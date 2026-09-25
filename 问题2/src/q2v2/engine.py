import contextlib
import copy
from datetime import datetime, timezone
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from q2.data import save_json, digest
from q2.engine import setup, batch_tensors, evaluate_model
from q2.missing import MAIN, augmentation_mask, scenario_mask
from q2.model import SentimentModel
from .model import PretrainedMultimodal
from .data import load_fold
from .losses import frequency_weights, microbatch_cross_entropy


def load_model(path, device="cpu"):
    record = torch.load(path, map_location="cpu", weights_only=True)
    config = record["config"]
    if config.get("kind") == "mlp":
        model = SentimentModel(config)
    else:
        model = PretrainedMultimodal(config, record["vocabulary"])
    model.load_state_dict(record["state_dict"])
    return model.to(device).eval(), record


def train_run(root, name, config, train_data, valid_data, vocabulary, source=None,
              teacher_path=None, fixed_epochs=None, reproduction=False, training_protocol=None):
    root = Path(root)
    # The portable repository may carry its frozen protocol in configs/ rather
    # than an untracked study/ directory. Existing callers keep their old path.
    protocol = copy.deepcopy(training_protocol) if training_protocol is not None else json.loads(
        (root / "study/protocol.json").read_text(encoding="utf-8"))
    settings = protocol["training"]
    run = root / "runs" / name
    result_path = run / "metrics.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    run.mkdir(parents=True, exist_ok=True)
    config = dict(config)
    setup(config.get("seed", 42))
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights = frequency_weights(train_data['labels'], config.get('class_weight_power', 0.))
    if class_weights is not None:
        class_weights = class_weights.to(device)
    is_mlp = config.get("kind") == "mlp"
    if is_mlp:
        model = SentimentModel(config).to(device)
    else:
        model = PretrainedMultimodal(config, vocabulary)
        if config.get("pretrained", True):
            if source is None:
                source = torch.load(root / "pretrained/bert_base_uncased.pt", map_location="cpu", weights_only=True)
            model.backbone.initialize(source)
        model = model.to(device)
    teacher = None
    if config.get("distillation"):
        if teacher_path is None:
            raise ValueError("Distillation requires a teacher fitted on the same fitting subset")
        teacher, teacher_record = load_model(teacher_path, device)
        for p in teacher.parameters():
            p.requires_grad_(False)
    batch_size = 64 if is_mlp else settings["batch_size"]
    microbatch_size = batch_size
    max_epochs = int(fixed_epochs or (40 if is_mlp else config.get("max_epochs", settings["max_epochs"])))
    patience_limit = 6 if is_mlp else settings["patience"]
    groups = []
    if is_mlp:
        groups = [{"params": list(model.parameters()), "lr": .001, "weight_decay": .01}]
    else:
        for backbone in [True, False]:
            for decay in [True, False]:
                parameters = [p for n,p in model.named_parameters() if n.startswith("backbone.") == backbone
                              and (not ("norm" in n or n.endswith("bias"))) == decay]
                groups.append({"params": parameters, "lr": settings["backbone_lr"] if backbone else settings["head_lr"],
                               "weight_decay": settings["weight_decay"] if decay else 0.})
    optimizer = torch.optim.AdamW(groups)
    steps = max_epochs * int(np.ceil(len(train_data["ids"]) / batch_size))
    warmup = max(1, int(steps * .1))
    schedule = None if is_mlp else torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step:
                min((step + 1) / warmup, max(0., (steps-step) / max(1, steps-warmup))))
    rng = np.random.default_rng(config.get("seed", 42))
    masks = {s: scenario_mask(valid_data, s) for s in ["clean"] + MAIN}
    best, best_epoch, stale = -float("inf"), 0, 0
    log = []
    started = time.time()
    metadata = {"name": name, "config": config, "fixed_epochs": fixed_epochs,
                "fit_ids": train_data["ids"].tolist(), "validation_ids": valid_data["ids"].tolist(),
                "vocabulary_size": len(vocabulary), "teacher_checkpoint_sha256": digest(teacher_path) if teacher_path else None,
                "training_precision": "FP32 MLP or BF16 CUDA autocast BERT; FP32 evaluation, TF32 disabled",
                "classification_weights": class_weights.detach().cpu().tolist() if class_weights is not None else None,
                "classification_weight_source": "fitting subset labels only",
                "scheduler": "constant MLP; BERT 10% linear warmup then linear decay, fixed before results"}
    save_json(run / "config.json", config)
    save_json(run / "training_protocol.json", protocol)
    save_json(run / "provenance.json", metadata)
    for epoch in range(1, max_epochs + 1):
        deadline = protocol.get("optimization_deadline")
        if not reproduction and deadline and datetime.now(timezone.utc) >= datetime.fromisoformat(deadline):
            raise TimeoutError("Predeclared optimization deadline reached")
        model.train()
        order = rng.permutation(len(train_data["ids"]))
        start, loss_sum, examples = 0, 0., 0
        while start < len(order):
            idx = order[start:start + batch_size]
            weight_sum = None if class_weights is None else float(class_weights[
                torch.from_numpy(train_data['labels'][idx]).to(device)].sum().item())
            optimizer.zero_grad(set_to_none=True)
            rng_state = copy.deepcopy(rng.bit_generator.state)
            cpu_state = torch.get_rng_state()
            cuda_state = torch.cuda.get_rng_state() if device.type == "cuda" else None
            try:
                drop = augmentation_mask(train_data["valid"][idx], rng, config.get("augmentation", "span"), .7)
                effective_loss = 0.
                for offset in range(0, len(idx), microbatch_size):
                    micro_idx = idx[offset:offset+microbatch_size]
                    batch = batch_tensors(train_data, micro_idx, device)
                    batch["drop"] = torch.from_numpy(drop[offset:offset+microbatch_size]).to(device)
                    labels = torch.from_numpy(train_data["labels"][micro_idx]).to(device)
                    targets = torch.from_numpy(train_data["targets"][micro_idx]).to(device)
                    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" and not is_mlp else contextlib.nullcontext()
                    with amp:
                        if teacher is not None:
                            logits, regression, features = model(**batch, return_features=True)
                            with torch.no_grad():
                                t_logits, t_regression, t_features = teacher(**batch, return_features=True)
                        else:
                            logits, regression = model(**batch)
                        loss = microbatch_cross_entropy(logits, labels, class_weights, len(idx), weight_sum) + F.huber_loss(regression.float(), targets)
                        if teacher is not None:
                            temperature = settings["kl_temperature"]
                            kd = F.kl_div(F.log_softmax(logits.float()/temperature, -1),
                                          F.softmax(t_logits.float()/temperature, -1), reduction="batchmean") * temperature**2
                            loss += settings["distill_kl_weight"] * kd
                            loss += settings["distill_reg_weight"] * F.mse_loss(regression.float(), t_regression.float())
                            loss += settings["distill_feature_weight"] * F.mse_loss(
                                F.layer_norm(features.float(), (768,)), F.layer_norm(t_features.float(), (768,)))
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite loss")
                    (loss * (len(micro_idx)/len(idx))).backward()
                    effective_loss += float(loss.item()) * len(micro_idx)
                nn.utils.clip_grad_norm_(model.parameters(), 5. if is_mlp else settings["gradient_clip"])
                optimizer.step()
                if schedule is not None:
                    schedule.step()
                loss_sum += effective_loss
                examples += len(idx)
                start += len(idx)
            except torch.cuda.OutOfMemoryError:
                optimizer.zero_grad(set_to_none=True)
                rng.bit_generator.state = rng_state
                torch.set_rng_state(cpu_state)
                if cuda_state is not None:
                    torch.cuda.set_rng_state(cuda_state)
                torch.cuda.empty_cache()
                if microbatch_size <= 8:
                    raise
                microbatch_size //= 2
                with (run / "events.log").open("a") as f:
                    f.write(f"Epoch {epoch}: CUDA OOM; microbatch {microbatch_size}, effective batch {batch_size}\n")
        # Confirmation fits have a pre-fixed epoch count; do not evaluate during
        # fitting or pick epochs using the official validation set.
        score, clean_f1, clean_mae = None, None, None
        if fixed_epochs is None:
            result = evaluate_model(model, valid_data, device, ["clean"] + MAIN, batch_size=64, masks=masks)
            score = result["selection"]["score"]
            clean_f1, clean_mae = result["clean"]["macro_f1"], result["clean"]["mae"]
        row = {"epoch": epoch, "loss": loss_sum/examples, "score": score, "clean_f1": clean_f1,
               "clean_mae": clean_mae, "seconds": time.time()-started, "batch_size": batch_size, "microbatch_size": microbatch_size}
        log.append(row)
        save_json(run / "epochs.json", log)
        print(json.dumps({"run":name, **row}), flush=True)
        improved = fixed_epochs is not None or score > best + 1e-6
        if improved:
            best_epoch, stale = epoch, 0
            if score is not None:
                best = score
            torch.save({"config": config, "vocabulary": torch.as_tensor(vocabulary),
                        "state_dict": model.state_dict(), "epoch": epoch}, run / "best.pt")
        else:
            stale += 1
        if fixed_epochs is None and stale >= patience_limit:
            break
    record = torch.load(run / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(record["state_dict"])
    result = evaluate_model(model, valid_data, device, ["clean"] + MAIN,
                            batch_size=64, masks=masks, out=run / "predictions")
    result["run_metadata"] = {"best_epoch": best_epoch, "epochs": len(log), "seconds": time.time()-started,
                              "parameters": sum(p.numel() for p in model.parameters()),
                              "checkpoint_sha256": digest(run / "best.pt"),
                              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0}
    save_json(result_path, result)
    return result


def train_fold(root, name, config, fold, teacher_path=None):
    training, validation = load_fold(root, fold, "train"), load_fold(root, fold, "valid")
    vocabulary = np.load(Path(root) / "folds" / f"fold{fold}" / "vocabulary.npy")
    return train_run(root, name, config, training, validation, vocabulary, teacher_path=teacher_path)
