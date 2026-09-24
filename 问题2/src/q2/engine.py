import csv
import json
import os
import random
import time
import traceback
from pathlib import Path
import numpy as np
# Set before the first CUDA context is created, including standalone inference.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch import nn
from .data import CLASSES, digest, load_data, save_json
from .metrics import metrics, selection_score
from .missing import MAIN, SUPPLEMENT, augmentation_mask, scenario_mask
from .model import SentimentModel

DEFAULT = dict(hidden=64, encoder="mean", fusion="concat", augmentation="none",
               augmentation_probability=.7, dropout=.2, lr=.001, batch_size=64,
               max_epochs=40, patience=6, regression_weight=1., class_weight=False,
               weight_decay=.01, seed=42)
TENSOR_KEYS = ["tokens", "audio", "vision", "valid", "observed"]


def setup(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Preserve the numerical training protocol used by the frozen study.
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def batch_tensors(data, indices, device):
    return {k: torch.from_numpy(data[k][indices]).to(device) for k in TENSOR_KEYS}


@torch.inference_mode()
def infer(model, data, device, drop=None, batch_size=128):
    model.eval()
    probabilities, values = [], []
    for start in range(0, len(data["ids"]), batch_size):
        indices = np.arange(start, min(start + batch_size, len(data["ids"])))
        batch = batch_tensors(data, indices, device)
        if drop is not None:
            batch["drop"] = torch.from_numpy(drop[indices]).to(device)
        logits, regression = model(**batch)
        probabilities.append(logits.softmax(-1).cpu().numpy())
        values.append(regression.cpu().numpy())
    return np.concatenate(probabilities), np.concatenate(values)


def write_predictions(path, data, probabilities, regression, scenario="clean"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["sample_id", "scenario", "class_id", "sentiment", "intensity",
              "prob_negative", "prob_neutral", "prob_positive"]
    labelled = "labels" in data
    if labelled:
        fields += ["true_class", "true_intensity"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, sample_id in enumerate(data["ids"]):
            c = int(probabilities[i].argmax())
            row = dict(sample_id=sample_id, scenario=scenario, class_id=c,
                       sentiment=CLASSES[c], intensity=float(regression[i]),
                       prob_negative=float(probabilities[i, 0]),
                       prob_neutral=float(probabilities[i, 1]), prob_positive=float(probabilities[i, 2]))
            if labelled:
                row.update(true_class=int(data["labels"][i]), true_intensity=float(data["targets"][i]))
            writer.writerow(row)


def evaluate_model(model, data, device, scenarios, out=None, batch_size=128, masks=None):
    results = {}
    for scenario in scenarios:
        drop = masks[scenario] if masks is not None else scenario_mask(data, scenario)
        p, y = infer(model, data, device, drop, batch_size)
        results[scenario] = metrics(data["labels"], data["targets"], p, y)
        if out is not None:
            write_predictions(Path(out) / f"{scenario}.csv", data, p, y, scenario)
    if all(s in results for s in MAIN):
        results["selection"] = selection_score(results, MAIN)
    return results


def save_scenarios(root, data, split):
    masks = {s: scenario_mask(data, s) for s in ["clean"] + MAIN + SUPPLEMENT}
    out = Path(root) / "scenarios"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{split}_masks.npz"
    np.savez_compressed(path, ids=data["ids"], **masks)
    definitions = {"seed": 20260924, "main": MAIN, "supplement": SUPPLEMENT,
                   "unit": "aligned valid feature positions, NOT seconds",
                   "rounding": "max(1, floor(valid_length * rate + 0.5)); empty rows remove zero",
                   "selection_score": "0.5 * mean_missing_macro_f1 + 0.5 * (1 - mean_missing_mae / 6)",
                   "mask_sha256": digest(path),
                   "actual_counts": {s: {"selected": int(m.sum()),
                                           "newly_unobserved": int((m & data["observed"]).sum())}
                                     for s, m in masks.items()}}
    save_json(out / f"{split}_definitions.json", definitions)
    return masks


def naive_baselines(root):
    train, valid = load_data(root, "train"), load_data(root, "valid")
    majority = int(np.bincount(train["labels"], minlength=3).argmax())
    p = np.eye(3, dtype=np.float32)[np.full(len(valid["ids"]), majority)]
    result = {}
    for name, value in [("mean", float(train["targets"].mean())), ("median", float(np.median(train["targets"])) )]:
        y = np.full(len(valid["ids"]), value, dtype=np.float32)
        result[name] = metrics(valid["labels"], valid["targets"], p, y)
        result[name].update(train_regression_constant=value, train_majority_class=majority)
        write_predictions(Path(root) / "audit" / f"naive_{name}_valid.csv", valid, p, y)
    save_json(Path(root) / "audit/naive_baselines.json", result)
    return result


def train(root, config, run_name, overwrite=False):
    root = Path(root)
    cfg = {**DEFAULT, **config}
    run = root / "runs" / run_name
    if (run / "metrics_valid.json").exists() and not overwrite:
        raise FileExistsError(f"Completed run is immutable: {run}")
    run.mkdir(parents=True, exist_ok=True)
    save_json(run / "config.json", cfg)
    setup(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training, valid = load_data(root, "train"), load_data(root, "valid")
    masks = {s: scenario_mask(valid, s) for s in ["clean"] + MAIN}
    model = SentimentModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    weights = None
    if cfg["class_weight"]:
        counts = np.bincount(training["labels"], minlength=3)
        weights = torch.tensor(len(training["ids"]) / (3 * counts), device=device, dtype=torch.float32)
    classification_loss = nn.CrossEntropyLoss(weight=weights)
    regression_loss = nn.HuberLoss(delta=1.)
    rng = np.random.default_rng(cfg["seed"])
    best, patience, best_epoch = -float("inf"), 0, 0
    started = time.time()
    log = []
    batch_size = cfg["batch_size"]
    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        order = rng.permutation(len(training["ids"]))
        loss_sum, count, start = 0., 0, 0
        while start < len(order):
            indices = order[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            try:
                batch = batch_tensors(training, indices, device)
                drop = augmentation_mask(training["valid"][indices], rng, cfg["augmentation"], cfg["augmentation_probability"])
                batch["drop"] = torch.from_numpy(drop).to(device)
                labels = torch.from_numpy(training["labels"][indices]).to(device)
                targets = torch.from_numpy(training["targets"][indices]).to(device)
                logits, values = model(**batch)
                loss = classification_loss(logits, labels) + cfg["regression_weight"] * regression_loss(values, targets)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
                loss_sum += loss.item() * len(indices)
                count += len(indices)
                start += len(indices)
            except torch.cuda.OutOfMemoryError:
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                if batch_size <= 4:
                    raise
                batch_size //= 2
                with (run / "events.log").open("a") as f:
                    f.write(f"epoch={epoch}: OOM, batch reduced to {batch_size}\n")
        val = evaluate_model(model, valid, device, ["clean"] + MAIN, masks=masks, batch_size=batch_size)
        score = val["selection"]["score"]
        row = dict(epoch=epoch, loss=loss_sum / count, valid_score=score,
                   clean_f1=val["clean"]["macro_f1"], clean_mae=val["clean"]["mae"],
                   missing_f1=val["selection"]["mean_missing_macro_f1"],
                   missing_mae=val["selection"]["mean_missing_mae"],
                   elapsed_seconds=time.time() - started, batch_size=batch_size)
        log.append(row)
        with (run / "epochs.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(log)
        print(json.dumps({"run": run_name, **row}), flush=True)
        if score > best + 1e-6:
            best, patience, best_epoch = score, 0, epoch
            torch.save({"state_dict": model.state_dict(), "config": cfg, "epoch": epoch,
                        "classes": CLASSES, "version": "1.0.0"}, run / "best.pt")
        else:
            patience += 1
        if patience >= cfg["patience"]:
            break
    checkpoint = torch.load(run / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    results = evaluate_model(model, valid, device, ["clean"] + MAIN,
                             out=run / "predictions/valid", masks=masks, batch_size=batch_size)
    save_json(run / "metrics_valid.json", results)
    metadata = dict(run=run_name, best_epoch=best_epoch, epochs=len(log),
                    elapsed_seconds=time.time() - started, parameters=sum(p.numel() for p in model.parameters()),
                    device=str(device), torch=torch.__version__, cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                    checkpoint_sha256=digest(run / "best.pt"),
                    source_sha256={p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
                    peak_cuda_memory_bytes=torch.cuda.max_memory_allocated() if device.type == "cuda" else 0)
    save_json(run / "metadata.json", metadata)
    return results


def load_model(checkpoint, device=None):
    # Full FP32 inference avoids cuDNN TF32 batch-shape drift for the GRU.
    # This changes no weights, thresholds, missing masks or model selection.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(4)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model = SentimentModel(saved["config"]).to(device)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model, device


def evaluate(root, checkpoint, split="valid", supplementary=False, final=False):
    root = Path(root)
    if split == "test" and (not final or not (root / "study/frozen.json").exists()):
        raise ValueError("Test is locked until study/frozen.json exists and --final is supplied")
    model, device = load_model(checkpoint)
    data = load_data(root, split)
    scenarios = ["clean"] + MAIN + (SUPPLEMENT if supplementary else [])
    masks = save_scenarios(root, data, split)
    run = Path(checkpoint).parent
    result = evaluate_model(model, data, device, scenarios, out=run / "predictions" / split, masks=masks)
    save_json(run / f"metrics_{split}{'_extended' if supplementary else ''}.json", result)
    return result


def predict(root, checkpoint, out, input_dir=None):
    root = Path(root)
    model, device = load_model(checkpoint)
    if input_dir:
        from .data import load_pickle, convert, normalize
        parts = []
        for path in sorted(Path(input_dir).glob("*.pkl")):
            raw = load_pickle(path)["test"]
            part, _ = convert(raw, [f"{path.name}::row{i:03d}" for i in range(len(raw["text_bert"]))])
            parts.append(part)
        if not parts:
            raise ValueError("Input directory contains no .pkl files")
        data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        with np.load(root / "data/processed/scaler.npz") as stats:
            data = normalize(data, stats)
    else:
        data = load_data(root, "attachment3")
    p, y = infer(model, data, device)
    p_single, y_single = infer(model, data, device, batch_size=1)
    again, _ = load_model(checkpoint, device)
    p_again, y_again = infer(again, data, device)
    if not np.allclose(p, p_single, atol=1e-5) or not np.allclose(y, y_single, atol=1e-5):
        raise AssertionError("Single/batch prediction mismatch")
    if not np.array_equal(p, p_again) or not np.array_equal(y, y_again):
        raise AssertionError("Reload prediction mismatch")
    if not np.isfinite(p).all() or not np.isfinite(y).all() or np.any(np.abs(y) > 3):
        raise AssertionError("Invalid predictions")
    if not np.allclose(p.sum(1), 1, atol=1e-5):
        raise AssertionError("Invalid class probabilities")
    write_predictions(out, data, p, y)
    check = {"rows": len(data["ids"]), "unique_ids": len(set(data["ids"])),
             "single_batch_max_probability_diff": float(np.max(np.abs(p - p_single))),
             "single_batch_max_intensity_diff": float(np.max(np.abs(y - y_single))),
             "reload_exact_match": True, "finite": True, "checkpoint_sha256": digest(checkpoint),
             "csv_sha256": digest(out)}
    save_json(Path(out).with_suffix(".checks.json"), check)
    return check
