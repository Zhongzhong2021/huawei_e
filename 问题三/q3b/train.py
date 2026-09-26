"""Training and validation-only selection for the aligned-feature Plan B model.

The loop adapts the train/valid/early-stop shape of MMSA EF_LSTM.py at commit
a94e65d07fa1ae0d44e552390074b29b0898edfd. Losses, masks, optimization,
calibration and the strict test boundary are specific to this project.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import fit_stats, load_samples, make_batch
from .model import build_model

PAIR_KINDS = ("plan_b", "contextual_evidence", "bert_evidence")


def _json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _csv(path: Path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _hash(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dependency_versions():
    versions = {}
    for package in ("torch", "numpy", "scikit-learn", "matplotlib", "torchaudio", "transformers"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def apply_decision(raw_intensity, tau_minus, tau_plus):
    if not (-1 <= tau_minus <= 0 <= tau_plus <= 1):
        raise ValueError("thresholds must satisfy -1 <= tau_minus <= 0 <= tau_plus <= 1")
    raw = np.asarray(raw_intensity, dtype=np.float64)
    if not np.isfinite(raw).all():
        raise ValueError("non-finite prediction")
    cls = np.ones(raw.shape, dtype=np.int64)
    cls[raw < tau_minus] = 0
    cls[raw > tau_plus] = 2
    return cls, np.where(cls == 1, 0.0, raw)


def apply_model_decision(raw_intensity, aux_logits, calibration):
    """Apply the frozen decision policy; old bundles retain their exact behavior."""
    mode = calibration.get("decision_mode", "regression")
    if mode == "regression":
        return apply_decision(raw_intensity, calibration["tau_minus"], calibration["tau_plus"])
    if mode != "classification":
        raise ValueError(f"unknown decision mode: {mode}")
    raw = np.asarray(raw_intensity, dtype=np.float64).reshape(-1)
    logits = np.array(aux_logits, dtype=np.float64, copy=True)
    bias = float(calibration["neutral_bias"])
    if logits.shape != (len(raw), 3) or not (np.isfinite(raw).all() and
            np.isfinite(logits).all() and math.isfinite(bias)):
        raise ValueError("classification decision needs finite [N,3] logits and [N] scores")
    logits[:, 1] += bias
    cls = logits.argmax(axis=1)
    # Project onto the predicted polarity; conflicting regression scores approach zero.
    intensity = np.where(cls == 0, np.minimum(raw, -1e-6),
                         np.where(cls == 2, np.maximum(raw, 1e-6), 0.0))
    return cls, intensity


def calibrate_model_decision(y, raw, aux, config):
    mode = config.get("train", {}).get("decision_mode", "regression")
    if mode == "regression":
        return calibrate_thresholds(y, raw, config)
    if mode != "classification":
        raise ValueError(f"unknown decision mode: {mode}")
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    raw = np.asarray(raw, dtype=np.float64).reshape(-1)
    if len(y) != len(raw) or not len(y) or not (np.isfinite(y).all() and np.isfinite(raw).all()):
        raise ValueError("calibration needs paired finite nonempty arrays")
    classes = np.where(y < 0, 0, np.where(y > 0, 2, 1))
    raw_mae = float(np.abs(y - raw).mean())
    tolerance = float(config.get("calibration", {}).get("max_mae_increase", .02))
    candidates = []
    for index in range(-15, 16):
        policy = {"decision_mode": mode, "neutral_bias": index / 10}
        cls, intensity = apply_model_decision(raw, aux, policy)
        metrics = evaluate(y, classes, intensity, cls)
        candidates.append({"neutral_bias": index / 10,
                           "feasible": metrics["mae"] <= raw_mae + tolerance + 1e-12,
                           **{k: metrics[k] for k in ("macro_f1", "accuracy", "mae")}})
    feasible = [r for r in candidates if r["feasible"]]
    # Early epochs can violate the MAE constraint for every class-based decision.
    # Keep training, record infeasibility and use minimum MAE for that epoch.
    best = min(feasible or candidates, key=lambda r:
               ((-r["macro_f1"], r["mae"]) if feasible else (r["mae"], -r["macro_f1"]))
               + (-r["accuracy"], abs(r["neutral_bias"])))
    return {"decision_mode": mode, "neutral_bias": best["neutral_bias"],
            "raw_mae": raw_mae, "max_mae_increase": tolerance,
            "feasible": bool(feasible), "intensity_rule": "project_to_predicted_sign_epsilon_1e-6",
            "selected_metrics": {k: best[k] for k in ("macro_f1", "accuracy", "mae")},
            "candidates": candidates}


def evaluate(y, c, pred_y, pred_c):
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    c = np.asarray(c, dtype=np.int64).reshape(-1)
    pred_y = np.asarray(pred_y, dtype=np.float64).reshape(-1)
    pred_c = np.asarray(pred_c, dtype=np.int64).reshape(-1)
    if not (len(y) == len(c) == len(pred_y) == len(pred_c)) or not len(y):
        raise ValueError("metric arrays must be nonempty and equal length")
    if not (np.isfinite(y).all() and np.isfinite(pred_y).all()):
        raise ValueError("non-finite regression value")
    if not (np.isin(c, [0, 1, 2]).all() and np.isin(pred_c, [0, 1, 2]).all()):
        raise ValueError("classes must be 0, 1, or 2")
    matrix = np.zeros((3, 3), dtype=np.int64)
    np.add.at(matrix, (c, pred_c), 1)
    precision, recall, f1 = [], [], []
    for label in range(3):
        tp = int(matrix[label, label])
        p = tp / int(matrix[:, label].sum()) if matrix[:, label].sum() else 0.0
        r = tp / int(matrix[label, :].sum()) if matrix[label, :].sum() else 0.0
        precision.append(p)
        recall.append(r)
        f1.append(2 * p * r / (p + r) if p + r else 0.0)
    counts = matrix.sum(axis=1)
    pearson_defined = bool(np.std(y) > 0 and np.std(pred_y) > 0)
    pearson = float(np.corrcoef(y, pred_y)[0, 1]) if pearson_defined else None
    return {
        "accuracy": float(np.trace(matrix) / len(y)),
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": float(np.dot(f1, counts) / len(y)),
        "mae": float(np.mean(np.abs(y - pred_y))),
        "pearson": pearson,
        "pearson_defined": pearson_defined,
        "per_class": {str(i): {"precision": precision[i], "recall": recall[i],
                               "f1": f1[i], "support": int(counts[i])} for i in range(3)},
        "confusion_matrix": matrix.tolist(),
    }


def calibrate_thresholds(y, raw_intensity, config):
    """Search exactly 441 integer-derived threshold pairs on valid data."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    raw = np.asarray(raw_intensity, dtype=np.float64).reshape(-1)
    if len(y) != len(raw) or not len(y):
        raise ValueError("calibration needs paired nonempty arrays")
    true_class = np.where(y < 0, 0, np.where(y > 0, 2, 1))
    raw_mae = float(np.mean(np.abs(y - raw)))
    cfg = config.get("calibration", config)
    denominator = int(cfg.get("step_denominator", 20))
    neg = cfg.get("negative_step_indices", [-20, 0])
    pos = cfg.get("positive_step_indices", [0, 20])
    tolerance = float(cfg.get("max_mae_increase", 0.02))
    candidates = []
    for ni in range(int(neg[0]), int(neg[1]) + 1):
        for pi in range(int(pos[0]), int(pos[1]) + 1):
            minus, plus = ni / denominator, pi / denominator
            pred_class, pred_y = apply_decision(raw, minus, plus)
            metrics = evaluate(y, true_class, pred_y, pred_class)
            candidates.append({"tau_minus": minus, "tau_plus": plus,
                               "feasible": metrics["mae"] <= raw_mae + tolerance + 1e-12,
                               "macro_f1": metrics["macro_f1"], "accuracy": metrics["accuracy"],
                               "mae": metrics["mae"]})
    feasible = [row for row in candidates if row["feasible"]]
    if not feasible:
        raise ValueError("no feasible calibration candidate")
    best = min(feasible, key=lambda row: (-row["macro_f1"], row["mae"],
                                          -row["accuracy"], row["tau_plus"] - row["tau_minus"],
                                          abs(row["tau_minus"])))
    return {"tau_minus": best["tau_minus"], "tau_plus": best["tau_plus"],
            "raw_mae": raw_mae, "max_mae_increase": tolerance,
            "selection_order": ["macro_f1_desc", "mae_asc", "accuracy_desc",
                                "neutral_width_asc", "abs_tau_minus_asc"],
            "selected_metrics": {k: best[k] for k in ("macro_f1", "accuracy", "mae")},
            "candidates": candidates}


def _tuning_options(train_cfg):
    weight = float(train_cfg.get("ordinal_weight", 0.0))
    band = float(train_cfg.get("ordinal_band", 0.3))
    temperature = float(train_cfg.get("ordinal_temperature", 0.2))
    metric = train_cfg.get("checkpoint_metric", "valid_raw_mae")
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("ordinal_weight must be finite and nonnegative")
    if not math.isfinite(band) or band <= 0:
        raise ValueError("ordinal_band must be finite and positive")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("ordinal_temperature must be finite and positive")
    if metric not in ("valid_raw_mae", "valid_calibrated_macro_f1"):
        raise ValueError("invalid checkpoint_metric")
    return weight, band, temperature, metric


def _ordinal_loss(raw_intensity, class_label, band, temperature):
    logits = torch.stack(((raw_intensity + band) / temperature,
                          (raw_intensity - band) / temperature), dim=-1)
    targets = torch.stack((class_label > 0, class_label > 1), dim=-1).to(logits.dtype)
    return F.binary_cross_entropy_with_logits(logits, targets)


def _checkpoint_rank(row):
    return (-row["valid_calibrated_macro_f1"], row["valid_final_mae"],
            row["valid_raw_mae"])


def _slice_batch(batch, indices):
    size = len(batch["window_weight"])
    def take(value):
        if isinstance(value, torch.Tensor) and value.ndim and len(value) == size:
            return value[indices]
        if isinstance(value, dict):
            return {key: take(item) for key, item in value.items()}
        if isinstance(value, list) and len(value) == size:
            return [value[int(i)] for i in indices]
        return value
    return take(batch)


def _device(name):
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _predict_prepared(model, batch, batch_size, gain=1.0):
    model.eval()
    raw, aux = [], []
    with torch.inference_mode():
        for start in range(0, len(batch["window_weight"]), batch_size):
            part = _slice_batch(batch, np.arange(start, min(start + batch_size, len(batch["window_weight"]))))
            out = model(part, interaction_gain=gain)
            raw.append(out["raw_intensity"].detach().cpu().numpy())
            aux.append(out["aux_logits"].detach().cpu().numpy())
    return np.concatenate(raw), np.concatenate(aux)


def predict_batches(model, samples, stats, config, device):
    """Return (raw intensity [N], auxiliary logits [N,3]) in input order."""
    batch = make_batch(samples, stats, config["model"]["window_size"], device)
    gain = 1.0 if config["model"]["kind"] in PAIR_KINDS else 0.0
    return _predict_prepared(model, batch, config["train"]["batch_size"], gain)


def load_checkpoint(run_dir, device="cpu"):
    """Return (model, effective_config, train_stats, thresholds) for inference."""
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    stats = json.loads((run_dir / "stats.json").read_text())
    thresholds = json.loads((run_dir / "thresholds.json").read_text())
    state = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model = build_model(config).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model, config, stats, thresholds


def fit(config: dict, run_dir: Path, stats_path: Path | None = None) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    config = copy.deepcopy(config)
    train_cfg, model_cfg = config["train"], config["model"]
    ordinal_weight, ordinal_band, ordinal_temperature, checkpoint_metric = _tuning_options(train_cfg)
    kind = model_cfg["kind"]
    if kind not in PAIR_KINDS:
        train_cfg["warmup_epochs"] = 0
        train_cfg["interaction_l1"] = 0.0
    seed = int(train_cfg["seed"])
    _seed(seed)
    device = _device(train_cfg["device"])
    source = Path(config["data"]["aligned_path"])
    train_samples = load_samples(source, "train")
    valid_samples = load_samples(source, "valid")
    stats = json.loads(Path(stats_path).read_text()) if stats_path else fit_stats(train_samples)
    if stats.get("split", "train") != "train":
        raise ValueError("normalization stats must be fitted on train")
    if stats_path:
        recorded = stats.get("data_sha256") or stats.get("source_sha256")
        if recorded and recorded != _hash(source):
            raise ValueError("normalization stats source hash mismatch")
    if train_cfg.get("smoke", False):
        train_samples, valid_samples = train_samples[:16], valid_samples[:16]
        train_cfg["max_epochs"] = 2
        train_cfg["warmup_epochs"] = 0
        train_cfg["smoke"] = True
    train_batch = make_batch(train_samples, stats, model_cfg["window_size"], device)
    valid_batch = make_batch(valid_samples, stats, model_cfg["window_size"], device)
    model = build_model(config).to(device)
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        lr = train_cfg.get("encoder_learning_rate", train_cfg["learning_rate"]) if name.startswith("bert.") else train_cfg["learning_rate"]
        decay = train_cfg["weight_decay"] if parameter.ndim >= 2 else 0.0
        groups.setdefault((lr, decay), []).append(parameter)
    optimizer = torch.optim.AdamW([{"params": params, "lr": lr, "weight_decay": decay}
                                  for (lr, decay), params in groups.items()])
    pair_params = list(model.interaction_parameters()) if kind in PAIR_KINDS else []
    _json(run_dir / "config.json", config)
    _json(run_dir / "stats.json", stats)
    environment = {"python": sys.version, "platform": platform.platform(),
                   "torch": torch.__version__, "numpy": np.__version__,
                   "dependencies": _dependency_versions(),
                   "cuda_available": torch.cuda.is_available(), "device": str(device),
                   "cuda_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                   "cudnn_deterministic": torch.backends.cudnn.deterministic,
                   "cudnn_benchmark": torch.backends.cudnn.benchmark,
                   "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
    _json(run_dir / "environment.txt", environment)
    history, best_mae, best_raw_mae, best_rank, best_epoch, stale = [], float("inf"), float("inf"), None, None, 0
    started = time.perf_counter()
    for epoch in range(1, int(train_cfg["max_epochs"]) + 1):
        warm = kind in PAIR_KINDS and epoch <= int(train_cfg["warmup_epochs"])
        for parameter in pair_params:
            parameter.requires_grad_(not warm)
        gain = 0.0 if (warm or kind not in PAIR_KINDS) else 1.0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        tick = time.perf_counter()
        model.train()
        totals = np.zeros(5, dtype=np.float64)
        order = np.random.permutation(len(train_samples))
        for start in range(0, len(order), int(train_cfg["batch_size"])):
            indices = order[start:start + int(train_cfg["batch_size"])]
            batch = _slice_batch(train_batch, indices)
            optimizer.zero_grad(set_to_none=True)
            out = model(batch, interaction_gain=gain)
            regression = F.huber_loss(out["raw_intensity"], batch["y"], delta=train_cfg["huber_delta"])
            classification = F.cross_entropy(out["aux_logits"], batch["class_label"])
            ordinal = (_ordinal_loss(out["raw_intensity"], batch["class_label"], ordinal_band,
                                     ordinal_temperature) if ordinal_weight else regression.new_zeros(()))
            interaction = (out["pair"][..., 0].abs().sum(dim=(1, 2)).mean()
                           if kind in PAIR_KINDS else out["q"].new_zeros(()))
            loss = (regression + train_cfg["classification_weight"] * classification
                    + train_cfg["interaction_l1"] * interaction + ordinal_weight * ordinal)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["gradient_clip"])
            optimizer.step()
            totals += np.array([loss.item(), regression.item(), classification.item(),
                                interaction.item(), ordinal.item()]) * len(indices)
        raw, aux = _predict_prepared(model, valid_batch, int(train_cfg["batch_size"]), gain)
        y = valid_batch["y"].detach().cpu().numpy()
        c = valid_batch["class_label"].detach().cpu().numpy()
        valid_mae = float(np.mean(np.abs(y - raw)))
        aux_accuracy = float(np.mean(aux.argmax(axis=1) == c))
        aux_metrics = evaluate(y, c, raw, aux.argmax(axis=1))
        eligible = not warm
        calibration = (calibrate_model_decision(y, raw, aux, config)
                       if eligible and checkpoint_metric == "valid_calibrated_macro_f1" else None)
        row = {"epoch": epoch, "interaction_gain": gain, "checkpoint_metric": checkpoint_metric,
               "train_loss": totals[0] / len(order),
               "train_huber": totals[1] / len(order), "train_ce": totals[2] / len(order),
               "train_interaction_l1": totals[3] / len(order), "train_ordinal": totals[4] / len(order),
               "valid_raw_mae": valid_mae,
               "valid_aux_accuracy": aux_accuracy, "valid_aux_macro_f1": aux_metrics["macro_f1"],
               "valid_calibrated_macro_f1": calibration["selected_metrics"]["macro_f1"] if calibration else None,
               "valid_final_mae": calibration["selected_metrics"]["mae"] if calibration else None,
               "valid_tau_minus": calibration.get("tau_minus") if calibration else None,
               "valid_tau_plus": calibration.get("tau_plus") if calibration else None,
               "valid_neutral_bias": calibration.get("neutral_bias") if calibration else None,
               "valid_decision_feasible": calibration.get("feasible", True) if calibration else None,
               "epoch_seconds": time.perf_counter() - tick,
               "peak_gpu_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0}
        if eligible:
            best_raw_mae = min(best_raw_mae, valid_mae)
        rank = ((0 if row["valid_decision_feasible"] else 1,) + _checkpoint_rank(row)) if calibration else None
        selected = (eligible and (valid_mae < best_mae - float(train_cfg["min_delta"])
                    if checkpoint_metric == "valid_raw_mae" else best_rank is None or rank < best_rank))
        row["checkpoint_selected"] = bool(selected)
        history.append(row)
        _csv(run_dir / "history.csv", history)
        print(json.dumps({"run": str(run_dir), **row}), flush=True)
        if selected:
            best_mae, best_rank, best_epoch, stale = valid_mae, rank, epoch, 0
            torch.save({"model_state_dict": model.state_dict(), "model_config": model_cfg,
                        "best_epoch": epoch, "valid_raw_mae": valid_mae,
                        "checkpoint_metric": checkpoint_metric,
                        "selected_checkpoint_rank": rank,
                        "schema_version": config["schema_version"]}, run_dir / "best.pt")
        elif eligible:
            stale += 1
            if stale >= int(train_cfg["patience"]):
                break
    if best_epoch is None:
        raise RuntimeError("no eligible checkpoint; increase max_epochs beyond warmup")
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device, weights_only=False)["model_state_dict"])
    raw, aux = _predict_prepared(model, valid_batch, int(train_cfg["batch_size"]), 1.0 if kind in PAIR_KINDS else 0.0)
    y = valid_batch["y"].detach().cpu().numpy()
    c = valid_batch["class_label"].detach().cpu().numpy()
    calibration = calibrate_model_decision(y, raw, aux, config)
    candidates = calibration.pop("candidates")
    _csv(run_dir / "threshold_candidates.csv", candidates)
    _json(run_dir / "thresholds.json", calibration)
    final_c, final_y = apply_model_decision(raw, aux, calibration)
    metrics = {"final": evaluate(y, c, final_y, final_c),
               "raw": evaluate(y, c, raw, np.where(raw < 0, 0, np.where(raw > 0, 2, 1))),
               "aux": evaluate(y, c, raw, aux.argmax(axis=1)),
               "best_epoch": best_epoch, "best_valid_raw_mae": best_mae,
               "minimum_eligible_valid_raw_mae": best_raw_mae,
               "checkpoint_metric": checkpoint_metric,
               "selected_valid_raw_mae": float(np.mean(np.abs(y - raw))),
               "selected_valid_calibrated_macro_f1": calibration["selected_metrics"]["macro_f1"],
               "selected_valid_final_mae": calibration["selected_metrics"]["mae"]}
    _json(run_dir / "metrics.json", metrics)
    _csv(run_dir / "valid_predictions.csv", [
        {"id": str(sample["id"]), "y": float(y[i]), "class_label": int(c[i]),
         "raw_intensity": float(raw[i]), "aux_class": int(aux[i].argmax()),
         "aux_logit_0": float(aux[i, 0]), "aux_logit_1": float(aux[i, 1]),
         "aux_logit_2": float(aux[i, 2]),
         "polarity": int(final_c[i]), "intensity": float(final_y[i])}
        for i, sample in enumerate(valid_samples)])
    _json(run_dir / "run_manifest.json", {"status": "complete", "kind": kind, "seed": seed,
          "smoke": bool(train_cfg.get("smoke", False)), "data_sha256": _hash(source),
          "stats_source_sha256": stats.get("source_sha256"),
          "processing_version": stats.get("processing_version"),
          "stats_sha256": _hash(run_dir / "stats.json"), "parameter_count": sum(p.numel() for p in model.parameters()),
          "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
          "epochs_run": len(history), "best_epoch": best_epoch, "elapsed_seconds": time.perf_counter() - started,
          "device": str(device), "checkpoint_metric": checkpoint_metric,
          "best_valid_raw_mae": best_mae,
          "minimum_eligible_valid_raw_mae": best_raw_mae,
          "selected_valid_raw_mae": metrics["selected_valid_raw_mae"],
          "config_sha256": _hash(run_dir / "config.json")})
    return run_dir / "best.pt"


def select(runs, out_dir):
    rows = []
    for run in map(Path, runs):
        config = json.loads((run / "config.json").read_text())
        manifest = json.loads((run / "run_manifest.json").read_text())
        metrics = json.loads((run / "metrics.json").read_text())["final"]
        if (manifest.get("status") != "complete" or config["model"]["kind"] != "plan_b"
                or config["train"]["seed"] != 17 or manifest.get("smoke")):
            raise ValueError(f"selection expects full Plan B seed 17 runs: {run}")
        rows.append({"run": str(run), "mae": metrics["mae"], "macro_f1": metrics["macro_f1"],
                     "accuracy": metrics["accuracy"], "parameters": manifest["parameter_count"]})
    if not rows:
        raise ValueError("no selection runs")
    minimum_mae = min(row["mae"] for row in rows)
    for row in rows:
        row["feasible"] = row["mae"] <= minimum_mae + 0.02 + 1e-12
    best = min((row for row in rows if row["feasible"]),
               key=lambda row: (-row["macro_f1"], row["mae"], row["parameters"]))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_config = json.loads((Path(best["run"]) / "config.json").read_text())
    _json(out_dir / "selected_config.json", selected_config)
    _json(out_dir / "selection.json", {"selected_run": best["run"], "selected_seed": 17,
          "minimum_mae": minimum_mae, "mae_tolerance": 0.02, "runs": rows})
    _csv(out_dir / "comparison.csv", rows)
    return best["run"]


def freeze(selection_path, out_dir, deployment_seed=17):
    selection = json.loads(Path(selection_path).read_text())
    if deployment_seed != 17 or selection["selected_seed"] != 17:
        raise ValueError("deployment seed is fixed to 17")
    source = Path(selection["selected_run"])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("best.pt", "stats.json", "config.json", "thresholds.json", "metrics.json"):
        shutil.copy2(source / name, out_dir / name)
    _json(out_dir / "freeze_manifest.json", {"source_run": str(source), "deployment_seed": 17,
          "files_sha256": {name: _hash(out_dir / name) for name in
                           ("best.pt", "stats.json", "config.json", "thresholds.json", "metrics.json")}})
    return out_dir


def evaluate_bundle(bundle, split, out):
    if split not in ("valid", "test"):
        raise ValueError("split must be valid or test")
    bundle = Path(bundle)
    if split == "test" and not (bundle / "freeze_manifest.json").exists():
        raise ValueError("test evaluation requires a frozen bundle")
    model, config, stats, thresholds = load_checkpoint(bundle, "cpu")
    device = _device(config["train"]["device"])
    model.to(device)
    samples = load_samples(Path(config["data"]["aligned_path"]), split)
    raw, aux = predict_batches(model, samples, stats, config, device)
    y = np.array([sample["y"] for sample in samples], dtype=np.float64)
    c = np.array([sample["class_label"] for sample in samples], dtype=np.int64)
    final_c, final_y = apply_model_decision(raw, aux, thresholds)
    result = {"split": split, "n": len(samples), "final": evaluate(y, c, final_y, final_c),
              "raw": evaluate(y, c, raw, np.where(raw < 0, 0, np.where(raw > 0, 2, 1))),
              "aux": evaluate(y, c, raw, aux.argmax(axis=1)), "thresholds": thresholds}
    _json(Path(out), result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit_cmd = commands.add_parser("fit")
    fit_cmd.add_argument("--config", type=Path, required=True)
    fit_cmd.add_argument("--stats", type=Path)
    fit_cmd.add_argument("--run-dir", type=Path, required=True)
    fit_cmd.add_argument("--seed", type=int)
    fit_cmd.add_argument("--window-size", type=int)
    fit_cmd.add_argument("--interaction-l1", type=float)
    fit_cmd.add_argument("--kind", choices=["plan_b", "additive", "free_fusion", "text_only",
                                          "contextual_evidence", "sequence_fusion", "bert_evidence", "bert_fusion"])
    fit_cmd.add_argument("--device")
    fit_cmd.add_argument("--smoke", action="store_true")
    select_cmd = commands.add_parser("select")
    select_cmd.add_argument("--runs", type=Path, nargs="+", required=True)
    select_cmd.add_argument("--out-dir", type=Path, required=True)
    freeze_cmd = commands.add_parser("freeze")
    freeze_cmd.add_argument("--selection", type=Path, required=True)
    freeze_cmd.add_argument("--deployment-seed", type=int, default=17)
    freeze_cmd.add_argument("--out-dir", type=Path, required=True)
    eval_cmd = commands.add_parser("evaluate")
    eval_cmd.add_argument("--bundle", type=Path, required=True)
    eval_cmd.add_argument("--split", choices=["valid", "test"], required=True)
    eval_cmd.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "fit":
        config = json.loads(args.config.read_text())
        for key, value in (("seed", args.seed), ("interaction_l1", args.interaction_l1),
                           ("device", args.device)):
            if value is not None:
                config["train"][key] = value
        if args.window_size is not None:
            config["model"]["window_size"] = args.window_size
        if args.kind is not None:
            config["model"]["kind"] = args.kind
        if args.smoke:
            config["train"]["smoke"] = True
        print(fit(config, args.run_dir, args.stats))
    elif args.command == "select":
        print(select(args.runs, args.out_dir))
    elif args.command == "freeze":
        print(freeze(args.selection, args.out_dir, args.deployment_seed))
    elif args.command == "evaluate":
        print(json.dumps(evaluate_bundle(args.bundle, args.split, args.out)))


if __name__ == "__main__":
    main()
