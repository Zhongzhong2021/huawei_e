"""Run isolated multiobjective experiments without changing legacy training."""
from __future__ import annotations

import argparse
import copy
import importlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import fit_stats, load_samples, make_batch
from .model import build_model
from .multiobjective import MultiObjectiveKeeper
from .train import _csv, _hash, _json, _ordinal_loss, _predict_prepared, _seed, _slice_batch


def build_experiment_model(config):
    factory = config["model"].get("factory")
    if factory:
        module, function = factory.split(":")
        return getattr(importlib.import_module(module), function)(config)
    return build_model(config)


def common_loss(out, batch, cfg):
    raw, y, c, logits = out["raw_intensity"], batch["y"], batch["class_label"], out["aux_logits"]
    weights = 1 + float(cfg.get("strength_weight", 0)) * y.abs()
    regression = (F.huber_loss(raw, y, delta=cfg.get("huber_delta", 1.0), reduction="none") * weights).mean() / weights.mean()
    loss = regression + cfg.get("classification_weight", .5) * F.cross_entropy(logits, c)
    if cfg.get("ordinal_weight", 0):
        loss = loss + cfg["ordinal_weight"] * _ordinal_loss(raw, c, cfg.get("ordinal_band", .3), cfg.get("ordinal_temperature", .2))
    if cfg.get("neutral_weight", 0):
        neutral = logits[:, 1] - torch.logsumexp(logits[:, [0, 2]], dim=1)
        loss = loss + cfg["neutral_weight"] * F.binary_cross_entropy_with_logits(neutral, (c == 1).float())
    if cfg.get("polarity_weight", 0) and (c != 1).any():
        mask = c != 1
        loss = loss + cfg["polarity_weight"] * F.cross_entropy(logits[mask][:, [0, 2]], (c[mask] == 2).long())
    if cfg.get("interaction_l1", 0) and "pair" in out:
        loss = loss + cfg["interaction_l1"] * out["pair"][..., 0].abs().sum(dim=(1, 2)).mean()
    return loss


def optimizer_groups(model, cfg):
    if hasattr(model, "optimizer_groups"):
        return model.optimizer_groups(cfg)
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        bert = name.startswith("bert.") or ".bert." in name or ".text_model." in name or name.startswith("text_model.")
        lr = cfg.get("encoder_learning_rate", cfg["learning_rate"]) if bert else cfg["learning_rate"]
        decay = cfg.get("weight_decay", .01) if p.ndim >= 2 else 0.0
        groups.setdefault((lr, decay), []).append(p)
    return [dict(params=p, lr=lr, weight_decay=decay) for (lr, decay), p in groups.items()]


def fit(config, run_dir):
    root = Path(run_dir)
    if root.exists():
        raise FileExistsError(f"refusing to overwrite {root}")
    config = copy.deepcopy(config)
    cfg = config["train"]
    torch.set_num_threads(1)
    _seed(int(cfg["seed"]))
    device = torch.device(cfg["device"])
    source = Path(config["data"]["aligned_path"])
    train_samples, valid_samples = load_samples(source, "train"), load_samples(source, "valid")
    stats = fit_stats(train_samples)
    if cfg.get("smoke"):
        train_samples, valid_samples = train_samples[:32], valid_samples[:32]
    train_batch = make_batch(train_samples, stats, config["model"]["window_size"], device)
    train_batch["sample_index"] = torch.arange(len(train_samples), device=device)
    valid_batch = make_batch(valid_samples, stats, config["model"]["window_size"], device)
    model = build_experiment_model(config).to(device)
    if hasattr(model, "prepare_training"):
        model.prepare_training(train_batch)
    optimizer = torch.optim.AdamW(optimizer_groups(model, cfg))
    keeper = MultiObjectiveKeeper(root, config, stats)
    _json(root / "environment.json", dict(python=sys.version, platform=platform.platform(),
          torch=torch.__version__, numpy=np.__version__, device=str(device),
          gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None))
    history, stale, started = [], 0, time.perf_counter()
    y = valid_batch["y"].cpu().numpy()
    ids = [str(s["id"]) for s in valid_samples]
    batch_size = int(cfg["batch_size"])
    for epoch in range(1, int(cfg["max_epochs"]) + 1):
        tick = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model.train()
        order, total_loss = np.random.permutation(len(train_samples)), 0.0
        for start in range(0, len(order), batch_size):
            batch = _slice_batch(train_batch, order[start:start + batch_size])
            optimizer.zero_grad(set_to_none=True)
            out = model(batch, interaction_gain=1.0)
            loss = model.training_loss(out, batch, epoch) if hasattr(model, "training_loss") else common_loss(out, batch, cfg)
            if loss.ndim != 0 or not torch.isfinite(loss):
                raise FloatingPointError(f"invalid loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("gradient_clip", 1.0), error_if_nonfinite=True)
            optimizer.step()
            if hasattr(model, "after_train_batch"):
                with torch.no_grad():
                    model.after_train_batch(out, batch, epoch)
            total_loss += float(loss.detach()) * len(batch["y"])
        if hasattr(model, "after_train_epoch"):
            with torch.no_grad():
                model.after_train_epoch(epoch)
        raw, aux = _predict_prepared(model, valid_batch, batch_size, 1.0)
        improved = keeper.consider(epoch, model, y, raw, aux, ids)
        stale = 0 if improved else stale + 1
        row = dict(epoch=epoch, train_loss=total_loss / len(train_samples),
                   valid_raw_mae=float(np.abs(y - raw).mean()), any_winner_improved=improved,
                   seconds=time.perf_counter() - tick,
                   peak_gpu_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                   **{f"best_{o}": w["metrics"][o] for o, w in keeper.winners.items()})
        history.append(row)
        _csv(root / "history.csv", history)
        print(json.dumps(dict(run=str(root), **row)), flush=True)
        if stale >= int(cfg.get("patience", 4)):
            break
    winners = keeper.finish()
    _json(root / "run_manifest.json", dict(status="complete", seed=cfg["seed"],
          smoke=bool(cfg.get("smoke")), model=config["model"],
          native_explanation_scope=getattr(model, "native_explanation_scope", "none"),
          epochs_run=len(history), elapsed_seconds=time.perf_counter() - started,
          data_sha256=_hash(source), stats_sha256=_hash(root / "stats.json"),
          parameter_count=sum(p.numel() for p in model.parameters()),
          trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
          train_n=len(train_samples), valid_n=len(valid_samples), test_used=False,
          source_sha256={p.name: _hash(p) for p in (Path(__file__), Path(__file__).with_name("multiobjective.py"))},
          winners=winners))
    return winners


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    fit(json.loads(args.config.read_text()), args.run_dir)
