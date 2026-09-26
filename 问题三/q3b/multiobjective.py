"""Validation-only policy selection and independent, reproducible metric winners."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from .train import _csv, _hash, _json, evaluate

OBJECTIVES = ("macro_f1", "accuracy", "mae")


def apply_policy(raw, aux, policy):
    raw, aux = np.asarray(raw), np.asarray(aux)
    if policy["decision_mode"] == "regression":
        c = np.where(raw < policy["tau_minus"], 0,
                     np.where(raw > policy["tau_plus"], 2, 1))
    elif policy["decision_mode"] == "classification":
        logits = aux.copy()
        logits[:, 1] += policy["neutral_bias"]
        c = logits.argmax(axis=1)
    else:
        raise ValueError("unknown decision mode")
    bounded = np.clip(raw, -3., 3.)
    pred = np.where(c == 0, np.minimum(bounded, -1e-6),
                    np.where(c == 2, np.maximum(bounded, 1e-6), 0.0))
    return c, pred


def policy_candidates(y, raw, aux):
    y, raw, aux = np.asarray(y), np.asarray(raw), np.asarray(aux)
    if (y.ndim != 1 or raw.shape != y.shape or aux.shape != (len(y), 3)
            or not len(y) or not all(np.isfinite(x).all() for x in (y, raw, aux))):
        raise ValueError("expected finite nonempty y/raw [N] and aux [N,3]")
    policies = [dict(decision_mode="regression", tau_minus=i / 20, tau_plus=j / 20)
                for i in range(-20, 1) for j in range(21)]
    policies += [dict(decision_mode="classification", neutral_bias=i / 10)
                 for i in range(-15, 16)]
    classes, values = zip(*(apply_policy(raw, aux, p) for p in policies))
    classes, values = np.asarray(classes), np.asarray(values)
    truth = np.where(y < 0, 0, np.where(y > 0, 2, 1))
    f1 = np.zeros(len(policies))
    for label in range(3):
        tp = ((classes == label) & (truth[None] == label)).sum(axis=1)
        denominator = (classes == label).sum(axis=1) + (truth == label).sum()
        f1 += np.divide(2 * tp, denominator, out=np.zeros_like(f1), where=denominator != 0) / 3
    accuracy = (classes == truth[None]).mean(axis=1)
    mae = np.abs(values - y[None]).mean(axis=1)
    return [dict(policy=p, metrics=dict(macro_f1=float(f1[i]), accuracy=float(accuracy[i]),
                                       mae=float(mae[i]))) for i, p in enumerate(policies)]


def objective_rank(metrics, objective):
    if objective == "macro_f1":
        return (-metrics["macro_f1"], metrics["mae"], -metrics["accuracy"])
    if objective == "accuracy":
        return (-metrics["accuracy"], -metrics["macro_f1"], metrics["mae"])
    if objective == "mae":
        return (metrics["mae"], -metrics["macro_f1"], -metrics["accuracy"])
    raise ValueError(f"unknown objective {objective}")


def select_policies(y, raw, aux):
    candidates = policy_candidates(y, raw, aux)
    return {objective: min(candidates, key=lambda row: objective_rank(row["metrics"], objective))
            for objective in OBJECTIVES}


class MultiObjectiveKeeper:
    """Keep distinct epochs/policies; delete only superseded states in this run."""

    def __init__(self, run_dir, config, stats):
        self.root = Path(run_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        if (self.root / "winners.json").exists():
            raise FileExistsError(f"refusing to replace existing experiment: {self.root}")
        self.config, self.stats, self.winners = config, stats, {}
        (self.root / "checkpoints").mkdir(exist_ok=True)
        (self.root / "epochs").mkdir(exist_ok=True)
        _json(self.root / "config.json", config)
        _json(self.root / "stats.json", stats)

    def consider(self, epoch, model, y, raw, aux, ids):
        if len(ids) != len(y) or len(set(ids)) != len(ids):
            raise ValueError("validation IDs must be unique and match predictions")
        candidates = policy_candidates(y, raw, aux)
        selected = {o: min(candidates, key=lambda row: objective_rank(row["metrics"], o))
                    for o in OBJECTIVES}
        improved = [o for o in OBJECTIVES if o not in self.winners or
                    objective_rank(selected[o]["metrics"], o) <
                    objective_rank(self.winners[o]["metrics"], o)]
        _json(self.root / "epochs" / f"epoch_{epoch:03d}_policies.json", candidates)
        np.savez_compressed(self.root / "epochs" / f"epoch_{epoch:03d}_valid.npz",
                            y=y, raw=raw, aux=aux, ids=np.asarray(ids, dtype=str))
        if not improved:
            return False
        checkpoint = self.root / "checkpoints" / f"epoch_{epoch:03d}.pt"
        temporary = checkpoint.with_suffix(".tmp")
        torch.save({"model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "epoch": epoch, "model_config": self.config["model"]}, temporary)
        os.replace(temporary, checkpoint)
        checkpoint_hash = _hash(checkpoint)
        truth = np.where(np.asarray(y) < 0, 0, np.where(np.asarray(y) > 0, 2, 1))
        for objective in improved:
            selection = selected[objective]
            classes, values = apply_policy(raw, aux, selection["policy"])
            metrics = evaluate(y, truth, values, classes)
            target = self.root / "best" / objective
            target.mkdir(parents=True, exist_ok=True)
            link = target / "best.tmp"
            if link.exists():
                link.unlink()
            os.link(checkpoint, link)
            os.replace(link, target / "best.pt")
            self.winners[objective] = dict(epoch=int(epoch), checkpoint=str(checkpoint.relative_to(self.root)),
                                          checkpoint_sha256=checkpoint_hash,
                                          policy=selection["policy"], metrics=metrics,
                                          raw_mae=float(np.mean(np.abs(np.asarray(y) - raw))),
                                          objective=objective, direction="min" if objective == "mae" else "max")
            for name, content in (("config.json", self.config), ("stats.json", self.stats),
                                  ("policy.json", selection["policy"]),
                                  ("metrics.json", self.winners[objective])):
                _json(target / name, content)
            _csv(target / "valid_predictions.csv", [dict(id=ids[i], true_intensity=float(y[i]),
                 true_class=int(truth[i]), raw_intensity=float(raw[i]),
                 aux_logit_0=float(aux[i, 0]), aux_logit_1=float(aux[i, 1]),
                 aux_logit_2=float(aux[i, 2]), polarity=int(classes[i]), intensity=float(values[i]))
                 for i in range(len(y))])
        _json(self.root / "winners.json", self.winners)
        live = {w["checkpoint"] for w in self.winners.values()}
        for old in (self.root / "checkpoints").glob("epoch_*.pt"):
            if str(old.relative_to(self.root)) not in live:
                old.unlink()
        return True

    def finish(self):
        if set(self.winners) != set(OBJECTIVES):
            raise RuntimeError("all three winners must exist before completion")
        _json(self.root / "selection_manifest.json", dict(status="complete", selection_split="valid",
              objectives=list(OBJECTIVES), intensity_policy="class_consistent_sign_and_neutral_zero",
              mae_constraint=False, config_sha256=_hash(self.root / "config.json"),
              winners=self.winners))
        return self.winners
