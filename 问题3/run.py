"""Portable inference and multiobjective training for the three delivered models."""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np
import torch

from q3b.data import load_samples, make_batch
from q3b.experiment import build_experiment_model, fit
from q3b.multiobjective import apply_policy
from q3b.train import _csv, _json, _slice_batch, evaluate as evaluate_metrics
from scripts.explanations import explanation_record


ROOT = Path(__file__).resolve().parent
MODELS = ("uncertainty", "selfmm", "tetfn")
FACTORIES = {"uncertainty": "q3b.ordinal_uncertainty:build",
             "selfmm": "q3b.selfmm_adapter:build",
             "tetfn": "q3b.tetfn_source_adapter:build"}


def bundle(model, selected=None):
    path = Path(selected).expanduser().resolve() if selected else ROOT / "weights" / model
    for name in ("best.pt", "config.json", "stats.json", "policy.json"):
        if not (path / name).is_file():
            raise FileNotFoundError(path / name)
    config = json.loads((path / "config.json").read_text())
    if config.get("model", {}).get("factory") != FACTORIES[model]:
        raise ValueError(f"bundle model factory does not match --model {model}")
    return path


def configuration(model, selected=None):
    config = json.loads((bundle(model, selected) / "config.json").read_text())
    config["model"]["pretrained_name"] = str(ROOT / "assets" / "bert-base-uncased")
    return config


def predict(args):
    torch.set_num_threads(1)
    output = args.output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"prediction output must be an empty directory: {output}")
    source = args.input.resolve()
    if args.split == "special" and not source.is_dir():
        raise ValueError("special input must be a directory containing 01.pkl through 20.pkl")
    if args.split != "special" and source.is_dir():
        raise ValueError("valid/test input must be one aligned pickle")
    path = bundle(args.model, args.bundle)
    config = configuration(args.model, args.bundle)
    stats = json.loads((path / "stats.json").read_text())
    policy = json.loads((path / "policy.json").read_text())
    samples = load_samples(source, None if args.split == "special" else args.split)
    prepared = make_batch(samples, stats, config["model"]["window_size"], args.device)
    model = build_experiment_model(config).to(args.device)
    state = torch.load(path / "best.pt", map_location=args.device, weights_only=True)
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.eval()
    output.mkdir(parents=True, exist_ok=True)
    rows, explanations = [], []
    scope = getattr(model, "native_explanation_scope", "none")
    with torch.inference_mode():
        size = int(config["train"]["batch_size"])
        for start in range(0, len(samples), size):
            indices = np.arange(start, min(start + size, len(samples)))
            batch = _slice_batch(prepared, indices)
            result = model(batch, interaction_gain=1.0)
            raw = result["raw_intensity"].cpu().numpy()
            aux = result["aux_logits"].cpu().numpy()
            classes, values = apply_policy(raw, aux, policy)
            for i, index in enumerate(indices):
                sample = samples[index]
                row = dict(id=sample["id"], polarity=int(classes[i]), intensity=float(values[i]),
                           raw_intensity=float(raw[i]),
                           **{f"aux_logit_{j}": float(aux[i, j]) for j in range(3)})
                if "y" in sample:
                    row.update(true_intensity=float(sample["y"]), true_class=int(sample["class_label"]))
                rows.append(row)
                if args.model == "uncertainty":
                    record = explanation_record(result, batch, i, policy, int(classes[i]), scope)
                    if record is None:
                        raise AssertionError("main model did not return native terms")
                    explanations.append(dict(id=sample["id"], polarity=int(classes[i]),
                                             intensity=float(values[i]), **record))
    _csv(output / "predictions.csv", rows)
    if explanations:
        with (output / "explanations.jsonl").open("w") as stream:
            for row in explanations:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(dict(model=args.model, split=args.split, samples=len(rows),
                          predictions=str(output / "predictions.csv"),
                          explanations=str(output / "explanations.jsonl") if explanations else None)))


def _rows(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or any(not row.get("id") for row in rows):
        raise ValueError("predictions must have nonempty id values")
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("prediction IDs must be unique")
    return rows


def evaluate(args):
    rows = _rows(args.predictions)
    if args.labels:
        with args.labels.open(newline="", encoding="utf-8-sig") as stream:
            labels = list(csv.DictReader(stream))
        ids = [row.get("sample_id") for row in labels]
        if not ids or any(not sid for sid in ids) or len(set(ids)) != len(ids):
            raise ValueError("label sample_id values must be nonempty and unique")
        if set(ids) != {row["id"] for row in rows}:
            raise ValueError("prediction and label IDs must match exactly")
        by_id = {row["sample_id"]: row for row in labels}
        truth = [by_id[row["id"]] for row in rows]
        y = np.asarray([float(row["sentiment"]) for row in truth])
        classes = np.asarray([int(row["class_id"]) for row in truth])
    else:
        y = np.asarray([float(row["true_intensity"]) for row in rows])
        classes = np.asarray([int(row["true_class"]) for row in rows])
    predicted = np.asarray([float(row["intensity"]) for row in rows])
    predicted_classes = np.asarray([int(row["polarity"]) for row in rows])
    raw = np.asarray([float(row["raw_intensity"]) for row in rows])
    if not np.isfinite(raw).all():
        raise ValueError("non-finite raw_intensity")
    if not np.array_equal(classes, np.where(y < 0, 0, np.where(y > 0, 2, 1))):
        raise ValueError("class labels must match sentiment sign")
    result = dict(n=len(rows), final=evaluate_metrics(y, classes, predicted, predicted_classes),
                  raw_mae=float(np.mean(np.abs(y - raw))))
    _json(args.output, result)
    print(json.dumps(dict(samples=len(rows), metrics=str(args.output.resolve()))))


def train(args):
    source = args.pretrained.expanduser().resolve()
    if not (source / "config.json").is_file() or not any(
            (source / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")):
        raise FileNotFoundError("--pretrained must be a local BERT directory with config and weights")
    if not args.aligned.is_file():
        raise FileNotFoundError(args.aligned)
    config = copy.deepcopy(configuration(args.model))
    config["model"]["pretrained_name"] = str(source)
    config["data"]["aligned_path"] = str(args.aligned.resolve())
    config["train"]["device"] = args.device
    config["train"]["max_epochs"] = args.epochs
    fit(config, args.output.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("predict")
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--split", choices=("special", "valid", "test"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--bundle", type=Path, help="selected best/<objective> directory from a training run")
    p.set_defaults(function=predict)
    p = commands.add_parser("evaluate")
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--labels", type=Path)
    p.set_defaults(function=evaluate)
    p = commands.add_parser("train")
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--aligned", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--pretrained", type=Path, required=True)
    p.set_defaults(function=train)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
