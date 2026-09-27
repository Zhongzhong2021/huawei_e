"""Fit the saved four-epoch recipe from raw aligned data and BERT initialization."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from q2.data import convert, digest, fit_scaler, load_pickle, normalize, save_json
from q2v2.engine import train_run


def fitting_data(raw):
    """Use only train/valid; fail on overlapping IDs before fitting any statistics."""
    datasets, audits = {}, {}
    for split in ("train", "valid"):
        datasets[split], audits[split] = convert(raw[split], raw[split]["id"])
    if set(datasets["train"]["ids"]) & set(datasets["valid"]["ids"]):
        raise ValueError("Overlapping train/valid sample IDs")
    stats = fit_scaler(datasets["train"])
    return {s: normalize(d, stats) for s, d in datasets.items()}, stats, audits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True,
                        help="Converted BERT state dictionary with initialization.json alongside")
    parser.add_argument("--output", type=Path, required=True,
                        help="New experiment root; existing directories are rejected")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new experiment root")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This recipe requires a CUDA GPU with BF16 support")
    recipe_path = ROOT / "configs/retrain_seed42.json"
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    config_path = recipe_path.parent / recipe["config"]
    protocol_path = recipe_path.parent / recipe["training_protocol"]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if config["seed"] != recipe["seed"] or config["vocabulary_mode"] != "full":
        raise ValueError("Recipe/config mismatch")
    initialization = json.loads(args.pretrained.with_name("initialization.json").read_text())
    if (initialization["repository"] != recipe["pretrained_repository"] or
            initialization["revision"] != recipe["pretrained_revision"] or
            initialization["converted_sha256"] != digest(args.pretrained)):
        raise ValueError("Pretrained initialization provenance mismatch")
    started = time.time()
    data, stats, audits = fitting_data(load_pickle(args.aligned))
    source = torch.load(args.pretrained, map_location="cpu", weights_only=True)
    args.output.mkdir(parents=True, exist_ok=False)
    processed = args.output / "data/processed"
    processed.mkdir(parents=True)
    np.savez_compressed(processed / "scaler.npz", **stats)
    for split, values in data.items():
        np.savez_compressed(processed / f"{split}.npz", **values)
    save_json(args.output / "recipe.json", recipe)
    save_json(args.output / "configs/frozen_protocol.json", protocol)
    save_json(args.output / "data_audit.json", audits)
    provenance = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "identity": recipe["identity"], "recipe_sha256": digest(recipe_path),
        "config_sha256": digest(config_path), "protocol_sha256": digest(protocol_path),
        "aligned_sha256": digest(args.aligned), "initialization": initialization,
        "scaler_fit_split": "train", "evaluation_split": "valid",
        "test_evaluated": False, "validation_epoch_selection": False,
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "packages": {p: importlib.metadata.version(p) for p in
                     ["numpy", "scipy", "scikit-learn", "safetensors"]},
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in
                          sorted((ROOT / "src").rglob("*.py"))},
        "runner_sha256": digest(Path(__file__)),
        "processed_sha256": {p.name: digest(p) for p in sorted(processed.glob("*.npz"))},
    }
    save_json(args.output / "provenance.json", provenance)
    print(json.dumps({"status": "training", "train": len(data["train"]["ids"]),
                      "valid": len(data["valid"]["ids"]), "epochs": recipe["fixed_epochs"]}), flush=True)
    result = train_run(args.output, "seed42", config, data["train"], data["valid"],
                       np.arange(recipe["vocabulary_size"], dtype=np.int64), source=source,
                       fixed_epochs=recipe["fixed_epochs"], reproduction=True,
                       training_protocol=protocol)
    final = args.output / "final"
    final.mkdir()
    shutil.copyfile(args.output / "runs/seed42/best.pt", final / "model.pt")
    shutil.copyfile(processed / "scaler.npz", final / "scaler.npz")
    save_json(args.output / "metadata/frozen_model.json", {
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": "fixed_recipe_seed42_retrained", "identity": recipe["identity"],
        "checkpoint_sha256": digest(final / "model.pt"),
        "scaler_sha256": digest(final / "scaler.npz"),
        "protocol_sha256": digest(protocol_path), "test_used_for_selection": False,
        "quantization": None, "fixed_epochs": recipe["fixed_epochs"],
    })
    provenance.update(status="completed", total_seconds=time.time() - started,
                      completed_utc=datetime.now(timezone.utc).isoformat())
    save_json(args.output / "provenance.json", provenance)
    print(json.dumps({"status": "completed", "clean": result["clean"],
                      "total_seconds": provenance["total_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
