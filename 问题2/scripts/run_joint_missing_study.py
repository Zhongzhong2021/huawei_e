"""One bounded, predeclared comparison; development gates validation confirmation."""
import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from q2.data import convert, digest, fit_scaler, load_pickle, normalize, save_json
from q2.engine import evaluate_model
from q2.joint_study import decision, joint_mask, scenarios, summarize
from q2.missing import scenario_mask
from q2v2.data import subset
from q2v2.engine import load_model, train_run


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def prepare_pair(root, name, fitting, validation, protocol):
    out = root / "data" / name
    out.mkdir(parents=True, exist_ok=False)
    stats = fit_scaler(fitting)
    np.savez_compressed(out / "scaler.npz", **stats)
    fitting, validation = normalize(fitting, stats), normalize(validation, stats)
    for split, data in [("train", fitting), ("valid", validation)]:
        np.savez_compressed(out / f"{split}.npz", **data)
    masks = {s: joint_mask(validation, s) if s.startswith("joint_") else scenario_mask(validation, s)
             for s in scenarios(protocol)}
    np.savez_compressed(out / "masks.npz", ids=validation["ids"], **masks)
    save_json(out / "manifest.json", {
        "fit_ids": fitting["ids"].tolist(), "evaluation_ids": validation["ids"].tolist(),
        "fit_class_counts": np.bincount(fitting["labels"], minlength=3).tolist(),
        "scaler_fit": "fitting subset only", "scaler_sha256": digest(out / "scaler.npz"),
        "mask_sha256": digest(out / "masks.npz"), "scenarios": len(masks),
    })
    return fitting, validation, masks


def pair(root, name, seed, data, source, config, training, protocol):
    fitting, validation, masks = data
    summaries = {}
    for arm, mode in [("reference", "span"), ("candidate", "mixed_joint")]:
        run_name = f"{name}_{arm}_s{seed}"
        cfg = {**config, "seed": seed, "augmentation": mode, "separate_augmentation_rng": True}
        train_run(root, run_name, cfg, fitting, validation, np.arange(30522), source=source,
                  fixed_epochs=protocol["fixed_epochs"], reproduction=True, training_protocol=training)
        gc.collect()
        torch.cuda.empty_cache()
        model, _ = load_model(root / "runs" / run_name / "best.pt", "cuda")
        out = root / "evaluation" / run_name
        scores = evaluate_model(model, validation, "cuda", list(masks), out=out / "predictions",
                                batch_size=32, masks=masks)
        save_json(out / "metrics.json", scores)
        summaries[arm] = summarize(scores, protocol)
        save_json(out / "summary.json", summaries[arm])
        print(json.dumps({"stage": name, "arm": arm, "seed": seed, "summary": summaries[arm]}), flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return summaries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--aligned", type=Path, required=True)
    p.add_argument("--pretrained", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError("Choose a fresh study root")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 is required")
    protocol_path = ROOT / "configs/joint_missing_study.json"
    protocol = read(protocol_path)
    config = read(ROOT / "configs/confirmed_seed42.json")
    training = read(ROOT / "configs/frozen_protocol.json")
    initialization = read(a.pretrained.with_name("initialization.json"))
    if initialization["converted_sha256"] != digest(a.pretrained):
        raise ValueError("Pretrained hash mismatch")
    a.output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    shutil.copyfile(protocol_path, a.output / "protocol.json")
    save_json(a.output / "started.json", {
        "started_utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": digest(protocol_path),
        "aligned_sha256": digest(a.aligned), "initialization": initialization,
        "training_protocol": training, "base_config": config,
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "source_sha256": {str(path.relative_to(ROOT)): digest(path) for folder in ("src", "scripts", "tests")
                          for path in sorted((ROOT / folder).rglob("*.py"))},
    })
    # One PKL contains all splits; only train is converted during development.
    raw = load_pickle(a.aligned)
    train, _ = convert(raw["train"], raw["train"]["id"])
    groups = np.array([str(sid).split("$_$")[0] for sid in train["ids"]])
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=protocol["fold_seed"])
    source = torch.load(a.pretrained, map_location="cpu", weights_only=True)
    records = []
    for fold, (fit_idx, dev_idx) in enumerate(splitter.split(train["tokens"], train["labels"], groups)):
        assert not set(groups[fit_idx]) & set(groups[dev_idx])
        data = prepare_pair(a.output, f"fold{fold}", subset(train, fit_idx), subset(train, dev_idx), protocol)
        scores = pair(a.output, f"fold{fold}", protocol["development_seed"], data, source, config, training, protocol)
        records.append({"fold": fold, "video_overlap": 0, **scores})
        save_json(a.output / "development.json", records)
    dev = decision([r["reference"] for r in records], [r["candidate"] for r in records], protocol)
    save_json(a.output / "development_decision.json", dev)
    terminal = {"development": dev, "test_evaluated": False, "special_labels_used": False,
                "validation_evaluated": False, "candidate_promoted": False}
    if dev["passed"]:
        save_json(a.output / "candidate_frozen.json", {
            "utc": datetime.now(timezone.utc).isoformat(), "augmentation": "mixed_joint",
            "fixed_epochs": 4, "inference_policy": "mask_unk",
            "protocol_sha256": digest(a.output / "protocol.json"),
        })
        valid, _ = convert(raw["valid"], raw["valid"]["id"])
        assert not set(train["ids"]) & set(valid["ids"])
        data = prepare_pair(a.output, "full", train, valid, protocol)
        confirmations = []
        for seed in protocol["confirmation_seeds"]:
            scores = pair(a.output, "full", seed, data, source, config, training, protocol)
            confirmations.append({"seed": seed, **scores})
            save_json(a.output / "confirmation.json", confirmations)
        confirmed = decision([r["reference"] for r in confirmations], [r["candidate"] for r in confirmations], protocol)
        save_json(a.output / "confirmation_decision.json", confirmed)
        terminal.update(confirmation=confirmed, validation_evaluated=True, candidate_promoted=confirmed["passed"])
        if confirmed["passed"]:
            final = a.output / "final"
            final.mkdir()
            shutil.copyfile(a.output / "runs/full_candidate_s42/best.pt", final / "model.pt")
            shutil.copyfile(a.output / "data/full/scaler.npz", final / "scaler.npz")
            save_json(final / "model.json", {"unknown_policy": "mask_unk", "seed": 42,
                "checkpoint_sha256": digest(final / "model.pt"), "scaler_sha256": digest(final / "scaler.npz")})
    terminal.update(status="completed", seconds=time.time() - started,
                    completed_utc=datetime.now(timezone.utc).isoformat())
    save_json(a.output / "terminal.json", terminal)
    print(json.dumps(terminal), flush=True)


if __name__ == "__main__":
    main()
