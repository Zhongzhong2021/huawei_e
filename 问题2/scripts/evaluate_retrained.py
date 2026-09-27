"""Evaluate one new recipe fit and preserve its independent result version."""
import argparse
import gc
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from q2.data import digest, save_json
from q2.engine import infer
from q2.metrics import metrics
from q2v2.deploy import load_deployment, predict, raw_inputs
from q2v3.analysis import arrays
from q2v3.evaluation import evaluate_frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--aligned", type=Path, required=True)
    parser.add_argument("--special", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists() or (args.run / "evaluation").exists():
        raise FileExistsError("Evaluation and report must use new directories")
    provenance = json.loads((args.run / "provenance.json").read_text())
    if provenance["aligned_sha256"] != digest(args.aligned):
        raise ValueError("Evaluation dataset differs from the training source")
    started = time.time()
    model_path, scaler = args.run / "final/model.pt", args.run / "final/scaler.npz"
    evaluation = args.run / "evaluation/valid"
    print("Evaluating 46 validation scenarios", flush=True)
    evaluate_frozen(args.run, data_root=args.run, output_dir=evaluation, device="cuda")
    gc.collect()
    torch.cuda.empty_cache()
    observation = args.run / "evaluation/observation"
    subprocess.run([sys.executable, str(ROOT / "scripts/evaluate_observation_policy.py"),
                    "--aligned", str(args.aligned), "--checkpoint", str(model_path),
                    "--scaler", str(scaler), "--special", str(args.special),
                    "--output", str(observation), "--device", "cuda"], check=True)
    prediction_path = args.run / "final/attachment3_predictions.csv"
    checks = predict(model_path, scaler, args.special, prediction_path, "cuda")
    data = raw_inputs(args.special, scaler)
    if len(data["ids"]) != 30:
        raise ValueError("Expected all 30 attachment-3 samples")
    gpu, _ = load_deployment(model_path, "cuda")
    gp, gy = infer(gpu, data, "cuda", batch_size=32)
    del gpu
    gc.collect()
    torch.cuda.empty_cache()
    cpu, _ = load_deployment(model_path, "cpu")
    cp, cy = infer(cpu, data, "cpu", batch_size=32)
    np.testing.assert_allclose(gp, cp, atol=1e-5, rtol=0)
    np.testing.assert_allclose(gy, cy, atol=1e-5, rtol=0)
    np.testing.assert_array_equal(gp.argmax(1), cp.argmax(1))
    checks.update(cpu_gpu_max_probability_difference=float(np.abs(gp - cp).max()),
                  cpu_gpu_max_intensity_difference=float(np.abs(gy - cy).max()),
                  cpu_gpu_class_equal=True, tolerance=1e-5)
    save_json(args.run / "evaluation/prediction_checks.json", checks)
    scores = json.loads((evaluation / "metrics.json").read_text())
    paired = json.loads((observation / "results.json").read_text())
    anchor = arrays(evaluation / "predictions/clean.csv")
    for policy in ("retain", "mask_unk"):
        clean = arrays(observation / f"predictions/clean__{policy}.csv")
        for key in anchor:
            np.testing.assert_array_equal(anchor[key], clean[key])
    verified = []
    # Recompute every aggregate from serialized per-sample outputs.
    for folder, rows in [(evaluation / "predictions", [
            {"file": f"{key}.csv", "metrics": value} for key, value in scores.items()
            if key != "selection"]), (observation / "predictions", [
            {"file": f"{r['scenario']}__{r['policy']}.csv", "metrics": r}
            for r in paired["metrics"]])]:
        for row in rows:
            p = folder / row["file"]
            values = arrays(p)
            actual = metrics(values["labels"], values["targets"],
                             values["probabilities"], values["predictions"])
            for key in ("accuracy", "macro_f1", "mae", "pearson"):
                if actual[key] is None:
                    assert row["metrics"][key] is None
                else:
                    assert abs(actual[key] - row["metrics"][key]) < 1e-10
            verified.append({"path": str(p.relative_to(args.run)), "sha256": digest(p)})
    historical_path = ROOT / "docs/round4/evidence/confirmed.json"
    historical = next(r["metrics"] for r in json.loads(historical_path.read_text())["records"]
                      if r["seed"] == 42)
    historical_model = json.loads((ROOT / "metadata/frozen_model.json").read_text())
    report = {
        "identity": provenance["identity"], "seed": 42, "fixed_epochs": 4,
        "model_root": str(args.run.resolve()), "validation_samples": scores["clean"]["n"],
        "validation_scenarios": 46, "paired_policy_conditions": 26,
        "clean": scores["clean"], "missing_main_mean": scores["selection"],
        "historical_seed42_clean": historical["clean"],
        "difference_from_historical_seed42": {k: scores["clean"][k] - historical["clean"][k]
                                             for k in ("accuracy", "macro_f1", "mae", "pearson")},
        "historical_source_sha256": digest(historical_path),
        "scaler_matches_historical": digest(scaler) == historical_model["scaler_sha256"],
        "historical_comparison_scope": "Aggregate comparison of separate fits; historical tensors were unavailable",
        "observation_mean_paired_intervals": [r for r in paired["paired_intervals"]
                                             if r["scenario"] == "perturbed_mean"],
        "policy_promoted": False, "attachment3_policy": "retain",
        "attachment3": checks, "test_evaluated": False,
        "verified_prediction_tables": len(verified), "prediction_tables": verified,
        "clean_policy_predictions_identical": True,
        "training_and_main_evaluation_seconds": json.loads(
            (args.run / "runs/seed42/metrics.json").read_text())["run_metadata"]["seconds"],
        "extended_evaluation_seconds": time.time() - started,
    }
    args.report.mkdir(parents=True, exist_ok=False)
    save_json(args.report / "summary.json", report)
    for src, name in [
        (evaluation / "metrics.json", "validation_metrics.json"),
        (observation / "results.json", "observation_results.json"),
        (args.run / "provenance.json", "provenance.json"),
        (args.run / "metadata/frozen_model.json", "model.json"),
        (args.run / "data_audit.json", "data_audit.json"),
        (args.run / "runs/seed42/epochs.json", "epochs.json"),
        (prediction_path, "attachment3_predictions.csv"),
        (args.run / "evaluation/prediction_checks.json", "prediction_checks.json"),
    ]:
        shutil.copyfile(src, args.report / name)
    save_json(args.report / "files.json", {p.name: digest(p) for p in sorted(args.report.iterdir())})
    print(json.dumps({"clean": report["clean"], "report": str(args.report),
                      "verified_prediction_tables": len(verified)}), flush=True)


if __name__ == "__main__":
    main()
