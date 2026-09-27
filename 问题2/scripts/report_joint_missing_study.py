"""Recompute paired results and conditional video-bootstrap intervals from saved predictions."""
import argparse
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from q2.data import convert, digest, fit_scaler, load_pickle, normalize, save_json
from q2.joint_study import decision, joint_mask, scenarios
from q2.metrics import metrics
from q2.missing import MAIN, scenario_mask
from q2v2.data import subset
from q2v3.analysis import arrays, boot_metrics, cluster_weights, read


def groups(protocol):
    result = {"clean": ["clean"], "single": MAIN}
    for group, rates in protocol["groups"].items():
        result[group] = [s for s in scenarios(protocol) if s.startswith("joint_")
                         and int(s.split("_")[1]) in rates]
    result["low_medium"] = result["low_joint"] + result["medium_joint"]
    return result


def analyze_stage(root, protocol, stage):
    development = stage == "development"
    records = read(root / f"{stage}.json")
    selection = decision([r["reference"] for r in records], [r["candidate"] for r in records], protocol)
    assert selection == read(root / f"{stage}_decision.json")
    deltas, verified = [], []
    manifests = []
    for index, record in enumerate(records):
        name = f"fold{record['fold']}" if development else "full"
        seed = 42 if development else record["seed"]
        paths = {arm: root / "evaluation" / f"{name}_{arm}_s{seed}" for arm in ("reference", "candidate")}
        anchor = arrays(paths["reference"] / "predictions/clean.csv")
        manifest = read(root / "data" / name / "manifest.json")
        np.testing.assert_array_equal(anchor["ids"], manifest["evaluation_ids"])
        assert not set(manifest["fit_ids"]) & set(manifest["evaluation_ids"])
        if development:
            fit_groups = {sid.split("$_$")[0] for sid in manifest["fit_ids"]}
            eval_groups = {sid.split("$_$")[0] for sid in manifest["evaluation_ids"]}
            assert not fit_groups & eval_groups
        manifests.append(manifest)
        # Same validation video draws across training seeds; independent draws
        # within disjoint development folds. Training-seed uncertainty excluded.
        weights, videos = cluster_weights(anchor["ids"], replicates=2000,
                                           seed=270927 + index if development else 270927)
        draws = {}
        for arm in paths:
            saved = read(paths[arm] / "metrics.json")
            for scenario in scenarios(protocol):
                path = paths[arm] / "predictions" / f"{scenario}.csv"
                data = arrays(path)
                for key in ("ids", "labels", "targets"):
                    np.testing.assert_array_equal(anchor[key], data[key])
                actual = metrics(data["labels"], data["targets"], data["probabilities"], data["predictions"])
                for metric in ("accuracy", "macro_f1", "mae", "pearson"):
                    assert (actual[metric] is None and saved[scenario][metric] is None) or abs(actual[metric] - saved[scenario][metric]) < 1e-10
                verified.append({"path": str(path.relative_to(root)), "sha256": digest(path)})
                draws[(arm, scenario)] = boot_metrics(data, weights)
        deltas.append({group: {metric: np.mean([
            draws[("candidate", s)][metric] - draws[("reference", s)][metric]
            for s in names], axis=0) for metric in ("macro_f1", "mae")}
            for group, names in groups(protocol).items()})
    if development:
        ids = [sid for m in manifests for sid in m["evaluation_ids"]]
        assert len(ids) == len(set(ids)) == 3395
    intervals = []
    for group in groups(protocol):
        for metric in ("macro_f1", "mae"):
            delta = np.mean([d[group][metric] for d in deltas], axis=0)
            intervals.append({"group": group, "metric": metric,
                "difference": selection["candidate_mean"][group][metric] - selection["reference_mean"][group][metric],
                "lower": float(np.quantile(delta, .025)), "upper": float(np.quantile(delta, .975))})
    return {"selection": selection, "paired_intervals": intervals, "verified_tables": verified,
        "bootstrap": {"replicates": 2000, "seed": 270927, "unit": "video",
            "scope": "Conditional on fixed fitted models and masks; not training-seed or selection-adjusted uncertainty",
            "development": "Independent within-fold video resampling, then equal fold mean",
            "confirmation": "Same video resamples across models, conditions and training seeds, then equal seed mean",
            "interval": "Pointwise percentile 95%; no multiplicity correction"}}


def audit_data(root, source, protocol):
    raw = load_pickle(source)
    train, _ = convert(raw["train"], raw["train"]["id"])
    by_id = {sid: i for i, sid in enumerate(train["ids"])}
    checks = []
    for folder in sorted((root / "data").iterdir()):
        manifest = read(folder / "manifest.json")
        fitting = subset(train, [by_id[sid] for sid in manifest["fit_ids"]])
        if folder.name == "full":
            evaluation, _ = convert(raw["valid"], raw["valid"]["id"])
        else:
            evaluation = subset(train, [by_id[sid] for sid in manifest["evaluation_ids"]])
        stats = fit_scaler(fitting)
        with np.load(folder / "scaler.npz", allow_pickle=False) as stored:
            for key in stats:
                np.testing.assert_array_equal(stats[key], stored[key])
        for split, original in [("train", fitting), ("valid", evaluation)]:
            expected = normalize(original, stats)
            with np.load(folder / f"{split}.npz", allow_pickle=False) as stored:
                assert set(expected) == set(stored.files)
                for key in expected:
                    np.testing.assert_array_equal(expected[key], stored[key])
        assert not fitting["text_unknown"].any() and not evaluation["text_unknown"].any()
        with np.load(folder / "masks.npz", allow_pickle=False) as masks:
            np.testing.assert_array_equal(masks["ids"], evaluation["ids"])
            for scenario in scenarios(protocol):
                expected = joint_mask(evaluation, scenario) if scenario.startswith("joint_") else scenario_mask(evaluation, scenario)
                np.testing.assert_array_equal(masks[scenario], expected)
        for run in (root / "runs").glob(f"{folder.name}_*/provenance.json"):
            record = read(run)
            np.testing.assert_array_equal(record["fit_ids"], fitting["ids"])
            counts = np.bincount(fitting["labels"], minlength=3)
            weights = len(fitting["ids"]) / (3 * counts)
            weights = (weights / weights.mean()).astype(np.float32)
            np.testing.assert_array_equal(record["classification_weights"], weights)
        checks.append({"data": folder.name, "train": len(fitting["ids"]),
                       "evaluation": len(evaluation["ids"]), "scaler_fit_only": True,
                       "normalized_arrays_exact": True, "masks_exact": True, "class_weights_fit_only": True})
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--aligned", type=Path, required=True)
    args = parser.parse_args()
    terminal = read(args.run / "terminal.json")
    if terminal["status"] != "completed" or args.output.exists():
        raise ValueError("Require completed study and fresh report directory")
    protocol = read(args.run / "protocol.json")
    started = read(args.run / "started.json")
    assert digest(args.run / "protocol.json") == started["protocol_sha256"]
    assert digest(args.aligned) == started["aligned_sha256"]
    for relative, sha in started["source_sha256"].items():
        assert digest(ROOT / relative) == sha, relative
    args.output.mkdir(parents=True)
    result = {"model_root": str(args.run.resolve()), "terminal": terminal,
              "data_audit": audit_data(args.run, args.aligned, protocol),
              "development": analyze_stage(args.run, protocol, "development"),
              "report_script_sha256": digest(Path(__file__))}
    if terminal["validation_evaluated"]:
        result["confirmation"] = analyze_stage(args.run, protocol, "confirmation")
    for filename in ("protocol.json", "started.json", "development.json", "development_decision.json",
                     "confirmation.json", "confirmation_decision.json", "candidate_frozen.json", "terminal.json"):
        if (args.run / filename).exists():
            shutil.copyfile(args.run / filename, args.output / filename)
    save_json(args.output / "analysis.json", result)
    save_json(args.output / "files.json", {p.name: digest(p) for p in sorted(args.output.iterdir())})
    print({"candidate_promoted": terminal["candidate_promoted"], "output": str(args.output)}, flush=True)


if __name__ == "__main__":
    main()
