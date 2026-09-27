"""Fixed joint-missing evaluation and paired promotion rules."""
import numpy as np
from .missing import MAIN, select_positions, stable_seed


def scenarios(protocol):
    return ["clean"] + MAIN + [
        f"joint_{rate}_{shape}_{seed}"
        for rate in protocol["evaluation_joint_rates"]
        for shape in protocol["evaluation_joint_shapes"]
        for seed in protocol["evaluation_mask_seeds"]]


def joint_mask(data, scenario):
    _, rate, shape, seed = scenario.split("_")
    drop = np.zeros_like(data["observed"], dtype=bool)
    for i, sid in enumerate(data["ids"]):
        rng = np.random.default_rng(stable_seed(sid, scenario, int(seed)))
        pos = select_positions(data["valid"][i], int(rate) / 100, shape, rng)
        drop[i, pos, :] = True
    return drop


def summarize(scores, protocol):
    keys = ("accuracy", "macro_f1", "mae", "pearson")
    result = {"clean": {k: scores["clean"][k] for k in keys}}
    sets = {"single": MAIN}
    for group, rates in protocol["groups"].items():
        sets[group] = [s for s in scores if s.startswith("joint_") and int(s.split("_")[1]) in rates]
    sets["low_medium"] = sets["low_joint"] + sets["medium_joint"]
    for group, names in sets.items():
        result[group] = {key: float(np.mean([scores[s][key] for s in names]))
                         if all(scores[s][key] is not None for s in names) else None for key in keys}
    result["score"] = .5 * result["low_medium"]["macro_f1"] + .5 * (1 - result["low_medium"]["mae"] / 6)
    return result


def mean_summary(records):
    return {key: float(np.mean([r[key] for r in records])) if key == "score" else {
        metric: float(np.mean([r[key][metric] for r in records]))
        if all(r[key][metric] is not None for r in records) else None
        for metric in records[0][key]} for key in records[0]}


def decision(reference, candidate, protocol):
    a, b = mean_summary(reference), mean_summary(candidate)
    g = protocol["guardrails"]
    checks = {
        "clean_f1": b["clean"]["macro_f1"] >= a["clean"]["macro_f1"] - g["clean_f1_drop"],
        "clean_accuracy": b["clean"]["accuracy"] >= a["clean"]["accuracy"] - g["clean_accuracy_drop"],
        "clean_mae": b["clean"]["mae"] <= a["clean"]["mae"] + g["clean_mae_increase"],
        "single_f1": b["single"]["macro_f1"] >= a["single"]["macro_f1"] - g["single_f1_drop"],
        "single_mae": b["single"]["mae"] <= a["single"]["mae"] + g["single_mae_increase"],
        "mean_score": b["score"] - a["score"] >= .002,
        "paired_successes": sum(y["score"] > x["score"] for x, y in zip(reference, candidate)) >= 2,
    }
    for group in protocol["groups"]:
        checks[group + "_f1"] = b[group]["macro_f1"] >= a[group]["macro_f1"] - g["joint_group_f1_drop"]
        checks[group + "_mae"] = b[group]["mae"] <= a[group]["mae"] + g["joint_group_mae_increase"]
    return {"passed": all(checks.values()), "checks": checks, "reference_mean": a,
            "candidate_mean": b, "paired_score_differences": [y["score"] - x["score"] for x, y in zip(reference, candidate)]}
