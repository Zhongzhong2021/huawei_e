"""Stable missing scenarios, applied before any text embedding or encoder."""
import hashlib
import numpy as np

MAIN = [f"{m}_{r}_random" for m in ["text", "audio", "vision"] for r in [10, 30, 50]]
SUPPLEMENT = [f"{m}_{r}_{shape}" for m in ["text", "audio", "vision"]
              for r in [10, 30, 50] for shape in ["front", "middle", "back", "multi"]]


def stable_seed(sample_id, scenario, seed=20260924):
    return int.from_bytes(hashlib.sha256(f"{seed}|{sample_id}|{scenario}".encode()).digest()[:8], "little")


def select_positions(valid, rate, shape, rng):
    idx = np.flatnonzero(valid)
    if not len(idx):
        return idx
    count = min(len(idx), max(1, int(np.floor(len(idx) * rate + 0.5))))
    if shape == "scattered":
        return np.sort(rng.choice(idx, count, replace=False))
    if shape == "multi":
        # Distribute exactly count removed positions over up to three short runs.
        groups = np.array_split(idx, min(3, len(idx)))
        counts = np.full(len(groups), count // len(groups))
        counts[:count % len(groups)] += 1
        selected = []
        for g, c in zip(groups, counts):
            start = int(rng.integers(0, len(g) - c + 1)) if c else 0
            selected.extend(g[start:start + c])
        return np.asarray(sorted(selected), dtype=np.int64)
    choices = {"front": 0, "middle": (len(idx) - count) // 2, "back": len(idx) - count}
    start = choices[shape] if shape in choices else int(rng.integers(0, len(idx) - count + 1))
    return idx[start:start + count]


def scenario_mask(data, scenario):
    drop = np.zeros_like(data["observed"], dtype=bool)
    if scenario == "clean":
        return drop
    name, percent, shape = scenario.split("_")
    modality = ["text", "audio", "vision"].index(name)
    for i, sample_id in enumerate(data["ids"]):
        rng = np.random.default_rng(stable_seed(sample_id, scenario))
        pos = select_positions(data["valid"][i], int(percent) / 100, shape, rng)
        drop[i, pos, modality] = True
    return drop


def augmentation_mask(valid, rng, mode, probability=0.7):
    drop = np.zeros((*valid.shape, 3), dtype=bool)
    if mode == "none":
        return drop
    shape = "scattered" if mode == "scattered" else "random"
    for i, row in enumerate(valid):
        if rng.random() < probability:
            modality = int(rng.integers(3))
            pos = select_positions(row, float(rng.choice([.1, .3, .5])), shape, rng)
            drop[i, pos, modality] = True
    return drop
