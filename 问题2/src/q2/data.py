"""One strict input contract for official splits and attachment 3."""
import hashlib
import importlib
import json
import pickle
from pathlib import Path
import numpy as np

CLASSES = ["Negative", "Neutral", "Positive"]
MODALITIES = ["text", "audio", "vision"]


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class NumpyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {"numpy": {"ndarray", "dtype", "asarray"},
                   "numpy.core.multiarray": {"_reconstruct", "scalar"},
                   "numpy._core.multiarray": {"_reconstruct", "scalar"},
                   "numpy.core.numeric": {"_frombuffer"},
                   "numpy._core.numeric": {"_frombuffer"}}
        if name not in allowed.get(module, set()):
            raise pickle.UnpicklingError(f"Disallowed global: {module}.{name}")
        try:
            return getattr(importlib.import_module(module), name)
        except ModuleNotFoundError:
            return getattr(importlib.import_module(module.replace("numpy._core", "numpy.core")), name)


def load_pickle(path):
    with open(path, "rb") as f:
        return NumpyUnpickler(f).load()


def convert(raw, ids):
    bert = np.asarray(raw["text_bert"])
    if bert.ndim != 3 or bert.shape[1:] != (3, 50):
        raise ValueError(f"Invalid text_bert shape: {bert.shape}")
    if not np.isfinite(bert).all() or not np.equal(bert, np.round(bert)).all():
        raise ValueError("text_bert must contain finite integers")
    bert = bert.astype(np.int64)
    tokens, attention, segment = bert[:, 0], bert[:, 1], bert[:, 2]
    if tokens.min() < 0 or tokens.max() >= 30522:
        raise ValueError("Token IDs outside bert-base-uncased vocabulary")
    if not np.isin(attention, [0, 1]).all() or not np.isin(segment, [0, 1]).all():
        raise ValueError("Invalid attention/segment channels")
    n, length = tokens.shape
    audio = np.asarray(raw["audio"], dtype=np.float32)
    vision = np.asarray(raw["vision"], dtype=np.float32)
    if audio.shape != (n, length, 74) or vision.shape != (n, length, 35):
        raise ValueError("Only aligned_50 interface is supported")
    evidence = (attention == 1) | (tokens != 0)
    evidence |= np.any(np.isfinite(audio) & (audio != 0), axis=-1)
    evidence |= np.any(np.isfinite(vision) & (vision != 0), axis=-1)
    positions = np.arange(length)[None, :]
    end = np.max(np.where(evidence, positions, -1), axis=1)
    # Internal zero runs remain valid positions. Boundary ambiguity is recorded.
    valid = (positions <= end[:, None]) & (positions > 0)
    valid &= (tokens != 101) & (tokens != 102)
    text_obs = valid & (attention == 1) & (tokens != 0)
    observations = [text_obs]
    audit = {"n": n, "all_empty_rows": int((end < 0).sum()),
             "boundary_without_sep": int((~np.any(tokens == 102, axis=1)).sum()),
             "attention_id_disagreement": int(((attention == 0) & (tokens != 0)).sum()),
             "valid_positions": int(valid.sum()), "modalities": {}}
    for name, x in [("audio", audio), ("vision", vision)]:
        finite = np.isfinite(x).all(axis=-1)
        zero = np.all(x == 0, axis=-1)
        # Statement defines ALL-ZERO feature vectors as missing; individual zero
        # coordinates, padding and known special-token positions are not missing.
        obs = valid & finite & ~zero
        observations.append(obs)
        audit["modalities"][name] = {
            "observed": int(obs.sum()), "interior_all_zero": int((valid & zero).sum()),
            "nonfinite_vectors": int((valid & ~finite).sum()),
            "nonzero_outside_valid": int((~valid & ~zero).sum())}
    audit["modalities"]["text"] = {"observed": int(text_obs.sum()),
                                          "missing_inside_extent": int((valid & ~text_obs).sum())}
    result = {"tokens": tokens, "audio": np.nan_to_num(audio, nan=0., posinf=0., neginf=0.),
              "vision": np.nan_to_num(vision, nan=0., posinf=0., neginf=0.),
              "valid": valid, "observed": np.stack(observations, axis=-1),
              "ids": np.asarray(ids, dtype=str)}
    if len(ids) != n or len(set(map(str, ids))) != n:
        raise ValueError("Sample IDs must be unique and cover all rows")
    if "classification_labels" in raw:
        cls = np.asarray(raw["classification_labels"])
        y = np.asarray(raw["regression_labels"], dtype=np.float32)
        if not np.isfinite(cls).all() or not np.isin(cls, [0, 1, 2]).all():
            raise ValueError("Invalid classification labels")
        if not np.isfinite(y).all() or np.any(np.abs(y) > 3):
            raise ValueError("Invalid regression labels")
        expected = np.where(y < 0, 0, np.where(y > 0, 2, 1))
        if not np.array_equal(cls, expected):
            raise ValueError("Polarity/sign label conflict")
        result.update(labels=cls.astype(np.int64), targets=y)
        audit["class_counts"] = np.bincount(result["labels"], minlength=3).tolist()
    return result, audit


def fit_scaler(train):
    stats = {}
    for m, name in enumerate(MODALITIES[1:], 1):
        values = train[name][train["observed"][:, :, m]].astype(np.float64)
        if not len(values):
            raise ValueError(f"No observed training values for {name}")
        stats[name + "_mean"] = values.mean(axis=0).astype(np.float32)
        stats[name + "_std"] = np.maximum(values.std(axis=0), 1e-5).astype(np.float32)
    return stats


def normalize(data, stats):
    data = dict(data)
    for m, name in enumerate(MODALITIES[1:], 1):
        x = (data[name] - stats[name + "_mean"]) / stats[name + "_std"]
        # Bounding extreme but finite measurements is a fixed preprocessing rule.
        x = np.clip(x, -10, 10)
        data[name] = np.where(data["observed"][:, :, m, None], x, 0).astype(np.float32)
    return data


def prepare(root, source, special):
    root, source, special = Path(root), Path(source), Path(special)
    out = root / "data/processed"
    out.mkdir(parents=True, exist_ok=True)
    raw = load_pickle(source)
    datasets, audits = {}, {}
    for split in ["train", "valid", "test"]:
        datasets[split], audits[split] = convert(raw[split], raw[split]["id"])
    for a, b in [("train", "valid"), ("train", "test"), ("valid", "test")]:
        if set(datasets[a]["ids"]) & set(datasets[b]["ids"]):
            raise ValueError(f"Sample ID leakage between {a} and {b}")
    stats = fit_scaler(datasets["train"])
    np.savez_compressed(out / "scaler.npz", **stats)
    file_records, parts = [], []
    for path in sorted(special.glob("*.pkl")):
        obj = load_pickle(path)["test"]
        n = len(obj["text_bert"])
        part, audits[path.name] = convert(obj, [f"{path.name}::row{i:03d}" for i in range(n)])
        parts.append(part)
        file_records.append({"file": path.name, "rows": n, "sha256": digest(path)})
    if not parts:
        raise ValueError("No attachment 3 files found")
    datasets["attachment3"] = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    for split, ds in datasets.items():
        np.savez_compressed(out / f"{split}.npz", **normalize(ds, stats))
    manifest = {"source_sha256": digest(source), "special_files": file_records,
                "classes": CLASSES, "audits": audits,
                "mask_policy": "Infer extent from union evidence; exclude position 0/CLS/SEP; entire zero A/V vector inside extent is unavailable per problem statement. Individual zero coordinates remain valid. Unknown trailing all-modal missing cannot be distinguished from padding; boundary_without_sep flags it.",
                "scaler_fit_split": "train", "clip_standardized": [-10, 10],
                "processed_sha256": {p.name: digest(p) for p in sorted(out.glob("*.npz"))}}
    save_json(root / "audit/data_audit.json", manifest)
    return manifest


def load_data(root, split):
    with np.load(Path(root) / "data/processed" / f"{split}.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
