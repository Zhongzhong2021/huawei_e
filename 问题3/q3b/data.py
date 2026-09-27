"""Aligned feature loading, train-only statistics, and fixed-window batches.

Field organization follows MMSA's MMDataset (MIT, commit a94e65d): read one
pickle split and expose its modalities, labels, IDs, and raw text per sample.
The masks, statistics, and window packing here are specific to Plan B.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import torch

FIELDS = {"T": ("text", 768), "A": ("audio", 74), "V": ("vision", 35)}
SPECIAL_IDS = (0, 101, 102)
PROCESSING_VERSION = "q3b-data-2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sample(record: dict, index: int | None, path: Path, split: str | None) -> dict:
    # Keep source precision until masks and normalization are complete: a tiny
    # nonzero float64 row must not become "missing" through early float32 cast.
    item = {key: np.asarray(record[key][index] if index is not None else record[key])
            for key, _ in FIELDS.values()}
    for key, dim in FIELDS.values():
        if item[key].ndim != 2 or item[key].shape[1] != dim:
            raise ValueError(f"{path}: {key} must have shape [length,{dim}]")
        if not np.isfinite(item[key]).all():
            raise ValueError(f"{path}: non-finite {key} at sample {index}")
    length = item["text"].shape[0]
    if any(item[key].shape[0] != length for key in ("audio", "vision")):
        raise ValueError(f"{path}: modality lengths disagree at sample {index}")
    bert = np.asarray(record["text_bert"][index] if index is not None else record["text_bert"])
    if bert.shape != (3, length) or not np.issubdtype(bert.dtype, np.integer):
        raise ValueError(f"{path}: text_bert must have shape [3,{length}] and integer IDs")
    if not np.isin(bert[1], (0, 1)).all():
        raise ValueError(f"{path}: attention mask must contain only 0/1")
    item["text_bert"] = bert.astype(np.int64, copy=False)
    for key in ("id", "raw_text"):
        value = record[key][index] if index is not None else record[key]
        item[key] = str(value)
    if "regression_labels" in record or "classification_labels" in record:
        if not {"regression_labels", "classification_labels"} <= record.keys():
            raise ValueError(f"{path}: incomplete labels at sample {index}")
        y = float(record["regression_labels"][index] if index is not None else record["regression_labels"])
        c = float(record["classification_labels"][index] if index is not None else record["classification_labels"])
        if not np.isfinite(y) or not -3 <= y <= 3 or not np.isfinite(c) or c not in (0, 1, 2):
            raise ValueError(f"{path}: invalid labels at sample {index}: {y}, {c}")
        if int(c) != (0 if y < 0 else 2 if y > 0 else 1):
            raise ValueError(f"{path}: label sign mismatch at sample {index}")
        item.update(y=y, class_label=int(c), regression_labels=y, classification_labels=int(c))
    item["_source_path"] = str(path.resolve())
    item["_split"] = split
    return item


def _from_split(record: dict, path: Path, split: str) -> list[dict]:
    required = {"text", "audio", "vision", "text_bert", "id", "raw_text"}
    if not required <= record.keys():
        raise ValueError(f"{path}: missing fields {sorted(required - record.keys())}")
    count = len(record["id"])
    if any(len(record[key]) != count for key in required):
        raise ValueError(f"{path}: inconsistent sample count in {split}")
    return [_sample(record, i, path, split) for i in range(count)]


def load_samples(path: Path, split: str | None = None) -> list[dict]:
    """Load an aligned split, a special pickle, or a directory of special pickles."""
    path = Path(path)
    if path.is_dir():
        if split is not None:
            raise ValueError("special directory has no split")
        files = sorted(path.glob("[0-9][0-9].pkl"))
        if [file.stem for file in files] != [f"{i:02d}" for i in range(1, 21)]:
            raise ValueError("special directory must contain 01.pkl through 20.pkl exactly once")
        samples = [load_samples(file)[0] for file in files]
        if [sample["id"] for sample in samples] != [file.stem for file in files]:
            raise ValueError("special sample ID does not match filename")
        return samples
    with path.open("rb") as stream:
        data = pickle.load(stream)
    if split is not None:
        if split not in ("train", "valid", "test") or split not in data:
            raise ValueError(f"unknown aligned split: {split}")
        return _from_split(data[split], path, split)
    if any(name in data for name in ("train", "valid", "test")):
        raise ValueError("aligned pickle requires split=train/valid/test")
    return [_sample(data, None, path, None)]


def _masks(sample: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    ids, attention = sample["text_bert"][:2]
    content = attention.astype(bool) & ~np.isin(ids, SPECIAL_IDS)
    observed = {"T": content}
    for modality, (key, _) in FIELDS.items():
        if modality != "T":
            observed[modality] = content & np.any(sample[key] != 0, axis=1)
    return content, observed


def fit_stats(train: list[dict]) -> dict:
    """Population mean/std of train's observed rows; provenance is mandatory."""
    if not train or any(sample.get("_split") != "train" for sample in train):
        raise ValueError("fit_stats accepts only a nonempty train split")
    paths = {sample.get("_source_path") for sample in train}
    if len(paths) != 1 or None in paths:
        raise ValueError("train samples must come from one source file")
    result = {"processing_version": PROCESSING_VERSION, "split": "train",
              "source_path": paths.pop(), "source_sha256": _sha256(Path(train[0]["_source_path"])),
              "modalities": {}}
    for modality, (key, dim) in FIELDS.items():
        count = 0
        total = np.zeros(dim, dtype=np.float64)
        square = np.zeros(dim, dtype=np.float64)
        for sample in train:
            rows = sample[key][_masks(sample)[1][modality]]
            count += len(rows)
            # Reduce a sample at a time; never hold a float64 copy of the dataset.
            total += rows.sum(axis=0, dtype=np.float64)
            square += np.square(rows, dtype=np.float64).sum(axis=0)
        mean = total / count if count else total
        scale = np.sqrt(np.maximum(square / count - mean * mean, 0)) if count else np.ones(dim)
        scale[scale < 1e-6] = 1
        result["modalities"][modality] = {"mean": mean.tolist(), "scale": scale.tolist(),
                                          "count": count}
    return result


def make_batch(samples: list[dict], stats: dict, window_size: int, device: str | torch.device) -> dict:
    if not samples or window_size < 1:
        raise ValueError("batch must be nonempty and window_size positive")
    if stats.get("split") != "train" or stats.get("processing_version") != PROCESSING_VERSION:
        raise ValueError("expected Plan B train statistics")
    b, w = len(samples), window_size
    length = samples[0]["text"].shape[0]
    if any(sample["text"].shape[0] != length for sample in samples):
        raise ValueError("batch sequence lengths disagree")
    k = (length + w - 1) // w
    x = {m: np.zeros((b, k, w, dim), dtype=np.float32) for m, (_, dim) in FIELDS.items()}
    observed = {m: np.zeros((b, k, w), dtype=bool) for m in FIELDS}
    content_batch = np.zeros((b, k, w), dtype=bool)
    source_index = np.full((b, k, w), -1, dtype=np.int64)
    weight = np.zeros((b, k), dtype=np.float32)
    meta = []
    for i, sample in enumerate(samples):
        content, masks = _masks(sample)
        indices = np.flatnonzero(content)
        n = len(indices)
        source_index[i].flat[:n] = indices
        content_batch[i].flat[:n] = True
        for m, (key, dim) in FIELDS.items():
            mask = masks[m][indices]
            observed[m][i].flat[:n] = mask
            norm = stats["modalities"][m]
            mean = np.asarray(norm["mean"], dtype=np.float64)
            scale = np.asarray(norm["scale"], dtype=np.float64)
            if mean.shape != (dim,) or scale.shape != (dim,) or np.any(scale <= 0):
                raise ValueError(f"invalid {m} train statistics")
            positions = np.flatnonzero(mask)
            x[m][i].reshape(k * w, dim)[positions] = (sample[key][indices[positions]] - mean) / scale
        if n:
            weight[i] = content_batch[i].sum(axis=1) / n
        ids, attention = sample["text_bert"][:2]
        meta.append({"id": sample["id"], "raw_text": sample["raw_text"],
                     "token_ids": ids.tolist(), "truncated": bool(attention[-1]),
                     "empty_content": n == 0, "content_count": n,
                     "observation_counts": {m: int(masks[m].sum()) for m in FIELDS},
                     "source_path": sample.get("_source_path")})
    if any(np.any(observed[m] & ~content_batch) or np.any(x[m][~observed[m]] != 0) for m in FIELDS):
        raise AssertionError("observed/content or zero-reference invariant failed")
    if not np.allclose(weight.sum(axis=1), (content_batch.sum(axis=(1, 2)) > 0).astype(float)):
        raise AssertionError("window weights do not sum to one")
    batch = {"x": {m: torch.as_tensor(a, device=device) for m, a in x.items()},
             "bert_inputs": torch.as_tensor(np.stack([s["text_bert"] for s in samples]),
                                              device=device),
             "observed": {m: torch.as_tensor(a, device=device) for m, a in observed.items()},
             "content": torch.as_tensor(content_batch, device=device),
             "window_weight": torch.as_tensor(weight, device=device),
             "source_index": torch.as_tensor(source_index, device=device), "meta": meta}
    labeled = ["y" in sample for sample in samples]
    if any(labeled) and not all(labeled):
        raise ValueError("cannot mix labeled and unlabeled samples")
    if all(labeled):
        batch["y"] = torch.tensor([sample["y"] for sample in samples], dtype=torch.float32, device=device)
        batch["class_label"] = torch.tensor([sample["class_label"] for sample in samples], dtype=torch.int64, device=device)
    return batch


def _audit_group(samples: list[dict]) -> dict:
    lengths = []
    zero = {m: 0 for m in FIELDS}
    labels = {str(i): 0 for i in range(3)}
    special_ids = set()
    truncated = []
    for sample in samples:
        content, masks = _masks(sample)
        lengths.append(int(content.sum()))
        special_ids.update(int(x) for x in np.unique(sample["text_bert"][0]) if x in SPECIAL_IDS)
        for m in FIELDS:
            zero[m] += not masks[m].any()
        if "class_label" in sample:
            labels[str(sample["class_label"])] += 1
        if sample["text_bert"][1, -1]:
            truncated.append(sample["id"])
    return {"count": len(samples), "label_counts": labels if samples and "y" in samples[0] else None,
            "zero_observation_samples": zero, "content_length": {"min": min(lengths),
            "max": max(lengths), "mean": float(np.mean(lengths))},
            "special_ids_seen": sorted(special_ids), "full_length_truncation_clues": truncated}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("audit", "fit-stats"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    aligned = Path(config["data"]["aligned_path"])
    special = Path(config["data"]["special_dir"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "fit-stats":
        result = fit_stats(load_samples(aligned, "train"))
    else:
        result = {"processing_version": PROCESSING_VERSION, "aligned_sha256": _sha256(aligned),
                  "splits": {name: _audit_group(load_samples(aligned, name)) for name in ("train", "valid", "test")},
                  "special": _audit_group(load_samples(special)),
                  "special_sha256": {p.name: _sha256(p) for p in sorted(special.glob("[0-9][0-9].pkl"))}}
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
