from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from q2.data import convert, fit_scaler, normalize, load_pickle, save_json, digest


def subset(data, indices):
    return {k: v[indices] for k,v in data.items()}


def prepare_folds(root):
    root = Path(root)
    raw = load_pickle(root.parent / "data/raw/aligned_50.pkl")["train"]
    data, _ = convert(raw, raw["id"])
    groups = np.array([str(x).split("$_$")[0] for x in data["ids"]])
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=240924)
    records = []
    for i, (train_idx, valid_idx) in enumerate(splitter.split(data["tokens"], data["labels"], groups)):
        fitting, validation = subset(data, train_idx), subset(data, valid_idx)
        assert not set(groups[train_idx]) & set(groups[valid_idx])
        stats = fit_scaler(fitting)
        out = root / "folds" / f"fold{i}"
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "train.npz", **normalize(fitting, stats))
        np.savez_compressed(out / "valid.npz", **normalize(validation, stats))
        np.savez_compressed(out / "scaler.npz", **stats)
        vocabulary = np.unique(np.concatenate([fitting["tokens"].ravel(), [0,100,101,102,103]]))
        np.save(out / "vocabulary.npy", vocabulary)
        record = {"fold": i, "train_ids": fitting["ids"].tolist(), "valid_ids": validation["ids"].tolist(),
                  "train_count": len(train_idx), "valid_count": len(valid_idx),
                  "train_class_counts": np.bincount(fitting["labels"], minlength=3).tolist(),
                  "valid_class_counts": np.bincount(validation["labels"], minlength=3).tolist(),
                  "group_overlap": 0, "vocabulary_size": len(vocabulary),
                  "scaler_sha256": digest(out / "scaler.npz"), "vocabulary_source": "fold fitting subset only"}
        save_json(out / "manifest.json", record)
        records.append({k:v for k,v in record.items() if not k.endswith("_ids")})
    save_json(root / "study/folds.json", records)
    print(records, flush=True)


def load_fold(root, fold, split):
    with np.load(Path(root) / "folds" / f"fold{fold}" / f"{split}.npz", allow_pickle=False) as f:
        return {k: f[k] for k in f.files}
