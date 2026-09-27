"""Describe attachment-3 missingness from its aligned inputs, without reference recovery."""
import argparse
from collections import Counter
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "问题2/src"))
from q2.data import convert, digest, load_pickle, save_json


def profile(directory):
    rows = []
    for path in sorted(Path(directory).glob("*.pkl")):
        raw = load_pickle(path)["test"]
        ids = [f"{path.name}::row{i:03d}" for i in range(len(raw["text_bert"]))]
        # Only released input channels enter this descriptive audit.
        inputs = {key: raw[key] for key in ("text_bert", "audio", "vision")}
        data, _ = convert(inputs, ids)
        for i, sid in enumerate(ids):
            valid, unknown, observed = data["valid"][i], data["text_unknown"][i], data["observed"][i]
            positions = np.flatnonzero(unknown)
            runs = []
            for pos in positions:
                if runs and runs[-1][1] == pos:
                    runs[-1][1] = int(pos + 1)
                else:
                    runs.append([int(pos), int(pos + 1)])
            length = int(valid.sum())
            rows.append({
                "sample_id": sid, "file_sha256": digest(path), "valid_positions": length,
                "unknown_positions": positions.tolist(), "unknown_count": len(positions),
                "unknown_rate": len(positions) / length if length else None,
                "unknown_runs_half_open": runs,
                "unknown_and_av_unobserved": int((unknown & ~observed[:, 1] & ~observed[:, 2]).sum()),
                "audio_unobserved_count": int((valid & ~observed[:, 1]).sum()),
                "vision_unobserved_count": int((valid & ~observed[:, 2]).sum()),
            })
    if not rows:
        raise ValueError("No aligned attachment files found")
    rates = [r["unknown_rate"] for r in rows if r["unknown_rate"] is not None]
    affected = [r["unknown_rate"] for r in rows if r["unknown_count"]]
    lengths = Counter(end - start for r in rows for start, end in r["unknown_runs_half_open"])
    valid_count = sum(r["valid_positions"] for r in rows)
    unknown_count = sum(r["unknown_count"] for r in rows)
    return {
        "scope": "Descriptive input audit; no source labels or cross-version recovery",
        "samples": len(rows), "affected_samples": len(affected),
        "valid_positions": valid_count, "unknown_positions": unknown_count,
        "unknown_and_av_unobserved": sum(r["unknown_and_av_unobserved"] for r in rows),
        "pooled_unknown_rate": unknown_count / valid_count if valid_count else None,
        "sample_mean_unknown_rate": float(np.mean(rates)) if rates else None,
        "affected_sample_median_unknown_rate": float(np.median(affected)) if affected else None,
        "run_length_counts": {str(k): v for k, v in sorted(lengths.items())},
        "rows": rows, "script_sha256": digest(Path(__file__)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new profile output")
    save_json(args.output, profile(args.input))
