"""Standalone compact-checkpoint inference, including raw attachment-3 PKLs."""
import argparse
from pathlib import Path
import numpy as np
import torch
from q2.data import convert, load_pickle, normalize, save_json, digest
from q2.engine import infer, write_predictions
from .model import PretrainedMultimodal
from .quantization import unpack_checkpoint


def load_deployment(path, device="cpu"):
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    record = unpack_checkpoint(torch.load(path, map_location="cpu", weights_only=True))
    model = PretrainedMultimodal(record["config"], record["vocabulary"])
    model.load_state_dict(record["state_dict"], strict=True)
    return model.to(device).eval(), record


def raw_inputs(input_dir, scaler):
    parts = []
    for path in sorted(Path(input_dir).glob("*.pkl")):
        raw = load_pickle(path)["test"]
        part, _ = convert(raw, [f"{path.name}::row{i:03d}" for i in range(len(raw["text_bert"]))])
        parts.append(part)
    if not parts:
        raise ValueError("No .pkl files in input directory")
    data = {k: np.concatenate([part[k] for part in parts]) for k in parts[0]}
    with np.load(scaler, allow_pickle=False) as stats:
        return normalize(data, stats)


def predict(checkpoint, scaler, input_dir, output, device="cpu"):
    model, _ = load_deployment(checkpoint, device)
    data = raw_inputs(input_dir, scaler)
    p, y = infer(model, data, device, batch_size=64)
    single_p, single_y = infer(model, data, device, batch_size=1)
    again, _ = load_deployment(checkpoint, device)
    reload_p, reload_y = infer(again, data, device, batch_size=64)
    assert np.isfinite(p).all() and np.isfinite(y).all()
    assert (np.abs(y) <= 3).all() and np.allclose(p.sum(1), 1, atol=1e-6)
    assert np.allclose(p, single_p, atol=1e-5, rtol=0) and np.allclose(y, single_y, atol=1e-5, rtol=0)
    assert np.array_equal(p.argmax(1), single_p.argmax(1)), "Single/batch class mismatch"
    assert np.array_equal(p, reload_p) and np.array_equal(y, reload_y)
    assert len(set(data["ids"])) == len(data["ids"])
    write_predictions(output, data, p, y)
    check = {"rows": len(y), "unique_ids":len(set(data["ids"])), "device":str(device),
             "single_batch_max_probability_diff":float(np.abs(p-single_p).max()),
             "single_batch_max_intensity_diff":float(np.abs(y-single_y).max()),
             "single_batch_class_equal":True, "reload_exact_match":True, "finite":True,
             "checkpoint_sha256":digest(checkpoint), "scaler_sha256":digest(scaler), "csv_sha256":digest(output)}
    save_json(Path(output).with_suffix(".checks.json"), check)
    return check


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scaler", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    arguments = parser.parse_args()
    print(predict(arguments.checkpoint, arguments.scaler, arguments.input_dir, arguments.output, arguments.device))
