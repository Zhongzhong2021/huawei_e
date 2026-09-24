"""Portable weight-only quantization; computation remains FP32 after loading.

No activation calibration and no external model download are required. This is
file-size compression, NOT a claim of faster or lower-memory FP32 inference.
"""
from pathlib import Path
import torch
from torch.nn import functional as F


def encode_tensor(weight, bits=8, group_size=None):
    if weight.ndim != 2 or not weight.is_floating_point() or bits not in (4, 8):
        raise ValueError("Expected a floating matrix and 4 or 8 bits")
    weight = weight.detach().cpu().float()
    rows, columns = weight.shape
    group_size = int(group_size or columns)
    if group_size < 1:
        raise ValueError("group_size must be positive")
    padded = F.pad(weight, (0, (-columns) % group_size))
    grouped = padded.reshape(rows, -1, group_size)
    qmax = 2 ** (bits-1) - 1
    scale = grouped.abs().amax(-1, keepdim=True) / qmax
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    integer = (grouped / scale).round().clamp(-qmax, qmax).to(torch.int8)
    if bits == 8:
        packed = integer
    else:
        unsigned = (integer.flatten().to(torch.int16) + 8).to(torch.uint8)
        if unsigned.numel() % 2:
            unsigned = F.pad(unsigned, (0, 1), value=8)
        packed = unsigned[::2] | (unsigned[1::2] << 4)
    return {"bits": bits, "shape": [rows, columns], "group_size": group_size,
            "group_shape": list(grouped.shape), "scale": scale, "packed": packed}


def decode_tensor(record):
    if record["bits"] == 8:
        integer = record["packed"].float()
    elif record["bits"] == 4:
        packed = record["packed"]
        pairs = torch.stack([packed & 15, packed >> 4], dim=1).flatten().to(torch.int16) - 8
        size = 1
        for n in record["group_shape"]:
            size *= n
        integer = pairs[:size].reshape(record["group_shape"]).float()
    else:
        raise ValueError("Unsupported quantization bits")
    return (integer * record["scale"]).reshape(record["shape"][0], -1)[:, :record["shape"][1]].contiguous()


def compress_checkpoint(source, destination, mode="int8"):
    if mode not in ("int8", "int8g128", "int8g32", "mixed4"):
        raise ValueError("Unsupported weight-only quantization mode")
    original = torch.load(source, map_location="cpu", weights_only=True)
    compressed = {k: v for k,v in original.items() if k != "state_dict"}
    compressed.update(format="q2-weight-only-v1", quantization=mode, tensors={})
    for name, tensor in original["state_dict"].items():
        if tensor.ndim == 2 and tensor.is_floating_point():
            bits = 4 if mode == "mixed4" and name.startswith("backbone.blocks.") else 8
            group_size = 128 if bits == 4 or mode == "int8g128" else 32 if mode == "int8g32" else None
            compressed["tensors"][name] = encode_tensor(tensor, bits, group_size)
        else:
            compressed["tensors"][name] = tensor.detach().cpu().clone()
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    torch.save(compressed, destination)
    return {"mode": mode, "source_bytes": Path(source).stat().st_size,
            "compressed_bytes": Path(destination).stat().st_size}


def unpack_checkpoint(record):
    if record.get("format") != "q2-weight-only-v1":
        return record
    result = {k: v for k,v in record.items() if k != "tensors"}
    result["state_dict"] = {name: decode_tensor(value) if isinstance(value, dict) else value
                            for name,value in record["tensors"].items()}
    return result
