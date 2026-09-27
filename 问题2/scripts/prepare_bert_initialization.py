"""Verify and convert the pinned public BERT checkpoint to the local model interface."""
import argparse
from pathlib import Path
import sys

import numpy as np
from safetensors.torch import load_file
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from q2.data import digest, save_json
from q2v2.model import CompactBert

REPOSITORY = "google-bert/bert-base-uncased"
REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"
SHA256 = "68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--safetensors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new output directory")
    if digest(args.safetensors) != SHA256:
        raise ValueError("Pinned pretrained checkpoint SHA256 mismatch")
    original = load_file(str(args.safetensors), device="cpu")
    source = {k.removeprefix("bert.").replace(".LayerNorm.gamma", ".LayerNorm.weight")
              .replace(".LayerNorm.beta", ".LayerNorm.bias"): v for k, v in original.items()
              if k.startswith(("bert.embeddings.", "bert.encoder."))}
    # Check all mapped keys, shapes and layer coverage before writing anything.
    model = CompactBert(np.arange(30522), list(range(12)))
    model.initialize(source)
    if len(source) != 197:
        raise ValueError(f"Unexpected BERT source tensor count: {len(source)}")
    args.output.mkdir(parents=True, exist_ok=False)
    converted = args.output / "bert_base_uncased.pt"
    torch.save(source, converted)
    restored = torch.load(converted, map_location="cpu", weights_only=True)
    if set(source) != set(restored) or any(not torch.equal(v, restored[k]) for k, v in source.items()):
        raise ValueError("Converted weights changed on reload")
    save_json(args.output / "initialization.json", {
        "repository": REPOSITORY, "revision": REVISION,
        "source_url": f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/model.safetensors",
        "source_sha256": SHA256, "converted_sha256": digest(converted),
        "source_tensors": len(source), "mapping": "Remove bert. prefix; LayerNorm gamma/beta to weight/bias; retain embeddings and encoder",
        "unused_heads": "pooler and masked-language-model head",
        "initialization_check": "12 layers and full vocabulary loaded; serialized tensors exactly equal",
    })
    print(converted)


if __name__ == "__main__":
    main()
