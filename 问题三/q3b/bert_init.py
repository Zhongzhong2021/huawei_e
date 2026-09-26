"""Initialize BERT locally for training or strict checkpoint restoration."""
from pathlib import Path

from transformers import AutoModel, BertConfig, BertModel


def load_bert(path: str):
    source = Path(path).expanduser().resolve()
    config = source / "config.json"
    if not config.is_file():
        raise FileNotFoundError(f"local BERT config missing: {config}")
    if any((source / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")):
        return AutoModel.from_pretrained(str(source), local_files_only=True)
    return BertModel(BertConfig.from_json_file(str(config)))
