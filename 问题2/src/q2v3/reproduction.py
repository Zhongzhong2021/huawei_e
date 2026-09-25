"""Read-only preflight for a fresh, fixed-epoch reproduction run."""
import json
import math
from pathlib import Path
import re


def preflight(root, name, seed, data_root=None, pretrained=None, checkpoint=None, protocol=None):
    root = Path(root).resolve()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", name):
        raise ValueError("Run name must be 1-80 ASCII letters/digits/dots/dashes/underscores, starting with a letter or digit")
    if name.split('.')[0].upper() in {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1, 10)], *[f"LPT{i}" for i in range(1, 10)]}:
        raise ValueError("Run name is reserved on Windows")
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32:
        raise ValueError("Seed must be an integer in [0, 2**32)")
    if (root / 'runs' / name).exists():
        raise FileExistsError("Run directories are immutable; select a fresh --name")
    data_root = Path(data_root).resolve() if data_root is not None else root.parent
    paths = {
        'checkpoint': Path(checkpoint) if checkpoint is not None else root / 'final/model.pt',
        'pretrained': Path(pretrained) if pretrained is not None else root.parent / 'round2/pretrained/bert_base_uncased.pt',
        'train_data': data_root / 'data/processed/train.npz',
        'valid_data': data_root / 'data/processed/valid.npz',
    }
    if protocol is None:
        candidates = [root / 'study/protocol.json', root / 'configs/frozen_protocol.json']
        paths['protocol'] = next((p for p in candidates if p.is_file()), candidates[-1])
    else:
        paths['protocol'] = Path(protocol)
    missing = [f"{key}: {path}" for key, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Reproduction inputs are incomplete (no training started):\n" + '\n'.join(missing))
    record = json.loads(paths['protocol'].read_text(encoding='utf-8'))
    settings = record.get('training', {})
    for key in ['batch_size', 'max_epochs', 'patience']:
        value = settings.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"Invalid protocol training.{key}: positive integer required")
    for key in ['backbone_lr', 'head_lr', 'weight_decay', 'gradient_clip']:
        value = settings.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or (value < 0 if key == 'weight_decay' else value <= 0):
            raise ValueError(f"Invalid protocol training.{key}")
    return {
        'name': name, 'seed': seed, 'data_root': str(data_root),
        'output_directory': str(root / 'runs' / name),
        'inputs': {key: str(path.resolve()) for key, path in paths.items()},
        'training_protocol': record,
        'fixed_epoch_source': 'checkpoint.epoch; never selected from validation during this run',
    }
