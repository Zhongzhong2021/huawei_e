"""Portable frozen evaluation: explicit data, no overwriting research predictions."""
import json
from pathlib import Path
import numpy as np
from q2.data import digest, load_data, save_json
from q2.engine import evaluate_model, save_scenarios
from q2.missing import MAIN, SUPPLEMENT
from q2v2.deploy import load_deployment


def evaluate_frozen(root, data_root=None, output_dir=None, split='valid', device='cpu'):
    root = Path(root).resolve()
    data_root = Path(data_root).resolve() if data_root is not None else root.parent
    output = Path(output_dir).resolve() if output_dir is not None else root / f'evaluation/recomputed_{split}'
    if split not in ['valid', 'test']: raise ValueError('Only valid or descriptive test evaluation is supported')
    if output.exists(): raise FileExistsError('Evaluation output already exists; choose a new --output-dir')
    freeze_path = next((root / p for p in ['study/frozen.json', 'metadata/frozen_model.json'] if (root / p).is_file()), None)
    if freeze_path is None: raise FileNotFoundError('Frozen model metadata is required')
    frozen = json.loads(freeze_path.read_text(encoding='utf-8'))
    model_path, scaler_path = root / 'final/model.pt', root / 'final/scaler.npz'
    if digest(model_path) != frozen['checkpoint_sha256']: raise ValueError('Frozen model hash mismatch')
    if digest(scaler_path) != frozen['scaler_sha256']: raise ValueError('Frozen scaler hash mismatch')
    # Compare arrays, not NPZ bytes: independently saved archives may have different metadata.
    with np.load(scaler_path, allow_pickle=False) as expected, np.load(data_root / 'data/processed/scaler.npz', allow_pickle=False) as actual:
        if set(expected.files) != set(actual.files): raise ValueError('Processed-data scaler fields differ')
        for key in expected.files:
            if not np.array_equal(expected[key], actual[key]): raise ValueError(f'Processed-data scaler mismatch: {key}')
    data = load_data(data_root, split)
    model, _ = load_deployment(model_path, device)
    output.mkdir(parents=True, exist_ok=False)
    masks = save_scenarios(output, data, split)
    result = evaluate_model(model, data, device, ['clean'] + MAIN + SUPPLEMENT, out=output / 'predictions', batch_size=32, masks=masks)
    save_json(output / 'metrics.json', result)
    save_json(output / 'provenance.json', {'checkpoint_sha256': frozen['checkpoint_sha256'],
        'data_sha256': digest(data_root / f'data/processed/{split}.npz'), 'split': split,
        'role': 'descriptive only; no model selection', 'scenarios': 46, 'device': device})
    return {'output': str(output), 'clean': result['clean'], 'scenarios': 46}
