"""Post-freeze inference, independently recomputed evidence and class diagnostics."""
import argparse
import gc
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from q2.data import digest, load_data, save_json
from q2.engine import evaluate_model, infer, save_scenarios
from q2.metrics import metrics
from q2.missing import MAIN, SUPPLEMENT
from q2v2.deploy import load_deployment, predict, raw_inputs
from q2v3.analysis import arrays, main as analyze, read, write_rows


def validate_prediction_file(path, expected):
    a = arrays(path)
    assert len(set(a['ids'])) == len(a['ids'])
    assert np.array_equal(a['probabilities'].argmax(1), a['classes'])
    actual = metrics(a['labels'], a['targets'], a['probabilities'], a['predictions'])
    for key in ['accuracy', 'macro_f1', 'mae', 'pearson']:
        if expected[key] is None: assert actual[key] is None
        else: assert abs(actual[key] - expected[key]) < 1e-10, (path, key)
    assert actual['confusion_matrix'] == expected['confusion_matrix']
    np.testing.assert_allclose(actual['class_f1'], expected['class_f1'], atol=1e-12, rtol=0)
    return actual


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--round3-root', type=Path, required=True)
    a = p.parse_args()
    root, old = a.root.resolve(), a.round3_root.resolve()
    freeze = read(root / 'study/frozen.json')
    assert digest(root / 'final/model.pt') == freeze['checkpoint_sha256']
    if (root / 'audit/postfreeze_complete.json').exists():
        raise FileExistsError('Post-freeze results already exist; inspect instead of overwriting')
    # Copy immutable prediction evidence; no retraining and no test-driven choice.
    shutil.copytree(old / 'evaluation/final', root / 'evaluation/reference', dirs_exist_ok=True)
    for name in [f'bert12_full_span_fold{f}' for f in range(3)] + [f'confirmed_bert12_full_span_s{s}' for s in [42, 2026, 3407]]:
        target = root / 'evidence/reference/runs' / name
        target.mkdir(parents=True, exist_ok=True)
        for source in (old / 'runs' / name).glob('*.json'):
            shutil.copy2(source, target / source.name)
        shutil.copytree(old / 'runs' / name / 'predictions', target / 'predictions', dirs_exist_ok=True)
    shutil.copy2(old / 'evidence/valid_metadata.npz', root / 'evidence/valid_metadata.npz')
    save_json(root / 'evidence/reference/frozen.json', read(old / 'study/frozen.json'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if freeze['stable_improvement']:
        model, _ = load_deployment(root / 'final/model.pt', device)
        for split in ['valid', 'test']:
            data = load_data(a.data_root, split)
            masks = save_scenarios(root, data, split)
            out = root / f'evaluation/final/{split}'
            result = evaluate_model(model, data, device, ['clean'] + MAIN + SUPPLEMENT, out=out / 'predictions', batch_size=32, masks=masks)
            save_json(out / 'metrics.json', result)
            print(json.dumps({'split': split, 'clean': result['clean']}, ensure_ascii=False), flush=True)
        del model
        gc.collect(); torch.cuda.empty_cache()
    else:
        assert digest(root / 'final/model.pt') == digest(old / 'final/model.pt')
        shutil.copytree(root / 'evaluation/reference', root / 'evaluation/final', dirs_exist_ok=True)
        shutil.copytree(old / 'scenarios', root / 'scenarios', dirs_exist_ok=True)
    checks = predict(root / 'final/model.pt', root / 'final/scaler.npz', a.data_root / 'data/raw/attachment3', root / 'final/attachment3_predictions.csv', device)
    save_json(root / 'audit/prediction_checks.json', checks)
    special = raw_inputs(a.data_root / 'data/raw/attachment3', root / 'final/scaler.npz')
    model, _ = load_deployment(root / 'final/model.pt', 'cpu')
    cpu_p, cpu_y = infer(model, special, 'cpu', batch_size=32)
    empty = {k: np.array(v[:2], copy=True) for k, v in special.items()}
    empty['valid'][:] = False; empty['observed'][:] = False
    ep, ey = infer(model, empty, 'cpu', batch_size=2)
    assert np.isfinite(ep).all() and np.isfinite(ey).all()
    del model
    model, _ = load_deployment(root / 'final/model.pt', device)
    gpu_p, gpu_y = infer(model, special, device, batch_size=32)
    assert np.array_equal(cpu_p.argmax(1), gpu_p.argmax(1))
    np.testing.assert_allclose(cpu_p, gpu_p, atol=1e-5, rtol=0)
    np.testing.assert_allclose(cpu_y, gpu_y, atol=1e-5, rtol=0)
    save_json(root / 'audit/cpu_gpu.json', {'class_equal': True,
        'max_probability_difference': float(abs(cpu_p - gpu_p).max()),
        'max_intensity_difference': float(abs(cpu_y - gpu_y).max()), 'empty_input_finite': True})
    del model
    gc.collect(); torch.cuda.empty_cache()
    analyze(root)
    verified = []
    for directory in [root / 'runs', root / 'evidence/reference/runs']:
        for run in sorted(directory.iterdir()):
            if not (run / 'metrics.json').exists(): continue
            m = read(run / 'metrics.json')
            for scenario in ['clean'] + MAIN:
                path = run / f'predictions/{scenario}.csv'
                validate_prediction_file(path, m[scenario])
                verified.append({'source': str(path.relative_to(root)), 'sha256': digest(path), 'passed': True})
    save_json(root / 'audit/training_prediction_recomputation.json', {'verified_count': len(verified), 'records': verified})
    development = read(root / 'study/development.json')
    fold_rows = []
    for name, power, records in [('reference', 0., development['reference'])] + [(entry['name'], entry['config']['class_weight_power'], entry['metrics']) for entry in development['entries']]:
        for fold, m in enumerate(records):
            cm = np.asarray(m['clean']['confusion_matrix'])
            fold_rows.append({'model': name, 'power': power, 'fold': fold, 'n': int(cm.sum()),
                'score': m['selection']['score'], 'clean_macro_f1': m['clean']['macro_f1'], 'clean_mae': m['clean']['mae'],
                'neutral_precision': float(cm[1, 1] / max(1, cm[:, 1].sum())), 'neutral_recall': float(cm[1, 1] / cm[1].sum()),
                'neutral_f1': m['clean']['class_f1'][1], 'negative_f1': m['clean']['class_f1'][0], 'positive_f1': m['clean']['class_f1'][2],
                'missing_macro_f1': m['selection']['mean_missing_macro_f1'], 'missing_mae': m['selection']['mean_missing_mae']})
    write_rows(root / 'analysis/internal_folds.csv', fold_rows)
    class_diagnostics = {}
    for role in ['reference', 'final']:
        values = arrays(root / f'evaluation/{role}/valid/predictions/clean.csv')
        pred_class, prediction = values['classes'], values['predictions']
        implied = np.where(prediction < 0, 0, np.where(prediction > 0, 2, 1))
        exact_mismatch = implied != pred_class
        neutral_component = (pred_class == 1) & (prediction != 0)
        polarity_component = ((pred_class == 0) & (prediction > 0)) | ((pred_class == 2) & (prediction < 0))
        zero_component = (pred_class != 1) & (prediction == 0)
        assert np.array_equal(exact_mismatch, neutral_component | polarity_component | zero_component)
        class_diagnostics[role] = {'n': len(prediction), 'exact_sign_mismatch': int(exact_mismatch.sum()),
            'neutral_class_with_nonzero_regression': int(neutral_component.sum()),
            'opposite_nonzero_polarities': int(polarity_component.sum()), 'nonneutral_class_with_exact_zero_regression': int(zero_component.sum()),
            'note': 'Neutral class outputs need not produce exactly zero continuous regression; total exact-sign mismatch is not a contradiction rate.',
            'class_mae': [float(abs(values['targets'][values['labels'] == c] - prediction[values['labels'] == c]).mean()) for c in range(3)]}
    save_json(root / 'analysis/output_consistency_decomposition.json', class_diagnostics)
    input_sources = {str(path.relative_to(root)): digest(path) for path in sorted(root.glob('runs/*/provenance.json'))}
    save_json(root / 'audit/postfreeze_complete.json', {
        'frozen_model_sha256': digest(root / 'final/model.pt'), 'attachment3_rows': checks['rows'],
        'evaluation_scenarios_recomputed': 184, 'training_scenarios_recomputed': len(verified),
        'cpu_gpu_checked': True, 'finite_empty_input': True, 'run_provenance_sha256': input_sources,
        'old_round3_model_unchanged': digest(old / 'final/model.pt') == read(old / 'study/frozen.json')['checkpoint_sha256']})
    print(json.dumps(read(root / 'audit/postfreeze_complete.json'), ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__': main()
