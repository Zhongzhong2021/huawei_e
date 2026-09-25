"""Independent decision, fitting-count and exact-retraining audit for round four."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

import numpy as np
from q2.data import digest, load_data, save_json
from q2v2.data import load_fold
from run_round4 import compare_tensors


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))


def mean(records):
    return {key: statistics.mean(values) for key, values in {
        'score': [m['selection']['score'] for m in records],
        'f1': [m['clean']['macro_f1'] for m in records],
        'mae': [m['clean']['mae'] for m in records]}.items()}


def clean_guard(a, b): return a['f1'] >= b['f1'] - .01 and a['mae'] <= b['mae'] + .03
def paired_guard(a, b): return a['score'] > b['score'] and clean_guard(a, b)


def main(root, data_root, old2, old3):
    frozen = read(root / 'study/frozen.json')
    assert digest(root / 'study/protocol.json') == frozen['protocol_sha256']
    assert digest(root / 'final/model.pt') == frozen['checkpoint_sha256']
    assert digest(root / 'final/scaler.npz') == frozen['scaler_sha256']
    original = read(root / 'audit/input_hashes.json')
    assert digest(old3 / 'final/model.pt') == original['reference_model_sha256']
    assert digest(old3 / 'study/frozen.json') == original['prior_frozen_record_sha256']
    assert digest(old2 / 'pretrained/bert_base_uncased.pt') == original['pretrained_sha256']
    for key, expected in original['folds'].items():
        fold, split = key.split('/')
        assert digest(old2 / f'folds/fold{fold}/{split}.npz') == expected
    old_sources = read(root / 'audit/source_hashes.json')
    # Analysis/report scripts may be added after freezing; training source must not change.
    for name, expected in old_sources.items():
        if name.startswith('src/'): assert digest(root / name) == expected, name
    dev = read(root / 'study/development.json')
    base = mean(dev['reference']); eligible = []; decisions = []
    for entry in dev['entries']:
        current = mean(entry['metrics'])
        improved = sum(a['selection']['score'] > b['selection']['score'] for a, b in zip(entry['metrics'], dev['reference']))
        passed = improved >= 2 and clean_guard(current, base)
        assert improved == entry['fold_improvements'] and passed == entry['eligible']
        assert entry['fixed_epochs'] == int(statistics.median(m['run_metadata']['best_epoch'] for m in entry['metrics']))
        if passed: eligible.append((current['score'], entry['name']))
        decisions.append({'name': entry['name'], 'score': current['score'], 'eligible': passed, 'improved_folds': improved})
    conf = read(root / 'study/confirmed.json')
    assert max(eligible)[1] == conf['selected']['name']
    assert [r['seed'] for r in conf['records']] == [42, 2026, 3407]
    passed = [paired_guard(mean([r['metrics']]), mean([r['reference']])) for r in conf['records']]
    assert passed == [r['passed'] for r in conf['pairs']]
    stable = sum(passed) >= 2 and paired_guard(mean([r['metrics'] for r in conf['records']]), mean([r['reference'] for r in conf['records']]))
    assert stable == conf['stable_improvement'] == frozen['stable_improvement']
    counts = []
    def check_fit(run, fitting, power):
        prov = read(root / f'runs/{run}/provenance.json')
        assert prov['fit_ids'] == fitting['ids'].tolist()
        n = np.bincount(fitting['labels'], minlength=3)
        u = (n.sum() / (3 * n)) ** power
        expected = u / u.mean()
        np.testing.assert_allclose(prov['classification_weights'], expected, rtol=1e-6, atol=1e-7)
        counts.append({'run': run, 'fit_class_counts': n.tolist(), 'weights': prov['classification_weights']})
    for fold in range(3):
        fit, valid = load_fold(old2, fold, 'train'), load_fold(old2, fold, 'valid')
        assert not set(str(x).split('$_$')[0] for x in fit['ids']) & set(str(x).split('$_$')[0] for x in valid['ids'])
        for entry in dev['entries']:
            check_fit(f"{entry['name']}_fold{fold}", fit, entry['config']['class_weight_power'])
    fit = load_data(data_root, 'train')
    for rec in conf['records']: check_fit(rec['run'], fit, conf['selected']['config']['class_weight_power'])
    check_fit('reproduction_s42', fit, conf['selected']['config']['class_weight_power'])
    result = compare_tensors(root / 'final/model.pt', root / 'runs/reproduction_s42/best.pt')
    assert result['all_tensors_exact']
    result.update({'checkpoint_sha256': digest(root / 'final/model.pt'), 'reproduction_sha256': digest(root / 'runs/reproduction_s42/best.pt'), 'same_environment_only': True})
    assert result['checkpoint_sha256'] == result['reproduction_sha256']
    save_json(root / 'audit/final_reproduction.json', result)
    checks = {'utc': datetime.now(timezone.utc).isoformat(), 'selection_independently_recomputed': True,
              'fitting_only_class_weights_recomputed': counts, 'video_group_overlap': 0,
              'training_sources_unchanged': True, 'reference_and_fold_hashes_unchanged': True,
              'candidate_decisions': decisions, 'confirmation_pairs_passed': sum(passed),
              'final_reproduction': result}
    save_json(root / 'audit/decision_integrity.json', checks)
    print(json.dumps({k: v for k, v in checks.items() if k != 'fitting_only_class_weights_recomputed'}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ['root', 'data-root', 'round2-root', 'round3-root']: p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args(); main(a.root, a.data_root, a.round2_root, a.round3_root)
