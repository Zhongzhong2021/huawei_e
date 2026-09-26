"""Independent post-study audit; never fit, select alternatives, or read test data."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prediction(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    ids = [row['sample_id'] for row in rows]
    assert ids and len(set(ids)) == len(ids)
    y = np.array([int(row['true_class']) for row in rows])
    target = np.array([float(row['true_intensity']) for row in rows])
    p = np.array([[float(row['prob_'+c]) for c in ['negative', 'neutral', 'positive']] for row in rows])
    estimate = np.array([float(row['intensity']) for row in rows])
    classes = np.array([int(row['class_id']) for row in rows])
    assert np.isfinite(p).all() and np.isfinite(estimate).all()
    assert np.isin(y, [0, 1, 2]).all() and np.max(np.abs(estimate)) <= 3
    assert ((p >= 0) & (p <= 1)).all()
    np.testing.assert_allclose(p.sum(1), 1, atol=1e-6, rtol=0)
    np.testing.assert_array_equal(classes, p.argmax(1))
    return ids, y, target, p, estimate


def recompute(values, expected):
    _, y, target, p, estimate = values
    cm = np.zeros((3, 3), dtype=np.int64)
    np.add.at(cm, (y, p.argmax(1)), 1)
    denominator = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2 * cm.diagonal(), denominator, out=np.zeros(3), where=denominator != 0)
    actual = {'n': len(y), 'accuracy': float(cm.trace()/len(y)), 'macro_f1': float(f1.mean()),
              'mae': float(np.abs(target-estimate).mean()), 'class_f1': f1.tolist(), 'confusion_matrix': cm.tolist()}
    for key, value in actual.items():
        np.testing.assert_allclose(value, expected[key], atol=1e-12, rtol=0)
    pearson = None if len(y) < 2 or np.ptp(target) == 0 or np.ptp(estimate) == 0 else float(np.corrcoef(target, estimate)[0, 1])
    if pearson is None:
        assert expected['pearson'] is None and expected['pearson_note']
    else:
        np.testing.assert_allclose(pearson, expected['pearson'], atol=1e-12, rtol=0)
    return actual


def summarize(records, scenes):
    return {key: float(np.mean(values)) for key, values in {
        'score': [.5*np.mean([r[s]['macro_f1'] for s in scenes])+.5*(1-np.mean([r[s]['mae'] for s in scenes])/6) for r in records],
        'clean_f1': [r['clean']['macro_f1'] for r in records],
        'clean_mae': [r['clean']['mae'] for r in records],
        'accuracy': [r['clean']['accuracy'] for r in records],
        'neutral_f1': [r['clean']['class_f1'][1] for r in records],
        'positive_f1': [r['clean']['class_f1'][2] for r in records],
        'missing_f1': [np.mean([r[s]['macro_f1'] for s in scenes]) for r in records],
        'missing_mae': [np.mean([r[s]['mae'] for s in scenes]) for r in records],
    }.items()}


def guard(a, b):
    d = {k: a[k]-b[k] for k in a}
    return (d['score'] > 0 and d['clean_f1'] >= -.01 and d['clean_mae'] <= .03
            and d['accuracy'] >= -.005 and d['neutral_f1'] >= -.01 and d['positive_f1'] > 0), d


def main(root, old3, old4, output):
    if output.exists():
        raise FileExistsError('Keep previous audits; choose a new output directory')
    protocol = read(root/'study/protocol.json')
    dev = read(root/'study/development.json')
    conf = read(root/'study/confirmed.json')
    terminal = read(root/'study/terminal.json')
    assert read(root/'study/started.json')['protocol_sha256'] == sha(root/'study/protocol.json')
    assert [(c['name'], c['round4_weight']) for c in protocol['candidates']] == [('r4_90_r3_10', .9), ('r4_75_r3_25', .75)]
    sources = read(root/'audit/prediction_sources.json')
    for path, expected in sources.items():
        assert '/predictions/' in path and '/test/' not in path and '/special/' not in path
        assert sha(Path(path)) == expected
    scenes = [f'{m}_{r}_random' for m in ['text', 'audio', 'vision'] for r in [10, 30, 50]]
    verified = set()
    pair_count = 0
    rows = []
    provenance_hashes = {}
    for entry in dev['entries']:
        for fold in range(3):
            rows.append((entry['name'], entry['round4_weight'], f'{entry["name"]}_fold{fold}',
                         f'balanced_inverse_fold{fold}', f'bert12_full_span_fold{fold}', True))
    for record in conf['records']:
        seed = record['seed']
        entry = conf['selected']
        rows.append((entry['name'], entry['round4_weight'], f'confirmed_{entry["name"]}_s{seed}',
                     f'confirmed_balanced_inverse_s{seed}', f'confirmed_bert12_full_span_s{seed}', False))
    for name, weight, mixed, r4name, r3name, internal in rows:
        runs = [root/'runs'/mixed, old4/'runs'/r4name, old3/'runs'/r3name]
        cp, pp = [read(run/'provenance.json') for run in runs[1:]]
        assert cp['fit_ids'] == pp['fit_ids'] and cp['validation_ids'] == pp['validation_ids']
        for run in runs[1:]:
            provenance_hashes[str(run/'provenance.json')] = sha(run/'provenance.json')
        assert not (set(cp['fit_ids']) & set(cp['validation_ids']))
        if internal:
            assert not ({i.split('$_$')[0] for i in cp['fit_ids']} & {i.split('$_$')[0] for i in cp['validation_ids']})
        for scene in ['clean']+scenes:
            values = [prediction(run/f'predictions/{scene}.csv') for run in runs]
            for run, v in zip(runs, values):
                recompute(v, read(run/'metrics.json')[scene])
                verified.add(str(run/f'predictions/{scene}.csv'))
            for v in values[1:]:
                assert values[0][0] == v[0]
                np.testing.assert_array_equal(values[0][1], v[1])
                np.testing.assert_array_equal(values[0][2], v[2])
            assert set(values[0][0]) == set(cp['validation_ids'])
            a = np.float32(weight)
            for column in [3, 4]:
                expected = a*values[1][column].astype(np.float32)+(np.float32(1)-a)*values[2][column].astype(np.float32)
                np.testing.assert_array_equal(values[0][column], expected)
            pair_count += 1
    baseline = summarize(dev['reference'], scenes)
    eligible = []
    for entry in dev['entries']:
        summary = summarize(entry['metrics'], scenes)
        passed, deltas = guard(summary, baseline)
        improved = sum(summarize([m], scenes)['score'] > summarize([b], scenes)['score'] for m, b in zip(entry['metrics'], dev['reference']))
        for key, value in summary.items():
            np.testing.assert_allclose(value, entry['summary'][key], atol=1e-12, rtol=0)
        assert improved == entry['fold_improvements'] and (passed and improved >= 2) == entry['eligible']
        if entry['eligible']:
            eligible.append(entry)
    selected = max(eligible, key=lambda e: e['summary']['score'])
    assert conf['selected']['name'] == selected['name'] == read(root/'study/internal_selection.json')['selected']['name']
    assert [r['seed'] for r in conf['records']] == [42, 2026, 3407]
    pairs = []
    for record, saved in zip(conf['records'], conf['pairs']):
        passed, deltas = guard(summarize([record['metrics']], scenes), summarize([record['reference']], scenes))
        assert passed == saved['passed']
        for key, value in deltas.items():
            np.testing.assert_allclose(value, saved['deltas'][key], atol=1e-12, rtol=0)
        pairs.append(passed)
    passed, deltas = guard(summarize([r['metrics'] for r in conf['records']], scenes), summarize([r['reference'] for r in conf['records']], scenes))
    assert passed == conf['mean_guard']['passed']
    success = sum(pairs) >= 2 and passed
    assert success == conf['statistical_conditions_passed'] and not success
    assert terminal['status'] == 'completed_no_promotion' and not terminal['deployment_changed']
    assert sorted(p.name for p in (root/'runs').iterdir()) == sorted(row[2] for row in rows)
    frozen = read(old4/'study/frozen.json')
    model_sha = sha(old4/'final/model.pt')
    assert model_sha == frozen['checkpoint_sha256']
    report = {'audited_utc': datetime.now(timezone.utc).isoformat(), 'status': 'passed',
              'protocol_sha256': sha(root/'study/protocol.json'), 'auditor_sha256': sha(Path(__file__)),
              'verified_unique_prediction_files': len(verified), 'verified_blend_scenarios': pair_count,
              'verified_original_source_hashes': len(sources), 'paired_seed_passes': sum(pairs),
              'mean_deltas': deltas, 'statistical_conditions_passed': bool(success),
              'official_model_sha256': model_sha, 'no_unselected_confirmation': True,
              'internal_video_groups_disjoint': True, 'component_fit_and_validation_ids_equal': True,
              'deployment_changed': False, 'test_read_by_audit': False, 'attachment3_read_by_audit': False,
              'source_provenance_hashes': provenance_hashes,
              'prediction_hashes': {p: sha(Path(p)) for p in sorted(verified)}}
    output.mkdir(parents=True)
    (output/'poststudy_checks.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if not k.endswith('hashes')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for arg in ['root', 'round3-root', 'round4-root', 'output']:
        parser.add_argument('--'+arg, type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.round3_root, args.round4_root, args.output)
