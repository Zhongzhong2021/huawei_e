"""Predeclared round-four class-balance study; no test-set model selection."""
import argparse
from datetime import datetime, timedelta, timezone
import gc
import json
from pathlib import Path
import shutil
import traceback

import numpy as np
import torch

from q2.data import digest, load_data, save_json
from q2v2.data import load_fold
from q2v2.engine import train_run


def now(): return datetime.now(timezone.utc)
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def summarize(records):
    return {
        'score': float(np.mean([r['selection']['score'] for r in records])),
        'clean_f1': float(np.mean([r['clean']['macro_f1'] for r in records])),
        'clean_mae': float(np.mean([r['clean']['mae'] for r in records])),
        'neutral_f1': float(np.mean([r['clean']['class_f1'][1] for r in records])),
        'missing_f1': float(np.mean([r['selection']['mean_missing_macro_f1'] for r in records])),
        'missing_mae': float(np.mean([r['selection']['mean_missing_mae'] for r in records])),
    }


def guards(candidate, reference):
    return candidate['score'] > reference['score'] and candidate['clean_f1'] >= reference['clean_f1'] - .01 and candidate['clean_mae'] <= reference['clean_mae'] + .03


def compare_tensors(first, second):
    a = torch.load(first, map_location='cpu', weights_only=True)['state_dict']
    b = torch.load(second, map_location='cpu', weights_only=True)['state_dict']
    assert a.keys() == b.keys()
    differences = {k: float((a[k].float() - b[k].float()).abs().max()) for k in a}
    return {'tensor_count': len(a), 'all_tensors_exact': all(v == 0 for v in differences.values()),
            'maximum_absolute_difference': max(differences.values())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--round2-root', type=Path, required=True)
    parser.add_argument('--round3-root', type=Path, required=True)
    args = parser.parse_args()
    root, data_root, old2, old3 = [p.resolve() for p in [args.root, args.data_root, args.round2_root, args.round3_root]]
    if root in [data_root, old2, old3] or root in old3.parents or root in old2.parents:
        raise ValueError('Round four requires an independent output directory')
    if (root / 'study/frozen.json').exists():
        raise RuntimeError('Round four already frozen; never restart its study')
    if (root / 'study/started.json').exists():
        raise RuntimeError('An earlier study started here; inspect its live process or terminal state before manual recovery')
    for folder in ['study', 'audit', 'runs', 'final']:
        (root / folder).mkdir(parents=True, exist_ok=True)
    old_protocol = read(old3 / 'study/protocol.json')
    base_config = read(old3 / 'runs/bert12_full_span_fold0/config.json')
    references = [read(old3 / f'runs/bert12_full_span_fold{i}/metrics.json') for i in range(3)]
    baseline = summarize(references)
    started = now()
    protocol = {
        'version': 'round4-class-balance-predeclared-v1', 'started_utc': started.isoformat(),
        'new_configuration_deadline': (started + timedelta(hours=2)).isoformat(),
        'optimization_deadline': (started + timedelta(hours=3)).isoformat(),
        'delivery_deadline': (started + timedelta(hours=4)).isoformat(),
        'reference': 'round3 full-vocabulary 12-layer span model, FP32 inference',
        'candidates': [{'name': 'balanced_sqrt', 'class_weight_power': .5}, {'name': 'balanced_inverse', 'class_weight_power': 1.}],
        'training': {k: v for k, v in old_protocol['training'].items() if k != 'class_weight'},
        'class_weight_definition': 'w_c proportional to (N/(3*n_c))**power, arithmetic mean normalized to 1; counts from fitting labels only',
        'classification_loss': 'sum(w_y * CE)/sum(w_y) over the effective batch; microbatch accumulation preserves the denominator',
        'fixed_other_factors': 'same initialization, folds, train-only scaler, full vocabulary, 0.7 span augmentation, CE+Huber, optimizer, batch32 and precision as round3',
        'selection_score': old_protocol['selection_score'], 'guardrails': old_protocol['guardrails'],
        'folds': 3, 'development_seed': 42, 'confirmation_seeds': [42, 2026, 3407],
        'selection_rule': 'at least 2/3 internal fold score improvements and mean clean guards; best eligible mean S; median internal best epoch; >=2/3 paired confirmation successes and mean success against round3',
        'selection_success': 'S strictly improves, clean Macro-F1 drop <=0.01, clean MAE increase <=0.03',
        'test_policy': 'No test reads before freeze; previously observed test is descriptive only',
        'attachment3_policy': 'No attachment3 predictions used for tuning',
        'baseline_control': 'Replay unweighted fold0 first; require every tensor identical before reusing old reference runs',
        'statistical_scope': 'official validation has been reused across rounds; improvements are conditional, not unbiased performance estimates',
        'class_weight_counts_never_use': ['internal validation', 'official validation', 'test', 'attachment3'],
    }
    source_hashes = {str(p.relative_to(root)): digest(p) for folder in ['src', 'scripts', 'tests'] for p in sorted((root / folder).rglob('*.py'))}
    save_json(root / 'study/protocol.json', protocol)
    save_json(root / 'audit/source_hashes.json', source_hashes)
    save_json(root / 'study/started.json', {'started_utc': started.isoformat(), 'protocol_sha256': digest(root / 'study/protocol.json')})
    stop = datetime.fromisoformat(protocol['optimization_deadline'])
    new_stop = datetime.fromisoformat(protocol['new_configuration_deadline'])
    source_path = old2 / 'pretrained/bert_base_uncased.pt'
    save_json(root / 'audit/input_hashes.json', {
        'pretrained_sha256': digest(source_path), 'reference_model_sha256': digest(old3 / 'final/model.pt'),
        'folds': {f'{f}/{s}': digest(old2 / f'folds/fold{f}/{s}.npz') for f in range(3) for s in ['train', 'valid', 'scaler']},
        'prior_frozen_record_sha256': digest(old3 / 'study/frozen.json')})
    source = torch.load(source_path, map_location='cpu', weights_only=True)
    fit0, valid0 = load_fold(old2, 0, 'train'), load_fold(old2, 0, 'valid')
    control = train_run(root, 'control_unweighted_fold0', base_config, fit0, valid0, np.arange(30522), source=source)
    comparison = compare_tensors(root / 'runs/control_unweighted_fold0/best.pt', old3 / 'runs/bert12_full_span_fold0/best.pt')
    comparison['selection_score_delta'] = control['selection']['score'] - references[0]['selection']['score']
    save_json(root / 'audit/control_reproduction.json', comparison)
    if not comparison['all_tensors_exact']:
        save_json(root / 'study/terminal.json', {'status': 'failed_baseline_control', **comparison})
        raise RuntimeError('Baseline replay changed: do not reuse old references or start weighted candidates')
    del fit0, valid0
    gc.collect(); torch.cuda.empty_cache()
    development = {'reference': references, 'reference_summary': baseline, 'entries': [], 'events': []}
    save_json(root / 'study/development.json', development)
    for candidate in protocol['candidates']:
        name = candidate['name']
        if now() >= new_stop:
            development['events'].append({'name': name, 'status': 'not_started_deadline'})
            save_json(root / 'study/development.json', development)
            continue
        config = {**base_config, 'class_weight_power': candidate['class_weight_power']}
        records = []
        try:
            for fold in range(3):
                if now() >= stop: raise TimeoutError('Round-four optimization deadline reached')
                fitting, validation = load_fold(old2, fold, 'train'), load_fold(old2, fold, 'valid')
                records.append(train_run(root, f'{name}_fold{fold}', config, fitting, validation, np.arange(30522), source=source))
                del fitting, validation
                gc.collect(); torch.cuda.empty_cache()
            mean = summarize(records)
            improvements = sum(r['selection']['score'] > ref['selection']['score'] for r, ref in zip(records, references))
            development['entries'].append({
                'name': name, 'config': config, 'metrics': records, 'summary': mean,
                'fold_improvements': improvements, 'eligible': improvements >= 2 and guards(mean, baseline),
                'fixed_epochs': int(np.floor(np.median([m['run_metadata']['best_epoch'] for m in records]) + .5)),
            })
        except Exception as exc:
            development['events'].append({'name': name, 'status': 'failed_or_incomplete', 'completed_folds': len(records),
                'error': repr(exc), 'traceback': traceback.format_exc()})
        save_json(root / 'study/development.json', development)
        print(json.dumps({'candidate_completed': name, 'entries': [{k: e[k] for k in ['name', 'summary', 'fold_improvements', 'eligible', 'fixed_epochs']} for e in development['entries']]}, ensure_ascii=False), flush=True)
    eligible = [e for e in development['entries'] if e['eligible']]
    selected = max(eligible, key=lambda e: e['summary']['score']) if eligible else None
    save_json(root / 'study/internal_selection.json', {'selected': selected, 'selected_utc': now().isoformat(), 'test_accessed': False})
    records, pairs, stable = [], [], False
    if selected is not None:
        training, valid = load_data(data_root, 'train'), load_data(data_root, 'valid')
        for seed in protocol['confirmation_seeds']:
            if now() >= stop: break
            name = f"confirmed_{selected['name']}_s{seed}"
            config = {**selected['config'], 'seed': seed}
            try:
                result = train_run(root, name, config, training, valid, np.arange(30522), source=source, fixed_epochs=selected['fixed_epochs'])
                reference = read(old3 / f'runs/confirmed_bert12_full_span_s{seed}/metrics.json')
                a, b = summarize([result]), summarize([reference])
                pairs.append({'seed': seed, 'deltas': {k: a[k] - b[k] for k in a}, 'passed': guards(a, b)})
                records.append({'seed': seed, 'run': name, 'metrics': result, 'reference': reference})
                save_json(root / 'study/confirmation_progress.json', {'records': records, 'pairs': pairs})
                gc.collect(); torch.cuda.empty_cache()
            except Exception as exc:
                save_json(root / 'audit/confirmation_failure.json', {'seed': seed, 'error': repr(exc), 'traceback': traceback.format_exc()})
                break
    if len(records) == 3 and now() <= stop:
        stable = sum(p['passed'] for p in pairs) >= 2 and guards(summarize([r['metrics'] for r in records]), summarize([r['reference'] for r in records]))
    confirmed = {'stable_improvement': stable, 'selected': selected, 'records': records, 'pairs': pairs, 'completed_utc': now().isoformat()}
    save_json(root / 'study/confirmed.json', confirmed)
    chosen = root / 'runs' / records[0]['run'] / 'best.pt' if stable else old3 / 'final/model.pt'
    shutil.copy2(chosen, root / 'final/model.pt')
    shutil.copy2(old3 / 'final/scaler.npz', root / 'final/scaler.npz')
    save_json(root / 'study/frozen.json', {
        'frozen_utc': now().isoformat(), 'selected_checkpoint': str(chosen),
        'model_name': selected['name'] if stable else 'round3_bert12_full_span',
        'stable_improvement': stable, 'checkpoint_sha256': digest(root / 'final/model.pt'),
        'scaler_sha256': digest(root / 'final/scaler.npz'), 'protocol_sha256': digest(root / 'study/protocol.json'),
        'test_used_for_selection': False, 'quantization': None,
    })
    save_json(root / 'study/terminal.json', {'status': 'completed', 'stable_improvement': stable, 'completed_utc': now().isoformat()})
    print(json.dumps({'frozen': read(root / 'study/frozen.json'), 'pairs': pairs}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__': main()
