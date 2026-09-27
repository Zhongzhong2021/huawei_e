"""Audit the complete seed-17 source-baseline parameter cohort and evaluate test.

Requires the original research runs (not included in the delivery) and aligned
attachment-two data. No training or policy search is performed.
"""
import argparse
import copy
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from q3b.data import load_samples, make_batch
from q3b.experiment import build_experiment_model
from q3b.multiobjective import apply_policy
from q3b.train import _hash, _json, _predict_prepared, evaluate

COHORT = [('S0', 'selfmm', 'control'), ('S1', 'selfmm', 'aux'),
          ('S2', 'selfmm', 'aux_gentle'), ('S3', 'selfmm', 'wide'),
          ('S4', 'selfmm', 'dropout'), ('S5', 'selfmm', 'dropout_source_lr'),
          ('T0', 'tetfn', 'source50_lr1'), ('T1', 'tetfn', 'source50_lr2')]


def csv_write(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research-root', required=True, type=Path)
    parser.add_argument('--aligned', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Output must be empty; existing results are never overwritten')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    samples = {s: load_samples(args.aligned, s) for s in ('valid', 'test')}
    assert [len(samples[s]) for s in ('valid', 'test')] == [728, 727]
    first_stats = args.research_root/'runs/q3b_next/selfmm/control/stats.json'
    stats = json.loads(first_stats.read_text())
    batches = {s: make_batch(v, stats, 1, args.device) for s, v in samples.items()}
    records, table = [], []
    for code, family, name in COHORT:
        relative = f'runs/q3b_next/{family}/{name}'
        run = args.research_root/relative
        config = json.loads((run/'config.json').read_text())
        winner = json.loads((run/'winners.json').read_text())['macro_f1']
        assert config['train']['seed'] == 17 and config['train']['max_epochs'] == 10
        assert config['train']['patience'] == 4 and config['model']['window_size'] == 1
        assert json.loads((run/'stats.json').read_text()) == stats
        checkpoint = run/winner['checkpoint']
        assert _hash(checkpoint) == winner['checkpoint_sha256']
        archive = run/f"epochs/epoch_{winner['epoch']:03d}_valid.npz"
        stored = np.load(archive)
        restored_config = copy.deepcopy(config)
        restored_config['model']['pretrained_name'] = str(ROOT/'assets/bert-base-uncased')
        model = build_experiment_model(restored_config).to(args.device)
        state = torch.load(checkpoint, map_location=args.device, weights_only=True)
        model.load_state_dict(state['model_state_dict'], strict=True)
        target = args.output/code
        target.mkdir()
        metrics = {}
        for split in ('valid', 'test'):
            raw, aux = _predict_prepared(model, batches[split], config['train']['batch_size'])
            ids = [str(s['id']) for s in samples[split]]
            y = np.asarray([s['y'] for s in samples[split]], dtype=np.float32)
            truth = np.where(y < 0, 0, np.where(y > 0, 2, 1))
            classes, values = apply_policy(raw, aux, winner['policy'])
            metrics[split] = evaluate(y, truth, values, classes)
            if split == 'valid':
                assert ids == stored['ids'].astype(str).tolist()
                np.testing.assert_array_equal(y, stored['y'])
                reload_delta = max(float(np.max(np.abs(raw-stored['raw']))),
                                   float(np.max(np.abs(aux-stored['aux']))))
                assert reload_delta < 3e-6, (code, reload_delta)
                old_classes, old_values = apply_policy(stored['raw'], stored['aux'], winner['policy'])
                np.testing.assert_array_equal(classes, old_classes)
                recalculated = evaluate(y, truth, old_values, old_classes)
                metric_delta = max(abs(recalculated[k]-winner['metrics'][k])
                                   for k in ('accuracy', 'macro_f1', 'mae'))
                assert metric_delta < 1e-12
                # Keep the exact historical validation metrics; test is newly evaluated.
                metrics['valid'] = winner['metrics']
            csv_write(target/f'{split}_predictions.csv', [dict(
                id=ids[i], true_intensity=float(y[i]), true_class=int(truth[i]),
                raw_intensity=float(raw[i]), aux_logit_0=float(aux[i, 0]),
                aux_logit_1=float(aux[i, 1]), aux_logit_2=float(aux[i, 2]),
                polarity=int(classes[i]), intensity=float(values[i])) for i in range(len(y))])
        provenance = {str(p.relative_to(args.research_root)): _hash(p) for p in
                      (run/'config.json', run/'winners.json', run/'stats.json', archive)}
        record = dict(code=code, family=family, name=name, run=relative,
                      config=config, epoch=winner['epoch'], policy=winner['policy'],
                      actual_epochs=len(list((run/'epochs').glob('*_valid.npz'))),
                      checkpoint_sha256=winner['checkpoint_sha256'], provenance=provenance,
                      validation_reload_max_delta=reload_delta,
                      validation_recalculation_max_delta=metric_delta, **metrics)
        _json(target/'record.json', record)
        records.append(record)
        table.append(dict(code=code, family=family, configuration=name, seed=17,
                          max_epochs=10, selected_epoch=winner['epoch'],
                          **{f'{s}_{k}': metrics[s][k] for s in ('valid', 'test')
                             for k in ('accuracy', 'macro_f1', 'mae')}))
        print(json.dumps(table[-1], ensure_ascii=False), flush=True)
        del model, state
        torch.cuda.empty_cache()
    _json(args.output/'comparisons.json', dict(
        protocol=dict(selection='historical_validation_macro_f1', new_training=False,
                      test_evaluation='frozen_checkpoint_and_policy_no_reselection',
                      selection_scope='all_seed17_SELF-MM_and_source_TETFN_configs_in_q3b_next',
                      data_sha256=_hash(args.aligned), seed=17, max_epochs=10,
                      train_n=3395, valid_n=728, test_n=727,
                      attachment_four_used=False), records=records))
    csv_write(args.output/'comparison.csv', table)


if __name__ == '__main__':
    main()
