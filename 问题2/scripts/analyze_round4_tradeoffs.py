"""Secondary, non-selection analysis of all classes and overall accuracy."""
import argparse
from pathlib import Path
import numpy as np
from q2.data import save_json
from q2v3.analysis import arrays, cluster_weights, write_rows


def class_statistics(a, weights):
    cells = np.stack([(a['labels'] == i) & (a['classes'] == j) for i in range(3) for j in range(3)], axis=1).astype(float)
    cm = (weights @ cells).reshape(-1, 3, 3)
    tp = np.diagonal(cm, axis1=1, axis2=2)
    den = cm.sum(1) + cm.sum(2)
    f1 = np.divide(2 * tp, den, out=np.zeros_like(tp), where=den != 0)
    return np.column_stack([tp.sum(1) / cm.sum((1, 2)), f1])


def main(root):
    root = Path(root)
    baseline = arrays(root / 'evaluation/reference/valid/predictions/clean.csv')
    final = arrays(root / 'evaluation/final/valid/predictions/clean.csv')
    assert np.array_equal(baseline['ids'], final['ids'])
    assert np.array_equal(baseline['labels'], final['labels'])
    weights, groups = cluster_weights(final['ids'], 2000, 240924)
    draws = class_statistics(final, weights) - class_statistics(baseline, weights)
    unit = np.ones((1, len(final['ids'])))
    old_point, new_point = class_statistics(baseline, unit)[0], class_statistics(final, unit)[0]
    records = []
    for i, metric in enumerate(['accuracy', 'negative_f1', 'neutral_f1', 'positive_f1']):
        records.append({'metric': metric, 'reference': float(old_point[i]), 'final': float(new_point[i]),
            'difference': float(new_point[i] - old_point[i]), 'lower': float(np.quantile(draws[:, i], .025)),
            'upper': float(np.quantile(draws[:, i], .975))})
    write_rows(root / 'analysis/class_tradeoff_intervals.csv', records)
    np.savez_compressed(root / 'analysis/class_tradeoff_bootstrap.npz', differences=draws)
    save_json(root / 'analysis/class_tradeoff_protocol.json', {'replicates': 2000, 'seed': 240924,
        'groups': len(groups), 'samples': len(final['ids']), 'model_seed': 42,
        'role': 'secondary description of all three classes and accuracy; never used for selection; no multiple-comparison correction',
        'records': records})
    print(records, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('root', type=Path); main(p.parse_args().root)
