"""Strict, paired probability/intensity blending of frozen model predictions."""
import numpy as np


def blend_predictions(current, previous, current_weight):
    if isinstance(current_weight, bool) or not np.isfinite(current_weight) or not 0 < current_weight < 1:
        raise ValueError('Current-model weight must lie strictly between zero and one')
    n = len(current['ids'])
    if not n:
        raise ValueError('Empty prediction table')
    for model in [current, previous]:
        if len(set(model['ids'])) != n or len(model['ids']) != n:
            raise ValueError('Duplicate IDs or mismatched sample count')
        p, y = model['probabilities'], model['predictions']
        if p.shape != (n, 3) or y.shape != (n,):
            raise ValueError('Prediction shape mismatch')
        if not np.isfinite(p).all() or not np.isfinite(y).all():
            raise ValueError('Nonfinite predictions')
        if (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(1), 1, atol=1e-6, rtol=0):
            raise ValueError('Invalid class probabilities')
        if (np.abs(y) > 3).any() or not np.array_equal(p.argmax(1), model['classes']):
            raise ValueError('Invalid intensity or inconsistent class/probability output')
    for key in ['ids', 'labels', 'targets']:
        if not np.array_equal(current[key], previous[key]):
            raise ValueError(f'Paired predictions do not align on {key}')
    a = np.float32(current_weight)
    p = a*current['probabilities'].astype(np.float32)+(np.float32(1)-a)*previous['probabilities'].astype(np.float32)
    y = a*current['predictions'].astype(np.float32)+(np.float32(1)-a)*previous['predictions'].astype(np.float32)
    return {**{k:current[k].copy() for k in ['ids','labels','targets']},
            'classes':p.argmax(1),'probabilities':p,'predictions':y}
