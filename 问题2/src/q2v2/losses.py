"""Fitting-only class weights and effective-batch-correct microbatch reduction."""
import math
import numpy as np
import torch
from torch.nn import functional as F


def epoch_class_weights(weights, epoch, delay_epochs=0):
    """Use unweighted CE for the first fixed number of epochs, then frozen weights."""
    if not isinstance(delay_epochs, int) or isinstance(delay_epochs, bool) or delay_epochs < 0:
        raise ValueError('class_weight_delay_epochs must be a nonnegative integer')
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 1:
        raise ValueError('epoch must be a positive integer')
    return None if epoch <= delay_epochs else weights


def frequency_weights(labels, power=0., classes=3):
    if not math.isfinite(power) or not 0 <= power <= 1:
        raise ValueError('class_weight_power must be finite and in [0, 1]')
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer) or np.any((labels < 0) | (labels >= classes)):
        raise ValueError('Fitting labels must be a one-dimensional integer class vector')
    if power == 0:
        return None
    counts = np.bincount(labels, minlength=classes)
    if np.any(counts == 0):
        raise ValueError('Cannot estimate inverse-frequency weights when a fitting class is absent')
    weights = (len(labels) / (classes * counts)) ** power
    return torch.tensor(weights / weights.mean(), dtype=torch.float32)


def microbatch_cross_entropy(logits, labels, weights=None, effective_size=None, effective_weight_sum=None):
    if weights is None:
        return F.cross_entropy(logits.float(), labels)
    if effective_size is None or effective_weight_sum is None or effective_size < len(labels) or effective_weight_sum <= 0:
        raise ValueError('Weighted CE requires the full effective batch size and weight sum')
    # The caller subsequently multiplies the entire loss by micro/effective size.
    # This factor makes the accumulated CE exactly sum(w_i*CE_i)/sum(w_i),
    # including imbalanced microbatches and a short final microbatch.
    return F.cross_entropy(logits.float(), labels, weight=weights, reduction='sum') * (
        effective_size / (len(labels) * effective_weight_sum))
