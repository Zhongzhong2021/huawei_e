import unittest
import numpy as np
import torch
from torch.nn import functional as F
from q2v2.losses import frequency_weights, microbatch_cross_entropy


class ClassBalanceTests(unittest.TestCase):
    def test_zero_power_preserves_unweighted_loss_exactly(self):
        labels = np.array([0, 1, 2], dtype=np.int64)
        self.assertIsNone(frequency_weights(labels, 0))
        logits = torch.tensor([[1., 2., 0.], [0., 3., 1.], [2., 1., 4.]])
        target = torch.from_numpy(labels)
        self.assertTrue(torch.equal(microbatch_cross_entropy(logits, target), F.cross_entropy(logits, target)))

    def test_weights_use_only_supplied_fitting_counts(self):
        labels = np.array([0, 0, 1, 2, 2, 2, 2])
        for power in [.5, 1.]:
            weights = frequency_weights(labels, power).numpy()
            expected = (7 / (3 * np.array([2, 1, 4]))) ** power
            np.testing.assert_allclose(weights, expected / expected.mean(), rtol=1e-6)

    def test_absent_class_and_invalid_power_are_rejected(self):
        with self.assertRaises(ValueError): frequency_weights(np.array([0, 1]), .5)
        for power in [-1, 2, float('nan')]:
            with self.assertRaises(ValueError): frequency_weights(np.array([0, 1, 2]), power)

    def test_weighted_accumulation_matches_full_batch_gradient(self):
        torch.manual_seed(20260925)
        x = torch.randn(11, 5)
        y = torch.tensor([0, 0, 0, 0, 1, 1, 2, 0, 2, 2, 2])
        weights = torch.tensor([.6, 1.7, .9])
        layer = torch.nn.Linear(5, 3)
        F.cross_entropy(layer(x), y, weight=weights).backward()
        expected = [p.grad.clone() for p in layer.parameters()]
        for micro in [1, 3, 8, 11]:
            layer.zero_grad()
            denominator = float(weights[y].sum())
            for start in range(0, len(y), micro):
                target = y[start:start + micro]
                loss = microbatch_cross_entropy(layer(x[start:start + micro]), target, weights, len(y), denominator)
                (loss * len(target) / len(y)).backward()
            for parameter, reference in zip(layer.parameters(), expected):
                self.assertTrue(torch.allclose(parameter.grad, reference, atol=1e-7, rtol=1e-6))


if __name__ == '__main__': unittest.main()
