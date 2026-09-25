import tempfile
from pathlib import Path
import unittest
import torch
from q2v2.losses import epoch_class_weights
from q2v2.engine import train_run


class DeferredWeightTests(unittest.TestCase):
    def test_zero_delay_returns_identical_tensor_without_copy(self):
        w = torch.tensor([1., 1.4, .6])
        for epoch in range(1, 7): self.assertIs(epoch_class_weights(w, epoch), w)

    def test_switch_uses_fixed_absolute_epoch(self):
        w = torch.tensor([1., 1.4, .6])
        for delay in [1, 2]:
            for epoch in range(1, 7):
                self.assertIs(epoch_class_weights(w, epoch, delay), None if epoch <= delay else w)
        self.assertIsNone(epoch_class_weights(None, 4, 2))

    def test_invalid_epochs_rejected(self):
        for delay in [-1, .5, True, float('nan')]:
            with self.assertRaises(ValueError): epoch_class_weights(None, 1, delay)
        for epoch in [0, -1, 1.5, False]:
            with self.assertRaises(ValueError): epoch_class_weights(None, epoch)

    def test_delay_cannot_consume_whole_run_and_creates_no_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, 'weighted epoch'):
                train_run(root, 'invalid', {'kind':'bert', 'class_weight_delay_epochs':2}, {}, {}, [],
                          fixed_epochs=2, training_protocol={'training':{'max_epochs':6}})
            self.assertEqual(list(root.iterdir()), [])
