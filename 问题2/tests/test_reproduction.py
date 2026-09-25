import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from q2v3.reproduction import preflight
from q2v3.__main__ import main


class ReproductionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.protocol = {'training': {
            'batch_size': 32, 'max_epochs': 6, 'patience': 3,
            'backbone_lr': 2e-5, 'head_lr': 5e-4,
            'weight_decay': .01, 'gradient_clip': 1.,
        }, 'optimization_deadline': '2026-09-24T13:26:25+00:00'}
        for folder in ['final', 'configs', 'data/processed', 'pretrained']:
            (self.root / folder).mkdir(parents=True)
        self.protocol_path = self.root / 'configs/frozen_protocol.json'
        self.protocol_path.write_text(json.dumps(self.protocol), encoding='utf-8')
        self.source = self.root / 'pretrained/bert_base_uncased.pt'
        torch.save({}, self.source)
        self.checkpoint = self.root / 'final/model.pt'
        torch.save({'config': {'kind': 'bert', 'distillation': False},
                    'vocabulary': torch.tensor([0, 100, 101, 102, 103]), 'epoch': 4}, self.checkpoint)
        for split in ['train', 'valid']:
            np.savez(self.root / f'data/processed/{split}.npz', ids=np.array(['synthetic']))

    def check(self, **extra):
        return preflight(self.root, 'replica', 42, self.root, self.source, **extra)

    def argv(self, *extra):
        return ['q2v3', '--root', str(self.root), 'train', '--name', 'replica',
                '--data-root', str(self.root), '--pretrained', str(self.source), *extra]

    def test_tracked_protocol_fallback_is_read_only(self):
        plan = self.check()
        self.assertEqual(plan['training_protocol'], self.protocol)
        self.assertEqual(Path(plan['inputs']['protocol']), self.protocol_path)
        self.assertFalse((self.root / 'runs').exists())

    def test_existing_study_protocol_has_priority(self):
        (self.root / 'study').mkdir()
        p = self.root / 'study/protocol.json'
        p.write_text(json.dumps(self.protocol), encoding='utf-8')
        self.assertEqual(Path(self.check()['inputs']['protocol']), p)

    def test_missing_input_fails_before_loading_any_model(self):
        self.protocol_path.unlink()
        with mock.patch('sys.argv', self.argv('--dry-run')), mock.patch('torch.load') as load:
            with self.assertRaisesRegex(FileNotFoundError, 'protocol'):
                main()
            load.assert_not_called()
        self.assertFalse((self.root / 'runs').exists())

    def test_reject_path_traversal_and_windows_reserved_names(self):
        for name in ['../old', '..\\old', '/root', 'a/b', 'a\\b', 'CON', 'lpt1.txt']:
            with self.subTest(name=name), self.assertRaises(ValueError):
                preflight(self.root, name, 42, self.root, self.source)

    def test_reject_existing_run_and_invalid_seed(self):
        (self.root / 'runs/replica').mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            self.check()
        for seed in [-1, 2**32, True]:
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                preflight(self.root, 'fresh', seed)

    def test_invalid_numeric_protocol_settings_are_rejected(self):
        for field, value in [('batch_size', 0), ('patience', True), ('head_lr', float('nan')), ('weight_decay', -.1)]:
            candidate = json.loads(json.dumps(self.protocol))
            candidate['training'][field] = value
            self.protocol_path.write_text(json.dumps(candidate), encoding='utf-8')
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                self.check()

    def test_dry_run_never_fits_or_creates_run(self):
        output = io.StringIO()
        with mock.patch('sys.argv', self.argv('--dry-run')), contextlib.redirect_stdout(output), \
                mock.patch('q2v2.engine.train_run') as fit, mock.patch('q2.data.load_data') as load:
            main()
            fit.assert_not_called()
            load.assert_not_called()
        plan = json.loads(output.getvalue())
        self.assertFalse(plan['training_started'])
        self.assertEqual(plan['fixed_epochs'], 4)
        self.assertFalse((self.root / 'runs').exists())

    def test_cli_passes_explicit_protocol_and_frozen_epoch(self):
        with mock.patch('sys.argv', self.argv()), contextlib.redirect_stdout(io.StringIO()), \
                mock.patch('q2v2.engine.train_run', return_value={}) as fit, \
                mock.patch('q2.data.load_data', return_value={}):
            main()
        fit.assert_called_once()
        self.assertEqual(fit.call_args.kwargs['training_protocol'], self.protocol)
        self.assertEqual(fit.call_args.kwargs['fixed_epochs'], 4)
        self.assertTrue(fit.call_args.kwargs['reproduction'])


if __name__ == '__main__':
    unittest.main()
