from collections import OrderedDict
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from q2v2.checkpoint import atomic_save, cpu_snapshot, equal_record
from q2v2.engine import train_run


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root / 'best.pt'
        self.record = {'config': {'kind': 'synthetic'}, 'epoch': 1,
                       'vocabulary': torch.arange(5),
                       'state_dict': OrderedDict(w=torch.arange(6.).reshape(2, 3))}
        self.record['state_dict']._metadata = {'': {'version': 1}}

    def test_roundtrip_snapshot_and_success_events(self):
        snap = cpu_snapshot(self.record)
        self.record['state_dict']['w'][0, 0] = 99
        self.assertEqual(snap['state_dict']['w'][0, 0].item(), 0)
        log = self.root / 'events.jsonl'
        atomic_save(snap, self.path, event_path=log)
        equal_record(snap, torch.load(self.path, weights_only=True))
        self.assertEqual([json.loads(s)['phase'] for s in log.read_text().splitlines()],
                         ['cpu_snapshot_started','cpu_snapshot_completed','serialize_started',
                          'serialize_completed','verified','committed'])
        self.assertEqual(list(self.root.glob('*.tmp')), [])

    def test_partial_write_does_not_replace_previous_best(self):
        atomic_save(self.record, self.path)
        old = self.path.read_bytes()
        def fail(value, stream):
            stream.write(b'partial checkpoint')
            raise OSError('synthetic disk full')
        with mock.patch('q2v2.checkpoint.torch.save', side_effect=fail):
            with self.assertRaisesRegex(OSError, 'disk full'):
                atomic_save(self.record, self.path)
        self.assertEqual(self.path.read_bytes(), old)
        self.assertEqual(len(list(self.root.glob('*.tmp'))), 1)

    def test_readback_failure_keeps_previous_best(self):
        atomic_save(self.record, self.path)
        old = self.path.read_bytes()
        with mock.patch('q2v2.checkpoint.torch.load', return_value={'epoch': 999}):
            with self.assertRaisesRegex(ValueError, 'keys mismatch'):
                atomic_save(self.record, self.path)
        self.assertEqual(self.path.read_bytes(), old)

    def test_failed_first_save_does_not_publish_checkpoint(self):
        with mock.patch('q2v2.checkpoint.torch.save', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                atomic_save(self.record, self.path)
        self.assertFalse(self.path.exists())

    def test_snapshot_error_happens_before_disk_checkpoint_creation(self):
        atomic_save(self.record, self.path)
        old = self.path.read_bytes()
        with mock.patch('q2v2.checkpoint.cpu_snapshot', side_effect=RuntimeError('copy failed')):
            with self.assertRaises(RuntimeError):
                atomic_save(self.record, self.path)
        self.assertEqual(self.path.read_bytes(), old)
        self.assertEqual(list(self.root.glob('*.tmp')), [])

    def test_replace_error_keeps_previous_best(self):
        atomic_save(self.record, self.path)
        old = self.path.read_bytes()
        with mock.patch('q2v2.checkpoint.os.replace', side_effect=PermissionError('locked')):
            with self.assertRaises(PermissionError):
                atomic_save(self.record, self.path)
        self.assertEqual(self.path.read_bytes(), old)

    def test_existing_run_is_not_silently_restarted_or_cached(self):
        run = self.root / 'runs/interrupted'
        run.mkdir(parents=True)
        for name in ['best.pt', 'metrics.json']:
            (run/name).write_bytes(b'original evidence')
        with self.assertRaises(FileExistsError):
            train_run(self.root, 'interrupted', {}, {}, {}, [],
                      training_protocol={'training': {'max_epochs': 6}})
        for name in ['best.pt', 'metrics.json']:
            self.assertEqual((run/name).read_bytes(), b'original evidence')

    def test_mismatched_tensor_and_container_types_are_rejected(self):
        with self.assertRaises(ValueError): equal_record(torch.ones(2), torch.ones(3))
        with self.assertRaises(ValueError): equal_record([1], (1,))


if __name__ == '__main__':
    unittest.main()
