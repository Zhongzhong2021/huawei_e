import json
import shutil
import sys
import tempfile
import unittest
import numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from delivery_reader import Dataset


class DeliveryReaderTests(unittest.TestCase):
    sid = '-AUZQgSxyPQ$_$2'

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        files = ['data/inputs.json', 'config/automatic_alignment.json',
                 'data/local_evidence_corroborated/manifest.json',
                 'scripts/local_evidence.py', 'scripts/run_local_evidence.py']
        for folder in ['features', 'temporal_index']:
            files.extend(f'data/{folder}/{self.sid}.{suffix}' for suffix in ['npz', 'json'])
        files.extend(f'data/{folder}/{self.sid}.json'
                     for folder in ['automatic_alignment', 'local_evidence_corroborated'])
        for relative in files:
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, target)

    def tearDown(self):
        self.temporary.cleanup()

    def test_actual_restored_words_keep_explicit_provenance(self):
        sample = Dataset(self.root).sample(self.sid)
        restored = [i for i, w in enumerate(sample.automatic['words']) if w['status'] == 'local_coherent_candidate']
        self.assertEqual(len(restored), 14)
        self.assertEqual(sample.baseline_automatic['automatic_eligible_words'], 0)
        for i in restored:
            r = sample.word(i)
            self.assertEqual(r['decision_evidence'], 'local_coherent_candidate')
            self.assertFalse(r['semantic_verified_by_this_function'])

    def test_modified_relation_rejected_instead_of_silently_read(self):
        path = self.root / 'data/local_evidence_corroborated' / (self.sid + '.json')
        data = json.loads(path.read_text())
        data['words'][0]['automatic_eligible'] = True
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'deterministic rule'):
            Dataset(self.root).sample(self.sid)

    def test_changed_config_rejected(self):
        path = self.root / 'config/automatic_alignment.json'
        data = json.loads(path.read_text()); data['boundary_agreement_s'] = 1
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'configuration differs'):
            Dataset(self.root)

    def test_real_zero_tracks_are_masked_in_public_batch_interface(self):
        dataset = Dataset(ROOT)
        silent = [sid for sid, record in dataset.records.items() if record['zero_audio']]
        normal = next(sid for sid, record in dataset.records.items() if not record['zero_audio'])
        batch = dataset.batch(silent + [normal], 'audio')
        self.assertEqual(len(silent), 2)
        self.assertTrue((batch['lengths'] > 0).all())
        self.assertFalse(batch['sample_available'][:-1].any())
        self.assertFalse(batch['dimension_mask'][:-1].any())
        self.assertFalse(batch['values'][:-1].any())
        self.assertTrue(batch['sample_available'][-1])
        self.assertTrue(batch['dimension_mask'][-1].any())
        np.testing.assert_array_equal(batch['sample_ids'], silent + [normal])


if __name__ == '__main__':
    unittest.main()
