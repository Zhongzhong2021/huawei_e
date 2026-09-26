"""Delivery reader: immutable baseline plus reproducible local corroboration."""
from common import read, digest
from q1_reader import Dataset as BaselineDataset, Sample
from local_evidence import corroborated_local_candidates
from feature_io import pad


class DeliverySample(Sample):
    def word(self, word_index):
        result = super().word(word_index)
        result['decision_evidence'] = self.automatic['words'][word_index]['status']
        result['decision_policy'] = self.automatic['decision_policy']
        return result


class Dataset(BaselineDataset):
    def __init__(self, root):
        super().__init__(root)
        self.config = read(self.root / 'config/automatic_alignment.json')
        self.manifest = read(self.root / 'data/local_evidence_corroborated/manifest.json')
        if digest(self.root / 'config/automatic_alignment.json') != self.manifest['config_sha256']:
            raise ValueError('Decision configuration differs from generated relations')
        for name, expected in self.manifest['scripts'].items():
            if digest(self.root / 'scripts' / name) != expected:
                raise ValueError('Local decision implementation changed: ' + name)

    def sample(self, sid):
        baseline = super().sample(sid)
        record = read(self.root / 'data/local_evidence_corroborated' / (sid + '.json'))
        if (record['sample_id'] != sid or record['source_sha256'] != baseline.source['sha256']
            or record['baseline_result_sha256'] != digest(self.root / 'data/automatic_alignment' / (sid + '.json'))):
            raise ValueError('Local relation source mismatch')
        expected = corroborated_local_candidates(baseline.automatic, self.config)
        for key, value in expected.items():
            if record[key] != value:
                raise ValueError('Stored relation differs from deterministic rule: ' + key)
        sample = DeliverySample(baseline.arrays, baseline.metadata, baseline.source,
                                record, baseline.temporal_index)
        sample.baseline_automatic = baseline.automatic
        return sample

    def batch(self, sample_ids, modality):
        """Pad native sequences with length, dimension and sample-availability masks.

        Availability refers to observations, not verified speech/text agreement.
        No word timing, face identity or missing modality is imputed.
        """
        samples = [self.sample(sid) for sid in sample_ids]
        return pad([(s.arrays, s.metadata) for s in samples], modality)
