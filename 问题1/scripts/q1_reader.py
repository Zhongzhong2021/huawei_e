"""Portable reading of native features, physical indices and selective candidates.

No model cache, source-media absolute path, or human review file is needed.
Candidate availability is deliberately not presented as semantic correctness.
"""
from pathlib import Path
import operator
import numpy as np
from common import read, digest
from feature_io import load
from temporal_support import interval_relation


class Dataset:
    def __init__(self, root):
        self.root = Path(root)
        records = read(self.root / 'data/inputs.json')['samples']
        self.records = {s['sample_id']: s for s in records}
        if len(records) != len(self.records):
            raise ValueError('Duplicate sample identifiers')

    def sample(self, sid):
        source = self.records[sid]
        base = self.root / 'data'
        arrays, metadata = load(base / 'features', sid)
        automatic = read(base / 'automatic_alignment' / (sid + '.json'))
        index_metadata = read(base / 'temporal_index' / (sid + '.json'))
        for record in (metadata, automatic, index_metadata):
            if record['sample_id'] != sid or record['source_sha256'] != source['sha256']:
                raise ValueError('Sample identity or source digest mismatch')
        for relative, key in [(f'features/{sid}.npz', 'native_features_sha256'),
                              (f'features/{sid}.json', 'native_metadata_sha256'),
                              (f'temporal_index/{sid}.npz', 'index_sha256')]:
            if digest(base / relative) != index_metadata[key]:
                raise ValueError('Feature or index digest mismatch: ' + relative)
        if metadata['original_text'] != source['text']:
            raise ValueError('Original text mismatch')
        if len(automatic['words']) != len(metadata['words']):
            raise ValueError('Candidate word count mismatch')
        for i, (word, native_word) in enumerate(zip(automatic['words'], metadata['words'])):
            if word['word_index'] != i or word['text'] != native_word['text']:
                raise ValueError('Candidate word identity mismatch')
        with np.load(base / 'temporal_index' / (sid + '.npz'), allow_pickle=False) as z:
            index = {k: z[k] for k in z.files}
        return Sample(arrays, metadata, source, automatic, index)


class Sample:
    def __init__(self, arrays, metadata, source, automatic, index):
        self.arrays, self.metadata, self.source = arrays, metadata, source
        self.automatic, self.temporal_index = automatic, index

    def window(self, start_s, end_s):
        """A physical interval carries no inferred text relationship."""
        return interval_relation(self.arrays, self.metadata, self.source,
                                 [start_s, end_s], 'physical_time_window')

    def word(self, word_index):
        """Return a selective candidate or explicit missing relationship.

        No fallback to forced-alignment diagnostics or human intervals.
        """
        word_index = operator.index(word_index)
        if not 0 <= word_index < len(self.automatic['words']):
            raise IndexError('Word index outside original text')
        word = self.automatic['words'][word_index]
        if not word['automatic_eligible']:
            return {'sample_id': self.metadata['sample_id'], 'word_index': word_index,
                    'status': word['status'], 'relation': None,
                    'semantic_verified_by_this_function': False}
        if self.source['zero_audio']:
            raise ValueError('Zero audio cannot carry an eligible speech relationship')
        relation = interval_relation(self.arrays, self.metadata, self.source,
                                     word['interval_s'], 'automatic_candidate', [word_index])
        return {'sample_id': self.metadata['sample_id'], 'word_index': word_index,
                'status': 'automatic_candidate', 'relation': relation,
                'semantic_verified_by_this_function': False}
