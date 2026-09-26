import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from q1_reader import Sample


class ReaderSemanticsTests(unittest.TestCase):
    def sample(self, zero=False):
        arrays = {'text': np.ones((2, 3)), 'audio': np.array([[5., 100.]]),
                  'audio_cells': [[0, 1]], 'audio_dim_mask': [[True, True]],
                  'scene': np.ones((1, 2)), 'video_cells': [[0, 1]],
                  'face_frame_rows': np.array([], dtype=int)}
        source = {'sha256': 'abc', 'zero_audio': zero,
                  'audio': {'start_s': 0, 'end_s': 1},
                  'video': {'time_base': '1/10', 'pts_ticks': [0], 'lengths': [10]}}
        automatic = {'words': [
            {'automatic_eligible': False, 'status': 'ambiguous', 'interval_s': None,
             'diagnostic_interval_s': [.1, .4]},
            {'automatic_eligible': True, 'interval_s': [.5, .9]}]}
        return Sample(arrays, {'source_sha256': 'abc', 'sample_id': 'example'},
                      source, automatic, {})

    def test_uncertain_word_does_not_fall_back_to_diagnostic_time(self):
        self.assertIsNone(self.sample().word(0)['relation'])

    def test_candidate_keeps_provenance_and_is_not_verified(self):
        result = self.sample().word(1)
        self.assertFalse(result['semantic_verified_by_this_function'])
        self.assertEqual(result['relation']['relation_kind'], 'automatic_candidate')
        np.testing.assert_equal(result['relation']['text_rows'], [1])

    def test_physical_window_has_no_assigned_text(self):
        result = self.sample().window(0, 1)
        self.assertEqual(len(result['text_rows']), 0)
        self.assertEqual(result['audio']['observed_seconds'], 1)

    def test_zero_audio_is_observed_but_unusable(self):
        s = self.sample(zero=True)
        self.assertFalse(s.window(0, 1)['audio']['usable'])
        with self.assertRaises(ValueError):
            s.word(1)

    def test_negative_index_cannot_silently_select_last_word(self):
        with self.assertRaises(IndexError):
            self.sample().word(-1)

    def test_resampling_rounding_cannot_create_original_audio_observation(self):
        sample = self.sample()
        sample.arrays['audio_cells'] = [[0, 1.00005]]
        result = sample.window(1, 1.00005)
        self.assertEqual(result['audio']['observed_seconds'], 0)
        self.assertFalse(result['audio']['usable'])
        self.assertFalse(result['audio']['dimension_mask'].any())


if __name__ == '__main__':
    unittest.main()
