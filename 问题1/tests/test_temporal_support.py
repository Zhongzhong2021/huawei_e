import sys
import unittest
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from temporal_support import aggregate, overlap_weights, union_intervals, interval_relation


class TemporalSupportTests(unittest.TestCase):
    def test_undefined_pitch_does_not_dilute_valid_pitch(self):
        output = aggregate([[10, 200], [30, 0]], [[0, 1], [1, 2]], [0, 2], [[1, 1], [1, 0]])
        np.testing.assert_allclose(output['values'], [20, 200])
        np.testing.assert_allclose(output['valid_seconds_per_dimension'], [2, 1])

    def test_video_gap_never_contributes(self):
        output = aggregate([[2], [8]], [[0, 1], [1, 2]], [0, 2], support=[[0, .5], [1.5, 2]])
        self.assertEqual(output['observed_seconds'], 1)
        self.assertEqual(output['observed_fraction'], .5)
        self.assertEqual(output['values'][0], 5)
        self.assertEqual(aggregate([[2]], [[0, 2]], [.6, 1.4], support=[[0, .5], [1.5, 2]])['observed_seconds'], 0)

    def test_overlapping_source_frames_do_not_double_count(self):
        np.testing.assert_allclose(overlap_weights([0, 2], [[0, 2]], [[0, 1.5], [1, 2]]), [2])

    def test_observed_zero_audio_is_not_usable_emotion_evidence(self):
        result = aggregate([[0, -20]], [[0, 1]], [0, 1], usable=False)
        self.assertTrue(result['observation_present'])
        self.assertFalse(result['usable'])
        self.assertFalse(result['dimension_mask'].any())

    def test_support_is_additive_and_translation_equivariant(self):
        cells = np.array([[0, 1], [1, 2]])
        full = overlap_weights([0, 2], cells, [[.2, 1.8]])
        left = overlap_weights([0, .7], cells, [[.2, 1.8]])
        right = overlap_weights([.7, 2], cells, [[.2, 1.8]])
        np.testing.assert_allclose(full, left + right)
        np.testing.assert_allclose(full, overlap_weights([5, 7], cells + 5, [[5.2, 6.8]]))

    def test_invalid_or_overlapping_feature_cells_rejected(self):
        with self.assertRaises(ValueError):
            overlap_weights([0, 2], [[0, 1.5], [1, 2]])
        with self.assertRaises(ValueError):
            union_intervals([[0, float('nan')]])

    def test_empty_features_have_explicit_missing_output(self):
        result = aggregate(np.zeros((0, 3)), [], [0, 1])
        self.assertFalse(result['usable'])
        np.testing.assert_equal(result['dimension_mask'], [False] * 3)


if __name__ == '__main__':
    unittest.main()
