import sys
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from partial_acoustic_mapping import map_partial_word_tier
from math_features import words


class PartialMappingTests(unittest.TestCase):
    def test_zero_length_word_does_not_erase_neighbors(self):
        result, status = map_partial_word_tier(words('a b c'), [[0, 1, 'a'], [1, 1, 'b'], [1, 2, 'c']], 5, 2)
        self.assertEqual(status, 'partially_mapped')
        self.assertEqual([w['interval_s'] for w in result], [[5, 6], None, [6, 7]])

    def test_overlapping_words_both_rejected(self):
        result, _ = map_partial_word_tier(words('a b c'), [[0, .5, 'a'], [.5, 1, 'b'], [.9, 1.2, 'c']], 0, 2)
        self.assertEqual([w['interval_s'] for w in result], [[0, .5], None, None])

    def test_invalid_middle_does_not_hide_order_conflict(self):
        result, status = map_partial_word_tier(words('a b c'), [[2, 3, 'a'], [float('nan'), 2, 'b'], [0, 1, 'c']], 0, 4)
        self.assertEqual(status, 'no_valid_intervals')
        self.assertTrue(all(w['interval_s'] is None for w in result))

    def test_merged_tool_word_does_not_invent_internal_time(self):
        result, _ = map_partial_word_tier(words('can not go'), [[0, 1, 'cannot'], [1, 2, 'go']], 0, 2)
        self.assertEqual([w['interval_s'] for w in result], [None, None, [1, 2]])

    def test_label_mismatch_cannot_be_repaired_with_position(self):
        result, status = map_partial_word_tier(words('a b'), [[0, 1, 'a'], [1, 2, 'wrong']], 0, 2)
        self.assertEqual(status, 'label_sequence_mismatch')
        self.assertTrue(all(w['interval_s'] is None for w in result))


if __name__ == '__main__':
    unittest.main()
