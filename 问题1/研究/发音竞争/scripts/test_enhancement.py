"""Structural safety and full-corpus reproducibility of the optional sidecar."""
import copy
import unittest
from itertools import combinations
from build_enhancement import R, Q, build, screen, relation_consistent
from common import read, digest
from phrase_selection import compatible, objective, select_phrases


def phrase(indices, interval):
    return {'word_indices': indices, 'interval_s': interval}


class EnhancementTests(unittest.TestCase):
    def test_order_and_containment(self):
        p = phrase([1, 2, 3], [1, 4])
        self.assertTrue(relation_consistent(p, phrase([2], [2, 3])))
        self.assertFalse(relation_consistent(p, phrase([2], [0.9, 3])))
        self.assertFalse(relation_consistent(p, phrase([4], [3.9, 5])))
        self.assertFalse(relation_consistent(p, phrase([3, 4], [3, 5])))
        self.assertTrue(relation_consistent(p, phrase([4], [4, 5])))

    def test_quarantine_and_physical_bounds(self):
        base = {'words': [{'text': str(i), 'status': 'unknown', 'automatic_eligible': False}
                          for i in range(3)], 'phrases': []}
        source = {'zero_audio': False, 'audio': {'start_s': 0, 'end_s': 4}}
        p = dict(phrase([0, 1, 2], [0, 3]), reference_words=['0', '1', '2'])
        self.assertEqual(screen(p, base, source), ('eligible', [0, 1, 2]))
        base['words'][1]['status'] = 'cross_model_time_conflict'
        self.assertEqual(screen(p, base, source)[0], 'existing_time_conflict')
        base['words'][1]['status'] = 'unknown'
        for interval in [[0, 5], [float('nan'), 3], [-1, 3], [3, 2]]:
            self.assertEqual(screen(dict(p, interval_s=interval), base, source)[0], 'unavailable_audio_range')
        self.assertEqual(screen(p, base, dict(source, zero_audio=True))[0], 'unavailable_audio_range')
        self.assertEqual(screen(dict(p, reference_words=['wrong'] * 3), base, source)[0], 'reference_mismatch')

    def test_corpus_invariants_and_exact_rebuild(self):
        result = build()
        self.assertEqual(result, read(R / 'data/enhancement.json'))
        identity = read(R / 'data/contextual/identity.json')['rows']
        for sample in result['samples']:
            sid = sample['sample_id']
            base = read(Q / 'data/multigranular_alignment' / (sid + '.json'))
            self.assertEqual(result['base_sha256'][sid], digest(Q / 'data/multigranular_alignment' / (sid + '.json')))
            fixed = base['phrases'] + [phrase([i], w['interval_s'])
                                      for i, w in enumerate(base['words']) if w['automatic_eligible']]
            for p in sample['phrases']:
                self.assertTrue(p['new_word_indices'])
                self.assertTrue(all(relation_consistent(p, q) for q in fixed))
                self.assertTrue(identity[p['evidence_row']]['accepted'])
                self.assertFalse(p['internal_word_times_assigned'])
            self.assertTrue(all(compatible(a, b) for a, b in combinations(sample['phrases'], 2)))

    def test_selection_exhaustive_and_order_independent(self):
        # Includes nested and crossed candidates, existing group coverage and ties.
        candidates = [phrase([0, 1], [0, 2]), phrase([1, 2], [1, 3]),
                      phrase([3, 4], [3, 5]), phrase([0, 1, 2, 3, 4], [0, 5]),
                      phrase([5, 6], [4, 6]), phrase([5, 6], [5, 7])]
        words = [{'automatic_eligible': i in {0, 3}} for i in range(7)]
        original = copy.deepcopy(candidates)
        possible = [()]
        for n in range(1, len(candidates) + 1):
            possible += [ps for ps in combinations(candidates, n)
                         if all(compatible(a, b) for a, b in combinations(ps, 2))]
        chosen = select_phrases(candidates, words)
        self.assertEqual(objective(chosen, words), max(objective(ps, words) for ps in possible))
        self.assertEqual(chosen, select_phrases(list(reversed(candidates)), words))
        self.assertEqual(candidates, original)


if __name__ == '__main__':
    unittest.main()
