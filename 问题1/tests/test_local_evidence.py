import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from local_evidence import local_candidates, corroborated_local_candidates


def result(tokens, indices=None):
    indices = indices or list(range(len(tokens)))
    words = [{'word_index': i, 'text': token, 'status': 'sample_support_insufficient',
              'automatic_eligible': False, 'interval_s': None,
              'diagnostic_interval_s': [j, j + .5], 'boundary_disagreement_s': 0,
              'recognition_options': [{'exact': True, 'stable': True,
                  'unmatched_possible': False, 'hypothesis_options': [j]}]}
             for i, (token, j) in enumerate(zip(tokens, indices))]
    hyp = [{'text': 'EXTRA'} for _ in range(max(indices) + 1)]
    for token, j in zip(tokens, indices):
        hyp[j] = {'text': token}
    return {'words': words, 'recognition': hyp, 'sample_status': 'insufficient_evidence'}


class LocalEvidenceTests(unittest.TestCase):
    cfg = {'boundary_agreement_s': .2, 'informative_min_chars': 4,
           'minimum_informative_anchors': 2, 'function_words': ['THIS', 'THAT']}

    def test_two_distinct_ordered_anchors_recover_local_run(self):
        original = result(['BALANCED', 'AND', 'HYDRATING'])
        output = local_candidates(original, self.cfg)
        self.assertEqual(output['new_local_candidate_words'], 3)
        self.assertTrue(all(not w['automatic_eligible'] for w in original['words']))

    def test_isolated_function_word_or_repeated_anchor_not_enough(self):
        for tokens in [['ALL'], ['THIS', 'THAT'], ['SMOOTH', 'SMOOTH']]:
            self.assertEqual(local_candidates(result(tokens), self.cfg)['new_local_candidate_words'], 0)

    def test_hypothesis_insertion_breaks_local_support(self):
        self.assertEqual(local_candidates(result(['BALANCED', 'HYDRATING'], [0, 2]), self.cfg)['new_local_candidate_words'], 0)

    def test_ambiguous_word_cannot_bridge_anchors(self):
        r = result(['BALANCED', 'AND', 'HYDRATING'])
        r['words'][1]['recognition_options'][0]['unmatched_possible'] = True
        self.assertEqual(local_candidates(r, self.cfg)['new_local_candidate_words'], 0)

    def test_boundary_disagreement_breaks_run(self):
        r = result(['BALANCED', 'HYDRATING'])
        r['words'][1]['boundary_disagreement_s'] = .3
        self.assertEqual(local_candidates(r, self.cfg)['new_local_candidate_words'], 0)

    def test_zero_signal_never_recovered(self):
        r = result(['BALANCED', 'HYDRATING']); r['sample_status'] = 'no_signal'
        self.assertEqual(local_candidates(r, self.cfg)['new_local_candidate_words'], 0)

    def test_single_coincidental_phrase_does_not_override_whole_text_rejection(self):
        r = result(['HIGH', 'SCHOOL'])
        self.assertEqual(corroborated_local_candidates(r, self.cfg)['new_local_candidate_words'], 0)

    def test_two_separated_runs_with_different_anchors_corroborate(self):
        r = result(['BALANCED', 'HYDRATING', 'SMOOTH', 'HEALING'], [0, 1, 3, 4])
        self.assertEqual(corroborated_local_candidates(r, self.cfg)['new_local_candidate_words'], 4)

    def test_repeated_anchor_vocabulary_is_not_separate_corroboration(self):
        r = result(['HIGH', 'SCHOOL', 'HIGH', 'SCHOOL'], [0, 1, 3, 4])
        self.assertEqual(corroborated_local_candidates(r, self.cfg)['new_local_candidate_words'], 0)


if __name__ == '__main__':
    unittest.main()
