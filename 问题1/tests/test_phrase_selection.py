import itertools
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from phrase_selection import compatible, objective, select_phrases
from multigranular_alignment import combine
from common import read
from math_features import words
from whisper_support import lexical_support

ROOT = Path(__file__).resolve().parents[1]


class PhraseSelectionTests(unittest.TestCase):
    def test_rejected_word_proposal_does_not_suppress_supported_group(self):
        reference = words('bright winter morning')
        cfg = read(ROOT / 'config/whisper_alignment.json')
        basecfg = read(ROOT / 'config/automatic_alignment.json')
        baseline = {'words': [{'word_index': i, 'text': w['text'], 'automatic_eligible': i == 0,
                              'interval_s': [0, 1.1] if i == 0 else None,
                              'status': 'automatic_candidate' if i == 0 else 'unresolved'}
                             for i, w in enumerate(reference)]}
        recognized = [{'text': w['text'], 'interval_s': [i, i+1]} for i, w in enumerate(reference)]
        support = lexical_support(reference, recognized, basecfg, 0, 3)
        runs = [{'word_indices': [0, 1, 2], 'anchors': ['BRIGHT', 'WINTER', 'MORNING'],
                 'run_index': 0, 'contexts': [[{'interval_s': [i, i+1]} for i in range(3)] for _ in range(2)]}]
        args = (reference, baseline, support, runs, cfg, [0, 3])
        old = combine(*args, phrase_policy='early_greedy')
        new = combine(*args)
        self.assertEqual(new['words'], old['words'])
        self.assertFalse(new['words'][1]['automatic_eligible'])
        self.assertIsNone(new['words'][1]['interval_s'])
        self.assertFalse(old['phrases'])
        self.assertEqual(new['phrase_only_words'], [1])

    def test_global_group_avoids_overlapping_short_groups(self):
        ws = [{'automatic_eligible': False} for _ in range(3)]
        ps = [{'word_indices': indices, 'interval_s': [indices[0], indices[-1]+1]}
              for indices in [[0, 1], [1, 2], [0, 1, 2]]]
        self.assertEqual(select_phrases(ps, ws), [ps[2]])
        self.assertEqual(len(select_phrases(ps, ws, greedy=True)), 2)

    def test_reference_order_alone_cannot_accept_crossing_audio(self):
        ws = [{'automatic_eligible': False} for _ in range(4)]
        ps = [{'word_indices': [0, 1], 'interval_s': [2, 4]},
              {'word_indices': [2, 3], 'interval_s': [0, 2]}]
        self.assertEqual(len(select_phrases(ps, ws)), 1)

    def test_empty_pool_is_empty(self):
        self.assertEqual(select_phrases([], []), [])

    def test_optimizer_matches_exhaustive_search_and_is_permutation_invariant(self):
        rng = random.Random(714)
        for _ in range(100):
            ws = [{'automatic_eligible': rng.choice([False, False, True])} for _ in range(8)]
            ps = []
            for j in range(8):
                lo = rng.randrange(7); hi = rng.randrange(lo+1, 8)
                t = rng.randrange(12) / 2
                ps.append({'word_indices': list(range(lo, hi+1)), 'interval_s': [t, t+rng.randrange(1, 5)/2], 'id': j})
            ps = [p for p in ps if any(not ws[i]['automatic_eligible'] for i in p['word_indices'])]
            actual = select_phrases(ps, ws)
            optimal = (0, 0, 0)
            for bits in itertools.product([False, True], repeat=len(ps)):
                chosen = sorted([p for p, on in zip(ps, bits) if on], key=lambda p: p['word_indices'][0])
                if all(compatible(a, b) for a, b in zip(chosen, chosen[1:])):
                    optimal = max(optimal, objective(chosen, ws))
            self.assertEqual(objective(actual, ws), optimal)
            rng.shuffle(ps)
            self.assertEqual(select_phrases(ps, ws), actual)


if __name__ == '__main__':
    unittest.main()
