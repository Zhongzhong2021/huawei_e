import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from evaluate_independent_content import evaluate


class ContentEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.protocol = {'samples': [{'sample_id': 'a', 'source_sha256': 'one',
            'reference_words': 10, 'candidate_words': 8},
            {'sample_id': 'b', 'source_sha256': 'two', 'reference_words': 5, 'candidate_words': 2}]}
        self.review = {'schema': 'q1-independent-content-review-v1', 'samples': [
            {'sample_id': 'a', 'source_sha256': 'one', 'content_relation': 'complete', 'listened_full_audio': True}]}

    def test_unfilled_is_not_a_negative_and_coverage_is_not_accuracy(self):
        r = evaluate(self.review, self.protocol)
        self.assertEqual(r['counts']['unfilled'], 1)
        self.assertEqual(r['content_groups']['complete']['candidate_coverage'], .8)
        self.assertIsNone(r['word_precision'])

    def test_absent_candidate_presence_counted_without_inventing_word_truth(self):
        self.review['samples'][0]['content_relation'] = 'absent'
        r = evaluate(self.review, self.protocol)
        self.assertEqual(r['absent_samples_with_any_candidate'], 1)
        self.assertIsNone(r['word_boundary_accuracy'])

    def test_definite_judgment_requires_full_audio(self):
        self.review['samples'][0]['listened_full_audio'] = False
        with self.assertRaises(ValueError):
            evaluate(self.review, self.protocol)

    def test_duplicate_or_wrong_source_rejected(self):
        self.review['samples'].append(self.review['samples'][0].copy())
        with self.assertRaises(ValueError):
            evaluate(self.review, self.protocol)
        self.review['samples'].pop()
        self.review['samples'][0]['source_sha256'] = 'wrong'
        with self.assertRaises(ValueError):
            evaluate(self.review, self.protocol)

    def test_uncertain_not_counted_as_absent(self):
        self.review['samples'][0].update(content_relation='uncertain', listened_full_audio=False)
        r = evaluate(self.review, self.protocol)
        self.assertEqual(r['counts']['uncertain'], 1)
        self.assertEqual(r['content_groups'], {})


if __name__ == '__main__':
    unittest.main()
