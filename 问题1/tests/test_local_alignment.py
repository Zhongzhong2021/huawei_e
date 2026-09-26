import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from local_alignment import reference_in_hypothesis
class LocalAlignmentTests(unittest.TestCase):
    def test_outer_words_not_reference_errors(self):
        r=reference_in_hypothesis(['GOOD','DAY'],['INTRO','GOOD','DAY','OUTRO'])
        self.assertEqual(r['error_rate'],0);self.assertEqual(r['hypothesis_span'],[1,3]);self.assertEqual(r['pairs'],[(0,1),(1,2)])
    def test_internal_word_not_free(self):
        r=reference_in_hypothesis(['GOOD','DAY'],['GOOD','BAD','DAY']);self.assertEqual(r['error_rate'],.5)
    def test_repetition_exposes_multiple_candidates(self):
        r=reference_in_hypothesis(['GO'],['GO','STOP','GO']);self.assertEqual(r['tied_endpoints'],[1,3])
    def test_empty_and_unrelated_not_confirmed(self):
        self.assertEqual(reference_in_hypothesis([],['A'])['status'],'empty_reference')
        self.assertEqual(reference_in_hypothesis(['A'],[])['error_rate'],1)
        self.assertEqual(reference_in_hypothesis(['A'],['B'])['pairs'],[])
if __name__=='__main__':unittest.main()
