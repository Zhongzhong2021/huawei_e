import itertools,unittest,math
import numpy as np
from phone_evidence import ctc_scores,mixture_score
class PhoneScoreTests(unittest.TestCase):
 def test_ctc_likelihood_matches_exhaustive_path_sum(self):
  probabilities=np.array([[.6,.3,.1],[.2,.7,.1],[.3,.2,.5],[.5,.2,.3]],np.float32)
  seqs=[(),(1,),(2,),(1,2),(1,1)]
  actual=ctc_scores(np.log(probabilities),seqs)
  for target,score in zip(seqs,actual):
   total=0.
   for path in itertools.product(range(3),repeat=4):
    labels=tuple(c for i,c in enumerate(path) if c!=0 and (i==0 or c!=path[i-1]))
    if labels==target:total+=math.prod(float(probabilities[i,c]) for i,c in enumerate(path))
   self.assertAlmostEqual(float(score),math.log(total),places=5)
 def test_equal_priors_do_not_reward_duplicating_variants(self):
  lp=np.log(np.array([[.1,.8,.1],[.8,.1,.1]],np.float32))
  self.assertAlmostEqual(mixture_score(lp,[(1,)]),mixture_score(lp,[(1,),(1,)]),places=5)
 def test_silent_blank_dominant_audio_favors_deletion(self):
  lp=np.log(np.array([[.99,.005,.005]]*8,np.float32))
  a,b=ctc_scores(lp,[(1,),(1,2,1)])
  self.assertGreater(a,b)

class PronunciationTests(unittest.TestCase):
 def setUp(self):
  import os,json
  from pathlib import Path
  from phone_evidence import Pronunciations
  self.pron=Pronunciations(json.loads((Path(os.environ.get('Q1_PHONE_MODEL','/workspace/q1-phone-model'))/'vocab.json').read_text()))
 def test_numeric_value_is_expanded_before_pronunciation(self):
  self.assertEqual(self.pron.phones('20,000'),self.pron.phones('twenty thousand'))
  self.assertEqual(self.pron.phones('1,234'),self.pron.phones('one thousand two hundred and thirty four'))
 def test_contextual_fusion_remains_a_group_path(self):
  phones=self.pron.phones('we could have arrived').replace('|',' ').split()
  ids=tuple(self.pron.vocab[p] for p in phones)
  self.assertIn(ids,self.pron.variants(('we','could','have','arrived')))
 def test_unrepresentable_word_has_no_fabricated_phone_path(self):
  self.assertEqual(self.pron.variants(('',)),())

if __name__=='__main__':unittest.main()
