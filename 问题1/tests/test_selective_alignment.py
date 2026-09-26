import json,sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from selective_alignment import optimal_relations,align_sample
from math_features import words
class SelectiveTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.cfg=json.loads((ROOT/'config/automatic_alignment.json').read_text())
 def test_internal_insertions_do_not_shift_reference(self):
  r=optimal_relations(['ALPHA','BETA'],['PREFIX','ALPHA','EXTRA','BETA','SUFFIX'],self.cfg)
  self.assertEqual([x['hypothesis_options'] for x in r['reference_tokens']],[[1],[3]]);self.assertEqual(r['edit_cost'],1)
 def test_missing_word_remains_unmatched(self):
  r=optimal_relations(['ALPHA','MISSING','BETA'],['ALPHA','BETA'],self.cfg)
  self.assertTrue(r['reference_tokens'][1]['unmatched_possible']);self.assertFalse(r['reference_tokens'][1]['stable'])
 def test_repeated_occurrences_never_pick_first_as_truth(self):
  r=optimal_relations(['ALPHA','BETA'],['ALPHA','BETA','ALPHA','BETA'],self.cfg)
  self.assertEqual(r['reference_tokens'][0]['hypothesis_options'],[0,2]);self.assertFalse(r['reference_tokens'][0]['stable'])
 def test_common_words_cannot_pass_content_gate(self):
  r=align_sample(words('the with this'),[{'text':t,'interval_s':[i,i+.5]} for i,t in enumerate(['THE','WITH','THIS'])],np.array([[0,.5],[1,1.5],[2,2.5]]),self.cfg)
  self.assertEqual(r['sample_status'],'no_informative_support');self.assertEqual(r['automatic_eligible_words'],0)
 def test_approximate_spelling_not_automatic_word_truth(self):
  r=optimal_relations(['PRODUCTS'],['PRODUCT'],self.cfg)
  self.assertTrue(r['reference_tokens'][0]['stable']);self.assertFalse(r['reference_tokens'][0]['exact'])
 def test_known_relation_stays_available_but_boundary_conflict_abstains(self):
  h=[{'text':'ALPHA','interval_s':[0,.5]},{'text':'BETA','interval_s':[1,1.5]}]
  r=align_sample(words('alpha beta'),h,np.array([[0,.5],[1,1.5]]),self.cfg);self.assertEqual(r['automatic_eligible_words'],2)
  r=align_sample(words('alpha beta'),h,np.array([[0,.5],[2,2.5]]),self.cfg);self.assertEqual(r['automatic_eligible_words'],1);self.assertEqual(r['words'][1]['status'],'boundary_disagreement')
 def test_silence_never_has_eligible_words(self):
  r=align_sample(words('alpha beta'),[],np.zeros((2,2)),self.cfg,no_signal=True);self.assertEqual(r['sample_status'],'no_signal');self.assertEqual(r['automatic_eligible_words'],0)
 def test_repetition_is_content_evidence_but_not_unique_location(self):
  h=[{'text':t,'interval_s':[i,i+.5]} for i,t in enumerate(['ALPHA','BETA','ALPHA','BETA'])]
  r=align_sample(words('alpha beta'),h,np.array([[0,.5],[1,1.5]]),self.cfg)
  self.assertEqual(r['sample_status'],'supported');self.assertEqual(r['automatic_eligible_words'],0)
  self.assertEqual(r['context_envelopes'][0]['status'],'context_only_not_alignment')
 def test_repeated_same_token_does_not_inflate_support(self):
  h=[{'text':'ALPHA','interval_s':[0,.5]}]
  r=align_sample(words('alpha alpha'),h,np.array([[0,.5],[0,.5]]),self.cfg)
  self.assertEqual(r['informative_anchors'],1);self.assertNotEqual(r['sample_status'],'supported')
 def test_all_optimal_paths_against_exhaustive_oracle(self):
  from itertools import product
  for n in range(1,4):
   for m in range(4):
    for ref in product('AB',repeat=n):
     for hyp in product('AB',repeat=m):
      paths=[]
      def visit(i,j,cost,mates):
       if i==n:paths.append((cost,mates));return
       visit(i+1,j,cost+1,mates+[None])
       if j<m:
        visit(i+1,j+1,cost+(ref[i]!=hyp[j]),mates+[j if ref[i]==hyp[j] else None])
        visit(i,j+1,cost+1,mates)
      for start in range(m+1):visit(0,start,0,[])
      best=min(c for c,_ in paths);optimal=[mates for c,mates in paths if c==best];r=optimal_relations(ref,hyp,self.cfg)
      self.assertEqual(r['edit_cost'],best)
      for i,item in enumerate(r['reference_tokens']):
       options={mates[i] for mates in optimal}
       self.assertEqual(set(item['hypothesis_options']),options-{None});self.assertEqual(item['unmatched_possible'],None in options)
if __name__=='__main__':unittest.main()
