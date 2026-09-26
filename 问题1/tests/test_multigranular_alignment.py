import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from multigranular_alignment import combine
from whisper_support import lexical_support
from common import read
from math_features import words

ROOT=Path(__file__).resolve().parents[1]

class MultigranularTests(unittest.TestCase):
    def setUp(self):
        self.cfg=read(ROOT/'config/whisper_alignment.json')
        self.basecfg=read(ROOT/'config/automatic_alignment.json')
        self.reference=words('bright winter morning')
        self.base={'words':[{'word_index':i,'text':w['text'],'automatic_eligible':False,'interval_s':None,'status':'unresolved'} for i,w in enumerate(self.reference)]}
        self.rec=[{'text':w['text'],'interval_s':[float(i),float(i+1)]} for i,w in enumerate(self.reference)]
        self.support=lexical_support(self.reference,self.rec,self.basecfg,0,5)
        self.run={'word_indices':[0,1,2],'anchors':['BRIGHT','WINTER','MORNING'],'run_index':0,
                  'contexts':[[{'interval_s':[float(i),float(i+1)]} for i in range(3)] for _ in range(2)]}
    def output(self):
        return combine(self.reference,self.base,self.support,[self.run],self.cfg,[0,5])
    def test_consistent_models_restore_words_without_mutating_baseline(self):
        self.assertEqual(self.output()['added_words'],3)
        self.assertFalse(any(w['automatic_eligible'] for w in self.base['words']))
    def test_internal_boundary_uncertainty_returns_group_without_invented_times(self):
        self.run['contexts'][0][0]['interval_s'][1]=0.6
        self.run['contexts'][0][1]['interval_s'][0]=0.6
        result=self.output()
        self.assertEqual(result['added_words'],1)
        self.assertEqual(result['phrase_only_words'],[0,1])
        self.assertTrue(all(not p['internal_word_times_assigned'] for p in result['phrases']))
        self.assertIsNone(result['words'][0]['interval_s'])
    def test_wrong_location_is_rejected_at_word_and_group_levels(self):
        for context in self.run['contexts']:
            for w in context:w['interval_s']=[t+1 for t in w['interval_s']]
        result=self.output()
        self.assertEqual(result['added_words'],0);self.assertFalse(result['phrases'])
    def test_existing_neighbor_prevents_new_overlap(self):
        self.base['words'][0].update(automatic_eligible=True,interval_s=[0,1.5],status='automatic_candidate')
        self.run['contexts'][0][0]['interval_s']=None
        result=self.output()
        self.assertFalse(result['words'][1]['automatic_eligible'])
        self.assertEqual(result['words'][0]['interval_s'],[0,1.5])
    def test_supported_conflicting_location_is_not_selected_by_model_preference(self):
        self.base['words'][0].update(automatic_eligible=True,interval_s=[3,4],status='automatic_candidate')
        result=self.output()
        self.assertFalse(result['words'][0]['automatic_eligible'])
        self.assertEqual(result['words'][0]['competing_intervals_s'],[[3,4],[0,1]])
        self.assertEqual(result['conflicting_baseline_words'],1)
        self.assertFalse(any(0 in p['word_indices'] for p in result['phrases']))
    def test_repeated_phrase_is_not_unique(self):
        rec=self.rec+[{'text':r['text'],'interval_s':[t+3 for t in r['interval_s']]} for r in self.rec]
        result=lexical_support(self.reference,rec,self.basecfg,0,6)
        self.assertEqual(result['exact_unique_words'],0)
        self.assertFalse(result['runs'])
    def test_context_alone_does_not_fill_omitted_middle_word(self):
        rec=[self.rec[0],self.rec[2]]
        result=lexical_support(self.reference,rec,self.basecfg,0,5)
        self.assertFalse(result['words'][1]['exact_unique'])
        self.assertFalse(result['runs'])
    def test_wrong_text_and_silence_produce_no_runs(self):
        for recognized in [[],[{'text':'distant river landscape','interval_s':[0,3]}]]:
            result=lexical_support(self.reference,recognized,self.basecfg,0,5)
            self.assertFalse(result['runs'])

if __name__=='__main__':unittest.main()
