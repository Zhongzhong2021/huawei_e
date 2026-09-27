"""Guard simplification against uncertain or subgroup-specific degradation."""
import copy
import json
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_no_augmentation_study import simplicity

class SimplicityTests(unittest.TestCase):
    def setUp(self):
        self.protocol=json.loads((ROOT/'configs/no_augmentation_study.json').read_text())
        self.analysis={'selection':{'checks':{'clean_accuracy':True,'mean_score':False,'paired_successes':False},'paired_score_differences':[0.,0.,0.]},'paired_intervals':[
            {'group':g,'metric':m,'lower':-.001,'upper':.001}
            for g in ('clean','single','low_joint','medium_joint','heavy_joint','low_medium') for m in ('macro_f1','mae')]}

    def test_equivalent_performance_can_simplify_without_superiority(self):
        self.assertTrue(simplicity(self.analysis,self.protocol)['passed'])

    def test_uncertain_heavy_degradation_rejects_simplification(self):
        a=copy.deepcopy(self.analysis)
        next(r for r in a['paired_intervals'] if r['group']=='heavy_joint' and r['metric']=='macro_f1')['lower']=-.006
        self.assertFalse(simplicity(a,self.protocol)['passed'])

    def test_regression_cost_or_fold_instability_rejects(self):
        a=copy.deepcopy(self.analysis)
        next(r for r in a['paired_intervals'] if r['group']=='clean' and r['metric']=='mae')['upper']=.021
        self.assertFalse(simplicity(a,self.protocol)['passed'])
        self.analysis['selection']['paired_score_differences']=[-.003,-.003,.01]
        self.assertFalse(simplicity(self.analysis,self.protocol)['passed'])
