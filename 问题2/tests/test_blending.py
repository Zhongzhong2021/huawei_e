import copy
import unittest
import numpy as np
from q2v3.blending import blend_predictions


class BlendingTests(unittest.TestCase):
    def setUp(self):
        self.a = {'ids':np.array(['v1$_$0','v2$_$0']), 'labels':np.array([0,1]), 'targets':np.array([-1.,0.]),
                  'probabilities':np.array([[.7,.2,.1],[.1,.6,.3]]), 'classes':np.array([0,1]),
                  'predictions':np.array([-1.,.1])}

    def test_exact_weighted_predictions_and_no_mutation(self):
        b=copy.deepcopy(self.a);b['probabilities']=np.array([[.4,.3,.3],[.2,.4,.4]]);b['predictions']+=.4
        b['classes']=b['probabilities'].argmax(1)
        before=copy.deepcopy(self.a)
        result=blend_predictions(self.a,b,.75)
        np.testing.assert_allclose(result['probabilities'], .75*self.a['probabilities']+.25*b['probabilities'],atol=1e-7)
        np.testing.assert_allclose(result['predictions'], [-.9,.2],atol=1e-7)
        for key in self.a:np.testing.assert_array_equal(self.a[key],before[key])

    def test_reject_invalid_weight(self):
        for value in [0,1,-.1,1.1,True,float('nan'),float('inf')]:
            with self.subTest(value=value),self.assertRaises(ValueError):blend_predictions(self.a,self.a,value)

    def test_reject_missing_duplicate_reordered_ids_or_labels(self):
        for key, replacement in [('ids',np.array(['v2$_$0','v1$_$0'])),('ids',np.array(['v1$_$0','v1$_$0'])),
                                 ('labels',np.array([0,2])),('targets',np.array([-1.,1.]))]:
            b=copy.deepcopy(self.a);b[key]=replacement
            with self.subTest(key=key),self.assertRaises(ValueError):blend_predictions(self.a,b,.9)

    def test_reject_invalid_probabilities_or_outputs(self):
        for key, replacement in [('probabilities',np.ones((2,3))), ('predictions',np.array([4.,0.])),
                                 ('predictions',np.array([float('nan'),0.])), ('classes',np.array([2,2]))]:
            b=copy.deepcopy(self.a);b[key]=replacement
            with self.subTest(key=key),self.assertRaises(ValueError):blend_predictions(self.a,b,.9)

    def test_identical_models_preserve_classes(self):
        result=blend_predictions(self.a,self.a,.9)
        np.testing.assert_array_equal(result['classes'],self.a['classes'])
        np.testing.assert_allclose(result['probabilities'],self.a['probabilities'],atol=1e-7)


if __name__=='__main__':unittest.main()
