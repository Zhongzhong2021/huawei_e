import unittest
import numpy as np
from check_attachments import difference,match_field
class ComparisonTests(unittest.TestCase):
 def test_dtype_rounding_is_not_a_missing_value(self):
  a=np.array([[1/3,1/7]],dtype=np.float64);r=difference(a,a.astype(np.float32))
  self.assertTrue(r['equal_after_target_cast']);self.assertEqual(r['changed_cells'],0)
 def test_new_zeros_are_distinct_from_changed_nonzero_values(self):
  r=difference(np.array([[1.,2.],[3.,4.]]),np.array([[0.,0.],[3.,100.]]))
  self.assertEqual(r['new_zero_row_spans'],[[0,1]]);self.assertEqual(r['changed_nonzero_target_cells'],1)
 def test_masked_matching_does_not_accept_arbitrary_replacement(self):
  bank=np.array([[[1.,2.],[3.,4.]],[[5.,6.],[7.,8.]]])
  self.assertEqual(match_field(bank,np.array([[0.,0.],[3.,4.]]))[0],[0])
  self.assertEqual(match_field(bank,np.array([[0.,0.],[3.,9.]]))[0],[])
 def test_token_sequence_axis_is_preserved(self):
  a=np.ones((3,50),dtype=int);b=a.copy();b[:,7:9]=0
  self.assertEqual(difference(a,b)['new_zero_row_spans'],[[7,9]])
if __name__=='__main__':unittest.main()
