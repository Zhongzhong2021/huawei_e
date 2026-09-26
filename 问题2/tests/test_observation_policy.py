import unittest
import numpy as np
from q2.data import convert
from q2.observation_study import apply_condition
from test_contract import fixture

class ObservationTests(unittest.TestCase):
 def test_placeholder_presence_is_distinct_from_lexical_availability(self):
  raw=fixture();raw['text_bert'][0,0,2]=100
  retain,_=convert(raw,['a','b']);masked,_=convert(raw,['a','b'],unknown_policy='mask_unk')
  self.assertTrue(retain['text_present'][0,2]);self.assertTrue(retain['text_unknown'][0,2]);self.assertFalse(retain['text_lexical_available'][0,2])
  self.assertTrue(retain['observed'][0,2,0]);self.assertFalse(masked['observed'][0,2,0])
  np.testing.assert_array_equal(retain['tokens'],masked['tokens'])
  np.testing.assert_array_equal(retain['observed'][:,:,1:],masked['observed'][:,:,1:])
 def test_paired_joint_masks_are_identical_and_do_not_mutate_source(self):
  data,_=convert(fixture(),['a','b']);original=data['tokens'].copy()
  a,ma=apply_condition(data,'joint_50_scattered','retain');b,mb=apply_condition(data,'joint_50_scattered','mask_unk')
  np.testing.assert_array_equal(ma,mb);np.testing.assert_array_equal(data['tokens'],original)
  self.assertEqual(int(ma.sum()),4);self.assertTrue((a['tokens'][ma]==100).all())
  self.assertFalse(a['observed'][:,:,1:][ma].any());self.assertFalse(b['observed'][mb].any())
  self.assertTrue(a['observed'][:,:,0][ma].all());self.assertFalse(ma[:,[0,5,49]].any())
 def test_invalid_policy_and_condition_fail(self):
  with self.assertRaises(ValueError):convert(fixture(),['a','b'],unknown_policy='guess')
  data,_=convert(fixture(),['a','b'])
  with self.assertRaises(ValueError):apply_condition(data,'special_15','retain')
