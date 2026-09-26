import sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from acoustic_mapping import map_word_tier
from math_features import words
class AcousticMappingTests(unittest.TestCase):
 def test_contraction_parts_and_original_offset(self):
  out,status=map_word_tier(words("It's good"),[[.1,.2,'it'],[.2,.3,"'s"],[.3,.8,'good']],2.,1.)
  self.assertEqual(status,'mapped');self.assertEqual(out[0]['interval_s'],[2.1,2.3]);self.assertEqual(out[1]['interval_s'],[2.3,2.8])
 def test_number_expansion_maps_back_to_single_original_word(self):
  out,status=map_word_tier(words('100'),[[0,.3,'one'],[.3,.8,'hundred']],0,1)
  self.assertEqual(out[0]['interval_s'],[0,.8]);self.assertEqual(out[0]['text'],'100')
 def test_no_invented_boundary_inside_tool_word(self):
  out,status=map_word_tier(words('can not'),[[0,1,'cannot']],0,1)
  self.assertEqual(status,'mapped');self.assertTrue(all(w['interval_s'] is None for w in out))
 def test_unknown_or_mismatched_labels_are_not_positionally_assigned(self):
  for labels in [[[0,1,'<unk>']],[[0,1,'beta']],[[0,.1,"'"],[.1,1,'alpha']]]:
   out,status=map_word_tier(words('alpha'),labels,0,1);self.assertEqual(status,'label_sequence_mismatch');self.assertIsNone(out[0]['interval_s'])
 def test_nonmonotone_and_out_of_range_rejected(self):
  for labels in [[[0,.8,'alpha'],[.5,1,'beta']],[[0,1,'alpha'],[1,2,'beta']]]:
   out,status=map_word_tier(words('alpha beta'),labels,0,1);self.assertEqual(status,'invalid_tool_interval')
if __name__=='__main__':unittest.main()
