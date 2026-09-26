import sys,unittest,itertools
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import torch
from ctc_alignment import align_tokens
class CTCAlignmentTests(unittest.TestCase):
    def test_empty_external_audio_does_not_consume_real_frame(self):
        p=torch.tensor([[[-9.,0.,-9.],[-9.,-9.,0.]]]).log_softmax(-1)
        s=align_tokens(p,[1,2],outer=True);self.assertEqual([(x.start,x.end) for x in s],[(0,1),(1,2)])
    def test_prefix_suffix_are_outside_reference(self):
        p=torch.tensor([[[-9.,-9.,-9.,0.],[-9.,0.,-9.,-9.],[-9.,-9.,0.,-9.],[-9.,-9.,-9.,0.]]]).log_softmax(-1)
        s=align_tokens(p,[1,2],outer=True);self.assertEqual([(x.start,x.end) for x in s],[(1,2),(2,3)])
    def test_repeated_tokens_require_real_blank(self):
        p=torch.tensor([[[-9.,0.],[0.,-9.],[-9.,0.]]]).log_softmax(-1)
        s=align_tokens(p,[1,1],outer=True);self.assertEqual([(x.start,x.end) for x in s],[(0,1),(2,3)])
if __name__=='__main__':unittest.main()
