import unittest
import numpy as np
import torch
from q2v2.model import PretrainedMultimodal


class PretrainedTests(unittest.TestCase):
    def test_hidden_token_cannot_influence_other_observed_tokens(self):
        torch.set_num_threads(4)
        torch.manual_seed(42)
        model = PretrainedMultimodal({"layers":[0],"multimodal":True},[0,100,101,102,103,2001,2002,2003]).eval()
        tokens = torch.tensor([[101,2001,2002,102]+[0]*46])
        valid = torch.zeros(1,50,dtype=torch.bool)
        valid[:,1:3] = True
        drop = torch.zeros(1,50,3,dtype=torch.bool)
        drop[:,1,:] = True
        batch = dict(tokens=tokens,audio=torch.randn(1,50,74),vision=torch.randn(1,50,35),
                     valid=valid,observed=valid[:,:,None].expand(-1,-1,3),drop=drop)
        with torch.no_grad():
            before = model(**batch)
            batch["tokens"][:,1] = 2003
            batch["audio"][:,1] = 1e6
            batch["vision"][:,1] = -1e6
            after = model(**batch)
        for x,y in zip(before,after):
            self.assertTrue(torch.equal(x,y))

    def test_empty_sequence_and_partial_zero_vectors_remain_finite(self):
        torch.set_num_threads(4)
        model = PretrainedMultimodal({"layers":[0],"multimodal":True},[0,100,101,102,103]).eval()
        batch = dict(tokens=torch.zeros(2,50,dtype=torch.long),audio=torch.zeros(2,50,74),
                     vision=torch.zeros(2,50,35),valid=torch.zeros(2,50,dtype=torch.bool),
                     observed=torch.zeros(2,50,3,dtype=torch.bool))
        batch["valid"][1,1] = True
        batch["observed"][1,1,1] = True
        batch["audio"][1,1,0] = 1
        with torch.no_grad():
            outputs = model(**batch)
        for value in outputs:
            self.assertTrue(torch.isfinite(value).all())

    def test_unknown_vocabulary_and_all_missing(self):
        torch.set_num_threads(4)
        model = PretrainedMultimodal({"layers":[0], "multimodal":True}, [0,100,101,102,103,2023]).eval()
        self.assertEqual(int(model.backbone.lookup[29999]), int(model.backbone.lookup[100]))
        tokens = torch.tensor([[101,2023,102] + [0]*47])
        valid = torch.zeros((1,50), dtype=torch.bool)
        valid[:,1] = True
        observed = valid.unsqueeze(-1).expand(-1,-1,3)
        batch = dict(tokens=tokens, audio=torch.randn(1,50,74), vision=torch.randn(1,50,35),
                     valid=valid, observed=observed, drop=torch.ones(1,50,3,dtype=torch.bool))
        with torch.no_grad():
            a = model(**batch)
            batch["tokens"] = tokens.clone()
            batch["tokens"][:,1] = 29999
            batch["audio"] *= 100
            b = model(**batch)
        for x,y in zip(a,b):
            self.assertTrue(torch.isfinite(x).all())
            self.assertTrue(torch.equal(x,y))


if __name__ == "__main__":
    unittest.main()
