import unittest
import numpy as np
import torch
from q2v3.analysis import cluster_weights, weighted_f1, boot_metrics
from q2.metrics import metrics
from q2v2.model import CompactBert

class ResearchTests(unittest.TestCase):
    def test_video_clusters_remain_together_and_paired(self):
        ids=np.array(['a$_$0','a$_$1','b$_$0','c$_$0'])
        a,names=cluster_weights(ids,37,8);b,_=cluster_weights(ids,37,8)
        self.assertTrue(np.array_equal(a,b));self.assertTrue(np.array_equal(a[:,0],a[:,1]))
        self.assertTrue(np.all(a[:,[0,2,3]].sum(1)==3))
    def test_bootstrap_metric_matches_literal_replication(self):
        labels=np.array([0,0,1,2]);classes=np.array([0,2,1,1]);w=np.array([[2,2,1,0],[0,0,0,3]])
        for i,row in enumerate(w):
            idx=np.repeat(np.arange(4),row)
            expected=metrics(labels[idx],np.arange(4)[idx],np.eye(3)[classes[idx]],np.arange(4)[idx])['macro_f1']
            self.assertAlmostEqual(weighted_f1(labels,classes,w)[i],expected,places=14)
    def test_identical_predictions_have_zero_paired_difference(self):
        a={'labels':np.array([0,1,2]),'classes':np.array([1,1,2]),'targets':np.array([-1.,0.,1.]),'predictions':np.array([0.,.1,1.])}
        w,_=cluster_weights(np.array(['a','b','c']),100,9)
        d=boot_metrics(a,w);self.assertTrue(np.array_equal(d['mae']-d['mae'],np.zeros(100)))
    def test_complete_pretrained_vocabulary_keeps_every_id(self):
        torch.set_num_threads(4)
        model=CompactBert(np.arange(30522),[])
        self.assertTrue(torch.equal(model.lookup,torch.arange(30522)))
        self.assertEqual(model.word.num_embeddings,30522)
    def test_accumulation_preserves_effective_batch_gradient(self):
        torch.manual_seed(5);layer=torch.nn.Linear(4,1)
        x=torch.randn(7,4);y=torch.randn(7,1)
        torch.nn.functional.mse_loss(layer(x),y).backward();reference=layer.weight.grad.clone();layer.zero_grad()
        for k in range(0,7,3):
            (torch.nn.functional.mse_loss(layer(x[k:k+3]),y[k:k+3])*(len(x[k:k+3])/7)).backward()
        self.assertTrue(torch.allclose(reference,layer.weight.grad,atol=1e-6,rtol=0))
if __name__=='__main__':unittest.main()
