import unittest,numpy as np,torch
from benchmarks.v4.phase_1_multiscale.src.metrics import confusion_matrix,summarize,soft_dice_loss
class MetricTest(unittest.TestCase):
 def test_confusion(self):
  c=confusion_matrix(np.array([[0,1,255]]),np.array([[0,0,1]]),2);self.assertEqual(c.tolist(),[[1,0],[1,0]]);self.assertAlmostEqual(summarize(c)['per_class_dice'][0],2/3)
 def test_ignore_dice(self):
  logits=torch.tensor([[[[10.]],[[0.]]]]);self.assertEqual(float(soft_dice_loss(logits,torch.tensor([[[255]]]),2,255,set())),0.)
if __name__=='__main__':unittest.main()
