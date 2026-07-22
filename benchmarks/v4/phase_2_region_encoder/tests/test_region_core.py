import unittest
import numpy as np
import torch

from benchmarks.v4.phase_2_region_encoder.src.losses import RegionizationLoss, assignment_boundary
from benchmarks.v4.phase_2_region_encoder.src.metrics import region_metrics
from benchmarks.v4.phase_2_region_encoder.src.dataset import canonicalize_slic
from benchmarks.v4.phase_2_region_encoder.train_phase2 import stratified_ids


class RegionCoreTests(unittest.TestCase):
    def test_assignment_boundary_shape(self):
        logits=torch.randn(2,4,16,16,requires_grad=True); a=torch.softmax(logits,1)
        edge=assignment_boundary(a)
        self.assertEqual(tuple(edge.shape),(2,16,16)); self.assertTrue(torch.all((edge>0)&(edge<1)))
        edge.mean().backward(); self.assertIsNotNone(logits.grad)

    def test_loss_is_finite(self):
        b,k,c,h,w=2,4,3,16,16; logits=torch.randn(b,k,h,w,requires_grad=True); a=torch.softmax(logits,1)
        out={"assignment":a,"assignment_low":a,"semantic_logits":torch.randn(b,c,h,w)}; target=torch.randint(0,c,(b,h,w)); slic=torch.randint(0,k,(b,h,w))
        loss,parts=RegionizationLoss(c,255,{2},{"slic":1,"boundary":1,"balance":.25,"compact":.1,"purity":.25,"semantic_ce":.5,"semantic_dice":1})(out,target,slic,1.)
        self.assertTrue(torch.isfinite(loss)); self.assertGreater(parts["loss"],0); loss.backward(); self.assertTrue(torch.isfinite(logits.grad).all())

    def test_perfect_hard_regions(self):
        target=np.array([[0,0,1,1],[0,0,1,1],[2,2,2,2],[2,2,2,2]])
        a=np.zeros((3,4,4)); a[0,:2,:2]=1; a[1,:2,2:]=1; a[2,2:]=1
        result=region_metrics(a,target,255,3)
        self.assertAlmostEqual(result["region_purity"],1.); self.assertAlmostEqual(result["oracle_region_dice"],1.)

    def test_balance_penalizes_slot_collapse(self):
        b,k,c,h,w=1,4,3,8,8; target=torch.randint(0,c,(b,h,w)); slic=torch.arange(h*w).reshape(b,h,w)%k; semantic=torch.randn(b,c,h,w)
        uniform=torch.full((b,k,h,w),1/k); collapsed=torch.zeros_like(uniform); collapsed[:,0]=1
        fn=RegionizationLoss(c,255,{2},{"slic":1,"boundary":1,"balance":.25,"compact":.1,"purity":.25,"semantic_ce":.5,"semantic_dice":1})
        _,uniform_parts=fn({"assignment_low":uniform,"semantic_logits":semantic},target,slic,1.)
        _,collapsed_parts=fn({"assignment_low":collapsed,"semantic_logits":semantic},target,slic,1.)
        self.assertLess(uniform_parts["balance"],1e-6); self.assertGreater(collapsed_parts["balance"],1.0)

    def test_slic_labels_are_canonical_after_flip(self):
        labels=np.array([[9,9,3,3],[9,9,3,3],[7,7,5,5],[7,7,5,5]],dtype=np.uint8)
        canonical=canonicalize_slic(labels)
        self.assertEqual(len(np.unique(canonical)),4)
        self.assertLess(int(canonical[0,0]),int(canonical[0,-1])); self.assertLess(int(canonical[0,0]),int(canonical[-1,0]))
        flipped=canonicalize_slic(np.fliplr(labels))
        self.assertLess(int(flipped[0,0]),int(flipped[0,-1])); self.assertLess(int(flipped[0,0]),int(flipped[-1,0]))

    def test_stratified_ids_keeps_each_group(self):
        rows=[{"sampling_group":"a"} for _ in range(4)]+[{"sampling_group":"b"} for _ in range(4)]
        chosen=stratified_ids(rows,4,7)
        self.assertEqual(len(chosen),4); self.assertEqual(len(set(chosen)),4)
        self.assertEqual({rows[i]["sampling_group"] for i in chosen},{"a","b"})


if __name__ == "__main__": unittest.main()
