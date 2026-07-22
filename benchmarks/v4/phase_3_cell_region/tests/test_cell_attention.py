import unittest
import torch
from benchmarks.v4.phase_3_cell_region.src.model import CellToRegionAttention, sample_assignment_at_cells


class CellAttentionTest(unittest.TestCase):
    def test_attention_is_finite_and_respects_padding(self):
        m=CellToRegionAttention(cell_dim=4); regions=torch.randn(2,64,256); cells=torch.randn(2,5,4); mass=torch.rand(2,64,5); valid=torch.tensor([[1,1,0,0,0],[0,0,0,0,0]],dtype=torch.bool)
        out=m(regions,mass,cells,valid,torch.tensor([10.,0.]))
        self.assertTrue(torch.isfinite(out["fused_tokens"]).all())
        self.assertTrue(torch.allclose(out["attention"][0,:,2:],torch.zeros_like(out["attention"][0,:,2:])))
        self.assertFalse(out["has_cells"][1])
        self.assertTrue(torch.allclose(out["region_cell_count"][0].sum(),torch.tensor(10.),atol=1e-4))
        self.assertTrue(torch.all(out["region_cell_count"][1]==0))

    def test_coordinate_sampling_preserves_probability_mass(self):
        assignment=torch.rand(1,64,8,8).softmax(1); cells=torch.tensor([[[.5,.5],[.2,.8],[.001,.999]]])
        sampled=sample_assignment_at_cells(assignment,cells)
        self.assertTrue(torch.allclose(sampled.sum(1),torch.ones(1,3),atol=1e-5))
