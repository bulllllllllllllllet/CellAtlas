import unittest
import numpy as np

from benchmarks.v4.phase_3_cell_region.tools.build_cell_patch_cache import extract_cell_features


def reference(inst, cls):
    cells=[]
    for cell_id in np.unique(inst):
        if cell_id==0: continue
        pixels=inst==cell_id; yy,xx=np.nonzero(pixels); labels=cls[pixels]; foreground=labels[labels>0]
        cell_class=int(np.bincount(foreground,minlength=7).argmax()) if len(foreground) else 0
        cells.append([xx.mean()/inst.shape[1],yy.mean()/inst.shape[0],np.log1p(len(xx)),cell_class])
    return np.asarray(cells,dtype=np.float32).reshape(-1,4)


class CellCacheTest(unittest.TestCase):
    def test_vectorized_matches_reference(self):
        rng=np.random.default_rng(7); inst=rng.integers(0,12,size=(37,41),dtype=np.int32); cls=rng.integers(0,7,size=inst.shape,dtype=np.uint8)
        np.testing.assert_allclose(extract_cell_features(inst,cls),reference(inst,cls),rtol=0,atol=1e-6)

    def test_empty(self):
        out=extract_cell_features(np.zeros((8,9),np.int32),np.zeros((8,9),np.uint8)); self.assertEqual(out.shape,(0,4))
