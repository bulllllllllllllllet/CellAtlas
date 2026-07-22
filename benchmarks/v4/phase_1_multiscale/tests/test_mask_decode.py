import tempfile,unittest
from pathlib import Path
import numpy as np
from PIL import Image
from benchmarks.v4.phase_1_multiscale.src.data import decode_gt, decode_gt_patch
class DecodeTest(unittest.TestCase):
 def test_unknown_is_ignore(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'a.png';Image.fromarray(np.array([[[1,2,3],[4,5,6]]],dtype=np.uint8)).save(p)
   self.assertEqual(decode_gt(p,[{'id':0,'rgb':[1,2,3]}],255).tolist(),[[0,255]])
 def test_patch_decode(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'a.png';Image.fromarray(np.array([[[1,2,3],[4,5,6]],[[4,5,6],[1,2,3]]],dtype=np.uint8)).save(p)
   self.assertEqual(decode_gt_patch(p,1,0,1,2,[{'id':0,'rgb':[1,2,3]}],255).tolist(),[[255],[0]])
if __name__=='__main__':unittest.main()
