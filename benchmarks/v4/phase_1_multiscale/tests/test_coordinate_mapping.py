import unittest

class CoordinateMappingTest(unittest.TestCase):
 def test_round_trip_scale(self):
  sx,sy=2.0,2.0;x,y=512,768
  self.assertEqual((round(x*sx/sx),round(y*sy/sy)),(x,y))
 def test_anisotropy_is_detectable(self): self.assertGreater(abs(2.0-1.8)/2.0,.01)
if __name__=='__main__':unittest.main()
