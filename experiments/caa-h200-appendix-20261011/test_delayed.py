import unittest
class Timing(unittest.TestCase):
 def test_delayed_addition_zero_and_expiry(self):
  import torch
  from temporal_hook import hook_for
  x=torch.zeros(1,1,4,dtype=torch.bfloat16);v=torch.ones(4);a=[]
  h=hook_for([(v,4,3)],a)
  for expected in [0,0,1,1,0,0]:self.assertTrue(torch.equal(h(None,None,x),x+expected))
  self.assertEqual([r['single_token_forward'] for r in a],[3,4])
  z=hook_for([(v*0,4,3)],[])
  for _ in range(6):self.assertTrue(torch.equal(z(None,None,x),x))
if __name__=='__main__':unittest.main()
