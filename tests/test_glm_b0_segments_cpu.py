"""B0 segmented routing and request isolation; CPU mathematical reader stand-in."""
import importlib.util
from pathlib import Path
import types,unittest,torch
spec=importlib.util.spec_from_file_location('dispatch_fixture',Path(__file__).with_name('test_glm_b0_dispatch_cpu.py'));d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
class Segments(unittest.TestCase):
 def setup_case(self):
  c=d.f.Caches(d.f.weights(indexed=False),48,latent_kv='b0_fp8');c.b0_claim('a',0,16);c.b0_claim('b',16,16)
  p=c.b0_pools[0];p.attention=types.MethodType(d.cpu_reader,p)
  views=[c.b0_view(0,base,16,8) for base in (0,16)]
  rows=types.SimpleNamespace(segments=[(0,0,3),(16,0,2)],pos=torch.tensor([0,1,2,0,1],dtype=torch.int32),base=torch.tensor([0,0,0,16,16],dtype=torch.int32))
  x=torch.arange(5*512).reshape(5,512).remainder(31).to(torch.bfloat16);q=torch.ones(5,2,512,dtype=torch.bfloat16)/32
  return c,p,views,rows,x,q
 def test_segmented_equals_independent_requests(self):
  c,p,views,rows,x,q=self.setup_case();d.cache.segmented_write(views,rows,x);out=torch.empty_like(q)
  d.cache.segmented_attention(views,rows,q,None,None,out,1/16)
  at=0
  for view,(_,_,n) in zip(views,rows.segments):
   expected=torch.empty_like(q[at:at+n]);view.dense(q[at:at+n],torch.tensor([0],dtype=torch.int32),expected,1/16)
   self.assertTrue(torch.equal(expected,out[at:at+n]));at+=n
  p.check()
 def test_segmented_sparse_preserves_other_request(self):
  c,p,views,rows,x,q=self.setup_case();d.cache.segmented_write(views,rows,x);out=torch.empty_like(q)
  tokens=torch.tensor([[-1,-1],[0,1],[0,2],[-1,-1],[0,1]],dtype=torch.int32);counts=torch.tensor([0,2,2,0,2],dtype=torch.int32)
  d.cache.segmented_attention(views,rows,q,tokens,counts,out,1/16)
  keys=d.codec.unpack_reference(p.storage[:p.rows]).double()
  k=keys[[16,17]];expected=(torch.softmax(q[4].double()@k.T/16,-1)@k).to(torch.bfloat16)
  self.assertTrue(torch.equal(expected,out[4]));p.check()
 def test_bad_host_segment_rejected_before_write(self):
  c,p,views,rows,x,q=self.setup_case();rows.segments[1]=(32,0,2)
  with self.assertRaises(ValueError):d.cache.segmented_write(views,rows,x)
  self.assertFalse(bool(p.valid.any()))
 def test_bad_device_table_poisoned_before_write(self):
  for field in ['base','pos']:
   c,p,views,rows,x,q=self.setup_case();getattr(rows,field)[3]=33
   with self.assertRaises(ValueError):d.cache.segmented_write(views,rows,x)
   self.assertFalse(bool(p.valid.any()))
   with self.assertRaises(ValueError):p.check()
 def test_stale_binding_rejected(self):
  c,p,views,rows,x,q=self.setup_case();c.b0_release('b')
  with self.assertRaises(ValueError):d.cache.segmented_write(views,rows,x)
  self.assertFalse(bool(p.valid.any()))
if __name__=='__main__':unittest.main(verbosity=2)
