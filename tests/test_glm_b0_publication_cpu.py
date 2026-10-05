"""CPU all-rank error protocol and real Engine.sample entry boundary; not NCCL proof."""
import ast,importlib.util
from pathlib import Path
import types,unittest,torch
spec=importlib.util.spec_from_file_location('b0alloc',Path(__file__).with_name('test_glm_b0_allocation_cpu.py'));f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
class Comm:
 def __init__(self, flags):self.flags=flags;self.calls=[]
 def all_gather(self, local, out):self.calls.append(local.clone());out.copy_(torch.tensor(self.flags,dtype=torch.int32))
class Publication(unittest.TestCase):
 def caches(self):return f.Caches(f.weights(indexed=False),16,latent_kv='b0_fp8')
 def test_all_four_clean_pass(self):
  for rank in range(4):
   c=self.caches();comm=Comm([0,0,0,0]);c.check_b0(4,comm)
   self.assertEqual(comm.calls[0].tolist(),[0])
 def test_remote_failure_rejected_on_every_rank_and_sticky(self):
  for rank in range(4):
   c=self.caches();c.b0_pools[0].error.fill_(rank==2);comm=Comm([0,0,1,0])
   with self.assertRaisesRegex(ValueError,'ranks \\[2\\]'):c.check_b0(4,comm)
   self.assertEqual(comm.calls[0].tolist(),[int(rank==2)])
   self.assertTrue(all(bool(p.error) for p in c.b0_pools))
 def test_missing_communicator_rejected(self):
  with self.assertRaises(ValueError):self.caches().check_b0(4,None)
 def test_default_cache_does_not_add_collective(self):
  c=f.Caches(f.weights(indexed=False),16);comm=Comm([1,1,1,1]);c.check_b0(4,comm);self.assertEqual(comm.calls,[])
 def test_serial_sample_stops_before_sampler_or_constraints(self):
  tree=ast.parse((f.SRC/'families/glm5_next/cuda/decode.py').read_text());cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='Engine');fn=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='sample')
  calls=[];ns={'sample_rows':lambda *args:calls.append('sample')};exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),fn],type_ignores=[])),'actual_sample','exec'),ns)
  e=types.SimpleNamespace(caches=self.caches(),w=types.SimpleNamespace(world=4,comm=Comm([0,1,0,0])),constraint=types.SimpleNamespace(mask=lambda *args:calls.append('mask')),window=object())
  with self.assertRaises(ValueError):ns['sample'](e,None,[0],None)
  self.assertEqual(calls,[])
if __name__=='__main__':unittest.main(verbosity=2)
