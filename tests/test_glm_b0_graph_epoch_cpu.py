"""Real graph selection methods, host-only; this does not execute CUDA graphs."""
import ast
from pathlib import Path
import types
import unittest
ROOT=Path(__file__).resolve().parents[1]/'src/tensorfold/families/glm5_next/cuda'
def method(file,cls,name):
 tree=ast.parse((ROOT/file).read_text());c=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name==cls)
 f=next(x for x in c.body if isinstance(x,ast.FunctionDef) and x.name==name);f.decorator_list=[]
 ns={};exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),f],type_ignores=[])),file,'exec'),ns);return ns[name]
graphed=method('decode.py','Engine','graphed')
# Resolve the actual verify class name without assuming an old handoff name.
tree=ast.parse((ROOT/'verify.py').read_text());cls=next(x.name for x in tree.body if isinstance(x,ast.ClassDef) and any(isinstance(f,ast.FunctionDef) and f.name=='graph_for' for f in x.body))
graph_for=method('verify.py',cls,'graph_for')
begin_capture=method('verify.py',cls,'_begin_capture')
class Epoch(unittest.TestCase):
 def test_single_request_epoch_and_pool_identity(self):
  caches=types.SimpleNamespace(b0_epoch=0);slots=object();st=types.SimpleNamespace(caches=caches,slots=slots,graph_key=(0,0))
  e=types.SimpleNamespace(caches=caches,slots=slots,st=st,home=st,graphs=types.SimpleNamespace(cache_epoch=0))
  self.assertTrue(graphed(e));caches.b0_epoch=1;self.assertFalse(graphed(e))
  e.graphs.cache_epoch=1;self.assertTrue(graphed(e))
  foreign=types.SimpleNamespace(caches=object(),slots=slots,graph_key=(0,0));self.assertFalse(graphed(e,foreign))
 def test_batched_graph_rejected_after_each_lifecycle_change(self):
  caches=types.SimpleNamespace(b0_epoch=0);g=object();v=types.SimpleNamespace(e=types.SimpleNamespace(caches=caches),cache_epoch=0,graphs={8:g})
  for epoch in range(1,6):
   caches.b0_epoch=epoch;self.assertIsNone(graph_for(v,8))
   v.cache_epoch=epoch;self.assertIs(graph_for(v,8),g)
  self.assertIsNone(graph_for(v,9))
 def test_partial_recapture_drops_old_shapes(self):
  caches=types.SimpleNamespace(b0_epoch=1);old=object();v=types.SimpleNamespace(e=types.SimpleNamespace(caches=caches),cache_epoch=0,graphs={4:old,8:old})
  begin_capture(v);v.graphs[4]=object()
  self.assertIsNone(graph_for(v,8));self.assertIs(graph_for(v,4),v.graphs[4])
  begin_capture(v);self.assertIn(4,v.graphs)
 def test_uncaptured_graph_never_selected(self):
  v=types.SimpleNamespace(e=types.SimpleNamespace(caches=types.SimpleNamespace(b0_epoch=0)),cache_epoch=None,graphs={8:object()})
  self.assertIsNone(graph_for(v,8))
if __name__=='__main__':unittest.main(verbosity=2)
