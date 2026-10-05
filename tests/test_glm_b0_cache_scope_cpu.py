"""Actual namespace lookup and ADMIT decoder, without importing CUDA kernels."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

root=Path(__file__).parents[1]/'src/tensorfold/families/glm5_next/cuda'
spec=importlib.util.spec_from_file_location('scope_profile',root/'cache_profile.py')
p=importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name]=p;spec.loader.exec_module(p)
tree=ast.parse((root/'multi.py').read_text())
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MultiDecoder')
cls.body=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('_resume','_parse_admit')]
ns={'SHARED_SLOTS':4,'unpack_sampling':lambda x:None}
exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),str(root/'multi.py'),'exec'),ns)
D=ns['MultiDecoder']

class Scope(unittest.TestCase):
 def test_prefix_reuse_requires_same_scope(self):
  a,b=p.cache_scope('a'),p.cache_scope('b')
  c=SimpleNamespace(ids=[1,2],key=[1,2],drafter_end=2,drafter_rows=[],head=object(),cache_scope=a)
  d=D();d.kept=[c];d.drafts=None;d.grid=0
  self.assertIs(d._resume([1,2,3],a),c)
  self.assertIsNone(d._resume([1,2,3],b))
  self.assertIsNone(d._resume([1,2,3]))
 def test_wire_roundtrip_and_malformed_scope(self):
  # Fixed admission fields, sampling, shared stops, policy, prompt, grammar, images.
  prefix=[0]*9+[0]*10+[0]*4+[0]*4+[2,1,2]+[0]+[0,0]
  scope=p.cache_scope('request-uuid')
  self.assertEqual(D._parse_admit(prefix+[8,*scope])['cache_scope'],scope)
  self.assertEqual(D._parse_admit(prefix+[0])['cache_scope'],())
  with self.assertRaises(ValueError):D._parse_admit(prefix+[8,*scope[:-1]])
 def test_salts_are_opaque_and_validated(self):
  self.assertEqual(p.cache_scope('abc'),p.cache_scope('abc'))
  self.assertNotEqual(p.cache_scope(''),p.cache_scope(None))
  for x in [7,[], 'a'*257]:
   with self.assertRaises(ValueError):p.cache_scope(x)

if __name__=='__main__':unittest.main()
