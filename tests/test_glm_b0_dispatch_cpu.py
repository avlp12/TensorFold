"""Real CacheView and dsa_rows dispatch with CPU mathematical attention stand-in.

Tests routing/masks/lifecycle only. Actual reader CUDA numerics are a separate gate.
"""
import ast,contextlib,importlib.util
from pathlib import Path
import types,unittest,torch
spec=importlib.util.spec_from_file_location('alloc',Path(__file__).with_name('test_glm_b0_allocation_cpu.py'));f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
cache=__import__(f.PKG+'.b0_cache',fromlist=['CacheView'])
codec=__import__(f.PKG+'.b0_fp8',fromlist=['unpack_reference'])
def cpu_reader(self,extent,q,indices,lengths,scale):
 mapped,lens=self.selected(extent,indices,lengths);keys=codec.unpack_reference(self.storage[:self.rows]).double();out=torch.zeros_like(q)
 for i in range(q.shape[0]):
  selected=mapped[i,:int(lens[i])];selected=selected[selected>=0]
  if len(selected):
   k=keys[selected];out[i]=(torch.softmax(q[i].double()@k.T*scale,-1)@k).to(q.dtype)
 return types.SimpleNamespace(_values=out)
class Dispatch(unittest.TestCase):
 def setUp(self):
  self.c=f.Caches(f.weights(indexed=False),32,latent_kv='b0_fp8');self.c.b0_claim('request',8,16)
  self.p=self.c.b0_pools[0];self.p.attention=types.MethodType(cpu_reader,self.p)
  self.view=self.c.b0_view(0,8,16,8)
  self.x=torch.arange(4*512).reshape(4,512).remainder(37).to(torch.bfloat16)/8
  self.q=torch.ones(4,2,512,dtype=torch.bfloat16)/32
  self.pos=torch.tensor([0],dtype=torch.int32)
 def test_causal_dense_request_local(self):
  self.view.write(self.x,self.pos);out=torch.empty_like(self.q);self.view.dense(self.q,self.pos,out,1/16)
  keys=codec.unpack_reference(self.p.storage[8:12]).double()
  for i in range(4):
   k=keys[:i+1];expect=(torch.softmax(self.q[i].double()@k.T/16,-1)@k).to(torch.bfloat16)
   self.assertTrue(torch.equal(expect,out[i]))
  self.p.check()
 def test_sparse_keeps_dense_rows_and_rejects_future(self):
  self.view.write(self.x,self.pos);out=torch.full_like(self.q,7)
  tokens=torch.tensor([[-1,-1],[0,1],[0,2],[1,3]],dtype=torch.int32);counts=torch.tensor([0,2,2,2],dtype=torch.int32)
  self.view.sparse(self.q,self.pos,tokens,counts,out,1/16);self.assertTrue(bool((out[0]==7).all()));self.p.check()
  tokens[1,0]=3;self.view.sparse(self.q,self.pos,tokens,counts,out,1/16)
  with self.assertRaises(ValueError):self.p.check()
 def test_unwritten_dense_history_rejected(self):
  self.view.write(self.x[:1],torch.tensor([3],dtype=torch.int32))
  self.view.dense(self.q[:1],torch.tensor([3],dtype=torch.int32),torch.empty_like(self.q[:1]),1/16)
  with self.assertRaises(ValueError):self.p.check()
 def test_dense_range_overflow_and_empty_sparse_window_rejected(self):
  self.view.dense(self.q[:1],torch.tensor([8],dtype=torch.int32),torch.empty_like(self.q[:1]),1/16)
  with self.assertRaises(ValueError):self.p.check()
  c=f.Caches(f.weights(indexed=False),32,latent_kv='b0_fp8');c.b0_claim('x',0,16);p=c.b0_pools[0];p.attention=types.MethodType(cpu_reader,p);v=c.b0_view(0,0,16,8)
  v.sparse(self.q[:1],torch.tensor([8],dtype=torch.int32),torch.tensor([[-1]]),torch.tensor([0],dtype=torch.int32),torch.empty_like(self.q[:1]),1/16)
  with self.assertRaises(ValueError):p.check()
 def test_actual_dsa_rows_uses_b0_writer_and_dense_reader(self):
  tree=ast.parse((f.SRC/'families/glm5_next/cuda/forward.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='dsa_rows')
  def forbidden(*a,**k):raise AssertionError('legacy cache path invoked')
  latent=types.SimpleNamespace(latent_write=forbidden,absorb_q=lambda q,a,o,**kw:o.copy_(q),expand_v=lambda q,a,o,**kw:o.copy_(q),PROMPT_MMA=False)
  ns={'torch':torch,'latent':latent,'prof':types.SimpleNamespace(timed=lambda _:contextlib.nullcontext()),'l2pf':types.SimpleNamespace(site=lambda *a:None)}
  exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),fn],type_ignores=[])),'actual_dsa_rows','exec'),ns)
  scratch=types.SimpleNamespace(qa=torch.empty_like(self.q),ol=torch.empty_like(self.q),nch=1)
  b=types.SimpleNamespace(lat=self.x,q=self.q,lat_s=scratch,prefill=False,vn=torch.empty_like(self.q))
  cfg=types.SimpleNamespace(dense_limit=8,qk_dim=256,v_dim=512)
  result=ns['dsa_rows'](types.SimpleNamespace(heads=2,absorb=object()),types.SimpleNamespace(cfg=cfg),self.p.storage[8:24],self.pos,b,4,1,None,0,b0=self.view)
  expected=torch.empty_like(self.q);self.view.dense(self.q,self.pos,expected,1/16)
  self.assertTrue(torch.equal(result,expected.view(4,1024)));self.p.check()
 def test_actual_index_front_reaches_b0_ring_and_selection(self):
  from unittest.mock import patch
  idx=__import__(f.PKG+'.b0_indexer',fromlist=['IndexerView'])
  tree=ast.parse((f.SRC/'families/glm5_next/cuda/forward.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='dsa_rows')
  raw=torch.arange(4*128).reshape(4,128).remainder(17).to(torch.bfloat16);gates=raw.float()/3;headweights=torch.ones(4,32,dtype=torch.bfloat16);query=torch.ones(4,32,128,dtype=torch.bfloat16)
  index=(torch.zeros(8,128,dtype=torch.bfloat16),torch.zeros(8,128,dtype=torch.bfloat16),torch.zeros(6,144,dtype=torch.uint8));called=[]
  def comp(k,g,ape,e):called.append(('compress',k.shape,g.dtype));return torch.ones(len(k),132,dtype=torch.uint8)
  def choose(q,w,pk,pos,e,**kw):
   called.append(('select',q.shape,pk.shape,kw));cols=torch.arange(4,dtype=torch.int32);tokens=torch.where(cols[None,:]<=pos[:,None],cols[None,:],-1);return tokens,(tokens>=0).sum(-1).to(torch.int32)
  def mm(b,x,weight,scales,out):out.copy_(torch.cat((raw,headweights),1) if weight=='kw' else query.reshape(4,-1))
  def forbidden(*a,**k):raise AssertionError('legacy index path used')
  latent=types.SimpleNamespace(latent_write=forbidden,absorb_q=lambda q,a,o,**kw:o.copy_(q),expand_v=lambda q,a,o,**kw:o.copy_(q),PROMPT_MMA=False)
  ns={'__package__':f.PKG,'torch':torch,'mm':mm,'latent':latent,'glue':types.SimpleNamespace(router=lambda x,g,o:o.copy_(gates)),'prof':types.SimpleNamespace(timed=lambda _:contextlib.nullcontext()),'l2pf':types.SimpleNamespace(site=lambda *a:None),'sparse':types.SimpleNamespace(index_update=forbidden,select_tokens=forbidden)}
  exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),fn],type_ignores=[])),'actual_dsa_rows_index','exec'),ns)
  scratch=types.SimpleNamespace(qa=torch.empty_like(self.q),ol=torch.empty_like(self.q),nch=1)
  b=types.SimpleNamespace(lat=self.x,q=self.q,lat_s=scratch,prefill=True,vn=torch.empty_like(self.q),normed=torch.zeros(4,1),xs=torch.zeros(4,1),ikr=torch.empty(4,160,dtype=torch.bfloat16),igr=torch.empty(4,128),qr=torch.zeros(4,1),xs_qr=torch.zeros(4,1),qi=torch.empty(4,4096,dtype=torch.bfloat16))
  ix=types.SimpleNamespace(kw='kw',qb='qb',gate=None,ln_w=torch.ones(128),ln_b=torch.zeros(128),ape=torch.zeros(4,128));cfg=types.SimpleNamespace(dense_limit=8,qk_dim=256,v_dim=512,index_dim=128)
  with patch.object(idx,'compress',comp),patch.object(idx,'select',choose):
   result=ns['dsa_rows'](types.SimpleNamespace(heads=2,absorb=object(),index=ix),types.SimpleNamespace(cfg=cfg),self.p.storage[8:24],self.pos,b,4,1,index,0,b0=self.view)
  self.assertEqual([c[0] for c in called],['compress','select']);self.assertEqual(called[0][2],torch.bfloat16);self.assertEqual(called[1][1],(4,32,128));self.assertTrue(called[1][3]['short_prefill'])
  expected=torch.empty_like(self.q);self.view.dense(self.q,self.pos,expected,1/16);self.assertTrue(torch.equal(result,expected.view(4,1024)))

if __name__=='__main__':unittest.main(verbosity=2)
