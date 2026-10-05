"""Indexer adapter metadata/page-layout tests; no CUDA, weights or Triton."""
import ast,importlib.util
from pathlib import Path
import struct,types,unittest,torch
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]/'src/tensorfold/families/glm5_next/cuda'
spec=importlib.util.spec_from_file_location('b0index',ROOT/'b0_indexer.py');i=importlib.util.module_from_spec(spec);spec.loader.exec_module(i)
torch.set_default_device('cpu');torch.set_num_threads(2)
class Indexer(unittest.TestCase):
 def test_original_page_codes_then_scales(self):
  for page_size in [1,2,16,32]:
   codes=torch.arange(2*page_size*128).reshape(2,page_size,128).remainder(251).to(torch.uint8)
   scales=torch.arange(1,2*page_size+1,dtype=torch.float32).reshape(2,page_size)/8
   pages=torch.cat((codes.reshape(2,-1),scales.contiguous().view(torch.uint8).reshape(2,-1)),dim=1).reshape(2,page_size,132)
   expected=[]
   for p in range(2):
    for row in range(page_size):expected.append(torch.cat((codes[p,row],torch.tensor(list(struct.pack('<f',float(scales[p,row]))),dtype=torch.uint8))))
   self.assertTrue(torch.equal(i.page_rows(pages),torch.stack(expected)))
   if page_size>1:self.assertFalse(torch.equal(pages.reshape(-1,132),torch.stack(expected)))
 def test_page_metadata_rejected(self):
  for x in [torch.zeros(1,16,132),torch.zeros(1,16,128,dtype=torch.uint8),torch.zeros(0,16,132,dtype=torch.uint8),torch.zeros(1,32,132,dtype=torch.uint8)[:,::2]]:
   with self.assertRaises(ValueError):i.page_rows(x)
 def test_query_metadata_and_dtype_not_silently_converted(self):
  q=torch.zeros(2,32,128,dtype=torch.bfloat16);w=torch.zeros(2,32,dtype=torch.bfloat16);error=torch.zeros((),dtype=torch.bool)
  i.validate_query(q,w,error)
  for badq,badw in [(q.float(),w),(q[:,:16],w[:,:16]),(q,w.float()),(q,w[:1])]:
   with self.assertRaises(ValueError):i.validate_query(badq,badw,error)
 def test_pool_precision_contract(self):
  k=torch.zeros(3,4,128,dtype=torch.bfloat16);g=k.clone();ape=torch.zeros(4,128,dtype=torch.float32);error=torch.zeros((),dtype=torch.bool)
  i.validate_pools(k,g,ape,error)
  for bk,bg,ba in [(k.float(),g,ape),(k,g.float(),ape),(k,g,ape.bfloat16()),(k[:,:2],g[:,:2],ape[:2])]:
   with self.assertRaises(ValueError):i.validate_pools(bk,bg,ba,error)
 def test_selection_metadata_and_modes(self):
  q=torch.zeros(2,32,128,dtype=torch.bfloat16);w=torch.zeros(2,32,dtype=torch.bfloat16);e=torch.zeros((),dtype=torch.bool);pk=torch.zeros(513,132,dtype=torch.uint8);pos=torch.tensor([2048,2051],dtype=torch.int64)
  i.validate_selection(q,w,pk,pos,e,True,False)
  i.validate_selection(q,w,pk[:0],pos,e,True,True)
  for args in [(pk.float(),pos,True,False),(pk,pos.float(),True,False),(pk,pos,False,True),(pk,pos,1,False)]:
   with self.assertRaises(ValueError):i.validate_selection(q,w,args[0],args[1],e,args[2],args[3])
 def test_ring_update_across_chunks_and_selection_views(self):
  keys=torch.zeros(8,128,dtype=torch.bfloat16);gates=keys.clone();pools=torch.full((6,144),91,dtype=torch.uint8);error=torch.zeros((),dtype=torch.bool)
  v=i.IndexerView((keys,gates,pools),16,error);raw=torch.arange(8*128).reshape(8,128).remainder(31).to(torch.bfloat16);gate=raw.float()/4+0.001
  weight=torch.ones(128);bias=torch.zeros(128);ape=torch.zeros(4,128);calls=[]
  def compress(k,g,a,e):
   calls.append((k.clone(),g.clone(),a.clone()));return torch.full((len(k),132),7,dtype=torch.uint8)
  with patch.object(i,'compress',compress):
   v.update(raw[:3],gate[:3],weight,bias,ape,0);self.assertEqual(calls,[])
   v.update(raw[3:],gate[3:],weight,bias,ape,3)
  self.assertEqual(len(calls),1)
  expected=i._key_norm(raw,weight,bias,128,1e-6).reshape(2,4,128)
  self.assertTrue(torch.equal(calls[0][0],expected));self.assertTrue(torch.equal(calls[0][1],gate.bfloat16().reshape(2,4,128)))
  self.assertTrue(bool((pools[:2,:132]==7).all()));self.assertTrue(bool((pools[:,132:]==91).all()))
  captured={}
  def choose(q,w,pk,pos,e,**kw):captured.update(shape=q.shape,pk=pk.clone(),pos=pos.tolist(),flags=kw);return None,None
  with patch.object(i,'select',choose):v.choose(torch.zeros(2,4096,dtype=torch.bfloat16),torch.zeros(2,32,dtype=torch.bfloat16),6,prefill=True)
  self.assertEqual(captured['shape'],(2,32,128));self.assertEqual(captured['pk'].shape,(2,132));self.assertEqual(captured['pos'],[6,7]);self.assertTrue(captured['flags']['short_prefill'])
 def test_ring_overwrite_rejected_before_mutation(self):
  keys=torch.zeros(8,128,dtype=torch.bfloat16);gates=keys.clone();pools=torch.zeros(6,144,dtype=torch.uint8);e=torch.zeros((),dtype=torch.bool);v=i.IndexerView((keys,gates,pools),16,e)
  with self.assertRaises(ValueError):v.update(torch.ones(6,128,dtype=torch.bfloat16),torch.ones(6,128,dtype=torch.bfloat16),torch.ones(128),torch.zeros(128),torch.zeros(4,128),3)
  self.assertFalse(bool(keys.any()));self.assertFalse(bool(pools.any()))
 def test_actual_loader_ape_preserves_f32_in_b0_mode(self):
  tree=ast.parse((ROOT/'weights.py').read_text());load=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='load')
  tf=next(x for x in load.body if isinstance(x,ast.FunctionDef) and x.name=='t');dsa=next(x for x in load.body if isinstance(x,ast.FunctionDef) and x.name=='dsa');ix=next(x for x in dsa.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ix' for t in x.targets))
  raw=torch.tensor([[1.0001,-0.00012345]],dtype=torch.float32)
  for mode,dtype in [('vendor',torch.bfloat16),('b0',torch.float32)]:
   ns={'torch':torch,'rd':types.SimpleNamespace(get=lambda _:raw),'PREFIX':'model.','dev':torch.device('cpu'),'index_ape_dtype':i.ape_dtype(mode),'p':'layers.0.self_attn.','IndexW':lambda *a:a,'stack':lambda _:None,'q4':lambda _:None}
   exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),tf,ix],type_ignores=[])),'actual_loader_ape','exec'),ns)
   actual=ns['ix'][-1];self.assertEqual(actual.dtype,dtype)
   if mode=='b0':self.assertTrue(torch.equal(actual,raw))
   else:self.assertFalse(torch.equal(actual.float(),raw))
  with self.assertRaises(ValueError):i.ape_dtype('unverified')
if __name__=='__main__':unittest.main(verbosity=2)
