"""CPU-only regression gates for the real Flash rank reader/request methods."""
import ast
import importlib.util
import json
import struct
import sys
import tempfile
import types
import unittest
from pathlib import Path
import pytest
np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "src/tensorfold/families/glm5_next/cuda"

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj

split = module("flash14_split", BASE / "split.py")
engine = module("flash14_engine", BASE / "engine.py")

class Tensor:
    def __init__(self, value): self.a = np.asarray(value)
    def numel(self): return self.a.size
    def clone(self): return Tensor(self.a.copy())
    def view(self, *shape): return Tensor(self.a.reshape(shape))
    def reshape(self, *shape): return self.view(*shape)
    def contiguous(self): return self.clone()
    def __getitem__(self, key): return Tensor(self.a[key])
    def tolist(self): return self.a.tolist()
    def item(self): return self.a.item()
    def copy_(self, other): self.a[...] = other.a

class Torch:
    int32 = np.int32
    float32 = np.float32
    def tensor(self, value, dtype=None, device=None): return Tensor(np.asarray(value, dtype=dtype))
    def empty(self, shape, dtype=None, device=None): return Tensor(np.empty(shape, dtype=dtype))
    def zeros(self, shape, dtype=None, device=None): return Tensor(np.zeros(shape, dtype=dtype))

class Store:
    def __init__(self): self.keys = {}
    def set(self, key, value): self.keys[key] = value
    def wait(self, keys, timeout=None):
        for key in keys:
            if key not in self.keys: raise AssertionError(f"missing doorbell {key}")
    def delete_key(self, key): del self.keys[key]

class RankGates(unittest.TestCase):
    def test_host_and_device_splits_reconstruct_original(self):
        for world in (2, 4):
            for kind, shape, axis in (("row", (8, 16), 0), ("col", (8, 16), 1), ("dim1", (4, 8, 3), 1)):
                array = np.arange(np.prod(shape), dtype=np.uint32).reshape(shape)
                raw = array.view(np.uint8).reshape(-1)
                parts = []
                for rank in range(world):
                    host, got_shape = split.split_bytes(raw, list(shape), 4, kind, rank, world)
                    device, device_shape = split.split_device(Tensor(raw), list(shape), 4, kind, rank, world)
                    self.assertEqual(got_shape, device_shape)
                    np.testing.assert_array_equal(host, device.a)
                    parts.append(host.view(np.uint32).reshape(got_shape))
                np.testing.assert_array_equal(np.concatenate(parts, axis=axis), array)
            raw = np.arange(32, dtype=np.uint8)
            for rank in range(world):
                part, shape = split.split_bytes(raw, [16], 2, "rep", rank, world)
                np.testing.assert_array_equal(part, raw)
                self.assertEqual(shape, [16])

    def test_malformed_topologies_and_dimensions_refused(self):
        for world, rank in ((1,0),(3,0),(5,0),(4,-1),(4,4),(2,2)):
            with self.assertRaises(ValueError): split.validate_topology(rank, world)
        for kind, shape in (("row", [6,8]), ("col", [8,6]), ("dim1", [2,6,4])):
            raw = np.zeros(np.prod(shape)*2, dtype=np.uint8)
            with self.assertRaises(ValueError): split.split_bytes(raw, shape, 2, kind, 0, 4)
            with self.assertRaises(ValueError): split.split_device(Tensor(raw), shape, 2, kind, 0, 4)
            split.split_bytes(raw, shape, 2, kind, 1, 2)

    def test_reader_selects_only_its_quarter_rows(self):
        name = "model.language_model.layers.0.mlp.gate_proj.weight"
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "model-00001.safetensors"
            header = json.dumps({name:{"dtype":"BF16", "shape":[8,16], "data_offsets":[0,256]}}).encode()
            path.write_bytes(struct.pack("<Q",len(header)) + header + bytes(256))
            for rank in range(4):
                rd = split.RankReader.__new__(split.RankReader)
                rd.dir, rd.rank, rd.world = Path(td), rank, 4
                rd.split, rd.files, rd.index = False, {}, {name:path.name}
                _, begin, end, kind, shape, dtype = rd._span(name)
                self.assertEqual((begin,end),(8+len(header)+rank*64,8+len(header)+(rank+1)*64))
                self.assertEqual((kind,shape,dtype),("rep",[2,16],"BF16"))

    def test_presplit_tp2_refused_before_io_for_tp4(self):
        names = ("tensorfold", "tensorfold.cuda", "tensorfold.cuda.direct_read")
        saved = {n:sys.modules.get(n) for n in names}
        for n in names: sys.modules[n] = types.ModuleType(n)
        dr = sys.modules[names[-1]]
        dr.ReadAhead = dr.Reader = dr.SafeTensors = lambda *a: (_ for _ in ()).throw(AssertionError("I/O before refusal"))
        try:
            with tempfile.TemporaryDirectory() as td:
                (Path(td)/"model.rank0.safetensors").touch()
                with self.assertRaisesRegex(ValueError,"full checkpoint"):
                    split.RankReader(td,3,4)
        finally:
            for n,v in saved.items():
                if v is None: sys.modules.pop(n,None)
                else: sys.modules[n]=v

    def test_all_four_metadata_gathered_and_header_shared(self):
        for world in (2,4):
            values = [[r, 10+r] for r in range(world)]
            obj = engine.GlmEngine.__new__(engine.GlmEngine)
            obj.torch, obj.world, obj.rank = Torch(), world, 0
            def gather(send, recv):
                self.assertEqual(recv.numel(),world*send.numel())
                recv.a[:] = np.asarray(values).reshape(-1)
            obj.comm = types.SimpleNamespace(all_gather=gather)
            self.assertEqual(obj._gather_ints(values[0]),values)
            self.assertEqual(obj._gather_floats(values[0]),values)
            header=[41,42,43]
            calls=[]
            def share(send,recv):
                self.assertEqual(recv.numel(),world*send.numel())
                block=[len(header)] if not calls else header
                recv.a[:]=np.concatenate([np.asarray(block)] + [np.zeros(len(block)) for _ in range(1,world)])
                calls.append(1)
            obj.comm=types.SimpleNamespace(all_gather=share)
            self.assertEqual(obj._share(header),header)
            self.assertEqual(len(calls),2)

    def test_independent_worker_bells_do_not_steal_each_other(self):
        for world in (2,4):
            store=Store()
            peers=[]
            for rank in range(world):
                obj=engine.GlmEngine.__new__(engine.GlmEngine)
                obj.comm=types.SimpleNamespace(store=store)
                obj.rank,obj.world=rank,world
                peers.append(obj)
            for _ in range(3):
                peers[0]._ring()
                self.assertEqual(len(store.keys),world-1)
                for obj in reversed(peers[1:]): obj._await_bell()
                self.assertEqual(store.keys,{})
            self.assertEqual([p._bell for p in peers],[3]*world)

    def test_every_rank_settings_are_checked(self):
        tree=ast.parse((BASE/"engine.py").read_text())
        condition=next(n.test for n in ast.walk(tree) if isinstance(n,ast.If) and
                       isinstance(n.test,ast.Call) and isinstance(n.test.func,ast.Name)
                       and n.test.func.id == "any" and "both[0][:-1]" in ast.unparse(n.test))
        code=compile(ast.Expression(condition),"actual-settings-condition","eval")
        same=[[1,2,1024]]*4
        self.assertFalse(eval(code,{"both":same}))
        for r in range(1,4):
            rows=[list(x) for x in same]; rows[r][0]=99
            self.assertTrue(eval(code,{"both":rows}))

if __name__ == "__main__": unittest.main()
