"""CPU/meta verification of real Caches/Arena allocation and geometry functions.

Forward's Caches/State classes are loaded from their AST to avoid importing CUDA
kernels; allocation, view and copy bodies are unmodified. No weights/GPU/Metal.
"""
import ast
import importlib.util
from pathlib import Path
import sys
import types
import unittest

import torch

torch.set_default_device('cpu')
torch.set_num_threads(2)
SRC = Path(__file__).resolve().parents[1] / 'src/tensorfold'
PKG = '_b0_allocation'
for name, path in [(PKG, SRC / 'families/glm5_next/cuda'), (PKG + '.cuda', SRC / 'cuda')]:
    p = types.ModuleType(name); p.__path__ = [str(path)]; sys.modules[name] = p


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
    return m


layout = load(PKG + '.cuda.mla_layout', SRC / 'cuda/mla_layout.py')
geometry = load(PKG + '.cuda.geometry', SRC / 'cuda/geometry.py')


def extract(path, names, env):
    tree = ast.parse(path.read_text())
    body = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]
    body += [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(path), 'exec'), env)


kv = {'torch': torch, 'PAD': 16, 'KINDS': ('bf16', 'fp8')}
extract(SRC / 'families/glm5_next/cuda/kv8.py', {'row_bytes', 'zeros'}, kv)
ns = {'__package__': PKG, 'torch': torch, 'kv8': types.SimpleNamespace(**kv),
      'latent': types.SimpleNamespace(ENABLED=True), 'tp': types.SimpleNamespace(of=lambda w: types.SimpleNamespace(heads=4)),
      'latent_kind': layout.latent_kind, 'mla_layout_row_bytes': layout.row_bytes}
extract(SRC / 'families/glm5_next/cuda/forward.py', {'Caches', 'State'}, ns)
Caches, State = ns['Caches'], ns['State']
extract(SRC / 'families/glm5_next/cuda/decode.py', {'_row_views'}, ns)


def weights(mtp=False, indexed=True, device='cpu', width=512):
    cfg = types.SimpleNamespace(kv_lora=width, index_dim=128, qk_dim=256, v_dim=256)
    return types.SimpleNamespace(cfg=cfg, device=device, mtp=object() if mtp else None,
                                 layers=[types.SimpleNamespace(kind='dsa') for _ in range(2)],
                                 meta={'long_context': indexed})


def config(mtp=False):
    return dict(hidden_size=512, num_attention_heads=8, num_hidden_layers=4,
                layer_types=['linear_attention', 'full_attention'] * 2, linear_num_heads=8,
                qk_nope_head_dim=256, v_head_dim=256, vocab_size=1024, kv_lora_rank=512,
                moe_intermediate_size=512, num_experts_per_tok=2, num_nextn_predict_layers=int(mtp))


class Allocation(unittest.TestCase):
    def test_layout_bytes_and_wrong_width_rejection(self):
        self.assertEqual([layout.row_bytes(512, k) for k in layout.KINDS], [1024, 528, 656])
        for width in (128, 640, 0, -1, True):
            with self.subTest(width=width), self.assertRaises(ValueError):
                layout.row_bytes(width, 'b0_fp8')
        with self.assertRaises(ValueError):
            layout.latent_kind('b0_fp8', None, 512)

    def test_actual_planes_match_geometry_delta(self):
        for mtp in (False, True):
            for kv in ('bf16', 'fp8'):
                for ring in (None, 64):
                    for rows in (64, 2048):
                        with self.subTest(mtp=mtp, kv=kv, ring=ring, rows=rows):
                            a = Caches(weights(mtp), rows, streams=2, ring=ring, kv=kv)
                            b = Caches(weights(mtp), rows, streams=2, ring=ring, kv=kv, latent_kv='b0_fp8')
                            count = 2 + int(mtp)
                            want = count * (rows * (656 - layout.row_bytes(512, kv)) + 9 * rows + 666)
                            self.assertEqual(b.allocation_bytes() - a.allocation_bytes(), want)
                            ga = geometry.mla_geometry(config(mtp), 2, 8, latent=True, kv=kv)
                            gb = geometry.mla_geometry(config(mtp), 2, 8, latent=True, kv=kv, latent_kv='b0_fp8')
                            self.assertEqual(gb.bytes_at(rows) - ga.bytes_at(rows), want)
                            for i in b.kc:
                                self.assertEqual(b.arena.planes[i].tensor.shape, (rows, 656))
                            for old, new in zip(a.index, b.index):
                                self.assertEqual(a.arena.planes[old[2]].tensor.shape,
                                                 b.arena.planes[new[2]].tensor.shape)

    def test_meta_large_capacity_without_allocating_model_memory(self):
        c = Caches(weights(device='meta'), 850000, kv='fp8', latent_kv='b0_fp8', ring=64)
        self.assertEqual(c.arena.planes[c.kc[0]].tensor.numel(), 850000 * 656)
        self.assertEqual(c.arena.planes[c.index[0][2]].tensor.shape, (850000 // 4 + 2, 144))

    def test_arena_copy_and_bind_preserve_packed_bytes(self):
        c = Caches(weights(indexed=False), 32, latent_kv='b0_fp8')
        plane = c.arena.planes[c.kc[0]].tensor
        data = (torch.arange(8 * 656).reshape(8, 656) % 251).to(torch.uint8)
        plane[:8].copy_(data)
        c.arena.copy(0, 4, 8)  # overlapping relocation through the real Arena
        self.assertTrue(torch.equal(plane[4:12], data))
        st = types.SimpleNamespace(slot=0)
        c.bind(st, 4, 8)
        self.assertTrue(torch.equal(st.kc[0], data))
        self.assertEqual(st.kc[0].data_ptr(), plane[4:].data_ptr())

    def test_state_cannot_execute_unwired_b0_planes(self):
        c = Caches(weights(), 64, latent_kv='b0_fp8')
        with self.assertRaisesRegex(NotImplementedError, 'integrated DFlash2 path'):
            State(weights(), 64, 8, caches=c)

    def test_display_and_expanded_cache_rejected(self):
        with self.assertRaises(ValueError):
            Caches(weights(), 64, latent_kv='b0_fp8', display=1)
        before = ns['latent'].ENABLED
        ns['latent'].ENABLED = False
        try:
            with self.assertRaises(ValueError):
                Caches(weights(), 64, latent_kv='b0_fp8')
        finally:
            ns['latent'].ENABLED = before
        with self.assertRaises(ValueError):
            geometry.mla_geometry(config(), 2, 8, latent=False, latent_kv='b0_fp8')

    def test_wrong_latent_width_rejected_by_allocator_and_estimator(self):
        with self.assertRaises(ValueError):
            Caches(weights(width=128), 64, latent_kv='b0_fp8')
        with self.assertRaises(ValueError):
            geometry.mla_geometry(dict(config(), kv_lora_rank=128), 2, 8, latent=True, latent_kv='b0_fp8')

    def test_writer_uses_actual_arena_storage_and_exact_guard_accounting(self):
        c = Caches(weights(indexed=False), 32, latent_kv='b0_fp8')
        p = c.b0_pools[0]
        extent = p.reserve('a', 0, 8)
        value = torch.arange(512).reshape(1, 512).to(torch.bfloat16)
        p.write(extent, torch.tensor([3]), value)
        self.assertEqual(c.arena.planes[c.kc[0]].tensor.data_ptr(), p.storage.data_ptr())
        self.assertTrue(bool(c.arena.planes[c.b0_valid_planes[0]].tensor[3]))
        tensors = [t for pool in c.b0_pools for t in
                   (pool.storage, pool.valid, pool.generations, pool.error)]
        actual = sum(t.numel() * t.element_size() for t in tensors)
        self.assertEqual(c.allocation_bytes(), actual)
        self.assertEqual(actual, 2 * ((32 + 1) * (656 + 1 + 8) + 1))
        p.check()

    def test_arena_moves_validity_but_not_ownership_generation(self):
        c = Caches(weights(indexed=False), 32, latent_kv='b0_fp8')
        p = c.b0_pools[0]
        a = p.reserve('a', 0, 8)
        b = p.reserve('b', 8, 8)
        p.write(a, torch.tensor([0, 3]), torch.ones(2, 512, dtype=torch.bfloat16))
        c.arena.copy(0, 8, 8)
        self.assertEqual(p.valid[8:16].tolist(), p.valid[:8].tolist())
        self.assertEqual(p.generations[8:16].tolist(), [b.generation] * 8)
        mapped, _ = p.selected(b, torch.tensor([[0, 3]]), torch.tensor([2], dtype=torch.int32))
        self.assertEqual(mapped.tolist(), [[8, 11]])
        p.check()

    def test_snapshot_rows_preserve_validity_without_restoring_old_generation(self):
        c = Caches(weights(indexed=False), 32, latent_kv='b0_fp8')
        p = c.b0_pools[0]
        a = p.reserve('a', 0, 8)
        p.write(a, torch.tensor([0, 3]), torch.ones(2, 512, dtype=torch.bfloat16))
        st = types.SimpleNamespace(slot=0)
        c.bind(st, 0, 8)
        saved = [x.clone() for x in ns['_row_views'](st, 4, 0)]
        p.release(a)
        b = p.reserve('b', 8, 8)
        c.bind(st, 8, 8)
        for dst, src in zip(ns['_row_views'](st, 4, 0), saved, strict=True):
            dst.copy_(src)
        self.assertEqual(p.valid[8:12].tolist(), [True, False, False, True])
        self.assertEqual(p.generations[8:12].tolist(), [b.generation] * 4)
        p.selected(b, torch.tensor([[0, 3]]), torch.tensor([2], dtype=torch.int32))
        p.check()

    def test_snapshot_includes_mtp_validity_only_when_mtp_rows_requested(self):
        c = Caches(weights(mtp=True, indexed=False), 32, latent_kv='b0_fp8')
        st = types.SimpleNamespace(slot=0)
        c.bind(st, 0, 8)
        main = ns['_row_views'](st, 4, 0)
        both = ns['_row_views'](st, 4, 2)
        self.assertEqual(len(main), 4)  # two payload + two validity planes
        self.assertEqual(len(both), 6)
        self.assertEqual([x.dtype for x in both[-2:]], [torch.uint8, torch.bool])
        self.assertEqual([x.shape[0] for x in both[-2:]], [2, 2])

    def test_default_layouts_unchanged(self):
        for kv, dtype, width in [('bf16', torch.bfloat16, 512), ('fp8', torch.uint8, 528)]:
            c = Caches(weights(), 64, kv=kv)
            p = c.arena.planes[c.kc[0]].tensor
            self.assertEqual(p.dtype, dtype)
            self.assertEqual(p.shape, (64, width))
            self.assertEqual(c.latent_kv, kv)


if __name__ == '__main__':
    unittest.main(verbosity=2)
