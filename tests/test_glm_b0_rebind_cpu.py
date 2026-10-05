"""Actual cache and MultiDecoder lifecycle methods on CPU, no model imports."""
import ast
import importlib.util
from pathlib import Path
import types
import unittest
import torch

spec = importlib.util.spec_from_file_location('b0_alloc_fixture', Path(__file__).with_name('test_glm_b0_allocation_cpu.py'))
f = importlib.util.module_from_spec(spec); spec.loader.exec_module(f)
poolmod = __import__(f.PKG + '.pool', fromlist=['Pool'])
tree = ast.parse((f.SRC / 'families/glm5_next/cuda/multi.py').read_text())
cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'MultiDecoder')
cls.body = [x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name in ('_used', '_move', '_resize', '_settle')]
ns = {'align_up': lambda n: n}
exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), cls], type_ignores=[])), 'real_multi_lifecycle', 'exec'), ns)


class Rebinding(unittest.TestCase):
    def setUp(self):
        self.c = f.Caches(f.weights(indexed=False), 64, latent_kv='b0_fp8')
        self.m = ns['MultiDecoder'].__new__(ns['MultiDecoder'])
        self.m.e = types.SimpleNamespace(caches=self.c)
        self.m.pool = poolmod.Pool(64, align=1)
        self.m.arena = self.c.arena
        self.m.lanes = {}; self.m.partial = None
        self.x = self.m.pool.add(0, 8)
        self.c.b0_claim(self.x.eid, 0, 8)
        self.x.kept = [types.SimpleNamespace(ids=[1, 2, 3, 4])]
        for p, e in zip(self.c.b0_pools, self.c.b0_bindings[self.x.eid]):
            p.write(e, torch.tensor([0, 3]), torch.ones(2, 512, dtype=torch.bfloat16))

    def test_move_renews_generation_and_preserves_only_used_rows(self):
        old = self.c.b0_bindings[self.x.eid][0]
        before = self.c.b0_pools[0].storage[:4].clone()
        self.m._move(self.x, 16, 8)
        new = self.c.b0_bindings[self.x.eid][0]
        p = self.c.b0_pools[0]
        self.assertTrue(torch.equal(p.storage[16:20], before))
        self.assertEqual(p.valid[16:24].tolist(), [True, False, False, True] + [False] * 4)
        self.assertFalse(bool(p.valid[:8].any()))
        self.assertNotEqual(old.generation, new.generation)
        with self.assertRaises(ValueError):p.write(old, torch.tensor([0]), torch.ones(1, 512, dtype=torch.bfloat16))
        p.check()

    def test_overlapping_move_both_directions(self):
        for first, second in [(2, 0), (4, 2)]:
            c = f.Caches(f.weights(indexed=False), 32, latent_kv='b0_fp8')
            c.b0_claim('x', first, 8); p = c.b0_pools[0]
            p.write(c.b0_bindings['x'][0], torch.tensor([0, 1, 3]), torch.ones(3, 512, dtype=torch.bfloat16))
            expected = p.storage[first:first + 4].clone()
            c.b0_reframe('x', second, 8, used=4, move=True)
            self.assertTrue(torch.equal(p.storage[second:second + 4], expected))
            self.assertEqual(p.valid[second:second + 8].tolist(), [True, True, False, True] + [False] * 4)
            p.check()

    def test_resize_growth_invalidates_tail_and_old_handle(self):
        old = self.c.b0_bindings[self.x.eid][0]
        self.m._resize(self.x, 16)
        p = self.c.b0_pools[0]
        self.assertEqual(self.x.size, 16)
        self.assertEqual(p.valid[:16].tolist(), [True, False, False, True] + [False] * 12)
        self.assertNotEqual(old.generation, self.c.b0_bindings[self.x.eid][0].generation)

    def test_invalid_shrink_rejected_before_host_pool_mutation(self):
        old = self.c.b0_bindings[self.x.eid][0]
        with self.assertRaises(ValueError):self.m._resize(self.x, 2)
        self.assertEqual(self.x.size, 8)
        self.assertIs(old, self.c.b0_bindings[self.x.eid][0])

    def test_other_owner_overlap_rejected_without_payload_change(self):
        y = self.m.pool.add(16, 8); self.c.b0_claim(y.eid, 16, 8)
        original = self.c.b0_pools[0].storage.clone()
        with self.assertRaises(ValueError):self.m._move(self.x, 12, 8)
        self.assertEqual(self.x.base, 0)
        self.assertTrue(torch.equal(original, self.c.b0_pools[0].storage))

    def test_settle_retains_kept_prefix_then_releases(self):
        self.m._settle(self.x)
        self.assertEqual(self.x.size, 4)
        self.assertEqual(self.c.b0_bindings[self.x.eid][0].capacity, 4)
        self.x.kept = []
        self.m._settle(self.x)
        self.assertNotIn(self.x.eid, self.c.b0_bindings)
        self.assertFalse(bool(self.c.b0_pools[0].valid.any()))
        self.assertFalse(bool(self.c.b0_pools[0].generations.any()))

    def test_admission_reclaim_invalidates_prior_request_but_keeps_prefix(self):
        old = self.c.b0_bindings[self.x.eid][0]
        self.c.b0_claim(self.x.eid, 0, 8, used=4)
        self.assertNotEqual(old.generation, self.c.b0_bindings[self.x.eid][0].generation)
        self.assertEqual(self.c.b0_pools[0].valid[:8].tolist(), [True, False, False, True] + [False] * 4)

    def test_failed_copy_poison_all_layers(self):
        def fail(*args):raise RuntimeError('injected copy failure')
        self.c.arena.copy = fail
        with self.assertRaises(RuntimeError):self.c.b0_reframe(self.x.eid, 16, 8, used=4, move=True)
        for p in self.c.b0_pools:
            with self.assertRaises(ValueError):p.check()


if __name__ == '__main__':unittest.main(verbosity=2)
