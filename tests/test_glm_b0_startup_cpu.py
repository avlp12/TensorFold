"""Changed startup wiring; real CPU allocator, no CUDA imports or model weights."""
import importlib.util
from pathlib import Path
import sys
import unittest

spec = importlib.util.spec_from_file_location('alloc_startup', Path(__file__).with_name('test_glm_b0_allocation_cpu.py'))
f = importlib.util.module_from_spec(spec); spec.loader.exec_module(f)
p = f.load('startup_cache_profile', f.SRC / 'families/glm5_next/cuda/cache_profile.py')


class Startup(unittest.TestCase):
    def test_profile_drives_real_allocation_and_geometry(self):
        mode = p.CacheProfile.from_env('fp8', {'TF_GLM_LATENT_KV': 'b0_fp8'})
        self.assertEqual(mode.indexer, 'b0')
        c = f.Caches(f.weights(indexed=False), 32, **mode.options())
        self.assertEqual(c.latent_kv, 'b0_fp8')
        self.assertEqual(c.b0_pools[0].storage.shape[-1], 656)
        vendor = f.geometry.mla_geometry(f.config(), 2, 8, latent=True, kv='fp8')
        target = f.geometry.mla_geometry(f.config(), 2, 8, latent=True, **mode.options())
        self.assertGreater(target.bytes_at(2048), vendor.bytes_at(2048))
        self.assertNotEqual(mode.code(), p.CacheProfile.from_env('fp8', {}).code())
        with self.assertRaises(NotImplementedError):
            mode.require_ready({})


    def test_integrated_state_needs_current_owner(self):
        import types
        w = f.weights(indexed=False)
        w.meta['b0_speculation'] = True
        w.cfg.dense_limit = 2048
        for i, layer in enumerate(w.layers): layer.index = i
        c = f.Caches(w, 32, kv='fp8', latent_kv='b0_fp8')
        z = f.torch.zeros
        slots = types.SimpleNamespace(pos_dev=z(1,1,dtype=f.torch.int32),
            mtp_pos_dev=z(1,1,dtype=f.torch.int32), conv=z(1,0,3,1),
            rec=z(1,2,0,1), proj=z(1,0,4,1), scratch=[None])
        with self.assertRaisesRegex(ValueError, 'ownership binding'):
            f.State(w,32,4,caches=c,slots=slots)
        c.b0_home_claim(32)
        state = f.State(w,32,4,caches=c,slots=slots)
        self.assertEqual((state.base,state.capacity),(0,32))
        c.b0_home_release()
        with self.assertRaisesRegex(ValueError, 'ownership binding'):
            f.State(w,32,4,caches=c,slots=slots)


    def test_vendor_defaults_and_invalid_combinations(self):
        for kind in ('bf16', 'fp8'):
            mode = p.CacheProfile.from_env(kind, {})
            self.assertEqual(mode.options(), {'kv': kind, 'latent_kv': kind})
            self.assertEqual(mode.indexer, 'vendor')
            mode.require_ready()
        for kv, override in [('bf16', 'b0_fp8'), ('fp8', 'bf16'), ('fp8', 'unknown')]:
            with self.subTest(kv=kv, override=override), self.assertRaises(ValueError):
                p.CacheProfile.from_env(kv, {'TF_GLM_LATENT_KV': override})

    def test_home_reset_and_handoff_to_real_request(self):
        c = f.Caches(f.weights(indexed=False), 32, kv='fp8', latent_kv='b0_fp8')
        c.b0_home_claim(32)
        old = c.b0_bindings['engine-home'][0]
        c.b0_pools[0].valid[:16].fill_(True)
        c.b0_home_reset(32)
        self.assertFalse(c.b0_pools[0].valid.any())
        self.assertNotEqual(old.generation, c.b0_bindings['engine-home'][0].generation)
        c.b0_home_release()
        c.b0_claim(0, 0, 16)
        self.assertEqual(list(c.b0_bindings), [0])
        with self.assertRaisesRegex(ValueError, 'handed to the scheduler'):
            c.b0_home_reset(32)
        with self.assertRaisesRegex(ValueError, 'empty pool'):
            c.b0_home_claim(32)

    def test_default_cache_home_hooks_remain_noops(self):
        c = f.Caches(f.weights(indexed=False), 32)
        c.b0_home_claim(32); c.b0_home_reset(32); c.b0_home_release()
        self.assertFalse(c.b0_bindings)


if __name__ == '__main__':
    unittest.main()
