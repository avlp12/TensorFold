"""Changed allocator/dispatch boundary; no CUDA, weights or numeric kernel claim."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('fi_alloc', Path(__file__).with_name('test_glm_b0_allocation_cpu.py'))
f = importlib.util.module_from_spec(spec); spec.loader.exec_module(f)
t = f.torch
reader = f.load(f.PKG + '.b0_flashinfer', f.SRC / 'families/glm5_next/cuda/b0_flashinfer.py')
profile = f.load(f.PKG + '.cache_profile', f.SRC / 'families/glm5_next/cuda/cache_profile.py')


class ChangedIntegration(unittest.TestCase):
    def test_empty_selection_uses_sink_and_padding_is_masked(self):
        indices = t.tensor([[0, 7, 999], [-1, -1, 123]])
        values, empty = reader.prepare(indices, t.tensor([2, 2], dtype=t.int32), 64)
        self.assertEqual(values.shape, (2, 2048))
        self.assertEqual(values[0, :4].tolist(), [0, 7, -1, -1])
        self.assertEqual(values[1, :4].tolist(), [64, -1, -1, -1])
        self.assertEqual(empty.tolist(), [False, True])

    def test_actual_allocator_padding_and_geometry_agree(self):
        for rows in (63, 64, 65, 8192):
            with self.subTest(rows=rows):
                plain = f.Caches(f.weights(indexed=False), rows, latent_kv='b0_fp8')
                fast = f.Caches(f.weights(indexed=False), rows, latent_kv='b0_fp8', b0_attention='flashinfer')
                pad = reader.page_padding(rows)
                self.assertEqual(fast.b0_pools[0].storage.shape, (rows+pad, 656))
                expected = 2 * ((pad-1)*656 + (rows//64+1)*4) + reader.WORKSPACE_BYTES
                self.assertEqual(fast.allocation_bytes()-plain.allocation_bytes(), expected)
                a = f.geometry.mla_geometry(f.config(), 2, 8, latent=True, latent_kv='b0_fp8')
                b = f.geometry.mla_geometry(f.config(), 2, 8, latent=True, latent_kv='b0_fp8', b0_attention='flashinfer')
                self.assertEqual(b.bytes_at(rows)-a.bytes_at(rows), expected)

    def test_owned_pool_routes_fast_reader_and_keeps_error_latch(self):
        c = f.Caches(f.weights(indexed=False), 64, latent_kv='b0_fp8', b0_attention='flashinfer')
        pool = c.b0_pools[0]; owner = pool.reserve('request', 8, 16)
        pool.write(owner, t.tensor([0, 1]), t.ones(2, 512, dtype=t.bfloat16))
        calls = []
        class Stub:
            def __init__(self, storage, rows):
                calls.append((storage.shape, rows))
            def __call__(self, q, indices, lengths, scale):
                calls.append((indices.tolist(), lengths.tolist(), scale))
                return t.zeros_like(q)
        with patch.object(reader, 'Reader', Stub):
            pool.attention(owner, t.zeros(1,16,512,dtype=t.bfloat16), t.tensor([[0,1]]), t.tensor([2],dtype=t.int32),0.1).checked()
            self.assertEqual(calls[1][0], [[8,9]])
            with self.assertRaises(ValueError):
                pool.attention(owner, t.zeros(1,16,512,dtype=t.bfloat16), t.tensor([[2]]), t.tensor([1],dtype=t.int32),0.1).checked()
        self.assertTrue(bool(pool.error))

    def test_total_pool_geometry_leaves_per_request_scratch_unchanged(self):
        capacity, pool = 8192, 40960
        a = f.geometry.mla_geometry(f.config(),2,8,latent=True,kv='fp8',latent_kv='b0_fp8')
        b = f.geometry.mla_geometry(f.config(),2,8,latent=True,kv='fp8',latent_kv='b0_fp8',pool_rows=pool)
        delta = 2 * ((pool-capacity)*(656+9) + (pool-capacity)//4*144)
        self.assertEqual(b.bytes_at(capacity)-a.bytes_at(capacity),delta)
        self.assertEqual(profile.explicit_pool_rows('5499901',850016,2048),5500928)
        for value in ('849999','-1','nan','2147483648'):
            with self.assertRaises(ValueError): profile.explicit_pool_rows(value,850016,2048)

    def test_2051_dense_transition_never_discards_active_keys(self):
        c = f.Caches(f.weights(indexed=False),4096,latent_kv='b0_fp8',b0_attention='flashinfer')
        pool = c.b0_pools[0]; owner = pool.reserve('request',0,4096)
        pool.valid[:4096].fill_(True)
        calls = []
        class Stub:
            def __call__(self,q,indices,lengths,scale):
                calls.append(('fast',indices.shape[1]));return t.zeros_like(q)
        pool.reader = Stub()
        att = sys.modules.get(f.PKG+'.b0_attention') or f.load(f.PKG+'.b0_attention',f.SRC/'families/glm5_next/cuda/b0_attention.py')
        def generic(q,cache,indices,lengths,scale,**kw):
            calls.append(('generic',indices.shape[1]));return att.AttentionResult(t.zeros_like(q),pool.error)
        with patch.object(att,'attention_capture',generic):
            q = t.zeros(1,16,512,dtype=t.bfloat16);keys = t.arange(2051).reshape(1,-1)
            for n in (2000,2051):pool.attention(owner,q,keys,t.tensor([n],dtype=t.int32),0.1).checked()
        self.assertEqual(calls,[('fast',2048),('generic',2051)])
        self.assertEqual(pool.fallback_calls,1)

    def test_all_ranks_agree_reader_profile_and_vendor_stays_default(self):
        a = profile.CacheProfile.from_env('fp8',{'TF_GLM_LATENT_KV':'b0_fp8'})
        b = profile.CacheProfile.from_env('fp8',{'TF_GLM_LATENT_KV':'b0_fp8','TF_GLM_B0_ATTENTION':'flashinfer'})
        self.assertNotEqual(a.code(),b.code())
        self.assertEqual(b.options()['b0_attention'],'flashinfer')
        with self.assertRaises(ValueError):
            profile.CacheProfile.from_env('fp8',{'TF_GLM_B0_ATTENTION':'flashinfer'})


if __name__ == '__main__': unittest.main(verbosity=2)
