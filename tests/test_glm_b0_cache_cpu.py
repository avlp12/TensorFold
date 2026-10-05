"""CPU-only lifecycle tests; no TensorFold package init, MLX, Triton or weights.

Run directly with python3 tests/test_glm_b0_cache_cpu.py. These tests exercise the
real component methods; CUDA capture/TP4 and scheduler wiring remain untested.
"""
import importlib.util
from pathlib import Path
import sys
import types
import unittest

import torch

torch.set_default_device('cpu')
torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parents[1] / 'src/tensorfold/families/glm5_next/cuda'
PKG = '_b0_cache_cpu'
package = types.ModuleType(PKG)
package.__path__ = [str(ROOT)]
sys.modules[PKG] = package

def load(name):
    spec = importlib.util.spec_from_file_location(f'{PKG}.{name}', ROOT / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

codec = load('b0_fp8')
cache = load('b0_cache')
attention = load('b0_attention')


def rows(count, offset=0):
    return (torch.arange(count * 512, dtype=torch.float32).reshape(count, 512) % 97 / 23 + offset).to(torch.bfloat16)


def ids(*xs):
    return torch.tensor(xs, dtype=torch.int64)


class CacheLifecycle(unittest.TestCase):
    def setUp(self):
        self.pool = cache.PackedCachePool(12)
        self.a = self.pool.reserve('a', 0, 6)
        self.b = self.pool.reserve('b', 6, 6)

    def test_disjoint_writes_are_exact_original_codec_bytes(self):
        self.pool.write(self.a, ids(1, 4), rows(2))
        self.pool.write(self.b, ids(0, 5), rows(2, 10))
        self.assertTrue(torch.equal(self.pool.storage[[1, 4]], codec.pack_reference(rows(2))))
        self.assertTrue(torch.equal(self.pool.storage[[6, 11]], codec.pack_reference(rows(2, 10))))
        self.pool.check()

    def test_selection_offsets_and_padding_and_inactive_keys(self):
        self.pool.write(self.b, ids(0, 5), rows(2))
        mapped, lengths = self.pool.selected(self.b, ids([5, -1, 0], [0, 999, -999]), torch.tensor([3, 1], dtype=torch.int32))
        self.assertEqual(mapped.tolist(), [[11, -1, 6], [6, -1, -1]])
        self.assertEqual(lengths.tolist(), [3, 1])
        self.pool.check()

    def test_overlap_foreign_stale_host_extent_rejected(self):
        with self.assertRaises(ValueError):
            self.pool.reserve('c', 5, 2)
        forged = cache.Extent('a', 0, 6, self.a.generation)
        with self.assertRaises(ValueError):
            self.pool.write(forged, ids(0), rows(1))
        self.pool.release(self.a)
        new = self.pool.reserve('a', 0, 6)
        self.assertNotEqual(new.generation, self.a.generation)
        with self.assertRaises(ValueError):
            self.pool.write(self.a, ids(0), rows(1))

    def test_outside_write_does_not_change_other_request(self):
        self.pool.write(self.b, ids(0), rows(1, 10))
        original = self.pool.storage[:12].clone()
        self.pool.write(self.a, ids(0, 6), rows(2))
        self.assertTrue(torch.equal(original, self.pool.storage[:12]))
        with self.assertRaises(ValueError):
            self.pool.check()

    def test_duplicate_write_poisoned_without_partial_commit(self):
        self.pool.write(self.a, ids(2, 2), rows(2))
        self.assertFalse(bool(self.pool.valid[:12].any()))
        with self.assertRaises(ValueError):
            self.pool.check()

    def test_nonfinite_write_poisoned(self):
        for bad in [float('nan'), float('inf'), -float('inf')]:
            with self.subTest(bad=bad):
                p = cache.PackedCachePool(2)
                e = p.reserve('a', 0, 2)
                x = rows(1); x[0, 4] = bad
                p.write(e, ids(0), x)
                self.assertFalse(bool(p.valid[:2].any()))
                with self.assertRaises(ValueError):
                    p.check()

    def test_unwritten_row_is_not_readable(self):
        mapped, _ = self.pool.selected(self.a, ids([0]), torch.tensor([1], dtype=torch.int32))
        self.assertEqual(mapped.tolist(), [[-1]])
        with self.assertRaises(ValueError):
            self.pool.check()

    def test_cross_request_read_rejected_even_when_other_request_resident(self):
        self.pool.write(self.b, ids(0), rows(1))
        mapped, _ = self.pool.selected(self.a, ids([6]), torch.tensor([1], dtype=torch.int32))
        self.assertEqual(mapped.tolist(), [[-1]])
        with self.assertRaises(ValueError):
            self.pool.check()

    def test_bad_lengths_and_negative_indices_rejected(self):
        for key, length in [(0, -1), (0, 2), (-2, 1)]:
            with self.subTest(key=key, length=length):
                p = cache.PackedCachePool(2); e = p.reserve('a', 0, 2)
                p.write(e, ids(0), rows(1))
                p.selected(e, ids([key]), torch.tensor([length], dtype=torch.int32))
                with self.assertRaises(ValueError):
                    p.check()

    def test_error_survives_later_success_and_slot_reuse(self):
        self.pool.write(self.a, ids(6), rows(1))
        self.pool.write(self.a, ids(0), rows(1))
        self.pool.release(self.a)
        c = self.pool.reserve('c', 0, 6)
        self.pool.write(c, ids(0), rows(1))
        with self.assertRaises(ValueError):
            self.pool.check()

    def test_release_clears_residency(self):
        self.pool.write(self.a, ids(0), rows(1))
        self.pool.release(self.a)
        c = self.pool.reserve('c', 0, 6)
        self.pool.selected(c, ids([0]), torch.tensor([1], dtype=torch.int32))
        with self.assertRaises(ValueError):
            self.pool.check()

    def test_device_generation_rejects_write_without_host_registry_change(self):
        # Captured graphs skip Python ownership checks on replay. Their device
        # generation predicate must still protect a subsequently reused slot.
        self.pool.generations[:6].add_(1)
        original = self.pool.storage.clone()
        self.pool.write(self.a, ids(0), rows(1))
        self.assertTrue(torch.equal(original[:12], self.pool.storage[:12]))
        with self.assertRaises(ValueError):
            self.pool.check()

    def test_device_generation_rejects_even_an_empty_selection(self):
        self.pool.generations[:6].zero_()
        mapped, _ = self.pool.selected(self.a, ids([-1]), torch.tensor([0], dtype=torch.int32))
        self.assertEqual(mapped.tolist(), [[-1]])
        with self.assertRaises(ValueError):
            self.pool.check()

    def test_attention_layer_error_latches_across_later_clean_results(self):
        zero = torch.zeros((), dtype=torch.bool)
        good = torch.ones(1, 2, 512, dtype=torch.bfloat16)
        first = attention.AttentionResult(good, zero, error_latch=self.pool.error)
        self.assertIs(first.checked(), good)
        attention.AttentionResult(good, ~zero, error_latch=self.pool.error)
        last = attention.AttentionResult(good, zero, error_latch=self.pool.error)
        for result in (first, last):
            with self.assertRaises(ValueError):
                result.checked()
        with self.assertRaises(ValueError):
            self.pool.check()

    def test_nonfinite_attention_poison_survives_buffer_reuse(self):
        values = torch.full((1, 2, 512), float('nan'), dtype=torch.bfloat16)
        result = attention.AttentionResult(values, torch.zeros((), dtype=torch.bool),
                                           error_latch=self.pool.error)
        values.fill_(1)
        with self.assertRaises(ValueError):
            result.checked()
        with self.assertRaises(ValueError):
            self.pool.check()

    def test_metadata_rejects_instead_of_converting(self):
        with self.assertRaises(ValueError):
            self.pool.write(self.a, ids(0).to(torch.int32), rows(1))
        with self.assertRaises(ValueError):
            self.pool.write(self.a, ids(0), rows(1).float())
        with self.assertRaises(ValueError):
            self.pool.selected(self.a, ids([0]), torch.tensor([1], dtype=torch.int64))


if __name__ == '__main__':
    unittest.main(verbosity=2)
