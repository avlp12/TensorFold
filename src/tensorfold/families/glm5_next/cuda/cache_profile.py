"""One cache selection for the loader, capacity estimate and allocator.

Selecting B0 bytes is not model admission: startup ownership and exact DFlash2
integration must be finished before the B0 execution guard can be removed.
"""
from dataclasses import dataclass
import os


@dataclass(frozen=True)
class CacheProfile:
    kv: str
    latent_kv: str
    b0_attention: str = 'triton'

    @classmethod
    def from_env(cls, kv, env=None):
        if kv not in ('bf16', 'fp8'):
            raise ValueError('index storage must be bf16 or fp8')
        env = os.environ if env is None else env
        kind = env.get('TF_GLM_LATENT_KV', '').strip() or kv
        if kind not in (kv, 'b0_fp8'):
            raise ValueError(f'unknown TF_GLM_LATENT_KV {kind!r}')
        if kind == 'b0_fp8' and kv != 'fp8':
            raise ValueError('B0 index rows need TF_GLM_KV=fp8 byte storage')
        reader = env.get('TF_GLM_B0_ATTENTION', '').strip() or 'triton'
        if reader not in ('triton', 'flashinfer') or (reader != 'triton' and kind != 'b0_fp8'):
            raise ValueError('FlashInfer B0 reader requires b0_fp8 latent storage')
        return cls(kv, kind, reader)

    @property
    def indexer(self):
        return 'b0' if self.latent_kv == 'b0_fp8' else 'vendor'

    def code(self):
        return [('bf16', 'fp8').index(self.kv),
                ('bf16', 'fp8', 'b0_fp8').index(self.latent_kv),
                ('triton', 'flashinfer').index(self.b0_attention)]

    def options(self):
        out = {'kv': self.kv, 'latent_kv': self.latent_kv}
        if self.b0_attention != 'triton':
            out['b0_attention'] = self.b0_attention
        return out

    def require_ready(self, env=None):
        if self.latent_kv != 'b0_fp8':
            return
        env = os.environ if env is None else env
        required = {'TF_GLM_LATENT': '1', 'TF_GLM_KV': 'fp8', 'TF_GLM_DENSE': 'bf16',
                    'TF_GLM_KVB': 'bf16', 'TF_GLM_DRAFT_QUANT': 'bf16',
                    'TF_GLM_DFLASH_BLOCK': '4', 'TF_GLM_DFLASH_EDGE': '1',
                    'TF_GLM_DFLASH_NOISE': '1', 'TF_GLM_DFLASH_POLICY': 'f3',
                    'TF_GLM_COPY_DRAFTS': '0', 'TF_GLM_MTP': '0',
                    'TF_GLM_MULTI_DEPTH': 'policy', 'TF_GLM_DISPLAY_KV_MIB': '0'}
        mismatched = [key for key, value in required.items() if env.get(key, '').strip() != value]
        if mismatched:
            raise NotImplementedError('B0 startup requires its integrated fixed DFlash2 profile: ' + ', '.join(mismatched))


def cache_scope(salt):
    """Opaque per-request prefix namespace, transmitted identically to TP peers."""
    if salt is None:
        return ()
    if not isinstance(salt, str) or len(salt) > 256:
        raise ValueError('cache_salt must be a string of at most256 characters')
    import hashlib
    import struct
    return tuple(struct.unpack('<8i', hashlib.sha256(salt.encode('utf-8')).digest()))


def aligned_pool_rows(capacity, spare_rows, alignment):
    if capacity <= 0 or spare_rows < 0 or alignment <= 0:
        raise ValueError('positive capacity/alignment and nonnegative spare rows required')
    minimum = -(-capacity // alignment) * alignment
    return max(minimum, (capacity + spare_rows) // alignment * alignment)


def explicit_pool_rows(value, minimum, alignment):
    """Optional total shared pool, independent of per-request scratch capacity."""
    if value in (None, ''):
        return None
    if not str(value).isdecimal() or not minimum <= int(value) < 2**31:
        raise ValueError('TF_GLM_POOL_TOKENS must preserve the request window and fit int32')
    return -(-int(value) // alignment) * alignment
