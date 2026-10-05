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
        return cls(kv, kind)

    @property
    def indexer(self):
        return 'b0' if self.latent_kv == 'b0_fp8' else 'vendor'

    def code(self):
        return [('bf16', 'fp8').index(self.kv),
                ('bf16', 'fp8', 'b0_fp8').index(self.latent_kv)]

    def options(self):
        return {'kv': self.kv, 'latent_kv': self.latent_kv}

    def require_ready(self):
        if self.latent_kv == 'b0_fp8':
            raise NotImplementedError('B0 model startup remains gated: DFlash2 parity and integrated startup validation pending')
