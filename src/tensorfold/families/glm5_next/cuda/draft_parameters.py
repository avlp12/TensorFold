"""Explicit DFlash2 forward/selector parameters, independent of GPU imports."""
from dataclasses import dataclass
import math
import os


@dataclass(frozen=True)
class DraftParameters:
    block: int | None
    edge: float
    noise: float

    def code(self) -> list[int]:
        # Float bit patterns, not rounded decimal millionths, for rank agreement.
        import struct
        return [self.block or 0, *struct.unpack('<2i', struct.pack('<d', self.edge)),
                *struct.unpack('<2i', struct.pack('<d', self.noise))]

    @classmethod
    def from_env(cls, env=None):
        env = os.environ if env is None else env
        raw = env.get('TF_GLM_DFLASH_BLOCK', '').strip()
        block = int(raw) if raw else None
        edge = float(env.get('TF_GLM_DFLASH_EDGE', '') or '0.6')
        noise = float(env.get('TF_GLM_DFLASH_NOISE', '') or '0.7')
        if block is not None and not 2 <= block <= 16:
            raise ValueError('TF_GLM_DFLASH_BLOCK must be 2..16 or unset')
        if not all(math.isfinite(v) and v >= 0 for v in (edge, noise)):
            raise ValueError('DFlash2 edge/noise must be finite and nonnegative')
        return cls(block, edge, noise)

    def config(self, config):
        if self.block is None:
            return config
        return dict(config, dflash_config=dict(config['dflash_config'], block_size=self.block))
