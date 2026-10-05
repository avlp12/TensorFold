"""Pure byte layout metadata shared by MLA allocation and memory estimates.

b0_fp8 describes only GLM's 512-wide latent plane. It says nothing about the
indexer, execution support, or whether an engine is B0-equivalent.
"""

KINDS = ("bf16", "fp8", "b0_fp8")


def row_bytes(width: int, kind: str) -> int:
    if type(width) is not int or width <= 0:
        raise ValueError("positive integer MLA width required")
    if kind == "bf16":
        return width * 2
    if kind == "fp8":
        if width % 4:
            raise ValueError("vendor FP8 width must be divisible by four")
        return width + 16
    if kind == "b0_fp8":
        if width != 512:
            raise ValueError("B0 packed MLA supports only 512 latent values")
        return 512 + 4 * 4 + 64 * 2
    raise ValueError(f"unknown MLA row layout {kind!r}")


def latent_kind(kv: str, override: str | None, width: int) -> str:
    # The indexer format remains explicit and separate. In particular, b0_fp8
    # must never imply that the vendor pooled indexer already matches B0.
    if kv not in ("bf16", "fp8"):
        raise ValueError("indexer cache format must remain bf16 or vendor fp8")
    kind = kv if override is None else override
    row_bytes(width, kind)
    return kind
