"""Generic fine-backward variant selection. No torch dependency, so CPU checks can load it alone.

Variants: 0 float2 atomic dQ, 1 word atomic dQ, 2 no fine-branch atomics (kv pass plus q->kv pass).
Precedence: per-call argument, ops.set_bwd_variant, ECHOSPARSE_VSA_BWD_VARIANT, automatic rule.
The TK fine backward on SM90 does not use any of these.

Automatic rule: SM120 uses variant 2 when the fp32 dQ footprint B*H*L_padded*D reaches
AUTO_V2_MIN_DQ_ELEMS, otherwise variant 0. Other devices use variant 0.

Evidence (RTX 5090, one build of this core, H20 D128 K64 B64 R16 BF16, VSA2 router):
    2.94e8 elements (98,820 tokens):  variant 0 faster, F+B  96.29 vs 112.64 ms
    6.29e8 elements (219,600 tokens): variant 2 faster, F+B 282.56 vs 252.46 ms
    1.22e9 elements (435,600 tokens): variant 2 faster, F+B 698.72 vs 555.01 ms
The threshold is a linear interpolation of per-pair fine-backward cost between the first two points,
not a measured crossover. The footprint metric is a choice: only L varied in those runs, so scaling
with B, H, D or K is unverified. Variant 2 has the lower peak memory at all three lengths; pass it
explicitly if memory matters more than time below the threshold. Original VSA (same NB*K pairs, per
row) uses the same rule; its crossover is unmeasured.
"""

import os
from typing import Optional, Tuple

AUTO_V2_MIN_DQ_ELEMS = 500_000_000
BWD_VARIANTS = (0, 1, 2)
ENV_NAME = "ECHOSPARSE_VSA_BWD_VARIANT"


def check_bwd_variant(v, what: str) -> int:
    if v not in BWD_VARIANTS:
        raise ValueError(f"{what} must be one of {BWD_VARIANTS}, got {v!r}")
    return int(v)


def auto_bwd_variant(B: int, H: int, L: int, D: int, gpu_major: int) -> int:
    return 2 if gpu_major == 12 and B * H * L * D >= AUTO_V2_MIN_DQ_ELEMS else 0


def select_bwd_variant(B: int, H: int, L: int, D: int, gpu_major: int, call: Optional[int] = None,
                       override: Optional[int] = None, environ=os.environ) -> Tuple[int, str]:
    """Returns (variant, source), source being call, set_bwd_variant, environment or auto."""
    if call is not None:
        return check_bwd_variant(call, "bwd_variant"), "call"
    if override is not None:
        return check_bwd_variant(override, "set_bwd_variant"), "set_bwd_variant"
    env = environ.get(ENV_NAME)
    if env is not None:
        try:
            value = int(env)
        except ValueError:
            raise ValueError(f"{ENV_NAME} must be one of {BWD_VARIANTS}, got {env!r}") from None
        return check_bwd_variant(value, ENV_NAME), "environment"
    return auto_bwd_variant(B, H, L, D, gpu_major), "auto"
