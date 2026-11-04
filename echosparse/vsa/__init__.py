"""Native CUDA original VSA and VSA2 for H100 (SM90) and RTX 5090 (SM120), stateless and separate from
EchoSparseAttention.

`vsa_attention` (original VSA: whole-block router, K kv blocks per query block, full direct weight) and
`vsa2_attention` (VSA2: R-token PSP router, exact global NB*K budget, scalar tanh gate) take the preblocked
[B, H, NB*64, D] layout; `layout` maps real tokens to it and back. The extension is built on request
(ECHOSPARSE_VSA_BUILD=1) and imported at the first call.
"""
from .layout import BlockLayout, build_layout, from_blocked, sequence_layout, to_blocked
from .ops import (Selection, fine_block_sparse_attention, mask_to_selection, resolve_bwd_variant, route, route_rows,
                  set_bwd_variant, vsa2_attention, vsa_attention)

__all__ = ["BlockLayout", "Selection", "build_layout", "fine_block_sparse_attention", "from_blocked",
           "mask_to_selection", "resolve_bwd_variant", "route", "route_rows", "sequence_layout", "set_bwd_variant",
           "to_blocked", "vsa2_attention", "vsa_attention"]
