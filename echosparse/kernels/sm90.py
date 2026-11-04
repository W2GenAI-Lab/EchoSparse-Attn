"""H100 (SM90) kernel set, in Triton.

SM90 has no tcgen05 MMAs or tensor memory either, so its attention kernel is a Triton kernel with ordinary pointer
loads, tuned for the H100: one 64-token exact block per iteration with 8 warps, and in the sparse step the exact
loop software-pipelined over three stages (the K / V gathers overlap the dots) and 128 compensation columns per
batch (128 mean blocks, or 32 refined blocks x 4 representatives). Same quantities and units as the other sets:
128-row query tiles, the previous step's log2 LSE as a fixed softmax offset, block masses summed over the tile's
rows, BF16 operands and FP32 accumulation.

The plans, fallback and the compensation lists are the RTX 5090 kernels; the block means are computed in their own
kernel before the representatives, and the estimates one query tile per program.
"""
import math

import torch
import triton
import triton.language as tl

from . import representatives
from .sm120 import exact_fallback, launched_tiles, pack_plan, rebuild_plan  # noqa: F401
from .triton_ops import block_means, compensate_fallback, estimate_skipped, full_step_scales, select_blocks  # noqa

BM = 128
BK = 64
LOG2E = 1.4426950408889634
CW = 128                 # compensation columns per batch: 128 means, or 32 refined blocks x 4
JS = 3                   # software-pipeline stages of the sparse step's exact loop


class Representatives(representatives.Representatives):
    def build(self, q, k, v, k_mean=None, v_mean=None):
        """Block means (if given k_mean / v_mean) in their own kernel, then the representatives of every block."""
        if k_mean is not None:
            block_means(k, v, k_mean, v_mean)
        super().build(q, k, v)


@triton.jit
def _exact_block(q, prior, valid, acc, total, K, V, MASS, b, t, h, L, NB, NQ, H: tl.constexpr, D: tl.constexpr,
                 SCALE: tl.constexpr, WRITE_MASS: tl.constexpr):
    row = b * 64 + tl.arange(0, 64)
    ko = (row[:, None] * H + h) * D + tl.arange(0, D)[None, :]
    k = tl.load(K + ko, mask=(row < L)[:, None], other=0)
    s = tl.dot(q, tl.trans(k)) * SCALE - prior[:, None]
    s = tl.where(valid[:, None] & (row < L)[None, :], s, -float('inf'))
    p = tl.exp2(s)
    total += tl.sum(p, 1)
    if WRITE_MASS:
        tl.store(MASS + (h * NQ + t) * NB + b, tl.sum(tl.sum(p, 1), 0))
    v = tl.load(V + ko, mask=(row < L)[:, None], other=0)
    return tl.dot(p.to(tl.bfloat16), v, acc), total


@triton.jit
def _attention_kernel(Q, K, V, KB, VB, O, CNT, IDX, PRIOR, LSE, FLAG, MASS, MIDX, MCNT, RIDX, RCNT, K4, V4, LC4, L,
                      NB, NQ, NK: tl.constexpr, RW: tl.constexpr, H: tl.constexpr, D: tl.constexpr, BM: tl.constexpr,
                      SCALE: tl.constexpr, TH: tl.constexpr, LAST_LOG2: tl.constexpr, COMP: tl.constexpr,
                      WRITE_MASS: tl.constexpr, CW: tl.constexpr, JS: tl.constexpr):
    # The listed 64-token blocks exactly (JS > 0: as `tl.range(..., num_stages=JS)`); with COMP the listed mean blocks
    # through their mean key and value and the listed refined blocks through four representatives each, CW columns at
    # a time. NQ: query tiles of the sequence (the grid may cover only the first ones).
    t, h = tl.program_id(0), tl.program_id(1)
    rr = t * BM + tl.arange(0, BM)
    dd = tl.arange(0, D)
    valid = rr < L
    qo = (rr[:, None] * H + h) * D + dd[None, :]
    q = tl.load(Q + qo, mask=valid[:, None], other=0)
    prior = tl.load(PRIOR + rr * H + h)
    acc = tl.zeros((BM, D), tl.float32)
    total = tl.zeros((BM,), tl.float32)
    n = tl.load(CNT + t * H + h)
    ib = (t * H + h) * NK
    if JS == 0:
        for j in range(0, n):
            acc, total = _exact_block(q, prior, valid, acc, total, K, V, MASS, tl.load(IDX + ib + j), t, h, L, NB,
                                      NQ, H, D, SCALE, WRITE_MASS)
    else:
        for j in tl.range(0, n, num_stages=JS, loop_unroll_factor=1, warp_specialize=False):
            acc, total = _exact_block(q, prior, valid, acc, total, K, V, MASS, tl.load(IDX + ib + j), t, h, L, NB,
                                      NQ, H, D, SCALE, WRITE_MASS)
    if COMP:
        cc = tl.arange(0, CW)
        nm = tl.load(MCNT + t * H + h)
        for c0 in tl.range(0, nm, CW, num_stages=1):
            colv = c0 + cc < nm
            b = tl.load(MIDX + ib + c0 + cc, mask=colv, other=NB)
            bo = (b[:, None] * H + h) * D + dd[None, :]
            kb = tl.load(KB + bo, mask=colv[:, None], other=0.0)
            vb = tl.load(VB + bo, mask=colv[:, None], other=0.0)
            lc = tl.where(colv, tl.where(b == NB - 1, LAST_LOG2, 6.0), -float('inf'))
            s = tl.dot(q, tl.trans(kb)) * SCALE + lc[None, :] - prior[:, None]
            s = tl.where(valid[:, None] & colv[None, :], s, -float('inf'))
            p = tl.exp2(s)
            total += tl.sum(p, 1)
            acc = tl.dot(p.to(tl.bfloat16), vb, acc)
        nr = tl.load(RCNT + t * H + h)
        for r0 in tl.range(0, nr, CW // 4, num_stages=1):
            colv = r0 + cc // 4 < nr
            b = tl.load(RIDX + (t * H + h) * RW + r0 + cc // 4, mask=colv, other=NB)
            c = b * 4 + cc % 4
            co = (h * ((NB + 1) * 4) + c[:, None]) * D + dd[None, :]
            kc = tl.load(K4 + co, mask=colv[:, None], other=0.0)
            vc = tl.load(V4 + co, mask=colv[:, None], other=0.0)
            lc = tl.load(LC4 + h * ((NB + 1) * 4) + c, mask=colv, other=-float('inf'))
            s = tl.dot(q, tl.trans(kc)) * SCALE + lc[None, :] - prior[:, None]
            s = tl.where(valid[:, None] & colv[None, :], s, -float('inf'))
            p = tl.exp2(s)
            total += tl.sum(p, 1)
            acc = tl.dot(p.to(tl.bfloat16), vc, acc)

    lg = tl.log2(total)
    bad = tl.max((valid & ~((lg >= -TH) & (lg <= TH))).to(tl.int32), 0)
    tl.store(FLAG + t * H + h, bad)
    tl.store(O + qo, (acc / total[:, None]).to(tl.bfloat16), mask=valid[:, None])
    tl.store(LSE + rr * H + h, tl.where(valid, prior + lg, 1e30))


def block_attention(q, k, v, out, counts, blocks, offset, lse, flags, mass=None, threshold=60.0):
    """Exact attention over a list of 64-token kv blocks with a fixed offset, plus per-block masses."""
    L, H, D = q.shape
    assert D == 128 and q.dtype == torch.bfloat16
    nq = triton.cdiv(L, BM)
    _attention_kernel[(nq, H)](q, k, v, q, v, out, counts, blocks, offset, lse, flags,
                               offset if mass is None else mass, blocks, counts, blocks, counts, q, v, offset, L,
                               triton.cdiv(L, BK), nq, NK=blocks.shape[-1], RW=0, H=H, D=D, BM=BM,
                               SCALE=LOG2E / math.sqrt(D), TH=float(threshold), LAST_LOG2=0.0, COMP=False,
                               WRITE_MASS=mass is not None, CW=CW, JS=0, num_warps=8, num_stages=1)


def sparse_attention(q, k, v, k_mean, v_mean, means, mean_counts, out, counts, blocks, offset, lse, flags, refine,
                     refine_counts, k_rep, v_rep, log_counts, mass=None, threshold=60.0, rows=None):
    """`sm120.sparse_attention` with the H100 tuning."""
    L, H, D = q.shape
    assert D == 128 and q.dtype == torch.bfloat16
    nq, nb = triton.cdiv(L, BM), triton.cdiv(L, BK)
    tiles = launched_tiles(out, lse, flags, offset, rows)
    if tiles == 0:
        return
    _attention_kernel[(tiles, H)](q, k, v, k_mean, v_mean, out, counts, blocks, offset, lse, flags,
                                  offset if mass is None else mass, means, mean_counts, refine, refine_counts,
                                  k_rep, v_rep, log_counts, L, nb, nq, NK=blocks.shape[-1], RW=refine.shape[-1], H=H,
                                  D=D, BM=BM, SCALE=LOG2E / math.sqrt(D), TH=float(threshold),
                                  LAST_LOG2=math.log2(L - (nb - 1) * BK), COMP=True, WRITE_MASS=mass is not None,
                                  CW=CW, JS=JS, num_warps=8, num_stages=1)
