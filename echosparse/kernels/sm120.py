"""RTX 5090 (SM120) kernel set, in Triton.

SM120 has no tcgen05 MMAs or tensor memory, so the two attention kernels are written with ordinary loads and
tensor-core dots. They compute the same quantities as the B200 kernels, in the same units: 128-row query tiles,
64-token kv blocks, the previous step's log2 LSE as a fixed softmax offset, block masses summed over the tile's
rows, BF16 operands and FP32 accumulation.

There are two sparse kernels. The 64-wide one takes each block in one piece and the block means and representatives
128 columns at a time (Wan2.1, HunyuanVideo). The MiniMax-H3 one walks the 64-token blocks as two 32-token halves:
the row sums of the two halves are added before they enter the normalizer and the block mass, and the block means
and representatives are added in groups of four 32-column pieces, so a 128-column group of the 64-wide formulation
is summed as (a + b) + (c + d).

The selection, compensation and Gram / Cholesky / probe kernels are shared with the B200 set.
"""
import math

import torch
import triton
import triton.language as tl

from . import representatives
from . import triton_ops
from .triton_ops import block_means, compensate_fallback, full_step_scales, select_blocks  # noqa: F401

BM = 128
BK = 64
LOG2E = 1.4426950408889634


@triton.jit
def _block_kernel(Q, K, V, KB, VB, BITS, O, CNT, IDX, PRIOR, LSE, FLAG, MASS, RIDX, K4, V4, LC4, L, NB,
                  NK: tl.constexpr, NW: tl.constexpr, RW: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                  BM: tl.constexpr, SCALE: tl.constexpr, TH: tl.constexpr, LAST_LOG2: tl.constexpr,
                  COMP: tl.constexpr, REFINE: tl.constexpr, WRITE_MASS: tl.constexpr):
    # listed 64-token blocks exactly; with COMP / REFINE also the other skipped blocks through their means and the
    # refined blocks through four representatives each, 128 columns at a time
    t, h = tl.program_id(0), tl.program_id(1)
    nq = tl.num_programs(0)
    rr = t * BM + tl.arange(0, BM)
    dd = tl.arange(0, D)
    valid = rr < L
    qo = (rr[:, None] * H + h) * D + dd[None, :]
    q = tl.load(Q + qo, mask=valid[:, None], other=0)
    prior = tl.load(PRIOR + rr * H + h)
    acc = tl.zeros((BM, D), tl.float32)
    total = tl.zeros((BM,), tl.float32)
    n = tl.load(CNT + t * H + h)
    kr = tl.arange(0, 64)
    for j in range(0, n):
        b = tl.load(IDX + (t * H + h) * NK + j)
        row = b * 64 + kr
        ko = (row[:, None] * H + h) * D + dd[None, :]
        k = tl.load(K + ko, mask=(row < L)[:, None], other=0)
        s = tl.dot(q, tl.trans(k)) * SCALE - prior[:, None]
        s = tl.where(valid[:, None] & (row < L)[None, :], s, -float('inf'))
        p = tl.exp2(s)
        total += tl.sum(p, 1)
        if WRITE_MASS:
            tl.store(MASS + (h * nq + t) * NB + b, tl.sum(tl.sum(p, 1), 0))
        v = tl.load(V + ko, mask=(row < L)[:, None], other=0)
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
    if COMP:
        cc = tl.arange(0, 128)
        for j0 in range(0, NB, 128):
            b = j0 + cc
            bo = (b[:, None] * H + h) * D + dd[None, :]
            kb = tl.load(KB + bo, mask=(b < NB)[:, None], other=0)
            vb = tl.load(VB + bo, mask=(b < NB)[:, None], other=0)
            word = tl.load(BITS + (h * nq + t) * NW + b // 32, mask=b < NB, other=-1)
            skip = ((word >> (b % 32)) & 1) != 0
            count_log = tl.where(b == NB - 1, LAST_LOG2, 6.0)
            s = tl.dot(q, tl.trans(kb)) * SCALE + count_log[None, :] - prior[:, None]
            s = tl.where(valid[:, None] & (b < NB)[None, :] & (~skip)[None, :], s, -float('inf'))
            p = tl.exp2(s)
            total += tl.sum(p, 1)
            acc = tl.dot(p.to(tl.bfloat16), vb, acc)
    if REFINE:
        cc = tl.arange(0, 128)
        for r0 in range(0, RW, 32):
            b = tl.load(RIDX + (t * H + h) * RW + r0 + cc // 4)
            c = b * 4 + cc % 4
            co = (h * ((NB + 1) * 4) + c[:, None]) * D + dd[None, :]
            kc = tl.load(K4 + co)
            vc = tl.load(V4 + co)
            lc = tl.load(LC4 + h * ((NB + 1) * 4) + c)
            s = tl.dot(q, tl.trans(kc)) * SCALE + lc[None, :] - prior[:, None]
            s = tl.where(valid[:, None], s, -float('inf'))
            p = tl.exp2(s)
            total += tl.sum(p, 1)
            acc = tl.dot(p.to(tl.bfloat16), vc, acc)
    lg = tl.log2(total)
    bad = tl.max((valid & ~((lg >= -TH) & (lg <= TH))).to(tl.int32), 0)
    tl.store(FLAG + t * H + h, bad)
    tl.store(O + qo, (acc / total[:, None]).to(tl.bfloat16), mask=valid[:, None])
    tl.store(LSE + rr * H + h, tl.where(valid, prior + lg, 1e30))


def block_attention(q, k, v, out, counts, blocks, offset, lse, flags, mass=None, threshold=60.0, stages=2):
    """Exact attention over a list of 64-token kv blocks with a fixed offset, plus per-block masses."""
    L, H, D = q.shape
    assert D == 128 and q.dtype == torch.bfloat16
    _block_kernel[(triton.cdiv(L, BM), H)](q, k, v, q, v, blocks, out, counts, blocks, offset, lse, flags,
                                          offset if mass is None else mass, blocks, q, v, offset, L,
                                          triton.cdiv(L, BK), NK=blocks.shape[-1], NW=0, RW=0, H=H, D=D, BM=BM,
                                          SCALE=LOG2E / math.sqrt(D), TH=float(threshold), LAST_LOG2=0.0,
                                          COMP=False, REFINE=False, WRITE_MASS=mass is not None, num_warps=8,
                                          num_stages=stages)


def _wide_sparse_attention(q, k, v, k_mean, v_mean, skip_bits, out, counts, blocks, offset, lse, flags, refine,
                           k_rep, v_rep, log_counts, mass, threshold):
    # the 64-wide kernel of the Wan2.1 and HunyuanVideo RTX 5090 measurements (8 warps, 1 stage)
    L, H, D = q.shape
    nb = triton.cdiv(L, BK)
    width = refine.shape[-1]
    use_refine = width > 0
    _block_kernel[(triton.cdiv(L, BM), H)](q, k, v, k_mean, v_mean, skip_bits, out, counts, blocks, offset, lse,
                                          flags, offset if mass is None else mass, refine if use_refine else blocks,
                                          k_rep if use_refine else q, v_rep if use_refine else v,
                                          log_counts if use_refine else offset, L, nb, NK=blocks.shape[-1],
                                          NW=skip_bits.shape[-1], RW=width, H=H, D=D, BM=BM,
                                          SCALE=LOG2E / math.sqrt(D), TH=float(threshold),
                                          LAST_LOG2=math.log2(L - (nb - 1) * BK), COMP=True, REFINE=use_refine,
                                          WRITE_MASS=mass is not None, num_warps=8, num_stages=1)


@triton.jit
def _group_of_four(j, rowsum, first, first_pair, third, total):
    # (a + b) + (c + d) over four consecutive 32-column pieces
    if j % 4 == 0:
        first = rowsum
    elif j % 4 == 1:
        first_pair = first + rowsum
    elif j % 4 == 2:
        third = rowsum
    else:
        total += first_pair + (third + rowsum)
    return first, first_pair, third, total


@triton.jit
def _sparse_kernel(Q, K, V, KB, VB, BITS, O, CNT, IDX, PRIOR, LSE, FLAG, MASS, RIDX, K4, V4, LC4, L, NB,
                   NK: tl.constexpr, NW: tl.constexpr, RW: tl.constexpr, NQ: tl.constexpr, H: tl.constexpr,
                   D: tl.constexpr, BM: tl.constexpr, SCALE: tl.constexpr, TH: tl.constexpr,
                   LAST_LOG2: tl.constexpr, REFINE: tl.constexpr, WRITE_MASS: tl.constexpr,
                   SKIP_PADDING: tl.constexpr):
    t, h = tl.program_id(0), tl.program_id(1)
    nq = NQ
    rr = t * BM + tl.arange(0, BM)
    dd = tl.arange(0, D)
    valid = rr < L
    qo = (rr[:, None] * H + h) * D + dd[None, :]
    q = tl.load(Q + qo, mask=valid[:, None], other=0)
    prior = tl.load(PRIOR + rr * H + h)
    acc = tl.zeros((BM, D), tl.float32)
    total = tl.zeros((BM,), tl.float32)
    cc = tl.arange(0, 32)

    # listed blocks, exactly: two 32-token halves per 64-token block
    n = tl.load(CNT + t * H + h)
    half = tl.zeros((BM,), tl.float32)
    for j in range(0, 2 * n):
        b = tl.load(IDX + (t * H + h) * NK + j // 2)
        row = b * 64 + (j % 2) * 32 + cc
        ko = (row[:, None] * H + h) * D + dd[None, :]
        k = tl.load(K + ko, mask=(row < L)[:, None], other=0)
        s = tl.dot(q, tl.trans(k)) * SCALE - prior[:, None]
        s = tl.where(valid[:, None] & (row < L)[None, :], s, -float('inf'))
        p = tl.exp2(s)
        rowsum = tl.sum(p, 1)
        if j % 2 == 0:
            half = rowsum
        else:
            block = half + rowsum
            total += block
            if WRITE_MASS:
                tl.store(MASS + (h * nq + t) * NB + b, tl.sum(block, 0))
        v = tl.load(V + ko, mask=(row < L)[:, None], other=0)
        acc = tl.dot(p.to(tl.bfloat16), v, acc)

    # every other skipped block through its mean key and value, weighted by its token count
    first = tl.zeros((BM,), tl.float32)
    first_pair = tl.zeros((BM,), tl.float32)
    third = tl.zeros((BM,), tl.float32)
    pieces = ((NB + 127) // 128) * 4
    for j in range(0, pieces - 1 if SKIP_PADDING else pieces):
        b = j * 32 + cc
        bo = (b[:, None] * H + h) * D + dd[None, :]
        kb = tl.load(KB + bo, mask=(b < NB)[:, None], other=0)
        vb = tl.load(VB + bo, mask=(b < NB)[:, None], other=0)
        word = tl.load(BITS + (h * nq + t) * NW + b // 32, mask=b < NB, other=-1)
        skip = ((word >> (b % 32)) & 1) != 0
        count_log = tl.where(b == NB - 1, LAST_LOG2, 6.0)
        s = tl.dot(q, tl.trans(kb)) * SCALE + count_log[None, :] - prior[:, None]
        s = tl.where(valid[:, None] & (b < NB)[None, :] & (~skip)[None, :], s, -float('inf'))
        p = tl.exp2(s)
        first, first_pair, third, total = _group_of_four(j, tl.sum(p, 1), first, first_pair, third, total)
        acc = tl.dot(p.to(tl.bfloat16), vb, acc)
    if SKIP_PADDING:                 # the last piece lies past NB: its row sum is zero
        total += first_pair + (third + tl.zeros((BM,), tl.float32))

    # refined blocks through four representatives each; 32 representatives (8 list entries) per piece
    if REFINE:
        first = tl.zeros((BM,), tl.float32)
        first_pair = tl.zeros((BM,), tl.float32)
        third = tl.zeros((BM,), tl.float32)
        for j in range(0, RW // 8 - 1 if SKIP_PADDING else RW // 8):
            b = tl.load(RIDX + (t * H + h) * RW + j * 8 + cc // 4)
            c = b * 4 + cc % 4
            co = (h * ((NB + 1) * 4) + c[:, None]) * D + dd[None, :]
            kc = tl.load(K4 + co)
            vc = tl.load(V4 + co)
            lc = tl.load(LC4 + h * ((NB + 1) * 4) + c)
            s = tl.dot(q, tl.trans(kc)) * SCALE + lc[None, :] - prior[:, None]
            s = tl.where(valid[:, None], s, -float('inf'))
            p = tl.exp2(s)
            first, first_pair, third, total = _group_of_four(j, tl.sum(p, 1), first, first_pair, third, total)
            acc = tl.dot(p.to(tl.bfloat16), vc, acc)
        if SKIP_PADDING:             # the last eight list entries are the empty sentinel block
            total += first_pair + (third + tl.zeros((BM,), tl.float32))

    lg = tl.log2(total)
    bad = tl.max((valid & ~((lg >= -TH) & (lg <= TH))).to(tl.int32), 0)
    tl.store(FLAG + t * H + h, bad)
    tl.store(O + qo, (acc / total[:, None]).to(tl.bfloat16), mask=valid[:, None])
    tl.store(LSE + rr * H + h, tl.where(valid, prior + lg, 1e30))


def sparse_attention(q, k, v, k_mean, v_mean, skip_bits, out, counts, blocks, offset, lse, flags, refine, k_rep,
                     v_rep, log_counts, mass=None, threshold=60.0, rows=None, refined=None):
    """Listed blocks exactly, the other skipped blocks through block means and the refined blocks through four
    representatives each, in one softmax at the fixed offset.

    `rows`: the caller only uses query rows < rows (it overwrites the others), so query tiles past them are not
    launched; their output is zero, their LSE the offset and their flag 0. `refined`: the number of refine-list
    entries before the sentinel padding. Only for the MiniMax-H3 shape at density 0.24 (598 kv blocks, 151 refined
    blocks in a 160-entry list) with `rows` given, the kernel leaves out the last block-mean piece (columns
    608-639, all past the 598 blocks) and the last eight refine entries (all sentinel); other shapes and
    densities run the complete lists. Without `rows` (Wan2.1, HunyuanVideo) the 64-wide kernel runs instead.
    """
    L, H, D = q.shape
    assert D == 128 and q.dtype == torch.bfloat16
    if rows is None:
        return _wide_sparse_attention(q, k, v, k_mean, v_mean, skip_bits, out, counts, blocks, offset, lse, flags,
                                      refine, k_rep, v_rep, log_counts, mass, threshold)
    nq, nb = triton.cdiv(L, BM), triton.cdiv(L, BK)
    width = refine.shape[-1]
    use_refine = width > 0
    tiles = nq if rows is None else triton.cdiv(rows, BM)
    if tiles < nq:
        flags[tiles:].zero_()
        out[tiles * BM:].zero_()
        lse[tiles * BM:].copy_(offset[tiles * BM:])
    skip_padding = rows is not None and (nb, refined, width) == (598, 151, 160)
    _sparse_kernel[(tiles, H)](q, k, v, k_mean, v_mean, skip_bits, out, counts, blocks, offset, lse, flags,
                               offset if mass is None else mass, refine if use_refine else blocks,
                               k_rep if use_refine else q, v_rep if use_refine else v,
                               log_counts if use_refine else offset, L, nb, NK=blocks.shape[-1],
                               NW=skip_bits.shape[-1], RW=width, NQ=nq, H=H, D=D, BM=BM,
                               SCALE=LOG2E / math.sqrt(D), TH=float(threshold),
                               LAST_LOG2=math.log2(L - (nb - 1) * BK), REFINE=use_refine,
                               WRITE_MASS=mass is not None, SKIP_PADDING=skip_padding, num_warps=4, num_stages=2)


@triton.jit
def _fallback_kernel(Q, K, V, O, CNT, IDX, LSE, FLAG, L, NQ: tl.constexpr, NK: tl.constexpr, H: tl.constexpr,
                     D: tl.constexpr, BM: tl.constexpr, SCALE: tl.constexpr):
    # grid-stride over (tile, head): cheap when no tile is flagged, which is the usual case
    for item in range(tl.program_id(0), NQ * H, tl.num_programs(0)):
        t = item % NQ
        h = item // NQ
        if tl.load(FLAG + t * H + h) != 0:
            rr = t * BM + tl.arange(0, BM)
            dd = tl.arange(0, D)
            qo = (rr[:, None] * H + h) * D + dd[None, :]
            q = tl.load(Q + qo, mask=(rr < L)[:, None], other=0)
            n = tl.load(CNT + t * H + h)
            maximum = tl.full((BM,), -float('inf'), tl.float32)
            total = tl.zeros((BM,), tl.float32)
            acc = tl.zeros((BM, D), tl.float32)
            cc = tl.arange(0, 64)
            for j in range(0, n):
                b = tl.load(IDX + (t * H + h) * NK + j)
                kr = b * 64 + cc
                ko = (kr[:, None] * H + h) * D + dd[None, :]
                k = tl.load(K + ko, mask=(kr < L)[:, None], other=0)
                v = tl.load(V + ko, mask=(kr < L)[:, None], other=0)
                s = tl.dot(q, tl.trans(k)) * SCALE
                s = tl.where((kr < L)[None, :], s, -float('inf'))
                new_max = tl.maximum(maximum, tl.max(s, 1))
                alpha = tl.exp2(maximum - new_max)
                p = tl.exp2(s - new_max[:, None])
                total = total * alpha + tl.sum(p, 1)
                acc = acc * alpha[:, None]
                acc = tl.dot(p.to(tl.bfloat16), v, acc)
                maximum = new_max
            tl.store(O + qo, (acc / total[:, None]).to(tl.bfloat16), mask=(rr < L)[:, None])
            tl.store(LSE + rr * H + h, tl.where(rr < L, maximum + tl.log2(total), 1e30))


def exact_fallback(q, k, v, out, counts, blocks, lse, flags, per_tile=False):
    """Online-softmax attention over the block list, for the tiles whose fixed offset was too far off.
    `per_tile`: one program per (tile, head), one stage, instead of a persistent grid of 1184 programs."""
    L, H, D = q.shape
    nq = triton.cdiv(L, BM)
    _fallback_kernel[(nq * H if per_tile else min(1184, nq * H),)](q, k, v, out, counts, blocks, lse, flags, L,
                                                                   NQ=nq, NK=blocks.shape[-1], H=H, D=D, BM=BM,
                                                                   SCALE=LOG2E / math.sqrt(D), num_warps=8,
                                                                   num_stages=1 if per_tile else 2)


@triton.jit
def _estimate_kernel(q_ptr, km_ptr, keep_ptr, m_ptr, mnew_ptr, map_ptr, scale_exact_ptr, scale_skip_ptr, scale_log2,
                     L, NB, lb_last, NQ, H: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, BC: tl.constexpr,
                     LB: tl.constexpr, NS: tl.constexpr, TPP: tl.constexpr):
    # TPP consecutive query tiles per program share the block-mean loads; NS rows sampled evenly from each tile
    t0 = tl.program_id(0) * TPP
    h = tl.program_id(1)
    nq = NQ
    R: tl.constexpr = TPP * NS
    ri = tl.arange(0, R)
    tt = t0 + ri // NS
    r = tt * BM + (ri % NS) * (BM // NS)
    rv = (r < L) & (tt < nq)
    d = tl.arange(0, D)
    q = tl.load(q_ptr + (r[:, None].to(tl.int64) * H + h) * D + d[None, :], mask=rv[:, None], other=0.0)
    ms = tl.load(m_ptr + r * H + h, mask=rv, other=0.0)
    tq = t0 + tl.arange(0, TPP)
    tv = tq < nq
    nsv = tl.sum(tl.reshape(rv.to(tl.float32), [TPP, NS]), 1)
    num = tl.zeros([TPP, BC], dtype=tl.float32)
    den = tl.zeros([TPP, BC], dtype=tl.float32)
    for c0 in range(0, NB, BC):
        cols = c0 + tl.arange(0, BC)
        cv = cols < NB
        km = tl.load(km_ptr + (cols[:, None] * H + h) * D + d[None, :], mask=cv[:, None], other=0.0)
        off = (h * nq + tq[:, None]) * NB + cols[None, :]
        ok = tv[:, None] & cv[None, :]
        kp = tl.load(keep_ptr + off, mask=ok, other=1)
        lb = tl.where(cols == NB - 1, lb_last, LB)
        s = tl.dot(q, km.T) * scale_log2
        e = tl.math.exp2(tl.minimum(s - ms[:, None] + lb[None, :], 100.0))
        e = tl.where(rv[:, None], e, 0.0)
        est = tl.sum(tl.reshape(e, [TPP, NS, BC]), 1) / tl.maximum(nsv, 1.0)[:, None]
        exact = ok & (kp != 0)
        num += tl.where(exact, tl.load(map_ptr + off, mask=exact, other=0.0), 0.0)
        den += tl.where(exact, est, 0.0)
        tl.store(map_ptr + off, est, mask=ok & (kp == 0))
    num_all = tl.sum(num, 1)
    den_all = tl.sum(den, 1)
    for i in tl.static_range(TPP):
        t = t0 + i
        if t < nq:
            nrows = tl.minimum(L - t * BM, BM)
            rows = t * BM + tl.arange(0, BM)
            okr = rows < L
            m = tl.load(m_ptr + rows * H + h)
            mnew = tl.load(mnew_ptr + rows * H + h)
            f = tl.sum(tl.where(okr, tl.math.exp2(tl.minimum(tl.maximum(m - mnew, -100.0), 100.0)), 0.0)) \
                / nrows.to(tl.float32)
            tl.store(scale_exact_ptr + h * nq + t, f / nrows.to(tl.float32))
            # calibration of the block-mean estimates on the blocks computed exactly, clamped to [1, 64]
            nn = tl.sum(tl.where(tl.arange(0, TPP) == i, num_all, 0.0)) / nrows.to(tl.float32)
            dd = tl.sum(tl.where(tl.arange(0, TPP) == i, den_all, 0.0))
            c = tl.where(dd > 0, tl.minimum(tl.maximum(nn / tl.maximum(dd, 1e-30), 1.0), 64.0), 1.0)
            tl.store(scale_skip_ptr + h * nq + t, f * c)


def estimate_skipped(q, k_mean, keep, offset, lse, mass, scale_exact, scale_skip, tiles_per_program=4):
    """Mass estimates of the skipped blocks for the next step's selection, and the per-tile scales; with
    tiles_per_program=1 the shared one-tile kernel, which gives the same values."""
    if tiles_per_program == 1:
        return triton_ops.estimate_skipped(q, k_mean, keep, offset, lse, mass, scale_exact, scale_skip)
    L, H, D = q.shape
    nq = triton.cdiv(L, BM)
    nb = k_mean.shape[0]
    tpp, ns = 4, 16
    _estimate_kernel[(triton.cdiv(nq, tpp), H)](q, k_mean, keep, offset, lse, mass, scale_exact, scale_skip,
                                                1.0 / math.sqrt(D) * LOG2E, L, nb, math.log2(L - (nb - 1) * BK), nq,
                                                H=H, D=D, BM=BM, BC=128, LB=math.log2(BK), NS=ns, TPP=tpp,
                                                num_warps=8)


@triton.jit
def _kmeans_kernel(k_ptr, v_ptr, m_ptr, p_ptr, k4_ptr, v4_ptr, c4_ptr, km_ptr, vm_ptr, L, NB, NB4, H: tl.constexpr,
                     D: tl.constexpr, BK: tl.constexpr, ITERS: tl.constexpr, NP: tl.constexpr, WRITE_MEANS: tl.constexpr):
    """The measured GB=1 Lloyd kernel; H3 also writes block means and reloads K after Lloyd."""
    GB: tl.constexpr = 1
    j0 = tl.program_id(0) * GB
    h = tl.program_id(1)
    R: tl.constexpr = GB * BK
    C: tl.constexpr = 16 if 4 * GB < 16 else 4 * GB
    tok = tl.arange(0, R)
    rows = j0 * BK + tok
    valid = rows < L
    cols = tl.arange(0, D)
    off = (rows[:, None].to(tl.int64) * H + h) * D + cols[None, :]
    kk = tl.load(k_ptr + off, mask=valid[:, None], other=0.0)                 # bf16 [R, D]
    if WRITE_MEANS:
        n = tl.minimum(L - j0 * BK, BK).to(tl.float32)
        mean = tl.zeros((D,), tl.float32)
        mean += tl.sum(kk.to(tl.float32), 0)
        tl.store(km_ptr + (j0 * H + h) * D + cols, (mean / n).to(tl.bfloat16))
    mh = tl.load(m_ptr + h * D * D + cols[:, None] * D + cols[None, :])        # bf16 [D, D]
    X = tl.dot(kk, mh)                                                         # query-metric coordinates, fp32
    # token weights: each probe's softmax over each block's tokens, summed over the probes, x valid / NP
    pr = tl.load(p_ptr + (h * NP + tl.arange(0, NP))[:, None] * D + cols[None, :])            # bf16 [NP, D]
    sp = tl.dot(pr, tl.trans(kk))                                                              # [NP, R]
    sp = tl.where(valid[None, :], sp, float("-inf"))
    w = tl.zeros((R,), dtype=tl.float32)
    for b in tl.static_range(GB):
        inb = (tok // BK) == b
        sb = tl.where(inb[None, :], sp, float("-inf"))
        e = tl.exp(sb - tl.max(sb, 1)[:, None])
        e = e / tl.sum(e, 1)[:, None]
        nval = tl.sum((valid & inb).to(tl.float32), 0)
        w += tl.where(inb, tl.sum(e, 0) * (nval / NP), 0.0)
    w = tl.where(valid, w, 0.0)
    cl = tl.arange(0, C)
    own = ((cl[None, :] // 4) == (tok // BK)[:, None]) & (cl < 4 * GB)[None, :]
    lab = (tok // (BK // 4)).to(tl.int32)
    oh = ((lab[:, None] == cl[None, :]) & valid[:, None]).to(tl.float32)
    ohw = oh * w[:, None]
    cw = tl.sum(ohw, 0)
    cen = tl.dot(tl.trans(ohw), X) / tl.maximum(cw, 1e-6)[:, None]             # [C, D]
    it = 0
    changed = 1
    while (it < ITERS) & (changed > 0):
        dist = tl.sum(cen * cen, 1)[None, :] - 2.0 * tl.dot(X, tl.trans(cen))
        dist = tl.where(own & (cw > 0)[None, :], dist, float("inf"))
        new = tl.argmin(dist, 1).to(tl.int32)
        changed = tl.sum(((new != lab) & valid).to(tl.int32))
        lab = new
        oh = ((lab[:, None] == cl[None, :]) & valid[:, None]).to(tl.float32)
        ohw = oh * w[:, None]
        cw2 = tl.sum(ohw, 0)
        cen = tl.where((cw2 > 0)[:, None], tl.dot(tl.trans(ohw), X) / tl.maximum(cw2, 1e-6)[:, None], cen)
        cw = cw2
        it += 1
    cnt = tl.sum(oh, 0)
    inv = 1.0 / tl.maximum(cnt, 1.0)
    vv = tl.load(v_ptr + off, mask=valid[:, None], other=0.0)
    if WRITE_MEANS:
        mean = tl.zeros((D,), tl.float32)
        mean += tl.sum(vv.to(tl.float32), 0)
        tl.store(vm_ptr + (j0 * H + h) * D + cols, (mean / n).to(tl.bfloat16))
        kk = tl.load(k_ptr + off, mask=valid[:, None], other=0.0, volatile=True)
    ohb = oh.to(tl.bfloat16)
    ck = tl.dot(tl.trans(ohb), kk) * inv[:, None]
    cv = tl.dot(tl.trans(ohb), vv) * inv[:, None]
    ok = (cl < 4 * GB) & (j0 + cl // 4 < NB)
    o = (h * NB4 + j0 * 4 + cl[:, None]).to(tl.int64) * D + cols[None, :]
    tl.store(k4_ptr + o, ck.to(tl.bfloat16), mask=ok[:, None])
    tl.store(v4_ptr + o, cv.to(tl.bfloat16), mask=ok[:, None])
    lc = tl.where(cnt > 0, tl.log2(tl.maximum(cnt, 1.0)), float("-inf"))
    tl.store(c4_ptr + h * NB4 + j0 * 4 + cl, lc, mask=ok)


class Representatives(representatives.Representatives):
    """representatives.Representatives; given k_mean / v_mean, `build` also writes the block means (nb, H, D)."""

    def build(self, q, k, v, k_mean=None, v_mean=None):
        self._metric_and_probes(q)
        means = k_mean is not None
        _kmeans_kernel[(self.nb, self.H)](k, v, self._metric, self._probes, self.k, self.v, self.log_counts,
                                          k if k_mean is None else k_mean, v if v_mean is None else v_mean,
                                          self.L, self.nb, self.k.shape[1], H=self.H, D=self.D, BK=BK,
                                          ITERS=self.iters, NP=representatives.NUM_PROBES, WRITE_MEANS=means,
                                          num_warps=4, num_stages=1 if means else 3)


# Compact plan state (Wan2.1, HunyuanVideo): the selection for the next step is made right after this step's
# estimates and kept as two bit masks per (tile, head): the exactly computed blocks, and the blocks the block-mean
# compensation leaves out (exact and refined; `select_blocks`' skip bits). The keep mask, block list and refine list
# are rebuilt from them at the next step, so the block-probability map, the lists and the scales are shared scratch
# instead of per-layer state.

@triton.jit
def _pack_kernel(KEEP, EXACT, NB: tl.constexpr, NQ: tl.constexpr, NW: tl.constexpr, NBP: tl.constexpr):
    t, h = tl.program_id(0), tl.program_id(1)
    j = tl.arange(0, NBP)
    keep = tl.load(KEEP + (h * NQ + t) * NB + j, mask=j < NB, other=1)
    bit = ((keep != 0) | (j >= NB)).to(tl.int32)
    words = tl.sum(tl.reshape(bit, [NBP // 32, 32]) << tl.arange(0, 32)[None, :], 1)
    tl.store(EXACT + (h * NQ + t) * NW + tl.arange(0, NBP // 32), words)


def pack_plan(keep, exact_bits):
    """Bit mask of the exactly computed blocks: keep (H, nq, nb) uint8 -> exact_bits (H, nq, NW) int32."""
    H, nq, nb = keep.shape
    nbp = triton.next_power_of_2(nb)
    _pack_kernel[(nq, H)](keep, exact_bits, NB=nb, NQ=nq, NW=exact_bits.shape[-1], NBP=nbp,
                          num_warps=4 if nbp <= 1024 else 8)


@triton.jit
def _rebuild_kernel(EXACT, SKIP, KEEP, CNT, IDX, RIDX, NB: tl.constexpr, NQ: tl.constexpr, H: tl.constexpr,
                    NW: tl.constexpr, RW: tl.constexpr, NBP: tl.constexpr):
    t, h = tl.program_id(0), tl.program_id(1)
    j = tl.arange(0, NBP)
    off = (h * NQ + t) * NW + j // 32
    exact = (((tl.load(EXACT + off) >> (j % 32)) & 1) != 0) & (j < NB)
    refined = (((tl.load(SKIP + off) >> (j % 32)) & 1) != 0) & ~exact & (j < NB)
    pos = tl.cumsum(exact.to(tl.int32), 0) - 1
    tl.store(IDX + (t * H + h) * NB + pos, j, mask=exact)
    tl.store(CNT + t * H + h, tl.sum(exact.to(tl.int32)))
    tl.store(KEEP + (h * NQ + t) * NB + j, exact.to(tl.uint8), mask=j < NB)
    rpos = tl.cumsum(refined.to(tl.int32), 0) - 1
    nref = tl.sum(refined.to(tl.int32))
    tl.store(RIDX + (t * H + h) * RW + j, tl.full([NBP], NB, tl.int32), mask=(j >= nref) & (j < RW))
    tl.store(RIDX + (t * H + h) * RW + rpos, j, mask=refined & (rpos < RW))


def rebuild_plan(exact_bits, skip_bits, keep, counts, blocks, refine):
    """keep mask, sorted block list and refine list (padded with nb) from the two bit masks of a plan."""
    H, nq, nb = keep.shape
    _rebuild_kernel[(nq, H)](exact_bits, skip_bits, keep, counts, blocks, refine, NB=nb, NQ=nq, H=H,
                             NW=exact_bits.shape[-1], RW=refine.shape[-1], NBP=triton.next_power_of_2(nb),
                             num_warps=4 if nb <= 1024 else 8)


@triton.jit
def _bf16(x):
    return tl.inline_asm_elementwise('cvt.rn.bf16.f32 $0, $1;', constraints='=h,f', args=[x], dtype=tl.bfloat16,
                                     is_pure=True, pack=1)


@triton.jit
def _mul(a, b):
    return tl.inline_asm_elementwise('mul.rn.f32 $0, $1, $2;', constraints='=f,f,f', args=[a, b], dtype=tl.float32,
                                     is_pure=True, pack=1)


@triton.jit
def _h3_rotary_kernel(X, COS, SIN, OUT, N, L, H: tl.constexpr, XB, XL, XH, XD, CL, CD, SL, SD,
                      D: tl.constexpr, ROT: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    valid = i < N
    d = i % D
    head = i // D % H
    seq = i // (D * H) % L
    batch = i // (D * H * L)
    base = batch * XB + seq * XL + head * XH
    native = tl.load(X + base + d * XD, mask=valid, other=0)
    rot = valid & (d < ROT)
    partner = tl.load(X + base + ((d + ROT // 2) % ROT) * XD, mask=rot, other=0).to(tl.float32)
    negated = (partner.to(tl.uint32, bitcast=True) ^ 0x80000000).to(tl.float32, bitcast=True)
    rotated = tl.where(d < ROT // 2, negated, partner)
    c = _bf16(tl.load(COS + seq * CL + d * CD, mask=rot, other=0)).to(tl.float32)
    s = _bf16(tl.load(SIN + seq * SL + d * SD, mask=rot, other=0)).to(tl.float32)
    # as diffusers' H3 rotary embedding: cos / sin and both products rounded to BF16 before the BF16 addition
    summed = tl.inline_asm_elementwise('add.rn.f32 $0, $1, $2;', constraints='=f,f,f',
                                       args=[_bf16(_mul(native.to(tl.float32), c)).to(tl.float32),
                                             _bf16(_mul(rotated, s)).to(tl.float32)],
                                       dtype=tl.float32, is_pure=True, pack=1)
    tl.store(OUT + i, tl.where(d < ROT, _bf16(summed), native), mask=valid)


def h3_rotary(x, cos, sin):
    """MiniMax-H3's rotary embedding (`_apply_rotary_emb`) in one kernel: x (B, L, H, 128) BF16, cos / sin (L, 96)
    FP32; the first 96 channels of every head are rotated (half-split), the other 32 pass through."""
    B, L, H, D = x.shape
    out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    _h3_rotary_kernel[(triton.cdiv(x.numel(), 1024),)](x, cos, sin, out, x.numel(), L, H, *x.stride(), *cos.stride(),
                                                       *sin.stride(), D=D, ROT=cos.shape[-1], BLOCK=1024,
                                                       num_warps=4, enable_fp_fusion=False)
    return out
