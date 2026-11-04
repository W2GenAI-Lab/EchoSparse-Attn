"""RTX 5090 (SM120) kernel set, in Triton.

SM120 has no tcgen05 MMAs or tensor memory, so the attention kernel is written with ordinary loads and tensor-core
dots. It computes the same quantities as the B200 kernels, in the same units: 128-row query tiles, 64-token kv
blocks, the previous step's log2 LSE as a fixed softmax offset, block masses summed over the tile's rows, BF16
operands and FP32 accumulation. The compensation columns come from compact per-tile lists that the plan rebuild
writes: the block ids left to their means (32 per batch) and the refined block ids (8 per batch, 32 representative
columns).

The selection, compensation and Gram / Cholesky / probe kernels are shared with the B200 set.
"""
import math

import torch
import triton
import triton.language as tl

from .representatives import Representatives  # noqa: F401
from .triton_ops import compensate_fallback, full_step_scales, select_blocks  # noqa: F401

BM = 128
BK = 64
LOG2E = 1.4426950408889634


@triton.jit
def _attention_kernel(Q, K, V, KB, VB, O, CNT, IDX, PRIOR, LSE, FLAG, MASS, MIDX, MCNT, RIDX, RCNT, K4, V4, LC4, L,
                      NB, NK: tl.constexpr, RW: tl.constexpr, NQ: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                      BM: tl.constexpr, SCALE: tl.constexpr, TH: tl.constexpr, LAST_LOG2: tl.constexpr,
                      COMP: tl.constexpr, WRITE_MASS: tl.constexpr):
    # The listed 64-token blocks exactly, as two 32-key halves; with COMP the listed mean blocks through their mean
    # key and value and the listed refined blocks through four representatives each, 32 columns at a time.
    t, h = tl.program_id(0), tl.program_id(1)
    rr = t * BM + tl.arange(0, BM)
    dd = tl.arange(0, D)
    valid = rr < L
    qo = (rr[:, None] * H + h) * D + dd[None, :]
    q = tl.load(Q + qo, mask=valid[:, None], other=0)
    prior = tl.load(PRIOR + rr * H + h)
    acc = tl.zeros((BM, D), tl.float32)
    total = tl.zeros((BM,), tl.float32)
    cc = tl.arange(0, 32)

    n = tl.load(CNT + t * H + h)
    block = tl.zeros((BM,), tl.float32)
    for j in range(0, 2 * n):
        b = tl.load(IDX + (t * H + h) * NK + j // 2)
        row = b * 64 + (j % 2) * 32 + cc
        ko = (row[:, None] * H + h) * D + dd[None, :]
        k = tl.load(K + ko, mask=(row < L)[:, None], other=0)
        s = tl.dot(q, tl.trans(k)) * SCALE - prior[:, None]
        s = tl.where(valid[:, None] & (row < L)[None, :], s, -float('inf'))
        p = tl.exp2(s)
        block += tl.sum(p, 1)
        v = tl.load(V + ko, mask=(row < L)[:, None], other=0)
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        if j % 2 == 1:                              # both halves done: the block's row sums are complete
            total += block
            if WRITE_MASS:
                tl.store(MASS + (h * NQ + t) * NB + b, tl.sum(block, 0))
            block = tl.zeros((BM,), tl.float32)

    if COMP:
        nm = tl.load(MCNT + t * H + h)
        for c0 in range(0, nm, 32):
            colv = c0 + cc < nm
            b = tl.load(MIDX + (t * H + h) * NK + c0 + cc, mask=colv, other=NB)
            bo = (b[:, None] * H + h) * D + dd[None, :]
            kb = tl.load(KB + bo, mask=colv[:, None] & (b < NB)[:, None], other=0)
            vb = tl.load(VB + bo, mask=colv[:, None] & (b < NB)[:, None], other=0)
            count_log = tl.where(b == NB - 1, LAST_LOG2, 6.0)
            s = tl.dot(q, tl.trans(kb)) * SCALE + count_log[None, :] - prior[:, None]
            s = tl.where(valid[:, None] & colv[None, :], s, -float('inf'))
            p = tl.exp2(s)
            total += tl.sum(p, 1)
            acc = tl.dot(p.to(tl.bfloat16), vb, acc)
        nr = tl.load(RCNT + t * H + h)
        for r0 in range(0, nr, 8):
            colv = r0 + cc // 4 < nr
            b = tl.load(RIDX + (t * H + h) * RW + r0 + cc // 4, mask=colv, other=NB)
            c = b * 4 + cc % 4
            co = (h * ((NB + 1) * 4) + c[:, None]) * D + dd[None, :]
            kc = tl.load(K4 + co, mask=colv[:, None], other=0)
            vc = tl.load(V4 + co, mask=colv[:, None], other=0)
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
                               triton.cdiv(L, BK), NK=blocks.shape[-1], RW=0, NQ=nq, H=H, D=D, BM=BM,
                               SCALE=LOG2E / math.sqrt(D), TH=float(threshold), LAST_LOG2=0.0, COMP=False,
                               WRITE_MASS=mass is not None, num_warps=4, num_stages=2)


def launched_tiles(out, lse, flags, offset, rows):
    """Query tiles to launch: the caller only uses query rows < rows (it overwrites the others), so the tiles past
    them are not launched; their output is zero, their LSE the offset and their flag 0."""
    nq = triton.cdiv(out.shape[0], BM)
    tiles = nq if rows is None else triton.cdiv(rows, BM)
    if tiles < nq:
        flags[tiles:].zero_()
        out[tiles * BM:].zero_()
        lse[tiles * BM:].copy_(offset[tiles * BM:])
    return tiles


def sparse_attention(q, k, v, k_mean, v_mean, means, mean_counts, out, counts, blocks, offset, lse, flags, refine,
                     refine_counts, k_rep, v_rep, log_counts, mass=None, threshold=60.0, rows=None):
    """Listed blocks exactly, the listed mean blocks through block means and the listed refined blocks through four
    representatives each, in one softmax at the fixed offset. The lists are `rebuild_plan`'s; `rows`: see
    `launched_tiles`.
    """
    L, H, D = q.shape
    assert D == 128 and q.dtype == torch.bfloat16
    nq, nb = triton.cdiv(L, BM), triton.cdiv(L, BK)
    tiles = launched_tiles(out, lse, flags, offset, rows)
    if tiles == 0:
        return
    _attention_kernel[(tiles, H)](q, k, v, k_mean, v_mean, out, counts, blocks, offset, lse, flags,
                                  offset if mass is None else mass, means, mean_counts, refine, refine_counts,
                                  k_rep, v_rep, log_counts, L, nb, NK=blocks.shape[-1], RW=refine.shape[-1], NQ=nq,
                                  H=H, D=D, BM=BM, SCALE=LOG2E / math.sqrt(D), TH=float(threshold),
                                  LAST_LOG2=math.log2(L - (nb - 1) * BK), COMP=True, WRITE_MASS=mass is not None,
                                  num_warps=4, num_stages=2)


@triton.jit
def _fallback_kernel(Q, K, V, O, CNT, IDX, LSE, FLAG, L, NK: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                     BM: tl.constexpr, SCALE: tl.constexpr):
    t, h = tl.program_id(0), tl.program_id(1)
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


def exact_fallback(q, k, v, out, counts, blocks, lse, flags):
    """Online-softmax attention over the block list, for the tiles whose fixed offset was too far off."""
    L, H, D = q.shape
    _fallback_kernel[(triton.cdiv(L, BM), H)](q, k, v, out, counts, blocks, lse, flags, L, NK=blocks.shape[-1], H=H,
                                              D=D, BM=BM, SCALE=LOG2E / math.sqrt(D), num_warps=8, num_stages=1)


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


def estimate_skipped(q, k_mean, keep, offset, lse, mass, scale_exact, scale_skip):
    """Mass estimates of the skipped blocks for the next step's selection, and the per-tile scales: the values of
    `triton_ops.estimate_skipped`, four query tiles per program."""
    L, H, D = q.shape
    nq = triton.cdiv(L, BM)
    nb = k_mean.shape[0]
    tpp, ns = 4, 16
    _estimate_kernel[(triton.cdiv(nq, tpp), H)](q, k_mean, keep, offset, lse, mass, scale_exact, scale_skip,
                                                1.0 / math.sqrt(D) * LOG2E, L, nb, math.log2(L - (nb - 1) * BK), nq,
                                                H=H, D=D, BM=BM, BC=128, LB=math.log2(BK), NS=ns, TPP=tpp,
                                                num_warps=8)


# Plans: the selection for the next step is made right after this step's estimates and kept as two bit masks per
# (tile, head): the exactly computed blocks, and the blocks the block-mean compensation leaves out (exact and refined;
# `select_blocks`' skip bits). The keep mask and the block, mean and refine lists are rebuilt from them at the next
# step, so the block-probability map, the lists and the scales are shared scratch instead of per-layer state.

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
def _rebuild_kernel(EXACT, SKIP, KEEP, CNT, IDX, RIDX, RCNT, MIDX, MCNT, NB: tl.constexpr, NQ: tl.constexpr,
                    H: tl.constexpr, NW: tl.constexpr, RW: tl.constexpr, NBP: tl.constexpr):
    t, h = tl.program_id(0), tl.program_id(1)
    j = tl.arange(0, NBP)
    off = (h * NQ + t) * NW + j // 32
    exact = (((tl.load(EXACT + off) >> (j % 32)) & 1) != 0) & (j < NB)
    skip = ((tl.load(SKIP + off) >> (j % 32)) & 1) != 0
    refined = skip & ~exact & (j < NB)
    mean = ~skip & (j < NB)
    tl.store(MIDX + (t * H + h) * NB + tl.cumsum(mean.to(tl.int32), 0) - 1, j, mask=mean)
    tl.store(MCNT + t * H + h, tl.sum(mean.to(tl.int32)))
    pos = tl.cumsum(exact.to(tl.int32), 0) - 1
    tl.store(IDX + (t * H + h) * NB + pos, j, mask=exact)
    tl.store(CNT + t * H + h, tl.sum(exact.to(tl.int32)))
    tl.store(KEEP + (h * NQ + t) * NB + j, exact.to(tl.uint8), mask=j < NB)
    rpos = tl.cumsum(refined.to(tl.int32), 0) - 1
    nref = tl.sum(refined.to(tl.int32))
    tl.store(RIDX + (t * H + h) * RW + j, tl.full([NBP], NB, tl.int32), mask=(j >= nref) & (j < RW))
    tl.store(RIDX + (t * H + h) * RW + rpos, j, mask=refined & (rpos < RW))
    tl.store(RCNT + t * H + h, tl.minimum(nref, RW))


def rebuild_plan(exact_bits, skip_bits, keep, counts, blocks, refine, refine_counts, means, mean_counts):
    """keep mask, sorted block list, refine list (padded with nb) and sorted list of the blocks left to their means
    (`means`, shaped as `blocks`), with the lengths of the last two, from the two bit masks of a plan."""
    H, nq, nb = keep.shape
    _rebuild_kernel[(nq, H)](exact_bits, skip_bits, keep, counts, blocks, refine, refine_counts, means, mean_counts,
                             NB=nb, NQ=nq, H=H, NW=exact_bits.shape[-1], RW=refine.shape[-1],
                             NBP=triton.next_power_of_2(nb), num_warps=4 if nb <= 1024 else 8)


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
