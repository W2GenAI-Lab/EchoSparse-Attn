"""Triton kernels that run around the CuTe attention kernels.

Shapes: q / k / v are (L, H, D) bf16. Query tiles have 128 rows (nq = cdiv(L, 128)), kv blocks have 64 tokens
(nb = cdiv(L, 64)). Offsets and LSEs are log2 values stored as (nq * 128, H) float32; rows >= L hold 1e30.

The block-mass map (H, nq, nb) holds, for blocks computed exactly, the sum over the tile's rows of 2^(s - offset)
written by the attention kernel, and for skipped blocks an estimate from the block-mean keys. The per-tile scales
`scale_exact` and `scale_skip` bring both kinds of entries to probabilities when the next step selects blocks.
"""
import math

import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

BM = 128
BK = 64
LOG2E = 1.4426950408889634


@triton.jit
def _block_mean_kernel(k_ptr, v_ptr, km_ptr, vm_ptr, L, H: tl.constexpr, D: tl.constexpr, BK: tl.constexpr):
    j = tl.program_id(0)
    h = tl.program_id(1)
    c = tl.arange(0, D)
    n = tl.minimum(L - j * BK, BK).to(tl.float32)
    r = j * BK + tl.arange(0, BK)
    ok = (r < L)[:, None]
    off = (r[:, None].to(tl.int64) * H + h) * D + c[None, :]
    sk = tl.sum(tl.load(k_ptr + off, mask=ok, other=0.0).to(tl.float32), 0)
    sv = tl.sum(tl.load(v_ptr + off, mask=ok, other=0.0).to(tl.float32), 0)
    o = (j * H + h) * D + c
    tl.store(km_ptr + o, (sk / n).to(tl.bfloat16))
    tl.store(vm_ptr + o, (sv / n).to(tl.bfloat16))


def block_means(k, v, k_mean, v_mean):
    """Mean key and value of every 64-token block: (L, H, D) -> (nb, H, D)."""
    L, H, D = k.shape
    _block_mean_kernel[(triton.cdiv(L, BK), H)](k, v, k_mean, v_mean, L, H=H, D=D, BK=BK, num_warps=4)


@triton.jit
def _kth_largest(bits, K):
    # Values are >= 0 (padding is -1) and non-negative floats order like their int32 bit patterns, so a bisection
    # over the bit patterns finds the K-th largest value without sorting.
    lo = tl.zeros((), dtype=tl.int32)
    hi = tl.full((), 0x7F800001, dtype=tl.int32)
    for _ in tl.static_range(31):
        mid = lo + (hi - lo) // 2
        ok = tl.sum((bits >= mid).to(tl.int32)) >= K
        lo = tl.where(ok, mid, lo)
        hi = tl.where(ok, hi, mid)
    return lo


@triton.jit
def _select_kernel(map_ptr, keep_prev_ptr, scale_exact_ptr, scale_skip_ptr, keep_ptr, cnt_ptr, idx_ptr, flag_ptr,
                   bits_ptr, ridx_ptr, NW, K, R, RW, NB, L, SINK, H: tl.constexpr, BK: tl.constexpr, NBP: tl.constexpr,
                   BM: tl.constexpr):
    t = tl.program_id(0)
    h = tl.program_id(1)
    nq = tl.num_programs(0)
    j = tl.arange(0, NBP)
    valid = j < NB
    base = (h * nq + t) * NB
    kept_before = tl.load(keep_prev_ptr + base + j, mask=valid, other=0) != 0
    # the blocks holding the tile's own tokens, one more on each side, and the sink are always computed
    lo = (t * BM) // BK - 1
    hi = (tl.minimum((t + 1) * BM, L) - 1) // BK + 1
    forced = ((j >= lo) & (j <= hi) | (j >= SINK)) & valid

    raw = tl.load(map_ptr + base + j, mask=valid, other=0.0)
    scale_exact = tl.load(scale_exact_ptr + h * nq + t)
    scale_skip = tl.load(scale_skip_ptr + h * nq + t)
    fell_back = tl.load(flag_ptr + t * H + h)
    # a tile that fell back to online softmax has no valid masses: keep its previous blocks first
    val = tl.where(kept_before, tl.where(fell_back != 0, float("inf"), raw * scale_exact), raw * scale_skip)
    val = tl.where(val == val, val, 0.0)
    val = tl.where(j >= SINK, float("inf"), val)
    val = tl.where(valid, val, -1.0)
    vbits = val.to(tl.int32, bitcast=True)
    thr = _kth_largest(vbits, K).to(tl.float32, bitcast=True)
    keep = valid & ((val >= thr) | forced)

    # refine list: the R skipped blocks with the largest map values, padded with NB
    r2 = tl.where(raw == raw, raw, 0.0)
    r2 = tl.where(valid & ~keep, r2, -1.0)
    rbits = r2.to(tl.int32, bitcast=True)
    ref = (rbits >= _kth_largest(rbits, R)) & (r2 >= 0.0)
    rpos = tl.cumsum(ref.to(tl.int32), 0) - 1
    ref = ref & (rpos < R)
    nref = tl.sum(ref.to(tl.int32))
    rbase = (t * H + h) * RW
    tl.store(ridx_ptr + rbase + j, tl.full([NBP], NB, dtype=tl.int32), mask=(j >= nref) & (j < RW))
    tl.store(ridx_ptr + rbase + rpos, j, mask=ref)

    pos = tl.cumsum(keep.to(tl.int32), 0) - 1
    tl.store(idx_ptr + (t * H + h) * NB + pos, j, mask=keep)
    tl.store(cnt_ptr + t * H + h, tl.sum(keep.to(tl.int32)))
    tl.store(keep_ptr + base + j, keep.to(tl.uint8), mask=valid)
    # bit set = block-mean compensation skips the block (exact, refined, or past nb)
    skip = (keep | (j >= NB) | ref).to(tl.int32)
    words = tl.sum(tl.reshape(skip, [NBP // 32, 32]) << tl.arange(0, 32)[None, :], 1)
    tl.store(bits_ptr + (h * nq + t) * NW + tl.arange(0, NBP // 32), words)


def select_blocks(mass, keep_prev, scale_exact, scale_skip, keep, counts, blocks, flags, skip_bits, refine, k, r, L,
                  sink=1 << 30):
    """Top-k blocks of every (query tile, head) by block probability, plus the forced ones.

    Writes the sorted block list (counts, blocks), the keep mask, the refine list (the top-r skipped blocks) and
    the bit mask of blocks the block-mean compensation leaves out. `sink`: first block every tile computes."""
    H, nq, nb = mass.shape
    nbp = triton.next_power_of_2(nb)
    assert refine.shape[-1] >= r and refine.shape[-1] <= nbp
    _select_kernel[(nq, H)](mass, keep_prev, scale_exact, scale_skip, keep, counts, blocks, flags, skip_bits, refine,
                            skip_bits.shape[-1], k, int(r), refine.shape[-1], nb, L, sink, H=H, BK=BK, NBP=nbp, BM=BM,
                            num_warps=4 if nbp <= 1024 else 8)


@triton.jit
def _estimate_kernel(q_ptr, km_ptr, keep_ptr, m_ptr, mnew_ptr, map_ptr, scale_exact_ptr, scale_skip_ptr, scale_log2, L,
                     NB, lb_last, H: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, BC: tl.constexpr,
                     LB: tl.constexpr, NS: tl.constexpr):
    t = tl.program_id(0)
    h = tl.program_id(1)
    nq = tl.num_programs(0)
    # NS rows sampled evenly from the tile
    r = t * BM + tl.arange(0, NS) * (BM // NS)
    rv = r < L
    d = tl.arange(0, D)
    q = tl.load(q_ptr + (r[:, None].to(tl.int64) * H + h) * D + d[None, :], mask=rv[:, None], other=0.0)
    ms = tl.load(m_ptr + r * H + h, mask=rv, other=0.0)
    nsv = tl.sum(tl.reshape(rv.to(tl.float32), [1, NS]), 1)
    num = tl.zeros([1, BC], dtype=tl.float32)
    den = tl.zeros([1, BC], dtype=tl.float32)
    for c0 in range(0, NB, BC):
        cols = c0 + tl.arange(0, BC)
        cv = cols < NB
        km = tl.load(km_ptr + (cols[:, None] * H + h) * D + d[None, :], mask=cv[:, None], other=0.0)
        off = ((h * nq + t) * NB + cols)[None, :]
        ok = cv[None, :]
        kp = tl.load(keep_ptr + off, mask=ok, other=1)
        lb = tl.where(cols == NB - 1, lb_last, LB)
        s = tl.dot(q, km.T) * scale_log2
        e = tl.math.exp2(tl.minimum(s - ms[:, None] + lb[None, :], 100.0))
        e = tl.where(rv[:, None], e, 0.0)
        est = tl.sum(tl.reshape(e, [1, NS, BC]), 1) / tl.maximum(nsv, 1.0)[:, None]
        exact = ok & (kp != 0)
        num += tl.where(exact, tl.load(map_ptr + off, mask=exact, other=0.0), 0.0)
        den += tl.where(exact, est, 0.0)
        tl.store(map_ptr + off, est, mask=ok & (kp == 0))

    nrows = tl.minimum(L - t * BM, BM).to(tl.float32)
    rows = t * BM + tl.arange(0, BM)
    okr = rows < L
    m = tl.load(m_ptr + rows * H + h)
    mnew = tl.load(mnew_ptr + rows * H + h)
    f = tl.sum(tl.where(okr, tl.math.exp2(tl.minimum(tl.maximum(m - mnew, -100.0), 100.0)), 0.0)) / nrows
    tl.store(scale_exact_ptr + h * nq + t, f / nrows)
    # The block-mean estimate underestimates block mass (Jensen). Calibrate it on the blocks computed exactly in
    # this call: their exact masses over their estimates, clamped to [1, 64].
    nn = tl.sum(tl.sum(num, 1)) / nrows
    dd = tl.sum(tl.sum(den, 1))
    c = tl.where(dd > 0, tl.minimum(tl.maximum(nn / tl.maximum(dd, 1e-30), 1.0), 64.0), 1.0)
    tl.store(scale_skip_ptr + h * nq + t, f * c)


def estimate_skipped(q, k_mean, keep, offset, lse, mass, scale_exact, scale_skip):
    """Mass estimates of the skipped blocks for the next step's selection, and the per-tile scales."""
    L, H, D = q.shape
    nq = triton.cdiv(L, BM)
    nb = k_mean.shape[0]
    tail = L - (nb - 1) * BK
    _estimate_kernel[(nq, H)](q, k_mean, keep, offset, lse, mass, scale_exact, scale_skip, 1.0 / math.sqrt(D) * LOG2E,
                              L, nb, math.log2(tail), H=H, D=D, BM=BM, BC=128, LB=math.log2(BK), NS=16, num_warps=4)


@triton.jit
def _full_scale_kernel(m_ptr, lse_ptr, scale_exact_ptr, scale_skip_ptr, L, H: tl.constexpr, BM: tl.constexpr):
    t = tl.program_id(0)
    h = tl.program_id(1)
    nq = tl.num_programs(0)
    rows = t * BM + tl.arange(0, BM)
    rv = rows < L
    nrows = tl.minimum(L - t * BM, BM).to(tl.float32)
    lse = tl.load(lse_ptr + rows * H + h)
    tl.store(lse_ptr + rows * H + h, tl.where(rv, lse, 1e30))
    m = tl.load(m_ptr + rows * H + h)
    f = tl.where(rv, tl.math.exp2(tl.minimum(tl.maximum(m - lse, -100.0), 100.0)), 0.0)
    tl.store(scale_exact_ptr + h * nq + t, tl.sum(f) / nrows / nrows)
    tl.store(scale_skip_ptr + h * nq + t, 1.0)


def full_step_scales(offset, lse, scale_exact, scale_skip, L):
    """Map scales after a step that computed every block; also resets the padding rows of `lse` to 1e30."""
    H, nq = scale_exact.shape
    _full_scale_kernel[(nq, H)](offset, lse, scale_exact, scale_skip, L, H=H, BM=BM, num_warps=8)


@triton.jit
def _fallback_kernel(q_desc, k_desc, v_desc, o_ptr, cnt_ptr, idx_ptr, lse_ptr, flag_ptr, qk_scale, NK, L, NT,
                     H: tl.constexpr, HD: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    # grid-stride over (tile, head): cheap when no tile is flagged, which is the usual case
    for it in range(tl.program_id(0), NT, tl.num_programs(0)):
        qb = it // H
        h = it % H
        if tl.load(flag_ptr + qb * H + h) != 0:
            n = tl.load(cnt_ptr + qb * H + h)
            q = q_desc.load([qb * BM, h * HD])
            m_i = tl.full([BM], -float("inf"), dtype=tl.float32)
            l_i = tl.zeros([BM], dtype=tl.float32)
            acc = tl.zeros([BM, HD], dtype=tl.float32)
            lst = idx_ptr + (qb * H + h) * NK
            for i in range(0, n):
                blk = tl.load(lst + i)
                k = k_desc.load([blk * BN, h * HD])
                s = tl.dot(q, k.T)
                kcol = blk * BN + tl.arange(0, BN)
                s = tl.where((kcol < L)[None, :], s, -float("inf"))
                m_ij = tl.maximum(m_i, tl.max(s, 1) * qk_scale)
                p = tl.math.exp2(s * qk_scale - m_ij[:, None])
                alpha = tl.math.exp2(m_i - m_ij)
                l_i = l_i * alpha + tl.sum(p, 1)
                acc = acc * alpha[:, None]
                v = v_desc.load([blk * BN, h * HD])
                acc = tl.dot(p.to(v.dtype), v, acc)
                m_i = m_ij
            acc = acc / tl.where(l_i == 0.0, 1.0, l_i)[:, None]
            rows = qb * BM + tl.arange(0, BM)
            cols = h * HD + tl.arange(0, HD)
            tl.store(o_ptr + rows[:, None] * (H * HD) + cols[None, :], acc.to(tl.bfloat16), mask=(rows < L)[:, None])
            tl.store(lse_ptr + rows * H + h, tl.where(l_i > 0.0, m_i + tl.math.log2(l_i), -1e30))


def exact_fallback(q, k, v, out, counts, blocks, lse, flags):
    """Online-softmax attention over the block list, for the tiles whose fixed offset was too far off."""
    L, H, D = q.shape
    nq = triton.cdiv(L, BM)
    q2, k2, v2 = q.view(L, H * D), k.view(L, H * D), v.view(L, H * D)
    qd = TensorDescriptor(q2, shape=[L, H * D], strides=[H * D, 1], block_shape=[BM, D])
    kd = TensorDescriptor(k2, shape=[L, H * D], strides=[H * D, 1], block_shape=[BK, D])
    vd = TensorDescriptor(v2, shape=[L, H * D], strides=[H * D, 1], block_shape=[BK, D])
    nt = nq * H
    _fallback_kernel[(min(1184, nt),)](qd, kd, vd, out.view(L, H * D), counts, blocks, lse, flags,
                                       1.0 / math.sqrt(D) * LOG2E, blocks.shape[-1], L, nt, H=H, HD=D, BM=BM, BN=BK,
                                       num_warps=8, num_stages=2)


@triton.jit
def _compensate_kernel(q_ptr, km_ptr, vm_ptr, keep_ptr, lse_ptr, out_ptr, flag_ptr, scale_log2, L, NB, NQ, NT,
                       lb_last, H: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, BC: tl.constexpr,
                       LB: tl.constexpr):
    for it in range(tl.program_id(0), NT, tl.num_programs(0)):
        t = it // H
        h = it % H
        if tl.load(flag_ptr + t * H + h) != 0:
            rows = t * BM + tl.arange(0, BM)
            rv = rows < L
            d = tl.arange(0, D)
            roff = (rows[:, None].to(tl.int64) * H + h) * D + d[None, :]
            lse = tl.load(lse_ptr + rows * H + h)
            lu = tl.zeros([BM], dtype=tl.float32)
            nu = tl.zeros([BM, D], dtype=tl.float32)
            q = tl.load(q_ptr + roff, mask=rv[:, None], other=0.0)
            for c0 in range(0, NB, BC):
                cols = c0 + tl.arange(0, BC)
                cv = cols < NB
                coff = (cols[:, None] * H + h) * D + d[None, :]
                km = tl.load(km_ptr + coff, mask=cv[:, None], other=0.0)
                kp = tl.load(keep_ptr + (h * NQ + t) * NB + cols, mask=cv, other=1)
                lb = tl.where(cols == NB - 1, lb_last, LB)
                s = tl.dot(q, km.T) * scale_log2
                e = tl.math.exp2(tl.minimum(s - lse[:, None] + lb[None, :], 100.0))
                e = tl.where((kp == 0)[None, :] & rv[:, None], e, 0.0)
                lu += tl.sum(e, 1)
                vm = tl.load(vm_ptr + coff, mask=cv[:, None], other=0.0)
                nu = tl.dot(e.to(tl.bfloat16), vm, nu)
            oe = tl.load(out_ptr + roff, mask=rv[:, None], other=0.0).to(tl.float32)
            tl.store(out_ptr + roff, ((oe + nu) / (1.0 + lu)[:, None]).to(tl.bfloat16), mask=rv[:, None])
            tl.store(lse_ptr + rows * H + h, tl.where(rv, lse + tl.math.log2(1.0 + lu), 1e30))


def compensate_fallback(q, k_mean, v_mean, keep, lse, out, flags):
    """Block-mean compensation of the skipped blocks for the fallback tiles, in place on `out` and `lse`."""
    L, H, D = q.shape
    nq = triton.cdiv(L, BM)
    nb = k_mean.shape[0]
    tail = L - (nb - 1) * BK
    nt = nq * H
    _compensate_kernel[(min(1184, nt),)](q, k_mean, v_mean, keep, lse, out, flags, 1.0 / math.sqrt(D) * LOG2E, L, nb,
                                         nq, nt, math.log2(tail), H=H, D=D, BM=BM,
                                         BC=128 if nb >= 128 else triton.next_power_of_2(nb), LB=math.log2(BK),
                                         num_warps=8, num_stages=2)
