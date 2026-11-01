"""Four representatives per 64-token kv block, clustered in the query metric.

For each head, G = E[q q^T] of the current (scaled) queries plus 1e-3 of its mean diagonal; with its Cholesky
factor M, distances in X = K M match the logit change the queries see. Token weights come from 64 probe queries
(means of evenly spaced query blocks): each probe's softmax over a block's tokens, summed over the probes. Weighted
Lloyd starts from the block's four contiguous 16-token groups. The representatives are the plain means of K and V
over each cluster, with log2 of the cluster size (-inf for an empty cluster).

Outputs follow the attention kernel's layout: (H, (nb + 1) * 4, D) bf16 keys and values and (H, (nb + 1) * 4)
float32 log2 counts. The extra block nb is empty (zeros, -inf) and pads the refine lists.
"""
import math

import torch
import triton
import triton.language as tl

BK = 64
NUM_PROBES = 64


@triton.jit
def _gram_kernel(q_ptr, s_ptr, L, RPS, H: tl.constexpr, D: tl.constexpr, BL: tl.constexpr):
    h = tl.program_id(0)
    r0 = tl.program_id(1) * RPS
    cols = tl.arange(0, D)
    acc = tl.zeros((D, D), dtype=tl.float32)
    for t in range(0, RPS, BL):
        rows = r0 + t + tl.arange(0, BL)
        qb = tl.load(q_ptr + (rows[:, None].to(tl.int64) * H + h) * D + cols[None, :],
                     mask=(rows < L)[:, None] & (rows < r0 + RPS)[:, None], other=0.0)
        acc = tl.dot(tl.trans(qb), qb, acc)
    tl.atomic_add(s_ptr + h * D * D + cols[:, None] * D + cols[None, :], acc)


@triton.jit
def _probe_kernel(q_ptr, p_ptr, L, scale, H: tl.constexpr, D: tl.constexpr, BK: tl.constexpr, NP: tl.constexpr, NB):
    i = tl.program_id(0)
    h = tl.program_id(1)
    blk = ((i * (NB - 1)).to(tl.float32) / (NP - 1) + 0.5).to(tl.int32)
    rows = blk * BK + tl.arange(0, BK)
    ok = rows < L
    cols = tl.arange(0, D)
    x = tl.load(q_ptr + (rows[:, None].to(tl.int64) * H + h) * D + cols[None, :], mask=ok[:, None], other=0.0)
    m = tl.sum(x.to(tl.float32), 0) * scale / tl.maximum(tl.sum(ok.to(tl.float32), 0), 1.0)
    tl.store(p_ptr + (h * NP + i) * D + cols, m.to(tl.bfloat16))


@triton.jit
def _cholesky_kernel(g_ptr, m_ptr, D: tl.constexpr):
    # right-looking rank-1 updates in registers; a non-positive pivot (rounding) is clamped instead of failing
    h = tl.program_id(0)
    r = tl.arange(0, D)
    A = tl.load(g_ptr + h * D * D + r[:, None] * D + r[None, :])
    eps = tl.sum(tl.where(r[:, None] == r[None, :], A, 0.0)) / D * 1e-12
    Lm = tl.zeros((D, D), dtype=tl.float32)
    for j in range(D):
        colj = tl.sum(tl.where(r[None, :] == j, A, 0.0), 1)
        piv = tl.maximum(tl.sum(tl.where(r == j, colj, 0.0), 0), eps)
        lj = tl.where(r >= j, colj / tl.sqrt(piv), 0.0)
        Lm = tl.where(r[None, :] == j, lj[:, None], Lm)
        A = A - lj[:, None] * lj[None, :]
    tl.store(m_ptr + h * D * D + r[:, None] * D + r[None, :], Lm.to(tl.bfloat16))


@triton.jit
def _kmeans_kernel(k_ptr, v_ptr, m_ptr, p_ptr, k4_ptr, v4_ptr, c4_ptr, km_ptr, vm_ptr, L, NB, NB4, H: tl.constexpr,
                   D: tl.constexpr, BK: tl.constexpr, ITERS: tl.constexpr, NP: tl.constexpr,
                   WRITE_MEANS: tl.constexpr):
    j = tl.program_id(0)
    h = tl.program_id(1)
    C: tl.constexpr = 16                      # 4 clusters, padded to the smallest tl.dot size
    tok = tl.arange(0, BK)
    rows = j * BK + tok
    valid = rows < L
    cols = tl.arange(0, D)
    off = (rows[:, None].to(tl.int64) * H + h) * D + cols[None, :]
    kk = tl.load(k_ptr + off, mask=valid[:, None], other=0.0)
    if WRITE_MEANS:
        n = tl.minimum(L - j * BK, BK).to(tl.float32)
        tl.store(km_ptr + (j * H + h) * D + cols, (tl.sum(kk.to(tl.float32), 0) / n).to(tl.bfloat16))
    mh = tl.load(m_ptr + h * D * D + cols[:, None] * D + cols[None, :])
    X = tl.dot(kk, mh)

    probes = tl.load(p_ptr + (h * NP + tl.arange(0, NP))[:, None] * D + cols[None, :])
    sp = tl.dot(probes, tl.trans(kk))
    sp = tl.where(valid[None, :], sp, float("-inf"))
    e = tl.exp(sp - tl.max(sp, 1)[:, None])
    e = e / tl.sum(e, 1)[:, None]
    w = tl.where(valid, tl.sum(e, 0) * (tl.sum(valid.to(tl.float32), 0) / NP), 0.0)

    cl = tl.arange(0, C)
    real = (cl < 4)[None, :]
    lab = (tok // (BK // 4)).to(tl.int32)
    oh = ((lab[:, None] == cl[None, :]) & valid[:, None]).to(tl.float32)
    ohw = oh * w[:, None]
    cw = tl.sum(ohw, 0)
    cen = tl.dot(tl.trans(ohw), X) / tl.maximum(cw, 1e-6)[:, None]
    it = 0
    changed = 1
    while (it < ITERS) & (changed > 0):
        dist = tl.sum(cen * cen, 1)[None, :] - 2.0 * tl.dot(X, tl.trans(cen))
        dist = tl.where(real & (cw > 0)[None, :], dist, float("inf"))
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
        tl.store(vm_ptr + (j * H + h) * D + cols, (tl.sum(vv.to(tl.float32), 0) / n).to(tl.bfloat16))
        kk = tl.load(k_ptr + off, mask=valid[:, None], other=0.0, volatile=True)    # reloaded, not kept live
    ohb = oh.to(tl.bfloat16)
    ck = tl.dot(tl.trans(ohb), kk) * inv[:, None]
    cv = tl.dot(tl.trans(ohb), vv) * inv[:, None]
    ok = cl < 4
    o = (h * NB4 + j * 4 + cl[:, None]).to(tl.int64) * D + cols[None, :]
    tl.store(k4_ptr + o, ck.to(tl.bfloat16), mask=ok[:, None])
    tl.store(v4_ptr + o, cv.to(tl.bfloat16), mask=ok[:, None])
    lc = tl.where(cnt > 0, tl.log2(tl.maximum(cnt, 1.0)), float("-inf"))
    tl.store(c4_ptr + h * NB4 + j * 4 + cl, lc, mask=ok)


class Representatives:
    """Builds the representatives of one sequence shape; buffers are reused across calls."""

    def __init__(self, L, H, D, device, iters=6):
        nb = triton.cdiv(L, BK)
        assert nb >= NUM_PROBES, "needs at least 64 kv blocks (4096 tokens)"
        self.L, self.H, self.D, self.nb, self.iters = L, H, D, nb, iters
        nb4 = (nb + 1) * 4
        self.k = torch.zeros(H, nb4, D, device=device, dtype=torch.bfloat16)
        self.v = torch.zeros(H, nb4, D, device=device, dtype=torch.bfloat16)
        self.log_counts = torch.full((H, nb4), -math.inf, device=device)
        self._gram = torch.empty(H, D, D, device=device)
        self._eye = torch.eye(D, device=device)
        self._probes = torch.empty(H, NUM_PROBES, D, device=device, dtype=torch.bfloat16)
        self._metric = torch.empty(H, D, D, device=device, dtype=torch.bfloat16)

    def build(self, q, k, v, k_mean=None, v_mean=None):
        """Representatives of every block; given k_mean / v_mean (nb, H, D), also the block means."""
        self._metric_and_probes(q)
        means = k_mean is not None
        _kmeans_kernel[(self.nb, self.H)](k, v, self._metric, self._probes, self.k, self.v, self.log_counts,
                                          k_mean if means else k, v_mean if means else v, self.L, self.nb,
                                          self.k.shape[1], H=self.H, D=self.D, BK=BK, ITERS=self.iters,
                                          NP=NUM_PROBES, WRITE_MEANS=means, num_warps=4, num_stages=1 if means else 3)

    def _metric_and_probes(self, q):
        L, H, D, nb = self.L, self.H, self.D, self.nb
        scale = D ** -0.5
        self._gram.zero_()
        rows_per_split = triton.cdiv(triton.cdiv(L, 32), 64) * 64
        _gram_kernel[(H, triton.cdiv(L, rows_per_split))](q, self._gram, L, rows_per_split, H=H, D=D, BL=64,
                                                           num_warps=8)
        g = self._gram * (scale * scale / L)
        g = (g + 1e-3 * g.diagonal(dim1=-2, dim2=-1).mean(-1)[:, None, None] * self._eye).contiguous()
        _cholesky_kernel[(H,)](g, self._metric, D=D, num_warps=8)
        _probe_kernel[(NUM_PROBES, H)](q, self._probes, L, scale, H=H, D=D, BK=BK, NP=NUM_PROBES, NB=nb, num_warps=4)
