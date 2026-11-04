"""Transparent PyTorch oracles for VSA2 and original VSA (tests only; materializes LxL, never timed).

Written directly from the method definition, independent of the kernels:
  router : pooled Q_r/K_r (mean of each R-token group), A_r = softmax over ALL
           valid pooled keys of (Q_r K_r^T / sqrt(D)), block score
           S[i,j] = norm_i * sum_{a in i, b in j} A_r[(i,a),(j,b)]
  select : exact global top-(NB*topk) of S per (b,h); ties -> smaller flat index
  fine   : token attention restricted to the selected blocks, padded keys
           masked; query blocks with no selected block -> O_f = 0
  coarse : dense attention over block means (Q_c, K_c, V_c), O_c per block
  fuse   : O = O_f + act(gate) * O_c[blk]; padded token rows -> 0

Original VSA (FastVideo 661d18ba ``video_sparse_attn``, 64-token blocks), from its source ops:
  router : S = (q_c @ k_c^T) / sqrt(D) in the token dtype over whole-block means
  select : per query block, exactly min(topk, NB) kv blocks; ties -> smaller kv index
  output : O = O_f + W * O_c[blk], W a full [B,H,L,D] direct weight (no activation)
"""

from __future__ import annotations

import math
from typing import Optional

import torch


def token_valid(blk_cnt: torch.Tensor, L: int) -> torch.Tensor:
    pos = torch.arange(L, device=blk_cnt.device)
    return (pos % 64) < blk_cnt.to(torch.int64)[pos // 64]


def group_mean(x: torch.Tensor, blk_cnt: torch.Tensor, R: int, round_to: Optional[torch.dtype]):
    """[B,H,L,D] -> ([B,H,NB*G,D] group means, [NB*G] valid)."""
    B, H, L, D = x.shape
    NB, G = L // 64, 64 // R
    valid = token_valid(blk_cnt, L).to(x.dtype)
    xs = (x * valid[None, None, :, None]).reshape(B, H, NB * G, R, D).sum(3)
    n = valid.reshape(NB * G, R).sum(1)
    m = xs / n.clamp(min=1)[None, None, :, None]
    if round_to is not None:
        m = m.to(round_to).to(x.dtype)
    return m, n > 0


def block_mean(x: torch.Tensor, blk_cnt: torch.Tensor, round_to: Optional[torch.dtype]):
    B, H, L, D = x.shape
    NB = L // 64
    valid = token_valid(blk_cnt, L).to(x.dtype)
    xs = (x * valid[None, None, :, None]).reshape(B, H, NB, 64, D).sum(3)
    m = xs / blk_cnt.to(x.dtype)[None, None, :, None]
    if round_to is not None:
        m = m.to(round_to).to(x.dtype)
    return m


def router_scores(q, k, blk_cnt, R, row_norm="paper", round_to=torch.bfloat16):
    """Dense oracle of the PSP block scores, fp32. q,k: [B,H,L,D]."""
    qf, kf = q.float(), k.float()
    D = q.shape[-1]
    G = 64 // R
    NB = q.shape[2] // 64
    qr, qv = group_mean(qf, blk_cnt, R, round_to)
    kr, kv = group_mean(kf, blk_cnt, R, round_to)
    s = torch.einsum("bhid,bhjd->bhij", qr, kr) / math.sqrt(D)
    s = s.masked_fill(~kv[None, None, None, :], float("-inf"))
    a = torch.softmax(s, dim=-1) * qv[None, None, :, None]
    B, H = q.shape[:2]
    S = a.reshape(B, H, NB, G, NB, G).sum((3, 5))
    if row_norm == "paper":
        S = S / (G * G)
    else:
        ngrp = (blk_cnt.to(torch.int64) + R - 1) // R
        S = S / ngrp.to(S.dtype)[None, None, :, None]
    return S


def global_topk_mask(scores: torch.Tensor, budget: int) -> torch.Tensor:
    """Exact global selection with the kernel's total order (score desc, flat index asc)."""
    B, H, NB, _ = scores.shape
    flat = scores.reshape(B * H, NB * NB)
    k = max(0, min(int(budget), NB * NB))
    mask = torch.zeros_like(flat, dtype=torch.bool)
    if k > 0:
        # stable descending sort keeps equal scores in ascending index order
        order = torch.sort(flat, dim=1, descending=True, stable=True).indices[:, :k]
        mask.scatter_(1, order, True)
    return mask.reshape(B, H, NB, NB)


def fine_attention(q, k, v, blk_cnt, block_mask):
    """Token-level fine branch in fp32. Returns (O_f, lse) with lse=-inf for empty rows."""
    qf, kf, vf = q.float(), k.float(), v.float()
    B, H, L, D = q.shape
    valid = token_valid(blk_cnt, L)
    blk = torch.arange(L, device=q.device) // 64
    tm = block_mask[:, :, blk][:, :, :, blk] & valid[None, None, None, :]
    s = torch.einsum("bhid,bhjd->bhij", qf, kf) / math.sqrt(D)
    s = s.masked_fill(~tm, float("-inf"))
    has = tm.any(-1, keepdim=True)
    s_safe = torch.where(has, s, torch.zeros_like(s))
    p = torch.softmax(s_safe, dim=-1) * has
    o = torch.einsum("bhij,bhjd->bhid", p, vf)
    lse = torch.where(has.squeeze(-1), torch.logsumexp(s_safe, -1), torch.full_like(s_safe[..., 0], float("-inf")))
    rowv = valid[None, None, :]
    o = o * rowv[..., None]
    lse = torch.where(rowv, lse, torch.full_like(lse, float("-inf")))
    return o, lse


def coarse_attention(q, k, v, blk_cnt, round_to=torch.bfloat16):
    qc = block_mean(q.float(), blk_cnt, round_to)
    kc = block_mean(k.float(), blk_cnt, round_to)
    vc = block_mean(v.float(), blk_cnt, round_to)
    D = q.shape[-1]
    s = torch.einsum("bhid,bhjd->bhij", qc, kc) / math.sqrt(D)
    return torch.einsum("bhij,bhjd->bhid", torch.softmax(s, -1), vc)


def vsa2_reference(q, k, v, gate, blk_cnt, block_mask, gate_act="tanh", round_pooled=True):
    """Full VSA2 output with a given (frozen) block mask; differentiable in fp32."""
    B, H, L, D = q.shape
    rt = q.dtype if round_pooled else None
    of, _ = fine_attention(q, k, v, blk_cnt, block_mask)
    oc = coarse_attention(q, k, v, blk_cnt, round_to=rt)
    blk = torch.arange(L, device=q.device) // 64
    ocr = oc[:, :, blk]
    if gate is None:
        a = torch.ones_like(of[..., :1])
    else:
        g = gate.float()
        a = torch.tanh(g) if gate_act == "tanh" else g
        if g.dim() == 3:
            a = a[..., None]
    valid = token_valid(blk_cnt, L)[None, None, :, None]
    return (of + a * ocr) * valid


def vsa2_reference_routed(q, k, v, gate, blk_cnt, R, budget, row_norm="paper", gate_act="tanh"):
    with torch.no_grad():
        S = router_scores(q, k, blk_cnt, R, row_norm, round_to=q.dtype)
        mask = global_topk_mask(S, budget)
    return vsa2_reference(q, k, v, gate, blk_cnt, mask, gate_act), mask, S


# --------------------------------------------------------------------------- original VSA


def block_scores_from_means(qc, kc):
    """The original's score op alone, on given block means [B, H, NB, D]: matmul, / sqrt(D), in their dtype."""
    return torch.matmul(qc, kc.transpose(-2, -1)) / (qc.shape[-1] ** 0.5)


def vsa_block_scores(q, k, blk_cnt):
    """The original's router scores, its own Torch ops in the token dtype (fused_block_mean, matmul, / sqrt(D))."""
    L, D = q.shape[2], q.shape[3]
    nb = L // 64
    valid = token_valid(blk_cnt, L)[None, None, :, None]
    c = blk_cnt.to(torch.float32)[None, None, :, None]
    mean = lambda x: ((x * valid).float().reshape(*x.shape[:2], nb, 64, D).sum(3) / c).to(x.dtype)
    return block_scores_from_means(mean(q), mean(k))


def row_topk_mask(scores: torch.Tensor, topk: int) -> torch.Tensor:
    """Exactly min(topk, NB) kv blocks per row: score desc, ties -> smaller kv index (stable sort)."""
    k = max(0, min(int(topk), scores.shape[-1]))
    # + 0.0 turns -0 into +0, so a bitwise (radix) sort also sees them as equal
    order = torch.sort(scores.float() + 0.0, dim=-1, descending=True, stable=True).indices[..., :k]
    return torch.zeros(scores.shape, dtype=torch.bool, device=scores.device).scatter_(-1, order, True)


def vsa_reference(q, k, v, weight, blk_cnt, block_mask, dtype=torch.bfloat16):
    """Original VSA output for a frozen block mask, float64 inputs, the original's rounding points to ``dtype``
    applied straight-through (STE, as its casts differentiate). Pools sum all 64 slots / count with padding
    held at zero, like the original on zero-padded inputs; padded slots are constants, and padded output
    rows and query blocks with no kv block give O_f = 0, the native contract. Returns (O, coarse logits)."""
    B, H, L, D = q.shape
    nb = L // 64
    valid = token_valid(blk_cnt, L)
    zero = valid[None, None, :, None].to(q.dtype)
    rnd = lambda x: x + (x.to(dtype).to(x.dtype) - x).detach()
    c = blk_cnt.to(q.dtype)[None, None, :, None]
    pool = lambda x: rnd((x * zero).reshape(B, H, nb, 64, D).sum(3) / c)
    qc, kc, vc = pool(q), pool(k), pool(v)
    logits = rnd(rnd(qc @ kc.transpose(-1, -2)) / math.sqrt(D))
    oc = rnd(rnd(torch.softmax(logits, -1)) @ vc).repeat_interleave(64, dim=2)
    fm = block_mask.repeat_interleave(64, -2).repeat_interleave(64, -1) & valid[None, None, None, :]
    s = ((q * zero) @ (k * zero).transpose(-1, -2)) / math.sqrt(D)
    s = s.masked_fill(~fm, float("-inf"))
    has = fm.any(-1, keepdim=True)  # rows with no kv block: O_f = 0 and no fine gradient (fine_attention's pattern)
    p = torch.softmax(torch.where(has, s, torch.zeros_like(s)), -1) * has
    of = rnd(p @ (v * zero))
    o = of + (oc if weight is None else weight * oc)
    return rnd(o) * zero, logits.detach()
