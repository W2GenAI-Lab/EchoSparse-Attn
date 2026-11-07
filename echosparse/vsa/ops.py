"""Original VSA and VSA2 operators: autograd wiring around the native CUDA kernels.

Both algorithms share every kernel except the router. Every production stage
(pooling, VSA2's PSP router and exact global top-k, original VSA's whole-block
scores and per-row top-K, CSR list construction and inversion, coarse
attention, fine block-sparse attention, gate fusion, and every backward kernel)
is a native CUDA kernel from ``echosparse.vsa._C``. PyTorch is used here only
for allocation, ``torch.zeros`` initialization of accumulators, and the
shape-only dense CSR of the coarse branch.

Preblocked layout contract (layout.py builds it from real tokens):
  q, k, v : [B, H, L, D] contiguous, bf16 (primary) or fp16, D in {64, 128}.
            L = NB * 64; tokens are already in block-major order (64-token
            hardware blocks, each made of G = 64/R contiguous R-token router
            groups); block i holds blk_cnt[i] valid tokens first, then padding.
            Padding slots must hold finite values (zeros recommended); they are
            never pooled, output 0 and receive zero gradient.
  gate    : optional pre-activation, [B, H, L] (per token/head, VSA2) or
            [B, H, L, D] (per element, original VSA). O = O_f + act(gate) * O_c.
  blk_cnt : int32 [NB], 1..64 valid tokens per block.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .dispatch import check_bwd_variant, select_bwd_variant

_C = None
_bwd_override: Optional[int] = None

# Opt-in speed toggles: environment variables read on every call, off unless set to "1".
TK_BWD_STORE_ENV = "ECHOSPARSE_VSA_TK_BWD_STORE"  # TK backward: single-writer dK/dV store, no kg/vg zero fill
TK_BWD_STORE_KV = 16  # the tk_fine_bwd opts bit


def _env_on(name: str) -> bool:
    return os.environ.get(name) == "1"


def _tk_bwd_kv(q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """dK/dV accumulators and opts for C.tk_fine_bwd. With ECHOSPARSE_VSA_TK_BWD_STORE=1 the kernel stores every
    element (its grid is (NB, H, B) and it is MHA, so each (b, h, kv block) has one owner CTA, which writes zeros
    for an empty list), so the accumulators are allocated uninitialised. dQ is unchanged; dK/dV are the same
    values except that a -0 element stays -0 instead of becoming +0."""
    store = _env_on(TK_BWD_STORE_ENV)
    alloc = torch.empty if store else torch.zeros
    return (alloc(q.shape, device=q.device, dtype=torch.float32), alloc(q.shape, device=q.device, dtype=torch.float32),
            TK_BWD_STORE_KV if store else 0)


def ext():
    """The native extension, imported on first use; ``import echosparse.vsa`` never needs it."""
    global _C
    if _C is None:
        try:
            from . import _C as C  # type: ignore
        except ImportError as e:
            raise ImportError("echosparse.vsa needs its native extension; build it with ECHOSPARSE_VSA_BUILD=1 "
                              "pip install -e . --no-build-isolation (see README)") from e
        _C = C
    return _C


GATE_ACTS = {"tanh": 0, "identity": 1}
ROW_NORMS = {"valid_groups": 0, "paper": 1}


@dataclass
class Selection:
    """Exact block selection, CSR. Absolute offsets into the idx arrays."""

    q2k_ptr: torch.Tensor  # [B, H, NB+1] int32
    q2k_idx: torch.Tensor  # [nnz] int32, kv block ids ascending per row
    k2q_ptr: torch.Tensor  # [B, H, NB+1]
    k2q_idx: torch.Tensor  # [nnz], query block ids ascending per column

    def row_counts(self) -> torch.Tensor:
        return (self.q2k_ptr[..., 1:] - self.q2k_ptr[..., :-1]).to(torch.int64)

    def to_mask(self, nb: int) -> torch.Tensor:
        """Dense bool block mask [B,H,NB,NB] (rows are stored contiguously in (b,h,i) order)."""
        B, H = self.q2k_ptr.shape[:2]
        cnt = self.row_counts().reshape(-1)
        rows = torch.repeat_interleave(torch.arange(B * H * nb, device=cnt.device), cnt)
        first = int(self.q2k_ptr.reshape(-1)[0])
        idx = self.q2k_idx[first : first + rows.numel()].to(torch.int64)
        mask = torch.zeros(B * H * nb, nb, dtype=torch.bool, device=cnt.device)
        mask[rows, idx] = True
        return mask.reshape(B, H, nb, nb)

    def col_mask(self, nb: int) -> torch.Tensor:
        """Mask rebuilt from the inverted kv->q lists, transposed back to [B,H,NBq,NBkv]."""
        inv = Selection(self.k2q_ptr, self.k2q_idx, self.q2k_ptr, self.q2k_idx)
        return inv.to_mask(nb).transpose(-1, -2)


def mask_to_selection(mask: torch.Tensor) -> Selection:
    """Bool block mask [B,H,NB,NB] -> CSR selection (test/diagnostic utility)."""
    assert mask.dim() == 4 and mask.dtype == torch.bool

    def csr(m):
        B, H, N, _ = m.shape
        cnt = m.reshape(-1, N).sum(-1).to(torch.int64)
        incl = torch.cumsum(cnt, 0)
        excl = incl - cnt
        ptr = torch.cat([excl.reshape(B * H, N), incl.reshape(B * H, N)[:, -1:]], dim=1)
        idx = m.reshape(-1, N).nonzero()[:, 1]
        return ptr.reshape(B, H, N + 1).to(torch.int32).contiguous(), idx.to(torch.int32).contiguous()

    qp, qi = csr(mask)
    kp, ki = csr(mask.transpose(-1, -2).contiguous())
    return Selection(qp, qi, kp, ki)


def _coarse_meta(B: int, H: int, nb: int, nbc_pad: int, device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dense CSR over the 64-row tiles of the pooled sequence (shape-only metadata)."""
    T = nbc_pad // 64
    cnt = torch.clamp(nb - 64 * torch.arange(T, device=device), max=64).to(torch.int32)
    bh = torch.arange(B * H, device=device, dtype=torch.int64)
    ptr = bh[:, None] * (T * T) + torch.arange(T + 1, device=device, dtype=torch.int64)[None, :] * T
    ptr = ptr.to(torch.int32).reshape(B, H, T + 1).contiguous()
    idx = torch.arange(T, device=device, dtype=torch.int32).repeat(B * H * T).contiguous()
    return cnt.contiguous(), ptr, idx


def resolve_backend(backend: str, q: torch.Tensor) -> bool:
    """Returns True when the SM90 ThunderKittens fine kernels should be used."""
    C = ext()
    if backend == "generic":
        return False
    tk_ok = C.has_tk() and q.is_cuda and torch.cuda.get_device_capability(q.device) == (9, 0) and q.dtype == torch.bfloat16
    if backend == "tk":
        if not tk_ok:
            raise RuntimeError("the tk backend requires an sm_90 GPU, bf16 inputs and an ECHOSPARSE_VSA_TK=1 build")
        return True
    if backend != "auto":
        raise ValueError(f"unknown backend {backend!r}")
    return tk_ok


def set_bwd_variant(v: Optional[int]) -> None:
    """Process-wide generic backward variant override; None restores the automatic rule."""
    global _bwd_override
    _bwd_override = None if v is None else check_bwd_variant(v, "set_bwd_variant")


def resolve_bwd_variant(q: torch.Tensor, call: Optional[int] = None) -> Tuple[int, str]:
    """(variant, source) for this tensor's shape and device; rule and evidence in dispatch.py."""
    B, H, L, D = q.shape
    major = torch.cuda.get_device_capability(q.device)[0] if q.is_cuda else -1
    return select_bwd_variant(B, H, L, D, major, call=call, override=_bwd_override)


def budget_from_topk(nb: int, topk: Optional[int], budget: Optional[int]) -> int:
    if budget is not None:
        return int(budget)
    if topk is None:
        raise ValueError("pass topk (average kv blocks per query block) or budget")
    return int(nb) * int(topk)


def route(q, k, blk_cnt, R=16, topk=None, budget=None, row_norm="paper", max_score_bytes=2 << 30) -> Selection:
    """Standalone VSA2 PSP router + exact global top-k (stop-gradient)."""
    C = ext()
    nb = q.shape[2] // 64
    qr, kr, _, _, _ = C.pool_fwd(q, k, k, blk_cnt, R)
    out = C.route(qr, kr, blk_cnt, R, budget_from_topk(nb, topk, budget), ROW_NORMS[row_norm], max_score_bytes)
    return Selection(*out)


# Original VSA only needs the whole-block means of pool_fwd; R=32 makes its unused router pools smallest.
_ROWS_POOL_R = 32


def route_rows(q, k, blk_cnt, topk, max_score_bytes=2 << 30) -> Selection:
    """Standalone original-VSA router (stop-gradient): whole-block scores, exactly min(topk, NB) blocks per row."""
    C = ext()
    _, _, qc, kc, _ = C.pool_fwd(q, k, k, blk_cnt, _ROWS_POOL_R)
    return Selection(*C.route_rows(qc, kc, q.shape[2] // 64, int(topk), int(max_score_bytes)))


def _generic_fine_bwd(C, variant, q, k, v, dout, lse, delta, q2k_ptr, q2k_idx, k2q_ptr, k2q_idx, blk_cnt, dqc, dkc, dvc):
    """Generic fine backward plus coarse unpooling. Returns (dq, dk, dv) in q's dtype."""
    if variant == 2:
        dk, dv = C.attn_bwd(q, k, v, dout, lse, delta, k2q_ptr, k2q_idx, blk_cnt, None, dkc, dvc, False, False)
        return C.attn_bwd_q(q, k, v, dout, lse, delta, q2k_ptr, q2k_idx, blk_cnt, dqc), dk, dv
    dq_acc = torch.zeros(q.shape, device=q.device, dtype=torch.float32)
    dk, dv = C.attn_bwd(q, k, v, dout, lse, delta, k2q_ptr, k2q_idx, blk_cnt, dq_acc, dkc, dvc, False, variant == 1)
    return C.finalize(dq_acc, dqc, blk_cnt, q), dk, dv


class _Function(torch.autograd.Function):
    """Shared forward/backward of both algorithms. ``router`` is ("global", budget, row_norm) for VSA2 or
    ("rows", topk) for original VSA; the backward never depends on it (it uses the frozen forward selection)."""

    @staticmethod
    def forward(ctx, q, k, v, gate, blk_cnt, R, router, gate_act, use_tk, max_score_bytes, sel_in, bwd):
        C = ext()
        B, H, L, D = q.shape
        nb = L // 64
        qr, kr, qc, kc, vc = C.pool_fwd(q, k, v, blk_cnt, R)
        c_cnt, c_ptr, c_idx = _coarse_meta(B, H, nb, qc.shape[2], q.device)
        oc, lse_c = C.attn_fwd(qc, kc, vc, c_ptr, c_idx, c_cnt, False)
        if sel_in is None and router[0] == "rows":
            q2k_ptr, q2k_idx, k2q_ptr, k2q_idx = C.route_rows(qc, kc, nb, router[1], max_score_bytes)
        elif sel_in is None:
            q2k_ptr, q2k_idx, k2q_ptr, k2q_idx = C.route(qr, kr, blk_cnt, R, router[1], router[2], max_score_bytes)
        else:
            q2k_ptr, q2k_idx, k2q_ptr, k2q_idx = sel_in.q2k_ptr, sel_in.q2k_idx, sel_in.k2q_ptr, sel_in.k2q_idx
        of, lse_f = C.attn_fwd(q, k, v, q2k_ptr, q2k_idx, blk_cnt, use_tk)
        o = C.fuse_fwd(of, oc, gate, blk_cnt, gate_act)
        ctx.save_for_backward(q, k, v, gate, blk_cnt)
        # Intermediates are held as ctx attributes (not inputs/outputs) so backward can
        # release O_f as soon as delta is computed (peak-memory: one [B,H,L,D] fewer).
        ctx.inter = [qc, kc, vc, oc, lse_c, of, lse_f, q2k_ptr, q2k_idx, k2q_ptr, k2q_idx, c_cnt, c_ptr, c_idx]
        ctx.has_gate = gate is not None
        ctx.gate_act = gate_act
        ctx.use_tk = use_tk
        # The choice made at forward: None for the TK fine path, else (variant, source). Inspect via o.grad_fn.
        ctx.bwd_variant, ctx.bwd_source = (None, "tk") if use_tk else bwd
        ctx.mark_non_differentiable(q2k_ptr, q2k_idx, k2q_ptr, k2q_idx)
        return o, q2k_ptr, q2k_idx, k2q_ptr, k2q_idx

    @staticmethod
    def backward(ctx, dout, *unused):
        C = ext()
        q, k, v, gate, blk_cnt = ctx.saved_tensors
        if ctx.inter is None:
            raise RuntimeError("VSA backward frees its intermediates after the first backward; "
                               "re-run the forward instead of backward(retain_graph=True) twice")
        (qc, kc, vc, oc, lse_c, of, lse_f, q2k_ptr, q2k_idx, k2q_ptr, k2q_idx, c_cnt, c_ptr, c_idx) = ctx.inter
        dout = dout.contiguous()
        dout_f, delta_f, dout_c, delta_c, dgate = C.fuse_bwd(
            dout, of, oc, gate if ctx.has_gate else None, blk_cnt, ctx.gate_act, ctx.use_tk)
        if not ctx.use_tk:
            dout_f = dout  # generic fine bwd zeroes P on padded/empty rows; no masked copy needed
        del of
        ctx.inter[5] = None
        # coarse branch (dense over pooled blocks) -> fp32 per-block grads
        dqc = torch.zeros(qc.shape, device=q.device, dtype=torch.float32)
        dkc, dvc = C.attn_bwd(qc, kc, vc, dout_c, lse_c, delta_c, c_ptr, c_idx, c_cnt, dqc, None, None, True,
                              ctx.bwd_variant == 1)
        # fine branch on the exact forward selection, + coarse unpooling
        if ctx.use_tk:
            dq_acc = torch.zeros(q.shape, device=q.device, dtype=torch.float32)
            dk_acc, dv_acc, opts = _tk_bwd_kv(q)
            C.tk_fine_bwd(q, k, v, dout_f, lse_f, delta_f, k2q_ptr, k2q_idx, blk_cnt, dq_acc, dk_acc, dv_acc, opts)
            dk = C.finalize(dk_acc, dkc, blk_cnt, k)
            dv = C.finalize(dv_acc, dvc, blk_cnt, v)
            dq = C.finalize(dq_acc, dqc, blk_cnt, q)
        else:
            dq, dk, dv = _generic_fine_bwd(C, ctx.bwd_variant, q, k, v, dout_f, lse_f, delta_f, q2k_ptr, q2k_idx,
                                           k2q_ptr, k2q_idx, blk_cnt, dqc, dkc, dvc)
        del dout_f
        ctx.inter = None
        return (dq, dk, dv, dgate if ctx.has_gate else None) + (None,) * 8


def _attention(q, k, v, gate, blk_cnt, R, router, gate_act, backend, selection, max_score_bytes, return_selection,
               bwd_variant):
    for name, t in (("q", q), ("k", k), ("v", v)):
        if not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous [B,H,L,D]")
    blk_cnt = blk_cnt.to(torch.int32).contiguous()
    if gate is not None:
        gate = gate.contiguous()
    o, qp, qi, kp, ki = _Function.apply(q, k, v, gate, blk_cnt, int(R), router, GATE_ACTS[gate_act],
                                        resolve_backend(backend, q), int(max_score_bytes), selection,
                                        resolve_bwd_variant(q, bwd_variant))
    if return_selection:
        return o, Selection(qp, qi, kp, ki)
    return o


def vsa2_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: Optional[torch.Tensor],
    blk_cnt: torch.Tensor,
    *,
    R: int = 16,
    topk: Optional[int] = None,
    budget: Optional[int] = None,
    row_norm: str = "paper",
    gate_act: str = "tanh",
    backend: str = "auto",
    selection: Optional[Selection] = None,
    max_score_bytes: int = 2 << 30,
    return_selection: bool = False,
    bwd_variant: Optional[int] = None,
):
    """Full VSA2 attention: O = O_fine + act(gate) * repeat(O_coarse).

    The router (pool R -> softmax over all pooled keys -> pool to 64-token
    blocks) and the exact global top-k (budget = NB*topk connections per
    (b, h)) run on every call unless a frozen ``selection`` is passed (testing).
    ``bwd_variant`` overrides the generic backward choice for this call (dispatch.py). The choice
    is fixed at forward and readable from ``o.grad_fn.bwd_variant`` and ``o.grad_fn.bwd_source``.
    """
    nb = q.shape[2] // 64
    router = ("global", budget_from_topk(nb, topk, budget) if selection is None else 0, ROW_NORMS[row_norm])
    return _attention(q, k, v, gate, blk_cnt, R, router, gate_act, backend, selection, max_score_bytes,
                      return_selection, bwd_variant)


def vsa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    blk_cnt: torch.Tensor,
    topk: int,
    *,
    weight: Optional[torch.Tensor] = None,
    backend: str = "auto",
    selection: Optional[Selection] = None,
    max_score_bytes: int = 2 << 30,
    return_selection: bool = False,
    bwd_variant: Optional[int] = None,
):
    """Original VSA attention (FastVideo ``video_sparse_attn`` with 64-token blocks): O = O_fine + W * repeat(O_coarse).

    Router: whole-block means q_c, k_c; S = (q_c @ k_c^T) / sqrt(D) rounded as in the original's bf16/fp16
    Torch ops; every query block keeps exactly min(topk, NB) kv blocks (score desc, ties -> smaller kv index).
    ``weight`` is the original ``compress_attn_weight``: a full [B, H, L, D] direct weight, no activation;
    None means W = 1. Coarse attention, fine attention, fusion and the backward are the kernels shared with
    VSA2. Unlike the original, padded slots are never pooled, output 0 and get zero gradient (see layout.py).
    """
    if weight is not None and weight.shape != q.shape:
        raise ValueError(f"weight must be [B, H, L, D] like q, got {tuple(weight.shape)}")
    return _attention(q, k, v, weight, blk_cnt, _ROWS_POOL_R, ("rows", int(topk)), "identity", backend, selection,
                      max_score_bytes, return_selection, bwd_variant)


class _FineFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, blk_cnt, sel, use_tk, bwd):
        C = ext()
        of, lse = C.attn_fwd(q, k, v, sel.q2k_ptr, sel.q2k_idx, blk_cnt, use_tk)
        ctx.save_for_backward(q, k, v, blk_cnt, of, lse, sel.q2k_ptr, sel.q2k_idx, sel.k2q_ptr, sel.k2q_idx)
        ctx.use_tk = use_tk
        ctx.bwd_variant, ctx.bwd_source = (None, "tk") if use_tk else bwd
        return of, lse

    @staticmethod
    def backward(ctx, dout, dlse):
        C = ext()
        q, k, v, blk_cnt, of, lse, qp, qi, kp, ki = ctx.saved_tensors
        B, H, L, D = q.shape
        nbc_pad = (L // 64 + 63) // 64 * 64
        oc0 = torch.zeros(B, H, nbc_pad, D, device=q.device, dtype=q.dtype)
        dout = dout.contiguous()
        dout_f, delta_f, _, _, _ = C.fuse_bwd(dout, of, oc0, None, blk_cnt, 0, ctx.use_tk)
        if not ctx.use_tk:
            dout_f = dout
        if ctx.use_tk:
            dq_acc = torch.zeros(q.shape, device=q.device, dtype=torch.float32)
            dk_acc, dv_acc, opts = _tk_bwd_kv(q)
            C.tk_fine_bwd(q, k, v, dout_f, lse, delta_f, kp, ki, blk_cnt, dq_acc, dk_acc, dv_acc, opts)
            dk = C.finalize(dk_acc, None, blk_cnt, k)
            dv = C.finalize(dv_acc, None, blk_cnt, v)
            dq = C.finalize(dq_acc, None, blk_cnt, q)
        else:
            dq, dk, dv = _generic_fine_bwd(C, ctx.bwd_variant, q, k, v, dout_f, lse, delta_f, qp, qi, kp, ki, blk_cnt,
                                           None, None, None)
        return dq, dk, dv, None, None, None, None


def fine_block_sparse_attention(q, k, v, blk_cnt, selection: Selection, backend: str = "auto",
                                bwd_variant: Optional[int] = None):
    """Diagnostic fine-only primitive (no coarse branch, no router). Returns (O_f, lse).

    Zero-edge query blocks return O_f = 0 and lse = -inf. With the TK backend
    padded query rows hold finite values; the generic backend zeroes them.
    """
    use_tk = resolve_backend(backend, q)
    return _FineFunction.apply(q, k, v, blk_cnt.to(torch.int32).contiguous(), selection, use_tk,
                               resolve_bwd_variant(q, bwd_variant))
