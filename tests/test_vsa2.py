"""Native VSA2 (echosparse.vsa) against the independent PyTorch oracle; also the shared fine/coarse/fusion kernels.

Run:  pytest -q tests/test_vsa2.py            (all)
      pytest -q tests/test_vsa2.py -k topk    (subset)
Metrics of every comparison are appended to $ECHOSPARSE_VSA_TEST_LOG (JSONL) if set.
"""

import json
import math
import os

import pytest
import torch

from echosparse.vsa import ops
import vsa_reference as ref
from echosparse.vsa.ops import ext


def _built():
    try:
        ext()
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not _built(),
                                reason="needs a CUDA GPU and the ECHOSPARSE_VSA_BUILD=1 extension")
DEV = "cuda"


def _backends():
    out = ["generic"]
    try:
        if ext().has_tk() and torch.cuda.get_device_capability() == (9, 0):
            out.append("tk")
    except Exception:
        pass
    return out


BACKENDS = _backends() if torch.cuda.is_available() else ["generic"]


def log_metric(name, **kw):
    path = os.environ.get("ECHOSPARSE_VSA_TEST_LOG")
    if path:
        with open(path, "a") as f:
            f.write(json.dumps({"test": name, **kw}) + "\n")


def counts(nb, kind, gen):
    c = torch.full((nb,), 64, dtype=torch.int32)
    if kind == "tail":
        c[-1] = 37
    elif kind == "mixed":
        c = torch.randint(1, 65, (nb,), generator=gen, dtype=torch.int32)
        c[0] = 64
        if nb > 2:
            c[1] = 1
    return c.to(DEV)


def inputs(B, H, nb, D, dtype, cnt, seed=0, gate_mode="token"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    L = nb * 64
    valid = ref.token_valid(cnt, L)[None, None, :, None].cpu()
    mk = lambda: (torch.randn(B, H, L, D, generator=g) * valid).to(dtype).to(DEV).contiguous()
    q, k, v = mk(), mk(), mk()
    if gate_mode == "token":
        gate = (torch.randn(B, H, L, generator=g) * 1.5).to(dtype).to(DEV)
    elif gate_mode == "elem":
        gate = (torch.randn(B, H, L, D, generator=g) * 1.5).to(dtype).to(DEV)
    elif gate_mode == "zero":
        gate = torch.zeros(B, H, L, dtype=dtype, device=DEV)
    else:
        gate = None
    return q, k, v, gate


def check(name, got, exp, rtol, valid_rows=None, **meta):
    got = got.float()
    exp = exp.float()
    if valid_rows is not None:
        got = got[..., valid_rows, :] if got.dim() == 4 else got[..., valid_rows]
        exp = exp[..., valid_rows, :] if exp.dim() == 4 else exp[..., valid_rows]
    assert torch.isfinite(got).all(), f"{name}: non-finite values"
    scale = exp.abs().max().clamp(min=1e-6).item()
    err = (got - exp).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(got.flatten(), exp.flatten(), dim=0).item() if exp.abs().max() > 0 else 1.0
    log_metric(name, max_abs_err=err, ref_max=scale, rel=err / scale, cos=cos, **meta)
    assert err <= rtol * scale + 1e-5, f"{name}: max err {err:.3e} > {rtol}*{scale:.3e} (cos {cos:.6f}) {meta}"


def random_mask(B, H, nb, gen, kind):
    if kind == "full":
        return torch.ones(B, H, nb, nb, dtype=torch.bool, device=DEV)
    if kind == "diag":
        return torch.eye(nb, dtype=torch.bool, device=DEV).expand(B, H, nb, nb).contiguous()
    m = torch.rand(B, H, nb, nb, generator=gen) < 0.3
    m[:, :, 0] = False          # zero-edge query block
    m[:, :, -1] = True          # a query block that sees everything
    if nb > 3:
        m[:, :, 2] = False
        m[:, :, 2, 1] = True    # single edge
    return m.to(DEV)


# --------------------------------------------------------------------------- top-k


@pytest.mark.parametrize("nb", [1, 7, 33, 130])
@pytest.mark.parametrize("ties", [False, True])
def test_topk_exact(nb, ties):
    gen = torch.Generator().manual_seed(nb)
    BH = 3
    if ties:
        scores = torch.randint(0, 4, (BH, nb, nb), generator=gen).float() / 4.0
    else:
        scores = torch.rand(BH, nb, nb, generator=gen)
    scores = scores.to(DEV)
    for budget in sorted({0, 1, nb, 3 * nb, nb * nb - 1, nb * nb, nb * nb + 5}):
        if budget < 0:
            continue
        keff = max(0, min(budget, nb * nb))
        q2k_ptr = torch.empty(1, BH, nb + 1, dtype=torch.int32, device=DEV)
        k2q_ptr = torch.empty_like(q2k_ptr)
        q2k_idx = torch.empty(max(BH * keff, 1), dtype=torch.int32, device=DEV)
        k2q_idx = torch.empty_like(q2k_idx)
        ext().global_topk_into(scores, budget, 0, q2k_ptr, q2k_idx, k2q_ptr, k2q_idx)
        sel = ops.Selection(q2k_ptr, q2k_idx[: BH * keff], k2q_ptr, k2q_idx[: BH * keff])
        exp = ref.global_topk_mask(scores.reshape(1, BH, nb, nb), budget)
        got = sel.to_mask(nb)
        assert torch.equal(got, exp), f"top-k mismatch nb={nb} budget={budget} ties={ties}"
        assert torch.equal(sel.col_mask(nb), exp), "k2q inversion mismatch"
        assert int(got.sum()) == BH * keff
        # ascending lists
        cnt = sel.row_counts().reshape(-1)
        idx = q2k_idx[: BH * keff].long()
        if idx.numel() > 1:
            rows = torch.repeat_interleave(torch.arange(cnt.numel(), device=DEV), cnt)
            same = rows[1:] == rows[:-1]
            assert bool((idx[1:][same] > idx[:-1][same]).all())


# --------------------------------------------------------------------------- router


@pytest.mark.parametrize("R", [4, 8, 16, 32])
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("kind", ["full", "mixed"])
@pytest.mark.parametrize("row_norm", ["valid_groups", "paper"])
def test_router_scores(R, D, kind, row_norm):
    gen = torch.Generator().manual_seed(R * D)
    nb = 11
    cnt = counts(nb, kind, gen)
    q, k, _, _ = inputs(2, 2, nb, D, torch.bfloat16, cnt, seed=R)
    C = ext()
    qr, kr, qc, kc, vc = C.pool_fwd(q, k, k, cnt, R)
    exp_qr, _ = ref.group_mean(q.float(), cnt, R, torch.bfloat16)
    check("pool_qr", qr[:, :, : nb * (64 // R)], exp_qr, 1e-2, R=R)
    exp_qc = ref.block_mean(q.float(), cnt, torch.bfloat16)
    check("pool_qc", qc[:, :, :nb], exp_qc, 1e-2, R=R)
    lse2 = C.router_lse(qr, kr, cnt, R)
    S = C.router_scores(qr, kr, lse2, cnt, R, ops.ROW_NORMS[row_norm], 0, 4).reshape(2, 2, nb, nb)
    Sref = ref.router_scores(q, k, cnt, R, row_norm, round_to=torch.bfloat16)
    check("router_scores", S, Sref, 2e-3, R=R, D=D, kind=kind, row_norm=row_norm)
    if row_norm == "valid_groups":
        assert torch.allclose(S.sum(-1), torch.ones_like(S[..., 0]), atol=2e-3)


@pytest.mark.parametrize("R", [8, 16])
@pytest.mark.parametrize("row_norm", ["paper", "valid_groups"])
def test_route_exact_and_deterministic(R, row_norm):
    gen = torch.Generator().manual_seed(5)
    nb, topk = 24, 4
    cnt = counts(nb, "mixed", gen)
    q, k, _, _ = inputs(2, 3, nb, 128, torch.bfloat16, cnt, seed=3)
    C = ext()
    sel = ops.route(q, k, cnt, R=R, topk=topk, row_norm=row_norm)
    sel2 = ops.route(q, k, cnt, R=R, topk=topk, row_norm=row_norm, max_score_bytes=nb * nb * 4)  # chunked: 1 (b,h) per chunk
    m = sel.to_mask(nb)
    assert torch.equal(m, sel2.to_mask(nb)), "routing not deterministic across chunking"
    assert torch.equal(sel.col_mask(nb), m)
    assert (m.sum((-1, -2)) == nb * topk).all(), "budget must be exactly NB*topk per (b,h)"
    # exact top-k of the kernel's own scores
    qr, kr, _, _, _ = C.pool_fwd(q, k, k, cnt, R)
    lse2 = C.router_lse(qr, kr, cnt, R)
    S = C.router_scores(qr, kr, lse2, cnt, R, ops.ROW_NORMS[row_norm], 0, 6).reshape(2, 3, nb, nb)
    assert torch.equal(m, ref.global_topk_mask(S, nb * topk))
    # agreement with the fp32 oracle scores (may differ only at near-ties)
    Sref = ref.router_scores(q, k, cnt, R, row_norm, round_to=torch.bfloat16)
    agree = (m & ref.global_topk_mask(Sref, nb * topk)).sum().item() / m.sum().item()
    log_metric("route_oracle_agreement", R=R, row_norm=row_norm, agreement=agree)
    assert agree > 0.97


# --------------------------------------------------------------------------- fine only


@pytest.fixture(params=[1, 2], ids=["fwdMT1", "fwdMT2"])
def fwd_variant(request):
    C = ext()
    old = C.get_fwd_variant()
    C.set_fwd_variant(request.param)
    yield request.param
    C.set_fwd_variant(old)


@pytest.fixture(params=[0, 1, 2], ids=["bwdF2", "bwdWord", "bwdSplit"])
def bwd_variant(request):
    ops.set_bwd_variant(request.param)
    yield request.param
    ops.set_bwd_variant(None)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("mask_kind", ["full", "diag", "random"])
@pytest.mark.parametrize("cnt_kind", ["full", "tail", "mixed"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fine_frozen(backend, D, mask_kind, cnt_kind, dtype, fwd_variant, bwd_variant):
    if backend == "tk" and (fwd_variant == 2 or bwd_variant != 0):
        pytest.skip("fwd/bwd variants apply to the generic kernels only")
    if backend == "tk" and dtype != torch.bfloat16:
        pytest.skip("TK path is bf16 only")
    gen = torch.Generator().manual_seed(D)
    B, H, nb = 1, 2, 9
    cnt = counts(nb, cnt_kind, gen)
    q, k, v, _ = inputs(B, H, nb, D, dtype, cnt, seed=1)
    mask = random_mask(B, H, nb, gen, mask_kind)
    sel = ops.mask_to_selection(mask)
    qq, kk, vv = (x.clone().requires_grad_(True) for x in (q, k, v))
    of, lse = ops.fine_block_sparse_attention(qq, kk, vv, cnt, sel, backend=backend)
    L = nb * 64
    valid = ref.token_valid(cnt, L)
    dout = torch.randn_like(q) * valid[None, None, :, None].to(dtype)
    dq, dk, dv = torch.autograd.grad(of, (qq, kk, vv), dout)
    qf, kf, vf = (x.float().clone().requires_grad_(True) for x in (q, k, v))
    of_r, lse_r = ref.fine_attention(qf, kf, vf, cnt, mask)
    dq_r, dk_r, dv_r = torch.autograd.grad(of_r, (qf, kf, vf), dout.float())
    rt = 2e-2 if dtype == torch.bfloat16 else 1e-2
    meta = dict(backend=backend, D=D, mask=mask_kind, cnt=cnt_kind, dtype=str(dtype), fwd_mt=fwd_variant, bwd=bwd_variant)
    rows = valid.nonzero().squeeze(1)
    check("fine_o", of, of_r, rt, rows, **meta)
    finite = torch.isfinite(lse_r[..., rows])
    assert torch.equal(torch.isfinite(lse[..., rows]), finite), "lse sentinel mismatch"
    check("fine_lse", torch.where(finite, lse[..., rows], 0), torch.where(finite, lse_r[..., rows], 0), 1e-2, **meta)
    check("fine_dq", dq, dq_r, 2 * rt, **meta)
    check("fine_dk", dk, dk_r, 2 * rt, **meta)
    check("fine_dv", dv, dv_r, 2 * rt, **meta)
    pad = (~valid).nonzero().squeeze(1)
    if pad.numel():
        for t in (dq, dk, dv):
            assert (t[..., pad, :] == 0).all(), "padded tokens must get zero gradient"


# --------------------------------------------------------------------------- full operator


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("gate_mode", ["token", "elem", "zero", "none"])
@pytest.mark.parametrize("cnt_kind", ["full", "mixed"])
def test_full_frozen(backend, D, gate_mode, cnt_kind, bwd_variant):
    if backend == "tk" and bwd_variant != 0:
        pytest.skip("bwd variant applies to the generic kernels only")
    gen = torch.Generator().manual_seed(11)
    B, H, nb = 2, 2, 10
    dtype = torch.bfloat16
    cnt = counts(nb, cnt_kind, gen)
    q, k, v, gate = inputs(B, H, nb, D, dtype, cnt, seed=2, gate_mode=gate_mode)
    mask = random_mask(B, H, nb, gen, "random")
    sel = ops.mask_to_selection(mask)
    leaves = [x.clone().requires_grad_(True) for x in (q, k, v)] + ([gate.clone().requires_grad_(True)] if gate is not None else [])
    o = ops.vsa2_attention(leaves[0], leaves[1], leaves[2], leaves[3] if gate is not None else None, cnt,
                           R=16, selection=sel, backend=backend)
    valid = ref.token_valid(cnt, nb * 64)
    dout = torch.randn_like(q)
    grads = torch.autograd.grad(o, leaves, dout)
    lf = [x.float().clone().requires_grad_(True) for x in (q, k, v)] + ([gate.float().clone().requires_grad_(True)] if gate is not None else [])
    o_r = ref.vsa2_reference(lf[0], lf[1], lf[2], lf[3] if gate is not None else None, cnt, mask)
    grads_r = torch.autograd.grad(o_r, lf, dout.float())
    meta = dict(backend=backend, D=D, gate=gate_mode, cnt=cnt_kind, bwd=bwd_variant)
    check("full_o", o, o_r, 2e-2, **meta)
    for name, g_, gr in zip(["dq", "dk", "dv", "dgate"], grads, grads_r):
        check(f"full_{name}", g_, gr, 4e-2, **meta)
    assert (o[..., (~valid).nonzero().squeeze(1), :] == 0).all()


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("R", [8, 16])
@pytest.mark.parametrize("cnt_kind", ["tail", "mixed"])
def test_full_routed(backend, R, cnt_kind, fwd_variant):
    gen = torch.Generator().manual_seed(R)
    B, H, nb, D, topk = 1, 4, 20, 128, 3
    cnt = counts(nb, cnt_kind, gen)
    q, k, v, gate = inputs(B, H, nb, D, torch.bfloat16, cnt, seed=7)
    leaves = [x.clone().requires_grad_(True) for x in (q, k, v, gate)]
    o, sel = ops.vsa2_attention(*leaves, cnt, R=R, topk=topk, backend=backend, return_selection=True)
    mask = sel.to_mask(nb)
    assert (mask.sum((-1, -2)) == nb * topk).all()
    dout = torch.randn_like(q)
    grads = torch.autograd.grad(o, leaves, dout)
    lf = [x.float().clone().requires_grad_(True) for x in (q, k, v, gate)]
    o_r = ref.vsa2_reference(*lf, cnt, mask)
    grads_r = torch.autograd.grad(o_r, lf, dout.float())
    meta = dict(backend=backend, R=R, cnt=cnt_kind, fwd_mt=fwd_variant, zero_edge_rows=int((mask.sum(-1) == 0).sum()))
    check("routed_o", o, o_r, 2e-2, **meta)
    for name, g_, gr in zip(["dq", "dk", "dv", "dgate"], grads, grads_r):
        check(f"routed_{name}", g_, gr, 4e-2, **meta)


@pytest.mark.parametrize("backend", BACKENDS)
def test_zero_edge_rows_coarse_only(backend):
    """Query blocks with no fine edge: O = tanh(g) * O_c exactly, finite grads."""
    gen = torch.Generator().manual_seed(3)
    B, H, nb, D = 1, 2, 6, 64
    cnt = counts(nb, "tail", gen)
    q, k, v, gate = inputs(B, H, nb, D, torch.bfloat16, cnt, seed=9)
    mask = torch.zeros(B, H, nb, nb, dtype=torch.bool, device=DEV)
    mask[:, :, 3, 3] = True  # only one query block has a fine edge
    sel = ops.mask_to_selection(mask)
    leaves = [x.clone().requires_grad_(True) for x in (q, k, v, gate)]
    o = ops.vsa2_attention(*leaves, cnt, selection=sel, backend=backend)
    grads = torch.autograd.grad(o, leaves, torch.randn_like(q))
    for g_ in grads:
        assert torch.isfinite(g_.float()).all()
    oc = ref.coarse_attention(q, k, v, cnt)
    blk = torch.arange(nb * 64, device=DEV) // 64
    exp = torch.tanh(gate.float())[..., None] * oc[:, :, blk]
    rows = ((blk != 3) & ref.token_valid(cnt, nb * 64)).nonzero().squeeze(1)
    check("zero_edge_o", o, exp, 2e-2, rows, backend=backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_dense_mask_gate_zero_is_dense_attention(backend):
    """Dense mask + gate 0 reduces to dense attention; nonzero gate must differ from it."""
    B, H, nb, D = 1, 2, 5, 128
    cnt = counts(nb, "full", None)
    q, k, v, _ = inputs(B, H, nb, D, torch.bfloat16, cnt, seed=4)
    sel = ops.mask_to_selection(torch.ones(B, H, nb, nb, dtype=torch.bool, device=DEV))
    dense = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float())
    o0 = ops.vsa2_attention(q, k, v, torch.zeros(B, H, nb * 64, dtype=q.dtype, device=DEV), cnt,
                            selection=sel, backend=backend)
    check("dense_gate0", o0, dense, 2e-2, backend=backend)
    o1 = ops.vsa2_attention(q, k, v, torch.full((B, H, nb * 64), 2.0, dtype=q.dtype, device=DEV), cnt,
                            selection=sel, backend=backend)
    assert (o1.float() - dense).abs().max() > 1e-2


def test_retain_graph_second_backward_raises():
    B, H, nb, D = 1, 1, 3, 64
    cnt = counts(nb, "full", None)
    q, k, v, gate = inputs(B, H, nb, D, torch.bfloat16, cnt, seed=6)
    q.requires_grad_(True)
    o = ops.vsa2_attention(q, k, v, gate, cnt, topk=1, backend="generic")
    o.sum().backward(retain_graph=True)
    with pytest.raises(RuntimeError):
        o.sum().backward()


# ------------------------------------------------- full-operator coverage (see audit/ROOT_TEST_COVERAGE_REVIEW.md)


def _compare_full(o, grads, leaves, cnt, mask, name, rt=2e-2, **meta):
    """O + every input gradient of a native run against the oracle on the same frozen mask."""
    dout = meta.pop("dout")
    lf = [x.detach().float().clone().requires_grad_(True) for x in leaves]
    gate_r = lf[3] if len(lf) > 3 else None
    o_r = ref.vsa2_reference(lf[0], lf[1], lf[2], gate_r, cnt, mask)
    grads_r = torch.autograd.grad(o_r, lf, dout.float())
    check(f"{name}_o", o, o_r, rt, **meta)
    for n, g_, gr in zip(["dq", "dk", "dv", "dgate"], grads, grads_r):
        check(f"{name}_{n}", g_, gr, 2 * rt, **meta)
    pad = (~ref.token_valid(cnt, o.shape[2])).nonzero().squeeze(1)
    if pad.numel():
        assert (o[..., pad, :] == 0).all(), "padded output must be zero"
        for g_ in grads[:3]:
            assert (g_[..., pad, :] == 0).all(), "padded tokens must get zero gradient"


def _boundary_counts(nb, gen):
    """Partial blocks around every 64-row coarse tile boundary and at the tail."""
    c = torch.full((nb,), 64, dtype=torch.int32)
    for i in range(nb):
        if i % 64 == 63 or i % 64 == 0:
            c[i] = int(torch.randint(1, 64, (1,), generator=gen))
    c[-1] = 37
    return c.to(DEV)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("R", [8, 16])
@pytest.mark.parametrize("row_norm", ["paper", "valid_groups"])
def test_full_routed_row_norm_partial_blocks(backend, R, row_norm, bwd_variant):
    """Gap 1: the requested normalization, partial blocks, through the complete routed operator."""
    if backend == "tk" and bwd_variant != 0:
        pytest.skip("bwd variant applies to the generic kernels only")
    gen = torch.Generator().manual_seed(100 + R)
    B, H, nb, D, topk = 1, 3, 24, 128, 3
    cnt = counts(nb, "mixed", gen)
    q, k, v, gate = inputs(B, H, nb, D, torch.bfloat16, cnt, seed=8)
    leaves = [x.clone().requires_grad_(True) for x in (q, k, v, gate)]
    o, sel = ops.vsa2_attention(*leaves, cnt, R=R, topk=topk, row_norm=row_norm, backend=backend, return_selection=True)
    mask = sel.to_mask(nb)
    assert (mask.sum((-1, -2)) == nb * topk).all(), "budget must be exactly NB*topk per (b,h)"
    assert torch.equal(mask, ops.route(q, k, cnt, R=R, topk=topk, row_norm=row_norm).to_mask(nb)), "row_norm not threaded"
    Sref = ref.router_scores(q, k, cnt, R, row_norm, round_to=torch.bfloat16)
    agree = (mask & ref.global_topk_mask(Sref, nb * topk)).sum().item() / mask.sum().item()
    log_metric("routed_row_norm_agreement", backend=backend, R=R, row_norm=row_norm, agreement=agree)
    assert agree > 0.97
    dout = torch.randn_like(q)
    grads = torch.autograd.grad(o, leaves, dout)
    _compare_full(o, grads, leaves, cnt, mask, "routed_norm", dout=dout, backend=backend, R=R, row_norm=row_norm,
                  zero_edge_rows=int((mask.sum(-1) == 0).sum()))


def test_paper_norm_is_default_and_differs_on_partial_blocks():
    gen = torch.Generator().manual_seed(21)
    nb, topk = 24, 3
    cnt = counts(nb, "mixed", gen)
    q, k, v, gate = inputs(1, 3, nb, 128, torch.bfloat16, cnt, seed=8)
    _, sel = ops.vsa2_attention(q, k, v, gate, cnt, topk=topk, backend="generic", return_selection=True)
    paper = ops.route(q, k, cnt, topk=topk, row_norm="paper").to_mask(nb)
    assert torch.equal(sel.to_mask(nb), paper), "operator default must be row_norm='paper'"
    assert torch.equal(ops.route(q, k, cnt, topk=topk).to_mask(nb), paper), "route default must be row_norm='paper'"
    assert not torch.equal(paper, ops.route(q, k, cnt, topk=topk, row_norm="valid_groups").to_mask(nb)), \
        "partial blocks must make the two normalizations select differently here"


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("nb", [64, 65, 130])
@pytest.mark.parametrize("mode", ["frozen", "routed"])
def test_full_multitile_coarse(backend, nb, mode, bwd_variant):
    """Gap 2: pooled coarse sequence crossing 64-row tiles, partial blocks at the boundaries and tail."""
    if backend == "tk" and bwd_variant != 0:
        pytest.skip("bwd variant applies to the generic kernels only")
    gen = torch.Generator().manual_seed(nb)
    B, H, D = 1, 2, 128
    cnt = _boundary_counts(nb, gen)
    q, k, v, gate = inputs(B, H, nb, D, torch.bfloat16, cnt, seed=12)
    leaves = [x.clone().requires_grad_(True) for x in (q, k, v, gate)]
    if mode == "frozen":
        mask = torch.rand(B, H, nb, nb, generator=gen).to(DEV) < 0.03
        mask[:, :, 1] = False
        o = ops.vsa2_attention(*leaves, cnt, R=16, selection=ops.mask_to_selection(mask), backend=backend)
    else:
        o, sel = ops.vsa2_attention(*leaves, cnt, R=16, topk=3, backend=backend, return_selection=True)
        mask = sel.to_mask(nb)
        assert (mask.sum((-1, -2)) == nb * 3).all()
    dout = torch.randn_like(q)
    grads = torch.autograd.grad(o, leaves, dout)
    _compare_full(o, grads, leaves, cnt, mask, "multitile", dout=dout, backend=backend, nb=nb, mode=mode, bwd=bwd_variant,
                  zero_edge_rows=int((mask.sum(-1) == 0).sum()))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("nb", [10, 70])
@pytest.mark.parametrize("cnt_kind", ["tail", "mixed"])
@pytest.mark.parametrize("gate_mode", ["token", "elem", "none"])
def test_all_zero_fine_mask_isolates_coarse(backend, nb, cnt_kind, gate_mode, bwd_variant):
    """Gap 3: nothing but coarse attention, repeat adjoint, 1/cnt mean-pool adjoint and dGate."""
    if backend == "tk" and bwd_variant != 0:
        pytest.skip("bwd variant applies to the generic kernels only")
    gen = torch.Generator().manual_seed(nb + 1)
    B, H, D = 2, 2, 64
    cnt = counts(nb, cnt_kind, gen)
    q, k, v, gate = inputs(B, H, nb, D, torch.bfloat16, cnt, seed=13, gate_mode=gate_mode)
    mask = torch.zeros(B, H, nb, nb, dtype=torch.bool, device=DEV)
    leaves = [x.clone().requires_grad_(True) for x in (q, k, v)] + ([gate.clone().requires_grad_(True)] if gate is not None else [])
    o = ops.vsa2_attention(leaves[0], leaves[1], leaves[2], leaves[3] if gate is not None else None, cnt,
                           selection=ops.mask_to_selection(mask), backend=backend)
    dout = torch.randn_like(q)
    grads = torch.autograd.grad(o, leaves, dout)
    for g_ in grads:
        assert g_.abs().max() > 0, "coarse branch must contribute a gradient to every input"
    _compare_full(o, grads, leaves, cnt, mask, "zero_fine", dout=dout, backend=backend, nb=nb, cnt_kind=cnt_kind, gate=gate_mode, bwd=bwd_variant)


@pytest.mark.parametrize("backend", BACKENDS)
def test_zero_budget_through_router(backend):
    """budget 0 runs the whole native router and yields an all-empty selection; coarse still contributes."""
    gen = torch.Generator().manual_seed(77)
    B, H, nb, D = 1, 2, 12, 128
    cnt = counts(nb, "mixed", gen)
    q, k, v, gate = inputs(B, H, nb, D, torch.bfloat16, cnt, seed=14)
    leaves = [x.clone().requires_grad_(True) for x in (q, k, v, gate)]
    o, sel = ops.vsa2_attention(*leaves, cnt, topk=0, backend=backend, return_selection=True)
    assert sel.q2k_idx.numel() == 0 and sel.k2q_idx.numel() == 0
    dout = torch.randn_like(q)
    grads = torch.autograd.grad(o, leaves, dout)
    _compare_full(o, grads, leaves, cnt, torch.zeros(B, H, nb, nb, dtype=torch.bool, device=DEV), "zero_budget",
                  dout=dout, backend=backend)


@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("mode", ["frozen", "routed"])
@pytest.mark.parametrize("cnt_kind", ["tail", "mixed"])
def test_full_fp16(D, mode, cnt_kind, bwd_variant):
    """FP16 through the complete operator (generic path; TK is bf16 only)."""
    gen = torch.Generator().manual_seed(D + 1)
    B, H, nb = 1, 2, 16
    cnt = counts(nb, cnt_kind, gen)
    q, k, v, gate = inputs(B, H, nb, D, torch.float16, cnt, seed=15)
    leaves = [x.clone().requires_grad_(True) for x in (q, k, v, gate)]
    if mode == "frozen":
        mask = random_mask(B, H, nb, gen, "random")
        o = ops.vsa2_attention(*leaves, cnt, selection=ops.mask_to_selection(mask), backend="generic")
    else:
        o, sel = ops.vsa2_attention(*leaves, cnt, topk=3, backend="generic", return_selection=True)
        mask = sel.to_mask(nb)
    dout = torch.randn_like(q)
    grads = torch.autograd.grad(o, leaves, dout)
    _compare_full(o, grads, leaves, cnt, mask, "full_fp16", dout=dout, D=D, mode=mode, cnt_kind=cnt_kind, bwd=bwd_variant)


@pytest.mark.parametrize("backend", BACKENDS)
def test_real_video_layout_partial_tiles(backend):
    """A real cube-major layout (non-multiple T, H, W): raster <-> blocked round trip, then the full operator."""
    from echosparse.vsa import layout

    lay = layout.build_layout((6, 9, 10), R=16, device=DEV)
    B, H, D = 1, 2, 128
    n = 6 * 9 * 10
    assert (lay.blk_cnt < 64).any() and int(lay.blk_cnt.sum()) == n
    g = torch.Generator().manual_seed(16)
    raster = [torch.randn(B, H, n, D, generator=g).to(DEV).to(torch.bfloat16) for _ in range(3)]
    q, k, v = (layout.to_blocked(x, lay).contiguous() for x in raster)
    assert torch.equal(layout.from_blocked(q, lay), raster[0])
    gate = (torch.randn(B, H, lay.padded_len, generator=g) * 1.5).to(DEV).to(torch.bfloat16)
    leaves = [x.clone().requires_grad_(True) for x in (q, k, v, gate)]
    o, sel = ops.vsa2_attention(*leaves, lay.blk_cnt, R=16, topk=4, backend=backend, return_selection=True)
    dout = torch.randn_like(q)
    grads = torch.autograd.grad(o, leaves, dout)
    _compare_full(o, grads, leaves, lay.blk_cnt, sel.to_mask(lay.num_blocks), "real_layout", dout=dout, backend=backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_budget_overflow_clamps_to_dense(backend):
    B, H, nb, D = 1, 2, 6, 128
    cnt = counts(nb, "tail", None)
    q, k, v, _ = inputs(B, H, nb, D, torch.bfloat16, cnt, seed=17)
    gate = torch.zeros(B, H, nb * 64, dtype=q.dtype, device=DEV)
    o, sel = ops.vsa2_attention(q, k, v, gate, cnt, topk=nb + 100, backend=backend, return_selection=True)
    assert sel.to_mask(nb).all()
    of_r, _ = ref.fine_attention(q, k, v, cnt, torch.ones(B, H, nb, nb, dtype=torch.bool, device=DEV))
    check("budget_overflow_o", o, of_r, 2e-2, backend=backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_head_and_sample_isolation(backend):
    """Perturbing one (sample, head) changes nothing else: selection, output, dK, dV."""
    gen = torch.Generator().manual_seed(18)
    B, H, nb, D = 2, 3, 20, 64
    cnt = counts(nb, "mixed", gen)
    q, k, v, gate = inputs(B, H, nb, D, torch.bfloat16, cnt, seed=18)
    o0, s0 = ops.vsa2_attention(q, k, v, gate, cnt, topk=3, backend=backend, return_selection=True)
    q2, k2 = q.clone(), k.clone()
    q2[1, 2] = q2[1, 2].flip(1)
    k2[1, 2] = -k2[1, 2]
    o1, s1 = ops.vsa2_attention(q2, k2, v, gate, cnt, topk=3, backend=backend, return_selection=True)
    m0, m1 = s0.to_mask(nb), s1.to_mask(nb)
    keep = torch.ones(B, H, dtype=torch.bool, device=DEV)
    keep[1, 2] = False
    assert torch.equal(m0[keep], m1[keep]) and torch.equal(o0[keep], o1[keep])
    assert not torch.equal(o0[1, 2], o1[1, 2])


@pytest.mark.parametrize("backend", BACKENDS)
def test_deterministic_rerun(backend, bwd_variant):
    if backend == "tk" and bwd_variant != 0:
        pytest.skip("bwd variant applies to the generic kernels only")
    gen = torch.Generator().manual_seed(19)
    B, H, nb, D = 1, 2, 18, 128
    cnt = counts(nb, "mixed", gen)
    q, k, v, gate = inputs(B, H, nb, D, torch.bfloat16, cnt, seed=19)
    dout = torch.randn_like(q)
    runs = []
    for _ in range(2):
        leaves = [x.clone().requires_grad_(True) for x in (q, k, v, gate)]
        o, sel = ops.vsa2_attention(*leaves, cnt, topk=3, backend=backend, return_selection=True)
        runs.append((o.detach(), sel.to_mask(nb), torch.autograd.grad(o, leaves, dout)))
    assert torch.equal(runs[0][1], runs[1][1]), "selection must be deterministic"
    assert torch.equal(runs[0][0], runs[1][0]), "output must be deterministic"
    for i in (1, 2, 3):  # dK, dV, dGate
        assert torch.equal(runs[0][2][i], runs[1][2][i])
    if bwd_variant == 2:  # no atomics in the fine branch. The small coarse dQ_c still uses them, so allow its rounding.
        check("deterministic_dq", runs[0][2][0], runs[1][2][0], 1e-3, backend=backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_non_default_stream(backend):
    gen = torch.Generator().manual_seed(20)
    B, H, nb, D = 1, 2, 12, 128
    cnt = counts(nb, "tail", gen)
    q, k, v, gate = inputs(B, H, nb, D, torch.bfloat16, cnt, seed=20)
    leaves = [x.clone().requires_grad_(True) for x in (q, k, v, gate)]
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        o, sel = ops.vsa2_attention(*leaves, cnt, topk=3, backend=backend, return_selection=True)
        dout = torch.randn_like(q)
        grads = torch.autograd.grad(o, leaves, dout)
    s.synchronize()
    _compare_full(o, grads, leaves, cnt, sel.to_mask(nb), "stream", dout=dout, backend=backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_output_ownership_and_input_versioning(backend):
    """Overwriting the returned O must not change gradients; mutating saved inputs must raise."""
    gen = torch.Generator().manual_seed(23)
    B, H, nb, D = 1, 2, 10, 64
    cnt = counts(nb, "tail", gen)
    q, k, v, gate = inputs(B, H, nb, D, torch.bfloat16, cnt, seed=23)
    dout = torch.randn_like(q)
    sel = ops.mask_to_selection(random_mask(B, H, nb, gen, "random"))
    leaves = [x.clone().requires_grad_(True) for x in (q, k, v, gate)]
    g0 = torch.autograd.grad(ops.vsa2_attention(*leaves, cnt, selection=sel, backend=backend), leaves, dout)
    o = ops.vsa2_attention(*leaves, cnt, selection=sel, backend=backend)
    o.detach().fill_(123.0)
    g1 = torch.autograd.grad(o, leaves, dout)
    for a, b in zip(g0[1:], g1[1:]):  # dQ differs only by atomic summation order
        assert torch.equal(a, b)
    check("ownership_dq", g1[0], g0[0], 1e-2, backend=backend)
    leaves = [x.clone().requires_grad_(True) for x in (q, k, v, gate)]
    o = ops.vsa2_attention(*leaves, cnt, selection=sel, backend=backend)
    with torch.no_grad():
        leaves[1].add_(1.0)
    with pytest.raises(RuntimeError):
        torch.autograd.grad(o, leaves, dout)


def test_no_triton_in_native_path():
    """Production sources name no Triton; the operator modules import only torch, the stdlib and each other."""
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    files = [p for d in ("csrc/vsa", "echosparse/vsa") for p in (root / d).rglob("*")
             if p.suffix in {".cu", ".cuh", ".h", ".cpp", ".py"}]
    assert files
    assert not [str(p) for p in files if "triton" in p.read_text().lower()]
    for p in files:
        if p.suffix == ".py":
            for node in ast.walk(ast.parse(p.read_text())):
                if isinstance(node, ast.ImportFrom) and node.level == 0:
                    assert node.module.split(".")[0] in {"__future__", "torch", "dataclasses", "typing"}, (p, node.module)
                elif isinstance(node, ast.Import):
                    assert all(a.name.split(".")[0] in {"math", "os", "torch"} for a in node.names), p


def _small_full(backend="generic", **kw):
    gen = torch.Generator().manual_seed(31)
    B, H, nb, D = 1, 2, 8, 64
    cnt = counts(nb, "tail", gen)
    q, k, v, gate = inputs(B, H, nb, D, torch.bfloat16, cnt, seed=31)
    leaves = [x.clone().requires_grad_(True) for x in (q, k, v, gate)]
    return leaves, cnt, ops.vsa2_attention(*leaves, cnt, topk=2, backend=backend, **kw)


def test_bwd_variant_precedence(monkeypatch):
    monkeypatch.delenv("ECHOSPARSE_VSA_BWD_VARIANT", raising=False)
    node = lambda **kw: _small_full(**kw)[2].grad_fn
    n = node()
    assert (n.bwd_variant, n.bwd_source) == (0, "auto"), "small shapes stay below the automatic threshold"
    monkeypatch.setenv("ECHOSPARSE_VSA_BWD_VARIANT", "2")
    n = node()
    assert (n.bwd_variant, n.bwd_source) == (2, "environment")
    try:
        ops.set_bwd_variant(1)
        n = node()
        assert (n.bwd_variant, n.bwd_source) == (1, "set_bwd_variant")
        n = node(bwd_variant=0)
        assert (n.bwd_variant, n.bwd_source) == (0, "call")
    finally:
        ops.set_bwd_variant(None)
    monkeypatch.setenv("ECHOSPARSE_VSA_BWD_VARIANT", "9")
    with pytest.raises(ValueError):
        node()


@pytest.mark.parametrize("first,later", [(0, 2), (2, 0)])
def test_bwd_choice_fixed_at_forward(first, later):
    """A global change between forward and backward must not alter the backward path or its result."""
    leaves, cnt, o = _small_full(bwd_variant=first)
    assert o.grad_fn.bwd_variant == first
    dout = torch.randn_like(leaves[0])
    leaves2, _, o2 = _small_full(bwd_variant=first)
    expected = torch.autograd.grad(o2, leaves2, dout)
    try:
        ops.set_bwd_variant(later)
        got = torch.autograd.grad(o, leaves, dout)
    finally:
        ops.set_bwd_variant(None)
    for i in (1, 2, 3):  # dK, dV, dGate have a fixed summation order
        assert torch.equal(got[i], expected[i])
    check("fixed_choice_dq", got[0], expected[0], 1e-3, first=first)


def test_tk_ignores_generic_bwd_variant():
    if "tk" not in BACKENDS:
        pytest.skip("needs the SM90 TK backend")
    _, _, o = _small_full(backend="tk", bwd_variant=2)
    assert (o.grad_fn.bwd_variant, o.grad_fn.bwd_source) == (None, "tk")
