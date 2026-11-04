"""Native original VSA (echosparse.vsa.vsa_attention) and the real-token layout adapter.

The router is checked bit for bit against the original's Torch formula and tie order, the operator against an
FP64 oracle with the original's rounding points (vsa_reference.vsa_reference) on the native mask. The tests marked
`gpu` need the ECHOSPARSE_VSA_BUILD=1 extension; the rest run on the CPU.
Run:  pytest -q tests/test_vsa.py tests/test_vsa2.py
"""
import subprocess
import sys

import pytest
import torch

from echosparse.vsa import layout, ops
import vsa_reference as ref
from echosparse.vsa.dispatch import auto_bwd_variant
from test_vsa2 import BACKENDS, DEV, _built, check, counts, inputs, log_metric, random_mask

gpu = pytest.mark.skipif(not torch.cuda.is_available() or not _built(),
                         reason="needs a CUDA GPU and the ECHOSPARSE_VSA_BUILD=1 extension")
DTYPES = [torch.bfloat16, torch.float16]


def ulp16(x, dtype):
    """One unit in the last place of x in a 16-bit float type."""
    mant = 7 if dtype == torch.bfloat16 else 10
    return torch.exp2(torch.floor(torch.log2(x.float().abs().clamp(min=2.0 ** -14))) - mant)


def leaves_of(*xs):
    return [x.clone().requires_grad_(True) for x in xs if x is not None]


def oracle(leaves, cnt, mask, dout, dtype=torch.bfloat16):
    """FP64 original-VSA output and gradients of the given leaves (q, k, v[, weight])."""
    lf = [x.detach().double().clone().requires_grad_(True) for x in leaves]
    o, logits = ref.vsa_reference(lf[0], lf[1], lf[2], lf[3] if len(lf) > 3 else None, cnt, mask, dtype)
    return o, torch.autograd.grad(o, lf, dout.double()), logits


def compare(name, o, grads, leaves, blk_cnt, mask, dout, rt=2e-2, **meta):
    """meta is logged only; it may hold a `cnt` (count kind) label, so the counts are `blk_cnt` here."""
    o_r, grads_r, _ = oracle(leaves, blk_cnt, mask, dout, leaves[0].dtype)
    rows = ref.token_valid(blk_cnt, o.shape[2]).nonzero().squeeze(1)
    check(f"{name}_o", o, o_r, rt, rows, **meta)
    for n, g, gr in zip(["dq", "dk", "dv", "dw"], grads, grads_r):
        check(f"{name}_{n}", g, gr, 2 * rt, rows, **meta)
    pad = (~ref.token_valid(blk_cnt, o.shape[2])).nonzero().squeeze(1)
    if pad.numel():
        assert (o[..., pad, :] == 0).all(), "padded output must be zero"
        for g in grads:
            assert (g[..., pad, :] == 0).all(), "padded slots must get zero gradient"


def masked_dout(q, cnt):
    return torch.randn_like(q) * ref.token_valid(cnt, q.shape[2])[None, None, :, None].to(q.dtype)


# --------------------------------------------------------------------------- CPU


def test_import_does_not_load_the_extension():
    code = "import sys, echosparse.vsa; assert 'echosparse.vsa._C' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_reference_row_topk_ties_and_signed_zero():
    s = torch.tensor([[[0.5, 0.0, -0.0, 0.5, 0.25, 0.0]]])
    assert ref.row_topk_mask(s, 3).tolist() == [[[True, False, False, True, True, False]]]
    assert ref.row_topk_mask(s, 4).tolist() == [[[True, True, False, True, True, False]]]
    assert ref.row_topk_mask(s, 5).tolist() == [[[True, True, True, True, True, False]]], "-0 ties with +0"
    assert ref.row_topk_mask(s, 6).all() and not ref.row_topk_mask(s, 0).any()
    assert ref.row_topk_mask(s, 99).all()


def test_reference_empty_fine_mask_is_coarse_only():
    """B1 H1 L64 D64, Q = K = V = 1, no weight, no kv block: O = O_c = 1 exactly, finite gradients."""
    q, k, v = (torch.ones(1, 1, 64, 64, dtype=torch.float64, requires_grad=True) for _ in range(3))
    cnt = torch.tensor([64], dtype=torch.int32)
    o, _ = ref.vsa_reference(q, k, v, None, cnt, torch.zeros(1, 1, 1, 1, dtype=torch.bool))
    assert torch.equal(o, torch.ones_like(o))
    for g in torch.autograd.grad(o.sum(), (q, k, v)):
        assert torch.isfinite(g).all()


@pytest.mark.parametrize("mask_kind", ["random", "empty"])
def test_reference_oracles_agree_without_rounding(mask_kind):
    """Two independently written oracles: original VSA in FP64 with no rounding == the VSA2 oracle with a full
    identity weight on the same mask, outputs and gradients; "empty" leaves only the coarse branch."""
    g = torch.Generator().manual_seed(0)
    B, H, nb, D = 1, 2, 5, 64
    cnt = torch.tensor([64, 1, 64, 37, 20], dtype=torch.int32)
    valid = ref.token_valid(cnt, nb * 64)[None, None, :, None].double()
    leaves = [(torch.randn(B, H, nb * 64, D, generator=g, dtype=torch.float64) * valid).requires_grad_(True)
              for _ in range(4)]
    mask = torch.rand(B, H, nb, nb, generator=g) < 0.5
    mask[..., 0] = True
    mask[:, :, 2] = False  # a query block with no kv block
    if mask_kind == "empty":
        mask[:] = False
    o1, _ = ref.vsa_reference(*leaves, cnt, mask, dtype=torch.float64)
    o2 = ref.vsa2_reference(*leaves, cnt, mask, gate_act="identity", round_pooled=False)
    torch.testing.assert_close(o1, o2.double(), rtol=1e-4, atol=1e-4)
    dout = torch.randn(o1.shape, generator=g, dtype=torch.float64)
    for g1, g2 in zip(torch.autograd.grad(o1, leaves, dout), torch.autograd.grad(o2, leaves, dout.float())):
        assert torch.isfinite(g1).all()
        torch.testing.assert_close(g1, g2.double(), rtol=1e-4, atol=1e-4)


def test_layout_round_trip_and_padding_contract():
    for lay in (layout.build_layout((6, 9, 10), R=16), layout.sequence_layout(300)):
        n = lay.scatter.numel()
        assert int(lay.blk_cnt.sum()) == n and lay.padded_len == lay.num_blocks * 64
        x = torch.randn(1, 2, n, 8, dtype=torch.float64, requires_grad=True)
        gate = torch.randn(1, 2, n)
        xb = layout.to_blocked(x, lay)
        assert (xb[:, :, lay.gather < 0] == 0).all()
        assert torch.equal(layout.from_blocked(xb, lay), x)
        assert torch.equal(layout.from_blocked(layout.to_blocked(gate, lay), lay), gate)
        valid = ref.token_valid(lay.blk_cnt, lay.padded_len)
        assert torch.equal(valid, lay.gather >= 0), "valid tokens first inside every block"
        (xb * torch.arange(lay.padded_len, dtype=torch.float64)[None, None, :, None]).sum().backward()
        assert torch.equal(x.grad[0, 0, :, 0], lay.scatter.double())


def test_auto_bwd_variant_rule():
    assert auto_bwd_variant(1, 20, 245760, 128, 12) == 2
    assert auto_bwd_variant(1, 20, 245760, 128, 9) == 0
    assert auto_bwd_variant(1, 20, 98880, 128, 12) == 0


# --------------------------------------------------------------------------- router


@gpu
@pytest.mark.parametrize("nb", [1, 7, 32, 33, 130, 1100])
@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("dtype", DTYPES)
def test_row_topk_exact(nb, ties, dtype):
    gen = torch.Generator().manual_seed(nb)
    BH = 3
    if ties:  # few distinct values, both signed zeros
        scores = (torch.randint(-2, 3, (BH, nb, nb), generator=gen).float() / 4.0)
        scores = torch.where(torch.rand(BH, nb, nb, generator=gen) < 0.3, -scores, scores)
    else:
        scores = torch.randn(BH, nb, nb, generator=gen) * 3
    scores = scores.to(dtype).to(DEV)
    C = ops.ext()
    for topk in sorted({0, 1, 3, nb - 1, nb, nb + 5}):
        if topk < 0:
            continue
        K = min(topk, nb)
        q2k_ptr = torch.empty(1, BH, nb + 1, dtype=torch.int32, device=DEV)
        k2q_ptr = torch.empty_like(q2k_ptr)
        q2k_idx = torch.empty(max(BH * nb * K, 1), dtype=torch.int32, device=DEV)
        k2q_idx = torch.empty_like(q2k_idx)
        C.row_topk_into(scores, topk, 0, q2k_ptr, q2k_idx, k2q_ptr, k2q_idx)
        sel = ops.Selection(q2k_ptr, q2k_idx[: BH * nb * K], k2q_ptr, k2q_idx[: BH * nb * K])
        exp = ref.row_topk_mask(scores.reshape(1, BH, nb, nb), topk)
        got = sel.to_mask(nb)
        assert torch.equal(got, exp), f"row top-k mismatch nb={nb} topk={topk} ties={ties}"
        assert torch.equal(sel.col_mask(nb), exp), "k2q inversion mismatch"
        assert (sel.row_counts() == K).all()
        ptr = (torch.arange(BH, device=DEV)[:, None] * nb + torch.arange(nb + 1, device=DEV)[None, :]) * K
        assert torch.equal(q2k_ptr.reshape(BH, nb + 1).long(), ptr), "K entries per row"
        if K > 1:
            idx = sel.q2k_idx.long().reshape(BH * nb, K)
            assert bool((idx[:, 1:] > idx[:, :-1]).all()), "q->kv lists must ascend"


@gpu
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kind", ["full", "mixed"])
def test_block_scores_are_the_original_formula(D, dtype, kind):
    """The score kernel against the original's Torch score op (matmul, / sqrt(D)) on the SAME pooled means; the
    pooling itself is checked by test_pool_means_against_the_exact_mean."""
    gen = torch.Generator().manual_seed(D)
    nb = 70
    cnt = counts(nb, kind, gen)
    q, k, _, _ = inputs(2, 2, nb, D, dtype, cnt, seed=3)
    C = ops.ext()
    _, _, qc, kc, _ = C.pool_fwd(q, k, k, cnt, 32)
    S = C.block_scores(qc, kc, nb, 0, 0).reshape(2, 2, nb, nb)
    Sref = ref.block_scores_from_means(qc[:, :, :nb], kc[:, :, :nb])
    assert S.dtype == Sref.dtype == dtype
    diff = (S.float() - Sref.float()).abs()
    ulp = ulp16(torch.maximum(S.float().abs(), Sref.float().abs()), dtype)
    frac = float((diff > 0).float().mean())
    log_metric("vsa_block_scores", D=D, dtype=str(dtype), kind=kind, mismatch_fraction=frac,
               max_ulps=float((diff / ulp).max()))
    assert (diff <= ulp * 1.01).all(), "scores may differ from cuBLAS only by fp32 summation order"
    assert frac < 0.02
    # integer-valued means: every dot product is exact, so with D=64 (scale 1/8) the scores must be identical
    if D == 64:
        qi, ki = (torch.randint(-3, 4, (2, 2, nb * 64, D), generator=gen).to(dtype).to(DEV) for _ in range(2))
        full = torch.full_like(cnt, 64)
        _, _, qc, kc, _ = C.pool_fwd(qi, ki, ki, full, 32)
        mm = torch.backends.cuda.matmul
        flags = mm.allow_bf16_reduced_precision_reduction, mm.allow_fp16_reduced_precision_reduction
        try:  # exact fp32 accumulation in cuBLAS too
            mm.allow_bf16_reduced_precision_reduction = mm.allow_fp16_reduced_precision_reduction = False
            exp = ref.vsa_block_scores(qi, ki, full)
        finally:
            mm.allow_bf16_reduced_precision_reduction, mm.allow_fp16_reduced_precision_reduction = flags
        assert torch.equal(C.block_scores(qc, kc, nb, 0, 0).reshape(2, 2, nb, nb), exp)


@gpu
@pytest.mark.parametrize("R", [16, 32])
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kind", ["full", "mixed"])
def test_pool_means_against_the_exact_mean(R, D, dtype, kind):
    """Whole-block means of pool_fwd against the FP64 mean of the real tokens. The kernel rounds an fp32 sum of
    cnt <= 64 terms / cnt once to the token dtype, so |native - exact| <= ulp/2 + the fp32 error: the sum is off by
    at most (cnt - 1) * 2^-24 * sum|x|, the division adds 2^-24 relative, together <= 64 * 2^-24 * mean|x| =
    2^-18 * mean|x|; 2^-17 leaves a factor 2. Rounding differently from the exact mean happens only next to a
    rounding boundary, so such elements must stay rare."""
    gen = torch.Generator().manual_seed(R + D)
    nb = 70
    cnt = counts(nb, kind, gen)
    q, k, v, _ = inputs(2, 2, nb, D, dtype, cnt, seed=3)
    _, _, qc, kc, vc = ops.ext().pool_fwd(q, k, v, cnt, R)
    valid = ref.token_valid(cnt, nb * 64)[None, None, :, None].double()
    c = cnt.double()[None, None, :, None]
    for name, x, m in (("q", q, qc), ("k", k, kc), ("v", v, vc)):
        xs = (x.double() * valid).reshape(2, 2, nb, 64, D)
        exact, mabs = xs.sum(3) / c, xs.abs().sum(3) / c
        got = m[:, :, :nb].double()
        assert (m[:, :, nb:] == 0).all(), "pooled rows past NB stay zero"
        err = (got - exact).abs()
        bound = 0.5 * ulp16(torch.maximum(got.abs(), exact.abs()), dtype) * 1.01 + 2.0 ** -17 * mabs
        frac = float((got != exact.to(dtype).double()).float().mean())
        log_metric("vsa_pool_means", tensor=name, R=R, D=D, dtype=str(dtype), kind=kind, misrounded_fraction=frac,
                   max_err_over_bound=float((err / bound).max()))
        assert (err <= bound).all(), f"{name}: pooled mean outside the fp32-accumulation bound"
        assert frac < 0.01, f"{name}: {frac:.4f} of the means not the correctly rounded exact mean"


@gpu
@pytest.mark.parametrize("nb", [24, 65, 130])
@pytest.mark.parametrize("cnt_kind", ["tail", "mixed"])
def test_route_rows_exact_and_deterministic(nb, cnt_kind):
    gen = torch.Generator().manual_seed(nb + 5)
    topk = 5
    cnt = counts(nb, cnt_kind, gen)
    q, k, _, _ = inputs(2, 3, nb, 128, torch.bfloat16, cnt, seed=4)
    C = ops.ext()
    sel = ops.route_rows(q, k, cnt, topk)
    m = sel.to_mask(nb)
    chunked = ops.route_rows(q, k, cnt, topk, max_score_bytes=nb * nb * 2)  # one (b,h) per chunk
    assert torch.equal(m, chunked.to_mask(nb)) and torch.equal(m, ops.route_rows(q, k, cnt, topk).to_mask(nb))
    assert torch.equal(sel.col_mask(nb), m)
    assert (m.sum(-1) == topk).all(), "every query block keeps exactly topk kv blocks"
    _, _, qc, kc, _ = C.pool_fwd(q, k, k, cnt, 32)
    S = C.block_scores(qc, kc, nb, 0, 0).reshape(2, 3, nb, nb)
    assert torch.equal(m, ref.row_topk_mask(S, topk)), "exact top-k of the kernel's own scores"
    # rank order under the original's score op on the same means: disagreements only within one ulp
    Ssame = ref.block_scores_from_means(qc[:, :, :nb], kc[:, :, :nb]).float()
    lo = Ssame.masked_fill(~m, float("inf")).min(-1).values
    hi = Ssame.masked_fill(m, float("-inf")).max(-1).values
    assert (lo >= hi - ulp16(hi, torch.bfloat16) * 1.01).all(), "rank order of the original score op"
    # end to end against the whole original Torch formula, whose own means may differ by an ulp
    exp = ref.row_topk_mask(ref.vsa_block_scores(q, k, cnt), topk)
    agree = float((m & exp).sum()) / float(m.sum())
    log_metric("route_rows_original_agreement", nb=nb, cnt=cnt_kind, agreement=agree)
    assert agree > 0.98


@gpu
@pytest.mark.parametrize("backend", BACKENDS)
def test_equal_scores_select_the_smallest_kv_blocks(backend):
    """Zero queries make every score 0 (both signs): the original's tie order keeps kv blocks 0..K-1."""
    B, H, nb, D, topk = 1, 2, 9, 128, 3
    cnt = counts(nb, "tail", None)
    q, k, v, w = inputs(B, H, nb, D, torch.bfloat16, cnt, seed=5, gate_mode="elem")
    q = torch.zeros_like(q)
    leaves = leaves_of(q, k, v, w)
    o, sel = ops.vsa_attention(*leaves[:3], cnt, topk, weight=leaves[3], backend=backend, return_selection=True)
    exp = (torch.arange(nb, device=DEV) < topk).expand(B, H, nb, nb)
    assert torch.equal(sel.to_mask(nb), exp)
    dout = masked_dout(q, cnt)
    compare("vsa_ties", o, torch.autograd.grad(o, leaves, dout), leaves, cnt, exp, dout, backend=backend)


# --------------------------------------------------------------------------- operator


@gpu
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("weight", ["full", "none"])
@pytest.mark.parametrize("cnt_kind", ["full", "mixed"])
def test_vsa_frozen(backend, D, weight, cnt_kind, bwd_variant):
    if backend == "tk" and bwd_variant != 0:
        pytest.skip("bwd variant applies to the generic kernels only")
    gen = torch.Generator().manual_seed(D + 3)
    B, H, nb = 2, 2, 10
    cnt = counts(nb, cnt_kind, gen)
    q, k, v, w = inputs(B, H, nb, D, torch.bfloat16, cnt, seed=6, gate_mode="elem" if weight == "full" else "none")
    mask = random_mask(B, H, nb, gen, "random")
    leaves = leaves_of(q, k, v, w)
    o = ops.vsa_attention(*leaves[:3], cnt, 3, weight=leaves[3] if w is not None else None,
                          selection=ops.mask_to_selection(mask), backend=backend)
    dout = masked_dout(q, cnt)
    compare("vsa_frozen", o, torch.autograd.grad(o, leaves, dout), leaves, cnt, mask, dout, backend=backend, D=D,
            weight=weight, cnt=cnt_kind, bwd=bwd_variant)


@gpu
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("nb", [10, 65, 130])
@pytest.mark.parametrize("cnt_kind", ["tail", "mixed"])
def test_vsa_routed(backend, nb, cnt_kind, bwd_variant):
    """The complete operator: per-row selection, the original's rank order, O and every gradient."""
    if backend == "tk" and bwd_variant != 0:
        pytest.skip("bwd variant applies to the generic kernels only")
    gen = torch.Generator().manual_seed(nb + 7)
    B, H, D, topk = 1, 2, 128, 4
    cnt = counts(nb, cnt_kind, gen)
    q, k, v, w = inputs(B, H, nb, D, torch.bfloat16, cnt, seed=7, gate_mode="elem")
    leaves = leaves_of(q, k, v, w)
    o, sel = ops.vsa_attention(*leaves[:3], cnt, topk, weight=leaves[3], backend=backend, return_selection=True)
    mask = sel.to_mask(nb)
    assert (mask.sum(-1) == topk).all()
    assert torch.equal(mask, ops.route_rows(q, k, cnt, topk).to_mask(nb))
    dout = masked_dout(q, cnt)
    grads = torch.autograd.grad(o, leaves, dout)
    compare("vsa_routed", o, grads, leaves, cnt, mask, dout, backend=backend, nb=nb, cnt=cnt_kind, bwd=bwd_variant)
    _, _, qc, kc, _ = ops.ext().pool_fwd(q, k, k, cnt, 32)
    Ssame = ref.block_scores_from_means(qc[:, :, :nb], kc[:, :, :nb]).float()
    lo = Ssame.masked_fill(~mask, float("inf")).min(-1).values
    hi = Ssame.masked_fill(mask, float("-inf")).max(-1).values
    assert (lo >= hi - ulp16(hi, torch.bfloat16) * 1.01).all(), "rank order of the original score op, same means"
    _, _, logits = oracle(leaves, cnt, mask, dout)  # FP64 means: may round an ulp apart near cancellation
    agree = float((mask & ref.row_topk_mask(logits, topk)).sum()) / float(mask.sum())
    log_metric("vsa_routed_oracle_agreement", backend=backend, nb=nb, cnt=cnt_kind, agreement=agree)
    assert agree > 0.98


@gpu
@pytest.mark.parametrize("backend", BACKENDS)
def test_backward_uses_the_forward_selection(backend):
    """Gradients of a routed call equal those of the same call on its returned (frozen) selection."""
    gen = torch.Generator().manual_seed(8)
    B, H, nb, D = 1, 2, 18, 128
    cnt = counts(nb, "mixed", gen)
    q, k, v, w = inputs(B, H, nb, D, torch.bfloat16, cnt, seed=8, gate_mode="elem")
    dout = masked_dout(q, cnt)
    a = leaves_of(q, k, v, w)
    o, sel = ops.vsa_attention(*a[:3], cnt, 3, weight=a[3], backend=backend, return_selection=True, bwd_variant=2)
    ga = torch.autograd.grad(o, a, dout)
    b = leaves_of(q, k, v, w)
    gb = torch.autograd.grad(ops.vsa_attention(*b[:3], cnt, 3, weight=b[3], selection=sel, backend=backend,
                                               bwd_variant=2), b, dout)
    for i in (1, 2, 3):
        assert torch.equal(ga[i], gb[i])
    check("frozen_bwd_dq", ga[0], gb[0], 1e-3, backend=backend)


@gpu
@pytest.mark.parametrize("backend", BACKENDS)
def test_full_weight_matches_scalar_tanh_gate(backend):
    """W = tanh(z) expanded over D through vsa_attention == VSA2's scalar gate z on the same selection; dz is
    (1 - tanh^2) * sum_d dW. Checks the two gate modes against each other through the shared fusion kernels."""
    gen = torch.Generator().manual_seed(9)
    B, H, nb, D = 1, 2, 12, 64
    cnt = counts(nb, "mixed", gen)
    q, k, v, z = inputs(B, H, nb, D, torch.bfloat16, cnt, seed=9)
    sel = ops.route_rows(q, k, cnt, 3)
    dout = masked_dout(q, cnt)
    a = leaves_of(q, k, v, z)
    o2 = ops.vsa2_attention(*a, cnt, selection=sel, backend=backend)
    g2 = torch.autograd.grad(o2, a, dout)
    w = torch.tanh(z.float()).to(z.dtype)[..., None].expand(B, H, nb * 64, D).contiguous()
    b = leaves_of(q, k, v, w)
    o1 = ops.vsa_attention(*b[:3], cnt, 3, weight=b[3], selection=sel, backend=backend)
    g1 = torch.autograd.grad(o1, b, dout)
    check("gate_modes_o", o1, o2, 1e-2, backend=backend)
    for n, x, y in zip(["dq", "dk", "dv"], g1, g2):
        check(f"gate_modes_{n}", x, y, 1e-2, backend=backend)
    dz = (1 - w[..., 0].float() ** 2) * g1[3].float().sum(-1)
    check("gate_modes_dz", dz, g2[3], 2e-2, backend=backend)


@gpu
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("weight", ["full", "none"])
def test_coarse_branch_alone(backend, weight):
    """topk=0 runs the whole router and leaves only O = W * O_c: the coarse attention, its repeat and pool adjoints."""
    gen = torch.Generator().manual_seed(10)
    B, H, nb, D = 2, 2, 70, 64
    cnt = counts(nb, "mixed", gen)
    q, k, v, w = inputs(B, H, nb, D, torch.bfloat16, cnt, seed=10, gate_mode="elem" if weight == "full" else "none")
    leaves = leaves_of(q, k, v, w)
    o, sel = ops.vsa_attention(*leaves[:3], cnt, 0, weight=leaves[3] if w is not None else None, backend=backend,
                               return_selection=True)
    assert sel.q2k_idx.numel() == 0 and sel.k2q_idx.numel() == 0
    dout = masked_dout(q, cnt)
    grads = torch.autograd.grad(o, leaves, dout)
    for g in grads:
        assert g.abs().max() > 0, "the coarse branch reaches every input"
    compare("vsa_coarse", o, grads, leaves, cnt, torch.zeros(B, H, nb, nb, dtype=torch.bool, device=DEV), dout,
            backend=backend, weight=weight)


@gpu
@pytest.mark.parametrize("D", [64, 128])
def test_vsa_fp16(D, bwd_variant):
    gen = torch.Generator().manual_seed(D + 11)
    B, H, nb = 1, 2, 16
    cnt = counts(nb, "mixed", gen)
    q, k, v, w = inputs(B, H, nb, D, torch.float16, cnt, seed=11, gate_mode="elem")
    leaves = leaves_of(q, k, v, w)
    o, sel = ops.vsa_attention(*leaves[:3], cnt, 3, weight=leaves[3], backend="generic", return_selection=True)
    mask = sel.to_mask(nb)
    _, _, qc, kc, _ = ops.ext().pool_fwd(q, k, k, cnt, 32)
    assert torch.equal(mask, ref.row_topk_mask(ops.ext().block_scores(qc, kc, nb, 0, 0).reshape(B, H, nb, nb), 3))
    dout = masked_dout(q, cnt)
    compare("vsa_fp16", o, torch.autograd.grad(o, leaves, dout), leaves, cnt, mask, dout, D=D, bwd=bwd_variant)


@gpu
@pytest.mark.parametrize("backend", BACKENDS)
def test_vsa_and_vsa2_are_distinct_algorithms(backend):
    gen = torch.Generator().manual_seed(12)
    B, H, nb, D, topk = 1, 3, 20, 128, 3
    cnt = counts(nb, "mixed", gen)
    q, k, v, z = inputs(B, H, nb, D, torch.bfloat16, cnt, seed=12)
    _, s1 = ops.vsa_attention(q, k, v, cnt, topk, backend=backend, return_selection=True)
    _, s2 = ops.vsa2_attention(q, k, v, z, cnt, R=16, topk=topk, backend=backend, return_selection=True)
    m1, m2 = s1.to_mask(nb), s2.to_mask(nb)
    assert (m1.sum(-1) == topk).all() and (m2.sum((-1, -2)) == nb * topk).all()
    assert not (m2.sum(-1) == topk).all(), "VSA2's global budget is not a per-row K"
    with pytest.raises(ValueError):
        ops.vsa_attention(q, k, v, cnt, topk, weight=z)  # the original weight is a full [B, H, L, D] tensor


# --------------------------------------------------------------------------- real-token layout


@gpu
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("shape", [(6, 9, 10), (300,)])
@pytest.mark.parametrize("algo", ["vsa", "vsa2"])
def test_real_token_layout_autograd(backend, shape, algo):
    """Real tokens [B, H, N, D] -> to_blocked -> operator -> from_blocked; O and the gradients of the real tokens
    (partial cubes / a 1-D tail) against the oracles composed with the same permutation."""
    lay = layout.build_layout(shape, R=16, device=DEV) if len(shape) == 3 else layout.sequence_layout(shape[0], DEV)
    B, H, D, topk = 1, 2, 128, 3
    n = lay.scatter.numel()
    g = torch.Generator().manual_seed(13)
    real = [torch.randn(B, H, n, D, generator=g).to(DEV).to(torch.bfloat16) for _ in range(3)]
    gate = (torch.randn(B, H, n, D if algo == "vsa" else 1, generator=g) * 1.5).to(DEV).to(torch.bfloat16)
    gate = gate if algo == "vsa" else gate[..., 0]
    leaves = leaves_of(*real, gate)
    qb, kb, vb, gb = (layout.to_blocked(x, lay) for x in leaves)
    if algo == "vsa":
        ob, sel = ops.vsa_attention(qb, kb, vb, lay.blk_cnt, topk, weight=gb, backend=backend, return_selection=True)
    else:
        ob, sel = ops.vsa2_attention(qb, kb, vb, gb, lay.blk_cnt, R=16, topk=topk, backend=backend,
                                     return_selection=True)
    o = layout.from_blocked(ob, lay)
    dout = torch.randn_like(o)
    grads = torch.autograd.grad(o, leaves, dout)
    assert [tuple(x.shape) for x in grads] == [tuple(x.shape) for x in leaves]
    mask = sel.to_mask(lay.num_blocks)
    lf = [x.detach().double().requires_grad_(True) for x in leaves]
    blocked = [layout.to_blocked(x, lay) for x in lf]
    if algo == "vsa":
        o_r = ref.vsa_reference(*blocked, lay.blk_cnt, mask)[0]
    else:
        o_r = ref.vsa2_reference(*(x.float() for x in blocked), lay.blk_cnt, mask)
    o_r = layout.from_blocked(o_r, lay)
    grads_r = torch.autograd.grad(o_r, lf, dout.to(o_r.dtype))
    meta = dict(backend=backend, shape=shape, algo=algo)
    check("layout_o", o, o_r, 2e-2, **meta)
    for name, x, y in zip(["dq", "dk", "dv", "dgate"], grads, grads_r):
        check(f"layout_{name}", x, y, 4e-2, **meta)


@pytest.fixture(params=[0, 1, 2], ids=["bwdF2", "bwdWord", "bwdSplit"])
def bwd_variant(request):
    ops.set_bwd_variant(request.param)
    yield request.param
    ops.set_bwd_variant(None)
