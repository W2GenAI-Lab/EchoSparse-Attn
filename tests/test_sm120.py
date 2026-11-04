"""RTX 5090 kernels and per-layer state: python -m pytest tests"""
from types import SimpleNamespace

import pytest
import torch
import triton

from echosparse import EchoSparseAttention
from echosparse.kernels import sm120, triton_ops

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12,
                                reason="needs an SM120 GPU")


def inputs(L, H, n, seed=0):
    """n denoising steps of local, slowly changing q / k / v"""
    g = torch.Generator(device="cuda").manual_seed(seed)
    pos = torch.arange(L, device="cuda", dtype=torch.float32)[:, None, None] / 300.0
    freq = torch.randn(1, H, 128, device="cuda", generator=g)
    base = [torch.sin(pos * freq + torch.randn(1, H, 128, device="cuda", generator=g)) * 3.0 for _ in range(3)]
    return [[(b + 0.05 * torch.randn(b.shape, device="cuda", generator=g)).to(torch.bfloat16)[None] for b in base]
            for _ in range(n)]


def same(a, b):
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def sparse_inputs(L, H, density=0.1, sink=None, rows=None):
    """q / k / v of the first sparse step and what that step's kernel reads: the plan's block, mean and refine
    lists, the block means, the representatives and the offset."""
    attn = EchoSparseAttention(density=density, dense_steps=2, dense_layers=())
    steps = inputs(L, H, 3)
    for step in range(2):
        attn(*steps[step], step, 0, layer=1, sink=sink, rows=rows)
    q, k, v = (x[0] for x in steps[2])
    st, buf = attn._states[1], attn._buffers
    nq, nb = triton.cdiv(L, 128), triton.cdiv(L, 64)
    _, refined = attn._budget(nb)
    exact, skip = st.plan(st.pending[0])
    keep = torch.empty(H, nq, nb, device="cuda", dtype=torch.uint8)
    counts = torch.empty(nq, H, device="cuda", dtype=torch.int32)
    blocks = torch.zeros(nq, H, nb, device="cuda", dtype=torch.int32)
    refine = torch.empty(nq, H, triton.cdiv(refined, 32) * 32, device="cuda", dtype=torch.int32)
    refine_counts, means, mean_counts = torch.empty_like(counts), torch.empty_like(blocks), torch.empty_like(counts)
    sm120.rebuild_plan(exact, skip, keep, counts, blocks, refine, refine_counts, means, mean_counts)
    buf.reps.build(q, k, v, buf.k_mean, buf.v_mean)
    return SimpleNamespace(q=q, k=k, v=v, keep=keep, counts=counts, blocks=blocks, refine=refine,
                           refine_counts=refine_counts, means=means, mean_counts=mean_counts, offset=st.offset(0),
                           buf=buf)


def run_kernel(x, rows=None):
    nq, nb, H = triton.cdiv(x.q.shape[0], 128), x.buf.k_mean.shape[0], x.q.shape[1]
    out, lse = torch.full_like(x.q, 7.0), torch.full_like(x.offset, 7.0)
    flags = torch.full((nq, H), 7, device="cuda", dtype=torch.int32)
    mass = torch.zeros(H, nq, nb, device="cuda")
    reps = x.buf.reps
    sm120.sparse_attention(x.q, x.k, x.v, x.buf.k_mean, x.buf.v_mean, x.means, x.mean_counts, out, x.counts,
                           x.blocks, x.offset, lse, flags, x.refine, x.refine_counts, reps.k, reps.v, reps.log_counts,
                           mass=mass, rows=rows)
    return out, lse, flags, mass


def test_fused_block_means_and_representatives():
    L, H = 4152, 3
    q, k, v = (x[0] for x in inputs(L, H, 1)[0])
    nb = triton.cdiv(L, 64)
    reps = sm120.Representatives(L, H, 128, "cuda")
    k_mean, v_mean = (torch.empty(nb, H, 128, device="cuda", dtype=torch.bfloat16) for _ in range(2))
    reps.build(q, k, v, k_mean, v_mean)
    km, vm = torch.empty_like(k_mean), torch.empty_like(v_mean)
    triton_ops.block_means(k, v, km, vm)
    assert same(k_mean, km) and same(v_mean, vm)
    # the four representatives of a block, weighted by their cluster sizes, add up to the block's sum
    counts = reps.log_counts[:, :4 * nb].view(H, nb, 4).exp2()
    sizes = torch.full((nb,), 64.0, device="cuda")
    sizes[-1] = L - 64 * (nb - 1)
    assert torch.allclose(counts.sum(-1), sizes.expand(H, nb), atol=1e-3)
    for rep, mean in ((reps.k, k_mean), (reps.v, v_mean)):
        total = (rep[:, :4 * nb].view(H, nb, 4, 128).float() * counts[..., None]).sum(2)
        ref = mean.float().permute(1, 0, 2) * sizes[:, None]
        assert ((total - ref).norm() / ref.norm()).item() < 1e-2


def test_estimate_matches_shared_kernel():
    x = sparse_inputs(4152, 2)
    _, lse, _, mass = run_kernel(x)
    outs = []
    for fn in (sm120.estimate_skipped, triton_ops.estimate_skipped):
        m, se, ss = mass.clone(), torch.empty(mass.shape[:2], device="cuda"), torch.empty(mass.shape[:2], device="cuda")
        fn(x.q, x.buf.k_mean, x.keep, x.offset, lse, m, se, ss)
        outs.append((m, se, ss))
    assert all(same(a, b) for a, b in zip(*outs))


def test_rows_skip_only_unused_tiles():
    L = 4152
    x = sparse_inputs(L, 2, sink=60)
    full = run_kernel(x, rows=L)                  # every tile
    rows = 4000                                   # 32 tiles; tile 31 holds rows 3968-4095 and is computed whole
    part = run_kernel(x, rows=rows)
    assert same(part[0][:3968 + 128], full[0][:3968 + 128]) and same(part[1][:4096], full[1][:4096])
    assert same(part[2][:32], full[2][:32]) and same(part[3][:, :32], full[3][:, :32])
    assert part[0][4096:].abs().max() == 0 and same(part[1][4096:], x.offset[4096:])
    assert (part[2][32:] == 0).all()


def test_plan_bit_masks_rebuild_the_selection():
    L, H = 4152, 3
    nq, nb = triton.cdiv(L, 128), triton.cdiv(L, 64)
    words = max(triton.next_power_of_2(nb) // 32, 4 * triton.cdiv(nb, 128))
    g = torch.Generator(device="cuda").manual_seed(0)
    mass = torch.rand(H, nq, nb, device="cuda", generator=g)
    keep_prev = (torch.rand(H, nq, nb, device="cuda", generator=g) < 0.1).to(torch.uint8)
    ones = torch.ones(H, nq, device="cuda")
    flags = torch.zeros(nq, H, device="cuda", dtype=torch.int32)
    keep = torch.empty(H, nq, nb, device="cuda", dtype=torch.uint8)
    counts = torch.empty(nq, H, device="cuda", dtype=torch.int32)
    blocks = torch.zeros(nq, H, nb, device="cuda", dtype=torch.int32)
    skip = torch.empty(H, nq, words, device="cuda", dtype=torch.int32)
    refine = torch.empty(nq, H, 32, device="cuda", dtype=torch.int32)
    triton_ops.select_blocks(mass, keep_prev, ones, ones, keep, counts, blocks, flags, skip, refine, 7, 20, L,
                             sink=60)
    exact = torch.empty_like(skip)
    sm120.pack_plan(keep, exact)
    keep2, counts2, blocks2, refine2, refine_counts, means, mean_counts = (
        torch.full_like(x, 7) for x in (keep, counts, blocks, refine, counts, blocks, counts))
    sm120.rebuild_plan(exact, skip, keep2, counts2, blocks2, refine2, refine_counts, means, mean_counts)
    used = torch.arange(nb, device="cuda") < counts[..., None]
    assert same(keep2, keep) and same(counts2, counts) and same(refine2, refine)
    assert torch.equal(blocks2[used], blocks[used])
    assert torch.equal(refine_counts, (refine < nb).sum(-1, dtype=torch.int32))
    # the mean list: the blocks that are neither exact nor refined, ascending
    bits = (skip.permute(1, 0, 2)[..., :, None] >> torch.arange(32, device="cuda")) & 1
    mean = bits.flatten(-2)[..., :nb] == 0
    assert torch.equal(mean_counts, mean.sum(-1, dtype=torch.int32))
    ids = torch.arange(nb, device="cuda").expand_as(mean)
    listed = torch.arange(nb, device="cuda") < mean_counts[..., None]
    assert torch.equal(means[listed], ids[mean])


def persistent_bytes(st):
    return sum(x.numel() * x.element_size() for x in [*st.offsets.values(), *(t for p in st.plans for t in p)])


def run_steps(attn, L, H, n, branches, sink=None):
    steps = [inputs(L, H, n, seed=3 + b) for b in range(branches)]
    for video in range(2):
        attn.reset()
        for step in range(n):
            for branch in range(branches):
                attn(*steps[branch][step], step, branch, 1, sink=sink)
    return attn._states[1]


def test_plans_with_cfg():
    """Wan2.1 form: two CFG branches, the second reusing the first's plan; two plans per layer"""
    L, H = 4152, 2
    st = run_steps(EchoSparseAttention(density=0.1, dense_steps=2, dense_layers=()), L, H, 6, branches=2)
    nq, words = triton.cdiv(L, 128), st.words
    assert sorted(st.offsets) == [0, 1] and st.ready == {0, 1} and len(st.plans) == 2 and st.pending[1] == 6
    assert persistent_bytes(st) == 2 * nq * 128 * H * 4 + 4 * H * nq * words * 4


def test_plans_single_branch():
    """HunyuanVideo form: one branch, a text sink at the end; one plan per layer, rewritten in place"""
    L, H = 4152, 2
    st = run_steps(EchoSparseAttention(density=0.1, dense_steps=2, dense_layers=()), L, H, 6, branches=1,
                   sink=4096 // 64)
    nq, words = triton.cdiv(L, 128), st.words
    assert sorted(st.offsets) == [0] and len(st.plans) == 1 and st.pending == (0, 6)
    assert persistent_bytes(st) == nq * 128 * H * 4 + 2 * H * nq * words * 4


def test_h3_rotary_matches_eager():
    """diffusers' MiniMax-H3 `_apply_rotary_emb`: BF16 cos / sin and products, half-split over 96 of 128 channels"""
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(1, 3001, 4, 128, device="cuda", generator=g).to(torch.bfloat16)
    cos, sin = (torch.rand(3001, 96, device="cuda", generator=g) * 2 - 1 for _ in range(2))
    rot, rest = x[..., :96], x[..., 96:]
    c, s = cos.to(x.dtype)[None, :, None, :], sin.to(x.dtype)[None, :, None, :]
    x1, x2 = rot.chunk(2, dim=-1)
    ref = torch.cat((rot * c + torch.cat((-x2, x1), dim=-1) * s, rest), dim=-1)
    assert same(sm120.h3_rotary(x, cos, sin), ref)
