"""RTX 5090 kernels against the shared Triton kernels and against complete lists: python -m pytest tests"""
import pytest
import torch
import triton

from echosparse import EchoSparseAttention
from echosparse.kernels import representatives, sm120, triton_ops

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


def test_kmeans_and_block_means_match_shared_kernels():
    L, H = 4152, 3
    q, k, v = (x[0] for x in inputs(L, H, 1)[0])
    nb = triton.cdiv(L, 64)
    reps = sm120.Representatives(L, H, 128, "cuda")
    k_mean, v_mean = (torch.empty(nb, H, 128, device="cuda", dtype=torch.bfloat16) for _ in range(2))
    reps.build(q, k, v, k_mean, v_mean)
    ref = representatives.Representatives(L, H, 128, "cuda")
    ref._metric_and_probes(q)
    ref._metric.copy_(reps._metric)               # the Gram matrix is summed with atomics: share one metric
    ref._probes.copy_(reps._probes)
    representatives._kmeans_kernel[(nb, H)](k, v, ref._metric, ref._probes, ref.k, ref.v, ref.log_counts, L, nb,
                                            ref.k.shape[1], H=H, D=128, BK=64, ITERS=6, NP=64, num_warps=4)
    assert same(reps.k, ref.k) and same(reps.v, ref.v) and same(reps.log_counts, ref.log_counts)
    km, vm = torch.empty_like(k_mean), torch.empty_like(v_mean)
    triton_ops.block_means(k, v, km, vm)
    assert same(k_mean, km) and same(v_mean, vm)


def run_to_sparse(attn, L, H, sink=None, rows=None):
    steps = inputs(L, H, attn.dense_steps + 1)
    for step, (q, k, v) in enumerate(steps):
        attn(q, k, v, step, 0, layer=1, sink=sink, rows=rows)
    return [x[0] for x in steps[-1]], attn._states[(0, 1)], attn._buffers


def kernel_inputs(q, k, v, st, buf):
    reps = buf.reps
    return (q, k, v, buf.k_mean, buf.v_mean, st.skip_bits, None, st.counts, st.blocks, st.offset, None, None,
            st.refine, reps.k, reps.v, reps.log_counts)


def run_kernel(args, rows=None, refined=None, mass_shape=None):
    q, offset = args[0], args[9]
    out, lse = torch.full_like(q, 7.0), torch.full_like(offset, 7.0)
    flags = torch.full((triton.cdiv(q.shape[0], 128), q.shape[1]), 7, device="cuda", dtype=torch.int32)
    mass = torch.zeros(mass_shape, device="cuda")
    a = list(args)
    a[6], a[10], a[11] = out, lse, flags
    sm120.sparse_attention(*a, mass=mass, rows=rows, refined=refined)
    return out, lse, flags, mass


def test_estimate_matches_single_tile_kernel():
    L, H = 4152, 2
    attn = EchoSparseAttention(density=0.1, dense_steps=2, dense_layers=())
    (q, k, v), st, buf = run_to_sparse(attn, L, H)
    outs = []
    for fn in (sm120.estimate_skipped, triton_ops.estimate_skipped):
        mass, se, ss = st.mass.clone(), torch.empty_like(st.scale_exact), torch.empty_like(st.scale_skip)
        fn(q, buf.k_mean, st.keep, st.offset, st.next_offset, mass, se, ss)
        outs.append((mass, se, ss))
    assert all(same(a, b) for a, b in zip(*outs))


def test_rows_skip_only_unused_tiles():
    L, H = 4152, 2
    attn = EchoSparseAttention(density=0.1, dense_steps=2, dense_layers=())
    (q, k, v), st, buf = run_to_sparse(attn, L, H, sink=60)
    args = kernel_inputs(q, k, v, st, buf)
    full = run_kernel(args, rows=L, mass_shape=st.mass.shape)          # every tile, complete lists
    rows = 4000                                   # 32 tiles; tile 31 holds rows 3968-4095 and is computed whole
    part = run_kernel(args, rows=rows, refined=st.refined, mass_shape=st.mass.shape)
    assert same(part[0][:3968 + 128], full[0][:3968 + 128]) and same(part[1][:4096], full[1][:4096])
    assert same(part[2][:32], full[2][:32]) and same(part[3][:, :32], full[3][:, :32])
    assert part[0][4096:].abs().max() == 0 and same(part[1][4096:], st.offset[4096:])
    assert (part[2][32:] == 0).all()


@pytest.mark.parametrize("density, refined", [(0.24, 151), (0.20, 159)])
def test_h3_padding_skip_matches_complete_lists(density, refined):
    """MiniMax-H3 shape: 37296 video rows, then 951 prefix rows that the caller overwrites (598 kv blocks)."""
    L, H, nv = 38247, 2, 37296
    attn = EchoSparseAttention(density=density, dense_steps=2, dense_layers=())
    (q, k, v), st, buf = run_to_sparse(attn, L, H, sink=nv // 64, rows=nv)
    assert (st.refined, st.refine.shape[-1]) == (refined, 160)
    args = kernel_inputs(q, k, v, st, buf)
    skipped = run_kernel(args, rows=nv, refined=st.refined, mass_shape=st.mass.shape)
    complete = run_kernel(args, rows=nv, mass_shape=st.mass.shape)          # never leaves anything out
    assert all(same(a, b) for a, b in zip(skipped, complete))
    tail = st.refine[:triton.cdiv(nv, 128), :, 152:]
    if refined == 151:
        assert bool((tail == 598).all())            # the entries the kernel leaves out are the sentinel block
    else:
        assert bool((tail < 598).any())             # real blocks: the kernel must run the complete list


def test_wide_kernel_matches_halves_kernel():
    """The 64-wide kernel (Wan2.1, HunyuanVideo) and the 32-token-halves kernel (MiniMax-H3) differ in FP32 order."""
    L, H = 4152, 2
    attn = EchoSparseAttention(density=0.1, dense_steps=2, dense_layers=())
    (q, k, v), st, buf = run_to_sparse(attn, L, H, sink=60)
    args = kernel_inputs(q, k, v, st, buf)
    wide, halves = run_kernel(args, mass_shape=st.mass.shape), run_kernel(args, rows=L, mass_shape=st.mass.shape)
    assert (wide[0].float() - halves[0].float()).norm() / halves[0].float().norm() < 1e-2
    assert (wide[1][:L] - halves[1][:L]).abs().max() < 1e-3
    assert torch.allclose(wide[3], halves[3], rtol=1e-3, atol=1e-6)


def test_compact_plan_rebuilds_the_selection():
    L, H = 4152, 3
    attn = EchoSparseAttention(density=0.1, dense_steps=2, dense_layers=())
    _, st, _ = run_to_sparse(attn, L, H, sink=60)
    exact = torch.empty_like(st.skip_bits)
    sm120.pack_plan(st.keep, exact)
    keep, counts, blocks, refine = (torch.full_like(x, 7) for x in (st.keep, st.counts, st.blocks, st.refine))
    sm120.rebuild_plan(exact, st.skip_bits, keep, counts, blocks, refine)
    used = torch.arange(blocks.shape[-1], device="cuda") < counts[..., None]
    assert same(keep, st.keep) and same(counts, st.counts) and same(refine, st.refine)
    assert torch.equal(blocks[used], st.blocks[used])


def compare_compact(L, H, steps, branches, sink=None):
    """full and compact state side by side over dense, full-block and sparse steps; then a second video"""
    full = EchoSparseAttention(density=0.1, dense_steps=2, dense_layers=())
    compact = EchoSparseAttention(density=0.1, dense_steps=2, dense_layers=(), compact=True)
    for video in range(2):
        for attn in (full, compact):
            attn.reset()
        for step in range(len(steps[0])):
            for branch in range(branches):
                qkv = steps[branch][step]
                a, b = full(*qkv, step, branch, 1, sink=sink), compact(*qkv, step, branch, 1, sink=sink)
                err = (a.float() - b.float()).norm() / a.float().norm()
                assert err < 2e-2, (video, step, branch, float(err))
    return compact._states[1]


def persistent_bytes(st):
    return sum(x.numel() * x.element_size() for x in [*st.offsets.values(), *(t for p in st.plans for t in p)])


def test_compact_state_follows_full_state_with_cfg():
    """Wan2.1 form: two CFG branches, the second reusing the first's plan; two plans per layer"""
    L, H = 4152, 2
    st = compare_compact(L, H, [inputs(L, H, 6, seed=3), inputs(L, H, 6, seed=4)], branches=2)
    nq, words = triton.cdiv(L, 128), st.words
    assert sorted(st.offsets) == [0, 1] and st.ready == {0, 1} and len(st.plans) == 2 and st.pending[1] == 6
    assert persistent_bytes(st) == 2 * nq * 128 * H * 4 + 4 * H * nq * words * 4


def test_compact_state_follows_full_state_single_branch():
    """HunyuanVideo form: one branch, a text sink at the end; one plan per layer, rewritten in place"""
    L, H = 4152, 2
    st = compare_compact(L, H, [inputs(L, H, 6, seed=5)], branches=1, sink=4096 // 64)
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
