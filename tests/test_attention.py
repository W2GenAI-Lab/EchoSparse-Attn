"""Correctness of the public attention on an H100 (SM90), B200 (SM100) or RTX 5090 (SM120) GPU:
python -m pytest tests"""
import math

import pytest
import torch
import triton

from echosparse import EchoSparseAttention
from echosparse.attention import dense_attention
from echosparse.models.common import grid_morton_order

pytestmark = pytest.mark.skipif(not torch.cuda.is_available()
                                or torch.cuda.get_device_capability()[0] not in (9, 10, 12),
                                reason="needs an SM90, SM100 or SM120 GPU")

L, H, D = 4152, 2, 128          # 65 kv blocks (the last has 56 tokens), 33 query tiles (the last has 56 rows)


def steps(n, seed=0):
    """Inputs of n consecutive denoising steps: a fixed base plus a small change per step, with local structure."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    pos = torch.arange(L, device="cuda", dtype=torch.float32)[:, None, None] / 300.0
    freq = torch.randn(1, H, D, device="cuda", generator=g)
    base = [torch.sin(pos * freq + torch.randn(1, H, D, device="cuda", generator=g)) * 3.0 for _ in range(3)]
    out = []
    for _ in range(n):
        out.append([(b + 0.05 * torch.randn(b.shape, device="cuda", generator=g)).to(torch.bfloat16)[None]
                    for b in base])
    return out


def rel_err(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def run(attn, inputs, branches=1):
    outs = []
    for step, (q, k, v) in enumerate(inputs):
        for branch in range(branches):
            outs.append(attn(q, k, v, step, branch, layer=1))
    return outs


def offset_of(attn, branch=0, layer=1):
    """The offset (previous log2 LSE) that the next call of (branch, layer) uses."""
    if attn.backend == "b200":
        return attn._states[(branch, layer)].offset
    return attn._states[layer].offset(branch)


def call_with_selection(attn, q, k, v, step, branch=0, layer=1):
    """One sparse call; returns its output and the block selection it used (counts, blocks, refine list)."""
    if attn.backend == "b200":
        out = attn(q, k, v, step, branch, layer=layer)
        st = attn._states[(0, layer)]
        return out, (st.counts.clone(), st.blocks.clone(), st.refine.clone())
    st = attn._states[layer]                      # the plan is rewritten during the call: take it before
    exact, skip = (x.clone() for x in st.plan(st.pending[0] if branch == 0 else st.current[0]))
    out = attn(q, k, v, step, branch, layer=layer)
    H, nq, _ = exact.shape
    nb = triton.cdiv(q.shape[1], 64)
    keep = torch.empty(H, nq, nb, device="cuda", dtype=torch.uint8)
    counts = torch.empty(nq, H, device="cuda", dtype=torch.int32)
    blocks = torch.zeros(nq, H, nb, device="cuda", dtype=torch.int32)
    refine = torch.empty_like(attn._plan_scratch.refine)
    lists = torch.empty_like(counts), torch.empty_like(blocks), torch.empty_like(counts)
    attn.K.rebuild_plan(exact, skip, keep, counts, blocks, refine, *lists)
    return out, (counts, blocks, refine)


def check_formula(out, qkv, selection, buf):
    """Exact tokens of the listed blocks, block means of the other skipped blocks and four representatives of each
    refined block, all in one softmax."""
    out = out[0].float()
    q, k, v = (x[0].float() for x in qkv)
    counts, blocks, refine = selection
    nb = buf.k_mean.shape[0]
    block_len = torch.full((nb,), 64.0, device="cuda")
    block_len[-1] = L - 64 * (nb - 1)
    scale = D ** -0.5
    for h in range(H):
        for t in (0, 7, 32):
            exact = blocks[t, h, :counts[t, h]].tolist()
            refined = [b for b in refine[t, h].tolist() if b < nb]
            means = sorted(set(range(nb)) - set(exact) - set(refined))
            rows = torch.arange(128 * t, min(128 * t + 128, L), 9, device="cuda")
            keys, vals, bias = [], [], []
            for b in exact:
                tok = torch.arange(64 * b, min(64 * b + 64, L), device="cuda")
                keys.append(k[tok, h]); vals.append(v[tok, h]); bias.append(torch.zeros(len(tok), device="cuda"))
            keys.append(buf.k_mean[means, h].float()); vals.append(buf.v_mean[means, h].float())
            bias.append(block_len[means].log())
            for b in refined:
                sl = slice(4 * b, 4 * b + 4)
                keys.append(buf.reps.k[h, sl].float()); vals.append(buf.reps.v[h, sl].float())
                bias.append(buf.reps.log_counts[h, sl] * math.log(2))
            logits = q[rows, h] @ torch.cat(keys).T * scale + torch.cat(bias)
            ref = torch.softmax(logits, -1) @ torch.cat(vals)
            assert rel_err(out[rows, h], ref) < 1e-2, (h, t)


def test_full_density_matches_dense():
    inputs = steps(4)
    attn = EchoSparseAttention(density=1.0, dense_steps=2, dense_layers=())
    outs = run(attn, inputs)
    for (q, k, v), out in zip(inputs, outs):
        assert rel_err(out, dense_attention(q, k, v)) < 1e-2


def test_cfg_branches_share_selection():
    inputs = steps(4)
    attn = EchoSparseAttention(density=1.0, dense_steps=2, dense_layers=())
    outs = run(attn, inputs, branches=2)
    for i, (q, k, v) in enumerate(inputs):
        ref = dense_attention(q, k, v)
        assert rel_err(outs[2 * i], ref) < 1e-2 and rel_err(outs[2 * i + 1], ref) < 1e-2


def test_sparse_step_matches_formula():
    inputs = steps(3)
    attn = EchoSparseAttention(density=0.1, dense_steps=2, dense_layers=())
    run(attn, inputs[:2])
    out, selection = call_with_selection(attn, *inputs[2], 2)
    check_formula(out, inputs[2], selection, attn._buffers)


def test_second_cfg_branch_matches_formula():
    """The unconditional branch uses the conditional branch's block selection with its own inputs and offsets."""
    cond, uncond = steps(3, seed=1), steps(3, seed=2)
    attn = EchoSparseAttention(density=0.1, dense_steps=2, dense_layers=())
    for step in range(2):
        attn(*cond[step], step, 0, layer=1)
        attn(*uncond[step], step, 1, layer=1)
    _, first = call_with_selection(attn, *cond[2], 2, branch=0)
    out, second = call_with_selection(attn, *uncond[2], 2, branch=1)
    assert all(torch.equal(a, b) for a, b in zip(first, second))
    check_formula(out, uncond[2], second, attn._buffers)


def test_fallback_tiles():
    """A far-off offset flags every tile; the online-softmax fallback must give the same result."""
    inputs = steps(4)
    attn = EchoSparseAttention(density=1.0, dense_steps=2, dense_layers=())
    run(attn, inputs[:3])
    offset_of(attn)[:L] += 200.0
    q, k, v = inputs[3]
    out = attn(q, k, v, 3, 0, layer=1)
    assert rel_err(out, dense_attention(q, k, v)) < 1e-2


def test_morton_order_is_a_permutation():
    perm, inv = grid_morton_order(5, 6, 7, "cuda")
    assert torch.equal(perm[inv], torch.arange(5 * 6 * 7, device="cuda"))
    assert perm[:8].tolist() == [0, 1, 7, 8, 42, 43, 49, 50]
