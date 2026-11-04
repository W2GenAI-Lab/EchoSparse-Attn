"""LSE-guided block-sparse self-attention.

Between two denoising steps, the log-sum-exp (LSE) of every attention row barely changes. Each (CFG branch, layer)
keeps the previous step's LSE and uses it three ways:

1. as a fixed softmax offset, so the kernel needs no running maximum and no rescaling;
2. with that offset the kernel's exp values are already close to probabilities, so it sums them per kv block
   while computing the selected blocks, which gives the block probabilities that select the next step's blocks;
3. the skipped blocks are compensated inside the same softmax: every skipped block by its mean key and value,
   and the most probable third of them by four clustered representatives each.

Schedule of a (branch, layer): FA4 for the first `dense_steps - 1` steps, then one step with every block through
the fixed-offset kernel (exact output and the first block-probability map), then sparse steps that compute the
top `density` blocks of every 128-row query tile.

The kernels run on H100 (SM90), B200 (SM100) and RTX 5090 (SM120) GPUs; `backends` picks them from the device of q.
They keep different state between steps; the computation is the same. The B200 keeps, for every (branch, layer), the
block-probability map, the block lists and the scales, and selects a step's blocks at its start. The RTX 5090
(32 GB) and the H100 keep per layer only the offsets of each branch and the next step's selection as two bit masks
per tile, made at the end of each step; the map, the lists and the scales are scratch shared by all layers.
"""
import torch
import triton

from . import backends

BM = 128
BK = 64
LOG2E = 1.4426950408889634
MIN_BLOCKS = 64          # sequences shorter than 64 kv blocks (4096 tokens) run dense


def dense_attention(q, k, v, return_lse=False):
    out = backends.flash_attn_func(q.device)(q, k, v, return_lse=return_lse)
    return out if return_lse else out[0] if isinstance(out, tuple) else out


class _LayerState:
    """B200 state of one (branch, layer)."""

    def __init__(self, L, H, device):
        nq, nb = triton.cdiv(L, BM), triton.cdiv(L, BK)
        self.L = L
        self.offsets = [torch.full((nq * BM, H), 1e30, device=device) for _ in range(2)]
        self.cur = 0
        self.mass = torch.zeros(H, nq, nb, device=device)
        self.keeps = [torch.ones(H, nq, nb, device=device, dtype=torch.uint8) for _ in range(2)]
        self.kc = 0
        self.scale_exact = torch.ones(H, nq, device=device)
        self.scale_skip = torch.ones(H, nq, device=device)
        self.counts = torch.empty(nq, H, device=device, dtype=torch.int32)
        self.blocks = torch.zeros(nq, H, nb, device=device, dtype=torch.int32)
        self.flags = torch.zeros(nq, H, device=device, dtype=torch.int32)
        words = max(triton.next_power_of_2(nb) // 32, 4 * triton.cdiv(nb, 128))
        self.skip_bits = torch.full((H, nq, words), -1, device=device, dtype=torch.int32)
        self.refine = torch.empty(nq, H, 0, device=device, dtype=torch.int32)
        self.ready = False

    @property
    def offset(self):
        return self.offsets[self.cur]

    @property
    def next_offset(self):
        return self.offsets[1 - self.cur]

    @property
    def keep(self):
        return self.keeps[self.kc]


class _CompactState:
    """RTX 5090 / H100 state of one layer: the offsets of every branch and the plans (exact and skip bit masks). One
    branch rewrites its plan in place once the step has used it; with CFG branch 0 writes the other of two plans while
    later branches still read the current one."""

    def __init__(self, L, H, device):
        nb = triton.cdiv(L, BK)
        self.L, self.H, self.device = L, H, device
        self.words = max(triton.next_power_of_2(nb) // 32, 4 * triton.cdiv(nb, 128))
        self.plans = []
        self.offsets = {}
        self.ready = set()          # branches that have run the full step
        self.current = None         # (plan, step) used by the branches of this step
        self.pending = None         # (plan, step) made by branch 0 for the next step

    def plan(self, i):
        while len(self.plans) <= i:
            shape = (self.H, triton.cdiv(self.L, BM), self.words)
            self.plans.append(tuple(torch.full(shape, -1, device=self.device, dtype=torch.int32) for _ in range(2)))
        return self.plans[i]

    def offset(self, branch):
        if branch not in self.offsets:
            self.offsets[branch] = torch.full((triton.cdiv(self.L, BM) * BM, self.H), 1e30, device=self.device)
        return self.offsets[branch]


class _PlanScratch:
    """Per-call tensors of the RTX 5090 / H100 path, shared by all layers."""

    def __init__(self, L, H, device, refine_width):
        nq, nb = triton.cdiv(L, BM), triton.cdiv(L, BK)
        self.shape = (L, H, refine_width)
        self.mass = torch.empty(H, nq, nb, device=device)
        self.keep = torch.empty(H, nq, nb, device=device, dtype=torch.uint8)
        self.next_keep = torch.empty(H, nq, nb, device=device, dtype=torch.uint8)
        self.scale_exact = torch.empty(H, nq, device=device)
        self.scale_skip = torch.empty(H, nq, device=device)
        self.counts = torch.empty(nq, H, device=device, dtype=torch.int32)
        self.blocks = torch.zeros(nq, H, nb, device=device, dtype=torch.int32)
        self.refine = torch.empty(nq, H, refine_width, device=device, dtype=torch.int32)
        self.refine_counts = torch.empty(nq, H, device=device, dtype=torch.int32)
        self.means = torch.empty(nq, H, nb, device=device, dtype=torch.int32)
        self.mean_counts = torch.empty(nq, H, device=device, dtype=torch.int32)
        self.next_offset = torch.empty(nq * BM, H, device=device)


class _Buffers:
    """Per-call scratch for one sequence shape."""

    def __init__(self, L, H, D, device, refine_iters, K):
        nq, nb = triton.cdiv(L, BM), triton.cdiv(L, BK)
        self.shape = (L, H, D)
        self.k_mean = torch.empty(nb, H, D, device=device, dtype=torch.bfloat16)
        self.v_mean = torch.empty(nb, H, D, device=device, dtype=torch.bfloat16)
        self.flags = torch.empty(nq, H, device=device, dtype=torch.int32)
        self.all_counts = torch.full((nq, H), nb, device=device, dtype=torch.int32)
        self.all_blocks = torch.arange(nb, device=device, dtype=torch.int32).expand(nq, H, nb).contiguous()
        self.reps = K.Representatives(L, H, D, device, iters=refine_iters)


class EchoSparseAttention:
    """Self-attention for one video at a time; call `reset()` before each video.

    `__call__(q, k, v, step, branch, layer)` takes [1, L, H, 128] bf16 tensors. `branch` numbers the transformer
    forwards of a denoising step (0: conditional, 1: unconditional); later branches reuse branch 0's block
    selection of the same step and layer. `sink`: first kv block that every query tile computes exactly (text or
    other conditioning tokens placed after the video tokens). `rows`: the caller uses only the first `rows` output
    rows and overwrites the others; the RTX 5090 / H100 kernels then skip the query tiles past them.

    `gpu`: "auto" (from the device of q), "h100", "b200" or "rtx5090"; see `echosparse.backends`.
    """

    def __init__(self, density=0.10, dense_steps=10, dense_layers=(0,), refine_iters=6, threshold=60.0, gpu="auto"):
        assert dense_steps >= 2, "the first sparse step needs the LSE of a dense step and a full-block step"
        self.density = density
        self.dense_steps = dense_steps
        self.dense_layers = set(dense_layers)
        self.refine_iters = refine_iters
        self.threshold = threshold
        backends.check_name(gpu)
        self.gpu = gpu
        self.backend = None             # resolved from the device of the first call
        self._device = None
        self.K = None
        self._states = {}
        self._buffers = None
        self._plan_scratch = None

    def reset(self):
        for st in self._states.values():
            if isinstance(st, _CompactState):
                st.ready.clear()
                st.current = st.pending = None
            else:
                st.ready = False

    def __call__(self, q, k, v, step, branch, layer, sink=None, rows=None):
        B, L, H, D = q.shape
        assert B == 1, "batch size 1 only"
        if q.device != self._device:
            self.backend = backends.resolve(self.gpu, q.device)
            self.K = backends.kernels(self.backend)
            self._states.clear()
            self._buffers = self._plan_scratch = None
            self._device = q.device
        if layer in self.dense_layers or triton.cdiv(L, BK) < MIN_BLOCKS:
            return dense_attention(q, k, v)
        if self.backend == "b200":
            return self._call_b200(q, k, v, step, branch, layer, sink)
        return self._call_compact(q, k, v, step, branch, layer, sink, rows)

    def _call_b200(self, q, k, v, step, branch, layer, sink):
        L, H, D = q[0].shape
        st = self._state(branch, layer, L, H, q.device)
        if step < self.dense_steps - 1 or (step >= self.dense_steps and not st.ready):
            out, lse = dense_attention(q, k, v, return_lse=True)
            st.offset[:L].copy_(lse[0].t() * LOG2E)
            return out
        q, k, v = q[0], k[0], v[0]
        buf = self._scratch(L, H, D, q.device)
        if step < self.dense_steps:
            out = self._full_step(q, k, v, st, buf)
        elif branch == 0:
            out = self._sparse_step(q, k, v, st, buf, sink)
        else:
            out = self._shared_step(q, k, v, st, self._states[(0, layer)], buf)
        st.cur ^= 1
        return out[None]

    def _state(self, branch, layer, L, H, device):
        st = self._states.get((branch, layer))
        if st is None or st.L != L:
            self._states[(branch, layer)] = None          # free the old buffers before allocating new ones
            st = self._states[(branch, layer)] = _LayerState(L, H, device)
        return st

    def _scratch(self, L, H, D, device):
        if self._buffers is None or self._buffers.shape != (L, H, D):
            self._buffers = None                           # one sequence shape at a time
            self._buffers = _Buffers(L, H, D, device, self.refine_iters, self.K)
        return self._buffers

    def _full_step(self, q, k, v, st, buf):
        """Every block at the fixed offset: exact output and the first block-probability map."""
        K = self.K
        out = torch.empty_like(q)
        st.kc ^= 1
        st.keep.fill_(1)
        st.mass.zero_()
        K.block_attention(q, k, v, out, buf.all_counts, buf.all_blocks, st.offset, st.next_offset, buf.flags,
                          mass=st.mass, threshold=self.threshold)
        K.exact_fallback(q, k, v, out, buf.all_counts, buf.all_blocks, st.next_offset, buf.flags)
        K.full_step_scales(st.offset, st.next_offset, st.scale_exact, st.scale_skip, st.L)
        st.flags.copy_(buf.flags)
        st.ready = True
        return out

    def _sparse_step(self, q, k, v, st, buf, sink):
        K = self.K
        L, H, _ = q.shape
        nq, nb = triton.cdiv(L, BM), triton.cdiv(L, BK)
        K.block_means(k, v, buf.k_mean, buf.v_mean)
        top, refined = self._budget(nb)
        width = -(-refined // 32) * 32
        if st.refine.shape[-1] != width:
            st.refine = torch.empty(nq, H, width, device=q.device, dtype=torch.int32)
        keep_prev = st.keep
        st.kc ^= 1
        K.select_blocks(st.mass, keep_prev, st.scale_exact, st.scale_skip, st.keep, st.counts, st.blocks, st.flags,
                        st.skip_bits, st.refine, top, refined, L, sink=1 << 30 if sink is None else sink)
        st.mass.zero_()
        out = torch.empty_like(q)
        self._attend(q, k, v, st, st, buf, out, mass=st.mass)
        st.flags.copy_(buf.flags)
        K.estimate_skipped(q, buf.k_mean, st.keep, st.offset, st.next_offset, st.mass, st.scale_exact, st.scale_skip)
        return out

    def _shared_step(self, q, k, v, st, sel, buf):
        """A later CFG branch: branch 0's block selection, this branch's own offsets."""
        self.K.block_means(k, v, buf.k_mean, buf.v_mean)
        out = torch.empty_like(q)
        self._attend(q, k, v, st, sel, buf, out)
        return out

    def _attend(self, q, k, v, st, sel, buf, out, mass=None):
        K, reps = self.K, buf.reps
        reps.build(q, k, v)
        K.sparse_attention(q, k, v, buf.k_mean, buf.v_mean, sel.skip_bits, out, sel.counts, sel.blocks, st.offset,
                           st.next_offset, buf.flags, sel.refine, reps.k, reps.v, reps.log_counts, mass=mass,
                           threshold=self.threshold)
        # tiles whose offset was too far off: online softmax over the same list, then block-mean compensation
        K.exact_fallback(q, k, v, out, sel.counts, sel.blocks, st.next_offset, buf.flags)
        K.compensate_fallback(q, buf.k_mean, buf.v_mean, sel.keep, st.next_offset, out, buf.flags)

    def _budget(self, nb):
        """Exactly computed blocks per tile, and refined blocks (four representatives each) per tile."""
        top = max(1, int(round(self.density * nb)))
        return top, max(0, int(round((nb - top) / 3)))

    def _call_compact(self, q, k, v, step, branch, layer, sink, rows):
        L, H, D = q[0].shape
        st = self._states.get(layer)
        if st is None or st.L != L:
            self._states[layer] = None
            st = self._states[layer] = _CompactState(L, H, q.device)
        offset = st.offset(branch)
        if branch == 0 and step >= self.dense_steps:
            st.current = st.pending if st.pending is not None and st.pending[1] == step else None
            st.pending = None
        full = step == self.dense_steps - 1
        sparse = step >= self.dense_steps and branch in st.ready and st.current is not None and st.current[1] == step
        if not (full or sparse):
            out, lse = dense_attention(q, k, v, return_lse=True)
            offset[:L].copy_(lse[0].t() * LOG2E)
            return out
        K = self.K
        q, k, v = q[0], k[0], v[0]
        top, refined = self._budget(triton.cdiv(L, BK))
        buf = self._scratch(L, H, D, q.device)
        ps = self._plan_scratch
        if ps is None or ps.shape != (L, H, -(-refined // 32) * 32):
            self._plan_scratch = None
            ps = self._plan_scratch = _PlanScratch(L, H, q.device, -(-refined // 32) * 32)
        out = torch.empty_like(q)
        if full:
            ps.keep.fill_(1)
            ps.mass.zero_()
            K.block_attention(q, k, v, out, buf.all_counts, buf.all_blocks, offset, ps.next_offset, buf.flags,
                              mass=ps.mass, threshold=self.threshold)
            K.exact_fallback(q, k, v, out, buf.all_counts, buf.all_blocks, ps.next_offset, buf.flags)
            K.full_step_scales(offset, ps.next_offset, ps.scale_exact, ps.scale_skip, L)
            st.ready.add(branch)
        else:
            exact, skip = st.plan(st.current[0])
            K.rebuild_plan(exact, skip, ps.keep, ps.counts, ps.blocks, ps.refine, ps.refine_counts, ps.means,
                           ps.mean_counts)
            buf.reps.build(q, k, v, buf.k_mean, buf.v_mean)
            if branch == 0:
                ps.mass.zero_()
            K.sparse_attention(q, k, v, buf.k_mean, buf.v_mean, ps.means, ps.mean_counts, out, ps.counts, ps.blocks,
                               offset, ps.next_offset, buf.flags, ps.refine, ps.refine_counts, buf.reps.k, buf.reps.v,
                               buf.reps.log_counts, mass=ps.mass if branch == 0 else None, threshold=self.threshold,
                               rows=rows)
            K.exact_fallback(q, k, v, out, ps.counts, ps.blocks, ps.next_offset, buf.flags)
            K.compensate_fallback(q, buf.k_mean, buf.v_mean, ps.keep, ps.next_offset, out, buf.flags)
            if branch == 0:
                K.estimate_skipped(q, buf.k_mean, ps.keep, offset, ps.next_offset, ps.mass, ps.scale_exact,
                                   ps.scale_skip)
        if branch == 0:
            # the next step's plan, after this step's last use of the current one in a single branch
            plan = 0 if st.current is None else st.current[0] ^ (len(st.offsets) > 1)
            exact, skip = st.plan(plan)
            K.select_blocks(ps.mass, ps.keep, ps.scale_exact, ps.scale_skip, ps.next_keep, ps.counts, ps.blocks,
                            buf.flags, skip, ps.refine, top, refined, L, sink=1 << 30 if sink is None else sink)
            K.pack_plan(ps.next_keep, exact)
            st.pending = (plan, step + 1)
        offset.copy_(ps.next_offset)
        return out[None]
