"""Block-sparse attention with a fixed softmax offset and the compensation of skipped blocks in one kernel.

For each (query tile, head) the kernel runs three kinds of iterations over the same QK / softmax / PV pipeline,
all at the offset of the previous step's LSE:

  1. exact: the listed 64-token blocks, two per iteration;
  2. block means: the mean key and value of every block, 128 per iteration, each standing for its block's tokens
     (+log2 of the block length in the exponent); blocks whose bit is set in `skip_bits` (exact, refined, or past
     nb) are masked;
  3. refine: the four clustered representatives of each block in the tile's refine list, 32 list entries per
     iteration, each with +log2 of its cluster size. The rows are gathered with cp.async by the load warp.

Output = (exact + compensation) / (their row sums), and lse = offset + log2(row sum) is the next step's offset.
"""
import math
import operator

import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass import BFloat16, Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.cute.nvgpu import cpasync, tcgen05
import flash_attn.cute.pipeline as fa_pipeline
import flash_attn.cute.utils as fa_utils

from ._cute import (D, HB, LOG2E, M, N, O_OFF, P_OFF, PV_TILE, QK_TILE, S_OFF, S_TILE, THREADS, TMEM_COLS,
                    emulate_exp2, gemm, kv_tma_atoms, make_mmas, make_smem_layouts, q_tma_atom, tmem_at,
                    tmem_zero_based, to_cute, wait_tmem_load, wait_tmem_store)


@dsl_user_op
def _cp_async16(dst: Int32, src: Int64, *, loc=None, ip=None):
    llvm.inline_asm(None, [Int32(dst).ir_value(loc=loc, ip=ip), Int64(src).ir_value(loc=loc, ip=ip)],
                    "cp.async.cg.shared.global [$0], [$1], 16;", "r,l", has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def _cp_async_arrive(mbar: Int32, *, loc=None, ip=None):
    """One more expected arrival on mbar, completed when this thread's earlier cp.async copies land."""
    llvm.inline_asm(None, [Int32(mbar).ir_value(loc=loc, ip=ip)], "cp.async.mbarrier.arrive.shared::cta.b64 [$0];",
                    "r", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def _mbar_arrive(mbar: Int32, *, loc=None, ip=None):
    llvm.inline_asm(None, [Int32(mbar).ir_value(loc=loc, ip=ip)], "mbarrier.arrive.shared::cta.b64 _, [$0];",
                    "r", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def _fence_async_smem(*, loc=None, ip=None):
    llvm.inline_asm(None, [], "fence.proxy.async.shared::cta;", "", has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT)


def _box_table(layout, kmajor):
    """Trace time: pre-swizzle element offsets of 4-key groups in a K (kmajor) or V shared-memory tile.

    Group g in [0, 16) of a 64-key half (keys 4g..4g+3), 64-column chunk ch and half hh start at
    base + sum_i bit_i(g) w[i] + ch * sc + hh * sh. Checks that every group is 4 runs of 64 contiguous elements,
    64 apart, and that the offsets are affine in the bits of g."""
    flat = lambda t: [x for e in t for x in flat(e)] if isinstance(t, tuple) else [int(t)]
    size = lambda t: math.prod(flat(t))

    def split(i, shape):            # index -> coordinate of a (nested) mode, first sub-mode fastest (as CuTe)
        if not isinstance(shape, tuple):
            assert 0 <= i < int(shape), (i, shape)
            return i
        out = []
        for e in shape:
            out.append(split(i % size(e), e))
            i //= size(e)
        assert i == 0
        return tuple(out)
    shape, std = layout.shape, flat(layout.stride)

    def off(key, col, hh):
        # modes: (MMA N, MMA K = 16), rest N, rest K (k-blocks of 16), stage (= 64-key half)
        n, k = (key, col) if kmajor else (col, key)
        crd = ((split(n, shape[0][0]), split(k % 16, shape[0][1])), split(0, shape[1]), split(k // 16, shape[2]),
               split(hh, shape[3]))
        return sum(c * d for c, d in zip(flat(crd), std))
    base = off(0, 0, 0)
    w = [off(4 << i, 0, 0) - base for i in range(4)]
    sc = off(0, 64, 0) - base
    sh = off(0, 0, 1) - base
    for hh in range(2):
        for ch in range(2):
            for g in range(16):
                line = base + sum(w[i] for i in range(4) if g >> i & 1) + ch * sc + hh * sh
                for r in range(4):
                    for kk in (0, 1, 7, 8, 31, 63):
                        assert off(4 * g + r, 64 * ch + kk, hh) == line + 64 * r + kk, (kmajor, hh, ch, g, r, kk)
    return base, w, sc, sh


def _line(tb, g):
    """Pre-swizzle element offset of rows 4g.. of a half tile, relative to row 0, chunk 0."""
    return sum(tb[1][q] for q in range(4) if g >> q & 1)


@cute.jit
def _gather(sbase: Int32, gbase: Int64, tb: cutlass.Constexpr, half: cutlass.Constexpr, my: Int32, lane: Int32,
            row0: Int32, bar: Int32):
    """cp.async gather of one 64-key half (16 list entries x 4 rows x 256 bytes) by the 32 lanes of a warp.

    Copy `it` is entry it // 2 (its block from lane 16 * half + it // 2 of `my`), rows 2 (it % 2) + lane // 16,
    64-column chunk (lane // 8) % 2, 16-byte column lane % 8, written at its swizzled place (128B swizzle: the
    16-byte column XOR the row within the 8-row atom). Each lane then adds one expected arrival on `bar`, completed
    by its copies, and lane 0 adds the producer's own arrival."""
    ch = (lane // Int32(8)) % Int32(2)
    c16 = lane % Int32(8)
    for it in cutlass.range_constexpr(32):
        g = it // 2
        bj = cute.arch.shuffle_sync(my, Int32(16 * half + g))
        r = Int32(2 * (it % 2)) + lane // Int32(16)
        dst = sbase + Int32(2 * _line(tb, g)) + ch * Int32(2 * tb[2]) + r * Int32(128) \
            + ((c16 ^ (Int32(4 * (g % 2)) + r)) * Int32(16))
        src = gbase + Int64(256) * Int64(row0 + Int32(4) * bj + r) + Int64(ch * Int32(128) + c16 * Int32(16))
        _cp_async16(dst, src)
    _cp_async_arrive(bar)
    cute.arch.sync_warp()
    if lane == Int32(0):
        _mbar_arrive(bar)


@cute.struct
class SharedStorage:
    q_mbar: cute.struct.MemRange[cutlass.Int64, 2]
    k_mbar: cute.struct.MemRange[cutlass.Int64, 2]
    v_mbar: cute.struct.MemRange[cutlass.Int64, 2]
    k2_mbar: cute.struct.MemRange[cutlass.Int64, 2]
    v2_mbar: cute.struct.MemRange[cutlass.Int64, 2]
    s_mbar: cute.struct.MemRange[cutlass.Int64, 2]
    s2_mbar: cute.struct.MemRange[cutlass.Int64, 2]
    o_mbar: cute.struct.MemRange[cutlass.Int64, 2]
    flags: cute.struct.Align[cute.struct.MemRange[Int32, 4], 16]
    tmem_holding_buf: Int32


@cute.kernel
def _kernel(tiled_qk: cute.TiledMma, tiled_qk_s: cute.TiledMma, tiled_pv: cute.TiledMma,
            tma_q: cute.CopyAtom, mQ: cute.Tensor, tma_k: cute.CopyAtom, mK: cute.Tensor,
            tma_v: cute.CopyAtom, mV: cute.Tensor,
            mO: cute.Tensor, mCnt: cute.Tensor, mIdx: cute.Tensor, mM: cute.Tensor, mLse: cute.Tensor,
            mFlag: cute.Tensor, mMass: cute.Tensor, scale_log2: Float32, th: Float32, L: Int32, nb_last: Int32,
            tail_valid: Int32, MASS: cutlass.Constexpr,
            tma_kb: cute.CopyAtom, mKb: cute.Tensor, tma_vb: cute.CopyAtom, mVb: cute.Tensor, mBits: cute.Tensor,
            nvirt: Int32, corr_last: Float32, mRidx: cute.Tensor, mLc4: cute.Tensor, nref: Int32, nb4p: Int32,
            k4p: Int64, v4p: Int64,
            q_layout: cute.ComposedLayout, k_layout: cute.ComposedLayout, v_layout: cute.ComposedLayout,
            p_layout: cute.ComposedLayout):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    bid, _, _ = cute.arch.block_idx()
    nq_ = cute.size(mCnt.shape[0])
    # adjacent CTAs are adjacent query tiles of one head, so their kv lists overlap in L2
    qb = cute.arch.make_warp_uniform(Int32(bid) % Int32(nq_))
    h = cute.arch.make_warp_uniform(Int32(bid) // Int32(nq_))

    smem = utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    sFlags = storage.flags.get_tensor(cute.make_layout((4,)))
    sQ = smem.allocate_tensor(element_type=BFloat16, layout=q_layout.outer, byte_alignment=1024,
                              swizzle=q_layout.inner)
    sK = smem.allocate_tensor(element_type=BFloat16, layout=k_layout.outer, byte_alignment=1024,
                              swizzle=k_layout.inner)
    sV = smem.allocate_tensor(element_type=BFloat16, layout=v_layout.outer, byte_alignment=1024,
                              swizzle=v_layout.inner)
    # log2(cluster size) / scale of the representatives, double-buffered over refine iterations
    sBias = smem.allocate_tensor(element_type=Float32, layout=cute.make_layout((128, 2)), byte_alignment=16)
    kt = _box_table(k_layout.outer, True)
    vt = _box_table(v_layout.outer, False)

    tmem_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=THREADS)
    p_a = pipeline.NamedBarrier(barrier_id=2, num_threads=5 * 32)
    p_b = pipeline.NamedBarrier(barrier_id=4, num_threads=5 * 32)
    epi_barrier = pipeline.NamedBarrier(barrier_id=3, num_threads=4 * 32)
    bias_barrier = pipeline.NamedBarrier(barrier_id=5, num_threads=4 * 32)
    tmem = utils.TmemAllocator(storage.tmem_holding_buf.ptr, barrier_for_retrieve=tmem_barrier)
    tmem.allocate(TMEM_COLS)

    # Each 64-key half of K and V has its own barrier, so the next half is loaded as soon as its MMA has read the
    # current one; S is two N = 64 halves with their own MMA -> softmax barriers (s_pipe, s_pipe_b) and softmax ->
    # MMA barriers (p_a, p_b), so PV of the first half overlaps the softmax of the second.
    one_thread = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
    owners = pipeline.CooperativeGroup(pipeline.Agent.Thread, 4 * 32)
    q_bytes = cute.size_in_bytes(BFloat16, cute.select(q_layout, mode=[0, 1, 2]))
    k_bytes = cute.size_in_bytes(BFloat16, cute.select(k_layout, mode=[0, 1, 2]))          # one half
    v_bytes = cute.size_in_bytes(BFloat16, cute.select(v_layout, mode=[0, 1, 2]))
    q_pipe = fa_pipeline.PipelineTmaUmma.create(num_stages=1, producer_group=one_thread, consumer_group=one_thread,
                                                tx_count=q_bytes, barrier_storage=storage.q_mbar.data_ptr())
    k_pipe = fa_pipeline.PipelineTmaUmma.create(num_stages=1, producer_group=one_thread, consumer_group=one_thread,
                                                tx_count=k_bytes, barrier_storage=storage.k_mbar.data_ptr())
    v_pipe = fa_pipeline.PipelineTmaUmma.create(num_stages=1, producer_group=one_thread, consumer_group=one_thread,
                                                tx_count=v_bytes, barrier_storage=storage.v_mbar.data_ptr())
    k_pipe2 = fa_pipeline.PipelineTmaUmma.create(num_stages=1, producer_group=one_thread, consumer_group=one_thread,
                                                 tx_count=k_bytes, barrier_storage=storage.k2_mbar.data_ptr())
    v_pipe2 = fa_pipeline.PipelineTmaUmma.create(num_stages=1, producer_group=one_thread, consumer_group=one_thread,
                                                 tx_count=v_bytes, barrier_storage=storage.v2_mbar.data_ptr())
    s_pipe = fa_pipeline.PipelineUmmaAsync.create(num_stages=1, producer_group=one_thread, consumer_group=owners,
                                                  barrier_storage=storage.s_mbar.data_ptr())
    s_pipe_b = fa_pipeline.PipelineUmmaAsync.create(num_stages=1, producer_group=one_thread, consumer_group=owners,
                                                    barrier_storage=storage.s2_mbar.data_ptr())
    o_pipe = fa_pipeline.PipelineUmmaAsync.create(num_stages=1, producer_group=one_thread, consumer_group=owners,
                                                  barrier_storage=storage.o_mbar.data_ptr())

    gQ = cute.local_tile(mQ[None, None, h], (M, D), (None, 0))
    gK = cute.local_tile(mK[None, None, h], (HB, D), (None, 0))
    gV = cute.local_tile(mV[None, None, h], (D, HB), (0, None))
    thr_qk = tiled_qk.get_slice(0)
    thr_pv = tiled_pv.get_slice(0)
    tCgQ = thr_qk.partition_A(gQ)
    tCgK = thr_qk.partition_B(gK)
    tCgV = thr_pv.partition_B(gV)
    tCrQ = tiled_qk.make_fragment_A(sQ)
    tCrK = tiled_qk.make_fragment_B(sK)
    tCrV = tiled_pv.make_fragment_B(sV)
    tQsQ, tQgQ = cpasync.tma_partition(tma_q, 0, cute.make_layout(1), cute.group_modes(sQ, 0, 3),
                                       cute.group_modes(tCgQ, 0, 3))
    tKsK, tKgK = cpasync.tma_partition(tma_k, 0, cute.make_layout(1), cute.group_modes(sK, 0, 3),
                                       cute.group_modes(tCgK, 0, 3))
    tVsV, tVgV = cpasync.tma_partition(tma_v, 0, cute.make_layout(1), cute.group_modes(sV, 0, 3),
                                       cute.group_modes(tCgV, 0, 3))
    gKb = cute.local_tile(mKb[None, None, h], (HB, D), (None, 0))
    gVb = cute.local_tile(mVb[None, None, h], (D, HB), (0, None))
    tKbsK, tKbgK = cpasync.tma_partition(tma_kb, 0, cute.make_layout(1), cute.group_modes(sK, 0, 3),
                                         cute.group_modes(thr_qk.partition_B(gKb), 0, 3))
    tVbsV, tVbgV = cpasync.tma_partition(tma_vb, 0, cute.make_layout(1), cute.group_modes(sV, 0, 3),
                                         cute.group_modes(thr_pv.partition_B(gVb), 0, 3))

    s_template = tiled_qk.make_fragment_C(tiled_qk.partition_shape_C(QK_TILE[:2]))          # 128 x 64 (one half)
    o_template = tiled_pv.make_fragment_C(tiled_pv.partition_shape_C(PV_TILE[:2]))

    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(Float32)
    tmem.relinquish_alloc_permit()
    tmem_base = tmem_ptr.toint()
    tSa = cute.make_tensor(cute.make_ptr(Float32, tmem_base + Int32(S_OFF), cute.AddressSpace.tmem, assumed_align=16),
                           s_template.layout)
    tSb = cute.make_tensor(cute.make_ptr(Float32, tmem_base + Int32(S_OFF + HB), cute.AddressSpace.tmem,
                                         assumed_align=16), s_template.layout)
    tO = cute.make_tensor(cute.make_ptr(Float32, tmem_base + Int32(O_OFF), cute.AddressSpace.tmem, assumed_align=16),
                          o_template.layout)
    tP_base = tiled_pv.make_fragment_A(cute.make_tensor(tSa.iterator, p_layout.outer))[None, None, None, 0]
    tP = cute.make_tensor(tP_base.iterator + tmem_base + tmem_base + Int32(P_OFF * 2), tP_base.layout)
    # P of the second half lives in its own columns (64..96), so the first half's PV can start early
    tPb = cute.make_tensor(tP_base.iterator + tmem_base + tmem_base + Int32((P_OFF + HB) * 2), tP_base.layout)

    n64 = Int32(mCnt[qb, h])
    n = (n64 + Int32(1)) // Int32(2)                  # exact iterations (pairs of 64-token blocks)
    nt = n + nvirt + nref                             # + block-mean iterations + refine iterations

    # ---------------------------------------------------------------- loads
    if warp_idx == Int32(5):
        cpasync.prefetch_descriptor(tma_q)
        cpasync.prefetch_descriptor(tma_k)
        cpasync.prefetch_descriptor(tma_v)
        cpasync.prefetch_descriptor(tma_kb)
        cpasync.prefetch_descriptor(tma_vb)
        q_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
        k_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
        v_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
        k_prod2 = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
        v_prod2 = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
        q_pipe.producer_acquire(q_prod)
        cute.copy(tma_q, tQgQ[(None, qb)], tQsQ[(None, q_prod.index)], tma_bar_ptr=q_pipe.producer_get_barrier(q_prod))
        q_prod.advance()
        for i in cutlass.range(n, unroll=1):
            ba = Int32(mIdx[qb, h, Int32(2) * i])
            bb = ba                                    # odd list: reload block a, masked in the softmax
            if Int32(2) * i + Int32(1) < n64:
                bb = Int32(mIdx[qb, h, Int32(2) * i + Int32(1)])
            k_pipe.producer_acquire(k_prod)
            cute.copy(tma_k, tKgK[(None, ba)], tKsK[(None, 0)], tma_bar_ptr=k_pipe.producer_get_barrier(k_prod))
            k_prod.advance()
            k_pipe2.producer_acquire(k_prod2)
            cute.copy(tma_k, tKgK[(None, bb)], tKsK[(None, 1)], tma_bar_ptr=k_pipe2.producer_get_barrier(k_prod2))
            k_prod2.advance()
            v_pipe.producer_acquire(v_prod)
            cute.copy(tma_v, tVgV[(None, ba)], tVsV[(None, 0)], tma_bar_ptr=v_pipe.producer_get_barrier(v_prod))
            v_prod.advance()
            v_pipe2.producer_acquire(v_prod2)
            cute.copy(tma_v, tVgV[(None, bb)], tVsV[(None, 1)], tma_bar_ptr=v_pipe2.producer_get_barrier(v_prod2))
            v_prod2.advance()
        for c in cutlass.range(nvirt, unroll=1):
            k_pipe.producer_acquire(k_prod)
            cute.copy(tma_kb, tKbgK[(None, Int32(2) * c)], tKbsK[(None, 0)], tma_bar_ptr=k_pipe.producer_get_barrier(k_prod))
            k_prod.advance()
            k_pipe2.producer_acquire(k_prod2)
            cute.copy(tma_kb, tKbgK[(None, Int32(2) * c + Int32(1))], tKbsK[(None, 1)],
                      tma_bar_ptr=k_pipe2.producer_get_barrier(k_prod2))
            k_prod2.advance()
            v_pipe.producer_acquire(v_prod)
            cute.copy(tma_vb, tVbgV[(None, Int32(2) * c)], tVbsV[(None, 0)], tma_bar_ptr=v_pipe.producer_get_barrier(v_prod))
            v_prod.advance()
            v_pipe2.producer_acquire(v_prod2)
            cute.copy(tma_vb, tVbgV[(None, Int32(2) * c + Int32(1))], tVbsV[(None, 1)],
                      tma_bar_ptr=v_pipe2.producer_get_barrier(v_prod2))
            v_prod2.advance()
        # refine iterations: list entry 32 c + e fills keys 4e..4e+3 of the 128-key tile
        lane = tidx % Int32(32)
        for c in cutlass.range(nref, unroll=1):
            my = Int32(mRidx[qb, h, Int32(32) * c + lane])
            row0 = h * nb4p
            ka = Int32(sK.iterator.toint()) + Int32(2 * kt[0])
            va_ = Int32(sV.iterator.toint()) + Int32(2 * vt[0])
            k_pipe.sync_object_empty.wait(k_prod.index, k_prod.phase)
            _gather(ka, k4p, kt, 0, my, lane, row0, Int32(k_pipe.producer_get_barrier(k_prod).toint()))
            k_prod.advance()
            k_pipe2.sync_object_empty.wait(k_prod2.index, k_prod2.phase)
            _gather(ka + Int32(2 * kt[3]), k4p, kt, 1, my, lane, row0,
                    Int32(k_pipe2.producer_get_barrier(k_prod2).toint()))
            k_prod2.advance()
            v_pipe.sync_object_empty.wait(v_prod.index, v_prod.phase)
            _gather(va_, v4p, vt, 0, my, lane, row0, Int32(v_pipe.producer_get_barrier(v_prod).toint()))
            v_prod.advance()
            v_pipe2.sync_object_empty.wait(v_prod2.index, v_prod2.phase)
            _gather(va_ + Int32(2 * vt[3]), v4p, vt, 1, my, lane, row0,
                    Int32(v_pipe2.producer_get_barrier(v_prod2).toint()))
            v_prod2.advance()

    # ---------------------------------------------------------------- MMAs
    if warp_idx == Int32(4):
        q_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
        k_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
        v_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
        k_cons2 = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
        v_cons2 = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
        s_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
        s_prod_b = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
        o_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
        q_pipe.consumer_wait(q_cons)
        k_pipe.consumer_wait(k_cons)
        s_pipe.producer_acquire(s_prod)
        gemm(tiled_qk, tSa, tCrQ[None, None, None, 0], tCrK[None, None, None, 0], True)
        s_pipe.producer_commit(s_prod)
        s_prod.advance()
        k_pipe.consumer_release(k_cons)
        k_cons.advance()
        k_pipe2.consumer_wait(k_cons2)
        s_pipe_b.producer_acquire(s_prod_b)
        gemm(tiled_qk, tSb, tCrQ[None, None, None, 0], tCrK[None, None, None, 1], True)
        s_pipe_b.producer_commit(s_prod_b)
        s_prod_b.advance()
        k_pipe2.consumer_release(k_cons2)
        k_cons2.advance()
        for i in cutlass.range(nt, unroll=1):
            # the cp.async gathers are generic-proxy writes: order them before the MMAs that read them
            gat = i >= n + nvirt
            nxg = i + Int32(1) >= n + nvirt
            v_pipe.consumer_wait(v_cons)
            if gat:
                _fence_async_smem()
            p_a.arrive_and_wait()                # P_a(i) stored: PV_a(i), then QK_a(i + 1)
            gemm(tiled_pv, tO, tP, tCrV[None, None, None, 0], i == Int32(0), 0)
            v_pipe.consumer_release(v_cons)
            v_cons.advance()
            if i + Int32(1) < nt:
                k_pipe.consumer_wait(k_cons)
                if nxg:
                    _fence_async_smem()
                s_pipe.producer_acquire(s_prod)
                gemm(tiled_qk, tSa, tCrQ[None, None, None, 0], tCrK[None, None, None, 0], True)
                s_pipe.producer_commit(s_prod)
                s_prod.advance()
                k_pipe.consumer_release(k_cons)
                k_cons.advance()
            v_pipe2.consumer_wait(v_cons2)
            if gat:
                _fence_async_smem()
            p_b.arrive_and_wait()                # P_b(i) stored: PV_b(i), then QK_b(i + 1)
            gemm(tiled_pv, tO, tPb, tCrV[None, None, None, 1], False, 0)
            if i + Int32(1) == nt:
                o_pipe.producer_commit(o_prod)
            v_pipe2.consumer_release(v_cons2)
            v_cons2.advance()
            if i + Int32(1) < nt:
                k_pipe2.consumer_wait(k_cons2)
                if nxg:
                    _fence_async_smem()
                s_pipe_b.producer_acquire(s_prod_b)
                gemm(tiled_qk, tSb, tCrQ[None, None, None, 0], tCrK[None, None, None, 1], True)
                s_pipe_b.producer_commit(s_prod_b)
                s_prod_b.advance()
                k_pipe2.consumer_release(k_cons2)
                k_cons2.advance()
        q_pipe.consumer_release(q_cons)

    # ---------------------------------------------------------------- softmax and epilogue (thread t = row t)
    if warp_idx < Int32(4):
        s_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
        s_cons_b = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
        o_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
        # the two N = 64 accumulators are adjacent TMEM columns: read them as one 128 x 128 tile
        thr_qk_s = tiled_qk_s.get_slice(0)
        s_full = tiled_qk_s.make_fragment_C(tiled_qk_s.partition_shape_C(S_TILE))
        tSacc = tmem_zero_based(Float32, s_full.layout)[(None, None), 0, 0]
        tScS = thr_qk_s.partition_C(cute.make_identity_tensor(S_TILE))[(None, None), 0, 0]
        ld_atom = cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), Float32)
        thr_ld = tcgen05.make_tmem_copy(ld_atom, tSacc).get_slice(tidx)
        tStS = tmem_at(thr_ld.partition_S(tSacc), tmem_base + Int32(S_OFF))
        tStP_rel = cute.make_tensor(tSacc.iterator, cute.composition(tSacc.layout, cute.make_layout((M, N // 2))))
        st_atom = cute.make_copy_atom(tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(16)), Float32)
        thr_st = tcgen05.make_tmem_copy(st_atom, tStP_rel).get_slice(tidx)
        tStP = tmem_at(thr_st.partition_D(tStP_rel), tmem_base + Int32(P_OFF))
        tStPb = tmem_at(thr_st.partition_D(tStP_rel), tmem_base + Int32(P_OFF + HB // 2))   # chunks 2, 3 -> 64, 80
        tSrS = cute.make_rmem_tensor(thr_ld.partition_D(tScS).shape, Float32)
        tScR = thr_ld.partition_D(tScS)                                                  # (row, col) coordinates
        tSrP32 = cute.make_rmem_tensor(thr_st.partition_S(cute.make_identity_tensor((M, N // 2))).shape, Float32)
        tSrP = cute.make_tensor(cute.recast_ptr(tSrP32.iterator, dtype=BFloat16), tSrS.layout)
        frS = cute.logical_divide(tSrS, cute.make_layout(32))
        frP = cute.logical_divide(tSrP, cute.make_layout(32))
        frC = cute.logical_divide(tScR, cute.make_layout(32))

        row = qb * Int32(M) + tidx
        m = Float32(mM[row, h])
        l0 = Float32(0.0)
        l1 = Float32(0.0)
        noff = -m
        inv_scale = Float32(1.0) / scale_log2
        ejn = Int32(0)
        lcn = Float32(0.0)
        for i in cutlass.range(nt, unroll=1):
            virt = i >= n
            cv = i - n
            ie = i
            if virt:
                ie = Int32(0)
                noff = -m + Float32(6.0)               # log2(64): a block mean stands for 64 tokens
            ba = Int32(mIdx[qb, h, Int32(2) * ie])
            has_b = Int32(2) * ie + Int32(1) < n64
            bb = ba
            if has_b:
                bb = Int32(mIdx[qb, h, Int32(2) * ie + Int32(1)])
            # valid keys of each half: 64, the last block's length, or 0 for an empty second half
            va = Int32(HB)
            if ba == nb_last:
                va = tail_valid
            vb = Int32(HB)
            if bb == nb_last:
                vb = tail_valid
            if not has_b:
                vb = Int32(0)
            masked = ((va < Int32(HB)) | (vb < Int32(HB))) & (i < n)
            lim = va
            a0 = Float32(0.0)
            a1 = Float32(0.0)
            c0 = Float32(0.0)
            c1 = Float32(0.0)
            pooled = virt & (i < n + nvirt)
            refi = i >= n + nvirt
            cr = i - n - nvirt
            # the next refine iteration's log2(cluster size) / scale, one column per thread (column 4e + r:
            # representative r of list entry 32 nx + e), stored in the other half of sBias at the end of this one
            nx = cr + Int32(1)
            pre = (nx >= Int32(0)) & (nx < nref)
            if pre:
                ejn = Int32(mRidx[qb, h, Int32(32) * nx + tidx // Int32(4)])
            if refi:
                noff = -m
                bias_barrier.arrive_and_wait()         # this iteration's biases, stored by every thread, visible
            for hh in cutlass.range_constexpr(2):
                if cutlass.const_expr(hh == 0):
                    s_pipe.consumer_wait(s_cons)
                else:
                    s_pipe_b.consumer_wait(s_cons_b)
                for j in cutlass.range_constexpr(2 * hh, 2 * hh + 2):
                    cute.copy(thr_ld, tStS[None, None, j], tSrS[None, None, j])
                wait_tmem_load()
                cute.arch.fence_view_async_tmem_load()
                if cutlass.const_expr(hh == 0):
                    s_pipe.consumer_release(s_cons)
                else:
                    s_pipe_b.consumer_release(s_cons_b)
                if masked:
                    lim = va if cutlass.const_expr(hh == 0) else vb + Int32(HB)
                    for j in cutlass.range_constexpr(2 * hh, 2 * hh + 2):
                        for k in cutlass.range_constexpr(32):
                            frS[k, j] = -Float32.inf if frC[k, j][1] >= lim else frS[k, j]
                if pooled:
                    for j in cutlass.range_constexpr(2 * hh, 2 * hh + 2):
                        w = Int32(mBits[h, qb, Int32(4) * cv + Int32(j)])
                        for k in cutlass.range_constexpr(32):
                            frS[k, j] = -Float32.inf if ((w >> Int32(k)) & Int32(1)) != Int32(0) else frS[k, j]
                    if cv == nvirt - Int32(1):
                        # the last block may be shorter than 64 tokens
                        for j in cutlass.range_constexpr(2 * hh, 2 * hh + 2):
                            for k in cutlass.range_constexpr(32):
                                g = Int32(128) * cv + Int32(32 * j + k)
                                frS[k, j] = frS[k, j] + corr_last if g == nb_last else frS[k, j]
                if refi:
                    for j in cutlass.range_constexpr(2 * hh, 2 * hh + 2):
                        rbias = cute.make_rmem_tensor((32,), Float32)
                        cute.autovec_copy(cute.local_tile(sBias[None, cr % Int32(2)], (32,), (j,)), rbias)
                        for k in cutlass.range_constexpr(32):
                            frS[k, j] = frS[k, j] + rbias[k]
                for j in cutlass.range_constexpr(2 * hh, 2 * hh + 2):
                    for k in cutlass.range_constexpr(0, 32, 2):
                        frS[k, j], frS[k + 1, j] = cute.arch.fma_packed_f32x2(
                            (frS[k, j], frS[k + 1, j]), (scale_log2, scale_log2), (noff, noff))
                        if cutlass.const_expr(emulate_exp2(k)):
                            frS[k, j], frS[k + 1, j] = fa_utils.ex2_emulation_2(frS[k, j], frS[k + 1, j])
                        else:
                            frS[k, j] = cute.math.exp2(frS[k, j], fastmath=True)
                            frS[k + 1, j] = cute.math.exp2(frS[k + 1, j], fastmath=True)
                    for k in cutlass.range_constexpr(0, 32, 2):
                        if cutlass.const_expr(hh == 0):
                            a0, a1 = cute.arch.add_packed_f32x2((a0, a1), (frS[k, j], frS[k + 1, j]))
                        else:
                            c0, c1 = cute.arch.add_packed_f32x2((c0, c1), (frS[k, j], frS[k + 1, j]))
                    frP[None, j].store(frS[None, j].load().to(BFloat16))
                    if cutlass.const_expr(hh == 0):
                        cute.copy(thr_st, tSrP32[None, None, j], tStP[None, None, j])
                    else:
                        cute.copy(thr_st, tSrP32[None, None, j], tStPb[None, None, j])
                wait_tmem_store()
                cute.arch.fence_view_async_tmem_store()
                if cutlass.const_expr(hh == 0):
                    p_a.arrive()
                    if pre:
                        lcn = Float32(mLc4[h, Int32(4) * ejn + tidx % Int32(4)])
                else:
                    p_b.arrive()
            s_cons.advance()
            s_cons_b.advance()
            if pre:
                sBias[tidx, nx % Int32(2)] = lcn * inv_scale
            l0 = l0 + (a0 + c0)
            l1 = l1 + (a1 + c1)
            if cutlass.const_expr(MASS):
                ma = fa_utils.warp_reduce(a0 + a1, operator.add)
                mb = fa_utils.warp_reduce(c0 + c1, operator.add)
                if tidx % Int32(32) == Int32(0):
                    if i < n:
                        fa_utils.atomic_add_fp32(ma, fa_utils.elem_pointer(mMass, (h, qb, ba)))
                        if has_b:
                            fa_utils.atomic_add_fp32(mb, fa_utils.elem_pointer(mMass, (h, qb, bb)))

        l_sum = l0 + l1
        lg = cute.math.log2(l_sum, fastmath=False)
        bad = Int32(0)
        if row < L:
            if not (lg >= -th and lg <= th):
                bad = Int32(1)
        o_pipe.consumer_wait(o_cons)
        tOacc = tmem_zero_based(Float32, o_template.layout)[(None, None), 0, 0]
        tOcO = thr_pv.partition_C(cute.make_identity_tensor(PV_TILE[:2]))[(None, None), 0, 0]
        thr_ldo = tcgen05.make_tmem_copy(ld_atom, tOacc).get_slice(tidx)
        tOtO = tmem_at(thr_ldo.partition_S(tOacc), tmem_base + Int32(O_OFF))
        tOrO = cute.make_rmem_tensor(thr_ldo.partition_D(tOcO).shape, Float32)
        tOcR = thr_ldo.partition_D(tOcO)
        cute.copy(thr_ldo, tOtO, tOrO)
        wait_tmem_load()
        cute.arch.fence_view_async_tmem_load()
        rO = cute.make_rmem_tensor((D,), BFloat16)
        inv = Float32(1.0) / l_sum
        for c in cutlass.range_constexpr(cute.size(tOrO)):
            rO[tOcR[c][1]] = (Float32(tOrO[c]) * inv).to(BFloat16)
        if row < L:
            cute.autovec_copy(rO, mO[row, h, None])
        mLse[row, h] = m + lg
        # a tile whose row sums leave 2^[-th, th] is flagged and recomputed with online softmax
        ballot = cute.arch.vote_ballot_sync(bad != Int32(0))
        if tidx % Int32(32) == Int32(0):
            sFlags[tidx // Int32(32)] = Int32(ballot != Int32(0))
        epi_barrier.arrive_and_wait()
        if tidx == Int32(0):
            mFlag[qb, h] = sFlags[0] | sFlags[1] | sFlags[2] | sFlags[3]

    cute.arch.barrier()
    tmem.free(tmem_ptr)


@cute.jit
def _host(q: cute.Tensor, k: cute.Tensor, v: cute.Tensor, o: cute.Tensor, cnt: cute.Tensor, idx: cute.Tensor,
          m: cute.Tensor, lse: cute.Tensor, flag: cute.Tensor, mass: cute.Tensor, kb: cute.Tensor, vb: cute.Tensor,
          bits: cute.Tensor, scale_log2: Float32, th: Float32, L: Int32, nb_last: Int32, tail_valid: Int32,
          nvirt: Int32, corr_last: Float32, ridx: cute.Tensor, lc4: cute.Tensor, nref: Int32, nb4p: Int32,
          k4p: Int64, v4p: Int64, MASS: cutlass.Constexpr, stream: cuda.CUstream):
    tiled_qk, tiled_qk_s, tiled_pv = make_mmas()
    q_layout, k_layout, v_layout, p_layout = make_smem_layouts(tiled_qk, tiled_pv)
    tma_q, tq = q_tma_atom(q, tiled_qk, q_layout)
    (tma_k, tk), (tma_v, tv) = kv_tma_atoms(k, v, tiled_qk, tiled_pv, k_layout, v_layout)
    (tma_kb, tkb), (tma_vb, tvb) = kv_tma_atoms(kb, vb, tiled_qk, tiled_pv, k_layout, v_layout)
    nq = cute.size(cnt.shape[0])
    nh = cute.size(cnt.shape[1])
    _kernel(tiled_qk, tiled_qk_s, tiled_pv, tma_q, tq, tma_k, tk, tma_v, tv, o, cnt, idx, m, lse, flag, mass, scale_log2,
            th, L, nb_last, tail_valid, MASS, tma_kb, tkb, tma_vb, tvb, bits, nvirt, corr_last, ridx, lc4, nref, nb4p,
            k4p, v4p, q_layout, k_layout, v_layout, p_layout).launch(
        grid=(nq * nh, 1, 1), block=(THREADS, 1, 1), stream=stream, min_blocks_per_mp=2)


_COMPILED = {}


def sparse_attention(q, k, v, k_mean, v_mean, skip_bits, out, counts, blocks, offset, lse, flags, refine, k_rep,
                     v_rep, log_counts, mass=None, threshold=60.0):
    """q, k, v, out: (L, H, 128) bf16. k_mean, v_mean: (nb, H, 128) block means. skip_bits: (H, nq, nw) int32 with
    nw >= 4 * cdiv(nb, 128). counts / blocks: sorted exact block lists. offset, lse: (nq * 128, H) float32 log2.
    refine: (nq, H, 32 * r) int32 refine lists padded with nb. k_rep, v_rep: (H, (nb + 1) * 4, 128) bf16 and
    log_counts (H, (nb + 1) * 4) float32 from `Representatives`. mass: as in `block_attention`, exact blocks only."""
    L, H, hd = q.shape
    assert hd == D and q.dtype == torch.bfloat16
    nb = -(-L // HB)
    nvirt = -(-nb // 128)
    assert skip_bits.shape[-1] >= 4 * nvirt
    assert refine.shape[-1] % 32 == 0 and k_rep.shape == v_rep.shape == (H, (nb + 1) * 4, hd)
    assert log_counts.shape == k_rep.shape[:2] and k_rep.is_contiguous() and v_rep.is_contiguous()
    use_mass = mass is not None
    if mass is None:
        mass = torch.empty(1, 1, 1, device=q.device, dtype=torch.float32)
    tensors = (q, k, v, out, counts, blocks, offset, lse, flags, mass, k_mean, v_mean, skip_bits)
    args = [to_cute(x) for x in tensors]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    scale_log2 = 1.0 / math.sqrt(hd) * LOG2E
    tail_valid = L - (nb - 1) * HB
    corr_last = (math.log2(tail_valid) - 6.0) / scale_log2
    scalars = (Float32(scale_log2), Float32(threshold), Int32(L), Int32(nb - 1), Int32(tail_valid), Int32(nvirt),
               Float32(corr_last))
    reps = (Int32(refine.shape[-1] // 32), Int32(k_rep.shape[1]), Int64(k_rep.data_ptr()), Int64(v_rep.data_ptr()))
    key = (tuple((tuple(x.shape), tuple(x.stride()), x.dtype) for x in tensors + (refine, log_counts)), use_mass)
    fn = _COMPILED.get(key)
    if fn is None:
        fn = cute.compile(_host, *args, *scalars, to_cute(refine), to_cute(log_counts), *reps, use_mass, stream)
        _COMPILED[key] = fn
    fn(*args, *scalars, to_cute(refine), to_cute(log_counts), *reps, stream)
    return out
