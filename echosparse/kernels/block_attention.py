"""Exact attention over a list of 64-token kv blocks with a fixed softmax offset, plus per-block masses.

Used once per layer, at the last dense step: every block is listed, the offset is the previous step's LSE, and
the kernel writes the first block-mass map for the selection of the next step.
"""
import math
import operator

import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass import BFloat16, Float32, Int32
from cutlass.cute.nvgpu import cpasync, tcgen05
import flash_attn.cute.pipeline as fa_pipeline
import flash_attn.cute.utils as fa_utils

from ._cute import (D, HB, LOG2E, M, N, O_OFF, P_OFF, PV_TILE, QK_TILE, S_OFF, S_TILE, THREADS, TMEM_COLS,
                    emulate_exp2, gemm, kv_tma_atoms, make_mmas, make_smem_layouts, q_tma_atom, tmem_at,
                    tmem_zero_based, to_cute, wait_tmem_load, wait_tmem_store)


@cute.struct
class SharedStorage:
    q_mbar: cute.struct.MemRange[cutlass.Int64, 2]
    k_mbar: cute.struct.MemRange[cutlass.Int64, 2]
    v_mbar: cute.struct.MemRange[cutlass.Int64, 2]
    s_mbar: cute.struct.MemRange[cutlass.Int64, 2]
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
    sQ = smem.allocate_tensor(element_type=BFloat16, layout=q_layout.outer, byte_alignment=128, swizzle=q_layout.inner)
    sK = smem.allocate_tensor(element_type=BFloat16, layout=k_layout.outer, byte_alignment=128, swizzle=k_layout.inner)
    sV = smem.allocate_tensor(element_type=BFloat16, layout=v_layout.outer, byte_alignment=128, swizzle=v_layout.inner)

    tmem_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=THREADS)
    p_ready = pipeline.NamedBarrier(barrier_id=2, num_threads=5 * 32)
    epi_barrier = pipeline.NamedBarrier(barrier_id=3, num_threads=4 * 32)
    tmem = utils.TmemAllocator(storage.tmem_holding_buf.ptr, barrier_for_retrieve=tmem_barrier)
    tmem.allocate(TMEM_COLS)

    one_thread = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
    owners = pipeline.CooperativeGroup(pipeline.Agent.Thread, 4 * 32)
    q_bytes = cute.size_in_bytes(BFloat16, cute.select(q_layout, mode=[0, 1, 2]))
    k_bytes = cute.size_in_bytes(BFloat16, cute.select(k_layout, mode=[0, 1, 2])) * 2      # two halves
    v_bytes = cute.size_in_bytes(BFloat16, cute.select(v_layout, mode=[0, 1, 2])) * 2
    q_pipe = fa_pipeline.PipelineTmaUmma.create(num_stages=1, producer_group=one_thread, consumer_group=one_thread,
                                                tx_count=q_bytes, barrier_storage=storage.q_mbar.data_ptr())
    k_pipe = fa_pipeline.PipelineTmaUmma.create(num_stages=1, producer_group=one_thread, consumer_group=one_thread,
                                                tx_count=k_bytes, barrier_storage=storage.k_mbar.data_ptr())
    v_pipe = fa_pipeline.PipelineTmaUmma.create(num_stages=1, producer_group=one_thread, consumer_group=one_thread,
                                                tx_count=v_bytes, barrier_storage=storage.v_mbar.data_ptr())
    s_pipe = fa_pipeline.PipelineUmmaAsync.create(num_stages=1, producer_group=one_thread, consumer_group=owners,
                                                  barrier_storage=storage.s_mbar.data_ptr())
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

    n64 = Int32(mCnt[qb, h])
    n = (n64 + Int32(1)) // Int32(2)                  # iterations (pairs of 64-token blocks)

    # ---------------------------------------------------------------- loads
    if warp_idx == Int32(5):
        cpasync.prefetch_descriptor(tma_q)
        cpasync.prefetch_descriptor(tma_k)
        cpasync.prefetch_descriptor(tma_v)
        q_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
        k_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
        v_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
        q_pipe.producer_acquire(q_prod)
        cute.copy(tma_q, tQgQ[(None, qb)], tQsQ[(None, q_prod.index)], tma_bar_ptr=q_pipe.producer_get_barrier(q_prod))
        q_prod.advance()
        for i in cutlass.range(n, unroll=1):
            ba = Int32(mIdx[qb, h, Int32(2) * i])
            bb = ba                                    # odd list: reload block a, masked in the softmax
            if Int32(2) * i + Int32(1) < n64:
                bb = Int32(mIdx[qb, h, Int32(2) * i + Int32(1)])
            k_pipe.producer_acquire(k_prod)
            kbar = k_pipe.producer_get_barrier(k_prod)
            cute.copy(tma_k, tKgK[(None, ba)], tKsK[(None, 0)], tma_bar_ptr=kbar)
            cute.copy(tma_k, tKgK[(None, bb)], tKsK[(None, 1)], tma_bar_ptr=kbar)
            k_prod.advance()
            v_pipe.producer_acquire(v_prod)
            vbar = v_pipe.producer_get_barrier(v_prod)
            cute.copy(tma_v, tVgV[(None, ba)], tVsV[(None, 0)], tma_bar_ptr=vbar)
            cute.copy(tma_v, tVgV[(None, bb)], tVsV[(None, 1)], tma_bar_ptr=vbar)
            v_prod.advance()

    # ---------------------------------------------------------------- MMAs
    if warp_idx == Int32(4):
        q_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
        k_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
        v_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
        s_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
        o_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
        q_pipe.consumer_wait(q_cons)
        k_pipe.consumer_wait(k_cons)
        s_pipe.producer_acquire(s_prod)
        gemm(tiled_qk, tSa, tCrQ[None, None, None, 0], tCrK[None, None, None, 0], True)
        gemm(tiled_qk, tSb, tCrQ[None, None, None, 0], tCrK[None, None, None, 1], True)
        s_pipe.producer_commit(s_prod)
        s_prod.advance()
        k_pipe.consumer_release(k_cons)
        k_cons.advance()
        for i in cutlass.range(n, unroll=1):
            v_pipe.consumer_wait(v_cons)
            p_ready.arrive_and_wait()            # P(i) is in TMEM: PV(i) before QK(i + 1) overwrites S / P
            gemm(tiled_pv, tO, tP, tCrV[None, None, None, 0], i == Int32(0), 0)
            gemm(tiled_pv, tO, tP, tCrV[None, None, None, 1], False, 4)
            if i + Int32(1) == n:
                o_pipe.producer_commit(o_prod)
            v_pipe.consumer_release(v_cons)
            v_cons.advance()
            if i + Int32(1) < n:
                k_pipe.consumer_wait(k_cons)
                s_pipe.producer_acquire(s_prod)
                gemm(tiled_qk, tSa, tCrQ[None, None, None, 0], tCrK[None, None, None, 0], True)
                gemm(tiled_qk, tSb, tCrQ[None, None, None, 0], tCrK[None, None, None, 1], True)
                s_pipe.producer_commit(s_prod)
                s_prod.advance()
                k_pipe.consumer_release(k_cons)
                k_cons.advance()
        q_pipe.consumer_release(q_cons)

    # ---------------------------------------------------------------- softmax and epilogue (thread t = row t)
    if warp_idx < Int32(4):
        s_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
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
        for i in cutlass.range(n, unroll=1):
            ba = Int32(mIdx[qb, h, Int32(2) * i])
            has_b = Int32(2) * i + Int32(1) < n64
            bb = ba
            if has_b:
                bb = Int32(mIdx[qb, h, Int32(2) * i + Int32(1)])
            # valid keys of each half: 64, the last block's length, or 0 for an empty second half
            va = Int32(HB)
            if ba == nb_last:
                va = tail_valid
            vb = Int32(HB)
            if bb == nb_last:
                vb = tail_valid
            if not has_b:
                vb = Int32(0)
            masked = (va < Int32(HB)) | (vb < Int32(HB))
            lim = va
            a0 = Float32(0.0)
            a1 = Float32(0.0)
            c0 = Float32(0.0)
            c1 = Float32(0.0)
            s_pipe.consumer_wait(s_cons)
            for hh in cutlass.range_constexpr(2):
                for j in cutlass.range_constexpr(2 * hh, 2 * hh + 2):
                    cute.copy(thr_ld, tStS[None, None, j], tSrS[None, None, j])
                wait_tmem_load()
                cute.arch.fence_view_async_tmem_load()
                if cutlass.const_expr(hh == 1):
                    s_pipe.consumer_release(s_cons)
                if masked:
                    lim = va if cutlass.const_expr(hh == 0) else vb + Int32(HB)
                    for j in cutlass.range_constexpr(2 * hh, 2 * hh + 2):
                        for k in cutlass.range_constexpr(32):
                            frS[k, j] = -Float32.inf if frC[k, j][1] >= lim else frS[k, j]
                for j in cutlass.range_constexpr(2 * hh, 2 * hh + 2):
                    for k in cutlass.range_constexpr(0, 32, 2):
                        frS[k, j], frS[k + 1, j] = cute.arch.fma_packed_f32x2(
                            (frS[k, j], frS[k + 1, j]), (scale_log2, scale_log2), (-m, -m))
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
                    cute.copy(thr_st, tSrP32[None, None, j], tStP[None, None, j])
                wait_tmem_store()
                cute.arch.fence_view_async_tmem_store()
            s_cons.advance()
            p_ready.arrive()
            l0 = l0 + (a0 + c0)
            l1 = l1 + (a1 + c1)
            if cutlass.const_expr(MASS):
                ma = fa_utils.warp_reduce(a0 + a1, operator.add)
                mb = fa_utils.warp_reduce(c0 + c1, operator.add)
                if tidx % Int32(32) == Int32(0):
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
          m: cute.Tensor, lse: cute.Tensor, flag: cute.Tensor, mass: cute.Tensor, scale_log2: Float32, th: Float32,
          L: Int32, nb_last: Int32, tail_valid: Int32, MASS: cutlass.Constexpr, stream: cuda.CUstream):
    tiled_qk, tiled_qk_s, tiled_pv = make_mmas()
    q_layout, k_layout, v_layout, p_layout = make_smem_layouts(tiled_qk, tiled_pv)
    tma_q, tq = q_tma_atom(q, tiled_qk, q_layout)
    (tma_k, tk), (tma_v, tv) = kv_tma_atoms(k, v, tiled_qk, tiled_pv, k_layout, v_layout)
    nq = cute.size(cnt.shape[0])
    nh = cute.size(cnt.shape[1])
    _kernel(tiled_qk, tiled_qk_s, tiled_pv, tma_q, tq, tma_k, tk, tma_v, tv, o, cnt, idx, m, lse, flag, mass, scale_log2,
            th, L, nb_last, tail_valid, MASS, q_layout, k_layout, v_layout, p_layout).launch(
        grid=(nq * nh, 1, 1), block=(THREADS, 1, 1), stream=stream, min_blocks_per_mp=2)


_COMPILED = {}


def block_attention(q, k, v, out, counts, blocks, offset, lse, flags, mass=None, threshold=60.0):
    """q, k, v, out: (L, H, 128) bf16, contiguous. counts (nq, H) and blocks (nq, H, nk) int32: sorted block list
    of every tile. offset, lse: (nq * 128, H) float32 log2; offset rows >= L must be 1e30. flags (nq, H) int32 out.
    mass (H, nq, nb) float32: zeroed by the caller, receives the per-block sums of 2^(s - offset) over the tile."""
    L, H, hd = q.shape
    assert hd == D and q.dtype == torch.bfloat16
    nb = -(-L // HB)
    use_mass = mass is not None
    if mass is None:
        mass = torch.empty(1, 1, 1, device=q.device, dtype=torch.float32)
    tensors = (q, k, v, out, counts, blocks, offset, lse, flags, mass)
    args = [to_cute(x) for x in tensors]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    scale_log2 = 1.0 / math.sqrt(hd) * LOG2E
    tail_valid = L - (nb - 1) * HB
    scalars = (Float32(scale_log2), Float32(threshold), Int32(L), Int32(nb - 1), Int32(tail_valid))
    key = (tuple((tuple(x.shape), tuple(x.stride()), x.dtype) for x in tensors), use_mass)
    fn = _COMPILED.get(key)
    if fn is None:
        fn = cute.compile(_host, *args, *scalars, use_mass, stream)
        _COMPILED[key] = fn
    fn(*args, *scalars, stream)
    return out
