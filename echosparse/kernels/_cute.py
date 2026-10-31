"""Constants, helpers and host-side setup shared by the two CuTe DSL kernels (SM100).

Both kernels run one CTA per (128-row query tile, head) with six warps: 0-3 compute the softmax (thread t owns
row t), 4 issues the MMAs, 5 issues the loads. Every iteration covers two 64-token kv blocks: QK is two N = 64
MMAs into adjacent TMEM columns and PV two K = 64 MMAs. TMEM holds S / P in columns [0, 128) and O in [128, 256),
so two CTAs fit on one SM. The softmax uses a fixed per-row offset (the previous step's log2 LSE) instead of a
running maximum, so there is no rescaling; every second exp2 pair runs on the FMA pipe (FA4's polynomial).
"""
import math

import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import BFloat16, Float32, Int32
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack

M = 128
N = 128
D = 128
HB = 64
THREADS = 192
TMEM_COLS = 256
S_OFF = 0
P_OFF = 0
O_OFF = 128
QK_INST = (M, HB, 16)
PV_INST = (M, D, 16)
QK_TILE = (M, HB, D)
PV_TILE = (M, D, HB)
S_TILE = (M, N)
LOG2E = math.log2(math.e)


def emulate_exp2(k):
    """Whether exp2 pair k // 2 of a 32-column chunk runs on the FMA pipe."""
    return (k // 2) % 2 == 1


def to_cute(t):
    return from_dlpack(t, assumed_align=16)


@cute.jit
def wait_tmem_load():
    llvm.inline_asm(None, [], "tcgen05.wait::ld.sync.aligned;", "", has_side_effects=True,
                    is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


@cute.jit
def wait_tmem_store():
    llvm.inline_asm(None, [], "tcgen05.wait::st.sync.aligned;", "", has_side_effects=True,
                    is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)


@cute.jit
def tmem_at(t: cute.Tensor, base: Int32):
    """Zero-based TMEM tensor at its physical address (the second CTA on an SM does not start at column 0)."""
    return cute.make_tensor(cute.make_ptr(t.element_type, base + t.iterator.toint(), cute.AddressSpace.tmem,
                                          assumed_align=16), t.layout)


@cute.jit
def tmem_zero_based(dtype, layout):
    return cute.make_tensor(cute.make_ptr(dtype, Int32(0), cute.AddressSpace.tmem, assumed_align=16), layout)


@cute.jit
def gemm(tiled_mma: cute.TiledMma, acc: cute.Tensor, a: cute.Tensor, b: cute.Tensor, zero_init,
         a_off: cutlass.Constexpr = 0):
    """acc (+)= sum over the k-blocks of b of a[.., kb + a_off] * b[.., kb]"""
    mma = cute.make_mma_atom(tiled_mma.op)
    for kb in cutlass.range_constexpr(cute.size(b.shape[2])):
        mma.set(tcgen05.Field.ACCUMULATE, not zero_init or kb != 0)
        cute.gemm(mma, acc, a[None, None, kb + a_off], b[None, None, kb], acc)


def make_mmas():
    """QK (one 64-key half), QK over a full 128-key S tile (used for TMEM layouts only), PV."""
    k_major = cute.nvgpu.OperandMajorMode.K
    qk = tcgen05.MmaF16BF16Op(BFloat16, Float32, QK_INST, tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
                              k_major, k_major)
    qk_s = tcgen05.MmaF16BF16Op(BFloat16, Float32, (M, N, 16), tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
                                k_major, k_major)
    pv = tcgen05.MmaF16BF16Op(BFloat16, Float32, PV_INST, tcgen05.CtaGroup.ONE, tcgen05.OperandSource.TMEM,
                              k_major, cute.nvgpu.OperandMajorMode.MN)
    return cute.make_tiled_mma(qk), cute.make_tiled_mma(qk_s), cute.make_tiled_mma(pv)


def make_smem_layouts(tiled_qk, tiled_pv):
    q_layout = sm100_utils.make_smem_layout_a(tiled_qk, QK_TILE, BFloat16, 1)
    k_layout = sm100_utils.make_smem_layout_b(tiled_qk, QK_TILE, BFloat16, 2)       # two 64-token halves
    v_layout = sm100_utils.make_smem_layout_b(tiled_pv, PV_TILE, BFloat16, 2)
    p_layout = sm100_utils.make_smem_layout_a(tiled_pv, (M, D, N), BFloat16, 1)     # P: 128 keys
    return q_layout, k_layout, v_layout, p_layout


def q_tma_atom(q, tiled_qk, q_layout):
    """TMA atom and its global tensor for 128-row query tiles."""
    op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
    q_mkl = cute.make_tensor(q.iterator, cute.select(q.layout, mode=[0, 2, 1]))
    return cute.nvgpu.make_tiled_tma_atom_A(op, q_mkl, cute.select(q_layout, mode=[0, 1, 2]), QK_TILE, tiled_qk)


def kv_tma_atoms(k, v, tiled_qk, tiled_pv, k_layout, v_layout):
    """TMA atoms and global tensors for 64-token key and value blocks."""
    op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
    k_nkl = cute.make_tensor(k.iterator, cute.select(k.layout, mode=[0, 2, 1]))
    v_nkl = cute.make_tensor(v.iterator, cute.select(v.layout, mode=[2, 0, 1]))
    tma_k = cute.nvgpu.make_tiled_tma_atom_B(op, k_nkl, cute.select(k_layout, mode=[0, 1, 2]), QK_TILE, tiled_qk)
    tma_v = cute.nvgpu.make_tiled_tma_atom_B(op, v_nkl, cute.select(v_layout, mode=[0, 1, 2]), PV_TILE, tiled_pv)
    return tma_k, tma_v
