// VSA2 native kernels: shared device helpers.
//
// Raw PTX wrappers for the SM80-class warp-level tensor core path
// (mma.sync m16n8k16, ldmatrix, cp.async). These instructions are supported on
// sm_80, sm_86, sm_89, sm_90(a) and sm_120(a), so the "generic" kernels built on
// them are the native Tensor Core path for RTX 5090 (SM120 has no wgmma/tcgen05)
// and a cross-check path on H100.
//
// Fragment layouts follow the PTX ISA "Matrix Fragments for mma.m16n8k16" for
// 16-bit A/B types and f32 C/D:
//   groupID g = lane >> 2, threadID_in_group t = lane & 3
//   A (16x16, row): a0 (g, 2t..2t+1) a1 (g+8, 2t..) a2 (g, 2t+8..) a3 (g+8, 2t+8..)
//   B (16x8, col):  b0 (k=2t..2t+1, n=g)  b1 (k=2t+8.., n=g)
//   C (16x8):       c0,c1 (g, 2t..2t+1)   c2,c3 (g+8, 2t..2t+1)
#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

namespace vsa2 {

constexpr float kLog2e = 1.4426950408889634f;
constexpr float kLn2 = 0.6931471805599453f;

__device__ __forceinline__ float neg_inf() { return __uint_as_float(0xff800000u); }

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// 16-byte async global->shared copy. When pred is false, 16 zero bytes are
// written (src-size 0) and no global memory is read.
__device__ __forceinline__ void cp_async_16(uint32_t dst, const void* src, bool pred = true) {
  int sz = pred ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(sz));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ void ldsm_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}

__device__ __forceinline__ float fast_exp2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
  return y;
}

// Element traits: bf16 (primary) and fp16.
template <typename Elt>
struct EltTraits;

template <>
struct EltTraits<__nv_bfloat16> {
  __device__ __forceinline__ static uint32_t pack2(float a, float b) {
    __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<uint32_t*>(&v);
  }
  __device__ __forceinline__ static float2 unpack2(uint32_t u) {
    __nv_bfloat162 v = *reinterpret_cast<__nv_bfloat162*>(&u);
    return __bfloat1622float2(v);
  }
  __device__ __forceinline__ static float to_f(__nv_bfloat16 x) { return __bfloat162float(x); }
  __device__ __forceinline__ static __nv_bfloat16 from_f(float x) { return __float2bfloat16_rn(x); }
  __device__ __forceinline__ static void mma(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
  }
};

template <>
struct EltTraits<__half> {
  __device__ __forceinline__ static uint32_t pack2(float a, float b) {
    __half2 v = __floats2half2_rn(a, b);
    return *reinterpret_cast<uint32_t*>(&v);
  }
  __device__ __forceinline__ static float2 unpack2(uint32_t u) {
    __half2 v = *reinterpret_cast<__half2*>(&u);
    return __half22float2(v);
  }
  __device__ __forceinline__ static float to_f(__half x) { return __half2float(x); }
  __device__ __forceinline__ static __half from_f(float x) { return __float2half_rn(x); }
  __device__ __forceinline__ static void mma(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
  }
};

// XOR-swizzled element offset inside a row-major [rows][COLS] tile of 16-bit
// elements. 16-byte chunks are permuted by (row & 7) so that ldmatrix of 8
// consecutive rows at the same logical chunk hits 8 distinct bank groups.
// Requires COLS >= 64 (at least 8 chunks per row).
template <int COLS>
__device__ __forceinline__ int swz(int row, int col) {
  static_assert(COLS % 64 == 0, "swizzle needs >= 8 16B chunks per row");
  int chunk = col >> 3;
  return row * COLS + (((chunk ^ (row & 7)) << 3) | (col & 7));
}

// Asynchronously copy a [ROWS][COLS] 16-bit tile from global (row stride
// COLS elements, contiguous) into swizzled shared memory. Rows >= valid_rows
// are zero-filled without touching global memory.
template <int ROWS, int COLS, int NTHREADS, typename Elt>
__device__ __forceinline__ void load_tile_async(Elt* smem, const Elt* gmem, int tid, int valid_rows = ROWS) {
  constexpr int CHUNKS_PER_ROW = COLS / 8;
  constexpr int TOTAL = ROWS * CHUNKS_PER_ROW;
  static_assert(TOTAL % NTHREADS == 0, "tile chunks must divide threads");
#pragma unroll
  for (int i = 0; i < TOTAL / NTHREADS; ++i) {
    int id = tid + i * NTHREADS;
    int row = id / CHUNKS_PER_ROW;
    int ch = id % CHUNKS_PER_ROW;
    bool ok = row < valid_rows;
    const Elt* src = gmem + (size_t)(ok ? row : 0) * COLS + ch * 8;
    cp_async_16(smem_u32(smem + swz<COLS>(row, ch * 8)), src, ok);
  }
}

__device__ __forceinline__ float warp_quad_max(float v) {
  v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 1));
  v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 2));
  return v;
}
__device__ __forceinline__ float warp_quad_sum(float v) {
  v += __shfl_xor_sync(0xffffffff, v, 1);
  v += __shfl_xor_sync(0xffffffff, v, 2);
  return v;
}

// fp32 atomic add of two consecutive floats (8B aligned).
__device__ __forceinline__ void atomic_add_f2(float* addr, float a, float b) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  atomicAdd(reinterpret_cast<float2*>(addr), make_float2(a, b));
#else
  atomicAdd(addr, a);
  atomicAdd(addr + 1, b);
#endif
}

}  // namespace vsa2
