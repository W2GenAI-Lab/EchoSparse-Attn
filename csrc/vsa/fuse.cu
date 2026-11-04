// VSA2 gate fusion and backward bookkeeping kernels.
//
//   forward : O[t] = valid(t) ? O_f[t] + act(gate[t]) * O_c[blk(t)] : 0
//   backward: dO_f[t]   = valid(t) ? dO[t] : 0   (optional; TK path only)  (fine-branch dO)
//             delta_f[t]= sum_d dO_f[t] * O_f[t]                    (fine softmax bwd)
//             dO_c[blk] = sum_{valid t in blk} act(gate[t]) * dO[t] (repeat^T)
//             delta_c   = sum_d dO_c[blk] * O_c[blk]                (coarse softmax bwd)
//             dgate[t]  = act'(gate[t]) * <dO[t], O_c[blk]>  (per-token)  or elementwise
//   finalize: dX[t] = valid(t) ? acc[t] + add[blk]/cnt(blk) : 0     (coarse unpooling)
// One CTA per (b*h, 64-token block), 128 threads, 16-byte vector accesses.
// Reductions are in a fixed order (deterministic).
#include "common.cuh"
#include "vsa2_kernels.h"

namespace vsa2 {

namespace {

__device__ __forceinline__ float gate_act(float g, int act) { return act == 0 ? tanhf(g) : g; }
__device__ __forceinline__ float gate_dact(float a, int act) { return act == 0 ? 1.f - a * a : 1.f; }

template <typename Elt>
__device__ __forceinline__ void load8(const Elt* src, float* x) {
  using T = EltTraits<Elt>;
  uint4 u = *reinterpret_cast<const uint4*>(src);
  float2 a = T::unpack2(u.x), b = T::unpack2(u.y), c = T::unpack2(u.z), d = T::unpack2(u.w);
  x[0] = a.x; x[1] = a.y; x[2] = b.x; x[3] = b.y; x[4] = c.x; x[5] = c.y; x[6] = d.x; x[7] = d.y;
}
template <typename Elt>
__device__ __forceinline__ void store8(Elt* dst, const float* x) {
  using T = EltTraits<Elt>;
  uint4 u;
  u.x = T::pack2(x[0], x[1]);
  u.y = T::pack2(x[2], x[3]);
  u.z = T::pack2(x[4], x[5]);
  u.w = T::pack2(x[6], x[7]);
  *reinterpret_cast<uint4*>(dst) = u;
}

}  // namespace

template <typename Elt, int D>
__global__ void __launch_bounds__(128) fuse_fwd_kernel(const FuseFwdParams p) {
  using T = EltTraits<Elt>;
  constexpr int CPR = D / 8, RSTEP = 128 / CPR, NPER = kBlock / RSTEP;
  const int blk = blockIdx.x, bh = blockIdx.y;
  const int ch = threadIdx.x % CPR, r0 = threadIdx.x / CPR;
  const int cnt = p.blk_cnt[blk];
  float oc[8];
  load8(reinterpret_cast<const Elt*>(p.oc) + ((size_t)bh * p.NBc_pad + blk) * D + ch * 8, oc);
  const Elt* gate = reinterpret_cast<const Elt*>(p.gate);
#pragma unroll
  for (int i = 0; i < NPER; ++i) {
    const int row = r0 + i * RSTEP;
    const size_t tok = (size_t)bh * p.L + (size_t)blk * kBlock + row;
    float o[8];
    if (row < cnt) {
      load8(reinterpret_cast<const Elt*>(p.of) + tok * D + ch * 8, o);
      if (p.gate_mode == 1) {
        float a = gate_act(T::to_f(gate[tok]), p.gate_act);
#pragma unroll
        for (int e = 0; e < 8; ++e) o[e] += a * oc[e];
      } else if (p.gate_mode == 2) {
        float gv[8];
        load8(gate + tok * D + ch * 8, gv);
#pragma unroll
        for (int e = 0; e < 8; ++e) o[e] += gate_act(gv[e], p.gate_act) * oc[e];
      } else {
#pragma unroll
        for (int e = 0; e < 8; ++e) o[e] += oc[e];
      }
    } else {
#pragma unroll
      for (int e = 0; e < 8; ++e) o[e] = 0.f;
    }
    store8(reinterpret_cast<Elt*>(p.o) + tok * D + ch * 8, o);
  }
}

template <typename Elt, int D>
__global__ void __launch_bounds__(128) fuse_bwd_kernel(const FuseBwdParams p) {
  using T = EltTraits<Elt>;
  constexpr int CPR = D / 8, RSTEP = 128 / CPR, NPER = kBlock / RSTEP;
  __shared__ float red[RSTEP][D];
  __shared__ float red2[128 / 32];
  const int blk = blockIdx.x, bh = blockIdx.y;
  const int ch = threadIdx.x % CPR, r0 = threadIdx.x / CPR;
  const int cnt = p.blk_cnt[blk];
  float oc[8];
  load8(reinterpret_cast<const Elt*>(p.oc) + ((size_t)bh * p.NBc_pad + blk) * D + ch * 8, oc);
  const Elt* gate = reinterpret_cast<const Elt*>(p.gate);
  Elt* dgate = reinterpret_cast<Elt*>(p.dgate);
  float dc[8];
#pragma unroll
  for (int e = 0; e < 8; ++e) dc[e] = 0.f;

#pragma unroll
  for (int i = 0; i < NPER; ++i) {
    const int row = r0 + i * RSTEP;
    const bool valid = row < cnt;
    const size_t tok = (size_t)bh * p.L + (size_t)blk * kBlock + row;
    float d[8], of[8];
    if (valid) {
      load8(reinterpret_cast<const Elt*>(p.dout) + tok * D + ch * 8, d);
      load8(reinterpret_cast<const Elt*>(p.of) + tok * D + ch * 8, of);
    } else {
#pragma unroll
      for (int e = 0; e < 8; ++e) d[e] = of[e] = 0.f;
    }
    if (p.dout_f) store8(reinterpret_cast<Elt*>(p.dout_f) + tok * D + ch * 8, d);
    float pdel = 0.f, pg = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) pdel += d[e] * of[e];
    if (p.gate_mode == 1) {
      float a = valid ? gate_act(T::to_f(gate[tok]), p.gate_act) : 0.f;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        pg += d[e] * oc[e];
        dc[e] += a * d[e];
      }
      pg *= gate_dact(a, p.gate_act);
    } else if (p.gate_mode == 2) {
      float gv[8], dg[8];
      if (valid) load8(gate + tok * D + ch * 8, gv);
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        float a = valid ? gate_act(gv[e], p.gate_act) : 0.f;
        dg[e] = valid ? gate_dact(a, p.gate_act) * d[e] * oc[e] : 0.f;
        dc[e] += a * d[e];
      }
      store8(dgate + tok * D + ch * 8, dg);
    } else {
#pragma unroll
      for (int e = 0; e < 8; ++e) dc[e] += d[e];
    }
    // reduce across the CPR threads that share this row (contiguous lanes)
#pragma unroll
    for (int off = CPR / 2; off > 0; off >>= 1) {
      pdel += __shfl_xor_sync(0xffffffff, pdel, off);
      pg += __shfl_xor_sync(0xffffffff, pg, off);
    }
    if (ch == 0) {
      p.delta_f[tok] = valid ? pdel : 0.f;
      if (p.gate_mode == 1) dgate[tok] = T::from_f(valid ? pg : 0.f);
    }
  }

  // dO_c[blk] = sum over rows, fixed order
#pragma unroll
  for (int e = 0; e < 8; ++e) red[r0][ch * 8 + e] = dc[e];
  __syncthreads();
  float dcr = 0.f, part = 0.f;
  if (threadIdx.x < D) {
    float s = 0.f;
#pragma unroll
    for (int r = 0; r < RSTEP; ++r) s += red[r][threadIdx.x];
    Elt sr = T::from_f(s);
    reinterpret_cast<Elt*>(p.dout_c)[((size_t)bh * p.NBc_pad + blk) * D + threadIdx.x] = sr;
    dcr = T::to_f(sr);
    float ocd = T::to_f(reinterpret_cast<const Elt*>(p.oc)[((size_t)bh * p.NBc_pad + blk) * D + threadIdx.x]);
    part = dcr * ocd;
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) part += __shfl_xor_sync(0xffffffff, part, off);
  if ((threadIdx.x & 31) == 0) red2[threadIdx.x >> 5] = part;
  __syncthreads();
  if (threadIdx.x == 0) {
    float s = 0.f;
#pragma unroll
    for (int w = 0; w < 4; ++w) s += red2[w];
    p.delta_c[(size_t)bh * p.NBc_pad + blk] = s;
  }
}

template <typename Elt, int D>
__global__ void __launch_bounds__(128) finalize_kernel(const FinalizeParams p) {
  constexpr int CPR = D / 8, RSTEP = 128 / CPR, NPER = kBlock / RSTEP;
  const int blk = blockIdx.x, bh = blockIdx.y;
  const int ch = threadIdx.x % CPR, r0 = threadIdx.x / CPR;
  const int cnt = p.blk_cnt[blk];
  const float inv = 1.f / (float)cnt;
  float add[8];
  if (p.add) {
    const float4* a = reinterpret_cast<const float4*>(p.add + ((size_t)bh * p.add_nb + blk) * D + ch * 8);
    float4 a0 = a[0], a1 = a[1];
    add[0] = a0.x * inv; add[1] = a0.y * inv; add[2] = a0.z * inv; add[3] = a0.w * inv;
    add[4] = a1.x * inv; add[5] = a1.y * inv; add[6] = a1.z * inv; add[7] = a1.w * inv;
  } else {
#pragma unroll
    for (int e = 0; e < 8; ++e) add[e] = 0.f;
  }
#pragma unroll
  for (int i = 0; i < NPER; ++i) {
    const int row = r0 + i * RSTEP;
    const size_t tok = (size_t)bh * p.L + (size_t)blk * kBlock + row;
    float x[8];
    if (row < cnt) {
      if (p.acc) {
        const float4* s = reinterpret_cast<const float4*>(p.acc + tok * D + ch * 8);
        float4 s0 = s[0], s1 = s[1];
        x[0] = s0.x; x[1] = s0.y; x[2] = s0.z; x[3] = s0.w; x[4] = s1.x; x[5] = s1.y; x[6] = s1.z; x[7] = s1.w;
      } else {
#pragma unroll
        for (int e = 0; e < 8; ++e) x[e] = 0.f;
      }
#pragma unroll
      for (int e = 0; e < 8; ++e) x[e] += add[e];
    } else {
#pragma unroll
      for (int e = 0; e < 8; ++e) x[e] = 0.f;
    }
    store8(reinterpret_cast<Elt*>(p.out) + tok * D + ch * 8, x);
  }
}

#define VSA2_DISPATCH(D, dt, KERNEL, ...)                                  \
  do {                                                                     \
    if (dt == DType::kBF16) {                                              \
      if (D == 64) KERNEL<__nv_bfloat16, 64><<<__VA_ARGS__>>>(p);          \
      else KERNEL<__nv_bfloat16, 128><<<__VA_ARGS__>>>(p);                 \
    } else {                                                               \
      if (D == 64) KERNEL<__half, 64><<<__VA_ARGS__>>>(p);                 \
      else KERNEL<__half, 128><<<__VA_ARGS__>>>(p);                        \
    }                                                                      \
  } while (0)

void launch_fuse_fwd(const FuseFwdParams& p, int D, DType dt, cudaStream_t s) {
  dim3 grid(p.NB, p.BH);
  VSA2_DISPATCH(D, dt, fuse_fwd_kernel, grid, 128, 0, s);
}
void launch_fuse_bwd(const FuseBwdParams& p, int D, DType dt, cudaStream_t s) {
  dim3 grid(p.NB, p.BH);
  VSA2_DISPATCH(D, dt, fuse_bwd_kernel, grid, 128, 0, s);
}
void launch_finalize(const FinalizeParams& p, int D, DType dt, cudaStream_t s) {
  dim3 grid(p.NB, p.BH);
  VSA2_DISPATCH(D, dt, finalize_kernel, grid, 128, 0, s);
}

}  // namespace vsa2
