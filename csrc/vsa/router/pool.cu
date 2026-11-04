// VSA2 pooling: one pass over the q/k/v tokens of a 64-token block produces
//   * router pools  qr/kr : mean of each R-token group (G = 64/R groups per block)
//   * coarse pools  qc/kc/vc : mean of the block's valid tokens
// Only the first blk_cnt[blk] tokens of a block are read (valid-first layout),
// so padding contents never matter. Empty router groups are written as 0 and
// masked by the router kernels. Router pooling is stop-gradient; coarse
// pooling is differentiated in the backward (unpool: grad/cnt per valid token).
#include "../common.cuh"
#include "../vsa2_kernels.h"

namespace vsa2 {

template <typename Elt, int D>
__global__ void pool_kernel(const PoolParams p) {
  using T = EltTraits<Elt>;
  constexpr int CP = D / 2;  // column pairs
  const int blk = blockIdx.x;
  const int bh = blockIdx.y;
  const int tensor = threadIdx.x / CP;  // 0 q, 1 k, 2 v
  const int cp = threadIdx.x % CP;
  const int cnt = p.blk_cnt[blk];
  const Elt* base = reinterpret_cast<const Elt*>(tensor == 0 ? p.q : (tensor == 1 ? p.k : p.v));
  const uint32_t* src =
      reinterpret_cast<const uint32_t*>(base + ((size_t)bh * p.L + (size_t)blk * kBlock) * D) + cp;
  Elt* rdst = tensor == 0 ? reinterpret_cast<Elt*>(p.qr) : reinterpret_cast<Elt*>(p.kr);
  float tot0 = 0.f, tot1 = 0.f;
  for (int gi = 0; gi < p.G; ++gi) {
    const int rb = gi * p.R;
    const int re = min(rb + p.R, cnt);
    float a0 = 0.f, a1 = 0.f;
    for (int r = rb; r < re; ++r) {
      float2 x = T::unpack2(src[(size_t)r * CP]);
      a0 += x.x;
      a1 += x.y;
    }
    if (tensor < 2) {
      float inv = re > rb ? 1.f / (float)(re - rb) : 0.f;
      size_t row = (size_t)bh * p.Lr_pad + (size_t)blk * p.G + gi;
      reinterpret_cast<uint32_t*>(rdst + row * D)[cp] = T::pack2(a0 * inv, a1 * inv);
    }
    tot0 += a0;
    tot1 += a1;
  }
  Elt* cdst = reinterpret_cast<Elt*>(tensor == 0 ? p.qc : (tensor == 1 ? p.kc : p.vc));
  float inv = 1.f / (float)cnt;
  size_t row = (size_t)bh * p.NBc_pad + blk;
  reinterpret_cast<uint32_t*>(cdst + row * D)[cp] = T::pack2(tot0 * inv, tot1 * inv);
}

template <typename Elt, int D>
static void run_pool(const PoolParams& p, cudaStream_t s) {
  dim3 grid(p.NB, p.BH);
  pool_kernel<Elt, D><<<grid, 3 * (D / 2), 0, s>>>(p);
}

void launch_pool(const PoolParams& p, int D, DType dt, cudaStream_t s) {
  if (dt == DType::kBF16) {
    if (D == 64) run_pool<__nv_bfloat16, 64>(p, s);
    else run_pool<__nv_bfloat16, 128>(p, s);
  } else {
    if (D == 64) run_pool<__half, 64>(p, s);
    else run_pool<__half, 128>(p, s);
  }
}

}  // namespace vsa2
