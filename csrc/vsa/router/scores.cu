// Original VSA router scores: S[i, j] = rnd(rnd(qc_i . kc_j) * scale) over whole 64-token block means.
//
// One CTA = one (b*h, 64 query blocks, 64 kv blocks) tile, 4 warps x 16 rows, the S = Q K^T fragment loop of
// the generic attention forward (mma.sync m16n8k16, fp32 accumulation). rnd rounds to the token dtype, as
// Torch does for the original's bf16 `(q_c @ k_c^T) / sqrt(D)`; the ranking then runs on these 16-bit values.
// The tile is staged in shared memory so the [NB, NB] rows are written contiguously.
#include "../common.cuh"
#include "../vsa2_kernels.h"

namespace vsa2 {

template <typename Elt, int D>
__global__ void __launch_bounds__(128) block_scores_kernel(const BlockScoreParams p) {
  using T = EltTraits<Elt>;
  constexpr int KD = D / 16, SO = kBlock + 8;
  extern __shared__ __align__(128) uint8_t smem_raw[];
  Elt* sQ = reinterpret_cast<Elt*>(smem_raw);
  Elt* sK = sQ + kBlock * D;
  const int cb = blockIdx.x, rb = blockIdx.y, bhl = blockIdx.z;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t = lane & 3, r0 = warp * 16;
  const size_t bh = (size_t)(p.bh0 + bhl);
  load_tile_async<kBlock, D, 128>(sQ, reinterpret_cast<const Elt*>(p.qc) + (bh * p.NBc_pad + rb * kBlock) * D, tid);
  load_tile_async<kBlock, D, 128>(sK, reinterpret_cast<const Elt*>(p.kc) + (bh * p.NBc_pad + cb * kBlock) * D, tid);
  cp_async_commit();
  cp_async_wait<0>();
  __syncthreads();

  float s[8][4];
#pragma unroll
  for (int i = 0; i < 8; ++i) s[i][0] = s[i][1] = s[i][2] = s[i][3] = 0.f;
#pragma unroll
  for (int kk = 0; kk < KD; ++kk) {
    uint32_t a[4];
    ldsm_x4(a[0], a[1], a[2], a[3], smem_u32(sQ + swz<D>(r0 + (lane & 15), kk * 16 + (lane >> 4) * 8)));
#pragma unroll
    for (int np = 0; np < 4; ++np) {
      uint32_t b0, b1, b2, b3;
      ldsm_x4(b0, b1, b2, b3,
              smem_u32(sK + swz<D>(np * 16 + (lane & 7) + ((lane >> 4) << 3), kk * 16 + ((lane >> 3) & 1) * 8)));
      T::mma(s[2 * np], a, b0, b1);
      T::mma(s[2 * np + 1], a, b2, b3);
    }
  }
  __syncthreads();  // every warp is done with sQ; reuse it as the [64][SO] output tile

  Elt* sO = sQ;
#pragma unroll
  for (int nt = 0; nt < 8; ++nt)
#pragma unroll
    for (int e = 0; e < 4; ++e)
      sO[(r0 + g + 8 * (e >> 1)) * SO + nt * 8 + 2 * t + (e & 1)] = T::from_f(T::to_f(T::from_f(s[nt][e])) * p.scale);
  __syncthreads();
  Elt* out = reinterpret_cast<Elt*>(p.scores) + (size_t)bhl * p.NB * p.NB;
  for (int i = tid; i < kBlock * kBlock; i += 128) {
    const int r = rb * kBlock + i / kBlock, c = cb * kBlock + i % kBlock;
    if (r < p.NB && c < p.NB) out[(size_t)r * p.NB + c] = sO[(i / kBlock) * SO + i % kBlock];
  }
}

template <typename Elt, int D>
static void run_block_scores(const BlockScoreParams& p, cudaStream_t s) {
  constexpr int smem = 2 * kBlock * D * sizeof(Elt);
  const int tiles = (p.NB + kBlock - 1) / kBlock;
  block_scores_kernel<Elt, D><<<dim3(tiles, tiles, p.bh_count), 128, smem, s>>>(p);
}

void launch_block_scores(const BlockScoreParams& p, int D, DType dt, cudaStream_t s) {
  if (dt == DType::kBF16) {
    if (D == 64) run_block_scores<__nv_bfloat16, 64>(p, s);
    else run_block_scores<__nv_bfloat16, 128>(p, s);
  } else {
    if (D == 64) run_block_scores<__half, 64>(p, s);
    else run_block_scores<__half, 128>(p, s);
  }
}

}  // namespace vsa2
