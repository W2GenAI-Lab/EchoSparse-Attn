// VSA2 generic block-sparse attention backward.
//
// Per selected pair (query block i, kv block j):
//   dP = dO V^T, S = Q K^T                        (64x64, warp tile 16x32)
//   P  = exp(scale*S - lse), dS = P*(dP - delta)*scale
//   dV += P^T dO, dK += dS^T Q                    (64xD, warp tile 16xD/2)
//   dQ += dS K                                    (64xD, warp tile 16xD/2)
//
// Two kernels share these stages:
//
// attn_bwd_kernel: one CTA = one (b*h, 64-row kv block) walking the inverted kv->q list, 8
// warps. dK/dV are complete inside the CTA and written once in the epilogue, optionally
// adding a per-block gradient (coarse unpooling: add[blk]/cnt for every valid token). dQ goes
// to dq_acc with fp32 atomics (DQ_F2: float2, DQ_WORD: one float at a time), or is not computed
// at all (DQ_NONE, when attn_bwd_q_kernel supplies it).
//
// attn_bwd_q_kernel: one CTA = one (b*h, 64-row query block) walking the q->kv list. It repeats
// S and dP (7 products per pair instead of 5 in total) but accumulates dQ in registers: no
// atomics, a deterministic dQ, no fp32 dq_acc and no separate finalize pass.
//
// Shared memory: an RTX 5090 reports sharedMemPerBlockOptin = 101376 B, so at D=128 these kernels
// (72.5-80.5 KB) fit one CTA per SM there. The stage order overlaps tile loads with tensor-core
// work instead of adding buffers. The first 220K RTX measurement shows no gain from this order on
// its own for atomic dQ (202.49 ms before, 202.33 ms after, unpaired):
//   attn_bwd_kernel  : dP | S, P, dS | dV | release dO | dK | release Q | dQ
//     next dO load overlaps dK + dQ; next Q load overlaps dQ + next dP
//   attn_bwd_q_kernel: dP | release V | S, dS | dQ | release K
//     next V load overlaps S + dQ; next K load overlaps next dP
//
// Rows with lse == -inf (padded query rows, zero-edge query blocks) get P = 0 and contribute
// nothing. Padded kv columns (>= blk_cnt) get P = 0.
#include "../common.cuh"
#include "../vsa2_kernels.h"

namespace vsa2 {

namespace {

enum { DQ_F2 = 0, DQ_WORD = 1, DQ_NONE = 2 };

// acc[4][4] = A B^T, warp tile: rows 16*wm of A, columns 32*wn = rows of B
template <typename Elt, int D>
__device__ __forceinline__ void mm_abt(float (&acc)[4][4], const Elt* sA, const Elt* sB, int wm, int wn, int lane) {
  using T = EltTraits<Elt>;
#pragma unroll
  for (int i = 0; i < 4; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
#pragma unroll
  for (int kk = 0; kk < D / 16; ++kk) {
    uint32_t a[4];
    ldsm_x4(a[0], a[1], a[2], a[3], smem_u32(sA + swz<D>(16 * wm + (lane & 15), kk * 16 + (lane >> 4) * 8)));
#pragma unroll
    for (int np = 0; np < 2; ++np) {
      uint32_t b0, b1, b2, b3;
      ldsm_x4(b0, b1, b2, b3,
              smem_u32(sB + swz<D>(32 * wn + np * 16 + (lane & 7) + ((lane >> 4) << 3), kk * 16 + ((lane >> 3) & 1) * 8)));
      T::mma(acc[2 * np], a, b0, b1);
      T::mma(acc[2 * np + 1], a, b2, b3);
    }
  }
}

// P[row,col] (optional) and dS[row,col] from S and dP, into swizzled shared memory
template <typename Elt, int D, bool STORE_P>
__device__ __forceinline__ void store_p_ds(float (&s)[4][4], float (&dp)[4][4], Elt* sP, Elt* sdS,
                                           const float* sL, const float* sDl, float scale_log2, float scale, int kcnt,
                                           int wm, int wn, int g, int t) {
  using T = EltTraits<Elt>;
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int row = 16 * wm + g + 8 * r;
    const float lse = sL[row];
    const float dl = sDl[row];
    const bool rv = lse != neg_inf();
    const float lse2 = lse * kLog2e;
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      const int col = 32 * wn + nt * 8 + 2 * t;
      float pv[2], dsv[2];
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const bool ok = rv && (col + e) < kcnt;
        pv[e] = ok ? fast_exp2(s[nt][2 * r + e] * scale_log2 - lse2) : 0.f;
        dsv[e] = pv[e] * (dp[nt][2 * r + e] - dl) * scale;
      }
      if constexpr (STORE_P) *reinterpret_cast<uint32_t*>(sP + swz<kBlock>(row, col)) = T::pack2(pv[0], pv[1]);
      *reinterpret_cast<uint32_t*>(sdS + swz<kBlock>(row, col)) = T::pack2(dsv[0], dsv[1]);
    }
  }
}

// acc += X^T B, warp tile: rows 16*wm of the result (columns of X), head-dim half wn
template <typename Elt, int D>
__device__ __forceinline__ void mma_xtb(float (&acc)[D / 16][4], const Elt* sX, const Elt* sB, int wm, int wn, int lane) {
  using T = EltTraits<Elt>;
#pragma unroll
  for (int kk = 0; kk < 4; ++kk) {
    uint32_t a[4];
    ldsm_x4_t(a[0], a[1], a[2], a[3],
              smem_u32(sX + swz<kBlock>(kk * 16 + (lane & 7) + ((lane >> 4) << 3), 16 * wm + ((lane >> 3) & 1) * 8)));
#pragma unroll
    for (int dq2 = 0; dq2 < D / 32; ++dq2) {
      uint32_t b0, b1, b2, b3;
      ldsm_x4_t(b0, b1, b2, b3, smem_u32(sB + swz<D>(kk * 16 + (lane & 15), wn * (D / 2) + dq2 * 16 + (lane >> 4) * 8)));
      T::mma(acc[2 * dq2], a, b0, b1);
      T::mma(acc[2 * dq2 + 1], a, b2, b3);
    }
  }
}

// acc += dS K, warp tile: rows 16*wm of dS, head-dim half wn
template <typename Elt, int D>
__device__ __forceinline__ void mma_dsk(float (&acc)[D / 16][4], const Elt* sdS, const Elt* sK, int wm, int wn, int lane) {
  using T = EltTraits<Elt>;
#pragma unroll
  for (int kk = 0; kk < 4; ++kk) {
    uint32_t a[4];
    ldsm_x4(a[0], a[1], a[2], a[3], smem_u32(sdS + swz<kBlock>(16 * wm + (lane & 15), kk * 16 + (lane >> 4) * 8)));
#pragma unroll
    for (int dq2 = 0; dq2 < D / 32; ++dq2) {
      uint32_t b0, b1, b2, b3;
      ldsm_x4_t(b0, b1, b2, b3, smem_u32(sK + swz<D>(kk * 16 + (lane & 15), wn * (D / 2) + dq2 * 16 + (lane >> 4) * 8)));
      T::mma(acc[2 * dq2], a, b0, b1);
      T::mma(acc[2 * dq2 + 1], a, b2, b3);
    }
  }
}

}  // namespace

template <typename Elt, int D, int DQ>
__global__ void __launch_bounds__(256) attn_bwd_kernel(const AttnBwdParams p) {
  using T = EltTraits<Elt>;
  constexpr int BM = kBlock, BN = kBlock, NT = 256;
  constexpr int NH = D / 16;  // 8-wide n tiles in half of the head dim
  extern __shared__ __align__(128) uint8_t smem_raw[];
  Elt* sK = reinterpret_cast<Elt*>(smem_raw);
  Elt* sV = sK + BN * D;
  Elt* sQ = sV + BN * D;
  Elt* sdO = sQ + BM * D;
  Elt* sP = sdO + BM * D;
  Elt* sdS = sP + BM * BN;
  float* sL = reinterpret_cast<float*>(sdS + BM * BN);
  float* sDl = sL + BM;

  const int kb = blockIdx.x;
  const int bh = blockIdx.y;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t = lane & 3;
  const size_t tok_base = (size_t)bh * p.L;
  const Elt* gQ = reinterpret_cast<const Elt*>(p.q) + tok_base * D;
  const Elt* gdO = reinterpret_cast<const Elt*>(p.dout) + tok_base * D;
  const float* gL = p.lse + tok_base;
  const float* gDl = p.delta + tok_base;

  const int* rp = p.row_ptr + (size_t)bh * (p.NB + 1) + kb;
  const int beg = rp[0];
  const int nq = rp[1] - beg;
  const int* cidx = p.col_idx + beg;
  const int kcnt = p.blk_cnt[kb];

  auto issue_do = [&](int qb) {
    load_tile_async<BM, D, NT>(sdO, gdO + (size_t)qb * BM * D, tid);
    if (tid < 16) cp_async_16(smem_u32(sDl + tid * 4), gDl + (size_t)qb * BM + tid * 4);
    cp_async_commit();
  };
  auto issue_q = [&](int qb) {
    load_tile_async<BM, D, NT>(sQ, gQ + (size_t)qb * BM * D, tid);
    if (tid < 16) cp_async_16(smem_u32(sL + tid * 4), gL + (size_t)qb * BM + tid * 4);
    cp_async_commit();
  };
  // A kv block no query selected issues no cp.async: no load traffic, and nothing is
  // still in flight into shared memory when the CTA exits through the epilogue.
  if (nq > 0) {
    load_tile_async<BN, D, NT>(sK, reinterpret_cast<const Elt*>(p.k) + (tok_base + (size_t)kb * BN) * D, tid);
    load_tile_async<BN, D, NT>(sV, reinterpret_cast<const Elt*>(p.v) + (tok_base + (size_t)kb * BN) * D, tid);
    issue_do(cidx[0]);  // first group: K, V, dO, delta
    issue_q(cidx[0]);   // second group: Q, lse
  }

  float dk[NH][4], dv[NH][4];
#pragma unroll
  for (int i = 0; i < NH; ++i) {
    dk[i][0] = dk[i][1] = dk[i][2] = dk[i][3] = 0.f;
    dv[i][0] = dv[i][1] = dv[i][2] = dv[i][3] = 0.f;
  }
  const int wm = warp & 3;   // 16-row slice (q rows for S/dQ, kv rows for dK/dV)
  const int wn = warp >> 2;  // half (kv cols for S, head-dim half for dK/dV/dQ)

  for (int it = 0; it < nq; ++it) {
    const int qb = cidx[it];
    const bool has_next = it + 1 < nq;
    float s[4][4], dp[4][4];

    cp_async_wait<1>();
    __syncthreads();  // this pair's dO landed; every warp finished dS K of the previous pair
    mm_abt<Elt, D>(dp, sdO, sV, wm, wn, lane);

    cp_async_wait<0>();
    __syncthreads();  // this pair's Q and lse landed
    mm_abt<Elt, D>(s, sQ, sK, wm, wn, lane);
    store_p_ds<Elt, D, true>(s, dp, sP, sdS, sL, sDl, p.scale_log2, p.scale, kcnt, wm, wn, g, t);
    __syncthreads();  // P and dS visible to every warp

    mma_xtb<Elt, D>(dv, sP, sdO, wm, wn, lane);
    __syncthreads();  // dO and delta free
    if (has_next) issue_do(cidx[it + 1]);

    mma_xtb<Elt, D>(dk, sdS, sQ, wm, wn, lane);
    __syncthreads();  // Q and lse free
    if (has_next) issue_q(cidx[it + 1]);

    if constexpr (DQ != DQ_NONE) {
      float dq[NH][4];
#pragma unroll
      for (int i = 0; i < NH; ++i) dq[i][0] = dq[i][1] = dq[i][2] = dq[i][3] = 0.f;
      mma_dsk<Elt, D>(dq, sdS, sK, wm, wn, lane);
      const int qcnt = p.blk_cnt[qb];
      float* dqg = p.dq_acc + (tok_base + (size_t)qb * BM) * D;
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        int row = 16 * wm + g + 8 * r;
        if (row < qcnt) {
#pragma unroll
          for (int nd = 0; nd < NH; ++nd) {
            int col = wn * (D / 2) + nd * 8 + 2 * t;
            float* dst = dqg + (size_t)row * D + col;
            if constexpr (DQ == DQ_F2) {
              atomic_add_f2(dst, dq[nd][2 * r], dq[nd][2 * r + 1]);
            } else {
              atomicAdd(dst, dq[nd][2 * r]);
              atomicAdd(dst + 1, dq[nd][2 * r + 1]);
            }
          }
        }
      }
    }
  }

  // ---- epilogue: dK, dV (+ coarse unpooling add)
  const float inv_cnt = 1.f / (float)kcnt;
  const float* addk = p.add_dk ? p.add_dk + ((size_t)bh * p.add_nb + kb) * D : nullptr;
  const float* addv = p.add_dv ? p.add_dv + ((size_t)bh * p.add_nb + kb) * D : nullptr;
  const size_t out_base = (tok_base + (size_t)kb * BN) * D;
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    int row = 16 * wm + g + 8 * r;
    bool valid = row < kcnt;
#pragma unroll
    for (int nd = 0; nd < NH; ++nd) {
      int col = wn * (D / 2) + nd * 8 + 2 * t;
      float k0 = dk[nd][2 * r], k1 = dk[nd][2 * r + 1];
      float v0 = dv[nd][2 * r], v1 = dv[nd][2 * r + 1];
      if (addk) { k0 += addk[col] * inv_cnt; k1 += addk[col + 1] * inv_cnt; }
      if (addv) { v0 += addv[col] * inv_cnt; v1 += addv[col + 1] * inv_cnt; }
      if (!valid) k0 = k1 = v0 = v1 = 0.f;
      size_t off = out_base + (size_t)row * D + col;
      if (p.dk_f32) {
        *reinterpret_cast<float2*>(p.dk_f32 + off) = make_float2(k0, k1);
        *reinterpret_cast<float2*>(p.dv_f32 + off) = make_float2(v0, v1);
      } else {
        *reinterpret_cast<uint32_t*>(reinterpret_cast<Elt*>(p.dk) + off) = T::pack2(k0, k1);
        *reinterpret_cast<uint32_t*>(reinterpret_cast<Elt*>(p.dv) + off) = T::pack2(v0, v1);
      }
    }
  }
}

template <typename Elt, int D>
__global__ void __launch_bounds__(256) attn_bwd_q_kernel(const AttnBwdQParams p) {
  using T = EltTraits<Elt>;
  constexpr int BM = kBlock, BN = kBlock, NT = 256;
  constexpr int NH = D / 16;
  extern __shared__ __align__(128) uint8_t smem_raw[];
  Elt* sQ = reinterpret_cast<Elt*>(smem_raw);
  Elt* sdO = sQ + BM * D;
  Elt* sK = sdO + BM * D;
  Elt* sV = sK + BN * D;
  Elt* sdS = sV + BN * D;
  float* sL = reinterpret_cast<float*>(sdS + BM * BN);
  float* sDl = sL + BM;

  const int qb = blockIdx.x;
  const int bh = blockIdx.y;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t = lane & 3;
  const size_t tok_base = (size_t)bh * p.L;
  const size_t q_base = tok_base + (size_t)qb * BM;
  const Elt* gK = reinterpret_cast<const Elt*>(p.k) + tok_base * D;
  const Elt* gV = reinterpret_cast<const Elt*>(p.v) + tok_base * D;

  const int* rp = p.row_ptr + (size_t)bh * (p.NB + 1) + qb;
  const int beg = rp[0];
  const int nkv = rp[1] - beg;
  const int* cidx = p.col_idx + beg;
  const int qcnt = p.blk_cnt[qb];

  auto issue_v = [&](int kvb) {
    load_tile_async<BN, D, NT>(sV, gV + (size_t)kvb * BN * D, tid);
    cp_async_commit();
  };
  auto issue_k = [&](int kvb) {
    load_tile_async<BN, D, NT>(sK, gK + (size_t)kvb * BN * D, tid);
    cp_async_commit();
  };
  if (nkv > 0) {
    load_tile_async<BM, D, NT>(sQ, reinterpret_cast<const Elt*>(p.q) + q_base * D, tid);
    load_tile_async<BM, D, NT>(sdO, reinterpret_cast<const Elt*>(p.dout) + q_base * D, tid);
    if (tid < 16) cp_async_16(smem_u32(sL + tid * 4), p.lse + q_base + tid * 4);
    else if (tid < 32) cp_async_16(smem_u32(sDl + (tid - 16) * 4), p.delta + q_base + (tid - 16) * 4);
    issue_v(cidx[0]);  // first group: Q, dO, lse, delta, V
    issue_k(cidx[0]);  // second group: K
  }

  float dq[NH][4];
#pragma unroll
  for (int i = 0; i < NH; ++i) dq[i][0] = dq[i][1] = dq[i][2] = dq[i][3] = 0.f;
  const int wm = warp & 3, wn = warp >> 2;

  for (int it = 0; it < nkv; ++it) {
    const int kvb = cidx[it];
    const bool has_next = it + 1 < nkv;
    float s[4][4], dp[4][4];

    cp_async_wait<1>();
    __syncthreads();  // this pair's V landed; every warp finished dS K of the previous pair
    mm_abt<Elt, D>(dp, sdO, sV, wm, wn, lane);

    cp_async_wait<0>();
    __syncthreads();  // this pair's K landed; V free
    if (has_next) issue_v(cidx[it + 1]);

    mm_abt<Elt, D>(s, sQ, sK, wm, wn, lane);
    store_p_ds<Elt, D, false>(s, dp, nullptr, sdS, sL, sDl, p.scale_log2, p.scale, p.blk_cnt[kvb], wm, wn, g, t);
    __syncthreads();  // dS visible to every warp

    mma_dsk<Elt, D>(dq, sdS, sK, wm, wn, lane);
    __syncthreads();  // K free
    if (has_next) issue_k(cidx[it + 1]);
  }

  // ---- epilogue: dQ (+ coarse unpooling add)
  const float inv_cnt = 1.f / (float)qcnt;
  const float* add = p.add_dq ? p.add_dq + ((size_t)bh * p.add_nb + qb) * D : nullptr;
  Elt* out = reinterpret_cast<Elt*>(p.dq) + q_base * D;
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    int row = 16 * wm + g + 8 * r;
    bool valid = row < qcnt;
#pragma unroll
    for (int nd = 0; nd < NH; ++nd) {
      int col = wn * (D / 2) + nd * 8 + 2 * t;
      float q0 = dq[nd][2 * r], q1 = dq[nd][2 * r + 1];
      if (add) { q0 += add[col] * inv_cnt; q1 += add[col + 1] * inv_cnt; }
      if (!valid) q0 = q1 = 0.f;
      *reinterpret_cast<uint32_t*>(out + (size_t)row * D + col) = T::pack2(q0, q1);
    }
  }
}

template <typename Elt, int D, int DQ>
static void run_bwd_mode(const AttnBwdParams& p, cudaStream_t s) {
  constexpr int smem = (2 * kBlock * D * 2 + 2 * kBlock * kBlock) * sizeof(Elt) + 2 * kBlock * sizeof(float);
  auto kern = attn_bwd_kernel<Elt, D, DQ>;
  cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  dim3 grid(p.NB, p.BH);
  kern<<<grid, 256, smem, s>>>(p);
}

template <typename Elt, int D>
static void run_bwd(const AttnBwdParams& p, cudaStream_t s) {
  if (!p.dq_acc) run_bwd_mode<Elt, D, DQ_NONE>(p, s);
  else if (p.dq_word) run_bwd_mode<Elt, D, DQ_WORD>(p, s);
  else run_bwd_mode<Elt, D, DQ_F2>(p, s);
}

void launch_attn_bwd(const AttnBwdParams& p, int D, DType dt, cudaStream_t s) {
  if (dt == DType::kBF16) {
    if (D == 64) run_bwd<__nv_bfloat16, 64>(p, s);
    else run_bwd<__nv_bfloat16, 128>(p, s);
  } else {
    if (D == 64) run_bwd<__half, 64>(p, s);
    else run_bwd<__half, 128>(p, s);
  }
}

template <typename Elt, int D>
static void run_bwd_q(const AttnBwdQParams& p, cudaStream_t s) {
  constexpr int smem = (4 * kBlock * D + kBlock * kBlock) * sizeof(Elt) + 2 * kBlock * sizeof(float);
  auto kern = attn_bwd_q_kernel<Elt, D>;
  cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  dim3 grid(p.NB, p.BH);
  kern<<<grid, 256, smem, s>>>(p);
}

void launch_attn_bwd_q(const AttnBwdQParams& p, int D, DType dt, cudaStream_t s) {
  if (dt == DType::kBF16) {
    if (D == 64) run_bwd_q<__nv_bfloat16, 64>(p, s);
    else run_bwd_q<__nv_bfloat16, 128>(p, s);
  } else {
    if (D == 64) run_bwd_q<__half, 64>(p, s);
    else run_bwd_q<__half, 128>(p, s);
  }
}

}  // namespace vsa2
