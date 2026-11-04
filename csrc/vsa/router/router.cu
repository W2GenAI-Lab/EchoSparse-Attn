// VSA2 pool -> softmax -> pool (PSP) router, two streaming passes, no Lr x Lr
// probability materialization.
//
//   pooled rows   : qr/kr [BH, Lr_pad, D], row = blk*G + g, Lr = NB*G
//   pass 1 (lse)  : lse2[r] = log2 sum_c exp2(scale_log2 * qr_r . kr_c) over ALL valid
//                   pooled keys c of the sequence (softmax over all pooled keys)
//   pass 2 (score): A[r,c] = exp2(scale_log2 * qr_r . kr_c - lse2[r]) recomputed tile by
//                   tile and summed into hardware-block bins in registers:
//                   S[i,j] = norm_i * sum_{a in grp(i), b in grp(j)} A[(i,a),(j,b)]
//                   norm_i = 1/#valid query groups of block i (row_norm=0, each row of S
//                   sums to 1) or 1/G^2 (row_norm=1, the paper's formula verbatim; equal
//                   to row_norm=0 up to the constant G for full blocks).
// A pooled key/query group is valid iff it holds >= 1 valid token.
// Each S[i,j] is produced by exactly one warp of one CTA in a fixed order, so the
// scores are deterministic run-to-run.
#include "../common.cuh"
#include "../vsa2_kernels.h"

namespace vsa2 {

template <typename Elt, int D, int G, bool SCORES>
__global__ void __launch_bounds__(128) router_kernel(const RouterParams p) {
  using T = EltTraits<Elt>;
  constexpr int BM = 64, BN = 64, NT = 128, KD = D / 16;
  extern __shared__ __align__(128) uint8_t smem_raw[];
  Elt* sQ = reinterpret_cast<Elt*>(smem_raw);
  Elt* sK0 = sQ + BM * D;
  float* sBias = reinterpret_cast<float*>(sK0 + 2 * BN * D);  // [2][64]

  const int qt = blockIdx.x;
  const int bh_local = blockIdx.y;
  const int bh = p.bh0 + bh_local;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t = lane & 3;
  const Elt* gQ = reinterpret_cast<const Elt*>(p.qr) + ((size_t)bh * p.Lr_pad + (size_t)qt * BM) * D;
  const Elt* gK = reinterpret_cast<const Elt*>(p.kr) + (size_t)bh * p.Lr_pad * D;
  const int nkt = p.Lr_pad / BN;

  auto issue_k = [&](int kt) {
    load_tile_async<BN, D, NT>(sK0 + (kt & 1) * BN * D, gK + (size_t)kt * BN * D, tid);
    if (tid < BN) {
      int c = kt * BN + tid;
      bool ok = c < p.Lr && (c % G) * p.R < p.blk_cnt[c / G];
      sBias[(kt & 1) * BN + tid] = ok ? 0.f : neg_inf();
    }
  };

  load_tile_async<BM, D, NT>(sQ, gQ, tid);
  issue_k(0);
  cp_async_commit();

  // this thread's two query rows
  bool rvalid[2];
  float rl2[2];
  float rnorm[2];
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    int rg = qt * BM + warp * 16 + g + 8 * r;
    rvalid[r] = rg < p.Lr && (rg % G) * p.R < p.blk_cnt[rg / G];
    rl2[r] = 0.f;
    rnorm[r] = 0.f;
    if (SCORES && rvalid[r]) {
      rl2[r] = p.lse2[(size_t)bh * p.Lr_pad + rg];
      int cnt = p.blk_cnt[rg / G];
      int ngrp = (cnt + p.R - 1) / p.R;
      rnorm[r] = p.row_norm == 0 ? 1.f / (float)ngrp : 1.f / (float)(G * G);
    }
  }
  float m_r[2] = {neg_inf(), neg_inf()};
  float l_r[2] = {0.f, 0.f};
  uint32_t qf[KD][4];
  float* out = SCORES ? p.scores + (size_t)bh_local * p.NB * p.NB : nullptr;

  for (int kt = 0; kt < nkt; ++kt) {
    cp_async_wait<0>();
    __syncthreads();
    if (kt == 0) {
#pragma unroll
      for (int kk = 0; kk < KD; ++kk)
        ldsm_x4(qf[kk][0], qf[kk][1], qf[kk][2], qf[kk][3],
                smem_u32(sQ + swz<D>(warp * 16 + (lane & 15), kk * 16 + (lane >> 4) * 8)));
    }
    if (kt + 1 < nkt) {
      issue_k(kt + 1);
      cp_async_commit();
    }
    const Elt* sK = sK0 + (kt & 1) * BN * D;
    const float* bias = sBias + (kt & 1) * BN;

    float s[8][4];
#pragma unroll
    for (int i = 0; i < 8; ++i) s[i][0] = s[i][1] = s[i][2] = s[i][3] = 0.f;
#pragma unroll
    for (int kk = 0; kk < KD; ++kk) {
#pragma unroll
      for (int np = 0; np < 4; ++np) {
        uint32_t b0, b1, b2, b3;
        ldsm_x4(b0, b1, b2, b3,
                smem_u32(sK + swz<D>(np * 16 + (lane & 7) + ((lane >> 4) << 3), kk * 16 + ((lane >> 3) & 1) * 8)));
        T::mma(s[2 * np], qf[kk], b0, b1);
        T::mma(s[2 * np + 1], qf[kk], b2, b3);
      }
    }
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
#pragma unroll
      for (int e = 0; e < 4; ++e) s[nt][e] = s[nt][e] * p.scale_log2 + bias[nt * 8 + 2 * t + (e & 1)];
    }

    if constexpr (!SCORES) {
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        float mx = neg_inf();
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) mx = fmaxf(mx, fmaxf(s[nt][2 * r], s[nt][2 * r + 1]));
        mx = warp_quad_max(mx);
        float m_new = fmaxf(m_r[r], mx);
        float m_use = m_new == neg_inf() ? 0.f : m_new;
        float acc = 0.f;
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) acc += fast_exp2(s[nt][2 * r] - m_use) + fast_exp2(s[nt][2 * r + 1] - m_use);
        l_r[r] = l_r[r] * fast_exp2(m_r[r] - m_use) + acc;
        m_r[r] = m_new;
      }
    } else {
      // probabilities, then column-pair sums cs[nt][rh] for cols (8nt+2t, +1)
      float cs[8][2];
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
#pragma unroll
        for (int rh = 0; rh < 2; ++rh) {
          float a = rvalid[rh] ? fast_exp2(s[nt][2 * rh] - rl2[rh]) : 0.f;
          float b = rvalid[rh] ? fast_exp2(s[nt][2 * rh + 1] - rl2[rh]) : 0.f;
          cs[nt][rh] = a + b;
        }
      }
      // key-group (column) reduction
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
#pragma unroll
        for (int rh = 0; rh < 2; ++rh) {
          if constexpr (G >= 4) cs[nt][rh] += __shfl_xor_sync(0xffffffff, cs[nt][rh], 1);
          if constexpr (G >= 8) cs[nt][rh] += __shfl_xor_sync(0xffffffff, cs[nt][rh], 2);
        }
      }
      if constexpr (G >= 16) {
#pragma unroll
        for (int m = 0; m < 4; ++m) {
          cs[2 * m][0] += cs[2 * m + 1][0];
          cs[2 * m][1] += cs[2 * m + 1][1];
        }
      }
      // query-group (row) reduction
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
#pragma unroll
        for (int rh = 0; rh < 2; ++rh) {
          cs[nt][rh] += __shfl_xor_sync(0xffffffff, cs[nt][rh], 4);
          if constexpr (G >= 4) cs[nt][rh] += __shfl_xor_sync(0xffffffff, cs[nt][rh], 8);
          if constexpr (G >= 8) cs[nt][rh] += __shfl_xor_sync(0xffffffff, cs[nt][rh], 16);
        }
        if constexpr (G >= 16) cs[nt][0] += cs[nt][1];
      }
      // group leaders write S[i, j]
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
#pragma unroll
        for (int rh = 0; rh < 2; ++rh) {
          int r = warp * 16 + g + 8 * rh;
          int c = nt * 8 + 2 * t;
          if ((r % G) == 0 && (c % G) == 0) {
            int qi = (qt * BM + r) / G;
            int kj = (kt * BN + c) / G;
            if (qi < p.NB && kj < p.NB) {
              // the norm is a property of the query block; leader row r is its first group
              float nrm = rnorm[rh];
              if (!rvalid[rh]) {
                int cnt = p.blk_cnt[qi];
                int ngrp = (cnt + p.R - 1) / p.R;
                nrm = p.row_norm == 0 ? 1.f / (float)ngrp : 1.f / (float)(G * G);
              }
              out[(size_t)qi * p.NB + kj] = cs[nt][rh] * nrm;
            }
          }
        }
      }
    }
  }

  if constexpr (!SCORES) {
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      float lsum = warp_quad_sum(l_r[r]);
      int rg = qt * BM + warp * 16 + g + 8 * r;
      if (t == 0 && rg < p.Lr_pad)
        p.lse2[(size_t)bh * p.Lr_pad + rg] = (rvalid[r] && lsum > 0.f) ? m_r[r] + __log2f(lsum) : neg_inf();
    }
  }
}

template <typename Elt, int D, int G, bool SCORES>
static void run_router_g(const RouterParams& p, cudaStream_t s) {
  constexpr int smem = 3 * 64 * D * sizeof(Elt) + 2 * 64 * sizeof(float);
  auto kern = router_kernel<Elt, D, G, SCORES>;
  cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  dim3 grid(p.Lr_pad / 64, SCORES ? p.bh_count : p.BH);
  kern<<<grid, 128, smem, s>>>(p);
}

template <typename Elt, int D, bool SCORES>
static void run_router(const RouterParams& p, cudaStream_t s) {
  switch (p.G) {
    case 2: run_router_g<Elt, D, 2, SCORES>(p, s); break;
    case 4: run_router_g<Elt, D, 4, SCORES>(p, s); break;
    case 8: run_router_g<Elt, D, 8, SCORES>(p, s); break;
    default: run_router_g<Elt, D, 16, SCORES>(p, s); break;
  }
}

template <bool SCORES>
static void dispatch_router(const RouterParams& p, int D, DType dt, cudaStream_t s) {
  if (dt == DType::kBF16) {
    if (D == 64) run_router<__nv_bfloat16, 64, SCORES>(p, s);
    else run_router<__nv_bfloat16, 128, SCORES>(p, s);
  } else {
    if (D == 64) run_router<__half, 64, SCORES>(p, s);
    else run_router<__half, 128, SCORES>(p, s);
  }
}

void launch_router_lse(const RouterParams& p, int D, DType dt, cudaStream_t s) {
  RouterParams q = p;
  q.bh0 = 0;
  dispatch_router<false>(q, D, dt, s);
}
void launch_router_scores(const RouterParams& p, int D, DType dt, cudaStream_t s) {
  dispatch_router<true>(p, D, dt, s);
}

}  // namespace vsa2
