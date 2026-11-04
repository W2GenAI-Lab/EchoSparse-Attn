// VSA2 generic block-sparse attention forward (CSR q->kv block lists).
//
// One CTA = one (b*h, 64-row query block). The CTA walks exactly the kv blocks
// listed for its query block; there is no dense LxL score materialization and no
// masking of a dense pass. Data movement is cp.async with a single Q/K/V buffer
// set in the FlashAttention-2 order (V_j load overlaps S_j = Q K_j^T, K_{j+1}
// load overlaps softmax and P V_j). MMA: mma.sync.m16n8k16 (bf16/fp16 in, fp32 acc).
//
// Variants (template MT = 16-row MMA tiles per warp):
//   MT=1: 4 warps x 16 rows, Q fragments held in registers (validated first version)
//   MT=2: 2 warps x 32 rows, Q fragments re-read from smem; each K/V ldmatrix fragment
//         feeds two MMAs (half the smem fragment traffic per MMA), mirroring the
//         32-rows-per-warp layout of FA4's tuned SM120 forward (128x64 tile, 4 warps).
// Selected at runtime by set_fwd_variant() / ECHOSPARSE_VSA_FWD_MT (default 1).
//
// Semantics:
//   * kv columns >= blk_cnt[kv] are masked (partial blocks / tails)
//   * query rows >= blk_cnt[q] produce o = 0, lse = -inf
//   * a query block with zero listed kv blocks produces o = 0, lse = -inf
#include <cstdlib>

#include "../common.cuh"
#include "../vsa2_kernels.h"

namespace vsa2 {

template <typename Elt, int D, int MT>
__global__ void __launch_bounds__(128 / MT) attn_fwd_kernel(const AttnFwdParams p) {
  using T = EltTraits<Elt>;
  constexpr int BM = kBlock, BN = kBlock, NT = 128 / MT, WR = 16 * MT;
  constexpr int KD = D / 16;
  constexpr int ND = D / 8;
  constexpr bool QREG = (MT == 1);
  extern __shared__ __align__(128) uint8_t smem_raw[];
  Elt* sQ = reinterpret_cast<Elt*>(smem_raw);
  Elt* sK = sQ + BM * D;
  Elt* sV = sK + BN * D;

  const int qb = blockIdx.x;
  const int bh = blockIdx.y;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t = lane & 3;
  const size_t tok_base = (size_t)bh * p.L;
  const Elt* gQ = reinterpret_cast<const Elt*>(p.q) + (tok_base + (size_t)qb * BM) * D;
  const Elt* gK = reinterpret_cast<const Elt*>(p.k) + tok_base * D;
  const Elt* gV = reinterpret_cast<const Elt*>(p.v) + tok_base * D;
  Elt* gO = reinterpret_cast<Elt*>(p.o) + (tok_base + (size_t)qb * BM) * D;
  float* gL = p.lse + tok_base + (size_t)qb * BM;

  const int* rp = p.row_ptr + (size_t)bh * (p.NB + 1) + qb;
  const int beg = rp[0];
  const int nkv = rp[1] - beg;
  const int* cidx = p.col_idx + beg;
  const int qcnt = p.blk_cnt[qb];

  if (nkv <= 0) {
    constexpr int CH = BM * D / 8;
    for (int i = tid; i < CH; i += NT) reinterpret_cast<uint4*>(gO)[i] = make_uint4(0, 0, 0, 0);
    for (int i = tid; i < BM; i += NT) gL[i] = neg_inf();
    return;
  }

  int kvb = cidx[0];
  load_tile_async<BM, D, NT>(sQ, gQ, tid);
  load_tile_async<BN, D, NT>(sK, gK + (size_t)kvb * BN * D, tid);
  cp_async_commit();

  float o_acc[MT][ND][4];
#pragma unroll
  for (int m = 0; m < MT; ++m)
#pragma unroll
    for (int i = 0; i < ND; ++i) o_acc[m][i][0] = o_acc[m][i][1] = o_acc[m][i][2] = o_acc[m][i][3] = 0.f;
  float m_r[MT][2], l_r[MT][2];
#pragma unroll
  for (int m = 0; m < MT; ++m) m_r[m][0] = m_r[m][1] = neg_inf(), l_r[m][0] = l_r[m][1] = 0.f;
  uint32_t qf[QREG ? KD : 1][4];
  const int r0 = warp * WR;

  for (int it = 0; it < nkv; ++it) {
    cp_async_wait<0>();
    __syncthreads();  // K_it (and Q) landed; every warp finished P V_{it-1}
    if constexpr (QREG) {
      if (it == 0) {
#pragma unroll
        for (int kk = 0; kk < KD; ++kk)
          ldsm_x4(qf[kk][0], qf[kk][1], qf[kk][2], qf[kk][3],
                  smem_u32(sQ + swz<D>(r0 + (lane & 15), kk * 16 + (lane >> 4) * 8)));
      }
    }
    load_tile_async<BN, D, NT>(sV, gV + (size_t)kvb * BN * D, tid);
    cp_async_commit();

    float s[MT][8][4];
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
      for (int i = 0; i < 8; ++i) s[m][i][0] = s[m][i][1] = s[m][i][2] = s[m][i][3] = 0.f;
#pragma unroll
    for (int kk = 0; kk < KD; ++kk) {
      uint32_t a[MT][4];
#pragma unroll
      for (int m = 0; m < MT; ++m) {
        if constexpr (QREG) {
          a[m][0] = qf[kk][0]; a[m][1] = qf[kk][1]; a[m][2] = qf[kk][2]; a[m][3] = qf[kk][3];
        } else {
          ldsm_x4(a[m][0], a[m][1], a[m][2], a[m][3],
                  smem_u32(sQ + swz<D>(r0 + 16 * m + (lane & 15), kk * 16 + (lane >> 4) * 8)));
        }
      }
#pragma unroll
      for (int np = 0; np < 4; ++np) {
        uint32_t b0, b1, b2, b3;
        ldsm_x4(b0, b1, b2, b3,
                smem_u32(sK + swz<D>(np * 16 + (lane & 7) + ((lane >> 4) << 3), kk * 16 + ((lane >> 3) & 1) * 8)));
#pragma unroll
        for (int m = 0; m < MT; ++m) {
          T::mma(s[m][2 * np], a[m], b0, b1);
          T::mma(s[m][2 * np + 1], a[m], b2, b3);
        }
      }
    }

    const int kcnt = p.blk_cnt[kvb];
#pragma unroll
    for (int m = 0; m < MT; ++m) {
      float mx[2] = {neg_inf(), neg_inf()};
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          int col = nt * 8 + 2 * t + (e & 1);
          float x = (col < kcnt) ? s[m][nt][e] * p.scale_log2 : neg_inf();
          s[m][nt][e] = x;
          mx[e >> 1] = fmaxf(mx[e >> 1], x);
        }
      }
      float mu[2];
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        float mrow = warp_quad_max(mx[r]);
        float m_new = fmaxf(m_r[m][r], mrow);
        float m_use = (m_new == neg_inf()) ? 0.f : m_new;
        float alpha = fast_exp2(m_r[m][r] - m_use);
        m_r[m][r] = m_new;
        mu[r] = m_use;
        l_r[m][r] *= alpha;
#pragma unroll
        for (int nd = 0; nd < ND; ++nd) {
          o_acc[m][nd][2 * r] *= alpha;
          o_acc[m][nd][2 * r + 1] *= alpha;
        }
      }
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          float pv = fast_exp2(s[m][nt][e] - mu[e >> 1]);
          s[m][nt][e] = pv;
          l_r[m][e >> 1] += pv;
        }
      }
    }

    cp_async_wait<0>();
    __syncthreads();  // V_it landed; every warp finished reading K_it
    int kvb_next = kvb;
    if (it + 1 < nkv) {
      kvb_next = cidx[it + 1];
      load_tile_async<BN, D, NT>(sK, gK + (size_t)kvb_next * BN * D, tid);
      cp_async_commit();
    }

    // pack P up front, ending the fp32 S lifetime before PV
    uint32_t pf[4][MT][4];
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
#pragma unroll
      for (int m = 0; m < MT; ++m) {
        pf[kk][m][0] = T::pack2(s[m][2 * kk][0], s[m][2 * kk][1]);
        pf[kk][m][1] = T::pack2(s[m][2 * kk][2], s[m][2 * kk][3]);
        pf[kk][m][2] = T::pack2(s[m][2 * kk + 1][0], s[m][2 * kk + 1][1]);
        pf[kk][m][3] = T::pack2(s[m][2 * kk + 1][2], s[m][2 * kk + 1][3]);
      }
    }
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
#pragma unroll
      for (int dp = 0; dp < D / 16; ++dp) {
        uint32_t b0, b1, b2, b3;
        ldsm_x4_t(b0, b1, b2, b3, smem_u32(sV + swz<D>(kk * 16 + (lane & 15), dp * 16 + (lane >> 4) * 8)));
#pragma unroll
        for (int m = 0; m < MT; ++m) {
          T::mma(o_acc[m][2 * dp], pf[kk][m], b0, b1);
          T::mma(o_acc[m][2 * dp + 1], pf[kk][m], b2, b3);
        }
      }
    }
    kvb = kvb_next;
  }

  // Epilogue: normalize, stage through this warp's rows of sQ, coalesced store.
  // (sQ is only read by this warp's own rows when MT == 2; in MT == 1 it was last read at it == 0.)
  __syncwarp();
#pragma unroll
  for (int m = 0; m < MT; ++m) {
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      float lsum = warp_quad_sum(l_r[m][r]);
      int row = r0 + 16 * m + g + 8 * r;
      bool valid = row < qcnt && lsum > 0.f;
      float inv = valid ? 1.f / lsum : 0.f;
#pragma unroll
      for (int nd = 0; nd < ND; ++nd) {
        int col = nd * 8 + 2 * t;
        *reinterpret_cast<uint32_t*>(sQ + swz<D>(row, col)) =
            T::pack2(o_acc[m][nd][2 * r] * inv, o_acc[m][nd][2 * r + 1] * inv);
      }
      if (t == 0) gL[row] = valid ? (m_r[m][r] + __log2f(lsum)) * kLn2 : neg_inf();
    }
  }
  __syncwarp();
  constexpr int CPR = D / 8;
#pragma unroll
  for (int i = lane; i < WR * CPR; i += 32) {
    int row = r0 + i / CPR;
    int ch = i % CPR;
    *reinterpret_cast<uint4*>(gO + (size_t)row * D + ch * 8) =
        *reinterpret_cast<const uint4*>(sQ + swz<D>(row, ch * 8));
  }
}

namespace {
int g_fwd_mt = -1;
int fwd_mt() {
  if (g_fwd_mt < 0) {
    const char* e = std::getenv("ECHOSPARSE_VSA_FWD_MT");
    g_fwd_mt = (e && std::atoi(e) == 2) ? 2 : 1;
  }
  return g_fwd_mt;
}
}  // namespace

void set_fwd_variant(int mt) { g_fwd_mt = (mt == 2) ? 2 : 1; }
int get_fwd_variant() { return fwd_mt(); }

template <typename Elt, int D, int MT>
static void run_fwd_mt(const AttnFwdParams& p, cudaStream_t s) {
  constexpr int smem = 3 * kBlock * D * sizeof(Elt);
  auto kern = attn_fwd_kernel<Elt, D, MT>;
  cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  dim3 grid(p.NB, p.BH);
  kern<<<grid, 128 / MT, smem, s>>>(p);
}

template <typename Elt, int D>
static void run_fwd(const AttnFwdParams& p, cudaStream_t s) {
  if (fwd_mt() == 2) run_fwd_mt<Elt, D, 2>(p, s);
  else run_fwd_mt<Elt, D, 1>(p, s);
}

void launch_attn_fwd(const AttnFwdParams& p, int D, DType dt, cudaStream_t s) {
  if (dt == DType::kBF16) {
    if (D == 64) run_fwd<__nv_bfloat16, 64>(p, s);
    else run_fwd<__nv_bfloat16, 128>(p, s);
  } else {
    if (D == 64) run_fwd<__half, 64>(p, s);
    else run_fwd<__half, 128>(p, s);
  }
}

}  // namespace vsa2
