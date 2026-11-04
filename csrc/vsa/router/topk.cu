// VSA2 exact global block selection.
//
// For every (b, h) independently, select exactly k_eff = min(budget, NB*NB)
// entries of the NB x NB block-score matrix S (budget = NB * topk: every query
// block receives topk connections on average, no per-row minimum or cap).
//
// Order: larger score first; ties broken by smaller flat index (i*NB + j), i.e.
// a total order, so the selection is unique and deterministic.
//
// Algorithm (no sort, no LxL token matrix):
//   1. radix select on the order-preserving uint32 key of the float score
//      (3 histogram passes: 11/11/10 bits) -> threshold key T and k_rem, the
//      number of entries equal to T that must be taken.
//   2. one warp per row: ballot -> bitmasks gt (key > T) and eq (key == T),
//      per-row popcounts.
//   3. per (b,h): scan of eq counts locates the k_rem-th tied entry in flat
//      order -> (cut_row, cut_col). Selected = gt | (eq & flat <= cut).
//      Row counts and the q->kv CSR row_ptr come from the same scan.
//   4. emit q->kv column lists (ascending) and the selection bitmask.
//   5. column counts from the bitmask, scan -> kv->q row_ptr, emit kv->q row
//      lists (ascending). Backward uses exactly this inverted selection.
#include <algorithm>

#include "../common.cuh"
#include "../vsa2_kernels.h"

namespace vsa2 {

namespace {

constexpr int kScanThreads = 1024;
enum { ST_PREFIX = 0, ST_KREM = 1, ST_CUTROW = 2, ST_CUTCOL = 3, ST_KEFF = 4, ST_STRIDE = 8 };

__device__ __forceinline__ uint32_t float_key(float f) {
  uint32_t u = __float_as_uint(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__device__ __forceinline__ int words_of(int nb) { return (nb + 31) >> 5; }

// Block-wide exclusive scan over kScanThreads ints. Returns exclusive prefix;
// *total receives the sum. smem must hold kScanThreads/32 ints.
__device__ int block_excl_scan(int v, int* smem, int* total) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  int x = v;
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    int y = __shfl_up_sync(0xffffffff, x, off);
    if (lane >= off) x += y;
  }
  if (lane == 31) smem[warp] = x;
  __syncthreads();
  if (warp == 0) {
    int w = lane < (kScanThreads / 32) ? smem[lane] : 0;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      int y = __shfl_up_sync(0xffffffff, w, off);
      if (lane >= off) w += y;
    }
    if (lane < (kScanThreads / 32)) smem[lane] = w;  // inclusive warp totals
  }
  __syncthreads();
  int warp_excl = warp > 0 ? smem[warp - 1] : 0;
  *total = smem[kScanThreads / 32 - 1];
  __syncthreads();
  return warp_excl + x - v;
}

__global__ void topk_init_kernel(int* state, int bh_count, long long nn, long long budget) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= bh_count) return;
  long long k = budget < nn ? budget : nn;
  if (k < 0) k = 0;
  int* st = state + i * ST_STRIDE;
  st[ST_PREFIX] = 0;
  st[ST_KREM] = (int)k;
  st[ST_CUTROW] = -1;
  st[ST_CUTCOL] = -1;
  st[ST_KEFF] = (int)k;
}

template <int PASS>
__global__ void __launch_bounds__(256) topk_hist_kernel(const float* scores, long long nn, const int* state,
                                                        uint32_t* hist) {
  constexpr int SHIFT = PASS == 0 ? 21 : (PASS == 1 ? 10 : 0);
  constexpr int NBINS = PASS == 2 ? 1024 : 2048;
  __shared__ uint32_t h[2048];
  const int bh = blockIdx.y;
  const int* st = state + bh * ST_STRIDE;
  if (st[ST_KREM] <= 0) return;
  for (int i = threadIdx.x; i < NBINS; i += blockDim.x) h[i] = 0;
  __syncthreads();
  const uint32_t prefix = (uint32_t)st[ST_PREFIX];
  const float* seg = scores + (size_t)bh * nn;
  // Each CTA owns a contiguous chunk, striped across its threads: neighbouring lanes read
  // neighbouring scores in every load, and the CTA walks memory sequentially.
  const long long per = (nn + gridDim.x - 1) / gridDim.x;
  const long long i1 = ((long long)blockIdx.x + 1) * per < nn ? ((long long)blockIdx.x + 1) * per : nn;
  for (long long i = (long long)blockIdx.x * per + threadIdx.x; i < i1; i += blockDim.x) {
    uint32_t key = float_key(seg[i]);
    bool match = PASS == 0 ? true : ((key >> (SHIFT + (PASS == 1 ? 11 : 10))) == (prefix >> (SHIFT + (PASS == 1 ? 11 : 10))));
    if (match) atomicAdd(&h[(key >> SHIFT) & (NBINS - 1)], 1u);
  }
  __syncthreads();
  uint32_t* gh = hist + ((size_t)PASS * gridDim.y + bh) * 2048;
  for (int i = threadIdx.x; i < NBINS; i += blockDim.x)
    if (h[i]) atomicAdd(&gh[i], h[i]);
}

template <int PASS>
__global__ void topk_find_kernel(const uint32_t* hist, int* state, int bh_count) {
  constexpr int SHIFT = PASS == 0 ? 21 : (PASS == 1 ? 10 : 0);
  constexpr int NBINS = PASS == 2 ? 1024 : 2048;
  constexpr int PER = NBINS / 32;
  const int bh = blockIdx.x;
  int* st = state + bh * ST_STRIDE;
  const int lane = threadIdx.x;
  const int krem = st[ST_KREM];
  if (krem <= 0) return;
  const uint32_t* h = hist + ((size_t)PASS * bh_count + bh) * 2048;
  long long sum = 0;
  for (int b = 0; b < PER; ++b) sum += h[lane * PER + b];
  // inclusive suffix sum over lanes
  long long suf = sum;
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    long long y = __shfl_down_sync(0xffffffff, suf, off);
    if (lane + off < 32) suf += y;
  }
  unsigned ball = __ballot_sync(0xffffffff, suf >= krem);
  int L = 31 - __clz(ball);  // highest lane whose suffix still reaches krem
  if (lane == L) {
    long long cum = suf - sum;  // entries in bins above this lane's chunk
    for (int b = PER - 1; b >= 0; --b) {
      long long c = h[lane * PER + b];
      if (cum + c >= krem) {
        uint32_t digit = (uint32_t)(lane * PER + b);
        st[ST_PREFIX] = (int)((uint32_t)st[ST_PREFIX] | (digit << SHIFT));
        st[ST_KREM] = (int)(krem - cum);
        break;
      }
      cum += c;
    }
  }
}

// one warp per row: gt / eq bitmasks and counts
__global__ void __launch_bounds__(256) topk_mask_kernel(const float* scores, int NB, const int* state,
                                                        uint32_t* gtbits, uint32_t* eqbits, int* gtcnt, int* eqcnt) {
  const int bh = blockIdx.y;
  const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (row >= NB) return;
  const int* st = state + bh * ST_STRIDE;
  const bool none = st[ST_KEFF] <= 0;
  const uint32_t T = (uint32_t)st[ST_PREFIX];
  const int W = words_of(NB);
  const float* src = scores + ((size_t)bh * NB + row) * NB;
  uint32_t* gw = gtbits + ((size_t)bh * NB + row) * W;
  uint32_t* ew = eqbits + ((size_t)bh * NB + row) * W;
  int ng = 0, ne = 0;
  for (int w = 0; w < W; ++w) {
    int col = w * 32 + lane;
    bool gt = false, eq = false;
    if (col < NB && !none) {
      uint32_t key = float_key(src[col]);
      gt = key > T;
      eq = key == T;
    }
    uint32_t bg = __ballot_sync(0xffffffff, gt);
    uint32_t be = __ballot_sync(0xffffffff, eq);
    if (lane == 0) {
      gw[w] = bg;
      ew[w] = be;
    }
    ng += __popc(bg);
    ne += __popc(be);
  }
  if (lane == 0) {
    gtcnt[(size_t)bh * NB + row] = ng;
    eqcnt[(size_t)bh * NB + row] = ne;
  }
}

// per (b,h): locate the tie cutoff, row counts, q->kv row_ptr
__global__ void __launch_bounds__(kScanThreads) topk_cut_kernel(int NB, int bh0, int* state, const uint32_t* eqbits,
                                                                const int* gtcnt, const int* eqcnt, int* q2k_ptr) {
  __shared__ int sscan[kScanThreads / 32];
  __shared__ int s_cut[3];
  const int bh = blockIdx.x;
  int* st = state + bh * ST_STRIDE;
  const int keff = st[ST_KEFF];
  const int krem = st[ST_KREM];
  const int W = words_of(NB);
  const int per = (NB + kScanThreads - 1) / kScanThreads;
  const int r0 = min(NB, (int)threadIdx.x * per), r1 = min(NB, r0 + per);
  const int* ec = eqcnt + (size_t)bh * NB;
  const int* gc = gtcnt + (size_t)bh * NB;
  if (threadIdx.x == 0) {
    s_cut[0] = -1;
    s_cut[1] = -1;
    s_cut[2] = 0;
  }
  int esum = 0;
  for (int i = r0; i < r1; ++i) esum += ec[i];
  int etot;
  int eex = block_excl_scan(esum, sscan, &etot);
  if (keff > 0 && krem > 0 && eex < krem && krem <= eex + esum) {
    int cum = eex;
    for (int i = r0; i < r1; ++i) {
      if (cum + ec[i] >= krem) {
        int need = krem - cum;  // 1-based rank of the cutoff tie within row i
        const uint32_t* ew = eqbits + ((size_t)bh * NB + i) * W;
        for (int w = 0; w < W; ++w) {
          uint32_t word = ew[w];
          int c = __popc(word);
          if (need <= c) {
            for (int j = 1; j < need; ++j) word &= word - 1;
            s_cut[0] = i;
            s_cut[1] = w * 32 + (__ffs(word) - 1);
            s_cut[2] = krem - cum;
            break;
          }
          need -= c;
        }
        break;
      }
      cum += ec[i];
    }
  }
  __syncthreads();
  const int cut_row = s_cut[0];
  const int cut_need = s_cut[2];
  int rsum = 0;
  for (int i = r0; i < r1; ++i) rsum += gc[i] + (i < cut_row ? ec[i] : (i == cut_row ? cut_need : 0));
  int rtot;
  int rex = block_excl_scan(rsum, sscan, &rtot);
  const long long base = (long long)(bh0 + bh) * keff;
  int* ptr = q2k_ptr + (size_t)(bh0 + bh) * (NB + 1);
  int run = rex;
  for (int i = r0; i < r1; ++i) {
    ptr[i] = (int)(base + run);
    run += gc[i] + (i < cut_row ? ec[i] : (i == cut_row ? cut_need : 0));
  }
  if (threadIdx.x == 0) {
    ptr[NB] = (int)(base + rtot);
    st[ST_CUTROW] = cut_row;
    st[ST_CUTCOL] = s_cut[1];
  }
}

__device__ __forceinline__ uint32_t cut_mask(int row, int w, int cut_row, int cut_col) {
  if (row < cut_row) return 0xffffffffu;
  if (row > cut_row) return 0u;
  int lo = w * 32;
  if (cut_col < lo) return 0u;
  if (cut_col >= lo + 31) return 0xffffffffu;
  return (2u << (cut_col - lo)) - 1u;
}

// one warp per row: final selection word, ascending q->kv lists
__global__ void __launch_bounds__(256) topk_emit_rows_kernel(int NB, int bh0, const int* state, const uint32_t* gtbits,
                                                             const uint32_t* eqbits, uint32_t* selbits,
                                                             const int* q2k_ptr, int* q2k_idx) {
  const int bh = blockIdx.y;
  const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (row >= NB) return;
  const int* st = state + bh * ST_STRIDE;
  const int cut_row = st[ST_CUTROW], cut_col = st[ST_CUTCOL];
  const int W = words_of(NB);
  const size_t roff = ((size_t)bh * NB + row) * W;
  int* out = q2k_idx + q2k_ptr[(size_t)(bh0 + bh) * (NB + 1) + row];
  int run = 0;
  for (int wb = 0; wb < W; wb += 32) {
    int w = wb + lane;
    uint32_t word = 0;
    if (w < W) {
      word = gtbits[roff + w] | (eqbits[roff + w] & cut_mask(row, w, cut_row, cut_col));
      selbits[roff + w] = word;
    }
    int c = __popc(word);
    int x = c;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      int y = __shfl_up_sync(0xffffffff, x, off);
      if (lane >= off) x += y;
    }
    int pos = run + x - c;
    while (word) {
      int b = __ffs(word) - 1;
      out[pos++] = w * 32 + b;
      word &= word - 1;
    }
    run += __shfl_sync(0xffffffff, x, 31);
  }
}

// one warp per selection-bitmask word column; lane = bit (column j = 32*wc + lane)
template <bool EMIT>
__global__ void __launch_bounds__(256) topk_cols_kernel(int NB, int bh0, const uint32_t* selbits, int* colcnt,
                                                        const int* k2q_ptr, int* k2q_idx) {
  const int bh = blockIdx.y;
  const int wc = blockIdx.x * 8 + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  const int W = words_of(NB);
  if (wc >= W) return;
  const int j = wc * 32 + lane;
  const uint32_t* col = selbits + (size_t)bh * NB * W + wc;
  int cnt = 0;
  int* out = nullptr;
  if (EMIT && j < NB) out = k2q_idx + k2q_ptr[(size_t)(bh0 + bh) * (NB + 1) + j];
  int i = 0;
  for (; i + 4 <= NB; i += 4) {
    uint32_t w0 = col[(size_t)(i + 0) * W], w1 = col[(size_t)(i + 1) * W];
    uint32_t w2 = col[(size_t)(i + 2) * W], w3 = col[(size_t)(i + 3) * W];
    uint32_t ws[4] = {w0, w1, w2, w3};
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      if ((ws[u] >> lane) & 1u) {
        if (EMIT && out) out[cnt] = i + u;
        ++cnt;
      }
    }
  }
  for (; i < NB; ++i) {
    if ((col[(size_t)i * W] >> lane) & 1u) {
      if (EMIT && out) out[cnt] = i;
      ++cnt;
    }
  }
  if (!EMIT && j < NB) colcnt[(size_t)bh * NB + j] = cnt;
}

__global__ void __launch_bounds__(kScanThreads) topk_colscan_kernel(int NB, int bh0, const int* state,
                                                                    const int* colcnt, int* k2q_ptr) {
  __shared__ int sscan[kScanThreads / 32];
  const int bh = blockIdx.x;
  const int keff = state[bh * ST_STRIDE + ST_KEFF];
  const int per = (NB + kScanThreads - 1) / kScanThreads;
  const int r0 = min(NB, (int)threadIdx.x * per), r1 = min(NB, r0 + per);
  const int* cc = colcnt + (size_t)bh * NB;
  int sum = 0;
  for (int i = r0; i < r1; ++i) sum += cc[i];
  int tot;
  int ex = block_excl_scan(sum, sscan, &tot);
  const long long base = (long long)(bh0 + bh) * keff;
  int* ptr = k2q_ptr + (size_t)(bh0 + bh) * (NB + 1);
  int run = ex;
  for (int i = r0; i < r1; ++i) {
    ptr[i] = (int)(base + run);
    run += cc[i];
  }
  if (threadIdx.x == 0) ptr[NB] = (int)(base + tot);
}

}  // namespace

size_t topk_workspace_bytes(int bh_count, int NB) {
  size_t W = (size_t)(NB + 31) / 32;
  size_t b = 0;
  b += (size_t)3 * bh_count * 2048 * sizeof(uint32_t);  // hist
  b += (size_t)bh_count * ST_STRIDE * sizeof(int);       // state
  b += (size_t)3 * bh_count * NB * W * sizeof(uint32_t); // gt/eq/sel bits
  b += (size_t)3 * bh_count * NB * sizeof(int);          // gtcnt/eqcnt/colcnt
  return b + 256;
}

void topk_carve_workspace(void* base, int bh_count, int NB, TopkWorkspace* ws) {
  size_t W = (size_t)(NB + 31) / 32;
  char* p = reinterpret_cast<char*>(base);
  ws->hist = reinterpret_cast<uint32_t*>(p);
  p += (size_t)3 * bh_count * 2048 * sizeof(uint32_t);
  ws->state = reinterpret_cast<int*>(p);
  p += (size_t)bh_count * ST_STRIDE * sizeof(int);
  ws->gtbits = reinterpret_cast<uint32_t*>(p);
  p += (size_t)bh_count * NB * W * sizeof(uint32_t);
  ws->eqbits = reinterpret_cast<uint32_t*>(p);
  p += (size_t)bh_count * NB * W * sizeof(uint32_t);
  ws->selbits = reinterpret_cast<uint32_t*>(p);
  p += (size_t)bh_count * NB * W * sizeof(uint32_t);
  ws->gtcnt = reinterpret_cast<int*>(p);
  p += (size_t)bh_count * NB * sizeof(int);
  ws->eqcnt = reinterpret_cast<int*>(p);
  p += (size_t)bh_count * NB * sizeof(int);
  ws->colcnt = reinterpret_cast<int*>(p);
}

void launch_global_topk(const TopkParams& p, const TopkWorkspace& ws, cudaStream_t s) {
  const int NB = p.NB, bhc = p.bh_count;
  const long long nn = (long long)NB * NB;
  const int W = (NB + 31) / 32;
  cudaMemsetAsync(ws.hist, 0, (size_t)3 * bhc * 2048 * sizeof(uint32_t), s);
  topk_init_kernel<<<(bhc + 127) / 128, 128, 0, s>>>(ws.state, bhc, nn, p.budget);

  int chunks = (int)std::min<long long>((nn + 256 * 16 - 1) / (256 * 16), 1024);
  chunks = std::max(chunks, 1);
  dim3 hgrid(chunks, bhc);
  topk_hist_kernel<0><<<hgrid, 256, 0, s>>>(p.scores, nn, ws.state, ws.hist);
  topk_find_kernel<0><<<bhc, 32, 0, s>>>(ws.hist, ws.state, bhc);
  topk_hist_kernel<1><<<hgrid, 256, 0, s>>>(p.scores, nn, ws.state, ws.hist);
  topk_find_kernel<1><<<bhc, 32, 0, s>>>(ws.hist, ws.state, bhc);
  topk_hist_kernel<2><<<hgrid, 256, 0, s>>>(p.scores, nn, ws.state, ws.hist);
  topk_find_kernel<2><<<bhc, 32, 0, s>>>(ws.hist, ws.state, bhc);

  dim3 rgrid((NB + 7) / 8, bhc);
  topk_mask_kernel<<<rgrid, 256, 0, s>>>(p.scores, NB, ws.state, ws.gtbits, ws.eqbits, ws.gtcnt, ws.eqcnt);
  topk_cut_kernel<<<bhc, kScanThreads, 0, s>>>(NB, p.bh0, ws.state, ws.eqbits, ws.gtcnt, ws.eqcnt, p.q2k_ptr);
  topk_emit_rows_kernel<<<rgrid, 256, 0, s>>>(NB, p.bh0, ws.state, ws.gtbits, ws.eqbits, ws.selbits, p.q2k_ptr,
                                              p.q2k_idx);
  dim3 cgrid((W + 7) / 8, bhc);
  topk_cols_kernel<false><<<cgrid, 256, 0, s>>>(NB, p.bh0, ws.selbits, ws.colcnt, nullptr, nullptr);
  topk_colscan_kernel<<<bhc, kScanThreads, 0, s>>>(NB, p.bh0, ws.state, ws.colcnt, p.k2q_ptr);
  topk_cols_kernel<true><<<cgrid, 256, 0, s>>>(NB, p.bh0, ws.selbits, nullptr, p.k2q_ptr, p.k2q_idx);
}

// ------------------------------------------------------------------ original VSA: per-row top-K
//
// One CTA per (b,h, query block). Two 8-bit radix passes over the order-preserving 16-bit keys give the
// K-th largest key T and k_rem, the number of entries equal to T to take. One ascending pass then selects
// key > T plus the first k_rem entries == T (ties -> smaller kv index), emitting the ascending q->kv list
// and the selection bitmask. The kv->q lists come from the column kernels of the global top-k.
namespace {

__device__ __forceinline__ uint32_t half_key(uint32_t u) {
  if ((u & 0x7fffu) == 0) u = 0;  // -0 ties with +0, as in a float comparison
  return (u & 0x8000u) ? (~u & 0xffffu) : (u | 0x8000u);
}

// warp 0: the bin of h[256] holding the krem-th largest entry -> s_out = {bin, rank inside the bin}
__device__ void row_find_digit(const uint32_t* h, int krem, int* s_out) {
  const int lane = threadIdx.x;
  int sum = 0;
  for (int b = 0; b < 8; ++b) sum += h[lane * 8 + b];
  int suf = sum;  // inclusive suffix sum over lanes
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    int y = __shfl_down_sync(0xffffffff, suf, off);
    if (lane + off < 32) suf += y;
  }
  unsigned ball = __ballot_sync(0xffffffff, suf >= krem);
  int L = 31 - __clz(ball);
  if (lane == L) {
    int cum = suf - sum;
    for (int b = 7; b >= 0; --b) {
      int c = h[lane * 8 + b];
      if (cum + c >= krem) {
        s_out[0] = lane * 8 + b;
        s_out[1] = krem - cum;
        break;
      }
      cum += c;
    }
  }
}

__global__ void __launch_bounds__(kScanThreads) row_topk_kernel(const RowTopkParams p, uint32_t* selbits) {
  __shared__ uint32_t hist[256];
  __shared__ int sscan[kScanThreads / 32];
  __shared__ int s_dig[2];
  const int row = blockIdx.x, bh = blockIdx.y, NB = p.NB, K = p.K, W = words_of(NB);
  const uint16_t* src = p.scores + ((size_t)bh * NB + row) * NB;
  const long long base = ((long long)(p.bh0 + bh) * NB + row) * K;
  if (threadIdx.x == 0) {
    int* ptr = p.q2k_ptr + (size_t)(p.bh0 + bh) * (NB + 1);
    ptr[row] = (int)base;
    if (row == NB - 1) ptr[NB] = (int)(base + K);
  }
  uint32_t T = 0x10000u;  // above every key: K == 0 selects nothing
  int krem = 0;
  if (K > 0) {
    uint32_t prefix = 0;
    krem = K;
    for (int pass = 0; pass < 2; ++pass) {
      for (int i = threadIdx.x; i < 256; i += kScanThreads) hist[i] = 0;
      __syncthreads();
      for (int j = threadIdx.x; j < NB; j += kScanThreads) {
        uint32_t key = half_key(src[j]);
        if (pass == 0) atomicAdd(&hist[key >> 8], 1u);
        else if ((key >> 8) == prefix) atomicAdd(&hist[key & 255u], 1u);
      }
      __syncthreads();
      if (threadIdx.x < 32) row_find_digit(hist, krem, s_dig);
      __syncthreads();
      prefix = pass == 0 ? (uint32_t)s_dig[0] : ((prefix << 8) | (uint32_t)s_dig[0]);
      krem = s_dig[1];
      __syncthreads();  // hist and s_dig are reused
    }
    T = prefix;
  }
  int* out = p.q2k_idx + base;
  uint32_t* bits = selbits + ((size_t)bh * NB + row) * W;
  int eq_run = 0, sel_run = 0;
  for (int c0 = 0; c0 < NB; c0 += kScanThreads) {
    const int j = c0 + threadIdx.x;
    const bool in = j < NB;
    const uint32_t key = in ? half_key(src[j]) : 0u;
    const int eq = in && key == T;
    int tot;
    const int eq_ex = block_excl_scan(eq, sscan, &tot);
    const bool sel = in && (key > T || (eq && eq_run + eq_ex < krem));
    eq_run += tot;
    const int sel_ex = block_excl_scan(sel ? 1 : 0, sscan, &tot);
    if (sel) out[sel_run + sel_ex] = j;
    sel_run += tot;
    const uint32_t word = __ballot_sync(0xffffffff, sel);
    const int w = (c0 >> 5) + (threadIdx.x >> 5);
    if ((threadIdx.x & 31) == 0 && w < W) bits[w] = word;
  }
}

}  // namespace

void launch_row_topk(const RowTopkParams& p, const TopkWorkspace& ws, cudaStream_t s) {
  const int NB = p.NB, bhc = p.bh_count, W = (NB + 31) / 32;
  topk_init_kernel<<<(bhc + 127) / 128, 128, 0, s>>>(ws.state, bhc, (long long)NB * NB, (long long)NB * p.K);
  row_topk_kernel<<<dim3(NB, bhc), kScanThreads, 0, s>>>(p, ws.selbits);
  dim3 cgrid((W + 7) / 8, bhc);
  topk_cols_kernel<false><<<cgrid, 256, 0, s>>>(NB, p.bh0, ws.selbits, ws.colcnt, nullptr, nullptr);
  topk_colscan_kernel<<<bhc, kScanThreads, 0, s>>>(NB, p.bh0, ws.state, ws.colcnt, p.k2q_ptr);
  topk_cols_kernel<true><<<cgrid, 256, 0, s>>>(NB, p.bh0, ws.selbits, nullptr, p.k2q_ptr, p.k2q_idx);
}

}  // namespace vsa2
