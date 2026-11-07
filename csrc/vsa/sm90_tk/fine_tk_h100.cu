// VSA2 fine branch, H100 (sm_90a) ThunderKittens wgmma + TMA kernels.
//
// Adapted from FastVideo fastvideo-kernel/csrc/attention/block_sparse_h100.cu
// (https://github.com/hao-ai-lab/FastVideo @ 661d18ba55ddd2fde1802960cf4343a2ca84f57e,
// Apache License 2.0, Copyright the FastVideo authors), built against
// ThunderKittens @ 6c27e28c8115d1839d9eeeb530913c184a75fc87 (MIT).
//
// Changes relative to the original VSA kernels (marked "VSA2:"):
//   * fixed-width q2k/k2q index tables -> CSR lists (row_ptr/col_idx), which hold
//     the variable per-query-block counts produced by global top-k
//   * zero-edge query blocks (empty list): o = 0, lse = -inf, no TMA issued
//     (the original kernel unconditionally read index[0] / index[-1])
//   * lse is stored as the natural log-sum-exp of scale*q.k (the original stored
//     -lse*sqrt(D)); the backward converts on load
//   * the backward prep kernel is removed: delta (and the masked dO) come from
//     the VSA2 fusion-backward kernel, which also folds in the gate
//   * MHA only (hr = 1), batch handled through gridDim.z
// The math, tiling, pipelining and TMA usage are otherwise unchanged.

#include "kittens.cuh"
#include <cooperative_groups.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

using namespace kittens;

#if !defined(__CUDA_ARCH__) || defined(__CUDA_ARCH_FEAT_SM90_ALL)
#define VSA2_TK_HOPPER 1
#else
#define VSA2_TK_HOPPER 0
#endif

namespace vsa2_tk {

constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 64;

template <int D>
struct fwd_attend_ker_tile_dims {
  constexpr static int tile_width = D;
  constexpr static int qo_height = 4 * 16;
  constexpr static int kv_height = 4 * 16;
};

template <int D>
struct fwd_globals {
  using q_tile = st_bf<fwd_attend_ker_tile_dims<D>::qo_height, fwd_attend_ker_tile_dims<D>::tile_width>;
  using k_tile = st_bf<fwd_attend_ker_tile_dims<D>::kv_height, fwd_attend_ker_tile_dims<D>::tile_width>;
  using v_tile = st_bf<fwd_attend_ker_tile_dims<D>::kv_height, fwd_attend_ker_tile_dims<D>::tile_width>;
  using l_col_vec = col_vec<st_fl<fwd_attend_ker_tile_dims<D>::qo_height, fwd_attend_ker_tile_dims<D>::tile_width>>;
  using o_tile = st_bf<fwd_attend_ker_tile_dims<D>::qo_height, fwd_attend_ker_tile_dims<D>::tile_width>;

  using q_gl = gl<bf16, -1, -1, -1, -1, q_tile>;
  using k_gl = gl<bf16, -1, -1, -1, -1, k_tile>;
  using v_gl = gl<bf16, -1, -1, -1, -1, v_tile>;
  using l_gl = gl<float, -1, -1, -1, -1, l_col_vec>;
  using o_gl = gl<bf16, -1, -1, -1, -1, o_tile>;

  q_gl q;
  k_gl k;
  v_gl v;
  l_gl l;
  o_gl o;

  const int L;
  const int NB;
  const int* __restrict__ row_ptr;  // VSA2: CSR q->kv
  const int* __restrict__ col_idx;
  const int* __restrict__ block_size;
};

template <int D>
__global__ __launch_bounds__(128, 4) void fwd_attend_ker(const __grid_constant__ fwd_globals<D> g) {
#if VSA2_TK_HOPPER
  constexpr float SCALE_LOG2 = (D == 64) ? 1.44269504089f * 0.125f : 1.44269504089f * 0.08838834764f;
  const int bh = blockIdx.z * gridDim.y + blockIdx.y;
  const int seq_idx = blockIdx.x;

  // VSA2: CSR row and zero-edge handling
  const int* rp = g.row_ptr + (size_t)bh * (g.NB + 1) + seq_idx;
  const int beg = rp[0];
  const int kv_blocks = rp[1] - beg;
  const int* q2k_block_sparse_index_ptr = g.col_idx + beg;
  if (kv_blocks <= 0) {
    bf16* o = g.o.raw_ptr + ((size_t)bh * g.L + (size_t)seq_idx * BLOCK_M) * D;
    for (int i = threadIdx.x; i < BLOCK_M * D / 8; i += blockDim.x) reinterpret_cast<uint4*>(o)[i] = make_uint4(0, 0, 0, 0);
    if (threadIdx.x < BLOCK_M) g.l.raw_ptr[(size_t)bh * g.L + (size_t)seq_idx * BLOCK_M + threadIdx.x] = -INFINITY;
    return;
  }

  extern __shared__ int __shm[];
  tma_swizzle_allocator al((int*)&__shm[0]);

  using K = fwd_attend_ker_tile_dims<D>;
  using q_tile = st_bf<64, K::tile_width>;
  using k_tile = st_bf<64, K::tile_width>;
  using v_tile = st_bf<64, K::tile_width>;
  using l_col_vec = col_vec<st_fl<64, K::tile_width>>;
  using o_tile = st_bf<64, K::tile_width>;

  q_tile(&q_smem)[1] = al.allocate<q_tile, 1>();
  k_tile(&k_smem)[1] = al.allocate<k_tile, 1>();
  v_tile(&v_smem)[1] = al.allocate<v_tile, 1>();
  l_col_vec(&l_smem)[1] = al.allocate<l_col_vec, 1>();
  auto(*o_smem) = reinterpret_cast<o_tile(*)>(q_smem);

  const int kv_head_idx = blockIdx.y;  // MHA

  __shared__ kittens::semaphore qsmem_semaphore, k_smem_arrived, v_smem_arrived;
  if (threadIdx.x == 0) {
    int32_t kv_block_index = q2k_block_sparse_index_ptr[0];
    init_semaphore(qsmem_semaphore, 0, 1);
    init_semaphore(k_smem_arrived, 0, 1);
    init_semaphore(v_smem_arrived, 0, 1);

    coord<q_tile> q_tile_idx = {blockIdx.z, blockIdx.y, seq_idx, 0};
    tma::expect_bytes(qsmem_semaphore, sizeof(q_smem));
    tma::load_async(q_smem[0], g.q, q_tile_idx, qsmem_semaphore);

    tma::expect_bytes(k_smem_arrived, sizeof(k_tile));
    coord<k_tile> k_tile_idx = {blockIdx.z, kv_head_idx, kv_block_index, 0};
    tma::load_async(k_smem[0], g.k, k_tile_idx, k_smem_arrived);

    tma::expect_bytes(v_smem_arrived, sizeof(v_tile));
    coord<v_tile> v_tile_idx = {blockIdx.z, kv_head_idx, kv_block_index, 0};
    tma::load_async(v_smem[0], g.v, v_tile_idx, v_smem_arrived);
  }
  __syncthreads();

  rt_fl<16, 64> att_block;
  rt_bf<16, 64> att_block_mma;
  rt_fl<16, K::tile_width> o_reg;
  col_vec<rt_fl<16, 64>> max_vec, norm_vec, max_vec_last_scaled, max_vec_scaled;

  neg_infty(max_vec);
  zero(norm_vec);
  zero(o_reg);

  wait(qsmem_semaphore, 0);

  for (int kv_idx = 0; kv_idx < kv_blocks; kv_idx++) {
    const bool has_next = kv_idx + 1 < kv_blocks;
    int32_t kv_block_index = has_next ? q2k_block_sparse_index_ptr[kv_idx + 1] : 0;

    wait(k_smem_arrived, kv_idx % 2);
    warpgroup::mm_ABt(att_block, q_smem[0], k_smem[0]);

    copy(max_vec_last_scaled, max_vec);
    mul(max_vec_last_scaled, max_vec_last_scaled, SCALE_LOG2);
    warpgroup::mma_async_wait();

    if (has_next && threadIdx.x == 0) {
      tma::expect_bytes(k_smem_arrived, sizeof(k_tile));
      coord<k_tile> k_tile_idx = {blockIdx.z, kv_head_idx, kv_block_index, 0};
      tma::load_async(k_smem[0], g.k, k_tile_idx, k_smem_arrived);
    }

    right_fill(att_block, att_block, g.block_size[q2k_block_sparse_index_ptr[kv_idx]],
               base_types::constants<float>::neg_infty());
    row_max(max_vec, att_block, max_vec);
    mul(att_block, att_block, SCALE_LOG2);
    mul(max_vec_scaled, max_vec, SCALE_LOG2);

    sub_row(att_block, att_block, max_vec_scaled);
    exp2(att_block, att_block);
    sub(max_vec_last_scaled, max_vec_last_scaled, max_vec_scaled);
    exp2(max_vec_last_scaled, max_vec_last_scaled);
    mul(norm_vec, norm_vec, max_vec_last_scaled);
    row_sum(norm_vec, att_block, norm_vec);
    add(att_block, att_block, 0.f);
    copy(att_block_mma, att_block);
    mul_row(o_reg, o_reg, max_vec_last_scaled);

    wait(v_smem_arrived, kv_idx % 2);
    warpgroup::mma_AB(o_reg, att_block_mma, v_smem[0]);
    warpgroup::mma_async_wait();

    if (has_next && threadIdx.x == 0) {
      tma::expect_bytes(v_smem_arrived, sizeof(v_tile));
      coord<v_tile> v_tile_idx = {blockIdx.z, kv_head_idx, kv_block_index, 0};
      tma::load_async(v_smem[0], g.v, v_tile_idx, v_smem_arrived);
    }
  }

  div_row(o_reg, o_reg, norm_vec);
  warpgroup::store(o_smem[0], o_reg);
  __syncthreads();

  if (threadIdx.x / 32 == 0) {
    coord<o_tile> o_tile_idx = {blockIdx.z, blockIdx.y, seq_idx, 0};
    tma::store_async(g.o, o_smem[0], o_tile_idx);
  }

  // VSA2: natural lse = max*scale + ln(norm)
  mul(max_vec_scaled, max_vec_scaled, 0.69314718056f);
  log(norm_vec, norm_vec);
  add(norm_vec, norm_vec, max_vec_scaled);

  warpgroup::store(l_smem[0], norm_vec);
  __syncthreads();

  if (threadIdx.x / 32 == 0) {
    coord<l_col_vec> tile_idx = {blockIdx.z, blockIdx.y, 0, seq_idx};
    tma::store_async(g.l, l_smem[0], tile_idx);
  }
  tma::store_async_wait();
#endif
}

template <int D>
struct bwd_attend_ker_tile_dims {
  constexpr static int tile_width = D;
  constexpr static int tile_h = 4 * 16;
  constexpr static int tile_h_qo = 4 * 16;
};

template <int D>
struct bwd_globals {
  using G = bwd_attend_ker_tile_dims<D>;
  using q_tile = st_bf<G::tile_h_qo, G::tile_width>;
  using k_tile = st_bf<G::tile_h, G::tile_width>;
  using v_tile = st_bf<G::tile_h, G::tile_width>;
  using og_tile = st_bf<G::tile_h_qo, G::tile_width>;
  using qg_tile = st_fl<G::tile_h_qo, G::tile_width>;
  using kg_tile = st_fl<G::tile_h, G::tile_width>;
  using vg_tile = st_fl<G::tile_h, G::tile_width>;
  using l_tile = row_vec<st_fl<G::tile_h_qo, G::tile_h>>;
  using d_tile = row_vec<st_fl<G::tile_h_qo, G::tile_h>>;

  using q_gl = gl<bf16, -1, -1, -1, -1, q_tile>;
  using k_gl = gl<bf16, -1, -1, -1, -1, k_tile>;
  using v_gl = gl<bf16, -1, -1, -1, -1, v_tile>;
  using og_gl = gl<bf16, -1, -1, -1, -1, og_tile>;
  using qg_gl = gl<float, -1, -1, -1, -1, qg_tile>;
  using kg_gl = gl<float, -1, -1, -1, -1, kg_tile>;
  using vg_gl = gl<float, -1, -1, -1, -1, vg_tile>;
  using l_gl = gl<float, -1, -1, -1, -1, l_tile>;
  using d_gl = gl<float, -1, -1, -1, -1, d_tile>;

  q_gl q;
  k_gl k;
  v_gl v;
  og_gl og;
  qg_gl qg;
  kg_gl kg;
  vg_gl vg;
  l_gl l;
  d_gl d;

  const int NB;
  const int* __restrict__ row_ptr;  // VSA2: CSR kv->q
  const int* __restrict__ col_idx;
  const int* __restrict__ block_size;
  // Single-writer dK/dV store (tk_fine_bwd opts bit 16): the grid is (NB, H, B) and the kernel is MHA, so each
  // (b, h, kv block) has exactly one CTA. It stores its dK/dV tiles instead of reduce-adding them, and a CTA with
  // an empty k2q list stores zeros, so kg/vg need no zero fill. Same values, except that a -0 element stays -0
  // (the reduce-add computes 0 + -0 = +0).
  const int store_kv;
};

__device__ static inline void stream_tile(auto& reg_tile, auto& smem_vec, int tic) {
#pragma unroll
  for (int i = 0; i < 4; i++) {
    int base_col = 16 * i + 2 * (kittens::laneid() % 4);
    reg_tile.tiles[0][i].data[0] = *(float2*)&smem_vec[tic][base_col + 0];
    reg_tile.tiles[0][i].data[1] = *(float2*)&smem_vec[tic][base_col + 0];
    reg_tile.tiles[0][i].data[2] = *(float2*)&smem_vec[tic][base_col + 8];
    reg_tile.tiles[0][i].data[3] = *(float2*)&smem_vec[tic][base_col + 8];
  }
}

__device__ static inline void stream_sub_tile(auto& reg_tile, auto& smem_vec, int tic) {
#pragma unroll
  for (int i = 0; i < 4; i++) {
    int base_col = 16 * i + 2 * (laneid() % 4);
    reg_tile.tiles[0][i].data[0] = base_ops::sub::template op<float2>(reg_tile.tiles[0][i].data[0], *(float2*)&smem_vec[tic][base_col + 0]);
    reg_tile.tiles[0][i].data[1] = base_ops::sub::template op<float2>(reg_tile.tiles[0][i].data[1], *(float2*)&smem_vec[tic][base_col + 0]);
    reg_tile.tiles[0][i].data[2] = base_ops::sub::template op<float2>(reg_tile.tiles[0][i].data[2], *(float2*)&smem_vec[tic][base_col + 8]);
    reg_tile.tiles[0][i].data[3] = base_ops::sub::template op<float2>(reg_tile.tiles[0][i].data[3], *(float2*)&smem_vec[tic][base_col + 8]);
  }
}

template <int D>
__global__ __launch_bounds__(128, (D == 64) ? 3 : 2) void bwd_attend_ker(const __grid_constant__ bwd_globals<D> g) {
#if VSA2_TK_HOPPER
  constexpr float SCALE = (D == 64) ? 0.125f : 0.08838834764f;
  constexpr float SCALE_LOG2 = 1.44269504089f * SCALE;
  constexpr float NEG_INV_SCALE = (D == 64) ? -8.0f : -11.313708499f;  // VSA2: natural lse -> -lse*sqrt(D)

  const int bh = blockIdx.z * gridDim.y + blockIdx.y;
  const int* rp = g.row_ptr + (size_t)bh * (g.NB + 1) + blockIdx.x;
  const int beg = rp[0];
  const int qo_blocks = rp[1] - beg;
  const int* k2q_block_sparse_index_ptr = g.col_idx + beg;
  if (qo_blocks <= 0) {  // dK/dV accumulators stay zero
    if (g.store_kv) {      // they were not zero-filled: write this block's zeros
      const size_t off = ((size_t)bh * g.NB + blockIdx.x) * (size_t)(BLOCK_N * D);
      float4* kz = reinterpret_cast<float4*>(g.kg.raw_ptr + off);
      float4* vz = reinterpret_cast<float4*>(g.vg.raw_ptr + off);
      for (int i = threadIdx.x; i < BLOCK_N * D / 4; i += blockDim.x) {
        kz[i] = make_float4(0.f, 0.f, 0.f, 0.f);
        vz[i] = make_float4(0.f, 0.f, 0.f, 0.f);
      }
    }
    return;
  }

  extern __shared__ int __shm[];
  tma_swizzle_allocator al((int*)&__shm[0]);

  using G = bwd_attend_ker_tile_dims<D>;
  using kg_tile = st_fl<G::tile_h, G::tile_width>;
  using vg_tile = st_fl<G::tile_h, G::tile_width>;
  using k_tile = st_bf<G::tile_h, G::tile_width>;
  using v_tile = st_bf<G::tile_h, G::tile_width>;
  using q_tile = st_bf<G::tile_h_qo, G::tile_width>;
  using og_tile = st_bf<G::tile_h_qo, G::tile_width>;
  using qg_tile = st_fl<G::tile_h_qo, G::tile_width>;
  using l_tile = row_vec<st_fl<G::tile_h_qo, G::tile_h>>;
  using d_tile = row_vec<st_fl<G::tile_h_qo, G::tile_h>>;
  using attn_tile = st_bf<G::tile_h_qo, G::tile_h>;

  k_tile(&k_smem)[1] = al.allocate<k_tile, 1>();
  v_tile(&v_smem)[1] = al.allocate<v_tile, 1>();
  q_tile(&q_smem)[1] = al.allocate<q_tile, 1>();
  og_tile(&og_smem)[1] = al.allocate<og_tile, 1>();
  qg_tile(&qg_smem) = al.allocate<qg_tile>();
  l_tile(&l_smem)[1] = al.allocate<l_tile, 1>();
  d_tile(&d_smem)[1] = al.allocate<d_tile, 1>();
  kg_tile(*kg_smem) = reinterpret_cast<kg_tile*>(&k_smem[0].data[0]);
  vg_tile(*vg_smem) = reinterpret_cast<vg_tile*>(&q_smem[0].data[0]);
  attn_tile(&ds_smem_t)[1] = al.allocate<attn_tile, 1>();

  const int kv_head_idx = blockIdx.y;  // MHA

  __shared__ kittens::semaphore kv_b, q_b[1], o_b[1], vec_b[1];
  int32_t store_qg_block_index;
  int32_t load_q_block_index = k2q_block_sparse_index_ptr[0];

  if (threadIdx.x == 0) {
    init_semaphore(kv_b, 0, 1);
    init_semaphore(q_b[0], 0, 1);
    init_semaphore(o_b[0], 0, 1);
    init_semaphore(vec_b[0], 0, 1);

    tma::expect_bytes(kv_b, sizeof(k_smem[0]) + sizeof(v_smem[0]));
    coord<k_tile> tile_idx_kv = {blockIdx.z, kv_head_idx, blockIdx.x, 0};
    tma::load_async(k_smem[0], g.k, tile_idx_kv, kv_b);
    tma::load_async(v_smem[0], g.v, tile_idx_kv, kv_b);

    coord<q_tile> tile_idx_qo = {blockIdx.z, blockIdx.y, load_q_block_index, 0};
    coord<l_tile> vec_idx = {blockIdx.z, blockIdx.y, 0, load_q_block_index};

    tma::expect_bytes(o_b[0], sizeof(og_smem[0]));
    tma::load_async(og_smem[0], g.og, tile_idx_qo, o_b[0]);

    tma::expect_bytes(vec_b[0], sizeof(l_smem[0]) + sizeof(d_smem[0]));
    tma::load_async(l_smem[0], g.l, vec_idx, vec_b[0]);
    tma::load_async(d_smem[0], g.d, vec_idx, vec_b[0]);

    tma::expect_bytes(q_b[0], sizeof(q_smem[0]));
    tma::load_async(q_smem[0], g.q, tile_idx_qo, q_b[0]);
  }
  __syncthreads();

  rt_fl<16, G::tile_width> kg_reg, vg_reg;
  rt_fl<16, 64> s_block_t, p_block_t;
  rt_fl<16, 64> ds_block_t, dp_block_t;
  rt_bf<16, 64> ds_block_t_mma, p_block_t_mma;

  zero(kg_reg);
  zero(vg_reg);

  wait(kv_b, 0);
  int fill_start = g.block_size[blockIdx.x] - 16 * kittens::warpid();

  for (int qo_idx = 0; qo_idx < qo_blocks; qo_idx++) {
    const bool has_next = qo_idx + 1 < qo_blocks;
    store_qg_block_index = load_q_block_index;
    if (has_next) load_q_block_index = k2q_block_sparse_index_ptr[qo_idx + 1];

    wait(o_b[0], qo_idx % 2);
    warpgroup::mm_ABt(dp_block_t, v_smem[0], og_smem[0]);  // dP^T = V dO^T
    warpgroup::mma_commit_group();

    wait(vec_b[0], qo_idx % 2);
    stream_tile(s_block_t, l_smem, 0);
    mul(s_block_t, s_block_t, NEG_INV_SCALE);  // VSA2: -lse/scale
    wait(q_b[0], qo_idx % 2);
    warpgroup::mma_ABt(s_block_t, k_smem[0], q_smem[0]);  // S^T = K Q^T - lse/scale
    warpgroup::mma_commit_group();
    warpgroup::mma_async_wait();

    mul(s_block_t, s_block_t, SCALE_LOG2);
    lower_fill(s_block_t, s_block_t, fill_start, base_types::constants<float>::neg_infty());
    exp2(s_block_t, s_block_t);
    copy(p_block_t, s_block_t);
    copy(p_block_t_mma, s_block_t);
    stream_sub_tile(dp_block_t, d_smem, 0);  // dP - delta
    mul(ds_block_t, p_block_t, dp_block_t);  // dS = P * (dP - delta)
    mul(ds_block_t, ds_block_t, SCALE);

    if (has_next && threadIdx.x == 0) {
      coord<l_tile> vec_idx = {blockIdx.z, blockIdx.y, 0, load_q_block_index};
      tma::expect_bytes(vec_b[0], sizeof(l_smem[0]) + sizeof(d_smem[0]));
      tma::load_async(l_smem[0], g.l, vec_idx, vec_b[0]);
      tma::load_async(d_smem[0], g.d, vec_idx, vec_b[0]);
    }

    warpgroup::mma_AB(vg_reg, p_block_t_mma, og_smem[0]);  // dV += P^T dO
    warpgroup::mma_commit_group();
    copy(ds_block_t_mma, ds_block_t);
    warpgroup::store(ds_smem_t[0], ds_block_t);
    warpgroup::mma_async_wait();

    if (has_next && threadIdx.x == 0) {
      coord<q_tile> tile_idx = {blockIdx.z, blockIdx.y, load_q_block_index, 0};
      tma::expect_bytes(o_b[0], sizeof(og_smem[0]));
      tma::load_async(og_smem[0], g.og, tile_idx, o_b[0]);
    }

    warpgroup::mma_AB(kg_reg, ds_block_t_mma, q_smem[0]);  // dK += dS^T Q
    warpgroup::mma_commit_group();
    warpgroup::mma_async_wait();

    if (has_next && threadIdx.x == 0) {
      coord<q_tile> q_tile_idx = {blockIdx.z, blockIdx.y, load_q_block_index, 0};
      tma::expect_bytes(q_b[0], sizeof(q_smem[0]));
      tma::load_async(q_smem[0], g.q, q_tile_idx, q_b[0]);
    }

    rt_fl<16, G::tile_width> qg_reg;
    __syncthreads();
    warpgroup::mm_AtB(qg_reg, ds_smem_t[0], k_smem[0]);  // dQ = dS K
    warpgroup::mma_commit_group();
    warpgroup::mma_async_wait();
    warpgroup::store(qg_smem, qg_reg);
    __syncthreads();

    if (threadIdx.x / 32 == 0) {
      coord<qg_tile> tile_idx = {blockIdx.z, blockIdx.y, store_qg_block_index, 0};
      tma::store_add_async(g.qg, qg_smem, tile_idx);
      tma::store_async_wait();
    }
  }

  __syncthreads();
  warpgroup::store(kg_smem[0], kg_reg);
  __syncthreads();
  if (threadIdx.x / 32 == 0) {
    coord<kg_tile> tile_idx = {blockIdx.z, kv_head_idx, blockIdx.x, 0};
    if (g.store_kv) tma::store_async(g.kg, kg_smem[0], tile_idx);
    else tma::store_add_async(g.kg, kg_smem[0], tile_idx);
    tma::store_commit_group();
  }

  warpgroup::store(vg_smem[0], vg_reg);
  __syncthreads();
  if (kittens::warpid() % 4 == 0) {
    coord<vg_tile> tile_idx = {blockIdx.z, kv_head_idx, blockIdx.x, 0};
    if (g.store_kv) tma::store_async(g.vg, vg_smem[0], tile_idx);
    else tma::store_add_async(g.vg, vg_smem[0], tile_idx);
    tma::store_commit_group();
  }
  tma::store_async_wait();
#endif
}

// ------------------------------------------------------------------ host

template <int D>
static void tk_fwd_impl(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o, torch::Tensor lse,
                        torch::Tensor row_ptr, torch::Tensor col_idx, torch::Tensor blk_cnt) {
  using G = fwd_globals<D>;
  const unsigned B = q.size(0), H = q.size(1), L = q.size(2);
  bf16* dq = reinterpret_cast<bf16*>(q.data_ptr<c10::BFloat16>());
  bf16* dk = reinterpret_cast<bf16*>(k.data_ptr<c10::BFloat16>());
  bf16* dv = reinterpret_cast<bf16*>(v.data_ptr<c10::BFloat16>());
  bf16* d_o = reinterpret_cast<bf16*>(o.data_ptr<c10::BFloat16>());
  float* dl = lse.data_ptr<float>();
  typename G::q_gl qg{dq, B, H, L, (unsigned)D};
  typename G::k_gl kg{dk, B, H, L, (unsigned)D};
  typename G::v_gl vg{dv, B, H, L, (unsigned)D};
  typename G::l_gl lg{dl, B, H, 1U, L};
  typename G::o_gl og{d_o, B, H, L, (unsigned)D};
  const int NB = L / BLOCK_M;
  G g{qg, kg, vg, lg, og, (int)L, NB, row_ptr.data_ptr<int>(), col_idx.data_ptr<int>(), blk_cnt.data_ptr<int>()};
  constexpr int mem_size = 54000;
  cudaFuncSetAttribute(fwd_attend_ker<D>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);
  dim3 grid(NB, H, B);
  fwd_attend_ker<D><<<grid, 128, mem_size, at::cuda::getCurrentCUDAStream().stream()>>>(g);
}

template <int D>
static void tk_bwd_impl(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor dout_f, torch::Tensor lse,
                        torch::Tensor delta, torch::Tensor row_ptr, torch::Tensor col_idx, torch::Tensor blk_cnt,
                        torch::Tensor qg_acc, torch::Tensor kg_acc, torch::Tensor vg_acc, bool store_kv) {
  using G = bwd_globals<D>;
  const unsigned B = q.size(0), H = q.size(1), L = q.size(2);
  typename G::q_gl a_q{reinterpret_cast<bf16*>(q.data_ptr<c10::BFloat16>()), B, H, L, (unsigned)D};
  typename G::k_gl a_k{reinterpret_cast<bf16*>(k.data_ptr<c10::BFloat16>()), B, H, L, (unsigned)D};
  typename G::v_gl a_v{reinterpret_cast<bf16*>(v.data_ptr<c10::BFloat16>()), B, H, L, (unsigned)D};
  typename G::og_gl a_og{reinterpret_cast<bf16*>(dout_f.data_ptr<c10::BFloat16>()), B, H, L, (unsigned)D};
  typename G::qg_gl a_qg{qg_acc.data_ptr<float>(), B, H, L, (unsigned)D};
  typename G::kg_gl a_kg{kg_acc.data_ptr<float>(), B, H, L, (unsigned)D};
  typename G::vg_gl a_vg{vg_acc.data_ptr<float>(), B, H, L, (unsigned)D};
  typename G::l_gl a_l{lse.data_ptr<float>(), B, H, 1U, L};
  typename G::d_gl a_d{delta.data_ptr<float>(), B, H, 1U, L};
  const int NB = L / BLOCK_N;
  G g{a_q, a_k, a_v, a_og, a_qg, a_kg, a_vg, a_l, a_d, NB,
      row_ptr.data_ptr<int>(), col_idx.data_ptr<int>(), blk_cnt.data_ptr<int>(), (int)store_kv};
  constexpr int mem_size = (D == 64) ? 72000 : 113000;
  cudaFuncSetAttribute(bwd_attend_ker<D>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);
  dim3 grid(NB, H, B);
  bwd_attend_ker<D><<<grid, 128, mem_size, at::cuda::getCurrentCUDAStream().stream()>>>(g);
}

}  // namespace vsa2_tk

// q/k/v/o: [B, H, L, D] bf16; lse [B, H, L] fp32 (written)
void vsa2_tk_fine_fwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o, torch::Tensor lse,
                      torch::Tensor row_ptr, torch::Tensor col_idx, torch::Tensor blk_cnt) {
  const c10::cuda::OptionalCUDAGuard guard(q.device());
  if (q.size(3) == 64) vsa2_tk::tk_fwd_impl<64>(q, k, v, o, lse, row_ptr, col_idx, blk_cnt);
  else vsa2_tk::tk_fwd_impl<128>(q, k, v, o, lse, row_ptr, col_idx, blk_cnt);
}

// qg: fp32 [B, H, L, D] zero-initialized accumulator (written with TMA reduce-add). kg/vg: the same, or, with
// store_kv, any fp32 [B, H, L, D] buffers (every element is stored; see bwd_globals::store_kv)
void vsa2_tk_fine_bwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor dout_f, torch::Tensor lse,
                      torch::Tensor delta, torch::Tensor row_ptr, torch::Tensor col_idx, torch::Tensor blk_cnt,
                      torch::Tensor qg, torch::Tensor kg, torch::Tensor vg, bool store_kv) {
  const c10::cuda::OptionalCUDAGuard guard(q.device());
  if (q.size(3) == 64)
    vsa2_tk::tk_bwd_impl<64>(q, k, v, dout_f, lse, delta, row_ptr, col_idx, blk_cnt, qg, kg, vg, store_kv);
  else
    vsa2_tk::tk_bwd_impl<128>(q, k, v, dout_f, lse, delta, row_ptr, col_idx, blk_cnt, qg, kg, vg, store_kv);
}
