"""Fast partial attention for decoding (one query token) on a long KV shard.

Same job as ``distributed_attention.local_partial_attention`` for L == 1, but
runs as a Metal kernel modelled on MLX's own ``sdpa_vector_2pass_1``: the keys
are split into blocks, each block yields (max, sum of exp, weighted V), and the
blocks are merged with the online-softmax rule. MLX's public fused attention
does not return those statistics, which the cross-machine merge needs, so we
build the kernel with ``mx.fast.metal_kernel`` (no MLX rebuild required).

The kernel reads K and V straight from the (possibly sliced, non-contiguous)
cache buffers through their strides, so no copy of the cache is made.
"""

import os

import mlx.core as mx

ENABLED = os.environ.get("MLX_LM_SHARD_FAST_DECODE", "1") != "0"
MIN_KEYS = 256  # below this the plain path is as fast
MAX_ROWS = 64  # query rows (heads of a group x tokens) that share one pass over the keys
MATRIX_SMEM = 30000  # bytes of threadgroup memory the matrix kernel may use
MATRIX_MIN_KEYS_ONE_TOKEN = 32768  # below this a single token uses the scalar kernel

_HEADER = "#include <metal_simdgroup>\n#include <metal_simdgroup_matrix>\n"

_SOURCE = r"""
constexpr int BD = 32;
constexpr int qk_per_thread = D / BD;
constexpr int v_per_thread = D / BD;
typedef float U;

thread U q[qk_per_thread];
thread U o[v_per_thread];
for (int i = 0; i < v_per_thread; i++) {
  o[i] = 0;
}

const int kv_head_idx = threadgroup_position_in_grid.x;
const int batch_idx = threadgroup_position_in_grid.y;
const int block_idx = threadgroup_position_in_grid.z;
const int gqa_factor = threads_per_threadgroup.y;
const int q_seq_len = threads_per_threadgroup.z;
const int q_seq_idx = thread_position_in_threadgroup.z;
const int q_head_idx = gqa_factor * kv_head_idx + thread_position_in_threadgroup.y;
const int num_kv_heads = threadgroups_per_grid.x;
const int num_q_heads = num_kv_heads * gqa_factor;
const int blocks = threadgroups_per_grid.z;
const int lane = thread_index_in_simdgroup;
const int N = keys_shape[2];
const int q_batch_head_idx = batch_idx * num_q_heads + q_head_idx;
const int o_offset = q_batch_head_idx * q_seq_len + q_seq_idx;

const device T* qp = queries + o_offset * D + lane * qk_per_thread;
const device T* kp = keys + batch_idx * keys_strides[0] + kv_head_idx * keys_strides[1]
    + block_idx * keys_strides[2] + lane * qk_per_thread;
const device T* vp = values + batch_idx * values_strides[0] + kv_head_idx * values_strides[1]
    + block_idx * values_strides[2] + lane * v_per_thread;
const long kstep = blocks * keys_strides[2];
const long vstep = blocks * values_strides[2];

const U scale = static_cast<U>(scale_in[0]);
for (int i = 0; i < qk_per_thread; i++) {
  q[i] = scale * static_cast<U>(qp[i]);
}

U max_score = -3.4028234e38f;
U sum_exp = 0;
for (int i = block_idx; i < N; i += blocks) {
  U score = 0;
  for (int j = 0; j < qk_per_thread; j++) {
    score += q[j] * static_cast<U>(kp[j]);
  }
  score = simd_sum(score);
  U new_max = max(max_score, score);
  U factor = metal::fast::exp(max_score - new_max);
  U exp_score = metal::fast::exp(score - new_max);
  max_score = new_max;
  sum_exp = sum_exp * factor + exp_score;
  for (int j = 0; j < v_per_thread; j++) {
    o[j] = o[j] * factor + exp_score * static_cast<U>(vp[j]);
  }
  kp += kstep;
  vp += vstep;
}

device float* op = partial + (o_offset * blocks + block_idx) * D + lane * v_per_thread;
for (int j = 0; j < v_per_thread; j++) {
  op[j] = o[j];
}
if (lane == 0) {
  sums[o_offset * blocks + block_idx] = sum_exp;
  maxs[o_offset * blocks + block_idx] = max_score;
}
"""

_kernel = None


def _get_kernel():
    global _kernel
    if _kernel is None:
        _kernel = mx.fast.metal_kernel(
            name="partial_decode_attention",
            input_names=["queries", "keys", "values", "scale_in"],
            output_names=["partial", "sums", "maxs"],
            source=_SOURCE,
            header=_HEADER,
            ensure_row_contiguous=False,
        )
    return _kernel


# Attention for several query rows with 8x8 matrix instructions. A simdgroup owns 8
# rows, reads each key and value tile once, and multiplies whole tiles: scores for
# 8 keys at a time, a softmax step over 32 keys, then the product with the values.
_SOURCE_MATRIX = r"""
constexpr int BK = 32;
constexpr int DT = D / 8;                                   // 8x8 tiles along the head dimension
constexpr int QWORDS = (8 * D * sizeof(T) + 3) / 4;
constexpr int SWORDS = 8 * BK;
constexpr int PWORDS = (8 * BK * sizeof(T) + 3) / 4;
constexpr int BASEW = QWORDS > SWORDS + PWORDS ? QWORDS : SWORDS + PWORDS;
constexpr int SC = BASEW + 128 + 128;   // + staging of the last, partly filled tile of keys and values

threadgroup float smem[NT * SC];

const int kv_head = threadgroup_position_in_grid.x;
const int block = threadgroup_position_in_grid.y;
const int blocks = threadgroups_per_grid.y;
const int tile = simdgroup_index_in_threadgroup;            // which 8 rows this simdgroup owns
const int lane = thread_index_in_simdgroup;
const int total_rows = G * L;
const int N = keys_shape[2];
const int chunk = (((N + 7) / 8 + blocks - 1) / blocks) * 8;  // keys per block, multiple of 8
const int start = block * chunk;
const int end = min(N, start + chunk);

threadgroup float* base = smem + tile * SC;
threadgroup T* qbuf = (threadgroup T*)base;
threadgroup float* sbuf = base;
threadgroup T* pbuf = (threadgroup T*)(base + SWORDS);
threadgroup float* dbuf = base + BASEW;           // 8x8 diagonal (row rescale factors)
threadgroup float* obuf = dbuf + 64;                        // 8x8 staging for the output
threadgroup T* kstage = (threadgroup T*)(dbuf + 128);       // zero padded 8x8 tile of keys
threadgroup T* vstage = kstage + 64;                        // ... and of values

// queries of this tile into threadgroup memory (rows beyond the last one are zero)
for (int e = lane; e < 8 * D; e += 32) {
  const int r = e / D;
  const int d = e - r * D;
  const int rr = tile * 8 + r;
  T value = 0;
  if (rr < total_rows) {
    const int g = rr / L;
    const int l = rr - g * L;
    value = queries[((kv_head * G + g) * L + l) * D + d];
  }
  qbuf[e] = value;
}
for (int e = lane; e < 64; e += 32) {
  dbuf[e] = 0;
}
simdgroup_barrier(mem_flags::mem_threadgroup);
simdgroup_matrix<T, 8, 8> qf[DT];
for (int dk = 0; dk < DT; dk++) {
  simdgroup_load(qf[dk], qbuf + 8 * dk, D);
}
simdgroup_barrier(mem_flags::mem_threadgroup);

simdgroup_matrix<float, 8, 8> o[DT];
for (int j = 0; j < DT; j++) {
  o[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
}
const int row = lane >> 2;                                   // row handled by this lane in the softmax
const int col0 = (lane & 3) * 8;
float m_run = -3.4028234e38f;
float l_run = 0;
const float scale = scale_in[0];

const long kstride = keys_strides[2];
const long vstride = values_strides[2];
const device T* kp = keys + kv_head * keys_strides[1];
const device T* vp = values + kv_head * values_strides[1];

for (int pos = start; pos < end; pos += BK) {
  const int valid = min(BK, end - pos);
  const int nsub = (valid + 7) / 8;
  for (int t = 0; t < nsub; t++) {
    const int filled = min(8, valid - 8 * t);      // keys of this tile that exist
    simdgroup_matrix<float, 8, 8> s = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (int dk = 0; dk < DT; dk++) {
      simdgroup_matrix<T, 8, 8> kt;
      if (filled == 8) {
        simdgroup_load(kt, kp + (pos + 8 * t) * kstride + 8 * dk, kstride, ulong2(0, 0), true);
      } else {
        for (int e = lane; e < 64; e += 32) {
          const int r = e >> 3;
          kstage[e] = r < filled ? kp[(pos + 8 * t + r) * kstride + 8 * dk + (e & 7)] : T(0);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_load(kt, kstage, 8, ulong2(0, 0), true);
        simdgroup_barrier(mem_flags::mem_threadgroup);
      }
      simdgroup_multiply_accumulate(s, qf[dk], kt, s);
    }
    simdgroup_store(s, sbuf + 8 * t, BK);
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);

  float sc[8];
  float rmax = -3.4028234e38f;
  for (int i = 0; i < 8; i++) {
    const int col = col0 + i;
    sc[i] = col < valid ? sbuf[row * BK + col] * scale : -3.4028234e38f;
    rmax = max(rmax, sc[i]);
  }
  rmax = max(rmax, simd_shuffle_xor(rmax, 1));
  rmax = max(rmax, simd_shuffle_xor(rmax, 2));
  const float new_m = max(m_run, rmax);
  const float alpha = metal::fast::exp(m_run - new_m);
  float psum = 0;
  for (int i = 0; i < 8; i++) {
    const float p = metal::fast::exp(sc[i] - new_m);
    pbuf[row * BK + col0 + i] = static_cast<T>(p);
    psum += p;
  }
  psum += simd_shuffle_xor(psum, 1);
  psum += simd_shuffle_xor(psum, 2);
  l_run = l_run * alpha + psum;
  m_run = new_m;
  if ((lane & 3) == 0) {
    dbuf[row * 8 + row] = alpha;
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);

  simdgroup_matrix<float, 8, 8> dm;
  simdgroup_load(dm, dbuf, 8);
  for (int j = 0; j < DT; j++) {
    simdgroup_matrix<float, 8, 8> tmp;
    simdgroup_multiply(tmp, dm, o[j]);
    o[j] = tmp;
  }
  for (int t = 0; t < nsub; t++) {
    const int filled = min(8, valid - 8 * t);
    simdgroup_matrix<T, 8, 8> pm;
    simdgroup_load(pm, pbuf + 8 * t, BK);
    for (int j = 0; j < DT; j++) {
      simdgroup_matrix<T, 8, 8> vm;
      if (filled == 8) {
        simdgroup_load(vm, vp + (pos + 8 * t) * vstride + 8 * j, vstride);
      } else {
        for (int e = lane; e < 64; e += 32) {
          const int r = e >> 3;
          vstage[e] = r < filled ? vp[(pos + 8 * t + r) * vstride + 8 * j + (e & 7)] : T(0);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_load(vm, vstage, 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
      }
      simdgroup_multiply_accumulate(o[j], pm, vm, o[j]);
    }
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
}

// write this tile's partial result: unnormalised output, running max, running sum
for (int j = 0; j < DT; j++) {
  simdgroup_store(o[j], obuf, 8);
  simdgroup_barrier(mem_flags::mem_threadgroup);
  for (int e = lane; e < 64; e += 32) {
    const int r = e >> 3;
    const int c = e & 7;
    const int rr = tile * 8 + r;
    if (rr < total_rows) {
      const int g = rr / L;
      const int l = rr - g * L;
      const int o_offset = (kv_head * G + g) * L + l;
      partial[(o_offset * blocks + block) * D + 8 * j + c] = obuf[e];
    }
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
}
if ((lane & 3) == 0) {
  const int rr = tile * 8 + row;
  if (rr < total_rows) {
    const int g = rr / L;
    const int l = rr - g * L;
    const int o_offset = (kv_head * G + g) * L + l;
    sums[o_offset * blocks + block] = end > start ? l_run : 0.0f;
    maxs[o_offset * blocks + block] = end > start ? m_run : -3.4028234e38f;
  }
}
"""

_kernel_matrix = None


def _get_kernel_matrix():
    global _kernel_matrix
    if _kernel_matrix is None:
        _kernel_matrix = mx.fast.metal_kernel(
            name="matrix_rows_attention",
            input_names=["queries", "keys", "values", "scale_in"],
            output_names=["partial", "sums", "maxs"],
            source=_SOURCE_MATRIX,
            header=_HEADER,
            ensure_row_contiguous=False,
        )
    return _kernel_matrix


def _matrix_smem_bytes(tiles: int, head_dim: int, itemsize: int) -> int:
    queries = (8 * head_dim * itemsize + 3) // 4
    steps = 8 * 32 + (8 * 32 * itemsize + 3) // 4
    return tiles * (max(queries, steps) + 256) * 4


def _use_matrix(queries, keys_shard) -> bool:
    B, H, L, D = queries.shape
    rows = (H // keys_shard.shape[1]) * L
    if B != 1 or rows > MAX_ROWS:
        return False
    if L == 1 and keys_shard.shape[2] < MATRIX_MIN_KEYS_ONE_TOKEN:
        return False
    return _matrix_smem_bytes(-(-rows // 8), D, queries.dtype.size) <= MATRIX_SMEM


def _pick_blocks(n_keys: int, group: int) -> int:
    """Number of key blocks (tuned on M3 Max: 256 up to ~90k keys, 1024 beyond)."""
    return 256 if n_keys < 90_000 else 1024


def _pick_blocks_matrix(n_keys: int) -> int:
    """Key blocks for the matrix kernel (tuned on M3 Max)."""
    return 128 if n_keys >= 2048 else 16


def supported(queries, keys_shard, values_shard, mask) -> bool:
    if not ENABLED or mask is not None:
        return False
    if isinstance(keys_shard, tuple) or keys_shard is None:
        return False
    if mx.default_device() != mx.gpu:
        return False
    B, H, L, D = queries.shape
    KVH = keys_shard.shape[1]
    if D not in (64, 128) or values_shard.shape[-1] != D or H % KVH != 0:
        return False
    if keys_shard.dtype != queries.dtype or values_shard.dtype != queries.dtype:
        return False
    if keys_shard.shape[2] < MIN_KEYS:
        return False
    if _use_matrix(queries, keys_shard):
        return True
    return L == 1 and H // KVH <= 32  # scalar kernel: one simdgroup per query head


def fast_partial_attention(queries, keys_shard, values_shard, scale, blocks=None):
    """(max, sum of exp, weighted V) of ``queries`` against a KV shard.

    Shapes match ``local_partial_attention``: (B,H,L,1), (B,H,L,1), (B,H,L,D).
    """
    B, H, L, D = queries.shape
    KVH = keys_shard.shape[1]
    G = H // KVH
    N = keys_shard.shape[2]
    q = mx.contiguous(queries)
    if _use_matrix(queries, keys_shard):
        blocks = blocks or _pick_blocks_matrix(N)
        tiles = -(-G * L // 8)
        partial, sums, maxs = _get_kernel_matrix()(
            inputs=[q, keys_shard, values_shard, mx.array([scale], dtype=mx.float32)],
            template=[("T", queries.dtype), ("D", D), ("G", G), ("L", L), ("NT", tiles)],
            grid=(KVH * 32 * tiles, blocks, 1),
            threadgroup=(32 * tiles, 1, 1),
            output_shapes=[(B, H, L, blocks, D), (B, H, L, blocks), (B, H, L, blocks)],
            output_dtypes=[mx.float32, mx.float32, mx.float32],
        )
        return _combine_blocks(partial, sums, maxs, queries.dtype)
    if L > 1:
        raise ValueError(
            f"{G * L} query rows do not fit the fast kernels (up to {MAX_ROWS}); check supported() first"
        )
    blocks = blocks or _pick_blocks(N, G)
    partial, sums, maxs = _get_kernel()(
        inputs=[q, keys_shard, values_shard, mx.array([scale], dtype=mx.float32)],
        template=[("T", queries.dtype), ("D", D)],
        grid=(KVH * 32, B * G, blocks * L),
        threadgroup=(32, G, L),
        output_shapes=[(B, H, L, blocks, D), (B, H, L, blocks), (B, H, L, blocks)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    return _combine_blocks(partial, sums, maxs, queries.dtype)


def _combine_blocks(partial, sums, maxs, dtype):
    m = mx.max(maxs, axis=-1, keepdims=True)
    f = mx.exp(maxs - m)
    total = mx.sum(sums * f, axis=-1, keepdims=True)
    wv = mx.sum(partial * f[..., None], axis=-2)
    return m.astype(dtype), total.astype(dtype), wv.astype(dtype)
