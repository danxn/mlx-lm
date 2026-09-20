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
MATRIX_SMEM = 32000  # bytes of threadgroup memory the matrix kernel may use
MATRIX_BK = 32  # keys per step of the matrix kernel
MIN_LOADERS = 4  # simdgroups per threadgroup, so that loading the keys is spread out
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
_MATRIX_TEMPLATE = r"""
constexpr int BK = BK_KEYS;                                 // keys per step
constexpr int CPL = BK / 4;                                 // score columns per lane in the softmax
constexpr int DT = D / 8;                                   // 8x8 tiles along the head dimension
constexpr int LD = D + 8;                                   // padded row of the shared K/V tiles
constexpr int NTH = 32 * NW;                               // extra simdgroups only help to load
constexpr int SWORDS = 8 * BK;                              // scores of one row tile (float)
constexpr int PWORDS = (8 * BK * sizeof(T) + 3) / 4;        // probabilities of one row tile
constexpr int SC = SWORDS + PWORDS + 64;                    // + 8x8 diagonal of row scale factors

constexpr int KVSIZE = 2 * BK * LD > NT * 8 * D ? 2 * BK * LD : NT * 8 * D;
threadgroup T kv[KVSIZE];                                   // shared keys and values (also holds the queries at the start)
threadgroup float smem[NT * SC];
threadgroup T* ks = kv;
threadgroup T* vs = kv + BK * LD;

const int kv_head = threadgroup_position_in_grid.x;
const int block = threadgroup_position_in_grid.y;
const int blocks = threadgroups_per_grid.y;
const int tile = simdgroup_index_in_threadgroup;            // which 8 rows this simdgroup owns
const int lane = thread_index_in_simdgroup;
const int tid = tile * 32 + lane;
const bool active = tile < NT;
const int total_rows = G * L;
const int N = keys_shape[2];
const int chunk = (((N + 7) / 8 + blocks - 1) / blocks) * 8;  // keys per block, multiple of 8
const int start = block * chunk;
const int end = min(N, start + chunk);

threadgroup float* sbuf = smem + tile * SC;
threadgroup T* pbuf = (threadgroup T*)(sbuf + SWORDS);
threadgroup float* dbuf = sbuf + SWORDS + PWORDS;

// queries: rows of a kv head are contiguous, the last row tile may be partly empty
for (int e = tid; e < NT * 8 * D; e += NTH) {
  const int rr = e / D;
  kv[e] = rr < total_rows ? queries[kv_head * total_rows * D + e] : T(0);
}
if (active) {
  for (int e = lane; e < 64; e += 32) {
    dbuf[e] = 0;
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
simdgroup_matrix<T, 8, 8> qf[DT];
if (active) {
  for (int dk = 0; dk < DT; dk++) {
    simdgroup_load(qf[dk], kv + tile * 8 * D + 8 * dk, D);
  }
}

simdgroup_matrix<float, 8, 8> o[DT];
for (int j = 0; j < DT; j++) {
  o[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
}
const int row = lane >> 2;                                   // row handled by this lane in the softmax
const int col0 = (lane & 3) * CPL;
float m_run = -3.4028234e38f;
float l_run = 0;
const float scale = scale_in[0];

//@DECLS

if (wide && start < end) {
  FETCH(start)
}

for (int pos = start; pos < end; pos += BK) {
  const int valid = min(BK, end - pos);
  const int nsub = (valid + 7) / 8;

  threadgroup_barrier(mem_flags::mem_threadgroup);
  //@STORE
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (wide && pos + BK < end) {
    FETCH(pos + BK)
  }

  if (active) {
    for (int t = 0; t < nsub; t++) {
      simdgroup_matrix<float, 8, 8> sm = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
      for (int dk = 0; dk < DT; dk++) {
        simdgroup_matrix<T, 8, 8> kt;
        simdgroup_load(kt, ks + 8 * t * LD + 8 * dk, LD, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(sm, qf[dk], kt, sm);
      }
      simdgroup_store(sm, sbuf + 8 * t, BK);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    float sc[CPL];
    float rmax = -3.4028234e38f;
    for (int i = 0; i < CPL; i++) {
      const int col = col0 + i;
      sc[i] = col < valid ? sbuf[row * BK + col] * scale : -3.4028234e38f;
      rmax = max(rmax, sc[i]);
    }
    rmax = max(rmax, simd_shuffle_xor(rmax, 1));
    rmax = max(rmax, simd_shuffle_xor(rmax, 2));
    const float new_m = max(m_run, rmax);
    const float alpha = metal::fast::exp(m_run - new_m);
    float psum = 0;
    for (int i = 0; i < CPL; i++) {
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
      simdgroup_matrix<T, 8, 8> pm;
      simdgroup_load(pm, pbuf + 8 * t, BK);
      for (int j = 0; j < DT; j++) {
        simdgroup_matrix<T, 8, 8> vm;
        simdgroup_load(vm, vs + 8 * t * LD + 8 * j, LD);
        simdgroup_multiply_accumulate(o[j], pm, vm, o[j]);
      }
    }
  }
}

// write this tile's partial result: unnormalised output, running max, running sum
if (active) {
  threadgroup float* obuf = sbuf;
  simdgroup_barrier(mem_flags::mem_threadgroup);
  for (int j = 0; j < DT; j++) {
    simdgroup_store(o[j], obuf, 8);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (int e = lane; e < 64; e += 32) {
      const int r = e >> 3;
      const int c = e & 7;
      const int rr = tile * 8 + r;
      if (rr < total_rows) {
        const int o_offset = kv_head * total_rows + rr;
        partial[(o_offset * blocks + block) * D + 8 * j + c] = obuf[e];
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
  if ((lane & 3) == 0) {
    const int rr = tile * 8 + row;
    if (rr < total_rows) {
      const int o_offset = kv_head * total_rows + rr;
      sums[o_offset * blocks + block] = end > start ? l_run : 0.0f;
      maxs[o_offset * blocks + block] = end > start ? m_run : -3.4028234e38f;
    }
  }
}
"""

_MATRIX_DECLS_PLAIN = r"""const long kstride = keys_strides[2];
const long vstride = values_strides[2];
const device T* kp = keys + kv_head * keys_strides[1];
const device T* vp = values + kv_head * values_strides[1];
const bool wide = (kstride % 8 == 0) && (vstride % 8 == 0) && (keys_strides[1] % 8 == 0) &&
    (values_strides[1] % 8 == 0);

constexpr int VPR = D * sizeof(T) / 16;                     // 16-byte vectors per row
constexpr int NV = (BK * VPR + NTH - 1) / NTH;              // vectors per thread and step
uint4 kreg[NV];
uint4 vreg[NV];

// Fetch the vectors of one step from device memory into registers, zero past the end.
#define FETCH(FROM)                                                             \
  for (int i = 0; i < NV; i++) {                                                \
    const int e = tid + i * NTH;                                                \
    const int r = e / VPR;                                                      \
    const int c = (e - r * VPR) * (16 / sizeof(T));                             \
    kreg[i] = uint4(0);                                                         \
    vreg[i] = uint4(0);                                                         \
    if (e < BK * VPR && (FROM) + r < end) {                                     \
      kreg[i] = *(const device uint4*)(kp + ((FROM) + r) * kstride + c);        \
      vreg[i] = *(const device uint4*)(vp + ((FROM) + r) * vstride + c);        \
    }                                                                           \
  }
"""

_MATRIX_STORE_PLAIN = r"""  if (wide) {
    for (int i = 0; i < NV; i++) {
      const int e = tid + i * NTH;
      const int r = e / VPR;
      const int c = (e - r * VPR) * (16 / sizeof(T));
      if (e < BK * VPR) {
        *(threadgroup uint4*)(ks + r * LD + c) = kreg[i];
        *(threadgroup uint4*)(vs + r * LD + c) = vreg[i];
      }
    }
  } else {
    for (int e = tid; e < BK * D; e += NTH) {
      const int r = e / D;
      const int c = e - r * D;
      const bool ok = pos + r < end;
      ks[r * LD + c] = ok ? kp[(pos + r) * kstride + c] : T(0);
      vs[r * LD + c] = ok ? vp[(pos + r) * vstride + c] : T(0);
    }
  }
"""

_MATRIX_DECLS_QUANTIZED = r"""
constexpr int EPW = 32 / BITS;                              // packed elements per 32-bit word
constexpr int UW = 8 / EPW;                                 // words per 8 elements
constexpr int UPR = D / 8;                                  // units of 8 elements per row
constexpr int NV = (BK * UPR + NTH - 1) / NTH;              // units per thread and step
const long kstride = kq_strides[2];
const long vstride = vq_strides[2];
const long ksstride = kscales_strides[2];
const long vsstride = vscales_strides[2];
const device uint32_t* kp = kq + kv_head * kq_strides[1];
const device uint32_t* vp = vq + kv_head * vq_strides[1];
const device T* ksp = kscales + kv_head * kscales_strides[1];
const device T* kbp = kbiases + kv_head * kbiases_strides[1];
const device T* vsp = vscales + kv_head * vscales_strides[1];
const device T* vbp = vbiases + kv_head * vbiases_strides[1];
const bool wide = true;

uint2 kreg[NV];
uint2 vreg[NV];
float kscale_reg[NV];
float kbias_reg[NV];
float vscale_reg[NV];
float vbias_reg[NV];

// Fetch the packed units of one step and their scales, zero past the end (dequantises to 0).
#define FETCH(FROM)                                                                  \
  for (int i = 0; i < NV; i++) {                                                     \
    const int e = tid + i * NTH;                                                     \
    const int r = e / UPR;                                                           \
    const int c = (e - r * UPR) * 8;                                                 \
    kreg[i] = uint2(0);                                                              \
    vreg[i] = uint2(0);                                                              \
    kscale_reg[i] = 0;                                                               \
    kbias_reg[i] = 0;                                                                \
    vscale_reg[i] = 0;                                                               \
    vbias_reg[i] = 0;                                                                \
    if (e < BK * UPR && (FROM) + r < end) {                                          \
      const long row = (FROM) + r;                                                   \
      kreg[i].x = kp[row * kstride + c / EPW];                                       \
      vreg[i].x = vp[row * vstride + c / EPW];                                       \
      if (UW == 2) {                                                                 \
        kreg[i].y = kp[row * kstride + c / EPW + 1];                                 \
        vreg[i].y = vp[row * vstride + c / EPW + 1];                                 \
      }                                                                              \
      kscale_reg[i] = ksp[row * ksstride + c / GS];                                  \
      kbias_reg[i] = kbp[row * ksstride + c / GS];                                   \
      vscale_reg[i] = vsp[row * vsstride + c / GS];                                  \
      vbias_reg[i] = vbp[row * vsstride + c / GS];                                   \
    }                                                                                \
  }
"""

_MATRIX_STORE_QUANTIZED = r"""  // unpack, scale and shift each unit of 8 values once here, so every row tile reads plain T
  for (int i = 0; i < NV; i++) {
    const int e = tid + i * NTH;
    const int r = e / UPR;
    const int c = (e - r * UPR) * 8;
    if (e < BK * UPR) {
      threadgroup T* kdst = ks + r * LD + c;
      threadgroup T* vdst = vs + r * LD + c;
      const float4 ksc = float4(kscale_reg[i]);
      const float4 kbi = float4(kbias_reg[i]);
      const float4 vsc = float4(vscale_reg[i]);
      const float4 vbi = float4(vbias_reg[i]);
      if (BITS == 8) {
        for (int h = 0; h < 2; h++) {
          const uint kw = h == 0 ? kreg[i].x : kreg[i].y;
          const uint vw = h == 0 ? vreg[i].x : vreg[i].y;
          *(threadgroup vec<T, 4>*)(kdst + 4 * h) = vec<T, 4>(fma(float4(as_type<uchar4>(kw)), ksc, kbi));
          *(threadgroup vec<T, 4>*)(vdst + 4 * h) = vec<T, 4>(fma(float4(as_type<uchar4>(vw)), vsc, vbi));
        }
      } else {
        // a byte holds two neighbours: low nibble first
        const float4 klo = float4(as_type<uchar4>(kreg[i].x & 0x0F0F0F0Fu));
        const float4 khi = float4(as_type<uchar4>((kreg[i].x >> 4) & 0x0F0F0F0Fu));
        const float4 vlo = float4(as_type<uchar4>(vreg[i].x & 0x0F0F0F0Fu));
        const float4 vhi = float4(as_type<uchar4>((vreg[i].x >> 4) & 0x0F0F0F0Fu));
        const vec<T, 4> kl = vec<T, 4>(fma(klo, ksc, kbi));
        const vec<T, 4> kh = vec<T, 4>(fma(khi, ksc, kbi));
        const vec<T, 4> vl = vec<T, 4>(fma(vlo, vsc, vbi));
        const vec<T, 4> vh = vec<T, 4>(fma(vhi, vsc, vbi));
        for (int j = 0; j < 4; j++) {
          *(threadgroup vec<T, 2>*)(kdst + 2 * j) = vec<T, 2>(kl[j], kh[j]);
          *(threadgroup vec<T, 2>*)(vdst + 2 * j) = vec<T, 2>(vl[j], vh[j]);
        }
      }
    }
  }
"""

_SOURCE_MATRIX = _MATRIX_TEMPLATE.replace("//@DECLS", _MATRIX_DECLS_PLAIN).replace("//@STORE", _MATRIX_STORE_PLAIN)
_SOURCE_MATRIX_QUANTIZED = (
    _MATRIX_TEMPLATE.replace("//@DECLS", _MATRIX_DECLS_QUANTIZED)
    .replace("//@STORE", _MATRIX_STORE_QUANTIZED)
    .replace("keys_shape[2]", "kq_shape[2]")
)

_kernel_matrix = None
_kernel_matrix_quantized = None


def _get_kernel_matrix_quantized():
    global _kernel_matrix_quantized
    if _kernel_matrix_quantized is None:
        _kernel_matrix_quantized = mx.fast.metal_kernel(
            name="matrix_rows_attention_quantized",
            input_names=[
                "queries", "kq", "kscales", "kbiases", "vq", "vscales", "vbiases", "scale_in",
            ],
            output_names=["partial", "sums", "maxs"],
            source=_SOURCE_MATRIX_QUANTIZED,
            header=_HEADER,
            ensure_row_contiguous=False,
        )
    return _kernel_matrix_quantized


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
    shared = max(2 * MATRIX_BK * (head_dim + 8), tiles * 8 * head_dim) * itemsize
    per_tile = (8 * MATRIX_BK + (8 * MATRIX_BK * itemsize + 3) // 4 + 64) * 4
    return shared + tiles * per_tile


def _kv_shape(keys):
    """Shape (B, KVH, N, ...) of a plain K/V array or of a quantized tuple."""
    return keys[0].shape if isinstance(keys, tuple) else keys.shape


def _use_matrix(queries, keys_shard) -> bool:
    B, H, L, D = queries.shape
    shape = _kv_shape(keys_shard)
    rows = (H // shape[1]) * L
    if B != 1 or rows > MAX_ROWS:
        return False
    if L == 1 and shape[2] < MATRIX_MIN_KEYS_ONE_TOKEN and not isinstance(keys_shard, tuple):
        return False
    return _matrix_smem_bytes(-(-rows // 8), D, queries.dtype.size) <= MATRIX_SMEM


def _pick_blocks(n_keys: int, group: int) -> int:
    """Number of key blocks (tuned on M3 Max: 256 up to ~90k keys, 1024 beyond)."""
    return 256 if n_keys < 90_000 else 1024


def _pick_blocks_matrix(n_keys: int) -> int:
    """Key blocks for the matrix kernel (tuned on M3 Max)."""
    return min(64, max(8, n_keys // 1024))


def _quantized_layout(queries, keys_shard, values_shard):
    """(bits, group size) of a quantized K/V pair the kernel can read, else None."""
    if not (isinstance(keys_shard, tuple) and isinstance(values_shard, tuple)):
        return None
    if len(keys_shard) != 3 or len(values_shard) != 3:
        return None
    D = queries.shape[-1]
    bits = keys_shard[0].shape[-1] * 32 // D
    group = D // keys_shard[1].shape[-1]
    if bits not in (4, 8) or group % 8 != 0 or D % group != 0:
        return None
    if values_shard[0].shape[-1] != keys_shard[0].shape[-1] or values_shard[1].shape[-1] != keys_shard[1].shape[-1]:
        return None
    if any(x.dtype != queries.dtype for x in (keys_shard[1], keys_shard[2], values_shard[1], values_shard[2])):
        return None
    if keys_shard[0].dtype != mx.uint32 or values_shard[0].dtype != mx.uint32:
        return None
    return bits, group


def supported(queries, keys_shard, values_shard, mask) -> bool:
    if not ENABLED or mask is not None or keys_shard is None:
        return False
    if mx.default_device() != mx.gpu:
        return False
    B, H, L, D = queries.shape
    quantized = isinstance(keys_shard, tuple)
    if quantized and _quantized_layout(queries, keys_shard, values_shard) is None:
        return False
    shape = _kv_shape(keys_shard)
    KVH = shape[1]
    if D not in (64, 128) or H % KVH != 0:
        return False
    if not quantized and (
        values_shard.shape[-1] != D
        or keys_shard.dtype != queries.dtype
        or values_shard.dtype != queries.dtype
    ):
        return False
    if shape[2] < MIN_KEYS:
        return False
    if _use_matrix(queries, keys_shard):
        return True
    return not quantized and L == 1 and H // KVH <= 32  # scalar kernel: one simdgroup per query head


def fast_partial_attention(queries, keys_shard, values_shard, scale, blocks=None):
    """(max, sum of exp, weighted V) of ``queries`` against a KV shard.

    Shapes match ``local_partial_attention``: (B,H,L,1), (B,H,L,1), (B,H,L,D).
    """
    B, H, L, D = queries.shape
    shape = _kv_shape(keys_shard)
    KVH, N = shape[1], shape[2]
    G = H // KVH
    q = mx.contiguous(queries)
    if _use_matrix(queries, keys_shard):
        blocks = blocks or _pick_blocks_matrix(N)
        tiles = -(-G * L // 8)
        loaders = max(tiles, MIN_LOADERS)
        template = [
            ("T", queries.dtype),
            ("D", D),
            ("G", G),
            ("L", L),
            ("NT", tiles),
            ("NW", loaders),
            ("BK_KEYS", MATRIX_BK),
        ]
        scale_in = mx.array([scale], dtype=mx.float32)
        if isinstance(keys_shard, tuple):
            bits, group = _quantized_layout(queries, keys_shard, values_shard)
            kernel = _get_kernel_matrix_quantized()
            inputs = [q, *keys_shard, *values_shard, scale_in]
            template += [("BITS", bits), ("GS", group)]
        else:
            kernel = _get_kernel_matrix()
            inputs = [q, keys_shard, values_shard, scale_in]
        partial, sums, maxs = kernel(
            inputs=inputs,
            template=template,
            grid=(KVH * 32 * loaders, blocks, 1),
            threadgroup=(32 * loaders, 1, 1),
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
