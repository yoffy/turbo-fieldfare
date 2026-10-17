#include <metal_stdlib>
using namespace metal;

// ============================================================================
// qwen38.metal — Qwen3.8-Flash-Next (`qwen4_exp` text) small kernels.
//
// All storage FP16/BF16, FP32 accumulation. Each kernel mirrors one exact
// equation in `Qwen38Reference` (the Swift port of transformers
// `modular_qwen4_exp.py`); the reference is the audit source of truth.
//
// Covered here (kept out of the shared modules so their tuned paths stay
// bit-identical):
//   grouped/single RMSNorm with (1 + w) scale (Qwen4ExpTextRMSNorm,
//     zeros-init — NOT the plain `* w` of rmsnorm.metal),
//   hyper-connection elementwise (repeat / mix-mean / inject-add /
//     silu-div / sigmoid / sigmoid2-div),
//   QSA indexer helpers (block score / row gather; pooling runs on CPU),
//   PLE helpers (per-branch dot / sigmoid-scale rows / dilated short conv).
// INT4 GEMVs (hyper down/up/inject, indexer qk, PLE key/value, QSA q/k/v/o,
// GDN in-proj, shared/routed experts, lm_head) reuse the shared 32-wide
// dequant bodies via function-constant group selection.
// ============================================================================

// -- RMSNorm with (1 + w) -----------------------------------------------
// y = x * rsqrt(mean(x^2) + eps) * (1 + w). Reference:
// Qwen4ExpTextRMSNorm (weight zeros-init).

// Grouped variant: weight [H, D] with one segment per group (hc_norm,
// ple norm_key/query/conv). Otherwise identical math.
kernel void qwen38_rmsnorm_1pw_grouped(
    device const half*   x      [[buffer(0)]],   // [H, D] FP16
    device const bfloat* weight [[buffer(1)]],   // [H, D] BF16
    device half*         out    [[buffer(2)]],   // [H, D] FP16
    constant uint&       headDim [[buffer(3)]],
    constant float&      eps    [[buffer(4)]],
    uint head [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]]
) {
    device const half* xh = x + head * headDim;
    device half* oh = out + head * headDim;
    device const bfloat* wh = weight + head * headDim;
    threadgroup float partial[8];
    threadgroup float tg;
    uint sg = lid >> 5;
    uint lane = lid & 31u;
    for (uint p = lid; p < 8u; p += lsize) partial[p] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float acc = 0.0f;
    for (uint i = lid; i < headDim; i += lsize) {
        float v = float(xh[i]);
        acc = fma(v, v, acc);
    }
    float laneSum = simd_sum(acc);
    if (lane == 0) partial[sg] = laneSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && lane == 0) {
        float sum = 0.0f;
        for (uint s = 0; s < 8u; ++s) sum += partial[s];
        tg = rsqrt(sum / float(headDim) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = tg;
    for (uint i = lid; i < headDim; i += lsize) {
        float xv = float(xh[i]);
        float wv = float(wh[i]);
        oh[i] = half(xv * inv * (1.0f + wv));
    }
}

// -- Hyper-connection elementwise ----------------------------------------

// y[h, d] = x[d] (broadcast the token embedding over hc branches).
kernel void qwen38_hc_repeat(
    device const half* x [[buffer(0)]],   // [D] FP16
    device half*       y [[buffer(1)]],   // [H, D] FP16
    constant uint&     D [[buffer(2)]],
    constant uint&     H [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = H * D;
    if (gid >= total) return;
    y[gid] = x[gid % D];
}

// mixed[d] = mean_h(mix[h, d] * normed[h, d]).
kernel void qwen38_hc_mix_mean(
    device const half* mix    [[buffer(0)]],   // [H, D] FP16 (sigmoid output)
    device const half* normed [[buffer(1)]],   // [H, D] FP16
    device half*       mixed  [[buffer(2)]],   // [D] FP16
    constant uint&     D [[buffer(3)]],
    constant uint&     H [[buffer(4)]],
    uint d [[thread_position_in_grid]]
) {
    if (d >= D) return;
    float acc = 0.0f;
    for (uint h = 0; h < H; ++h) {
        acc += float(mix[h * D + d]) * float(normed[h * D + d]);
    }
    mixed[d] = half(acc / float(H));
}

// hyper[h, d] += sub[d] * inj[h] (in-place safe).
kernel void qwen38_inject_add(
    device half*       hyper [[buffer(0)]],   // [H, D] FP16, in/out
    device const half* sub   [[buffer(1)]],   // [D] FP16
    device const half* inj   [[buffer(2)]],   // [H] FP16
    constant uint&     D [[buffer(3)]],
    constant uint&     H [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = H * D;
    if (gid >= total) return;
    uint h = gid / D;
    uint d = gid % D;
    hyper[gid] = half(float(hyper[gid]) + float(sub[d]) * float(inj[h]));
}

// y[i] = silu(x[i] / div).
kernel void qwen38_silu_div(
    device const half* x [[buffer(0)]],
    device half*       y [[buffer(1)]],
    constant float&    div [[buffer(2)]],
    constant uint&     n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    float v = float(x[gid]) / div;
    y[gid] = half(v / (1.0f + exp(-v)));
}

// y[i] = sigmoid(x[i]).
kernel void qwen38_sigmoid(
    device const half* x [[buffer(0)]],
    device half*       y [[buffer(1)]],
    constant uint&     n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    float v = float(x[gid]);
    y[gid] = half(1.0f / (1.0f + exp(-v)));
}

// y[i] = 2 * sigmoid(x[i] / div) (block injection weights).
kernel void qwen38_sigmoid2_div(
    device const half* x [[buffer(0)]],
    device half*       y [[buffer(1)]],
    constant float&    div [[buffer(2)]],
    constant uint&     n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    float v = float(x[gid]) / div;
    y[gid] = half(2.0f / (1.0f + exp(-v)));
}

// -- QSA indexer helpers --------------------------------------------------

// scores[b] = sum_h relu(dot(q[h], k[b])) / sqrt(HD). Reference:
// Qwen4ExpTextQSAIndexer (post-layernorm, post-RoPE, FP32 matmul).
kernel void qsa_block_score(
    device const half* q      [[buffer(0)]],   // [nHeads, HD] FP16
    device const half* k      [[buffer(1)]],   // [nBlocks, HD] FP16
    device float*      scores [[buffer(2)]],   // [nBlocks] FP32
    constant uint&     headDim [[buffer(3)]],
    constant uint&     nHeads [[buffer(4)]],
    uint b [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]]
) {
    threadgroup float partial[8];
    uint sg = lid >> 5;
    uint lane = lid & 31u;
    for (uint p = lid; p < 8u; p += lsize) partial[p] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float laneSum = 0.0f;
    for (uint h = 0; h < nHeads; ++h) {
        float dot = 0.0f;
        for (uint d = lid; d < headDim; d += lsize) {
            dot = fma(float(q[h * headDim + d]), float(k[b * headDim + d]), dot);
        }
        dot = simd_sum(dot);
        if (lane == 0) laneSum += fmax(dot, 0.0f);
    }
    if (lane == 0) partial[sg] = laneSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && lane == 0) {
        float sum = 0.0f;
        for (uint s = 0; s < 8u; ++s) sum += partial[s];
        scores[b] = sum / sqrt(float(headDim));
    }
}

// dst[i, :] = src[sel[i], :] (compact selected KV for dense attention).
kernel void qsa_gather_rows(
    device const half* src [[buffer(0)]],
    device const uint* sel [[buffer(1)]],
    device half*       dst [[buffer(2)]],
    constant uint&     rowElems [[buffer(3)]],
    constant uint&     rows [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = rows * rowElems;
    if (gid >= total) return;
    uint i = gid / rowElems;
    uint j = gid % rowElems;
    dst[gid] = src[sel[i] * rowElems + j];
}

// -- PLE helpers ------------------------------------------------------------

// dots[h] = sum_d(key[h, d] * query[h, d]) / sqrt(D) (FP32 out).
kernel void ple_gate_dot(
    device const half* key  [[buffer(0)]],   // [H, D] FP16
    device const half* query [[buffer(1)]],  // [H, D] FP16
    device float*      dots [[buffer(2)]],   // [H] FP32
    constant uint&     D [[buffer(3)]],
    uint h [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]]
) {
    threadgroup float partial[8];
    uint sg = lid >> 5;
    uint lane = lid & 31u;
    for (uint p = lid; p < 8u; p += lsize) partial[p] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float acc = 0.0f;
    for (uint d = lid; d < D; d += lsize) {
        acc = fma(float(key[h * D + d]), float(query[h * D + d]), acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) partial[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && lane == 0) {
        float sum = 0.0f;
        for (uint s = 0; s < 8u; ++s) sum += partial[s];
        dots[h] = sum / sqrt(float(D));
    }
}

// out[h, d] = sigmoid(gate[h]) * value[d].
kernel void ple_scale_rows(
    device const half*  value [[buffer(0)]],   // [D] FP16
    device const float* gate  [[buffer(1)]],   // [H] FP32 (signed-sqrt gate)
    device half*        out   [[buffer(2)]],   // [H, D] FP16
    constant uint&      D [[buffer(3)]],
    constant uint&      H [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = H * D;
    if (gid >= total) return;
    uint h = gid / D;
    uint d = gid % D;
    float g = gate[h];
    float s = 1.0f / (1.0f + exp(-g));
    out[gid] = half(s * float(value[d]));
}

// Dilated depthwise short conv, single decode step. State holds the previous
// (kernel-1)*dilation normed inputs; on entry state = [.., oldest..newest],
// input = current. Taps are cross-correlation taps: tap t weights the lag
// (K-1-t)*dilation, so w[K-1] lands on the current input and w[0] on the
// oldest state row. After the taps, state shifts left by one and appends
// input. Reference: Qwen4ExpTextPLELayer._short_conv (dilation = ngram_size).
kernel void ple_short_conv_step(
    device half*       state  [[buffer(0)]],   // [S, C] FP16, in/out
    device const half* input [[buffer(1)]],   // [C] FP16
    device const bfloat* weight [[buffer(2)]], // [C, K] BF16
    device half*       out    [[buffer(3)]],   // [C] FP16
    constant uint&     channels [[buffer(4)]],
    constant uint&     kernelSize [[buffer(5)]],
    constant uint&     dilation [[buffer(6)]],
    constant uint&     stateLen [[buffer(7)]],
    uint c [[thread_position_in_grid]]
) {
    if (c >= channels) return;
    // Gather taps from [state | input]. Cross-correlation (F.conv1d): tap t
    // weights the lag (K-1-t)*dilation, so the current input carries w[K-1]
    // and the oldest state row carries w[0].
    float acc = 0.0f;
    for (uint t = 0; t < kernelSize; ++t) {
        uint back = (kernelSize - 1u - t) * dilation;
        float v;
        if (back == 0) {
            v = float(input[c]);
        } else {
            // back >= 1: position (stateLen - back) in state; back <= stateLen
            // by construction ((K-1)*dilation == stateLen).
            v = float(state[(stateLen - back) * channels + c]);
        }
        acc = fma(float(weight[c * kernelSize + t]), v, acc);
    }
    float s = acc / (1.0f + exp(-acc));
    out[c] = half(s);
    // Shift state left by one row and append input (per-channel element).
    // Done by every channel thread for its own column: threads cooperate on
    // disjoint columns, so no barrier is needed.
    for (uint r = 0; r + 1u < stateLen; ++r) {
        state[r * channels + c] = state[(r + 1u) * channels + c];
    }
    state[(stateLen - 1u) * channels + c] = input[c];
}
