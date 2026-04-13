#include "cutie/ort/core/cuda_kernels.h"

#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstdio>

// ── 辅助宏 ────────────────────────────────────────────────────────────

#define CUDA_CHECK(call)                                                                   \
    do                                                                                     \
    {                                                                                      \
        cudaError_t err = (call);                                                          \
        if (err != cudaSuccess)                                                            \
        {                                                                                  \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,               \
                    cudaGetErrorString(err));                                               \
        }                                                                                  \
    } while (0)

static constexpr int kBlockSize = 256;

static inline int div_ceil(int64_t n, int block)
{
    return static_cast<int>((n + block - 1) / block);
}

namespace cutie
{
namespace cuda
{

// ── concat_last_dim ─────────────────────────────────────────────────

__global__ void concat_last_dim_kernel(const float* a, int64_t a_inner, const float* b,
                                       int64_t b_inner, float* out, int64_t total_inner,
                                       int64_t total)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;

    int64_t outer_idx = idx / total_inner;
    int64_t inner_idx = idx % total_inner;

    if (inner_idx < a_inner)
    {
        out[idx] = a[outer_idx * a_inner + inner_idx];
    }
    else
    {
        out[idx] = b[outer_idx * b_inner + (inner_idx - a_inner)];
    }
}

void concat_last_dim(const float* a, int64_t a_inner, const float* b, int64_t b_inner,
                     float* out, int64_t outer)
{
    int64_t total_inner = a_inner + b_inner;
    int64_t total = outer * total_inner;
    concat_last_dim_kernel<<<div_ceil(total, kBlockSize), kBlockSize>>>(a, a_inner, b, b_inner, out,
                                                                        total_inner, total);
    CUDA_CHECK(cudaGetLastError());
}

// ── slice_last_dim ──────────────────────────────────────────────────

__global__ void slice_last_dim_kernel(const float* src, int64_t src_inner, float* dst,
                                      int64_t slice_len, int64_t offset, int64_t total)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;

    int64_t outer_idx = idx / slice_len;
    int64_t inner_idx = idx % slice_len;
    dst[idx] = src[outer_idx * src_inner + offset + inner_idx];
}

void slice_last_dim(const float* src, int64_t src_inner, float* dst, int64_t slice_len,
                    int64_t offset, int64_t outer)
{
    int64_t total = outer * slice_len;
    slice_last_dim_kernel<<<div_ceil(total, kBlockSize), kBlockSize>>>(src, src_inner, dst,
                                                                       slice_len, offset, total);
    CUDA_CHECK(cudaGetLastError());
}

// ── sigmoid ─────────────────────────────────────────────────────────

__global__ void sigmoid_kernel(const float* x, float* out, int64_t n)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;
    out[idx] = 1.0f / (1.0f + expf(-x[idx]));
}

void sigmoid(const float* x, float* out, int64_t n)
{
    sigmoid_kernel<<<div_ceil(n, kBlockSize), kBlockSize>>>(x, out, n);
    CUDA_CHECK(cudaGetLastError());
}

// ── aggregate_softmax ───────────────────────────────────────────────
// prob_no_bg: [num_obj, HW] → out: [num_obj+1, HW]
// 对每个像素位置：prob→logit，加 bg(logit=0)，softmax

__global__ void aggregate_softmax_kernel(const float* prob_no_bg, float* out, int num_obj, int hw)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    if (px >= hw)
        return;

    int num_ch = num_obj + 1;

    // 1. prob → logit，找 max
    float max_val = 0.0f;  // bg logit = 0
    for (int o = 0; o < num_obj; ++o)
    {
        float p = prob_no_bg[o * hw + px];
        p = fmaxf(1e-7f, fminf(1.0f - 1e-7f, p));
        float logit = logf(p / (1.0f - p));
        if (logit > max_val)
            max_val = logit;
    }

    // 2. softmax
    float sum_exp = expf(-max_val);  // bg: exp(0 - max)
    out[px] = sum_exp;               // bg channel
    for (int o = 0; o < num_obj; ++o)
    {
        float p = prob_no_bg[o * hw + px];
        p = fmaxf(1e-7f, fminf(1.0f - 1.0e-7f, p));
        float logit = logf(p / (1.0f - p));
        float e = expf(logit - max_val);
        out[(o + 1) * hw + px] = e;
        sum_exp += e;
    }
    float inv = 1.0f / sum_exp;
    for (int c = 0; c < num_ch; ++c)
    {
        out[c * hw + px] *= inv;
    }
}

void aggregate_softmax(const float* prob_no_bg, float* out, int num_obj, int hw)
{
    aggregate_softmax_kernel<<<div_ceil(hw, kBlockSize), kBlockSize>>>(prob_no_bg, out, num_obj, hw);
    CUDA_CHECK(cudaGetLastError());
}

// ── get_similarity ──────────────────────────────────────────────────
// memory_key: [B, CK, N], memory_shrinkage: [B, 1, N]
// query_key: [B, CK, HW], query_selection: [B, CK, HW]
// out: [B, N, HW]
// similarity(b,n,hw) = sum_c( mk[b,c,n] * ms[b,0,n] * qk[b,c,hw] * qs[b,c,hw] )

__global__ void get_similarity_kernel(const float* mk, const float* ms, const float* qk,
                                      const float* qs, float* out, int B, int CK, int N, int HW)
{
    // 每个线程处理一个 (b, n, hw) 位置
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)B * N * HW;
    if (idx >= total)
        return;

    int hw = idx % HW;
    int n = (idx / HW) % N;
    int b = idx / (N * HW);

    float shrink = ms[b * N + n];
    float sum = 0.0f;
    for (int c = 0; c < CK; ++c)
    {
        float m = mk[b * CK * N + c * N + n] * shrink;
        float q = qk[b * CK * HW + c * HW + hw] * qs[b * CK * HW + c * HW + hw];
        sum += m * q;
    }
    out[idx] = sum;
}

void get_similarity(const float* memory_key, const float* memory_shrinkage,
                    const float* query_key, const float* query_selection, float* out, int B,
                    int CK, int N, int HW)
{
    int64_t total = (int64_t)B * N * HW;
    get_similarity_kernel<<<div_ceil(total, kBlockSize), kBlockSize>>>(
        memory_key, memory_shrinkage, query_key, query_selection, out, B, CK, N, HW);
    CUDA_CHECK(cudaGetLastError());
}

// ── top_k_softmax ───────────────────────────────────────────────────
// similarity: [B, N, HW] → affinity: [B, N, HW]
// 对每个 (b, hw) 位置，只保留 top_k 个最大值做 softmax，其余置零。
// usage: [B, N] 累加每个 token 的 affinity 权重之和（可选）。

__global__ void top_k_softmax_kernel(const float* similarity, float* affinity, float* usage, int B,
                                     int N, int HW, int top_k)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)B * HW;
    if (idx >= total)
        return;

    int hw_idx = idx % HW;
    int b = idx / HW;

    const float* sim_col = similarity + (int64_t)b * N * HW + hw_idx;

    // 找第 top_k 大的值作为阈值
    float threshold = -FLT_MAX;
    if (top_k < N)
    {
        float cur_threshold = FLT_MAX;
        for (int k = 0; k < top_k; ++k)
        {
            float next_max = -FLT_MAX;
            for (int n = 0; n < N; ++n)
            {
                float v = sim_col[n * HW];
                if (v < cur_threshold && v > next_max)
                    next_max = v;
                if (k == 0 && v > next_max)
                    next_max = v;
            }
            cur_threshold = next_max;
        }
        threshold = cur_threshold;
    }

    // 稳定 softmax
    float max_val = -FLT_MAX;
    for (int n = 0; n < N; ++n)
    {
        float v = sim_col[n * HW];
        if (v >= threshold && v > max_val)
            max_val = v;
    }

    float sum_exp = 0.0f;
    for (int n = 0; n < N; ++n)
    {
        float v = sim_col[n * HW];
        if (v >= threshold)
            sum_exp += expf(v - max_val);
    }

    float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;

    for (int n = 0; n < N; ++n)
    {
        float v = sim_col[n * HW];
        float a = 0.0f;
        if (v >= threshold)
            a = expf(v - max_val) * inv_sum;
        affinity[(int64_t)b * N * HW + n * HW + hw_idx] = a;
        if (usage != nullptr && a > 0.0f)
            atomicAdd(&usage[b * N + n], a);
    }
}

void top_k_softmax(const float* similarity, float* affinity, float* usage, int B, int N, int HW,
                   int top_k)
{
    int64_t total = (int64_t)B * HW;
    if (usage != nullptr)
        CUDA_CHECK(cudaMemset(usage, 0, B * N * sizeof(float)));
    top_k_softmax_kernel<<<div_ceil(total, kBlockSize), kBlockSize>>>(similarity, affinity, usage, B,
                                                                      N, HW, top_k);
    CUDA_CHECK(cudaGetLastError());
}

// ── one_hot_encode ──────────────────────────────────────────────────

__global__ void one_hot_encode_kernel(const int32_t* mask, const int32_t* objects, float* out,
                                      int num_obj, int hw)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    if (px >= hw)
        return;
    int32_t val = mask[px];
    for (int o = 0; o < num_obj; ++o)
        out[o * hw + px] = (val == objects[o]) ? 1.0f : 0.0f;
}

void one_hot_encode(const int32_t* mask, const int32_t* objects, float* out, int num_obj, int hw)
{
    one_hot_encode_kernel<<<div_ceil(hw, kBlockSize), kBlockSize>>>(mask, objects, out, num_obj, hw);
    CUDA_CHECK(cudaGetLastError());
}

// ── mask_merge_zero ─────────────────────────────────────────────────

__global__ void mask_merge_zero_kernel(float* pred_no_bg, const int32_t* input_mask,
                                       int num_existing, int hw)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    if (px >= hw)
        return;
    if (input_mask[px] > 0)
    {
        for (int o = 0; o < num_existing; ++o)
            pred_no_bg[o * hw + px] = 0.0f;
    }
}

void mask_merge_zero(float* pred_no_bg, const int32_t* input_mask, int num_existing, int hw)
{
    mask_merge_zero_kernel<<<div_ceil(hw, kBlockSize), kBlockSize>>>(pred_no_bg, input_mask,
                                                                     num_existing, hw);
    CUDA_CHECK(cudaGetLastError());
}

// ── fill_zero / copy_d2d ────────────────────────────────────────────

void fill_zero(float* ptr, int64_t n)
{
    CUDA_CHECK(cudaMemset(ptr, 0, n * sizeof(float)));
}

void copy_d2d(float* dst, const float* src, int64_t n)
{
    CUDA_CHECK(cudaMemcpy(dst, src, n * sizeof(float), cudaMemcpyDeviceToDevice));
}

// ── add_inplace ─────────────────────────────────────────────────────

__global__ void add_inplace_kernel(float* dst, const float* src, int64_t n)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;
    dst[idx] += src[idx];
}

void add_inplace(float* dst, const float* src, int64_t n)
{
    add_inplace_kernel<<<div_ceil(n, kBlockSize), kBlockSize>>>(dst, src, n);
    CUDA_CHECK(cudaGetLastError());
}

// ── bilinear_resize ─────────────────────────────────────────────────

__global__ void bilinear_resize_kernel(const float* src, int src_h, int src_w, float* dst,
                                       int dst_h, int dst_w, int channels)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = static_cast<int64_t>(channels) * dst_h * dst_w;
    if (idx >= total)
        return;

    int c = static_cast<int>(idx / (dst_h * dst_w));
    int rem = static_cast<int>(idx % (dst_h * dst_w));
    int dy = rem / dst_w;
    int dx = rem % dst_w;

    // 目标像素中心 → 源坐标
    float sy = (dy + 0.5f) * src_h / dst_h - 0.5f;
    float sx = (dx + 0.5f) * src_w / dst_w - 0.5f;

    int y0 = static_cast<int>(floorf(sy));
    int x0 = static_cast<int>(floorf(sx));
    int y1 = y0 + 1;
    int x1 = x0 + 1;

    float fy = sy - y0;
    float fx = sx - x0;

    // clamp
    y0 = max(0, min(y0, src_h - 1));
    y1 = max(0, min(y1, src_h - 1));
    x0 = max(0, min(x0, src_w - 1));
    x1 = max(0, min(x1, src_w - 1));

    const float* src_c = src + static_cast<int64_t>(c) * src_h * src_w;
    float v00 = src_c[y0 * src_w + x0];
    float v01 = src_c[y0 * src_w + x1];
    float v10 = src_c[y1 * src_w + x0];
    float v11 = src_c[y1 * src_w + x1];

    float val = (1.0f - fy) * ((1.0f - fx) * v00 + fx * v01) +
                fy * ((1.0f - fx) * v10 + fx * v11);
    dst[static_cast<int64_t>(c) * dst_h * dst_w + dy * dst_w + dx] = val;
}

void bilinear_resize(const float* src, int src_h, int src_w, float* dst, int dst_h, int dst_w,
                     int channels)
{
    int64_t total = static_cast<int64_t>(channels) * dst_h * dst_w;
    bilinear_resize_kernel<<<div_ceil(total, kBlockSize), kBlockSize>>>(
        src, src_h, src_w, dst, dst_h, dst_w, channels);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace cuda
}  // namespace cutie
