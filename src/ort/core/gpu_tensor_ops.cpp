/**
 * @file gpu_tensor_ops.cpp
 * @brief GPU tensor operation implementations.
 *
 * Implements high-level GPU tensor operations for memory management and inference,
 * including similarity computation, softmax, readout, aggregation, and stacking.
 * All operations use CUDA kernels and cuBLAS for efficient GPU computation.
 */

#include "cutie/ort/core/gpu_tensor_ops.h"
#include "cutie/ort/core/cuda_kernels.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstring>
#include <numeric>
#include <stdexcept>

namespace cutie
{
namespace ortcore
{

using GA = GpuMemoryAllocator;

// ── gpu_get_similarity ──────────────────────────────────────────────
// 计算各向异性 L2 相似度，用于内存查询中的特征匹配。
{
    auto mk_shape = GA::shape(mk);
    auto qk_shape = GA::shape(qk);
    int B = static_cast<int>(mk_shape[0]);
    int CK = static_cast<int>(mk_shape[1]);
    int N = static_cast<int>(mk_shape[2]);
    int HW = static_cast<int>(qk_shape[2]);

    auto result = alloc.allocate({B, N, HW});

    cuda::get_similarity(GA::data_ptr(mk), GA::data_ptr(ms), GA::data_ptr(qk), GA::data_ptr(qe),
                         GA::data_ptr(result), B, CK, N, HW);

    return result;
}

// ── gpu_do_softmax ──────────────────────────────────────────────────

std::pair<Ort::Value, Ort::Value> gpu_do_softmax(GA& alloc, const Ort::Value& similarity,
                                                  int top_k, bool return_usage)
{
    auto shape = GA::shape(similarity);
    int B = static_cast<int>(shape[0]);
    int N = static_cast<int>(shape[1]);
    int HW = static_cast<int>(shape[2]);

    auto affinity = alloc.allocate({B, N, HW});

    Ort::Value usage{nullptr};
    float* usage_ptr = nullptr;
    if (return_usage)
    {
        usage = alloc.allocate({B, N});
        usage_ptr = GA::data_ptr(usage);
    }

    int actual_top_k = (top_k > 0 && top_k < N) ? top_k : N;
    cuda::top_k_softmax(GA::data_ptr(similarity), GA::data_ptr(affinity), usage_ptr, B, N, HW,
                        actual_top_k);

    return {std::move(affinity), std::move(usage)};
}

// ── gpu_readout (cuBLAS) ────────────────────────────────────────────

Ort::Value gpu_readout(GA& alloc, const Ort::Value& affinity, const Ort::Value& mv)
{
    // affinity: [B, N, HW], mv: [B, CV, N]
    // result: [B, CV, HW] = mv @ affinity  (batch matmul)
    auto aff_shape = GA::shape(affinity);
    auto mv_shape = GA::shape(mv);

    int B = static_cast<int>(mv_shape[0]);
    int CV = static_cast<int>(mv_shape[1]);
    int N = static_cast<int>(mv_shape[2]);
    int HW = static_cast<int>(aff_shape[2]);

    auto result = alloc.allocate({B, CV, HW});

    // cuBLAS: C = alpha * A * B + beta * C
    // A = mv[b]:  CV x N  (row-major → cublas sees as N x CV col-major)
    // B = aff[b]: N x HW  (row-major → cublas sees as HW x N col-major)
    // C = out[b]: CV x HW (row-major → cublas sees as HW x CV col-major)
    //
    // In col-major terms: C^T = B^T * A^T → cublas: C_col = B_col * A_col
    // cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, HW, CV, N, ...)
    //   A_col = aff (HW x N), B_col = mv (N x CV), C_col = out (HW x CV)

    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f;
    float beta = 0.0f;

    const float* aff_ptr = GA::data_ptr(affinity);
    const float* mv_ptr = GA::data_ptr(mv);
    float* out_ptr = GA::data_ptr(result);

    for (int b = 0; b < B; ++b)
    {
        // mv[b]: CV rows, N cols (row-major) → col-major: N rows, CV cols
        // aff[b]: N rows, HW cols (row-major) → col-major: HW rows, N cols
        // out[b]: CV rows, HW cols (row-major) → col-major: HW rows, CV cols
        // C_col(HW,CV) = A_col(HW,N) * B_col(N,CV)
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    HW, CV, N,
                    &alpha,
                    aff_ptr + b * N * HW, HW,        // A_col: aff[b], lda=HW
                    mv_ptr + b * CV * N, N,            // B_col: mv[b], ldb=N
                    &beta,
                    out_ptr + b * CV * HW, HW);        // C_col: out[b], ldc=HW
    }

    cublasDestroy(handle);
    return result;
}

// ── gpu_readout_4d ──────────────────────────────────────────────────

Ort::Value gpu_readout_4d(GA& alloc, const Ort::Value& affinity, const Ort::Value& mv_4d)
{
    // affinity: [B, N, HW], mv_4d: [B, num_obj, CV, N]
    // → reshape mv to [B, num_obj*CV, N], readout, reshape back
    auto mv_shape = GA::shape(mv_4d);
    int B = static_cast<int>(mv_shape[0]);
    int num_obj = static_cast<int>(mv_shape[1]);
    int CV = static_cast<int>(mv_shape[2]);
    int N = static_cast<int>(mv_shape[3]);
    int HW = static_cast<int>(GA::shape(affinity)[2]);

    // Reshape: [B, num_obj*CV, N] — 零拷贝 reshape（数据布局不变）
    auto mv_3d = Ort::Value::CreateTensor<float>(
        alloc.memory_info(),
        const_cast<float*>(GA::data_ptr(mv_4d)),
        B * num_obj * CV * N,
        std::vector<int64_t>{B, (int64_t)num_obj * CV, N}.data(), 3);

    auto out_3d = gpu_readout(alloc, affinity, mv_3d);

    // Reshape: [B, num_obj, CV, HW]
    auto out_4d = Ort::Value::CreateTensor<float>(
        alloc.memory_info(),
        GA::data_ptr(out_3d),
        B * num_obj * CV * HW,
        std::vector<int64_t>{B, num_obj, CV, HW}.data(), 4);

    // 这里 out_4d 共享 out_3d 的内存，需要保持 out_3d 存活。
    // 为安全起见，做一次 clone。
    return alloc.clone(out_4d);
}

// ── gpu_aggregate_logits ────────────────────────────────────────────

Ort::Value gpu_aggregate_logits(GA& alloc, const Ort::Value& prob_no_bg)
{
    auto shape = GA::shape(prob_no_bg);
    int num_obj = static_cast<int>(shape[0]);
    int H = static_cast<int>(shape[1]);
    int W = static_cast<int>(shape[2]);
    int HW = H * W;

    auto result = alloc.allocate({num_obj + 1, H, W});
    cuda::aggregate_logits(GA::data_ptr(prob_no_bg), GA::data_ptr(result), num_obj, HW);

    return result;
}

// ── gpu_aggregate ───────────────────────────────────────────────────

Ort::Value gpu_aggregate(GA& alloc, const Ort::Value& prob_no_bg)
{
    auto shape = GA::shape(prob_no_bg);
    int num_obj = static_cast<int>(shape[0]);
    int H = static_cast<int>(shape[1]);
    int W = static_cast<int>(shape[2]);
    int HW = H * W;

    auto result = alloc.allocate({num_obj + 1, H, W});
    cuda::aggregate_softmax(GA::data_ptr(prob_no_bg), GA::data_ptr(result), num_obj, HW);

    return result;
}

// ── gpu_softmax_channels ────────────────────────────────────────────

Ort::Value gpu_softmax_channels(GA& alloc, const Ort::Value& logits)
{
    auto shape = GA::shape(logits);
    int C = static_cast<int>(shape[0]);
    int H = static_cast<int>(shape[1]);
    int W = static_cast<int>(shape[2]);
    int HW = H * W;

    auto result = alloc.allocate(shape);
    cuda::softmax_channels(GA::data_ptr(logits), GA::data_ptr(result), C, HW);

    return result;
}

// ── gpu_sigmoid ─────────────────────────────────────────────────────

Ort::Value gpu_sigmoid(GA& alloc, const Ort::Value& x)
{
    auto shape = GA::shape(x);
    int64_t n = GA::numel(shape);
    auto result = alloc.allocate(shape);
    cuda::sigmoid(GA::data_ptr(x), GA::data_ptr(result), n);
    return result;
}

// ── gpu_flatten_spatial ─────────────────────────────────────────────

Ort::Value gpu_flatten_spatial(GA& alloc, const Ort::Value& tensor_4d)
{
    auto shape = GA::shape(tensor_4d);
    // [B, C, H, W] → [B, C, H*W]
    int64_t B = shape[0];
    int64_t C = shape[1];
    int64_t HW = shape[2] * shape[3];

    // 零拷贝 reshape：数据布局不变（连续内存）
    // 但 Ort::Value 不支持 view，需要 clone 到新 shape
    auto result = alloc.allocate({B, C, HW});
    cuda::copy_d2d(GA::data_ptr(result), GA::data_ptr(tensor_4d), B * C * HW);
    return result;
}

// ── gpu_stack ────────────────────────────────────────────────────────

Ort::Value gpu_stack(GA& alloc, const std::vector<Ort::Value*>& tensors, int dim)
{
    if (tensors.empty())
        throw std::runtime_error("gpu_stack: empty input");

    int N = static_cast<int>(tensors.size());
    auto first_shape = GA::shape(*tensors[0]);
    int ndim = static_cast<int>(first_shape.size());

    // 计算输出 shape：在 dim 位置插入 N
    std::vector<int64_t> out_shape;
    for (int i = 0; i < dim; ++i) out_shape.push_back(first_shape[i]);
    out_shape.push_back(N);
    for (int i = dim; i < ndim; ++i) out_shape.push_back(first_shape[i]);

    int64_t per_tensor = GA::numel(first_shape);

    // outer = 前 dim 维乘积
    int64_t outer = 1;
    for (int i = 0; i < dim; ++i) outer *= first_shape[i];
    // inner = dim 及之后维度乘积
    int64_t inner = per_tensor / outer;

    auto result = alloc.allocate(out_shape);
    float* dst = GA::data_ptr(result);

    for (int64_t o = 0; o < outer; ++o)
    {
        for (int n = 0; n < N; ++n)
        {
            cuda::copy_d2d(dst + (o * N + n) * inner,
                           GA::data_ptr(*tensors[n]) + o * inner, inner);
        }
    }

    return result;
}

// ── gpu_split ────────────────────────────────────────────────────────

std::vector<Ort::Value> gpu_split(GA& alloc, const Ort::Value& tensor, int dim)
{
    auto shape = GA::shape(tensor);
    int ndim = static_cast<int>(shape.size());
    int N = static_cast<int>(shape[dim]);

    // 输出 shape：去掉 dim 维度
    std::vector<int64_t> out_shape;
    for (int i = 0; i < ndim; ++i)
    {
        if (i != dim) out_shape.push_back(shape[i]);
    }

    int64_t per_slice = GA::numel(out_shape);
    int64_t outer = 1;
    for (int i = 0; i < dim; ++i) outer *= shape[i];
    int64_t inner = per_slice / outer;

    std::vector<Ort::Value> results;
    results.reserve(N);

    const float* src = GA::data_ptr(tensor);
    for (int n = 0; n < N; ++n)
    {
        auto slice = alloc.allocate(out_shape);
        float* dst = GA::data_ptr(slice);
        for (int64_t o = 0; o < outer; ++o)
        {
            cuda::copy_d2d(dst + o * inner, src + (o * N + n) * inner, inner);
        }
        results.push_back(std::move(slice));
    }

    return results;
}

}  // namespace ortcore
}  // namespace cutie
