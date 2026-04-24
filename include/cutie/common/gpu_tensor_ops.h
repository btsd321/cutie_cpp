/**
 * @file gpu_tensor_ops.h
 * @brief High-level GPU tensor operation primitives.
 *
 * GPU implementations of tensor operations for memory management and inference.
 * All inputs and outputs are GPU Ort::Value tensors.
 * Corresponds to CPU versions in memory_utils.h.
 *
 * GPU 上的高层张量操作原语，对应 memory_utils.h 中的 CPU 版本。
 * 所有输入输出均为 GPU Ort::Value。
 */

#pragma once

#include <unordered_map>
#include <utility>
#include <vector>

#include "cutie/common/gpu_memory.h"
#include "cutie/types.h"

namespace cutie
{
namespace ortcore
{

/**
 * @brief Compute anisotropic L2 similarity (GPU version).
 *
 * 计算各向异性 L2 相似度，用于内存查询。
 *
 * @param alloc GPU memory allocator
 * @param mk Memory key [B, CK, N]
 * @param ms Memory shrinkage [B, 1, N]
 * @param qk Query key [B, CK, HW]
 * @param qe Query selection [B, CK, HW]
 * @return Similarity tensor [B, N, HW]
 */
Ort::Value gpu_get_similarity(GpuMemoryAllocator& alloc, const Ort::Value& mk,
                              const Ort::Value& ms, const Ort::Value& qk, const Ort::Value& qe);

/**
 * @brief Top-K sparse softmax (GPU version).
 *
 * 计算 Top-K 稀疏 Softmax，用于内存读取的注意力权重。
 *
 * @param alloc GPU memory allocator
 * @param similarity Similarity tensor [B, N, HW]
 * @param top_k Number of top elements to keep (-1 = all)
 * @param return_usage Whether to return usage statistics
 * @return Pair of (affinity [B, N, HW], usage [B, N] or empty)
 */
std::pair<Ort::Value, Ort::Value> gpu_do_softmax(GpuMemoryAllocator& alloc,
                                                  const Ort::Value& similarity, int top_k = -1,
                                                  bool return_usage = false);

/**
 * @brief Attention-weighted readout (GPU version, using cuBLAS).
 *
 * 使用注意力权重从内存中读取特征。
 *
 * @param alloc GPU memory allocator
 * @param affinity Attention weights [B, N, HW]
 * @param mv Memory values [B, CV, N]
 * @return Readout tensor [B, CV, HW] (= mv @ affinity)
 */
Ort::Value gpu_readout(GpuMemoryAllocator& alloc, const Ort::Value& affinity,
                       const Ort::Value& mv);

/**
 * @brief Multi-object aggregation to logits (GPU version).
 *
 * 将多个对象的概率图聚合为 logits。
 * 计算背景概率 bg = prod(1 - prob)，拼接 [bg, prob]，转为 logits（不做 softmax）。
 *
 * @param alloc GPU memory allocator
 * @param prob_no_bg Per-object probability [num_obj, H, W]
 * @return Logits tensor [num_obj+1, H, W]
 */
Ort::Value gpu_aggregate_logits(GpuMemoryAllocator& alloc, const Ort::Value& prob_no_bg);

/**
 * @brief Multi-object aggregation with softmax (GPU version).
 *
 * 聚合多个对象的概率图并应用 Softmax。
 *
 * @param alloc GPU memory allocator
 * @param prob_no_bg Per-object probability [num_obj, H, W]
 * @return Probability tensor [num_obj+1, H, W]
 */
Ort::Value gpu_aggregate(GpuMemoryAllocator& alloc, const Ort::Value& prob_no_bg);

/**
 * @brief Softmax along channel dimension (GPU version).
 *
 * 沿通道维度应用 Softmax。
 *
 * @param alloc GPU memory allocator
 * @param logits Logits tensor [C, H, W]
 * @return Probability tensor [C, H, W]
 */
Ort::Value gpu_softmax_channels(GpuMemoryAllocator& alloc, const Ort::Value& logits);

/**
 * @brief Sigmoid activation (GPU version).
 *
 * 应用 Sigmoid 激活函数。
 *
 * @param alloc GPU memory allocator
 * @param x Input tensor [N]
 * @return Output tensor [N]
 */
Ort::Value gpu_sigmoid(GpuMemoryAllocator& alloc, const Ort::Value& x);

/**
 * @brief Flatten spatial dimensions.
 *
 * 展平空间维度：[B, C, H, W] → [B, C, H*W]。
 *
 * @param alloc GPU memory allocator
 * @param tensor_4d Input tensor [B, C, H, W]
 * @return Flattened tensor [B, C, H*W]
 */
Ort::Value gpu_flatten_spatial(GpuMemoryAllocator& alloc, const Ort::Value& tensor_4d);

/**
 * @brief Stack multiple tensors along specified dimension.
 *
 * 沿指定维度堆叠多个张量。
 * 例如: 多个 [B, C, H, W] 沿 dim=1 → [B, N, C, H, W]。
 *
 * @param alloc GPU memory allocator
 * @param tensors Vector of tensor pointers
 * @param dim Stacking dimension
 * @return Stacked tensor
 */
Ort::Value gpu_stack(GpuMemoryAllocator& alloc, const std::vector<Ort::Value*>& tensors, int dim);

/**
 * @brief Split tensor along specified dimension.
 *
 * 沿指定维度拆分张量为多个张量。
 * 例如: [B, N, C, H, W] 沿 dim=1 → N 个 [B, C, H, W]。
 *
 * @param alloc GPU memory allocator
 * @param tensor Input tensor
 * @param dim Split dimension
 * @return Vector of split tensors
 */
std::vector<Ort::Value> gpu_split(GpuMemoryAllocator& alloc, const Ort::Value& tensor, int dim);

/**
 * @brief 4D matrix readout (GPU version).
 *
 * 4D 矩阵读取: affinity [B, N, HW] @ mv [B, num_obj, CV, N] → [B, num_obj, CV, HW]。
 * 先 reshape 为 3D 再调用 gpu_readout，然后 reshape 回来。
 *
 * @param alloc GPU memory allocator
 * @param affinity Attention weights [B, N, HW]
 * @param mv_4d Memory values [B, num_obj, CV, N]
 * @return Readout tensor [B, num_obj, CV, HW]
 */
Ort::Value gpu_readout_4d(GpuMemoryAllocator& alloc, const Ort::Value& affinity,
                          const Ort::Value& mv_4d);

/**
 * @brief Get size of specified dimension from tensor shape.
 *
 * 从 Ort::Value 获取形状中指定维度的大小。
 *
 * @param t Input tensor
 * @param axis Dimension index
 * @return Size of dimension
 */
inline int64_t dim_size(const Ort::Value& t, int axis)
{
    return t.GetTensorTypeAndShapeInfo().GetShape()[axis];
}

}  // namespace ortcore
}  // namespace cutie
