#ifndef CUTIE_ORT_CORE_GPU_TENSOR_OPS_H
#define CUTIE_ORT_CORE_GPU_TENSOR_OPS_H

#include <unordered_map>
#include <utility>
#include <vector>

#include "cutie/ort/core/gpu_memory.h"
#include "cutie/types.h"

/// @file gpu_tensor_ops.h
/// GPU 上的高层张量操作原语，对应 memory_utils.h 中的 CPU 版本。
/// 所有输入输出均为 GPU Ort::Value。

namespace cutie
{
namespace ortcore
{

/// 各向异性 L2 相似度（GPU 版本）。
/// mk: [B, CK, N], ms: [B, 1, N], qk: [B, CK, HW], qe: [B, CK, HW]
/// 返回: [B, N, HW]
Ort::Value gpu_get_similarity(GpuMemoryAllocator& alloc, const Ort::Value& mk,
                              const Ort::Value& ms, const Ort::Value& qk, const Ort::Value& qe);

/// Top-K 稀疏 Softmax（GPU 版本）。
/// similarity: [B, N, HW]
/// 返回: (affinity [B, N, HW], usage [B, N]（可选，return_usage=false 时为空）)
std::pair<Ort::Value, Ort::Value> gpu_do_softmax(GpuMemoryAllocator& alloc,
                                                  const Ort::Value& similarity, int top_k = -1,
                                                  bool return_usage = false);

/// 注意力加权读取（GPU 版本，使用 cuBLAS）。
/// affinity: [B, N, HW], mv: [B, CV, N]
/// 返回: [B, CV, HW]  (= mv @ affinity)
Ort::Value gpu_readout(GpuMemoryAllocator& alloc, const Ort::Value& affinity,
                       const Ort::Value& mv);

/// 多对象聚合（GPU 版本）：prob_no_bg [num_obj, H, W] → logits [num_obj+1, H, W]
/// 计算 bg = prod(1 - prob)，拼接 [bg, prob]，转为 logits（不做 softmax）
Ort::Value gpu_aggregate_logits(GpuMemoryAllocator& alloc, const Ort::Value& prob_no_bg);

/// 多对象聚合 + Softmax（GPU 版本）。
/// prob_no_bg: [num_obj, H, W]  →  [num_obj+1, H, W]
Ort::Value gpu_aggregate(GpuMemoryAllocator& alloc, const Ort::Value& prob_no_bg);

/// Softmax 沿通道维度（GPU 版本）：logits [C, H, W] → prob [C, H, W]
Ort::Value gpu_softmax_channels(GpuMemoryAllocator& alloc, const Ort::Value& logits);

/// Sigmoid 激活（GPU 版本）。
/// x: [N] → out: [N]，就地或新分配。
Ort::Value gpu_sigmoid(GpuMemoryAllocator& alloc, const Ort::Value& x);

/// 空间维度展平: [B, C, H, W] → [B, C, H*W]
Ort::Value gpu_flatten_spatial(GpuMemoryAllocator& alloc, const Ort::Value& tensor_4d);

/// 多个 Ort::Value 沿指定维度堆叠。
/// 例如: 多个 [B, C, H, W] 沿 dim=1 → [B, N, C, H, W]
Ort::Value gpu_stack(GpuMemoryAllocator& alloc, const std::vector<Ort::Value*>& tensors, int dim);

/// 沿指定维度拆分为多个 Ort::Value。
/// 例如: [B, N, C, H, W] 沿 dim=1 → N 个 [B, C, H, W]
std::vector<Ort::Value> gpu_split(GpuMemoryAllocator& alloc, const Ort::Value& tensor, int dim);

/// 4D 矩阵读取: affinity [B, N, HW] @ mv [B, num_obj, CV, N] → [B, num_obj, CV, HW]
/// 先 reshape 为 3D 再调用 gpu_readout，然后 reshape 回来。
Ort::Value gpu_readout_4d(GpuMemoryAllocator& alloc, const Ort::Value& affinity,
                          const Ort::Value& mv_4d);

/// 从 Ort::Value 获取形状中指定维度的大小（便捷函数）。
inline int64_t dim_size(const Ort::Value& t, int axis)
{
    return t.GetTensorTypeAndShapeInfo().GetShape()[axis];
}

}  // namespace ortcore
}  // namespace cutie

#endif  // CUTIE_ORT_CORE_GPU_TENSOR_OPS_H
