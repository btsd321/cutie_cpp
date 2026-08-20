#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include "cutie/ort/core/ort_config.h"

namespace cutie
{
namespace ortcore
{

/**
 * @brief 获取进程级唯一的 ONNX Runtime 环境（Ort::Env），永生单例。
 *
 * Ort::Env 的析构会触发已加载 CUDA execution provider 的卸载。若每个推理实例
 * 各自创建/销毁 Env，在同一个进程内反复创建/销毁 VideoSegmenter（约 3 次以上）
 * 会在 provider 卸载阶段段错误（exit 139），与是否调用 close() 无关。
 *
 * 此处改为进程内共享同一个 Env，且**永不析构**（堆分配、故意泄漏）：
 * - provider 全程只 dlopen 加载一次、运行期永不 dlclose 卸载；
 * - 进程退出时由 OS 直接回收内存与已加载的 provider .so，不执行 ORT 析构逻辑，
 *   从而同时规避运行期与退出期的 provider 卸载段错误。
 *
 * 多个并存的推理实例共用同一 Env 创建各自 Session，是 ORT 官方推荐用法。
 *
 * @return 对永生 Ort::Env 的引用。
 */
Ort::Env& ort_global_env();

/// Create an Ort::Value tensor from a contiguous float buffer.
Ort::Value create_tensor(const float* data, const std::vector<int64_t>& shape,
                         const Ort::MemoryInfo& memory_info);

/// Create an Ort::Value tensor from a cv::Mat blob (assumed NCHW float32).
Ort::Value mat_to_tensor(const cv::Mat& blob, const Ort::MemoryInfo& memory_info);

/// Extract tensor shape from an Ort::Value.
std::vector<int64_t> get_tensor_shape(const Ort::Value& tensor);

/// Total element count from shape.
int64_t shape_numel(const std::vector<int64_t>& shape);

/// Copy an Ort::Value tensor (deep copy).
Ort::Value clone_tensor(const Ort::Value& src, const Ort::MemoryInfo& memory_info);

/// Create a zero-filled float tensor.
Ort::Value zeros(const std::vector<int64_t>& shape, const Ort::MemoryInfo& memory_info);

/// Concatenate tensors along a given dim.
Ort::Value concat_tensors(const std::vector<Ort::Value>& tensors, int dim,
                          const Ort::MemoryInfo& memory_info);

/// Convert an Ort::Value float tensor to a multi-dimensional cv::Mat.
/// The returned Mat is contiguous with shape matching the tensor dimensions.
cv::Mat tensor_to_mat(const Ort::Value& tensor);

/// Convert a multi-dimensional cv::Mat (float32) to an Ort::Value.
/// Handles arbitrary dimensions, not just NCHW blobs.
Ort::Value mat_to_tensor_nd(const cv::Mat& mat, const Ort::MemoryInfo& memory_info);

/// Zero-pad tensor along axis `axis` so that dim[axis] == target_size.
/// If dim[axis] >= target_size the tensor is returned unchanged (clone).
Ort::Value pad_tensor_dim(const Ort::Value& src, int axis, int64_t target_size);

/// Slice tensor along axis `axis` keeping only indices [0, actual_size).
/// If actual_size >= dim[axis] the tensor is returned unchanged (clone).
Ort::Value slice_tensor_dim(const Ort::Value& src, int axis, int64_t actual_size);

}  // namespace ortcore
}  // namespace cutie