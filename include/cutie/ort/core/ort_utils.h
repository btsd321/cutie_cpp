#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include "cutie/ort/core/ort_config.h"

namespace cutie
{
namespace ortcore
{

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