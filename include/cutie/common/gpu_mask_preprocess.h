#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>
#include <opencv2/core/cuda.hpp>

#include "cutie/common/gpu_memory.h"
#include "cutie/types.h"

/// @file gpu_mask_preprocess.h
/// GPU mask 预处理：nearest resize + pad + one_hot_encode + aggregate。
/// 替代 InferenceCore::step 中 mask 分支的全部 CPU 代码 + 手动 cudaMalloc。

namespace cutie
{
namespace cuda
{

/// GPU mask nearest resize：int32 index mask。
/// cv::cuda::resize 不支持 int32 类型，需自写 kernel。
/// @param src      GPU int32 源 mask [src_h, src_w]
/// @param src_h, src_w  源尺寸
/// @param dst      GPU int32 目标 mask [dst_h, dst_w]
/// @param dst_h, dst_w  目标尺寸
/// @param stream   CUDA stream
void launch_resize_mask_nearest(const int32_t* src, int src_h, int src_w, int32_t* dst, int dst_h,
                                int dst_w, cudaStream_t stream = nullptr);

/// GPU mask pad：int32 mask → 扩展尺寸，填充 0。
/// @param src      GPU int32 源 mask [src_h, src_w]
/// @param src_h, src_w  源尺寸
/// @param dst      GPU int32 目标 mask [dst_h, dst_w]
/// @param dst_h, dst_w  目标尺寸（含 pad）
/// @param pad_top  顶部 pad 行数
/// @param pad_left 左侧 pad 列数
/// @param stream   CUDA stream
void launch_pad_mask(const int32_t* src, int src_h, int src_w, int32_t* dst, int dst_h, int dst_w,
                     int pad_top, int pad_left, cudaStream_t stream = nullptr);

}  // namespace cuda

namespace ortcore
{

/// GPU mask nearest resize：GpuMat CV_32SC1 → GpuMat CV_32SC1。
/// @param mask      GPU index mask [orig_H, orig_W]
/// @param target_h  目标高度
/// @param target_w  目标宽度
/// @param stream    CUDA stream
/// @return GpuMat CV_32SC1 [target_h, target_w]
cv::cuda::GpuMat gpu_resize_mask_nearest(const cv::cuda::GpuMat& mask, int target_h, int target_w,
                                         cudaStream_t stream = nullptr);

/// GPU mask pad：GpuMat CV_32SC1 → GpuMat CV_32SC1（填充 0）。
/// @param mask  GPU index mask [H, W]
/// @param pad   [top, bottom, left, right]
/// @param stream CUDA stream
/// @return GpuMat CV_32SC1 [H+top+bottom, W+left+right]
cv::cuda::GpuMat gpu_pad_mask(const cv::cuda::GpuMat& mask, const std::array<int, 4>& pad,
                              cudaStream_t stream = nullptr);

/// 完整 GPU mask 预处理流程：resize + pad + upload objects + one_hot + aggregate。
/// 替代 InferenceCore::step 中 mask 分支的全部 CPU 代码。
/// @param alloc    GPU 内存分配器
/// @param gpu_mask GPU index mask (CV_32SC1, 原图尺寸)
/// @param objects  此帧 mask 中的 ObjectId 列表
/// @param target_h 缩放目标高度
/// @param target_w 缩放目标宽度
/// @param pad      [top, bottom, left, right]
/// @param total_num_objects 当前所有已知对象总数（决定输出通道数）
/// @param stream   CUDA stream
/// @return GPU Ort::Value [total_num_objects+1, padH, padW] float32 概率图
Ort::Value gpu_preprocess_mask(GpuMemoryAllocator& alloc, const cv::cuda::GpuMat& gpu_mask,
                               const std::vector<ObjectId>& objects, int target_h, int target_w,
                               const std::array<int, 4>& pad, int total_num_objects,
                               cudaStream_t stream = nullptr);

}  // namespace ortcore
}  // namespace cutie

