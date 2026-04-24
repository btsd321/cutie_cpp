#ifndef CUTIE_ORT_CORE_GPU_POSTPROCESS_H
#define CUTIE_ORT_CORE_GPU_POSTPROCESS_H

#include <array>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>
#include <opencv2/core/cuda.hpp>

#include "cutie/common/gpu_memory.h"
#include "cutie/types.h"

/// @file gpu_postprocess.h
/// GPU 后处理：unpad + argmax index mask 生成。
/// 消除 InferenceCore 中的 D2H + CPU unpad/resize/argmax 瓶颈。

namespace cutie
{
namespace cuda
{

/// GPU argmax kernel：prob [num_ch, H, W] → index_mask [H, W] int32。
/// channel 0 = background (输出 0)，channel i → objects[i-1]。
/// @param prob     GPU float32 概率图，CHW 连续布局
/// @param num_ch   通道数（num_objects + 1）
/// @param H, W     空间尺寸
/// @param objects  GPU int32 数组，长度 num_ch-1，ObjectId 映射
/// @param out      GPU int32 输出 [H, W]
/// @param stream   CUDA stream
void launch_argmax_index(const float* prob, int num_ch, int H, int W, const int32_t* objects,
                         int32_t* out, cudaStream_t stream = nullptr);

/// GPU unpad kernel：[C, padH, padW] → [C, H, W] 逐通道 D2D 拷贝。
/// @param src      GPU float32 源张量 [C, padH, padW]
/// @param dst      GPU float32 目标张量 [C, H, W]
/// @param C        通道数
/// @param src_h, src_w  源空间尺寸（含 pad）
/// @param dst_h, dst_w  目标空间尺寸（去 pad 后）
/// @param pad_top  顶部 pad 行数
/// @param pad_left 左侧 pad 列数
/// @param stream   CUDA stream
void launch_unpad(const float* src, float* dst, int C, int src_h, int src_w, int dst_h, int dst_w,
                  int pad_top, int pad_left, cudaStream_t stream = nullptr);

}  // namespace cuda

namespace ortcore
{

/// GPU unpad：从带 pad 的张量 [C, padH, padW] 中取出有效区域 [C, H, W]。
/// @param alloc  GPU 内存分配器
/// @param src    GPU Ort::Value [C, padH, padW] 或 [num_ch, padH, padW]
/// @param pad    [top, bottom, left, right] padding 尺寸
/// @param stream CUDA stream
/// @return GPU Ort::Value [C, H, W]
Ort::Value gpu_unpad(GpuMemoryAllocator& alloc, const Ort::Value& src,
                     const std::array<int, 4>& pad, cudaStream_t stream = nullptr);

/// GPU argmax：prob [num_obj+1, H, W] → index_mask GpuMat [H, W] CV_32SC1。
/// 逐像素在 channel 维做 argmax，channel 0 = background (输出 0)。
/// @param alloc    GPU 内存分配器
/// @param prob_with_bg  GPU Ort::Value [num_obj+1, H, W] float32
/// @param objects  CPU ObjectId 列表（长度 = num_obj）
/// @param stream   CUDA stream
/// @return GpuMat CV_32SC1 [H, W]，像素值为 ObjectId（bg=0）
cv::cuda::GpuMat gpu_prob_to_index_mask(GpuMemoryAllocator& alloc,
                                        const Ort::Value& prob_with_bg,
                                        const std::vector<ObjectId>& objects,
                                        cudaStream_t stream = nullptr);

}  // namespace ortcore
}  // namespace cutie

#endif  // CUTIE_ORT_CORE_GPU_POSTPROCESS_H
