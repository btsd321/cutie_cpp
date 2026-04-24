#pragma once

#include <array>
#include <cstdint>
#include <utility>

#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

#include "cutie/common/gpu_buffer.h"
#include "cutie/common/gpu_memory.h"

/// @file gpu_image_preprocess.h
/// 融合 GPU 图像预处理：单 CUDA kernel 完成 bilinear resize + pad + BGR→RGB
/// + /255 + ImageNet normalize + HWC→CHW。替代 GpuMemoryAllocator::preprocess_image_gpu
/// 中多步 OpenCV CUDA 调用。

namespace cutie
{
namespace cuda
{

/// 融合预处理 kernel 参数
struct ImagePreprocessParams
{
    int src_w, src_h;           ///< 原始图像尺寸
    int src_pitch;              ///< 源图像行步长（字节，GpuMat.step）
    int resize_w, resize_h;     ///< 缩放后尺寸（pad 前）
    int dst_w, dst_h;           ///< 最终尺寸（含 pad）
    int pad_left, pad_top;      ///< pad 偏移
    float mean[3];              ///< ImageNet 均值 (RGB 顺序)
    float inv_std[3];           ///< ImageNet 标准差的倒数 (RGB 顺序)
};

/// 融合预处理 CUDA kernel：一次完成 resize + pad + BGR→RGB + normalize + HWC→CHW。
/// @param src_bgr   源图像 GPU 指针 (uint8, HWC, BGR)
/// @param dst_chw   目标 CHW float32 (GPU)，布局 [3, dst_h, dst_w]
/// @param p         预处理参数
/// @param stream    CUDA stream
void launch_image_preprocess(const uint8_t* src_bgr, float* dst_chw,
                             const ImagePreprocessParams& p, cudaStream_t stream = nullptr);

}  // namespace cuda

namespace ortcore
{

/// 融合 GPU 图像预处理器。
/// 单 CUDA kernel 完成完整预处理流程：
///   bilinear resize → letterbox pad → BGR→RGB → /255 → ImageNet norm → HWC→CHW
///
/// 支持两种输入：
///   - cv::cuda::GpuMat BGR (零拷贝，跳过 upload)
///   - cv::Mat BGR (自动 upload 到持久化 GpuBuffer)
class GpuImagePreprocessor
{
public:
    GpuImagePreprocessor() = default;

    /// 从 GPU 图像预处理（零拷贝输入）。
    /// @param gpu_bgr   GPU 上的 BGR uint8 图像
    /// @param target_h  缩放目标高度（pad 前）
    /// @param target_w  缩放目标宽度（pad 前）
    /// @param alloc     GPU 内存分配器
    /// @param divisor   pad 到此值的倍数
    /// @param stream    CUDA stream
    /// @return (GPU Ort::Value [1,3,padH,padW], pad [top,bottom,left,right])
    std::pair<Ort::Value, std::array<int, 4>> preprocess(const cv::cuda::GpuMat& gpu_bgr,
                                                         int target_h, int target_w,
                                                         GpuMemoryAllocator& alloc,
                                                         int divisor = 16,
                                                         cudaStream_t stream = nullptr);

    /// 从 CPU 图像预处理（自动上传到 GPU）。
    /// @param bgr_image CPU 上的 BGR uint8 图像
    /// @param target_h  缩放目标高度（pad 前）
    /// @param target_w  缩放目标宽度（pad 前）
    /// @param alloc     GPU 内存分配器
    /// @param divisor   pad 到此值的倍数
    /// @param stream    CUDA stream
    /// @return (GPU Ort::Value [1,3,padH,padW], pad [top,bottom,left,right])
    std::pair<Ort::Value, std::array<int, 4>> preprocess(const cv::Mat& bgr_image, int target_h,
                                                         int target_w,
                                                         GpuMemoryAllocator& alloc,
                                                         int divisor = 16,
                                                         cudaStream_t stream = nullptr);

private:
    GpuBuffer upload_buf_;  ///< 持久化 GPU 缓冲，用于 CPU→GPU 图像上传

    /// ImageNet 归一化常量 (RGB 顺序)
    static constexpr float kMean[3] = {0.485f, 0.456f, 0.406f};
    static constexpr float kStd[3] = {0.229f, 0.224f, 0.225f};

    /// 计算 pad 和输出尺寸，填充 kernel 参数。
    /// @return (padded_h, padded_w, pad[top,bottom,left,right])
    std::tuple<int, int, std::array<int, 4>> compute_params(int src_h, int src_w, int target_h,
                                                            int target_w, int divisor) const;
};

}  // namespace ortcore
}  // namespace cutie

