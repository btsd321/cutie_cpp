#ifndef CUTIE_ORT_CORE_GPU_MEMORY_H
#define CUTIE_ORT_CORE_GPU_MEMORY_H

#include <cstdint>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

#include "cutie/ort/core/ort_config.h"

/// @file gpu_memory.h
/// GPU 内存分配器：管理 Ort::Value 在 GPU 上的创建、上传、下载，
/// 以及与 cv::cuda::GpuMat 的零拷贝转换。

namespace cutie
{
namespace ortcore
{

class GpuMemoryAllocator
{
public:
    /// 构造时指定 CUDA 设备 ID。
    explicit GpuMemoryAllocator(int device_id);
    ~GpuMemoryAllocator();

    GpuMemoryAllocator(const GpuMemoryAllocator&) = delete;
    GpuMemoryAllocator& operator=(const GpuMemoryAllocator&) = delete;
    GpuMemoryAllocator(GpuMemoryAllocator&&) noexcept;
    GpuMemoryAllocator& operator=(GpuMemoryAllocator&&) noexcept;

    // ── 基本分配 ────────────────────────────────────────────────────

    /// 在 GPU 上分配一个 float32 张量（未初始化）。
    Ort::Value allocate(const std::vector<int64_t>& shape);

    /// 在 GPU 上分配一个全零 float32 张量。
    Ort::Value zeros(const std::vector<int64_t>& shape);

    // ── CPU ↔ GPU 传输 ──────────────────────────────────────────────

    /// 从 CPU cv::Mat（float32，多维）上传到 GPU Ort::Value。
    Ort::Value upload(const cv::Mat& mat);

    /// 从 GPU Ort::Value 下载到 CPU cv::Mat（float32，多维）。
    cv::Mat download(const Ort::Value& tensor);

    // ── GPU Ort::Value 操作 ─────────────────────────────────────────

    /// 深拷贝一个 GPU 张量。
    Ort::Value clone(const Ort::Value& src);

    /// 沿指定维度拼接两个 GPU 张量（相当于 torch.cat）。
    Ort::Value concat(const Ort::Value& a, const Ort::Value& b, int dim);

    /// 沿最后维度切片：取 [offset, offset+length) 范围。
    Ort::Value slice_last(const Ort::Value& src, int64_t offset, int64_t length);

    /// 沿指定维度 pad 到 target_size（零填充）。
    /// 如果 dim[axis] >= target_size，返回 clone。
    Ort::Value pad_dim(const Ort::Value& src, int axis, int64_t target_size);

    /// 沿指定维度 slice 到 actual_size。
    /// 如果 dim[axis] <= actual_size，返回 clone。
    Ort::Value slice_dim(const Ort::Value& src, int axis, int64_t actual_size);

    // ── cv::cuda::GpuMat ↔ Ort::Value ─────────────────────────────

    /// 将 GpuMat 包装为 Ort::Value（零拷贝，共享 GPU 内存）。
    /// 注意：GpuMat 的生命周期必须超过返回的 Ort::Value。
    /// shape 指定张量形状（元素总数必须匹配 GpuMat）。
    Ort::Value wrap_gpumat(const cv::cuda::GpuMat& gpu_mat,
                           const std::vector<int64_t>& shape);

    /// 将 Ort::Value 的 GPU 指针包装为 GpuMat（零拷贝）。
    /// 注意：Ort::Value 的生命周期必须超过返回的 GpuMat。
    cv::cuda::GpuMat wrap_as_gpumat(const Ort::Value& tensor, int rows, int cols, int cv_type);

    // ── GPU resize ───────────────────────────────────────────────────

    /// 对多通道张量 [C, H, W] 逐通道 GPU resize。
    /// 返回: [C, target_h, target_w]
    Ort::Value resize_channels(const Ort::Value& src, int target_h, int target_w,
                               int interpolation = 1 /*cv::INTER_LINEAR*/);

    // ── 图像预处理（GPU） ──────────────────────────────────────────

    /// GPU 图像预处理：BGR uint8 → RGB float32 [1, 3, H, W] ImageNet 归一化 + pad。
    /// 1. 上传到 GPU
    /// 2. cv::cuda::resize 到 target_h × target_w（如果需要）
    /// 3. BGR→RGB + float32 归一化 + ImageNet normalize
    /// 4. pad 到 divisor 的倍数
    /// 返回: (GPU Ort::Value [1, 3, padH, padW], pad [top, bottom, left, right])
    std::pair<Ort::Value, std::array<int, 4>> preprocess_image_gpu(
        const cv::Mat& bgr_image, int target_h, int target_w, int divisor = 16);

    // ── 访问器 ──────────────────────────────────────────────────────

    const Ort::MemoryInfo& memory_info() const { return gpu_memory_info_; }
    int device_id() const { return device_id_; }

    // ── 静态工具 ────────────────────────────────────────────────────

    /// 获取张量的形状。
    static std::vector<int64_t> shape(const Ort::Value& tensor);

    /// 计算形状中的元素总数。
    static int64_t numel(const std::vector<int64_t>& shape);

    /// 获取 GPU 张量的 float* 数据指针。
    static float* data_ptr(Ort::Value& tensor);
    static const float* data_ptr(const Ort::Value& tensor);

private:
    int device_id_;
    Ort::MemoryInfo gpu_memory_info_;

    /// 计算指定维度的 outer 和 inner 步幅。
    static void compute_strides(const std::vector<int64_t>& shape, int axis, int64_t& outer,
                                int64_t& inner);
};

}  // namespace ortcore
}  // namespace cutie

#endif  // CUTIE_ORT_CORE_GPU_MEMORY_H
