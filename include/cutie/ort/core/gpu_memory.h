#ifndef CUTIE_ORT_CORE_GPU_MEMORY_H
#define CUTIE_ORT_CORE_GPU_MEMORY_H

#include <cstdint>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

#include "cutie/ort/core/ort_config.h"

/**
 * @file gpu_memory.h
 * @brief GPU memory allocator and tensor management.
 *
 * Manages GPU memory allocation, CPU↔GPU data transfer, and zero-copy
 * conversions between Ort::Value and cv::cuda::GpuMat. Provides high-level
 * tensor operations (concat, slice, resize, preprocess).
 *
 * GPU 内存分配器：管理 Ort::Value 在 GPU 上的创建、上传、下载，
 * 以及与 cv::cuda::GpuMat 的零拷贝转换。
 */

#ifndef CUTIE_ORT_CORE_GPU_MEMORY_H
#define CUTIE_ORT_CORE_GPU_MEMORY_H

#include <cstdint>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

#include "cutie/ort/core/ort_config.h"

namespace cutie
{
namespace ortcore
{

/**
 * @class GpuMemoryAllocator
 * @brief GPU memory allocator for Ort::Value tensors.
 *
 * Manages GPU memory allocation, CPU↔GPU transfers, and tensor operations.
 * Provides zero-copy conversions between Ort::Value and cv::cuda::GpuMat.
 * 所有张量操作都在 GPU 上执行，支持高效的内存管理和数据传输。
 */
class GpuMemoryAllocator
{
public:
    /**
     * @brief Construct allocator for specified CUDA device.
     * @param device_id CUDA device ID
     */
    explicit GpuMemoryAllocator(int device_id);
    ~GpuMemoryAllocator();

    GpuMemoryAllocator(const GpuMemoryAllocator&) = delete;
    GpuMemoryAllocator& operator=(const GpuMemoryAllocator&) = delete;
    GpuMemoryAllocator(GpuMemoryAllocator&&) noexcept;
    GpuMemoryAllocator& operator=(GpuMemoryAllocator&&) noexcept;

    // ── Basic Allocation ────────────────────────────────────────────

    /**
     * @brief Allocate uninitialized float32 tensor on GPU.
     *
     * 在 GPU 上分配一个 float32 张量（未初始化）。
     *
     * @param shape Tensor shape
     * @return GPU Ort::Value
     */
    Ort::Value allocate(const std::vector<int64_t>& shape);

    /**
     * @brief Allocate zero-filled float32 tensor on GPU.
     *
     * 在 GPU 上分配一个全零 float32 张量。
     *
     * @param shape Tensor shape
     * @return GPU Ort::Value (all zeros)
     */
    Ort::Value zeros(const std::vector<int64_t>& shape);

    // ── CPU ↔ GPU Transfer ──────────────────────────────────────────

    /**
     * @brief Upload CPU tensor to GPU.
     *
     * 从 CPU cv::Mat（float32，多维）上传到 GPU Ort::Value。
     *
     * @param mat CPU tensor (float32, any shape)
     * @return GPU Ort::Value
     */
    Ort::Value upload(const cv::Mat& mat);

    /**
     * @brief Download GPU tensor to CPU.
     *
     * 从 GPU Ort::Value 下载到 CPU cv::Mat（float32，多维）。
     *
     * @param tensor GPU Ort::Value
     * @return CPU tensor (float32, same shape)
     */
    cv::Mat download(const Ort::Value& tensor);

    // ── GPU Ort::Value Operations ───────────────────────────────────

    /**
     * @brief Deep copy GPU tensor.
     *
     * 深拷贝一个 GPU 张量。
     *
     * @param src Source tensor
     * @return Cloned tensor
     */
    Ort::Value clone(const Ort::Value& src);

    /**
     * @brief Concatenate two GPU tensors along dimension.
     *
     * 沿指定维度拼接两个 GPU 张量（相当于 torch.cat）。
     *
     * @param a First tensor
     * @param b Second tensor
     * @param dim Concatenation dimension
     * @return Concatenated tensor
     */
    Ort::Value concat(const Ort::Value& a, const Ort::Value& b, int dim);

    /**
     * @brief Slice GPU tensor along last dimension.
     *
     * 沿最后维度切片：取 [offset, offset+length) 范围。
     *
     * @param src Source tensor
     * @param offset Start index
     * @param length Number of elements
     * @return Sliced tensor
     */
    Ort::Value slice_last(const Ort::Value& src, int64_t offset, int64_t length);

    /**
     * @brief Pad dimension to target size (zero-fill).
     *
     * 沿指定维度 pad 到 target_size（零填充）。
     * 如果 dim[axis] >= target_size，返回 clone。
     *
     * @param src Source tensor
     * @param axis Dimension to pad
     * @param target_size Target size
     * @return Padded tensor
     */
    Ort::Value pad_dim(const Ort::Value& src, int axis, int64_t target_size);

    /**
     * @brief Slice dimension to actual size.
     *
     * 沿指定维度 slice 到 actual_size。
     * 如果 dim[axis] <= actual_size，返回 clone。
     *
     * @param src Source tensor
     * @param axis Dimension to slice
     * @param actual_size Target size
     * @return Sliced tensor
     */
    Ort::Value slice_dim(const Ort::Value& src, int axis, int64_t actual_size);

    // ── cv::cuda::GpuMat ↔ Ort::Value ──────────────────────────────

    /**
     * @brief Wrap GpuMat as Ort::Value (zero-copy).
     *
     * 将 GpuMat 包装为 Ort::Value（零拷贝，共享 GPU 内存）。
     * 注意：GpuMat 的生命周期必须超过返回的 Ort::Value。
     *
     * @param gpu_mat Source GpuMat
     * @param shape Tensor shape (total elements must match GpuMat)
     * @return GPU Ort::Value (shares memory with GpuMat)
     */
    Ort::Value wrap_gpumat(const cv::cuda::GpuMat& gpu_mat,
                           const std::vector<int64_t>& shape);

    /**
     * @brief Wrap Ort::Value as GpuMat (zero-copy).
     *
     * 将 Ort::Value 的 GPU 指针包装为 GpuMat（零拷贝）。
     * 注意：Ort::Value 的生命周期必须超过返回的 GpuMat。
     *
     * @param tensor Source GPU Ort::Value
     * @param rows Number of rows
     * @param cols Number of columns
     * @param cv_type OpenCV data type
     * @return GpuMat (shares memory with Ort::Value)
     */
    cv::cuda::GpuMat wrap_as_gpumat(const Ort::Value& tensor, int rows, int cols, int cv_type);

    // ── GPU Resize ──────────────────────────────────────────────────

    /**
     * @brief Resize multi-channel tensor per-channel.
     *
     * 对多通道张量 [C, H, W] 逐通道 GPU resize。
     *
     * @param src Source tensor [C, H, W]
     * @param target_h Target height
     * @param target_w Target width
     * @param interpolation Interpolation method (default cv::INTER_LINEAR)
     * @return Resized tensor [C, target_h, target_w]
     */
    Ort::Value resize_channels(const Ort::Value& src, int target_h, int target_w,
                               int interpolation = 1 /*cv::INTER_LINEAR*/);

    // ── Image Preprocessing (GPU) ───────────────────────────────────

    /**
     * @brief GPU image preprocessing pipeline.
     *
     * 完整的 GPU 图像预处理流程：
     * 1. 上传到 GPU
     * 2. cv::cuda::resize 到 target_h × target_w（如果需要）
     * 3. BGR→RGB + float32 归一化 + ImageNet normalize
     * 4. pad 到 divisor 的倍数
     *
     * @param bgr_image Input image (BGR, uint8, CPU)
     * @param target_h Target height
     * @param target_w Target width
     * @param divisor Padding divisor (default 16)
     * @return Pair of (GPU Ort::Value [1, 3, padH, padW], pad [top, bottom, left, right])
     */
    std::pair<Ort::Value, std::array<int, 4>> preprocess_image_gpu(
        const cv::Mat& bgr_image, int target_h, int target_w, int divisor = 16);

    // ── Accessors ───────────────────────────────────────────────────

    /**
     * @brief Get GPU memory info.
     * @return Reference to Ort::MemoryInfo
     */
    const Ort::MemoryInfo& memory_info() const { return gpu_memory_info_; }

    /**
     * @brief Get CUDA device ID.
     * @return Device ID
     */
    int device_id() const { return device_id_; }

    // ── Static Utilities ────────────────────────────────────────────

    /**
     * @brief Get tensor shape.
     *
     * 获取张量的形状。
     *
     * @param tensor Input tensor
     * @return Shape vector
     */
    static std::vector<int64_t> shape(const Ort::Value& tensor);

    /**
     * @brief Calculate total number of elements in shape.
     *
     * 计算形状中的元素总数。
     *
     * @param shape Shape vector
     * @return Total element count
     */
    static int64_t numel(const std::vector<int64_t>& shape);

    /**
     * @brief Get float pointer to GPU tensor data.
     *
     * 获取 GPU 张量的 float* 数据指针。
     *
     * @param tensor GPU Ort::Value
     * @return Mutable float pointer
     */
    static float* data_ptr(Ort::Value& tensor);

    /**
     * @brief Get const float pointer to GPU tensor data.
     * @param tensor GPU Ort::Value
     * @return Const float pointer
     */
    static const float* data_ptr(const Ort::Value& tensor);

private:
    int device_id_;  ///< CUDA device ID
    Ort::MemoryInfo gpu_memory_info_;  ///< GPU memory info

    /**
     * @brief Compute outer and inner strides for dimension.
     *
     * 计算指定维度的 outer 和 inner 步幅。
     *
     * @param shape Tensor shape
     * @param axis Target dimension
     * @param outer Output: outer stride
     * @param inner Output: inner stride
     */
    static void compute_strides(const std::vector<int64_t>& shape, int axis, int64_t& outer,
                                int64_t& inner);
};

}  // namespace ortcore
}  // namespace cutie

#endif  // CUTIE_ORT_CORE_GPU_MEMORY_H
