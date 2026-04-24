/**
 * @file gpu_memory.cpp
 * @brief GpuMemoryAllocator implementation.
 *
 * Implements GPU memory allocation, CPU↔GPU data transfer, and tensor operations.
 * Provides zero-copy conversions between Ort::Value and cv::cuda::GpuMat.
 * Includes image preprocessing and tensor manipulation on GPU.
 */

#include "cutie/common/gpu_memory.h"
#include "cutie/common/cuda_kernels.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <numeric>
#include <stdexcept>

#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

namespace cutie
{
namespace ortcore
{

// ── 构造 / 析构 ────────────────────────────────────────────────────
// 初始化 GPU 内存分配器，设置 CUDA 设备和内存信息。
    : device_id_(device_id),
      gpu_memory_info_(
          Ort::MemoryInfo("Cuda", OrtAllocatorType::OrtDeviceAllocator, device_id, OrtMemTypeDefault))
{
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string("GpuMemoryAllocator: cudaSetDevice failed: ") +
                                 cudaGetErrorString(err));
    }
}

GpuMemoryAllocator::~GpuMemoryAllocator() = default;

GpuMemoryAllocator::GpuMemoryAllocator(GpuMemoryAllocator&&) noexcept = default;
GpuMemoryAllocator& GpuMemoryAllocator::operator=(GpuMemoryAllocator&&) noexcept = default;

// ── 静态工具 ────────────────────────────────────────────────────────

std::vector<int64_t> GpuMemoryAllocator::shape(const Ort::Value& tensor)
{
    return tensor.GetTensorTypeAndShapeInfo().GetShape();
}

int64_t GpuMemoryAllocator::numel(const std::vector<int64_t>& s)
{
    if (s.empty())
        return 0;
    return std::accumulate(s.begin(), s.end(), int64_t(1), std::multiplies<int64_t>());
}

float* GpuMemoryAllocator::data_ptr(Ort::Value& tensor)
{
    return tensor.GetTensorMutableData<float>();
}

const float* GpuMemoryAllocator::data_ptr(const Ort::Value& tensor)
{
    return tensor.GetTensorData<float>();
}

void GpuMemoryAllocator::compute_strides(const std::vector<int64_t>& s, int axis, int64_t& outer,
                                          int64_t& inner)
{
    outer = 1;
    for (int i = 0; i < axis; ++i) outer *= s[i];
    inner = 1;
    for (int i = axis + 1; i < static_cast<int>(s.size()); ++i) inner *= s[i];
}

// ── 基本分配 ────────────────────────────────────────────────────────

Ort::Value GpuMemoryAllocator::allocate(const std::vector<int64_t>& s)
{
    int64_t total = numel(s);
    float* gpu_ptr = nullptr;
    cudaError_t err = cudaMalloc(&gpu_ptr, total * sizeof(float));
    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string("GpuMemoryAllocator::allocate cudaMalloc failed: ") +
                                 cudaGetErrorString(err));
    }
    // 用外部指针创建 Ort::Value（不拥有内存，需要手动管理）
    // 注意：这里使用 ORT 的外部内存接口。
    // CreateTensor 通过 memory_info 告诉 ORT 数据在 GPU 上。
    return Ort::Value::CreateTensor<float>(gpu_memory_info_, gpu_ptr, total, s.data(), s.size());
}

Ort::Value GpuMemoryAllocator::zeros(const std::vector<int64_t>& s)
{
    auto tensor = allocate(s);
    cuda::fill_zero(data_ptr(tensor), numel(s));
    return tensor;
}

// ── CPU ↔ GPU 传输 ──────────────────────────────────────────────────

Ort::Value GpuMemoryAllocator::upload(const cv::Mat& mat)
{
    if (mat.empty())
        throw std::runtime_error("GpuMemoryAllocator::upload: empty input");
    if (mat.type() != CV_32FC1)
        throw std::runtime_error("GpuMemoryAllocator::upload: expected CV_32FC1");

    std::vector<int64_t> s;
    for (int i = 0; i < mat.dims; ++i)
        s.push_back(static_cast<int64_t>(mat.size[i]));

    int64_t total = numel(s);
    auto tensor = allocate(s);
    cudaError_t err = cudaMemcpy(data_ptr(tensor), mat.ptr<float>(), total * sizeof(float),
                                 cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string("GpuMemoryAllocator::upload cudaMemcpy failed: ") +
                                 cudaGetErrorString(err));
    }
    return tensor;
}

cv::Mat GpuMemoryAllocator::download(const Ort::Value& tensor)
{
    auto s = shape(tensor);
    int64_t total = numel(s);
    std::vector<int> cv_sizes(s.begin(), s.end());

    cv::Mat result(static_cast<int>(cv_sizes.size()), cv_sizes.data(), CV_32FC1);
    cudaError_t err = cudaMemcpy(result.ptr<float>(), data_ptr(tensor), total * sizeof(float),
                                 cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string("GpuMemoryAllocator::download cudaMemcpy failed: ") +
                                 cudaGetErrorString(err));
    }
    return result;
}

// ── GPU 张量操作 ────────────────────────────────────────────────────

Ort::Value GpuMemoryAllocator::clone(const Ort::Value& src)
{
    auto s = shape(src);
    int64_t total = numel(s);
    auto dst = allocate(s);
    cuda::copy_d2d(data_ptr(dst), data_ptr(src), total);
    return dst;
}

Ort::Value GpuMemoryAllocator::concat(const Ort::Value& a, const Ort::Value& b, int dim)
{
    auto sa = shape(a);
    auto sb = shape(b);
    int ndim = static_cast<int>(sa.size());
    if (dim < 0)
        dim += ndim;

    // 计算输出形状
    std::vector<int64_t> out_shape = sa;
    out_shape[dim] = sa[dim] + sb[dim];

    // 对于最后一维 concat，使用 CUDA kernel
    // 对于其他维度，转换为等效的最后一维操作
    int64_t outer = 1;
    for (int i = 0; i < dim; ++i) outer *= sa[i];
    int64_t inner_a = sa[dim];
    int64_t inner_b = sb[dim];
    for (int i = dim + 1; i < ndim; ++i)
    {
        inner_a *= sa[i];
        inner_b *= sb[i];
    }

    auto result = allocate(out_shape);
    cuda::concat_last_dim(data_ptr(a), inner_a, data_ptr(b), inner_b, data_ptr(result), outer);
    return result;
}

Ort::Value GpuMemoryAllocator::slice_last(const Ort::Value& src, int64_t offset, int64_t length)
{
    auto s = shape(src);
    int ndim = static_cast<int>(s.size());
    int64_t src_inner = s[ndim - 1];
    int64_t outer = numel(s) / src_inner;

    std::vector<int64_t> out_shape = s;
    out_shape[ndim - 1] = length;

    auto result = allocate(out_shape);
    cuda::slice_last_dim(data_ptr(src), src_inner, data_ptr(result), length, offset, outer);
    return result;
}

Ort::Value GpuMemoryAllocator::pad_dim(const Ort::Value& src, int axis, int64_t target_size)
{
    auto s = shape(src);
    int ndim = static_cast<int>(s.size());
    if (axis < 0)
        axis += ndim;

    if (s[axis] >= target_size)
        return clone(src);

    // 输出形状
    std::vector<int64_t> out_shape = s;
    out_shape[axis] = target_size;

    auto result = zeros(out_shape);

    // 将 src 数据复制到 result 的前 s[axis] 部分
    int64_t outer, inner;
    compute_strides(s, axis, outer, inner);

    int64_t src_stride = s[axis] * inner;
    int64_t dst_stride = target_size * inner;

    const float* sp = data_ptr(src);
    float* dp = data_ptr(result);

    for (int64_t o = 0; o < outer; ++o)
    {
        cuda::copy_d2d(dp + o * dst_stride, sp + o * src_stride, s[axis] * inner);
    }

    return result;
}

Ort::Value GpuMemoryAllocator::slice_dim(const Ort::Value& src, int axis, int64_t actual_size)
{
    auto s = shape(src);
    int ndim = static_cast<int>(s.size());
    if (axis < 0)
        axis += ndim;

    if (s[axis] <= actual_size)
        return clone(src);

    std::vector<int64_t> out_shape = s;
    out_shape[axis] = actual_size;

    auto result = allocate(out_shape);

    int64_t outer, inner;
    compute_strides(s, axis, outer, inner);

    int64_t src_stride = s[axis] * inner;
    int64_t dst_stride = actual_size * inner;

    const float* sp = data_ptr(src);
    float* dp = data_ptr(result);

    for (int64_t o = 0; o < outer; ++o)
    {
        cuda::copy_d2d(dp + o * dst_stride, sp + o * src_stride, actual_size * inner);
    }

    return result;
}

// ── cv::cuda::GpuMat ↔ Ort::Value ─────────────────────────────────

Ort::Value GpuMemoryAllocator::wrap_gpumat(const cv::cuda::GpuMat& gpu_mat,
                                            const std::vector<int64_t>& s)
{
    int64_t total = numel(s);
    int64_t mat_elems = static_cast<int64_t>(gpu_mat.rows) * gpu_mat.cols * gpu_mat.channels();
    if (total != mat_elems)
    {
        throw std::runtime_error("GpuMemoryAllocator::wrap_gpumat: shape/size mismatch");
    }

    float* gpu_ptr = reinterpret_cast<float*>(gpu_mat.data);
    return Ort::Value::CreateTensor<float>(gpu_memory_info_, gpu_ptr, total, s.data(), s.size());
}

cv::cuda::GpuMat GpuMemoryAllocator::wrap_as_gpumat(const Ort::Value& tensor, int rows, int cols,
                                                      int cv_type)
{
    // 注意：这里使用 const_cast 因为 GpuMat 需要非 const 指针。
    // 调用者需确保不通过 GpuMat 修改只读数据。
    float* ptr = const_cast<float*>(data_ptr(tensor));
    return cv::cuda::GpuMat(rows, cols, cv_type, ptr);
}

// ── GPU resize 多通道 ───────────────────────────────────────────────

Ort::Value GpuMemoryAllocator::resize_channels(const Ort::Value& src, int target_h, int target_w,
                                                int interpolation)
{
    auto s = shape(src);
    int C = static_cast<int>(s[0]);
    int H = static_cast<int>(s[1]);
    int W = static_cast<int>(s[2]);

    if (H == target_h && W == target_w)
        return clone(src);

    auto result = allocate({C, target_h, target_w});

    // INTER_LINEAR 使用自定义 CUDA kernel，直接操作连续 GPU 内存，
    // 不依赖 OpenCV 的 Texture Object，无 pitch 对齐要求。
    if (interpolation == 1 /* cv::INTER_LINEAR */)
    {
        cuda::bilinear_resize(data_ptr(src), H, W, data_ptr(result), target_h, target_w, C);
        return result;
    }

    for (int c = 0; c < C; ++c)
    {
        // 包装为 GpuMat（零拷贝）
        cv::cuda::GpuMat src_slice(H, W, CV_32FC1,
                                   const_cast<float*>(data_ptr(src)) + c * H * W);
        cv::cuda::GpuMat dst_slice(target_h, target_w, CV_32FC1,
                                   data_ptr(result) + c * target_h * target_w);
        cv::cuda::resize(src_slice, dst_slice, cv::Size(target_w, target_h), 0, 0, interpolation);
    }

    return result;
}

// ── GPU 图像预处理 ──────────────────────────────────────────────────

std::pair<Ort::Value, std::array<int, 4>> GpuMemoryAllocator::preprocess_image_gpu(
    const cv::Mat& bgr_image, int target_h, int target_w, int divisor)
{
    // ImageNet 归一化常量
    static const float kMean[] = {0.485f, 0.456f, 0.406f};
    static const float kStd[] = {0.229f, 0.224f, 0.225f};

    // 1. 上传到 GPU
    cv::cuda::GpuMat gpu_bgr;
    gpu_bgr.upload(bgr_image);

    // 2. Resize（如果需要）
    if (gpu_bgr.rows != target_h || gpu_bgr.cols != target_w)
    {
        cv::cuda::GpuMat resized;
        cv::cuda::resize(gpu_bgr, resized, cv::Size(target_w, target_h), 0, 0, cv::INTER_LINEAR);
        gpu_bgr = resized;
    }

    int h = gpu_bgr.rows;
    int w = gpu_bgr.cols;

    // 3. BGR→RGB + float32 [0,1]
    cv::cuda::GpuMat gpu_rgb;
    cv::cuda::cvtColor(gpu_bgr, gpu_rgb, cv::COLOR_BGR2RGB);

    cv::cuda::GpuMat gpu_float;
    gpu_rgb.convertTo(gpu_float, CV_32FC3, 1.0 / 255.0);

    // 4. ImageNet normalize（分通道）
    std::vector<cv::cuda::GpuMat> channels(3);
    cv::cuda::split(gpu_float, channels);
    for (int c = 0; c < 3; ++c)
    {
        cv::cuda::subtract(channels[c], cv::Scalar(kMean[c]), channels[c]);
        cv::cuda::divide(channels[c], cv::Scalar(kStd[c]), channels[c]);
    }
    cv::cuda::merge(channels, gpu_float);

    // 5. Pad 到 divisor 的倍数
    int pad_h = (divisor - h % divisor) % divisor;
    int pad_w = (divisor - w % divisor) % divisor;
    int pad_top = pad_h / 2;
    int pad_bottom = pad_h - pad_top;
    int pad_left = pad_w / 2;
    int pad_right = pad_w - pad_left;

    if (pad_h > 0 || pad_w > 0)
    {
        cv::cuda::GpuMat padded;
        cv::cuda::copyMakeBorder(gpu_float, padded, pad_top, pad_bottom, pad_left, pad_right,
                                 cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
        gpu_float = padded;
    }

    int padded_h = gpu_float.rows;
    int padded_w = gpu_float.cols;

    // 6. HWC → NCHW: 分通道排列到连续 GPU 内存
    auto result = allocate({1, 3, padded_h, padded_w});
    float* dst = data_ptr(result);

    cv::cuda::split(gpu_float, channels);
    for (int c = 0; c < 3; ++c)
    {
        // channels[c] 是 padded_h × padded_w 的连续 GpuMat
        // 如果 step != cols * elemSize，需要逐行拷贝
        if (channels[c].isContinuous())
        {
            cudaMemcpy(dst + c * padded_h * padded_w, channels[c].data,
                       padded_h * padded_w * sizeof(float), cudaMemcpyDeviceToDevice);
        }
        else
        {
            for (int r = 0; r < padded_h; ++r)
            {
                cudaMemcpy(dst + c * padded_h * padded_w + r * padded_w,
                           channels[c].ptr<float>(r), padded_w * sizeof(float),
                           cudaMemcpyDeviceToDevice);
            }
        }
    }

    std::array<int, 4> pad = {pad_top, pad_bottom, pad_left, pad_right};
    return {std::move(result), pad};
}

}  // namespace ortcore
}  // namespace cutie
