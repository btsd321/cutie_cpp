/// @file gpu_image_preprocess.cu
/// 融合 GPU 图像预处理实现：单 kernel 完成 bilinear resize + pad + BGR→RGB
/// + /255 + ImageNet normalize + HWC→CHW。

#include "cutie/ort/core/gpu_image_preprocess.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <stdexcept>

// ── CUDA kernel ─────────────────────────────────────────────────────

namespace cutie
{
namespace cuda
{

/// 融合预处理 kernel：每个线程处理输出 tensor 中的一个 (x, y) 位置。
/// 一次完成：bilinear resize → letterbox pad → BGR→RGB → /255 → ImageNet norm → CHW 写入。
__global__ void fused_image_preprocess_kernel(const uint8_t* __restrict__ src,
                                              float* __restrict__ dst,
                                              ImagePreprocessParams p)
{
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= p.dst_w || dy >= p.dst_h) return;

    float c0, c1, c2;  // 三通道插值结果（归一化后，RGB 顺序）

    // 在 letterbox 内的相对坐标
    int rx = dx - p.pad_left;
    int ry = dy - p.pad_top;

    if (rx < 0 || rx >= p.resize_w || ry < 0 || ry >= p.resize_h)
    {
        // pad 区域：归一化后的零值 = (0 - mean) / std = -mean * inv_std
        c0 = -p.mean[0] * p.inv_std[0];
        c1 = -p.mean[1] * p.inv_std[1];
        c2 = -p.mean[2] * p.inv_std[2];
    }
    else
    {
        // 映射到源图坐标（half-pixel center 对齐，与 cv::INTER_LINEAR 一致）
        float sx = (rx + 0.5f) * p.src_w / p.resize_w - 0.5f;
        float sy = (ry + 0.5f) * p.src_h / p.resize_h - 0.5f;

        // 钳位到 [0, src_size - 1]
        sx = fmaxf(0.f, fminf(sx, (float)(p.src_w - 1)));
        sy = fmaxf(0.f, fminf(sy, (float)(p.src_h - 1)));

        int x0 = (int)floorf(sx);
        int y0 = (int)floorf(sy);
        int x1 = min(x0 + 1, p.src_w - 1);
        int y1 = min(y0 + 1, p.src_h - 1);

        float fx = sx - x0;
        float fy = sy - y0;
        float w00 = (1.f - fx) * (1.f - fy);
        float w10 = fx * (1.f - fy);
        float w01 = (1.f - fx) * fy;
        float w11 = fx * fy;

        // 采样 4 个邻域像素（BGR HWC 布局，带 pitch）
        const uint8_t* row0 = src + y0 * p.src_pitch;
        const uint8_t* row1 = src + y1 * p.src_pitch;

        // BGR → RGB：通道 0=B, 1=G, 2=R → 输出 c0=R, c1=G, c2=B
        float b = (w00 * row0[x0 * 3 + 0] + w10 * row0[x1 * 3 + 0] +
                   w01 * row1[x0 * 3 + 0] + w11 * row1[x1 * 3 + 0]);
        float g = (w00 * row0[x0 * 3 + 1] + w10 * row0[x1 * 3 + 1] +
                   w01 * row1[x0 * 3 + 1] + w11 * row1[x1 * 3 + 1]);
        float r = (w00 * row0[x0 * 3 + 2] + w10 * row0[x1 * 3 + 2] +
                   w01 * row1[x0 * 3 + 2] + w11 * row1[x1 * 3 + 2]);

        // /255 + ImageNet normalize: (pixel/255 - mean) / std = pixel/255 * inv_std - mean * inv_std
        constexpr float inv255 = 1.0f / 255.0f;
        c0 = (r * inv255 - p.mean[0]) * p.inv_std[0];  // R
        c1 = (g * inv255 - p.mean[1]) * p.inv_std[1];  // G
        c2 = (b * inv255 - p.mean[2]) * p.inv_std[2];  // B
    }

    // 写入 CHW 布局: [C, H, W]，通道顺序 RGB
    int area = p.dst_h * p.dst_w;
    int offset = dy * p.dst_w + dx;
    dst[0 * area + offset] = c0;  // R
    dst[1 * area + offset] = c1;  // G
    dst[2 * area + offset] = c2;  // B
}

void launch_image_preprocess(const uint8_t* src_bgr, float* dst_chw,
                             const ImagePreprocessParams& p, cudaStream_t stream)
{
    dim3 block(32, 8);  // 256 threads per block
    dim3 grid((p.dst_w + block.x - 1) / block.x, (p.dst_h + block.y - 1) / block.y);

    fused_image_preprocess_kernel<<<grid, block, 0, stream>>>(src_bgr, dst_chw, p);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "launch_image_preprocess kernel error: %s\n", cudaGetErrorString(err));
    }
}

}  // namespace cuda

// ── GpuImagePreprocessor ────────────────────────────────────────────

namespace ortcore
{

constexpr float GpuImagePreprocessor::kMean[3];
constexpr float GpuImagePreprocessor::kStd[3];

std::tuple<int, int, std::array<int, 4>> GpuImagePreprocessor::compute_params(
    int src_h, int src_w, int target_h, int target_w, int divisor) const
{
    // Pad 到 divisor 的倍数
    int pad_h = (divisor - target_h % divisor) % divisor;
    int pad_w = (divisor - target_w % divisor) % divisor;
    int pad_top = pad_h / 2;
    int pad_bottom = pad_h - pad_top;
    int pad_left = pad_w / 2;
    int pad_right = pad_w - pad_left;

    int padded_h = target_h + pad_h;
    int padded_w = target_w + pad_w;

    return {padded_h, padded_w, {pad_top, pad_bottom, pad_left, pad_right}};
}

std::pair<Ort::Value, std::array<int, 4>> GpuImagePreprocessor::preprocess(
    const cv::cuda::GpuMat& gpu_bgr, int target_h, int target_w, GpuMemoryAllocator& alloc,
    int divisor, cudaStream_t stream)
{
    if (gpu_bgr.empty())
    {
        throw std::runtime_error("GpuImagePreprocessor::preprocess: empty GpuMat input");
    }

    auto [padded_h, padded_w, pad] = compute_params(gpu_bgr.rows, gpu_bgr.cols, target_h,
                                                     target_w, divisor);

    // 分配输出张量 [1, 3, padded_h, padded_w]
    auto result = alloc.allocate({1, 3, padded_h, padded_w});

    // 填充 kernel 参数
    cuda::ImagePreprocessParams p;
    p.src_w = gpu_bgr.cols;
    p.src_h = gpu_bgr.rows;
    p.src_pitch = static_cast<int>(gpu_bgr.step);
    p.resize_w = target_w;
    p.resize_h = target_h;
    p.dst_w = padded_w;
    p.dst_h = padded_h;
    p.pad_left = pad[2];
    p.pad_top = pad[0];
    for (int c = 0; c < 3; ++c)
    {
        p.mean[c] = kMean[c];
        p.inv_std[c] = 1.0f / kStd[c];
    }

    cuda::launch_image_preprocess(gpu_bgr.data, GpuMemoryAllocator::data_ptr(result), p, stream);

    return {std::move(result), pad};
}

std::pair<Ort::Value, std::array<int, 4>> GpuImagePreprocessor::preprocess(
    const cv::Mat& bgr_image, int target_h, int target_w, GpuMemoryAllocator& alloc, int divisor,
    cudaStream_t stream)
{
    if (bgr_image.empty())
    {
        throw std::runtime_error("GpuImagePreprocessor::preprocess: empty cv::Mat input");
    }

    // 使用持久化 GpuBuffer 上传，避免每帧 cudaMalloc
    size_t img_bytes = bgr_image.total() * bgr_image.elemSize();
    upload_buf_.reserve(img_bytes);

    // 构造一个连续的 GpuMat 指向 upload_buf_ 的 GPU 内存
    // 先确保 CPU 图像连续
    cv::Mat continuous = bgr_image.isContinuous() ? bgr_image : bgr_image.clone();

    cudaError_t err =
        cudaMemcpy(upload_buf_.data(), continuous.data, img_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string("GpuImagePreprocessor upload failed: ") +
                                 cudaGetErrorString(err));
    }

    // 包装为 GpuMat（零拷贝，指向 upload_buf_）
    cv::cuda::GpuMat gpu_bgr(continuous.rows, continuous.cols, continuous.type(),
                              upload_buf_.data(), continuous.step);

    return preprocess(gpu_bgr, target_h, target_w, alloc, divisor, stream);
}

}  // namespace ortcore
}  // namespace cutie
