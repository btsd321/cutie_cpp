/// @file gpu_mask_preprocess.cu
/// GPU mask 预处理 kernel 实现：nearest resize + pad + one_hot + aggregate。

#include "cutie/ort/core/gpu_mask_preprocess.h"
#include "cutie/ort/core/cuda_kernels.h"
#include "cutie/ort/core/gpu_tensor_ops.h"

#include <cuda_runtime.h>

#include <cstdio>

// ── CUDA kernels ────────────────────────────────────────────────────

namespace cutie
{
namespace cuda
{

/// nearest neighbor resize kernel（int32 mask）。
/// 每个线程处理一个输出像素。
__global__ void resize_mask_nearest_kernel(const int32_t* __restrict__ src, int src_h, int src_w,
                                           int32_t* __restrict__ dst, int dst_h, int dst_w)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) return;

    // nearest neighbor 采样：直接整数除法映射
    int sx = x * src_w / dst_w;
    int sy = y * src_h / dst_h;
    sx = min(sx, src_w - 1);
    sy = min(sy, src_h - 1);

    dst[y * dst_w + x] = src[sy * src_w + sx];
}

void launch_resize_mask_nearest(const int32_t* src, int src_h, int src_w, int32_t* dst, int dst_h,
                                int dst_w, cudaStream_t stream)
{
    dim3 block(32, 8);
    dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);

    resize_mask_nearest_kernel<<<grid, block, 0, stream>>>(src, src_h, src_w, dst, dst_h, dst_w);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "launch_resize_mask_nearest kernel error: %s\n", cudaGetErrorString(err));
    }
}

/// pad mask kernel：每个线程处理输出 mask 的一个像素。
/// pad 区域填 0，有效区域从 src 拷贝。
__global__ void pad_mask_kernel(const int32_t* __restrict__ src, int src_h, int src_w,
                                int32_t* __restrict__ dst, int dst_h, int dst_w, int pad_top,
                                int pad_left)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) return;

    int rx = x - pad_left;
    int ry = y - pad_top;

    if (rx < 0 || rx >= src_w || ry < 0 || ry >= src_h)
    {
        dst[y * dst_w + x] = 0;
    }
    else
    {
        dst[y * dst_w + x] = src[ry * src_w + rx];
    }
}

void launch_pad_mask(const int32_t* src, int src_h, int src_w, int32_t* dst, int dst_h, int dst_w,
                     int pad_top, int pad_left, cudaStream_t stream)
{
    dim3 block(32, 8);
    dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);

    pad_mask_kernel<<<grid, block, 0, stream>>>(src, src_h, src_w, dst, dst_h, dst_w, pad_top,
                                                 pad_left);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "launch_pad_mask kernel error: %s\n", cudaGetErrorString(err));
    }
}

}  // namespace cuda

// ── 高层接口 ────────────────────────────────────────────────────────

namespace ortcore
{

cv::cuda::GpuMat gpu_resize_mask_nearest(const cv::cuda::GpuMat& mask, int target_h, int target_w,
                                         cudaStream_t stream)
{
    if (mask.rows == target_h && mask.cols == target_w)
    {
        return mask.clone();
    }

    cv::cuda::GpuMat result(target_h, target_w, CV_32SC1);

    // GpuMat 可能有 step 对齐（非连续）。对于 CV_32SC1 连续分配的情况直接使用 data 指针。
    // 如果 mask 有 padding（step > cols * elemSize），需确保 kernel 正确处理。
    // 此处假设 mask 是连续的（GpuMat(H,W,type) 默认连续分配）。
    cuda::launch_resize_mask_nearest(reinterpret_cast<const int32_t*>(mask.data), mask.rows,
                                     mask.cols, reinterpret_cast<int32_t*>(result.data), target_h,
                                     target_w, stream);

    return result;
}

cv::cuda::GpuMat gpu_pad_mask(const cv::cuda::GpuMat& mask, const std::array<int, 4>& pad,
                              cudaStream_t stream)
{
    int top = pad[0], bottom = pad[1], left = pad[2], right = pad[3];

    if (top == 0 && bottom == 0 && left == 0 && right == 0)
    {
        return mask.clone();
    }

    int dst_h = mask.rows + top + bottom;
    int dst_w = mask.cols + left + right;
    cv::cuda::GpuMat result(dst_h, dst_w, CV_32SC1);

    cuda::launch_pad_mask(reinterpret_cast<const int32_t*>(mask.data), mask.rows, mask.cols,
                          reinterpret_cast<int32_t*>(result.data), dst_h, dst_w, top, left, stream);

    return result;
}

Ort::Value gpu_preprocess_mask(GpuMemoryAllocator& alloc, const cv::cuda::GpuMat& gpu_mask,
                               const std::vector<ObjectId>& objects, int target_h, int target_w,
                               const std::array<int, 4>& pad, int total_num_objects,
                               cudaStream_t stream)
{
    // ① GPU resize (nearest)
    cv::cuda::GpuMat resized = gpu_resize_mask_nearest(gpu_mask, target_h, target_w, stream);

    // ② GPU pad
    cv::cuda::GpuMat padded = gpu_pad_mask(resized, pad, stream);

    int H = padded.rows;
    int W = padded.cols;
    int HW = H * W;

    // ③ 上传 objects 到 GPU（通常 <20 个 int32）
    std::vector<int32_t> obj_i32(objects.begin(), objects.end());
    int num_input_obj = static_cast<int>(obj_i32.size());
    int32_t* d_objects = nullptr;
    cudaMalloc(&d_objects, num_input_obj * sizeof(int32_t));
    cudaMemcpy(d_objects, obj_i32.data(), num_input_obj * sizeof(int32_t),
               cudaMemcpyHostToDevice);

    // ④ one_hot_encode：每个输入对象写入对应通道
    // 输出 [total_num_objects, H, W] 全零，然后按 object 写入
    auto one_hot_gpu = alloc.zeros({total_num_objects, H, W});

    const int32_t* d_mask = reinterpret_cast<const int32_t*>(padded.data);
    for (int mi = 0; mi < num_input_obj; ++mi)
    {
        // 找到 objects[mi] 在全局对象列表中的通道索引
        // 注意：one_hot 通道 0 对应 object_manager 的第一个 obj，不是 bg
        // 调用者需确保 objects 的 channel 映射正确。
        // 这里简单地按 objects 顺序写入对应的通道（调用者负责映射）。
        // 实际使用中，InferenceCore::step_gpu_impl 中会像原 step() 一样处理通道映射。
        cuda::one_hot_encode(d_mask, d_objects + mi,
                             GpuMemoryAllocator::data_ptr(one_hot_gpu) + mi * HW, 1, HW);
    }

    cudaFree(d_objects);

    // ⑤ aggregate: add background + softmax
    auto prob_with_bg = gpu_aggregate(alloc, one_hot_gpu);

    return prob_with_bg;
}

}  // namespace ortcore
}  // namespace cutie
