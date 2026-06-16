/// @file gpu_mask_preprocess.cu
/// GPU mask 预处理 kernel 实现：nearest resize + pad + one_hot + aggregate。

#include "cutie/common/gpu_mask_preprocess.h"
#include "cutie/common/cuda_kernels.h"
#include "cutie/common/gpu_tensor_ops.h"

#include <cuda_runtime.h>

#include <cstdio>

// ── CUDA kernels ────────────────────────────────────────────────────

namespace cutie
{
namespace cuda
{

/// nearest neighbor resize kernel（int32 mask）。
/// 每个线程处理一个输出像素。
/// src_pitch / dst_pitch：GpuMat 内存每行实际 int32 元素数（step / sizeof(int32_t)）。
/// CUDA 分配的 GpuMat 行尾可能有对齐 padding，必须用 pitch 而非 cols 寻址。
__global__ void resize_mask_nearest_kernel(const int32_t* __restrict__ src, int src_h, int src_w,
                                           int src_pitch, int32_t* __restrict__ dst, int dst_h,
                                           int dst_w, int dst_pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) return;

    // nearest neighbor 采样：直接整数除法映射
    int sx = x * src_w / dst_w;
    int sy = y * src_h / dst_h;
    sx = min(sx, src_w - 1);
    sy = min(sy, src_h - 1);

    // 使用 pitch（含 padding）寻址，与 GpuMat::step 一致
    dst[y * dst_pitch + x] = src[sy * src_pitch + sx];
}

void launch_resize_mask_nearest(const int32_t* src, int src_h, int src_w, int src_pitch,
                                int32_t* dst, int dst_h, int dst_w, int dst_pitch,
                                cudaStream_t stream)
{
    // [DIAG] 打印完整参数（含 pitch）
    printf("[launch_resize_mask_nearest] src: h=%d w=%d pitch=%d | dst: h=%d w=%d pitch=%d\n",
           src_h, src_w, src_pitch, dst_h, dst_w, dst_pitch);
    if (src_pitch != src_w)
        printf("[launch_resize_mask_nearest] WARNING: src_pitch(%d) != src_w(%d), 有 padding\n",
               src_pitch, src_w);
    if (dst_pitch != dst_w)
        printf("[launch_resize_mask_nearest] WARNING: dst_pitch(%d) != dst_w(%d), 有 padding\n",
               dst_pitch, dst_w);

    dim3 block(32, 8);
    dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);

    resize_mask_nearest_kernel<<<grid, block, 0, stream>>>(src, src_h, src_w, src_pitch,
                                                            dst, dst_h, dst_w, dst_pitch);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "launch_resize_mask_nearest kernel error: %s\n", cudaGetErrorString(err));
    }

    // [DIAG] kernel 执行后：用 pitch 正确寻址验证几个采样点
    cudaStreamSynchronize(stream);
    if (dst_h > 10 && dst_w > 10) {
        int32_t v[8];
        // dst 用 pitch 寻址（与 kernel 一致）
        cudaMemcpy(&v[0], &dst[0 * dst_pitch + 0],   sizeof(int32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&v[1], &dst[0 * dst_pitch + 1],   sizeof(int32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&v[2], &dst[1 * dst_pitch + 0],   sizeof(int32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&v[3], &dst[10 * dst_pitch + 10], sizeof(int32_t), cudaMemcpyDeviceToHost);
        // src 也用 pitch 寻址
        cudaMemcpy(&v[4], &src[0 * src_pitch + 0],   sizeof(int32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&v[5], &src[0 * src_pitch + 1],   sizeof(int32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&v[6], &src[1 * src_pitch + 0],   sizeof(int32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&v[7], &src[10 * src_pitch + 10], sizeof(int32_t), cudaMemcpyDeviceToHost);
        printf("[launch_resize_mask_nearest] 验证(pitch寻址): "
               "dst(r0c0)=%d dst(r1c0)=%d dst(r0c1)=%d dst(r10c10)=%d | "
               "src(r0c0)=%d src(r1c0)=%d src(r0c1)=%d src(r10c10)=%d\n",
               v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
    }
}

/// pad mask kernel：每个线程处理输出 mask 的一个像素。
/// pad 区域填 0，有效区域从 src 拷贝。
/// src_pitch / dst_pitch：GpuMat 内存每行实际 int32 元素数（step / sizeof(int32_t)）。
__global__ void pad_mask_kernel(const int32_t* __restrict__ src, int src_h, int src_w,
                                int src_pitch, int32_t* __restrict__ dst, int dst_h, int dst_w,
                                int dst_pitch, int pad_top, int pad_left)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) return;

    int rx = x - pad_left;
    int ry = y - pad_top;

    // 使用 pitch（含 padding）寻址，与 GpuMat::step 一致
    if (rx < 0 || rx >= src_w || ry < 0 || ry >= src_h)
    {
        dst[y * dst_pitch + x] = 0;
    }
    else
    {
        dst[y * dst_pitch + x] = src[ry * src_pitch + rx];
    }
}

void launch_pad_mask(const int32_t* src, int src_h, int src_w, int src_pitch,
                     int32_t* dst, int dst_h, int dst_w, int dst_pitch,
                     int pad_top, int pad_left, cudaStream_t stream)
{
    // [DIAG] 打印完整参数（含 pitch）
    printf("[launch_pad_mask] src: h=%d w=%d pitch=%d | dst: h=%d w=%d pitch=%d | "
           "pad_top=%d pad_left=%d\n",
           src_h, src_w, src_pitch, dst_h, dst_w, dst_pitch, pad_top, pad_left);
    if (src_pitch != src_w)
        printf("[launch_pad_mask] WARNING: src_pitch(%d) != src_w(%d), 有 padding\n",
               src_pitch, src_w);
    if (dst_pitch != dst_w)
        printf("[launch_pad_mask] WARNING: dst_pitch(%d) != dst_w(%d), 有 padding\n",
               dst_pitch, dst_w);

    dim3 block(32, 8);
    dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);

    pad_mask_kernel<<<grid, block, 0, stream>>>(src, src_h, src_w, src_pitch,
                                                dst, dst_h, dst_w, dst_pitch,
                                                pad_top, pad_left);

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

    // [DIAG] 打印调用参数
    printf("[gpu_resize_mask_nearest] 输入: mask.rows=%d cols=%d (实际 H×W), "
           "target_h=%d target_w=%d, step=%zu, continuous=%d\n",
           mask.rows, mask.cols, target_h, target_w, mask.step, mask.isContinuous());

    // [DIAG] 检查 step 是否影响寻址
    int src_pitch = static_cast<int>(mask.step / sizeof(int32_t));
    if (mask.step != mask.cols * sizeof(int32_t)) {
        printf("[gpu_resize_mask_nearest] WARNING: step(%zu) != cols*4(%zu), 有 padding! "
               "src_pitch=%d (实际每行元素数)\n",
               mask.step, mask.cols * sizeof(int32_t), src_pitch);
    }

    cv::cuda::GpuMat result(target_h, target_w, CV_32SC1);

    // [DIAG] 检查输出 result 的 step
    int dst_pitch = static_cast<int>(result.step / sizeof(int32_t));
    printf("[gpu_resize_mask_nearest] 输出 result: rows=%d cols=%d step=%zu pitch=%d, continuous=%d\n",
           result.rows, result.cols, result.step, dst_pitch, result.isContinuous());

    // GpuMat 可能有 step 对齐（非连续）。对于 CV_32SC1 连续分配的情况直接使用 data 指针。
    // 如果 mask 有 padding（step > cols * elemSize），需确保 kernel 正确处理。
    // 传入 src_pitch (step/sizeof(int32_t)) 而不是 cols，处理非连续内存。
    cuda::launch_resize_mask_nearest(reinterpret_cast<const int32_t*>(mask.data), mask.rows,
                                     mask.cols, src_pitch,
                                     reinterpret_cast<int32_t*>(result.data), target_h,
                                     target_w, dst_pitch, stream);

    cudaStreamSynchronize(stream);
    printf("[gpu_resize_mask_nearest] 输出: result.rows=%d cols=%d step=%zu pitch=%d\n",
           result.rows, result.cols, result.step, dst_pitch);

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

    // [DIAG] 检查 src/dst step vs cols*4
    int src_pitch = static_cast<int>(mask.step / sizeof(int32_t));
    int dst_pitch = static_cast<int>(result.step / sizeof(int32_t));
    printf("[gpu_pad_mask] src: rows=%d cols=%d step=%zu pitch=%d | "
           "dst: rows=%d cols=%d step=%zu pitch=%d\n",
           mask.rows, mask.cols, mask.step, src_pitch,
           result.rows, result.cols, result.step, dst_pitch);
    if (mask.step != mask.cols * sizeof(int32_t))
        printf("[gpu_pad_mask] WARNING: src step(%zu) != cols*4(%zu), src 有 padding!\n",
               mask.step, mask.cols * sizeof(int32_t));
    if (result.step != result.cols * sizeof(int32_t))
        printf("[gpu_pad_mask] WARNING: dst step(%zu) != cols*4(%zu), dst 有 padding!\n",
               result.step, result.cols * sizeof(int32_t));

    cuda::launch_pad_mask(reinterpret_cast<const int32_t*>(mask.data), mask.rows, mask.cols,
                          src_pitch,
                          reinterpret_cast<int32_t*>(result.data), dst_h, dst_w, dst_pitch,
                          top, left, stream);

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
    int mask_pitch = static_cast<int>(padded.step / sizeof(int32_t));

    for (int mi = 0; mi < num_input_obj; ++mi)
    {
        // 找到 objects[mi] 在全局对象列表中的通道索引
        // 注意：one_hot 通道 0 对应 object_manager 的第一个 obj，不是 bg
        // 调用者需确保 objects 的 channel 映射正确。
        // 这里简单地按 objects 顺序写入对应的通道（调用者负责映射）。
        // 实际使用中，InferenceCore::step_gpu_impl 中会像原 step() 一样处理通道映射。
        cuda::one_hot_encode(d_mask, mask_pitch, H, W, d_objects + mi,
                             GpuMemoryAllocator::data_ptr(one_hot_gpu) + mi * HW, 1);
    }

    cudaFree(d_objects);

    // ⑤ aggregate: add background + softmax
    auto prob_with_bg = gpu_aggregate(alloc, one_hot_gpu);

    return prob_with_bg;
}

}  // namespace ortcore
}  // namespace cutie
