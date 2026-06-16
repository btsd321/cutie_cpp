/// @file gpu_postprocess.cu
/// GPU 后处理 kernel 实现：unpad + argmax index mask。

#include "cutie/common/gpu_postprocess.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <stdexcept>

// ── CUDA kernels ────────────────────────────────────────────────────

namespace cutie
{
namespace cuda
{

/// argmax kernel：每个线程处理一个像素 (x, y)，遍历 num_ch 个通道找最大值。
/// 对于典型场景 (num_obj < 20)，简单遍历比 reduction 更快。
/// out_pitch：输出 GpuMat 每行实际元素数（step/sizeof(int32_t)），处理行对齐 padding。
/// prob 是扁平 Ort tensor [C, H, W]（无 padding），out 是 GpuMat（可能有 pitch）。
__global__ void argmax_index_kernel(const float* __restrict__ prob, int num_ch, int H, int W,
                                    const int32_t* __restrict__ objects,
                                    int32_t* __restrict__ out, int out_pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    int hw = H * W;
    int prob_idx = y * W + x;  // prob 是紧密布局

    float max_val = prob[prob_idx];  // channel 0 = background
    int max_ch = 0;

    for (int c = 1; c < num_ch; ++c)
    {
        float v = prob[c * hw + prob_idx];
        if (v > max_val)
        {
            max_val = v;
            max_ch = c;
        }
    }

    // out 用 pitch 寻址（与 GpuMat::step 一致）
    out[y * out_pitch + x] = (max_ch == 0) ? 0 : objects[max_ch - 1];
}

void launch_argmax_index(const float* prob, int num_ch, int H, int W, const int32_t* objects,
                         int32_t* out, int out_pitch, cudaStream_t stream)
{
    // [DIAG]
    printf("[launch_argmax_index] H=%d W=%d num_ch=%d out_pitch=%d\n",
           H, W, num_ch, out_pitch);
    if (out_pitch != W)
        printf("[launch_argmax_index] WARNING: out_pitch(%d) != W(%d), 输出有 padding\n",
               out_pitch, W);

    dim3 block(32, 8);
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);

    argmax_index_kernel<<<grid, block, 0, stream>>>(prob, num_ch, H, W, objects, out, out_pitch);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "launch_argmax_index kernel error: %s\n", cudaGetErrorString(err));
    }
}

/// unpad kernel：每个线程处理输出 (c, y, x)，从源张量对应位置拷贝。
__global__ void unpad_kernel(const float* __restrict__ src, float* __restrict__ dst, int C,
                             int src_h, int src_w, int dst_h, int dst_w, int pad_top, int pad_left)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) return;

    int src_hw = src_h * src_w;
    int dst_hw = dst_h * dst_w;
    int sy = y + pad_top;
    int sx = x + pad_left;

    for (int c = 0; c < C; ++c)
    {
        dst[c * dst_hw + y * dst_w + x] = src[c * src_hw + sy * src_w + sx];
    }
}

void launch_unpad(const float* src, float* dst, int C, int src_h, int src_w, int dst_h, int dst_w,
                  int pad_top, int pad_left, cudaStream_t stream)
{
    dim3 block(32, 8);
    dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);

    unpad_kernel<<<grid, block, 0, stream>>>(src, dst, C, src_h, src_w, dst_h, dst_w, pad_top,
                                              pad_left);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "launch_unpad kernel error: %s\n", cudaGetErrorString(err));
    }
}

}  // namespace cuda

// ── 高层接口 ────────────────────────────────────────────────────────

namespace ortcore
{

Ort::Value gpu_unpad(GpuMemoryAllocator& alloc, const Ort::Value& src,
                     const std::array<int, 4>& pad, cudaStream_t stream)
{
    int top = pad[0], bottom = pad[1], left = pad[2], right = pad[3];

    if (top == 0 && bottom == 0 && left == 0 && right == 0)
    {
        return alloc.clone(src);
    }

    auto s = GpuMemoryAllocator::shape(src);
    int C = static_cast<int>(s[0]);
    int H = static_cast<int>(s[1]);
    int W = static_cast<int>(s[2]);
    int new_h = H - top - bottom;
    int new_w = W - left - right;

    if (new_h <= 0 || new_w <= 0)
    {
        return alloc.clone(src);
    }

    auto result = alloc.allocate({C, new_h, new_w});
    cuda::launch_unpad(GpuMemoryAllocator::data_ptr(src), GpuMemoryAllocator::data_ptr(result), C,
                       H, W, new_h, new_w, top, left, stream);

    return result;
}

cv::cuda::GpuMat gpu_prob_to_index_mask(GpuMemoryAllocator& alloc, const Ort::Value& prob_with_bg,
                                        const std::vector<ObjectId>& objects, cudaStream_t stream)
{
    auto s = GpuMemoryAllocator::shape(prob_with_bg);
    int num_ch = static_cast<int>(s[0]);
    int H = static_cast<int>(s[1]);
    int W = static_cast<int>(s[2]);

    // 分配输出 GpuMat
    cv::cuda::GpuMat index_mask(H, W, CV_32SC1);

    // [DIAG] 检查 index_mask 的 step 对齐
    int out_pitch = static_cast<int>(index_mask.step / sizeof(int32_t));
    if (index_mask.step != index_mask.cols * sizeof(int32_t))
    {
        printf("[gpu_prob_to_index_mask] WARNING: index_mask step=%zu vs cols*4=%zu, "
               "有 padding! pitch=%d\n",
               index_mask.step, index_mask.cols * sizeof(int32_t), out_pitch);
    }

    if (num_ch <= 1 || objects.empty())
    {
        // 只有背景通道或无对象，全部填零
        // 注意：cudaMemset 必须按 step 填充整块（含 padding 区也置零）
        cudaMemset2D(index_mask.data, index_mask.step, 0, W * sizeof(int32_t), H);
        return index_mask;
    }

    // 上传 objects 数组到 GPU（通常 <20 个 int32，开销可忽略）
    int num_obj = static_cast<int>(objects.size());
    int32_t* d_objects = nullptr;
    cudaMalloc(&d_objects, num_obj * sizeof(int32_t));
    cudaMemcpy(d_objects, objects.data(), num_obj * sizeof(int32_t), cudaMemcpyHostToDevice);

    cuda::launch_argmax_index(GpuMemoryAllocator::data_ptr(prob_with_bg), num_ch, H, W, d_objects,
                              reinterpret_cast<int32_t*>(index_mask.data), out_pitch, stream);

    // [DIAG] argmax 后采样验证：在 prob 上找 ID 中心区域，看看 argmax 输出是否对应
    cudaStreamSynchronize(stream);
    {
        // 下载部分 prob[c=1] 数据 + index_mask 第一行/中间行做对比
        std::vector<float> prob_ch1(static_cast<size_t>(H) * W);
        cudaMemcpy(prob_ch1.data(),
                   GpuMemoryAllocator::data_ptr(prob_with_bg) + static_cast<size_t>(H) * W,
                   prob_ch1.size() * sizeof(float), cudaMemcpyDeviceToHost);
        // 找前景中心
        double sum_x = 0, sum_y = 0;
        int cnt = 0;
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                if (prob_ch1[static_cast<size_t>(y) * W + x] > 0.5f) {
                    sum_x += x; sum_y += y; cnt++;
                }
            }
        }
        if (cnt > 0) {
            printf("[gpu_prob_to_index_mask] DIAG: prob[ch=1]>0.5 像素数=%d, "
                   "中心=(%.1f, %.1f) (H=%d W=%d)\n",
                   cnt, sum_x / cnt, sum_y / cnt, H, W);
        }

        // 下载 index_mask 验证 argmax 输出位置
        std::vector<int32_t> idx_cpu(static_cast<size_t>(H) * out_pitch);
        cudaMemcpy(idx_cpu.data(), index_mask.data,
                   static_cast<size_t>(H) * index_mask.step, cudaMemcpyDeviceToHost);
        long long sum_x2 = 0, sum_y2 = 0;
        int cnt2 = 0;
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                if (idx_cpu[static_cast<size_t>(y) * out_pitch + x] != 0) {
                    sum_x2 += x; sum_y2 += y; cnt2++;
                }
            }
        }
        if (cnt2 > 0) {
            printf("[gpu_prob_to_index_mask] DIAG: index_mask 非零像素=%d, "
                   "中心=(%.1f, %.1f) (H=%d W=%d, step=%zu, pitch=%d)\n",
                   cnt2, static_cast<double>(sum_x2) / cnt2,
                   static_cast<double>(sum_y2) / cnt2, H, W, index_mask.step, out_pitch);
        }
    }

    cudaFree(d_objects);

    return index_mask;
}

}  // namespace ortcore
}  // namespace cutie
