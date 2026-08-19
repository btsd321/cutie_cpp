/**
 * @file py_convert.cpp
 * @brief py_convert.h 的实现。
 *
 * 拷贝策略实现细节见 py_convert.h 的文件级说明。
 */

#include "py_convert.h"

#include <algorithm>
#include <set>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

#include "cutie/common/gpu_memory.h"

namespace cutie
{
namespace python
{

namespace
{

/// 把 cudaError_t 转成带上下文的异常，避免静默产生错误数据。
void check_cuda(cudaError_t err, const char* what)
{
    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string(what) + " 失败: " + cudaGetErrorString(err));
    }
}

}  // namespace

// ── 输入：numpy → cv::Mat（零拷贝）────────────────────────────────────

cv::Mat image_from_numpy(const ImageArray& array)
{
    // py::array::c_style 已保证 C 连续，此处仅校验形状语义。
    if (array.ndim() != 3)
    {
        throw std::invalid_argument("图像数组必须是 3 维 (H, W, 3)，实际为 " +
                                    std::to_string(array.ndim()) + " 维");
    }
    if (array.shape(2) != 3)
    {
        throw std::invalid_argument("图像数组必须有 3 个通道（BGR），实际为 " +
                                    std::to_string(array.shape(2)) + " 个");
    }
    if (array.shape(0) <= 0 || array.shape(1) <= 0)
    {
        throw std::invalid_argument("图像尺寸不能为 0");
    }

    const int rows = static_cast<int>(array.shape(0));
    const int cols = static_cast<int>(array.shape(1));

    // 借用 numpy 缓冲区构造 cv::Mat：不拷贝，也不接管所有权。
    // const_cast 是安全的——推理路径只读取输入图像。
    return cv::Mat(rows, cols, CV_8UC3, const_cast<uint8_t*>(array.data()));
}

cv::Mat mask_from_numpy(const MaskArray& array)
{
    if (array.ndim() != 2)
    {
        throw std::invalid_argument("掩码数组必须是 2 维 (H, W)，实际为 " +
                                    std::to_string(array.ndim()) + " 维");
    }
    if (array.shape(0) <= 0 || array.shape(1) <= 0)
    {
        throw std::invalid_argument("掩码尺寸不能为 0");
    }

    const int rows = static_cast<int>(array.shape(0));
    const int cols = static_cast<int>(array.shape(1));

    return cv::Mat(rows, cols, CV_32SC1, const_cast<int32_t*>(array.data()));
}

// ── 输出：GPU → numpy（单次 D2H）──────────────────────────────────────

py::array index_mask_to_numpy(const cv::cuda::GpuMat& gpu_mask)
{
    if (gpu_mask.empty())
    {
        return py::array_t<int32_t>(std::vector<py::ssize_t>{0, 0});
    }
    if (gpu_mask.type() != CV_32SC1)
    {
        throw std::runtime_error("GPU 索引掩码类型必须是 CV_32SC1，实际 type=" +
                                 std::to_string(gpu_mask.type()));
    }

    const int rows = gpu_mask.rows;
    const int cols = gpu_mask.cols;

    // 先分配 numpy 输出缓冲区，D2H 直接写入，避免中间 cv::Mat。
    py::array_t<int32_t> out(std::vector<py::ssize_t>{rows, cols});
    auto info = out.request();

    // GpuMat 是 pitched 内存（step 可能大于 cols * 4），必须用 cudaMemcpy2D
    // 按行拷贝；用 cudaMemcpy 会把 padding 区一并拷入，产生错位。
    check_cuda(cudaMemcpy2D(info.ptr,                          // dst
                            static_cast<size_t>(cols) * sizeof(int32_t),  // dst pitch（紧凑）
                            gpu_mask.data,                     // src
                            gpu_mask.step,                     // src pitch（含 padding）
                            static_cast<size_t>(cols) * sizeof(int32_t),  // 每行有效字节
                            static_cast<size_t>(rows),         // 行数
                            cudaMemcpyDeviceToHost),
                "索引掩码 D2H 拷贝");

    return std::move(out);
}

py::array prob_to_numpy(const Ort::Value& gpu_prob)
{
    using GA = ortcore::GpuMemoryAllocator;

    if (!gpu_prob.IsTensor())
    {
        return py::array_t<float>(std::vector<py::ssize_t>{0});
    }

    const std::vector<int64_t> shape = GA::shape(gpu_prob);
    const int64_t total = GA::numel(shape);
    if (total <= 0)
    {
        return py::array_t<float>(std::vector<py::ssize_t>{0});
    }

    const std::vector<py::ssize_t> py_shape(shape.begin(), shape.end());
    py::array_t<float> out(py_shape);
    auto info = out.request();

    // 概率张量由 GpuMemoryAllocator 分配，是紧凑连续内存，可直接线性拷贝。
    check_cuda(cudaMemcpy(info.ptr, GA::data_ptr(gpu_prob),
                          static_cast<size_t>(total) * sizeof(float), cudaMemcpyDeviceToHost),
               "概率张量 D2H 拷贝");

    return std::move(out);
}

// ── 辅助：目标 ID 推导 ────────────────────────────────────────────────

std::vector<ObjectId> unique_object_ids(const cv::Mat& mask)
{
    std::vector<ObjectId> ids;
    if (mask.empty())
    {
        return ids;
    }
    if (mask.type() != CV_32SC1)
    {
        throw std::runtime_error("推导目标 ID 要求掩码为 CV_32SC1");
    }

    // std::set 天然去重且有序，掩码通常只有个位数目标，开销可忽略。
    std::set<ObjectId> unique_ids;
    for (int r = 0; r < mask.rows; ++r)
    {
        const int32_t* row = mask.ptr<int32_t>(r);
        for (int c = 0; c < mask.cols; ++c)
        {
            if (row[c] != 0)  // 0 是背景，不作为目标
            {
                unique_ids.insert(static_cast<ObjectId>(row[c]));
            }
        }
    }

    ids.assign(unique_ids.begin(), unique_ids.end());
    return ids;
}

}  // namespace python
}  // namespace cutie
