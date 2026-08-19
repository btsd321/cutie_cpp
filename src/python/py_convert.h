/**
 * @file py_convert.h
 * @brief numpy 数组与 OpenCV / GPU 张量之间的转换工具。
 *
 * 本文件是 Python 绑定层的数据转换核心，设计目标是把每帧的内存拷贝次数
 * 压到最低：**1 次 H2D（不可避免）+ 1 次 D2H**。
 *
 * 关键约定：
 * - 输入 numpy → cv::Mat 采用**借用指针**方式（`cv::Mat` 的外部数据构造函数），
 *   不发生拷贝。因此返回的 `cv::Mat` 生命周期**依附于传入的 numpy 数组**，
 *   调用方必须保证 numpy 对象在 cv::Mat 使用期间存活。
 * - 输出 GPU 张量 → numpy 直接 `cudaMemcpy2D` 写入预分配的 numpy 缓冲区，
 *   不经过中间 `cv::Mat`。故意**不复用** `types::GpuCutieMask::download()`
 *   （见 src/core/processor.cpp），因为它会自行分配 cv::Mat，导致二次拷贝。
 * - `cv::cuda::GpuMat` 是 pitched 内存（`step != cols * elemSize()`），
 *   D2H 必须用 `cudaMemcpy2D`，不能用 `cudaMemcpy`。
 *
 * @note 本文件所有函数均不获取 GIL，调用方需自行保证在持有 GIL 的上下文中调用
 *       （涉及 numpy 对象创建与访问的函数）。
 */

#ifndef CUTIE_PYTHON_PY_CONVERT_H
#define CUTIE_PYTHON_PY_CONVERT_H

#include <cstdint>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

#include "cutie/types.h"

namespace cutie
{
namespace python
{

namespace py = pybind11;

/// 要求 C 连续、不允许隐式类型转换的 uint8 数组（BGR 图像）
using ImageArray = py::array_t<uint8_t, py::array::c_style>;

/// 要求 C 连续、不允许隐式类型转换的 int32 数组（索引掩码）
using MaskArray = py::array_t<int32_t, py::array::c_style>;

/**
 * @brief 将 numpy BGR 图像零拷贝包装为 cv::Mat。
 *
 * 校验数组为 (H, W, 3) uint8 且 C 连续，随后借用其数据指针构造 cv::Mat。
 * **不发生内存拷贝。**
 *
 * @param array 输入图像数组，形状 (H, W, 3)，dtype uint8
 * @return 借用 `array` 缓冲区的 cv::Mat（CV_8UC3）
 *
 * @throws std::invalid_argument 维度、通道数或连续性不满足要求时抛出
 *
 * @warning 返回的 cv::Mat 不持有内存所有权，`array` 必须在其使用期间保持存活。
 */
cv::Mat image_from_numpy(const ImageArray& array);

/**
 * @brief 将 numpy 索引掩码零拷贝包装为 cv::Mat。
 *
 * 校验数组为 (H, W) int32 且 C 连续，随后借用其数据指针构造 cv::Mat。
 * **不发生内存拷贝。**
 *
 * @param array 输入掩码数组，形状 (H, W)，dtype int32
 * @return 借用 `array` 缓冲区的 cv::Mat（CV_32SC1）
 *
 * @throws std::invalid_argument 维度或连续性不满足要求时抛出
 *
 * @warning 返回的 cv::Mat 不持有内存所有权，`array` 必须在其使用期间保持存活。
 */
cv::Mat mask_from_numpy(const MaskArray& array);

/**
 * @brief 将 GPU 索引掩码单次 D2H 拷贝到新建的 numpy 数组。
 *
 * 先按 `gpu_mask` 尺寸分配 (H, W) int32 numpy 数组，再用 `cudaMemcpy2D`
 * 直接写入其缓冲区，正确处理 GpuMat 的 pitch 对齐。
 *
 * @param gpu_mask GPU 索引掩码（CV_32SC1）
 * @return 形状 (H, W)、dtype int32 的 numpy 数组；`gpu_mask` 为空时返回形状 (0, 0) 的数组
 *
 * @throws std::runtime_error 掩码类型非 CV_32SC1，或 cudaMemcpy2D 失败时抛出
 */
py::array index_mask_to_numpy(const cv::cuda::GpuMat& gpu_mask);

/**
 * @brief 将 GPU 概率张量单次 D2H 拷贝到新建的 numpy 数组。
 *
 * 概率张量是连续的 device 内存（非 pitched），形状为 [num_objects+1, H, W]。
 *
 * @param gpu_prob GPU 概率张量（Ort::Value，float32）
 * @return 形状 [num_objects+1, H, W]、dtype float32 的 numpy 数组；
 *         `gpu_prob` 非张量时返回空数组
 *
 * @throws std::runtime_error cudaMemcpy 失败时抛出
 */
py::array prob_to_numpy(const Ort::Value& gpu_prob);

/**
 * @brief 从索引掩码中提取所有非零目标 ID（升序去重）。
 *
 * 用于 Python 侧省略 `object_ids` 参数时自动推导要跟踪的目标。
 * 背景值 0 会被排除。
 *
 * @param mask 索引掩码（CV_32SC1）
 * @return 升序排列的唯一目标 ID 列表
 */
std::vector<ObjectId> unique_object_ids(const cv::Mat& mask);

}  // namespace python
}  // namespace cutie

#endif  // CUTIE_PYTHON_PY_CONVERT_H
