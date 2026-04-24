/**
 * @file trt_types.h
 * @brief TensorRT 类型定义和工具函数
 *
 * 定义 TensorRT 相关的类型别名和辅助函数。
 */

#ifndef CUTIE_TRT_CORE_TRT_TYPES_H
#define CUTIE_TRT_CORE_TRT_TYPES_H

#include <NvInfer.h>

#include <memory>

namespace cutie
{
namespace trtcore
{

/// TensorRT 智能指针删除器
struct TrtDeleter
{
    template <typename T>
    void operator()(T* obj) const
    {
        if (obj) delete obj;
    }
};

/// TensorRT 对象智能指针类型
template <typename T>
using TrtUniquePtr = std::unique_ptr<T, TrtDeleter>;

}  // namespace trtcore
}  // namespace cutie

#endif  // CUTIE_TRT_CORE_TRT_TYPES_H
