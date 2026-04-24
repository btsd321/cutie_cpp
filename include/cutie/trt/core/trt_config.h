/**
 * @file trt_config.h
 * @brief TensorRT 配置和类型定义
 *
 * 包含 TensorRT 相关的头文件、类型定义和配置结构体。
 */

#ifndef CUTIE_TRT_CORE_TRT_CONFIG_H
#define CUTIE_TRT_CORE_TRT_CONFIG_H

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <memory>
#include <string>
#include <vector>

namespace cutie
{
namespace trtcore
{

/**
 * @struct BuildConfig
 * @brief TensorRT 引擎构建配置
 */
struct BuildConfig
{
    /// 工作空间大小（字节），默认 1GB
    size_t max_workspace_size = 1ULL << 30;

    /// 是否启用 FP16 精度
    bool enable_fp16 = false;

    /// 是否启用 INT8 精度
    bool enable_int8 = false;

    /// 是否启用严格类型约束
    bool strict_types = false;

    /// 日志级别
    nvinfer1::ILogger::Severity log_level = nvinfer1::ILogger::Severity::kWARNING;
};

}  // namespace trtcore
}  // namespace cutie

#endif  // CUTIE_TRT_CORE_TRT_CONFIG_H
