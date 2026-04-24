/**
 * @file trt_engine_builder.h
 * @brief TensorRT 引擎构建器
 *
 * 负责从 ONNX 模型构建 TensorRT 引擎，支持序列化和反序列化。
 * 提供引擎缓存机制，避免重复构建。
 */

#ifndef CUTIE_TRT_CORE_TRT_ENGINE_BUILDER_H
#define CUTIE_TRT_CORE_TRT_ENGINE_BUILDER_H

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <linden_logger/logger_interface.hpp>

#include <memory>
#include <string>

#include "cutie/trt/core/trt_config.h"
#include "cutie/trt/core/trt_types.h"

namespace cutie
{
namespace trtcore
{

/**
 * @class TrtEngineBuilder
 * @brief TensorRT 引擎构建器
 *
 * 从 ONNX 模型构建 TensorRT 引擎，支持序列化和反序列化。
 * 提供引擎缓存机制，避免每次启动都重新构建引擎。
 */
class TrtEngineBuilder
{
public:
    /**
     * @brief 构造函数
     * @param logger 日志记录器
     */
    explicit TrtEngineBuilder(std::shared_ptr<linden::log::ILogger> logger = nullptr);

    ~TrtEngineBuilder();

    TrtEngineBuilder(const TrtEngineBuilder&) = delete;
    TrtEngineBuilder& operator=(const TrtEngineBuilder&) = delete;

    /**
     * @brief 从 ONNX 文件构建 TensorRT 引擎
     *
     * 解析 ONNX 模型并构建优化的 TensorRT 引擎。
     *
     * @param onnx_path ONNX 模型文件路径
     * @param config 构建配置
     * @return TensorRT 引擎智能指针
     * @throws std::runtime_error 构建失败时抛出异常
     */
    TrtUniquePtr<nvinfer1::ICudaEngine> build_from_onnx(const std::string& onnx_path,
                                                        const BuildConfig& config);

    /**
     * @brief 序列化引擎到文件
     *
     * 将构建好的引擎序列化保存到文件，用于后续快速加载。
     *
     * @param engine TensorRT 引擎指针
     * @param output_path 输出文件路径
     * @throws std::runtime_error 序列化失败时抛出异常
     */
    void serialize_engine(nvinfer1::ICudaEngine* engine, const std::string& output_path);

    /**
     * @brief 从文件反序列化引擎
     *
     * 从序列化文件加载 TensorRT 引擎，避免重新构建。
     *
     * @param engine_path 引擎文件路径
     * @return TensorRT 引擎智能指针
     * @throws std::runtime_error 反序列化失败时抛出异常
     */
    TrtUniquePtr<nvinfer1::ICudaEngine> deserialize_engine(const std::string& engine_path);

    /**
     * @brief 获取或构建引擎（带缓存）
     *
     * 智能缓存机制：
     * - 如果 .engine 文件存在且时间戳晚于 .onnx，直接加载
     * - 否则从 ONNX 构建并序列化
     *
     * @param onnx_path ONNX 模型文件路径
     * @param engine_path 引擎缓存文件路径
     * @param config 构建配置
     * @return TensorRT 引擎智能指针
     */
    TrtUniquePtr<nvinfer1::ICudaEngine> get_or_build_engine(const std::string& onnx_path,
                                                            const std::string& engine_path,
                                                            const BuildConfig& config);

private:
    class TrtLogger;  ///< TensorRT 日志适配器（前向声明）

    std::shared_ptr<linden::log::ILogger> logger_;  ///< linden_logger 实例
    std::unique_ptr<TrtLogger> trt_logger_;         ///< TensorRT 日志适配器
    TrtUniquePtr<nvinfer1::IBuilder> builder_;      ///< TensorRT 构建器
    TrtUniquePtr<nvinfer1::IRuntime> runtime_;      ///< TensorRT 运行时
};

}  // namespace trtcore
}  // namespace cutie

#endif  // CUTIE_TRT_CORE_TRT_ENGINE_BUILDER_H
