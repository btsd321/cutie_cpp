/**
 * @file trt_engine_builder.cpp
 * @brief TrtEngineBuilder 实现
 *
 * 实现 ONNX 到 TensorRT 引擎的转换、序列化和反序列化。
 * 提供引擎缓存机制，避免重复构建。
 */

#include "cutie/trt/core/trt_engine_builder.h"

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <linden_logger/logger_interface.hpp>

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace cutie
{
namespace trtcore
{

// ── TensorRT 日志适配器 ────────────────────────────────────────
// 将 TensorRT 日志输出集成到 linden_logger 系统

class TrtEngineBuilder::TrtLogger : public nvinfer1::ILogger
{
public:
    explicit TrtLogger(std::shared_ptr<linden::log::ILogger> logger) : logger_(logger) {}

    void log(Severity severity, const char* msg) noexcept override
    {
        if (!logger_) return;

        // 将 TensorRT 日志级别映射到 linden_logger 级别
        linden::log::LogLevel level;
        switch (severity)
        {
            case Severity::kINTERNAL_ERROR:
            case Severity::kERROR:
                level = linden::log::LogLevel::ERROR;
                break;
            case Severity::kWARNING:
                level = linden::log::LogLevel::WARN;
                break;
            case Severity::kINFO:
                level = linden::log::LogLevel::INFO;
                break;
            case Severity::kVERBOSE:
                level = linden::log::LogLevel::DEBUG;
                break;
            default:
                level = linden::log::LogLevel::INFO;
        }

        logger_->logf(level, "[TRT] {}", fmt::make_format_args(msg));
    }

private:
    std::shared_ptr<linden::log::ILogger> logger_;
};

// ── 构造 / 析构 ────────────────────────────────────────────────

TrtEngineBuilder::TrtEngineBuilder(std::shared_ptr<linden::log::ILogger> logger)
    : logger_(logger ? std::move(logger) : linden::log::StdLogger::instance()),
      trt_logger_(std::make_unique<TrtLogger>(logger_))
{
    // 创建 TensorRT Builder
    builder_.reset(nvinfer1::createInferBuilder(*trt_logger_));
    if (!builder_)
    {
        logger_->error("TrtEngineBuilder: 创建 IBuilder 失败");
        throw std::runtime_error("TrtEngineBuilder: 创建 IBuilder 失败");
    }

    // 创建 TensorRT Runtime
    runtime_.reset(nvinfer1::createInferRuntime(*trt_logger_));
    if (!runtime_)
    {
        logger_->error("TrtEngineBuilder: 创建 IRuntime 失败");
        throw std::runtime_error("TrtEngineBuilder: 创建 IRuntime 失败");
    }

    logger_->info("TrtEngineBuilder: 初始化成功");
}

TrtEngineBuilder::~TrtEngineBuilder() = default;

// ── 从 ONNX 构建引擎 ────────────────────────────────────────────

TrtUniquePtr<nvinfer1::ICudaEngine> TrtEngineBuilder::build_from_onnx(
    const std::string& onnx_path, const BuildConfig& config)
{
    namespace fs = std::filesystem;

    if (!fs::exists(onnx_path))
    {
        logger_->error("TrtEngineBuilder: ONNX 文件不存在: {}", onnx_path);
        throw std::runtime_error("TrtEngineBuilder: ONNX 文件不存在: " + onnx_path);
    }

    logger_->info("TrtEngineBuilder: 开始从 ONNX 构建引擎: {}", onnx_path);

    // 创建网络定义（TensorRT 10.x 默认使用显式 batch）
    TrtUniquePtr<nvinfer1::INetworkDefinition> network(
        builder_->createNetworkV2(0U));
    if (!network)
    {
        logger_->error("TrtEngineBuilder: 创建 INetworkDefinition 失败");
        throw std::runtime_error("TrtEngineBuilder: 创建 INetworkDefinition 失败");
    }

    // 创建 ONNX 解析器
    TrtUniquePtr<nvonnxparser::IParser> parser(
        nvonnxparser::createParser(*network, *trt_logger_));
    if (!parser)
    {
        logger_->error("TrtEngineBuilder: 创建 ONNX Parser 失败");
        throw std::runtime_error("TrtEngineBuilder: 创建 ONNX Parser 失败");
    }

    // 解析 ONNX 文件
    logger_->debug("TrtEngineBuilder: 解析 ONNX 文件...");
    if (!parser->parseFromFile(onnx_path.c_str(),
                               static_cast<int>(config.log_level)))
    {
        logger_->error("TrtEngineBuilder: ONNX 解析失败");
        for (int i = 0; i < parser->getNbErrors(); ++i)
        {
            logger_->error("  - {}", parser->getError(i)->desc());
        }
        throw std::runtime_error("TrtEngineBuilder: ONNX 解析失败");
    }

    // 创建构建配置
    TrtUniquePtr<nvinfer1::IBuilderConfig> builder_config(
        builder_->createBuilderConfig());
    if (!builder_config)
    {
        logger_->error("TrtEngineBuilder: 创建 IBuilderConfig 失败");
        throw std::runtime_error("TrtEngineBuilder: 创建 IBuilderConfig 失败");
    }

    // 设置工作空间大小
    builder_config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                                       config.max_workspace_size);
    logger_->debug("TrtEngineBuilder: 工作空间大小 = {} MB",
                   config.max_workspace_size / (1024 * 1024));

    // 启用 FP16 精度（TensorRT 10.x API）
    if (config.enable_fp16)
    {
        builder_config->setFlag(nvinfer1::BuilderFlag::kFP16);
        logger_->info("TrtEngineBuilder: 启用 FP16 精度");
    }

    // 启用 INT8 精度（TensorRT 10.x API）
    if (config.enable_int8)
    {
        builder_config->setFlag(nvinfer1::BuilderFlag::kINT8);
        logger_->info("TrtEngineBuilder: 启用 INT8 精度");
    }

    // 构建序列化网络
    logger_->info("TrtEngineBuilder: 开始构建引擎（可能需要几分钟）...");
    TrtUniquePtr<nvinfer1::IHostMemory> serialized_engine(
        builder_->buildSerializedNetwork(*network, *builder_config));
    if (!serialized_engine)
    {
        logger_->error("TrtEngineBuilder: 引擎构建失败");
        throw std::runtime_error("TrtEngineBuilder: 引擎构建失败");
    }

    // 反序列化引擎
    TrtUniquePtr<nvinfer1::ICudaEngine> engine(runtime_->deserializeCudaEngine(
        serialized_engine->data(), serialized_engine->size()));
    if (!engine)
    {
        logger_->error("TrtEngineBuilder: 引擎反序列化失败");
        throw std::runtime_error("TrtEngineBuilder: 引擎反序列化失败");
    }

    logger_->info("TrtEngineBuilder: 引擎构建成功");
    return engine;
}

// ── 序列化引擎到文件 ────────────────────────────────────────────

void TrtEngineBuilder::serialize_engine(nvinfer1::ICudaEngine* engine,
                                        const std::string& output_path)
{
    if (!engine)
    {
        logger_->error("TrtEngineBuilder: 引擎指针为空，无法序列化");
        throw std::runtime_error("TrtEngineBuilder: 引擎指针为空");
    }

    logger_->info("TrtEngineBuilder: 序列化引擎到文件: {}", output_path);

    // 序列化引擎
    TrtUniquePtr<nvinfer1::IHostMemory> serialized(engine->serialize());
    if (!serialized)
    {
        logger_->error("TrtEngineBuilder: 引擎序列化失败");
        throw std::runtime_error("TrtEngineBuilder: 引擎序列化失败");
    }

    // 写入文件
    std::ofstream file(output_path, std::ios::binary);
    if (!file)
    {
        logger_->error("TrtEngineBuilder: 无法打开文件: {}", output_path);
        throw std::runtime_error("TrtEngineBuilder: 无法打开文件: " + output_path);
    }

    file.write(static_cast<const char*>(serialized->data()), serialized->size());
    file.close();

    logger_->info("TrtEngineBuilder: 引擎序列化成功，大小 = {} MB",
                  serialized->size() / (1024.0 * 1024.0));
}

// ── 从文件反序列化引擎 ──────────────────────────────────────────

TrtUniquePtr<nvinfer1::ICudaEngine> TrtEngineBuilder::deserialize_engine(
    const std::string& engine_path)
{
    namespace fs = std::filesystem;

    if (!fs::exists(engine_path))
    {
        logger_->error("TrtEngineBuilder: 引擎文件不存在: {}", engine_path);
        throw std::runtime_error("TrtEngineBuilder: 引擎文件不存在: " + engine_path);
    }

    logger_->info("TrtEngineBuilder: 从文件反序列化引擎: {}", engine_path);

    // 读取文件
    std::ifstream file(engine_path, std::ios::binary);
    if (!file)
    {
        logger_->error("TrtEngineBuilder: 无法打开文件: {}", engine_path);
        throw std::runtime_error("TrtEngineBuilder: 无法打开文件: " + engine_path);
    }

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    file.close();

    logger_->debug("TrtEngineBuilder: 读取引擎文件，大小 = {} MB", size / (1024.0 * 1024.0));

    // 反序列化引擎
    TrtUniquePtr<nvinfer1::ICudaEngine> engine(
        runtime_->deserializeCudaEngine(buffer.data(), size));
    if (!engine)
    {
        logger_->error("TrtEngineBuilder: 引擎反序列化失败");
        throw std::runtime_error("TrtEngineBuilder: 引擎反序列化失败");
    }

    logger_->info("TrtEngineBuilder: 引擎反序列化成功");
    return engine;
}

// ── 获取或构建引擎（带缓存） ────────────────────────────────────

TrtUniquePtr<nvinfer1::ICudaEngine> TrtEngineBuilder::get_or_build_engine(
    const std::string& onnx_path, const std::string& engine_path, const BuildConfig& config)
{
    namespace fs = std::filesystem;

    // 检查引擎缓存是否存在且有效
    if (fs::exists(engine_path))
    {
        // 比较时间戳：如果 .engine 文件晚于 .onnx，直接加载
        if (fs::exists(onnx_path))
        {
            auto engine_time = fs::last_write_time(engine_path);
            auto onnx_time = fs::last_write_time(onnx_path);

            if (engine_time >= onnx_time)
            {
                logger_->info("TrtEngineBuilder: 发现有效的引擎缓存，直接加载");
                try
                {
                    return deserialize_engine(engine_path);
                }
                catch (const std::exception& e)
                {
                    logger_->warn("TrtEngineBuilder: 加载缓存引擎失败: {}", e.what());
                    logger_->warn("TrtEngineBuilder: 将重新构建引擎");
                }
            }
            else
            {
                logger_->info("TrtEngineBuilder: ONNX 文件已更新，需要重新构建引擎");
            }
        }
        else
        {
            logger_->warn("TrtEngineBuilder: ONNX 文件不存在，但引擎缓存存在，尝试加载");
            try
            {
                return deserialize_engine(engine_path);
            }
            catch (const std::exception& e)
            {
                logger_->error("TrtEngineBuilder: 加载缓存引擎失败且 ONNX 文件不存在");
                throw;
            }
        }
    }

    // 从 ONNX 构建引擎
    logger_->info("TrtEngineBuilder: 未找到有效缓存，从 ONNX 构建引擎");
    auto engine = build_from_onnx(onnx_path, config);

    // 序列化引擎到文件
    try
    {
        serialize_engine(engine.get(), engine_path);
    }
    catch (const std::exception& e)
    {
        logger_->warn("TrtEngineBuilder: 序列化引擎失败: {}", e.what());
        logger_->warn("TrtEngineBuilder: 引擎未缓存，下次启动将重新构建");
    }

    return engine;
}

}  // namespace trtcore
}  // namespace cutie
