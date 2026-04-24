/**
 * @file trt_handler.cpp
 * @brief TrtHandler 实现
 *
 * 实现 TensorRT 引擎的执行上下文管理和推理接口。
 * 支持同步和异步推理，支持动态形状输入。
 */

#include "cutie/trt/core/trt_handler.h"

#include <NvInfer.h>

#include <cuda_runtime.h>
#include <linden_logger/logger_interface.hpp>

#include <stdexcept>

#include "cutie/trt/core/trt_engine_builder.h"

namespace cutie
{
namespace trtcore
{

// ── 构造 / 析构 ────────────────────────────────────────────────

TrtHandler::TrtHandler(const std::string& engine_path, int device_id,
                       std::shared_ptr<linden::log::ILogger> logger)
    : logger_(logger ? std::move(logger) : linden::log::StdLogger::instance())
{
    // 使用 TrtEngineBuilder 加载引擎
    TrtEngineBuilder builder(logger_);
    engine_ = builder.deserialize_engine(engine_path);

    initialize(device_id);
}

TrtHandler::TrtHandler(TrtUniquePtr<nvinfer1::ICudaEngine> engine, int device_id,
                       std::shared_ptr<linden::log::ILogger> logger)
    : logger_(logger ? std::move(logger) : linden::log::StdLogger::instance()),
      engine_(std::move(engine))
{
    if (!engine_)
    {
        logger_->error("TrtHandler: 引擎指针为空");
        throw std::runtime_error("TrtHandler: 引擎指针为空");
    }

    initialize(device_id);
}

TrtHandler::~TrtHandler()
{
    if (stream_)
    {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
}

TrtHandler::TrtHandler(TrtHandler&&) noexcept = default;
TrtHandler& TrtHandler::operator=(TrtHandler&&) noexcept = default;

// ── 初始化 ──────────────────────────────────────────────────────

void TrtHandler::initialize(int device_id)
{
    device_id_ = device_id;

    // 设置 CUDA 设备
    cudaError_t err = cudaSetDevice(device_id_);
    if (err != cudaSuccess)
    {
        logger_->error("TrtHandler: cudaSetDevice 失败: {}", cudaGetErrorString(err));
        throw std::runtime_error("TrtHandler: cudaSetDevice 失败");
    }

    // 创建执行上下文
    context_.reset(engine_->createExecutionContext());
    if (!context_)
    {
        logger_->error("TrtHandler: 创建 IExecutionContext 失败");
        throw std::runtime_error("TrtHandler: 创建 IExecutionContext 失败");
    }

    // 创建 CUDA stream（用于同步推理）
    err = cudaStreamCreate(&stream_);
    if (err != cudaSuccess)
    {
        logger_->error("TrtHandler: cudaStreamCreate 失败: {}", cudaGetErrorString(err));
        throw std::runtime_error("TrtHandler: cudaStreamCreate 失败");
    }

    // 收集输入输出绑定信息
    collect_binding_info();

    logger_->info("TrtHandler: 初始化成功，设备 ID = {}", device_id_);
}

void TrtHandler::collect_binding_info()
{
    int32_t num_io_tensors = engine_->getNbIOTensors();

    for (int32_t i = 0; i < num_io_tensors; ++i)
    {
        const char* name = engine_->getIOTensorName(i);
        if (!name)
        {
            logger_->warn("TrtHandler: 无法获取张量名称，索引 = {}", i);
            continue;
        }

        std::string tensor_name(name);
        name_to_index_[tensor_name] = i;

        nvinfer1::TensorIOMode mode = engine_->getTensorIOMode(name);
        if (mode == nvinfer1::TensorIOMode::kINPUT)
        {
            input_names_.push_back(tensor_name);
            logger_->debug("TrtHandler: 输入张量 [{}] = {}", input_names_.size() - 1, tensor_name);
        }
        else if (mode == nvinfer1::TensorIOMode::kOUTPUT)
        {
            output_names_.push_back(tensor_name);
            logger_->debug("TrtHandler: 输出张量 [{}] = {}", output_names_.size() - 1, tensor_name);
        }
    }

    logger_->info("TrtHandler: 输入张量数 = {}, 输出张量数 = {}", input_names_.size(),
                  output_names_.size());
}

// ── 推理接口 ────────────────────────────────────────────────────

void TrtHandler::infer(const std::unordered_map<std::string, void*>& input_buffers,
                       const std::unordered_map<std::string, void*>& output_buffers)
{
    infer_async(input_buffers, output_buffers, stream_);

    // 同步等待完成
    cudaError_t err = cudaStreamSynchronize(stream_);
    if (err != cudaSuccess)
    {
        logger_->error("TrtHandler: cudaStreamSynchronize 失败: {}", cudaGetErrorString(err));
        throw std::runtime_error("TrtHandler: cudaStreamSynchronize 失败");
    }
}

void TrtHandler::infer_async(const std::unordered_map<std::string, void*>& input_buffers,
                             const std::unordered_map<std::string, void*>& output_buffers,
                             cudaStream_t stream)
{
    // 绑定输入张量
    for (const auto& name : input_names_)
    {
        auto it = input_buffers.find(name);
        if (it == input_buffers.end())
        {
            logger_->error("TrtHandler: 缺少输入张量: {}", name);
            throw std::runtime_error("TrtHandler: 缺少输入张量: " + name);
        }

        if (!context_->setTensorAddress(name.c_str(), it->second))
        {
            logger_->error("TrtHandler: setTensorAddress 失败，输入: {}", name);
            throw std::runtime_error("TrtHandler: setTensorAddress 失败");
        }
    }

    // 绑定输出张量
    for (const auto& name : output_names_)
    {
        auto it = output_buffers.find(name);
        if (it == output_buffers.end())
        {
            logger_->error("TrtHandler: 缺少输出张量: {}", name);
            throw std::runtime_error("TrtHandler: 缺少输出张量: " + name);
        }

        if (!context_->setTensorAddress(name.c_str(), it->second))
        {
            logger_->error("TrtHandler: setTensorAddress 失败，输出: {}", name);
            throw std::runtime_error("TrtHandler: setTensorAddress 失败");
        }
    }

    // 执行推理
    if (!context_->enqueueV3(stream))
    {
        logger_->error("TrtHandler: enqueueV3 失败");
        throw std::runtime_error("TrtHandler: enqueueV3 失败");
    }
}

// ── 动态形状设置 ────────────────────────────────────────────────

void TrtHandler::set_input_shape(const std::string& name, const std::vector<int64_t>& shape)
{
    // 转换为 nvinfer1::Dims
    nvinfer1::Dims dims;
    dims.nbDims = static_cast<int32_t>(shape.size());
    for (size_t i = 0; i < shape.size(); ++i)
    {
        dims.d[i] = static_cast<int32_t>(shape[i]);
    }

    if (!context_->setInputShape(name.c_str(), dims))
    {
        logger_->error("TrtHandler: setInputShape 失败，张量: {}", name);
        throw std::runtime_error("TrtHandler: setInputShape 失败");
    }

    logger_->debug("TrtHandler: 设置输入形状 {} = [{}, {}, {}, {}]", name, shape[0], shape[1],
                   shape[2], shape[3]);
}

// ── 访问器 ──────────────────────────────────────────────────────

std::vector<int64_t> TrtHandler::get_binding_shape(const std::string& name) const
{
    nvinfer1::Dims dims = engine_->getTensorShape(name.c_str());

    std::vector<int64_t> shape;
    shape.reserve(dims.nbDims);
    for (int32_t i = 0; i < dims.nbDims; ++i)
    {
        shape.push_back(static_cast<int64_t>(dims.d[i]));
    }

    return shape;
}

}  // namespace trtcore
}  // namespace cutie
