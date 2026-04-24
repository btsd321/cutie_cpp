/**
 * @file trt_handler.h
 * @brief TensorRT 引擎管理器
 *
 * 管理单个 TensorRT 引擎的执行上下文，提供同步和异步推理接口。
 * 对应 ONNX Runtime 后端的 BasicOrtHandler。
 */

#ifndef CUTIE_TRT_CORE_TRT_HANDLER_H
#define CUTIE_TRT_CORE_TRT_HANDLER_H

#include <NvInfer.h>

#include <cuda_runtime.h>
#include <linden_logger/logger_interface.hpp>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "cutie/trt/core/trt_types.h"

namespace cutie
{
namespace trtcore
{

/**
 * @class TrtHandler
 * @brief TensorRT 引擎管理器
 *
 * 管理单个 TensorRT 引擎的执行上下文，提供推理接口。
 * 支持同步和异步推理，支持动态形状输入。
 */
class TrtHandler
{
public:
    /**
     * @brief 从引擎文件构造
     * @param engine_path TensorRT 引擎文件路径
     * @param device_id GPU 设备 ID
     * @param logger 日志记录器
     */
    TrtHandler(const std::string& engine_path, int device_id,
               std::shared_ptr<linden::log::ILogger> logger = nullptr);

    /**
     * @brief 从引擎对象构造
     * @param engine TensorRT 引擎智能指针
     * @param device_id GPU 设备 ID
     * @param logger 日志记录器
     */
    TrtHandler(TrtUniquePtr<nvinfer1::ICudaEngine> engine, int device_id,
               std::shared_ptr<linden::log::ILogger> logger = nullptr);

    ~TrtHandler();

    TrtHandler(const TrtHandler&) = delete;
    TrtHandler& operator=(const TrtHandler&) = delete;
    TrtHandler(TrtHandler&&) noexcept;
    TrtHandler& operator=(TrtHandler&&) noexcept;

    /**
     * @brief 同步推理（内部管理 CUDA stream）
     *
     * 绑定输入输出 GPU 指针并执行推理，阻塞直到完成。
     *
     * @param input_buffers 输入张量名称 → GPU 指针映射
     * @param output_buffers 输出张量名称 → GPU 指针映射
     */
    void infer(const std::unordered_map<std::string, void*>& input_buffers,
               const std::unordered_map<std::string, void*>& output_buffers);

    /**
     * @brief 异步推理（用户提供 CUDA stream）
     *
     * 绑定输入输出 GPU 指针并在指定 stream 上执行推理。
     * 不阻塞，用户需自行同步 stream。
     *
     * @param input_buffers 输入张量名称 → GPU 指针映射
     * @param output_buffers 输出张量名称 → GPU 指针映射
     * @param stream CUDA stream
     */
    void infer_async(const std::unordered_map<std::string, void*>& input_buffers,
                     const std::unordered_map<std::string, void*>& output_buffers,
                     cudaStream_t stream);

    /**
     * @brief 设置输入张量的动态形状
     *
     * 用于动态形状模型，在推理前调用。
     *
     * @param name 输入张量名称
     * @param shape 张量形状
     */
    void set_input_shape(const std::string& name, const std::vector<int64_t>& shape);

    // ── 访问器 ──────────────────────────────────────────────────

    /// 获取输入张量名称列表
    const std::vector<std::string>& input_names() const { return input_names_; }

    /// 获取输出张量名称列表
    const std::vector<std::string>& output_names() const { return output_names_; }

    /// 获取张量的形状（支持输入和输出）
    std::vector<int64_t> get_binding_shape(const std::string& name) const;

    /// 获取 GPU 设备 ID
    int device_id() const { return device_id_; }

private:
    void initialize(int device_id);
    void collect_binding_info();

    std::shared_ptr<linden::log::ILogger> logger_;
    TrtUniquePtr<nvinfer1::ICudaEngine> engine_;
    TrtUniquePtr<nvinfer1::IExecutionContext> context_;

    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::unordered_map<std::string, int32_t> name_to_index_;  ///< 名称 → 绑定索引

    cudaStream_t stream_ = nullptr;  ///< 内部 CUDA stream（用于同步推理）
    int device_id_;
};

}  // namespace trtcore
}  // namespace cutie

#endif  // CUTIE_TRT_CORE_TRT_HANDLER_H
