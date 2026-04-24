/**
 * @file ort_handler.h
 * @brief Basic ONNX Runtime session management.
 *
 * Provides a base class for managing ONNX Runtime sessions with
 * automatic model loading, input/output handling, and inference execution.
 */

#ifndef CUTIE_ORT_HANDLER_H
#define CUTIE_ORT_HANDLER_H

#include <memory>
#include <string>
#include <vector>

#include "cutie/ort/core/ort_config.h"
#include "cutie/ort/core/ort_types.h"

namespace cutie
{
namespace ortcore
{

/**
 * @class BasicOrtHandler
 * @brief Basic single-session ONNX Runtime handler.
 *
 * Manages a single ONNX Runtime session with automatic model loading,
 * session configuration, and inference execution. Provides both positional
 * and named input/output interfaces.
 *
 * 基础 ONNX Runtime 会话管理器，负责模型加载、会话配置和推理执行。
 * Reference: lite.ai.toolkit BasicOrtHandler
 */
class BasicOrtHandler
{
protected:
    Ort::Env ort_env;  ///< ONNX Runtime environment
    Ort::SessionOptions session_options;  ///< Session configuration
    std::unique_ptr<Ort::Session> ort_session;  ///< ONNX Runtime session
    Ort::MemoryInfo memory_info_handler =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);  ///< CPU memory info

    OrtSessionInfo session_info;  ///< Session metadata (inputs/outputs)
    unsigned int num_threads;  ///< Number of threads for inference

protected:
    /**
     * @brief Construct handler and load ONNX model.
     *
     * @param onnx_path Path to ONNX model file
     * @param num_threads Number of threads for inference (default 1)
     * @param device_id GPU device ID (-1 = CPU)
     * @throws std::runtime_error if model loading fails
     */
    explicit BasicOrtHandler(const std::string& onnx_path, unsigned int num_threads = 1,
                             int device_id = -1);
    virtual ~BasicOrtHandler() = default;

    BasicOrtHandler(const BasicOrtHandler&) = delete;
    BasicOrtHandler& operator=(const BasicOrtHandler&) = delete;
    BasicOrtHandler(BasicOrtHandler&&) noexcept = default;
    BasicOrtHandler& operator=(BasicOrtHandler&&) noexcept = default;

    /**
     * @brief Run inference with positional inputs/outputs.
     *
     * 使用位置参数运行推理，返回所有输出张量。
     *
     * @param input_values Input tensors in order
     * @return Output tensors in order
     */
    std::vector<Ort::Value> run(const std::vector<Ort::Value>& input_values);

    /**
     * @brief Run inference with named inputs/outputs.
     *
     * 使用命名参数运行推理，提供更灵活的输入/输出处理。
     *
     * @param input_names Input tensor names
     * @param input_values Input tensors
     * @param output_names Output tensor names
     * @return Output tensors in specified order
     */
    std::vector<Ort::Value> run(const std::vector<const char*>& input_names,
                                const std::vector<Ort::Value>& input_values,
                                const std::vector<const char*>& output_names);

private:
    /**
     * @brief Initialize handler and load model.
     *
     * 初始化处理器并加载 ONNX 模型。
     *
     * @param onnx_path Path to ONNX model file
     * @param device_id GPU device ID
     */
    void initialize_handler(const std::string& onnx_path, int device_id);

    // Owned strings backing the const char* pointers in session_info
    std::vector<std::string> input_name_strings_;  ///< Input name storage
    std::vector<std::string> output_name_strings_;  ///< Output name storage
    std::vector<const char*> input_name_ptrs_;  ///< Input name pointers
    std::vector<const char*> output_name_ptrs_;  ///< Output name pointers
};

}  // namespace ortcore
}  // namespace cutie

#endif  // CUTIE_ORT_HANDLER_H
