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

/// Basic single-session ORT handler.
/// Reference: lite.ai.toolkit BasicOrtHandler
class BasicOrtHandler
{
protected:
    Ort::Env ort_env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> ort_session;
    Ort::MemoryInfo memory_info_handler =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    OrtSessionInfo session_info;
    unsigned int num_threads;

protected:
    explicit BasicOrtHandler(const std::string& onnx_path, unsigned int num_threads = 1,
                             int device_id = -1);
    virtual ~BasicOrtHandler() = default;

    BasicOrtHandler(const BasicOrtHandler&) = delete;
    BasicOrtHandler& operator=(const BasicOrtHandler&) = delete;
    BasicOrtHandler(BasicOrtHandler&&) noexcept = default;
    BasicOrtHandler& operator=(BasicOrtHandler&&) noexcept = default;

    /// Run inference with the given inputs. Returns all output tensors.
    std::vector<Ort::Value> run(const std::vector<Ort::Value>& input_values);

    /// Run inference with named inputs/outputs for flexibility.
    std::vector<Ort::Value> run(const std::vector<const char*>& input_names,
                                const std::vector<Ort::Value>& input_values,
                                const std::vector<const char*>& output_names);

private:
    void initialize_handler(const std::string& onnx_path, int device_id);

    // Owned strings backing the const char* pointers in session_info
    std::vector<std::string> input_name_strings_;
    std::vector<std::string> output_name_strings_;
    std::vector<const char*> input_name_ptrs_;
    std::vector<const char*> output_name_ptrs_;
};

}  // namespace ortcore
}  // namespace cutie

#endif  // CUTIE_ORT_HANDLER_H
