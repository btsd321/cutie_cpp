#include <iostream>
#include <stdexcept>

#include "cutie/ort/core/ort_handler.h"

namespace cutie
{
namespace ortcore
{

BasicOrtHandler::BasicOrtHandler(const std::string& onnx_path, unsigned int num_threads,
                                 int device_id)
    : ort_env(ORT_LOGGING_LEVEL_WARNING, "cutie"), num_threads(num_threads)
{
    initialize_handler(onnx_path, device_id);
}

void BasicOrtHandler::initialize_handler(const std::string& onnx_path, int device_id)
{
    session_options.SetIntraOpNumThreads(static_cast<int>(num_threads));
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // Enable CUDA EP if device_id >= 0
    if (device_id >= 0)
    {
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = device_id;
        session_options.AppendExecutionProvider_CUDA(cuda_options);
    }

    ort_session = std::make_unique<Ort::Session>(ort_env, onnx_path.c_str(), session_options);

    Ort::AllocatorWithDefaultOptions allocator;

    // Collect input names and shapes
    size_t num_inputs = ort_session->GetInputCount();
    input_name_strings_.reserve(num_inputs);
    input_name_ptrs_.reserve(num_inputs);
    session_info.input_names.reserve(num_inputs);
    session_info.input_shapes.reserve(num_inputs);

    for (size_t i = 0; i < num_inputs; ++i)
    {
        auto name = ort_session->GetInputNameAllocated(i, allocator);
        input_name_strings_.emplace_back(name.get());
        session_info.input_names.push_back(input_name_strings_.back());

        auto type_info = ort_session->GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        session_info.input_shapes.push_back(tensor_info.GetShape());
    }

    for (auto& s : input_name_strings_)
    {
        input_name_ptrs_.push_back(s.c_str());
    }

    // Collect output names and shapes
    size_t num_outputs = ort_session->GetOutputCount();
    output_name_strings_.reserve(num_outputs);
    output_name_ptrs_.reserve(num_outputs);
    session_info.output_names.reserve(num_outputs);
    session_info.output_shapes.reserve(num_outputs);

    for (size_t i = 0; i < num_outputs; ++i)
    {
        auto name = ort_session->GetOutputNameAllocated(i, allocator);
        output_name_strings_.emplace_back(name.get());
        session_info.output_names.push_back(output_name_strings_.back());

        auto type_info = ort_session->GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        session_info.output_shapes.push_back(tensor_info.GetShape());
    }

    for (auto& s : output_name_strings_)
    {
        output_name_ptrs_.push_back(s.c_str());
    }
}

std::vector<Ort::Value> BasicOrtHandler::run(const std::vector<Ort::Value>& input_values)
{
    return ort_session->Run(Ort::RunOptions{nullptr}, input_name_ptrs_.data(), input_values.data(),
                            input_values.size(), output_name_ptrs_.data(),
                            output_name_ptrs_.size());
}

std::vector<Ort::Value> BasicOrtHandler::run(const std::vector<const char*>& input_names,
                                             const std::vector<Ort::Value>& input_values,
                                             const std::vector<const char*>& output_names)
{
    return ort_session->Run(Ort::RunOptions{nullptr}, input_names.data(), input_values.data(),
                            input_values.size(), output_names.data(), output_names.size());
}

}  // namespace ortcore
}  // namespace cutie
