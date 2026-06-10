/**
 * @file ort_cutie.cpp
 * @brief OrtCutie implementation (ONNX Runtime model wrapper).
 *
 * Implements the wrapper for 6 ONNX Runtime sessions managing the complete
 * Cutie inference pipeline. Uses IO Binding for zero-copy GPU inference.
 * Includes ONNX Runtime logging integration and session management.
 */

#include <filesystem>
#include <stdexcept>

#include "cutie/core/processor.h"
#include "cutie/ort/cv/ort_cutie.h"

namespace cutie
{
namespace ortcv
{

// ── ONNX Runtime 日志回调 ────────────────────────────────────────
// 将 ONNX Runtime 的日志输出集成到 linden_logger 系统。

static void ORT_API_CALL ort_logging_callback(void* param, OrtLoggingLevel severity,
                                               const char* category, const char* logid,
                                               const char* code_location, const char* message)
{
    auto* logger = static_cast<linden::log::ILogger*>(param);
    if (!logger) return;

    // 将 ORT 日志级别映射到 linden_logger 级别
    linden::log::LogLevel level;
    switch (severity)
    {
        case ORT_LOGGING_LEVEL_VERBOSE:
        case ORT_LOGGING_LEVEL_INFO:
            level = linden::log::LogLevel::DEBUG;
            break;
        case ORT_LOGGING_LEVEL_WARNING:
            level = linden::log::LogLevel::WARN;
            break;
        case ORT_LOGGING_LEVEL_ERROR:
        case ORT_LOGGING_LEVEL_FATAL:
            level = linden::log::LogLevel::ERROR;
            break;
        default:
            level = linden::log::LogLevel::INFO;
    }

    // 格式化输出：[ORT][category] message (location)
    if (code_location && code_location[0] != '\0')
    {
        logger->logf(level, "[ORT][{}] {} ({})", fmt::make_format_args(category, message, code_location));
    }
    else
    {
        logger->logf(level, "[ORT][{}] {}", fmt::make_format_args(category, message));
    }
}

// ── Internal session wrapper ────────────────────────────────────────

struct OrtCutie::SessionBundle
{
    std::unique_ptr<Ort::Session> session;
    Ort::SessionOptions options;

    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::vector<const char*> input_name_ptrs;
    std::vector<const char*> output_name_ptrs;

    void collect_names()
    {
        Ort::AllocatorWithDefaultOptions allocator;

        size_t num_inputs = session->GetInputCount();
        input_names.reserve(num_inputs);
        input_name_ptrs.reserve(num_inputs);
        for (size_t i = 0; i < num_inputs; ++i)
        {
            auto name = session->GetInputNameAllocated(i, allocator);
            input_names.emplace_back(name.get());
        }
        for (auto& s : input_names) input_name_ptrs.push_back(s.c_str());

        size_t num_outputs = session->GetOutputCount();
        output_names.reserve(num_outputs);
        output_name_ptrs.reserve(num_outputs);
        for (size_t i = 0; i < num_outputs; ++i)
        {
            auto name = session->GetOutputNameAllocated(i, allocator);
            output_names.emplace_back(name.get());
        }
        for (auto& s : output_names) output_name_ptrs.push_back(s.c_str());
    }
};

static OrtLoggingLevel to_ort_level(linden::log::LogLevel level)
{
    switch (level)
    {
        case linden::log::LogLevel::DEBUG: return ORT_LOGGING_LEVEL_VERBOSE;
        case linden::log::LogLevel::INFO:  return ORT_LOGGING_LEVEL_INFO;
        case linden::log::LogLevel::WARN:  return ORT_LOGGING_LEVEL_WARNING;
        case linden::log::LogLevel::ERROR: return ORT_LOGGING_LEVEL_ERROR;
        default:                           return ORT_LOGGING_LEVEL_WARNING;
    }
}

// ── Construction / destruction ──────────────────────────────────────

OrtCutie::OrtCutie(const core::CutieConfig& config, std::shared_ptr<linden::log::ILogger> logger)
    : logger_(logger ? std::move(logger) : linden::log::StdLogger::instance()),
      env_(to_ort_level(logger_->get_level()), "cutie", ort_logging_callback, logger_.get())
{
    namespace fs = std::filesystem;
    const std::string& dir = config.model_dir;
    int dev = (config.device == Device::kCUDA) ? config.device_id : -1;

    if (dev < 0)
    {
        logger_->error("OrtCutie: GPU mode requires device_id >= 0 (Device::kCUDA)");
        throw std::runtime_error("OrtCutie: GPU mode requires CUDA device");
    }

    gpu_alloc_ = std::make_unique<ortcore::GpuMemoryAllocator>(dev);

    auto require = [&](const std::string& name) -> std::string
    {
        if (config.model_prefix.empty())
        {
            logger_->error("OrtCutie: model_prefix must be set (e.g. \"cutie-base-mega\")");
            throw std::runtime_error("OrtCutie: model_prefix is empty");
        }
        std::string path = (fs::path(dir) / (config.model_prefix + "_" + name)).string();
        if (!fs::exists(path))
        {
            logger_->error("OrtCutie: ONNX file not found: {}", path);
            throw std::runtime_error("OrtCutie: ONNX file not found: " + path);
        }
        return path;
    };

    pixel_encoder_ = create_session(require("pixel_encoder.onnx"), dev);
    logger_->info("OrtCutie: loaded pixel_encoder.onnx");

    // Read expected input H/W from pixel_encoder ONNX input shape [1, 3, H, W]
    {
        auto shape = pixel_encoder_->session->GetInputTypeInfo(0)
                         .GetTensorTypeAndShapeInfo()
                         .GetShape();
        if (shape.size() >= 4 && shape[2] > 0 && shape[3] > 0)
        {
            model_h_ = static_cast<int>(shape[2]);
            model_w_ = static_cast<int>(shape[3]);
            logger_->info("OrtCutie: model input size = {}x{}", model_h_, model_w_);
        }
        else
        {
            logger_->warn("OrtCutie: could not read static input shape from pixel_encoder");
        }
    }

    key_projection_ = create_session(require("key_projection.onnx"), dev);
    logger_->info("OrtCutie: loaded key_projection.onnx");
    mask_encoder_ = create_session(require("mask_encoder.onnx"), dev);
    logger_->info("OrtCutie: loaded mask_encoder.onnx");

    pixel_fuser_ = create_session(require("pixel_fuser.onnx"), dev);
    logger_->info("OrtCutie: loaded pixel_fuser.onnx");
    object_transformer_ = create_session(require("object_transformer.onnx"), dev);
    logger_->info("OrtCutie: loaded object_transformer.onnx");
    mask_decoder_ = create_session(require("mask_decoder.onnx"), dev);
    logger_->info("OrtCutie: loaded mask_decoder.onnx");

    logger_->info("OrtCutie: all 6 ONNX submodules loaded (GPU mode, dir={})", dir);
}

OrtCutie::~OrtCutie() = default;

OrtCutie::OrtCutie(OrtCutie&&) noexcept = default;
OrtCutie& OrtCutie::operator=(OrtCutie&&) noexcept = default;

OrtCutie::SessionPtr OrtCutie::create_session(const std::string& onnx_path, int device_id)
{
    auto bundle = std::make_unique<SessionBundle>();
    bundle->options.SetIntraOpNumThreads(1);
    bundle->options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    OrtCUDAProviderOptions cuda_opts;
    cuda_opts.device_id = device_id;
    bundle->options.AppendExecutionProvider_CUDA(cuda_opts);

    bundle->session = std::make_unique<Ort::Session>(env_, onnx_path.c_str(), bundle->options);
    bundle->collect_names();
    return bundle;
}

// ── IO Binding based session run ────────────────────────────────────

std::vector<Ort::Value> OrtCutie::run_session(SessionBundle& b,
                                              std::vector<Ort::Value>& inputs)
{
    Ort::IoBinding io_binding(*b.session);

    // Bind inputs (GPU tensors)
    for (size_t i = 0; i < inputs.size(); ++i)
    {
        io_binding.BindInput(b.input_name_ptrs[i], inputs[i]);
    }

    // Bind outputs to GPU memory
    for (size_t i = 0; i < b.output_name_ptrs.size(); ++i)
    {
        io_binding.BindOutput(b.output_name_ptrs[i], gpu_alloc_->memory_info());
    }

    b.session->Run(Ort::RunOptions{nullptr}, io_binding);

    return io_binding.GetOutputValues();
}

// ── Submodule inference (GPU in/out) ────────────────────────────────

OrtCutie::ImageFeatures OrtCutie::encode_image(Ort::Value& image)
{
    // Inputs:  image [1, 3, H, W] (GPU)
    // Outputs: f16, f8, f4, pix_feat (GPU)
    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(image));

    auto outputs = run_session(*pixel_encoder_, inputs);

    // Restore input ownership
    image = std::move(inputs[0]);

    ImageFeatures feat;
    feat.f16 = std::move(outputs[0]);
    feat.f8 = std::move(outputs[1]);
    feat.f4 = std::move(outputs[2]);
    feat.pix_feat = std::move(outputs[3]);
    return feat;
}

OrtCutie::KeyFeatures OrtCutie::transform_key(Ort::Value& f16)
{
    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(f16));

    auto outputs = run_session(*key_projection_, inputs);

    f16 = std::move(inputs[0]);

    KeyFeatures kf;
    kf.key = std::move(outputs[0]);
    kf.shrinkage = std::move(outputs[1]);
    kf.selection = std::move(outputs[2]);
    return kf;
}

OrtCutie::MaskEncoded OrtCutie::encode_mask(Ort::Value& image, Ort::Value& pix_feat,
                                            Ort::Value& sensory, Ort::Value& masks)
{
    // 动态 N 模型:直接传递真实对象数的张量,无需 pad/slice
    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(image));
    inputs.push_back(std::move(pix_feat));
    inputs.push_back(std::move(sensory));
    inputs.push_back(std::move(masks));

    auto outputs = run_session(*mask_encoder_, inputs);

    // Restore input ownership
    image = std::move(inputs[0]);
    pix_feat = std::move(inputs[1]);

    MaskEncoded enc;
    enc.mask_value = std::move(outputs[0]);
    enc.new_sensory = std::move(outputs[1]);
    enc.obj_summaries = std::move(outputs[2]);
    return enc;
}

Ort::Value OrtCutie::pixel_fusion(Ort::Value& pix_feat, Ort::Value& pixel, Ort::Value& sensory,
                                  Ort::Value& last_mask)
{
    // 动态 N 模型:直接传递真实对象数的张量,无需 pad
    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(pix_feat));
    inputs.push_back(std::move(pixel));
    inputs.push_back(std::move(sensory));
    inputs.push_back(std::move(last_mask));

    auto outputs = run_session(*pixel_fuser_, inputs);

    pix_feat = std::move(inputs[0]);

    return std::move(outputs[0]);
}

Ort::Value OrtCutie::readout_query(Ort::Value& pixel_readout, Ort::Value& obj_memory)
{
    // 动态 N 模型:直接传递真实对象数的张量,无需 pad
    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(pixel_readout));
    inputs.push_back(std::move(obj_memory));

    auto outputs = run_session(*object_transformer_, inputs);

    return std::move(outputs[0]);
}

OrtCutie::SegmentResult OrtCutie::segment(Ort::Value& f8, Ort::Value& f4,
                                          Ort::Value& memory_readout, Ort::Value& sensory)
{
    // 动态 N 模型:直接传递真实对象数的张量,无需 pad
    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(f8));
    inputs.push_back(std::move(f4));
    inputs.push_back(std::move(memory_readout));
    inputs.push_back(std::move(sensory));

    auto outputs = run_session(*mask_decoder_, inputs);

    f8 = std::move(inputs[0]);
    f4 = std::move(inputs[1]);

    SegmentResult seg;
    seg.new_sensory = std::move(outputs[0]);
    seg.logits = std::move(outputs[1]);
    return seg;
}

}  // namespace ortcv
}  // namespace cutie
