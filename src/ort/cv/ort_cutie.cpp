#include <filesystem>
#include <stdexcept>

#include "cutie/core/processor.h"
#include "cutie/ort/cv/ort_cutie.h"

namespace cutie
{
namespace ortcv
{

// ── ONNX Runtime 日志回调 ────────────────────────────────────────

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

// ── Construction / destruction ──────────────────────────────────────

OrtCutie::OrtCutie(const core::CutieConfig& config, std::shared_ptr<linden::log::ILogger> logger)
    : logger_(logger ? std::move(logger) : linden::log::StdLogger::instance()),
      env_(ORT_LOGGING_LEVEL_WARNING, "cutie", ort_logging_callback, logger_.get())
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

    // Read static N from mask_encoder sensory input shape [1, N, C, H, W] (index 2, dim 1)
    {
        auto sensory_shape = mask_encoder_->session->GetInputTypeInfo(2)
                                 .GetTensorTypeAndShapeInfo()
                                 .GetShape();
        if (sensory_shape.size() >= 2 && sensory_shape[1] > 0)
        {
            model_n_obj_ = static_cast<int>(sensory_shape[1]);
            logger_->info("OrtCutie: model compiled for N={} objects (static)", model_n_obj_);
        }
    }
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
    int64_t actual_n = ortcore::GpuMemoryAllocator::shape(sensory)[1];

    // Pad N dimension to model_n_obj_ on GPU
    auto padded_sensory = gpu_alloc_->pad_dim(sensory, 1, model_n_obj_);
    auto padded_masks = gpu_alloc_->pad_dim(masks, 1, model_n_obj_);

    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(image));
    inputs.push_back(std::move(pix_feat));
    inputs.push_back(std::move(padded_sensory));
    inputs.push_back(std::move(padded_masks));

    auto outputs = run_session(*mask_encoder_, inputs);

    // Restore input ownership
    image = std::move(inputs[0]);
    pix_feat = std::move(inputs[1]);

    MaskEncoded enc;
    enc.mask_value = gpu_alloc_->slice_dim(outputs[0], 1, actual_n);
    enc.new_sensory = gpu_alloc_->slice_dim(outputs[1], 1, actual_n);
    enc.obj_summaries = gpu_alloc_->slice_dim(outputs[2], 1, actual_n);
    return enc;
}

Ort::Value OrtCutie::pixel_fusion(Ort::Value& pix_feat, Ort::Value& pixel, Ort::Value& sensory,
                                  Ort::Value& last_mask)
{
    int64_t actual_n = ortcore::GpuMemoryAllocator::shape(pixel)[1];

    auto padded_pixel = gpu_alloc_->pad_dim(pixel, 1, model_n_obj_);
    auto padded_sensory = gpu_alloc_->pad_dim(sensory, 1, model_n_obj_);
    auto padded_mask = gpu_alloc_->pad_dim(last_mask, 1, model_n_obj_);

    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(pix_feat));
    inputs.push_back(std::move(padded_pixel));
    inputs.push_back(std::move(padded_sensory));
    inputs.push_back(std::move(padded_mask));

    auto outputs = run_session(*pixel_fuser_, inputs);

    pix_feat = std::move(inputs[0]);

    return gpu_alloc_->slice_dim(outputs[0], 1, actual_n);
}

Ort::Value OrtCutie::readout_query(Ort::Value& pixel_readout, Ort::Value& obj_memory)
{
    int64_t actual_n = ortcore::GpuMemoryAllocator::shape(pixel_readout)[1];

    auto padded_readout = gpu_alloc_->pad_dim(pixel_readout, 1, model_n_obj_);
    auto padded_obj_mem = gpu_alloc_->pad_dim(obj_memory, 1, model_n_obj_);

    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(padded_readout));
    inputs.push_back(std::move(padded_obj_mem));

    auto outputs = run_session(*object_transformer_, inputs);

    return gpu_alloc_->slice_dim(outputs[0], 1, actual_n);
}

OrtCutie::SegmentResult OrtCutie::segment(Ort::Value& f8, Ort::Value& f4,
                                          Ort::Value& memory_readout, Ort::Value& sensory)
{
    int64_t actual_n = ortcore::GpuMemoryAllocator::shape(memory_readout)[1];

    auto padded_readout = gpu_alloc_->pad_dim(memory_readout, 1, model_n_obj_);
    auto padded_sensory = gpu_alloc_->pad_dim(sensory, 1, model_n_obj_);

    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(f8));
    inputs.push_back(std::move(f4));
    inputs.push_back(std::move(padded_readout));
    inputs.push_back(std::move(padded_sensory));

    auto outputs = run_session(*mask_decoder_, inputs);

    f8 = std::move(inputs[0]);
    f4 = std::move(inputs[1]);

    SegmentResult seg;
    seg.new_sensory = gpu_alloc_->slice_dim(outputs[0], 1, actual_n);
    seg.logits = gpu_alloc_->slice_dim(outputs[1], 1, actual_n);
    return seg;
}

}  // namespace ortcv
}  // namespace cutie
