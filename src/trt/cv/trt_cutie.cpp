/**
 * @file trt_cutie.cpp
 * @brief TrtCutie 实现（TensorRT Cutie 模型封装）
 *
 * 实现 6 个 TensorRT 子模块的封装，提供与 OrtCutie 相同的接口。
 * 管理完整的 Cutie 推理流程，支持全 GPU 数据流。
 */

#include "cutie/trt/cv/trt_cutie.h"

#include <filesystem>
#include <stdexcept>

#include "cutie/core/processor.h"
#include "cutie/trt/core/trt_engine_builder.h"

namespace cutie
{
namespace trtcv
{

// ── 构造 / 析构 ────────────────────────────────────────────────

TrtCutie::TrtCutie(const core::CutieConfig& config,
                   std::shared_ptr<linden::log::ILogger> logger)
    : logger_(logger ? std::move(logger) : linden::log::StdLogger::instance())
{
    namespace fs = std::filesystem;
    const std::string& dir = config.model_dir;
    int dev = (config.device == Device::kCUDA) ? config.device_id : -1;

    if (dev < 0)
    {
        logger_->error("TrtCutie: GPU 模式需要 device_id >= 0 (Device::kCUDA)");
        throw std::runtime_error("TrtCutie: GPU 模式需要 CUDA 设备");
    }

    // 初始化 GPU 内存分配器
    gpu_alloc_ = std::make_unique<ortcore::GpuMemoryAllocator>(dev);

    // 创建 CUDA stream
    cudaError_t err = cudaStreamCreate(&stream_);
    if (err != cudaSuccess)
    {
        logger_->error("TrtCutie: cudaStreamCreate 失败: {}", cudaGetErrorString(err));
        throw std::runtime_error("TrtCutie: cudaStreamCreate 失败");
    }

    // 辅助函数：构建 ONNX 和 Engine 文件路径
    auto get_paths = [&](const std::string& name) -> std::pair<std::string, std::string>
    {
        if (config.model_prefix.empty())
        {
            logger_->error("TrtCutie: model_prefix 必须设置（例如 \"cutie-base-mega\"）");
            throw std::runtime_error("TrtCutie: model_prefix 为空");
        }
        std::string onnx_path = (fs::path(dir) / (config.model_prefix + "_" + name + ".onnx")).string();
        std::string engine_path = (fs::path(dir) / (config.model_prefix + "_" + name + ".engine")).string();

        if (!fs::exists(onnx_path))
        {
            logger_->error("TrtCutie: ONNX 文件不存在: {}", onnx_path);
            throw std::runtime_error("TrtCutie: ONNX 文件不存在: " + onnx_path);
        }

        return {onnx_path, engine_path};
    };

    // TensorRT 引擎构建配置
    trtcore::BuildConfig build_config;
    build_config.max_workspace_size = 2ULL << 30;  // 2GB
    build_config.enable_fp16 = false;  // 默认 FP32，后续可配置
    build_config.log_level = nvinfer1::ILogger::Severity::kWARNING;

    // 创建引擎构建器
    trtcore::TrtEngineBuilder builder(logger_);

    // 加载 pixel_encoder
    {
        auto [onnx_path, engine_path] = get_paths("pixel_encoder");
        logger_->info("TrtCutie: 加载 pixel_encoder...");
        auto engine = builder.get_or_build_engine(onnx_path, engine_path, build_config);
        pixel_encoder_ = std::make_unique<trtcore::TrtHandler>(std::move(engine), dev, logger_);
        logger_->info("TrtCutie: pixel_encoder 加载成功");

        // 读取模型输入尺寸
        auto shape = pixel_encoder_->get_binding_shape("image");
        if (shape.size() >= 4 && shape[2] > 0 && shape[3] > 0)
        {
            model_h_ = static_cast<int>(shape[2]);
            model_w_ = static_cast<int>(shape[3]);
            logger_->info("TrtCutie: 模型输入尺寸 = {}x{}", model_h_, model_w_);
        }
        else
        {
            logger_->warn("TrtCutie: 无法从 pixel_encoder 读取静态输入尺寸");
        }
    }

    // 加载 key_projection
    {
        auto [onnx_path, engine_path] = get_paths("key_projection");
        logger_->info("TrtCutie: 加载 key_projection...");
        auto engine = builder.get_or_build_engine(onnx_path, engine_path, build_config);
        key_projection_ = std::make_unique<trtcore::TrtHandler>(std::move(engine), dev, logger_);
        logger_->info("TrtCutie: key_projection 加载成功");
    }

    // 加载 mask_encoder
    {
        auto [onnx_path, engine_path] = get_paths("mask_encoder");
        logger_->info("TrtCutie: 加载 mask_encoder...");
        auto engine = builder.get_or_build_engine(onnx_path, engine_path, build_config);
        mask_encoder_ = std::make_unique<trtcore::TrtHandler>(std::move(engine), dev, logger_);
        logger_->info("TrtCutie: mask_encoder 加载成功");

        // 读取静态对象数 N
        auto sensory_shape = mask_encoder_->get_binding_shape("sensory");
        if (sensory_shape.size() >= 2 && sensory_shape[1] > 0)
        {
            model_n_obj_ = static_cast<int>(sensory_shape[1]);
            logger_->info("TrtCutie: 模型编译支持 N={} 个对象（静态）", model_n_obj_);
        }
    }

    // 加载 pixel_fuser
    {
        auto [onnx_path, engine_path] = get_paths("pixel_fuser");
        logger_->info("TrtCutie: 加载 pixel_fuser...");
        auto engine = builder.get_or_build_engine(onnx_path, engine_path, build_config);
        pixel_fuser_ = std::make_unique<trtcore::TrtHandler>(std::move(engine), dev, logger_);
        logger_->info("TrtCutie: pixel_fuser 加载成功");
    }

    // 加载 object_transformer
    {
        auto [onnx_path, engine_path] = get_paths("object_transformer");
        logger_->info("TrtCutie: 加载 object_transformer...");
        auto engine = builder.get_or_build_engine(onnx_path, engine_path, build_config);
        object_transformer_ = std::make_unique<trtcore::TrtHandler>(std::move(engine), dev, logger_);
        logger_->info("TrtCutie: object_transformer 加载成功");
    }

    // 加载 mask_decoder
    {
        auto [onnx_path, engine_path] = get_paths("mask_decoder");
        logger_->info("TrtCutie: 加载 mask_decoder...");
        auto engine = builder.get_or_build_engine(onnx_path, engine_path, build_config);
        mask_decoder_ = std::make_unique<trtcore::TrtHandler>(std::move(engine), dev, logger_);
        logger_->info("TrtCutie: mask_decoder 加载成功");
    }

    logger_->info("TrtCutie: 所有 6 个 TensorRT 子模块加载完成（GPU 模式，dir={}）", dir);
}

TrtCutie::~TrtCutie()
{
    if (stream_)
    {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
}

TrtCutie::TrtCutie(TrtCutie&&) noexcept = default;
TrtCutie& TrtCutie::operator=(TrtCutie&&) noexcept = default;

// ── 子模块推理（GPU in/out）────────────────────────────────────

TrtCutie::ImageFeatures TrtCutie::encode_image(Ort::Value& image)
{
    // 输入：image [1, 3, H, W] (GPU)
    // 输出：f16, f8, f4, pix_feat (GPU)

    void* image_ptr = image.GetTensorMutableData<float>();

    // 准备输出张量
    ImageFeatures feat;
    // 从 pixel_encoder 获取输出形状
    auto f16_shape = pixel_encoder_->get_binding_shape("f16");
    auto f8_shape = pixel_encoder_->get_binding_shape("f8");
    auto f4_shape = pixel_encoder_->get_binding_shape("f4");
    auto pix_feat_shape = pixel_encoder_->get_binding_shape("pix_feat");

    feat.f16 = gpu_alloc_->allocate(f16_shape);
    feat.f8 = gpu_alloc_->allocate(f8_shape);
    feat.f4 = gpu_alloc_->allocate(f4_shape);
    feat.pix_feat = gpu_alloc_->allocate(pix_feat_shape);

    // 构建输入输出映射
    std::unordered_map<std::string, void*> inputs = {{"image", image_ptr}};
    std::unordered_map<std::string, void*> outputs = {
        {"f16", feat.f16.GetTensorMutableData<float>()},
        {"f8", feat.f8.GetTensorMutableData<float>()},
        {"f4", feat.f4.GetTensorMutableData<float>()},
        {"pix_feat", feat.pix_feat.GetTensorMutableData<float>()}
    };

    // 执行推理
    pixel_encoder_->infer_async(inputs, outputs, stream_);
    cudaStreamSynchronize(stream_);

    return feat;
}

TrtCutie::KeyFeatures TrtCutie::transform_key(Ort::Value& f16)
{
    // 输入：f16 [1, C16, H/16, W/16] (GPU)
    // 输出：key, shrinkage, selection (GPU)

    void* f16_ptr = f16.GetTensorMutableData<float>();

    KeyFeatures feat;
    auto key_shape = key_projection_->get_binding_shape("key");
    auto shrinkage_shape = key_projection_->get_binding_shape("shrinkage");
    auto selection_shape = key_projection_->get_binding_shape("selection");

    feat.key = gpu_alloc_->allocate(key_shape);
    feat.shrinkage = gpu_alloc_->allocate(shrinkage_shape);
    feat.selection = gpu_alloc_->allocate(selection_shape);

    std::unordered_map<std::string, void*> inputs = {{"f16", f16_ptr}};
    std::unordered_map<std::string, void*> outputs = {
        {"key", feat.key.GetTensorMutableData<float>()},
        {"shrinkage", feat.shrinkage.GetTensorMutableData<float>()},
        {"selection", feat.selection.GetTensorMutableData<float>()}
    };

    key_projection_->infer_async(inputs, outputs, stream_);
    cudaStreamSynchronize(stream_);

    return feat;
}

TrtCutie::MaskEncoded TrtCutie::encode_mask(Ort::Value& image, Ort::Value& pix_feat,
                                            Ort::Value& sensory, Ort::Value& masks)
{
    // 输入：image [1, 3, H, W], pix_feat [1, pixel_dim, H/16, W/16],
    //       sensory [num_objects, sensory_dim, H/16, W/16], masks [num_objects, 1, H/16, W/16]
    // 输出：mask_value, new_sensory, obj_summaries

    void* image_ptr = image.GetTensorMutableData<float>();
    void* pix_feat_ptr = pix_feat.GetTensorMutableData<float>();
    void* sensory_ptr = sensory.GetTensorMutableData<float>();
    void* masks_ptr = masks.GetTensorMutableData<float>();

    MaskEncoded feat;
    auto mask_value_shape = mask_encoder_->get_binding_shape("mask_value");
    auto new_sensory_shape = mask_encoder_->get_binding_shape("new_sensory");
    auto obj_summaries_shape = mask_encoder_->get_binding_shape("obj_summaries");

    feat.mask_value = gpu_alloc_->allocate(mask_value_shape);
    feat.new_sensory = gpu_alloc_->allocate(new_sensory_shape);
    feat.obj_summaries = gpu_alloc_->allocate(obj_summaries_shape);

    std::unordered_map<std::string, void*> inputs = {
        {"image", image_ptr},
        {"pix_feat", pix_feat_ptr},
        {"sensory", sensory_ptr},
        {"masks", masks_ptr}
    };
    std::unordered_map<std::string, void*> outputs = {
        {"mask_value", feat.mask_value.GetTensorMutableData<float>()},
        {"new_sensory", feat.new_sensory.GetTensorMutableData<float>()},
        {"obj_summaries", feat.obj_summaries.GetTensorMutableData<float>()}
    };

    mask_encoder_->infer_async(inputs, outputs, stream_);
    cudaStreamSynchronize(stream_);

    return feat;
}

Ort::Value TrtCutie::pixel_fusion(Ort::Value& pix_feat, Ort::Value& pixel,
                                  Ort::Value& sensory, Ort::Value& last_mask)
{
    // 输入：pix_feat [1, pixel_dim, H/16, W/16], pixel [1, C, H/16, W/16],
    //       sensory [1, sensory_dim, H/16, W/16], last_mask [1, 1, H/16, W/16]
    // 输出：fused_pixel [1, C, H/16, W/16]

    void* pix_feat_ptr = pix_feat.GetTensorMutableData<float>();
    void* pixel_ptr = pixel.GetTensorMutableData<float>();
    void* sensory_ptr = sensory.GetTensorMutableData<float>();
    void* last_mask_ptr = last_mask.GetTensorMutableData<float>();

    auto output_shape = pixel_fuser_->get_binding_shape("output");
    Ort::Value output = gpu_alloc_->allocate(output_shape);

    std::unordered_map<std::string, void*> inputs = {
        {"pix_feat", pix_feat_ptr},
        {"pixel", pixel_ptr},
        {"sensory", sensory_ptr},
        {"last_mask", last_mask_ptr}
    };
    std::unordered_map<std::string, void*> outputs = {
        {"output", output.GetTensorMutableData<float>()}
    };

    pixel_fuser_->infer_async(inputs, outputs, stream_);
    cudaStreamSynchronize(stream_);

    return output;
}

Ort::Value TrtCutie::readout_query(Ort::Value& pixel_readout, Ort::Value& obj_memory)
{
    // 输入：pixel_readout [1, C, H/16, W/16], obj_memory [1, num_objects, Q, C]
    // 输出：refined_readout [1, num_objects, C, H/16, W/16]

    void* pixel_readout_ptr = pixel_readout.GetTensorMutableData<float>();
    void* obj_memory_ptr = obj_memory.GetTensorMutableData<float>();

    auto output_shape = object_transformer_->get_binding_shape("output");
    Ort::Value output = gpu_alloc_->allocate(output_shape);

    std::unordered_map<std::string, void*> inputs = {
        {"pixel_readout", pixel_readout_ptr},
        {"obj_memory", obj_memory_ptr}
    };
    std::unordered_map<std::string, void*> outputs = {
        {"output", output.GetTensorMutableData<float>()}
    };

    object_transformer_->infer_async(inputs, outputs, stream_);
    cudaStreamSynchronize(stream_);

    return output;
}

TrtCutie::SegmentResult TrtCutie::segment(Ort::Value& f8, Ort::Value& f4,
                                          Ort::Value& memory_readout, Ort::Value& sensory)
{
    // 输入：f8 [1, C8, H/8, W/8], f4 [1, C4, H/4, W/4],
    //       memory_readout [1, num_objects, C, H/16, W/16],
    //       sensory [num_objects, sensory_dim, H/16, W/16]
    // 输出：new_sensory, logits

    void* f8_ptr = f8.GetTensorMutableData<float>();
    void* f4_ptr = f4.GetTensorMutableData<float>();
    void* memory_readout_ptr = memory_readout.GetTensorMutableData<float>();
    void* sensory_ptr = sensory.GetTensorMutableData<float>();

    SegmentResult result;
    auto new_sensory_shape = mask_decoder_->get_binding_shape("new_sensory");
    auto logits_shape = mask_decoder_->get_binding_shape("logits");

    result.new_sensory = gpu_alloc_->allocate(new_sensory_shape);
    result.logits = gpu_alloc_->allocate(logits_shape);

    std::unordered_map<std::string, void*> inputs = {
        {"f8", f8_ptr},
        {"f4", f4_ptr},
        {"memory_readout", memory_readout_ptr},
        {"sensory", sensory_ptr}
    };
    std::unordered_map<std::string, void*> outputs = {
        {"new_sensory", result.new_sensory.GetTensorMutableData<float>()},
        {"logits", result.logits.GetTensorMutableData<float>()}
    };

    mask_decoder_->infer_async(inputs, outputs, stream_);
    cudaStreamSynchronize(stream_);

    return result;
}

}  // namespace trtcv
}  // namespace cutie
