/**
 * @file trt_cutie.h
 * @brief TensorRT Cutie 模型封装
 *
 * 封装 6 个 TensorRT 子模块，提供与 OrtCutie 相同的接口。
 * 管理完整的 Cutie 推理流程，支持全 GPU 数据流。
 */

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <linden_logger/logger_interface.hpp>

#include "cutie/common/gpu_memory.h"
#include "cutie/ort/core/ort_config.h"
#include "cutie/trt/core/trt_handler.h"

namespace cutie
{

// Forward declare
namespace core
{
struct CutieConfig;
}

namespace trtcv
{

/**
 * @class TrtCutie
 * @brief TensorRT Cutie 模型封装
 *
 * 管理 6 个 TensorRT 引擎，实现完整的 Cutie 推理流程：
 * 1. **pixel_encoder** - Image → multi-scale features + pixel features
 * 2. **key_projection** - f16 → key, shrinkage, selection
 * 3. **mask_encoder** - Image + features + sensory + masks → value, sensory, summaries
 * 4. **pixel_fuser** - Fuses pixel features with memory readout (per-object)
 * 5. **object_transformer** - Refines readout with object queries
 * 6. **mask_decoder** - Decodes final mask + sensory update
 *
 * 所有张量操作都在 GPU 上进行，支持零拷贝推理。
 * 提供与 OrtCutie 相同的接口，便于后端切换。
 */
class TrtCutie
{
public:
    // ── Result Structures ───────────────────────────────────────────

    /**
     * @struct ImageFeatures
     * @brief Multi-scale image features from pixel encoder.
     */
    struct ImageFeatures
    {
        Ort::Value f16{nullptr};       ///< [1, C16, H/16, W/16] features at 1/16 scale
        Ort::Value f8{nullptr};        ///< [1, C8, H/8, W/8] features at 1/8 scale
        Ort::Value f4{nullptr};        ///< [1, C4, H/4, W/4] features at 1/4 scale
        Ort::Value pix_feat{nullptr};  ///< [1, pixel_dim, H/16, W/16] pixel features
    };

    /**
     * @struct KeyFeatures
     * @brief Key projection features for memory matching.
     */
    struct KeyFeatures
    {
        Ort::Value key{nullptr};        ///< [1, key_dim, H/16, W/16] memory key
        Ort::Value shrinkage{nullptr};  ///< [1, 1, H/16, W/16] shrinkage factor
        Ort::Value selection{nullptr};  ///< [1, key_dim, H/16, W/16] selection mask
    };

    /**
     * @struct MaskEncoded
     * @brief Encoded mask features and object summaries.
     */
    struct MaskEncoded
    {
        Ort::Value mask_value{nullptr};     ///< [1, value_dim, num_objects, H/16, W/16] mask values
        Ort::Value new_sensory{nullptr};    ///< [num_objects, sensory_dim, H/16, W/16] sensory state
        Ort::Value obj_summaries{nullptr};  ///< [num_objects, 1, num_queries, embed_dim+1] object summaries
    };

    /**
     * @struct SegmentResult
     * @brief Final segmentation result.
     */
    struct SegmentResult
    {
        Ort::Value new_sensory{nullptr};  ///< [num_objects, sensory_dim, H/16, W/16] updated sensory
        Ort::Value logits{nullptr};       ///< [num_objects, 1, H/4, W/4] segmentation logits
    };

    // ── Construction ────────────────────────────────────────────────

    /**
     * @brief Construct TrtCutie and load all submodule engines.
     *
     * 从 ONNX 模型构建或加载缓存的 TensorRT 引擎。
     *
     * @param config Cutie configuration (contains model paths)
     * @param logger Optional logger instance
     * @throws std::runtime_error if engine loading fails
     */
    explicit TrtCutie(const core::CutieConfig& config,
                      std::shared_ptr<linden::log::ILogger> logger = nullptr);
    ~TrtCutie();

    TrtCutie(const TrtCutie&) = delete;
    TrtCutie& operator=(const TrtCutie&) = delete;
    TrtCutie(TrtCutie&&) noexcept;
    TrtCutie& operator=(TrtCutie&&) noexcept;

    // ── Submodule Inference Methods (GPU in/out) ───────────────────

    /**
     * @brief Encode image through pixel encoder backbone.
     *
     * 通过像素编码器提取多尺度图像特征。
     *
     * @param image Input image [1, 3, H, W] float32 (GPU, RGB, ImageNet-normalized)
     * @return Multi-scale features (f16, f8, f4, pix_feat)
     */
    ImageFeatures encode_image(Ort::Value& image);

    /**
     * @brief Project f16 features to key, shrinkage, and selection.
     *
     * 将 f16 特征投影为内存匹配所需的 key、shrinkage 和 selection。
     *
     * @param f16 Input features [1, C16, H/16, W/16] float32 (GPU)
     * @return Key features (key, shrinkage, selection)
     */
    KeyFeatures transform_key(Ort::Value& f16);

    /**
     * @brief Encode mask to memory values and sensory state.
     *
     * 将掩码编码为内存值和感知状态。
     * 所有输入都是 GPU 张量。
     *
     * @param image Input image [1, 3, H, W]
     * @param pix_feat Pixel features [1, pixel_dim, H/16, W/16]
     * @param sensory Sensory state [num_objects, sensory_dim, H/16, W/16]
     * @param masks Object masks [num_objects, 1, H/16, W/16]
     * @return Encoded mask features (mask_value, new_sensory, obj_summaries)
     */
    MaskEncoded encode_mask(Ort::Value& image, Ort::Value& pix_feat, Ort::Value& sensory,
                            Ort::Value& masks);

    /**
     * @brief Fuse pixel features with memory readout (per-object).
     *
     * 融合像素特征与内存读出。
     * 所有输入都是 GPU 张量。
     *
     * @param pix_feat Pixel features [1, pixel_dim, H/16, W/16]
     * @param pixel Memory readout [1, C, H/16, W/16]
     * @param sensory Sensory state [1, sensory_dim, H/16, W/16]
     * @param last_mask Last frame mask [1, 1, H/16, W/16]
     * @return Fused pixel readout [1, C, H/16, W/16]
     */
    Ort::Value pixel_fusion(Ort::Value& pix_feat, Ort::Value& pixel, Ort::Value& sensory,
                            Ort::Value& last_mask);

    /**
     * @brief Apply object transformer to refine pixel readout.
     *
     * 使用对象变换器精化像素读出。
     * 所有输入都是 GPU 张量。
     *
     * @param pixel_readout Pixel readout [1, C, H/16, W/16]
     * @param obj_memory Object memory [1, num_objects, Q, C]
     * @return Refined readout [1, num_objects, C, H/16, W/16]
     */
    Ort::Value readout_query(Ort::Value& pixel_readout, Ort::Value& obj_memory);

    /**
     * @brief Decode mask and update sensory state.
     *
     * 解码最终分割掩码并更新感知状态。
     * 所有输入都是 GPU 张量。
     *
     * @param f8 Features at 1/8 scale [1, C8, H/8, W/8]
     * @param f4 Features at 1/4 scale [1, C4, H/4, W/4]
     * @param memory_readout Memory readout [1, num_objects, C, H/16, W/16]
     * @param sensory Sensory state [num_objects, sensory_dim, H/16, W/16]
     * @return Segmentation result (new_sensory, logits)
     */
    SegmentResult segment(Ort::Value& f8, Ort::Value& f4, Ort::Value& memory_readout,
                          Ort::Value& sensory);

    // ── Accessors ───────────────────────────────────────────────────

    /**
     * @brief Get GPU memory allocator.
     * @return Reference to GpuMemoryAllocator
     */
    ortcore::GpuMemoryAllocator& gpu_alloc() { return *gpu_alloc_; }

    /**
     * @brief Get GPU memory allocator (const).
     * @return Const reference to GpuMemoryAllocator
     */
    const ortcore::GpuMemoryAllocator& gpu_alloc() const { return *gpu_alloc_; }

    /**
     * @brief Get GPU memory info for tensor creation.
     * @return Reference to Ort::MemoryInfo
     */
    const Ort::MemoryInfo& memory_info() const { return gpu_alloc_->memory_info(); }

    /**
     * @brief Get expected input height of pixel encoder.
     *
     * 获取像素编码器的预期输入高度（从模型元数据读取）。
     *
     * @return Input height (0 if dynamic)
     */
    int model_input_h() const { return model_h_; }

    /**
     * @brief Get expected input width of pixel encoder.
     *
     * 获取像素编码器的预期输入宽度（从模型元数据读取）。
     *
     * @return Input width (0 if dynamic)
     */
    int model_input_w() const { return model_w_; }

private:
    std::shared_ptr<linden::log::ILogger> logger_;
    std::unique_ptr<ortcore::GpuMemoryAllocator> gpu_alloc_;

    int model_h_ = 0;  ///< Model input height
    int model_w_ = 0;  ///< Model input width
    int model_n_obj_ = 0;  ///< Model max objects

    // TensorRT handlers for each submodule
    std::unique_ptr<trtcore::TrtHandler> pixel_encoder_;
    std::unique_ptr<trtcore::TrtHandler> key_projection_;
    std::unique_ptr<trtcore::TrtHandler> mask_encoder_;
    std::unique_ptr<trtcore::TrtHandler> pixel_fuser_;
    std::unique_ptr<trtcore::TrtHandler> object_transformer_;
    std::unique_ptr<trtcore::TrtHandler> mask_decoder_;

    cudaStream_t stream_ = nullptr;  ///< CUDA stream for inference
};

}  // namespace trtcv
}  // namespace cutie
