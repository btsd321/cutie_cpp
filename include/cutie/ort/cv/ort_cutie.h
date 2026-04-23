#ifndef CUTIE_ORT_CV_ORT_CUTIE_H
#define CUTIE_ORT_CV_ORT_CUTIE_H

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <linden_logger/logger_interface.hpp>

#include "cutie/ort/core/gpu_memory.h"
#include "cutie/ort/core/ort_config.h"
#include "cutie/ort/core/ort_utils.h"

namespace cutie
{

// Forward declare
namespace core
{
struct CutieConfig;
}

namespace ortcv
{

/// Manages 6 ONNX Runtime sessions for the Cutie model submodules.
/// All inputs and outputs are GPU Ort::Value tensors (CUDA memory).
///
/// Submodules:
///   1. pixel_encoder   - image → multi-scale features + pix_feat
///   2. key_projection  - f16 → key, shrinkage, selection
///   3. mask_encoder    - image + features + sensory + masks → value, sensory, summaries
///   4. pixel_fuser     - fuses pixel features with memory readout (per-object)
///   5. object_transformer - refines readout with object queries
///   6. mask_decoder    - decodes final mask + sensory update
class OrtCutie
{
public:
    // ── Result structs ──────────────────────────────────────────────

    struct ImageFeatures
    {
        Ort::Value f16{nullptr};       // [1, C16, H/16, W/16]
        Ort::Value f8{nullptr};        // [1, C8, H/8, W/8]
        Ort::Value f4{nullptr};        // [1, C4, H/4, W/4]
        Ort::Value pix_feat{nullptr};  // [1, pixel_dim, H/16, W/16]
    };

    struct KeyFeatures
    {
        Ort::Value key{nullptr};        // [1, key_dim, H/16, W/16]
        Ort::Value shrinkage{nullptr};  // [1, 1, H/16, W/16]
        Ort::Value selection{nullptr};  // [1, key_dim, H/16, W/16]
    };

    struct MaskEncoded
    {
        Ort::Value mask_value{nullptr};     // [1, value_dim, num_objects, H/16, W/16]
        Ort::Value new_sensory{nullptr};    // [num_objects, sensory_dim, H/16, W/16]
        Ort::Value obj_summaries{nullptr};  // [num_objects, 1, num_queries, embed_dim+1]
    };

    struct SegmentResult
    {
        Ort::Value new_sensory{nullptr};  // [num_objects, sensory_dim, H/16, W/16]
        Ort::Value logits{nullptr};       // [num_objects, 1, H/4, W/4]
    };

    // ── Construction ────────────────────────────────────────────────

    explicit OrtCutie(const core::CutieConfig& config,
                      std::shared_ptr<linden::log::ILogger> logger = nullptr);
    ~OrtCutie();

    OrtCutie(const OrtCutie&) = delete;
    OrtCutie& operator=(const OrtCutie&) = delete;
    OrtCutie(OrtCutie&&) noexcept;
    OrtCutie& operator=(OrtCutie&&) noexcept;

    // ── Submodule inference methods (GPU in/out) ───────────────────

    /// Encode image through the pixel encoder backbone.
    /// Input:  image [1, 3, H, W] float32 (GPU, RGB, ImageNet-normalized)
    ImageFeatures encode_image(Ort::Value& image);

    /// Project f16 features to key, shrinkage, and selection.
    /// Input:  f16 [1, C16, H/16, W/16] float32 (GPU)
    KeyFeatures transform_key(Ort::Value& f16);

    /// Encode mask (with all objects batched) to memory values.
    /// All inputs are GPU tensors.
    MaskEncoded encode_mask(Ort::Value& image, Ort::Value& pix_feat, Ort::Value& sensory,
                            Ort::Value& masks);

    /// Fuse pixel features with memory readout (per-object call).
    /// All inputs are GPU tensors.
    Ort::Value pixel_fusion(Ort::Value& pix_feat, Ort::Value& pixel, Ort::Value& sensory,
                            Ort::Value& last_mask);

    /// Apply object transformer to refine pixel readout.
    /// All inputs are GPU tensors.
    Ort::Value readout_query(Ort::Value& pixel_readout, Ort::Value& obj_memory);

    /// Decode mask and update sensory.
    /// All inputs are GPU tensors.
    SegmentResult segment(Ort::Value& f8, Ort::Value& f4, Ort::Value& memory_readout,
                          Ort::Value& sensory);

    /// Access the GPU memory allocator.
    ortcore::GpuMemoryAllocator& gpu_alloc() { return *gpu_alloc_; }
    const ortcore::GpuMemoryAllocator& gpu_alloc() const { return *gpu_alloc_; }

    /// Access the GPU memory info (for tensor creation).
    const Ort::MemoryInfo& memory_info() const { return gpu_alloc_->memory_info(); }

    /// Return the expected input height/width of pixel_encoder (read from ONNX metadata).
    int model_input_h() const { return model_h_; }
    int model_input_w() const { return model_w_; }

private:
    struct SessionBundle;  // fwd decl for internal session wrapper
    using SessionPtr = std::unique_ptr<SessionBundle>;

    std::shared_ptr<linden::log::ILogger> logger_;  // 必须在 env_ 之前，因为 env_ 构造时引用 logger_
    Ort::Env env_;
    std::unique_ptr<ortcore::GpuMemoryAllocator> gpu_alloc_;

    int model_h_ = 0;
    int model_w_ = 0;
    int model_n_obj_ = 0;

    SessionPtr pixel_encoder_;
    SessionPtr key_projection_;
    SessionPtr mask_encoder_;
    SessionPtr pixel_fuser_;
    SessionPtr object_transformer_;
    SessionPtr mask_decoder_;

    SessionPtr create_session(const std::string& onnx_path, int device_id);

    /// Run a session using IO Binding (GPU in/out).
    std::vector<Ort::Value> run_session(SessionBundle& bundle,
                                        std::vector<Ort::Value>& inputs);
};

}  // namespace ortcv
}  // namespace cutie

#endif  // CUTIE_ORT_CV_ORT_CUTIE_H
