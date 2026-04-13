#ifndef CUTIE_CORE_PROCESSOR_H
#define CUTIE_CORE_PROCESSOR_H

#include <memory>
#include <string>
#include <vector>

#include <linden_logger/logger_interface.hpp>
#include <opencv2/core.hpp>

#include "cutie/types.h"

namespace cutie
{
namespace core
{

struct CutieConfig
{
    ModelVariant variant = ModelVariant::kBase;
    std::string model_dir;
    /// Prefix for ONNX submodule filenames (required).
    /// Derived from the .pth weights filename, e.g. "cutie-base-mega".
    /// Files must be named "<model_prefix>_pixel_encoder.onnx" etc.
    std::string model_prefix;
    Device device = Device::kCUDA;
    int device_id = 0;
    bool single_object = false;

    // Inference parameters
    int max_internal_size = 480;
    int mem_every = 5;
    int top_k = 30;
    int chunk_size = -1;
    int stagger_updates = 5;

    // Working memory
    int max_mem_frames = 5;

    // Long-term memory
    bool use_long_term = false;
    struct LongTermConfig
    {
        bool count_usage = true;
        int max_mem_frames = 10;
        int min_mem_frames = 5;
        int num_prototypes = 128;
        int max_num_tokens = 10000;
        int buffer_tokens = 2000;
    } long_term;

    // Model dimensions (auto-filled based on variant)
    struct ModelDims
    {
        int key_dim = 64;
        int value_dim = 256;
        int sensory_dim = 256;
        int pixel_dim = 256;
        int f16_dim = 1024;
        int f8_dim = 512;
        int f4_dim = 256;
        int num_queries = 16;
    } model;

    static CutieConfig base_default(const std::string& model_dir);
    static CutieConfig small_default(const std::string& model_dir);
};

struct StepOptions
{
    bool idx_mask = true;
    bool end = false;
    bool force_permanent = false;
};

/// Stateful video object segmentation processor.
/// One instance per video. Not thread-safe.
class CutieProcessor
{
public:
    explicit CutieProcessor(const CutieConfig& config,
                            std::shared_ptr<linden::log::ILogger> logger = nullptr);
    ~CutieProcessor();

    CutieProcessor(const CutieProcessor&) = delete;
    CutieProcessor& operator=(const CutieProcessor&) = delete;
    CutieProcessor(CutieProcessor&&) noexcept;
    CutieProcessor& operator=(CutieProcessor&&) noexcept;

    // Core inference
    types::CutieMask step(const cv::Mat& image, const cv::Mat& mask = cv::Mat(),
                          const std::vector<ObjectId>& objects = {});

    types::CutieMask step(const cv::Mat& image, const cv::Mat& mask,
                          const std::vector<ObjectId>& objects, const StepOptions& options);

    // Object management
    void delete_objects(const std::vector<ObjectId>& objects);
    std::vector<ObjectId> active_objects() const;
    int num_objects() const;

    // Memory management
    void clear_memory();
    void clear_non_permanent_memory();
    void clear_sensory_memory();

    // Configuration
    void update_config(const CutieConfig& config);
    const CutieConfig& config() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace core
}  // namespace cutie

#endif  // CUTIE_CORE_PROCESSOR_H
