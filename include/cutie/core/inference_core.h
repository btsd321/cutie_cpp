#ifndef CUTIE_CORE_INFERENCE_CORE_H
#define CUTIE_CORE_INFERENCE_CORE_H

#include <memory>
#include <vector>

#include <linden_logger/logger_interface.hpp>
#include <opencv2/core.hpp>

#include "cutie/types.h"

namespace cutie
{

namespace core
{

struct CutieConfig;
class ObjectManager;
class MemoryManager;
class ImageFeatureStore;

}  // namespace core

namespace ortcv
{
class OrtCutie;
}

namespace core
{

/// Orchestrates the full Cutie inference pipeline:
///   image encoding → memory read → segmentation → memory write.
/// Manages frame-level state: ti counter, memory scheduling, feature cache.
class InferenceCore
{
public:
    explicit InferenceCore(const CutieConfig& config,
                           std::shared_ptr<linden::log::ILogger> logger = nullptr);
    ~InferenceCore();

    InferenceCore(const InferenceCore&) = delete;
    InferenceCore& operator=(const InferenceCore&) = delete;
    InferenceCore(InferenceCore&&) noexcept;
    InferenceCore& operator=(InferenceCore&&) noexcept;

    /// Process one frame. Returns probability map [num_objects+1, H, W].
    /// If mask is provided, it is used as the ground-truth for this frame.
    /// objects: list of object IDs in the mask (only needed when mask is given).
    cv::Mat step(const cv::Mat& image, const cv::Mat& mask = cv::Mat(),
                 const std::vector<ObjectId>& objects = {}, bool end = false,
                 bool force_permanent = false);

    // Object management
    void delete_objects(const std::vector<ObjectId>& objects);
    std::vector<ObjectId> active_objects() const;
    int num_objects() const;

    // Memory management
    void clear_memory();
    void clear_non_permanent_memory();
    void clear_sensory_memory();

    // Config hot-update
    void update_config(const CutieConfig& config);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace core
}  // namespace cutie

#endif  // CUTIE_CORE_INFERENCE_CORE_H
