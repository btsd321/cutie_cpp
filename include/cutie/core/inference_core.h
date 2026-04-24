/**
 * @file inference_core.h
 * @brief Main inference pipeline orchestration.
 *
 * Coordinates the full Cutie inference pipeline: image encoding, memory read,
 * segmentation, and memory write. Manages frame-level state (frame counter,
 * memory scheduling, feature caching).
 */

#pragma once

#include <memory>
#include <vector>

#include <linden_logger/logger_interface.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

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

/**
 * @class InferenceCore
 * @brief Orchestrates the full Cutie inference pipeline.
 *
 * Manages the complete inference flow for each frame:
 * 1. Image encoding (pixel_encoder)
 * 2. Memory read (key_projection, memory query)
 * 3. Segmentation (mask_encoder, pixel_fuser, object_transformer, mask_decoder)
 * 4. Memory write (feature storage)
 *
 * Maintains frame-level state: frame counter (ti), memory scheduling,
 * feature caching, and object tracking.
 *
 * 推理核心协调完整的推理流程，包括图像编码、内存读取、分割和内存写入。
 * 支持 CPU 和 GPU 推理路径，管理帧级状态和内存调度。
 */
class InferenceCore
{
public:
    /**
     * @brief Construct inference core.
     *
     * @param config Inference configuration
     * @param logger Optional logger instance
     * @throws std::runtime_error if model initialization fails
     */
    explicit InferenceCore(const CutieConfig& config,
                           std::shared_ptr<linden::log::ILogger> logger = nullptr);
    ~InferenceCore();

    InferenceCore(const InferenceCore&) = delete;
    InferenceCore& operator=(const InferenceCore&) = delete;
    InferenceCore(InferenceCore&&) noexcept;
    InferenceCore& operator=(InferenceCore&&) noexcept;

    // ─── CPU Inference Interface ────────────────────────────────────

    /**
     * @brief Process one frame (CPU path).
     *
     * Runs full inference pipeline on CPU input, returns CPU output.
     * 在 CPU 上运行完整推理流程。
     *
     * @param image Input frame (BGR, CV_8UC3)
     * @param mask Optional: First-frame mask (CV_8UC1, pixel value = ObjectId)
     * @param objects Optional: List of object IDs in mask (required if mask provided)
     * @param end Mark as end-of-sequence (triggers memory consolidation)
     * @param force_permanent Force this frame to permanent memory
     * @return Probability map [num_objects+1, H, W] (CV_32FC1)
     */
    cv::Mat step(const cv::Mat& image, const cv::Mat& mask = cv::Mat(),
                 const std::vector<ObjectId>& objects = {}, bool end = false,
                 bool force_permanent = false);

    // ─── GPU Inference Interface ────────────────────────────────────

    /**
     * @brief Process one frame (GPU path, zero-copy).
     *
     * Full GPU pipeline: input/output remain on GPU, no CPU transfers during inference.
     * 在 GPU 上运行完整推理流程，所有数据保持在 GPU 上。
     *
     * @param image_gpu Input frame on GPU (BGR, CV_8UC3)
     * @param mask_gpu Optional: First-frame mask on GPU (CV_32SC1, pixel value = ObjectId)
     * @param objects Optional: List of object IDs in mask
     * @param end Mark as end-of-sequence
     * @param force_permanent Force this frame to permanent memory
     * @return GPU segmentation result (index_mask + prob + object_ids)
     */
    types::GpuCutieMask step_gpu(const cv::cuda::GpuMat& image_gpu,
                                 const cv::cuda::GpuMat& mask_gpu = cv::cuda::GpuMat(),
                                 const std::vector<ObjectId>& objects = {}, bool end = false,
                                 bool force_permanent = false);

    // ─── Object Management ──────────────────────────────────────────

    /**
     * @brief Stop tracking specified objects.
     *
     * 删除指定对象，释放相关内存。
     *
     * @param objects List of ObjectIds to delete
     */
    void delete_objects(const std::vector<ObjectId>& objects);

    /**
     * @brief Get list of currently tracked objects.
     * @return Vector of active ObjectIds
     */
    std::vector<ObjectId> active_objects() const;

    /**
     * @brief Get number of currently tracked objects.
     * @return Count of active objects
     */
    int num_objects() const;

    // ─── Memory Management ──────────────────────────────────────────

    /**
     * @brief Clear all memory (working, long-term, sensory).
     *
     * 清除所有内存，重置为初始状态。
     */
    void clear_memory();

    /**
     * @brief Clear non-permanent memory (working + long-term).
     *
     * 清除非永久内存，保留永久标记的特征。
     */
    void clear_non_permanent_memory();

    /**
     * @brief Clear sensory memory (per-object visual context).
     *
     * 清除感知记忆，重置对象的 GRU 隐藏状态。
     */
    void clear_sensory_memory();

    // ─── Configuration ──────────────────────────────────────────────

    /**
     * @brief Update configuration (hot-update).
     *
     * 更新配置参数，某些参数可以在推理过程中修改。
     *
     * @param config New configuration
     */
    void update_config(const CutieConfig& config);

private:
    /// Implementation (PIMPL pattern)
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace core
}  // namespace cutie
