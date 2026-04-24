#ifndef CUTIE_CORE_PROCESSOR_H
#define CUTIE_CORE_PROCESSOR_H

#include <memory>
#include <string>
#include <vector>

#include <linden_logger/logger_interface.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

#include "cutie/types.h"

namespace cutie
{
namespace core
{

/**
 * @struct CutieConfig
 * @brief Complete configuration for Cutie video object segmentation.
 *
 * Encapsulates all parameters controlling model behavior, memory management,
 * and inference strategy. Create instances using factory methods `base_default()`
 * or `small_default()`, then customize as needed.
 *
 * **Model Selection:**
 * - `variant`: Choose between base (higher accuracy) or small (faster)
 * - `model_dir`: Directory containing 6 ONNX submodule files
 * - `model_prefix`: Prefix for ONNX filenames (e.g., "cutie-base-mega")
 *
 * **Device & Compute:**
 * - `device`: CPU or CUDA GPU
 * - `device_id`: GPU device index (for multi-GPU systems)
 *
 * **Inference Parameters:**
 * - `max_internal_size`: Short-edge limit for dynamic resolution (default 480)
 * - `mem_every`: Add to memory every N frames (default 5)
 * - `top_k`: Top-K affinity selection for memory read (default 30)
 * - `chunk_size`: Batch size for transformer (default -1 = no chunking)
 * - `stagger_updates`: Spread memory updates across N frames (default 5)
 *
 * **Memory Configuration:**
 * - `max_mem_frames`: Working memory frame count (default 5)
 * - `use_long_term`: Enable long-term memory with consolidation (default false)
 * - `long_term`: Long-term memory parameters (prototypes, tokens, etc.)
 *
 * **Example:**
 * ```cpp
 * CutieConfig config = CutieConfig::base_default("./models/");
 * config.max_internal_size = 640;  // Support up to 640p
 * config.mem_every = 3;             // Update memory every 3 frames
 * CutieProcessor processor(config);
 * ```
 *
 * @see CutieProcessor for usage
 */
struct CutieConfig
{
    /// Model variant (base or small)
    ModelVariant variant = ModelVariant::kBase;

    /// Directory containing ONNX submodule files (required)
    std::string model_dir;

    /**
     * @brief Prefix for ONNX submodule filenames (required).
     *
     * Derived from the .pth weights filename, e.g., "cutie-base-mega".
     * ONNX files must be named:
     * - `{model_prefix}_pixel_encoder.onnx`
     * - `{model_prefix}_key_projection.onnx`
     * - `{model_prefix}_mask_encoder.onnx`
     * - `{model_prefix}_pixel_fuser.onnx`
     * - `{model_prefix}_object_transformer.onnx`
     * - `{model_prefix}_mask_decoder.onnx`
     */
    std::string model_prefix;

    /// Compute device (CPU or CUDA)
    Device device = Device::kCUDA;

    /// GPU device index (for multi-GPU systems)
    int device_id = 0;

    /// Single-object mode (optimization for single target)
    bool single_object = false;

    // ─── Inference Parameters ───────────────────────────────────────

    /**
     * @brief Short-edge limit for dynamic resolution (pixels).
     *
     * Input images are resized so the short edge ≤ max_internal_size,
     * preserving aspect ratio. Larger values increase memory/compute.
     * Default: 480 (supports up to 1080p with 16:9 aspect ratio).
     */
    int max_internal_size = 480;

    /**
     * @brief Add frame to memory every N frames.
     *
     * Controls memory update frequency. Larger values reduce memory growth
     * but may miss important frames. Default: 5.
     */
    int mem_every = 5;

    /**
     * @brief Top-K affinity selection for memory read.
     *
     * When reading from memory, select top-K most similar frames.
     * Larger values increase compute but may improve accuracy. Default: 30.
     */
    int top_k = 30;

    /**
     * @brief Batch size for transformer (chunked processing).
     *
     * If > 0, process transformer in chunks of this size.
     * If -1 (default), process all at once (no chunking).
     */
    int chunk_size = -1;

    /**
     * @brief Spread memory updates across N frames.
     *
     * Stagger memory operations to avoid compute spikes.
     * Default: 5 (update 1/5 of memory each frame).
     */
    int stagger_updates = 5;

    // ─── Working Memory ─────────────────────────────────────────────

    /**
     * @brief Maximum frames in working memory (FIFO buffer).
     *
     * Recent frames are stored in fast-access working memory.
     * Older frames are moved to long-term memory (if enabled).
     * Default: 5.
     */
    int max_mem_frames = 5;

    // ─── Long-Term Memory ───────────────────────────────────────────

    /// Enable long-term memory with consolidation
    bool use_long_term = false;

    /**
     * @struct LongTermConfig
     * @brief Long-term memory consolidation parameters.
     *
     * Long-term memory uses prototype-based compression to store
     * older frames efficiently. Reduces memory growth for long videos.
     */
    struct LongTermConfig
    {
        /// Track usage frequency for prototype selection
        bool count_usage = true;

        /// Max frames before consolidation
        int max_mem_frames = 10;

        /// Min frames to keep after consolidation
        int min_mem_frames = 5;

        /// Number of prototype clusters
        int num_prototypes = 128;

        /// Maximum tokens in long-term memory
        int max_num_tokens = 10000;

        /// Buffer tokens for consolidation
        int buffer_tokens = 2000;
    } long_term;

    // ─── Model Dimensions ───────────────────────────────────────────

    /**
     * @struct ModelDims
     * @brief Model architecture dimensions (auto-filled based on variant).
     *
     * These are automatically set by `base_default()` or `small_default()`.
     * Modify only if using custom model variants.
     */
    struct ModelDims
    {
        int key_dim = 64;        ///< Key projection dimension
        int value_dim = 256;     ///< Value dimension
        int sensory_dim = 256;   ///< Sensory memory dimension
        int pixel_dim = 256;     ///< Pixel feature dimension
        int f16_dim = 1024;      ///< 1/16 feature dimension (base variant)
        int f8_dim = 512;        ///< 1/8 feature dimension (base variant)
        int f4_dim = 256;        ///< 1/4 feature dimension (base variant)
        int num_queries = 16;    ///< Object transformer queries
    } model;

    /**
     * @brief Create default configuration for base variant.
     * @param model_dir Directory containing ONNX files
     * @return CutieConfig with base variant defaults
     */
    static CutieConfig base_default(const std::string& model_dir);

    /**
     * @brief Create default configuration for small variant.
     * @param model_dir Directory containing ONNX files
     * @return CutieConfig with small variant defaults
     */
    static CutieConfig small_default(const std::string& model_dir);
};

/**
 * @struct StepOptions
 * @brief Per-frame inference options.
 *
 * Customize behavior for individual frames without modifying global config.
 */
struct StepOptions
{
    /// Include index mask in output (default true)
    bool idx_mask = true;

    /// Mark this frame as end-of-sequence (triggers memory consolidation)
    bool end = false;

    /// Force this frame to be added to permanent memory
    bool force_permanent = false;
};

/**
 * @class CutieProcessor
 * @brief Stateful video object segmentation processor.
 *
 * Main entry point for Cutie inference. Maintains state across frames
 * (memory, object tracking, feature caching) for efficient video processing.
 *
 * **Key Properties:**
 * - **Stateful**: One instance per video sequence. Maintains frame counter,
 *   memory, and object tracking state.
 * - **Not thread-safe**: Use separate instances for parallel video processing.
 * - **PIMPL pattern**: Implementation hidden in `Impl` struct for ABI stability.
 *
 * **Typical Usage (CPU Path):**
 * ```cpp
 * CutieConfig config = CutieConfig::base_default("./models/");
 * CutieProcessor processor(config);
 *
 * // First frame: provide initial mask
 * cv::Mat first_frame = cv::imread("frame_0.jpg");
 * cv::Mat first_mask = cv::imread("mask_0.png", cv::IMREAD_GRAYSCALE);
 * std::vector<ObjectId> objects = {1, 2};  // Track 2 objects
 *
 * auto result = processor.step(first_frame, first_mask, objects);
 * // result.index_mask: H×W matrix with pixel values = ObjectId
 * // result.object_ids: {1, 2}
 *
 * // Subsequent frames: no mask needed
 * for (int i = 1; i < num_frames; ++i) {
 *     cv::Mat frame = cv::imread(fmt::format("frame_{}.jpg", i));
 *     auto result = processor.step(frame);
 *     // Process result...
 * }
 * ```
 *
 * **GPU Path (Zero-Copy):**
 * ```cpp
 * cv::cuda::GpuMat gpu_frame, gpu_mask;
 * // ... upload frames to GPU ...
 *
 * auto gpu_result = processor.step_gpu(gpu_frame, gpu_mask, objects);
 * // All data remains on GPU; download only when needed:
 * auto cpu_result = gpu_result.download();
 * ```
 *
 * **Memory Management:**
 * - Call `clear_memory()` to reset state (e.g., between videos)
 * - Call `clear_sensory_memory()` to reset per-object visual context
 * - Call `delete_objects()` to stop tracking specific objects
 *
 * @see CutieConfig for configuration
 * @see types::CutieMask for output format
 * @see types::GpuCutieMask for GPU output format
 */
class CutieProcessor
{
public:
    /**
     * @brief Construct processor with configuration.
     * @param config Inference configuration
     * @param logger Optional logger instance (uses default if nullptr)
     * @throws std::runtime_error if model files not found or ONNX parsing fails
     */
    explicit CutieProcessor(const CutieConfig& config,
                            std::shared_ptr<linden::log::ILogger> logger = nullptr);

    /// Destructor
    ~CutieProcessor();

    // Non-copyable, movable
    CutieProcessor(const CutieProcessor&) = delete;
    CutieProcessor& operator=(const CutieProcessor&) = delete;
    CutieProcessor(CutieProcessor&&) noexcept;
    CutieProcessor& operator=(CutieProcessor&&) noexcept;

    // ─── CPU Inference Interface ────────────────────────────────────

    /**
     * @brief Process frame (CPU path, default options).
     *
     * @param image Input frame (BGR, CV_8UC3)
     * @param mask Optional: First-frame mask (CV_8UC1, pixel value = ObjectId)
     * @param objects Optional: List of object IDs to track
     * @return Segmentation result (CPU-resident)
     *
     * @note On first frame, provide both `mask` and `objects`.
     *       On subsequent frames, omit both (or provide empty vectors).
     */
    types::CutieMask step(const cv::Mat& image, const cv::Mat& mask = cv::Mat(),
                          const std::vector<ObjectId>& objects = {});

    /**
     * @brief Process frame (CPU path, custom options).
     *
     * @param image Input frame (BGR, CV_8UC3)
     * @param mask First-frame mask (CV_8UC1, pixel value = ObjectId)
     * @param objects List of object IDs to track
     * @param options Per-frame options (memory update, end-of-sequence, etc.)
     * @return Segmentation result (CPU-resident)
     */
    types::CutieMask step(const cv::Mat& image, const cv::Mat& mask,
                          const std::vector<ObjectId>& objects, const StepOptions& options);

    // ─── GPU Inference Interface ────────────────────────────────────

    /**
     * @brief Process GPU frame (GpuMat input, default options).
     *
     * Full GPU pipeline: no CPU transfers during inference.
     * Results remain on GPU; call `download()` only when needed.
     *
     * @param gpu_image Input frame on GPU (BGR, CV_8UC3)
     * @param gpu_mask Optional: First-frame mask on GPU (CV_8UC1)
     * @param objects Optional: List of object IDs to track
     * @return Segmentation result (GPU-resident)
     *
     * @note Requires CUDA-capable GPU and ONNX Runtime CUDA EP.
     */
    types::GpuCutieMask step_gpu(const cv::cuda::GpuMat& gpu_image,
                                 const cv::cuda::GpuMat& gpu_mask = cv::cuda::GpuMat(),
                                 const std::vector<ObjectId>& objects = {});

    /**
     * @brief Process GPU frame (GpuMat input, custom options).
     *
     * @param gpu_image Input frame on GPU (BGR, CV_8UC3)
     * @param gpu_mask First-frame mask on GPU (CV_8UC1)
     * @param objects List of object IDs to track
     * @param options Per-frame options
     * @return Segmentation result (GPU-resident)
     */
    types::GpuCutieMask step_gpu(const cv::cuda::GpuMat& gpu_image,
                                 const cv::cuda::GpuMat& gpu_mask,
                                 const std::vector<ObjectId>& objects,
                                 const StepOptions& options);

    /**
     * @brief Process CPU frame with automatic GPU upload (convenience).
     *
     * Automatically uploads CPU image to GPU, runs inference, and returns
     * GPU-resident result. Useful for mixed CPU/GPU pipelines.
     *
     * @param image Input frame (CPU, BGR, CV_8UC3)
     * @param mask Optional: First-frame mask (CPU, CV_8UC1)
     * @param objects Optional: List of object IDs to track
     * @return Segmentation result (GPU-resident)
     *
     * @note Includes CPU→GPU transfer overhead. For high-throughput,
     *       prefer uploading frames to GPU beforehand.
     */
    types::GpuCutieMask step_gpu(const cv::Mat& image, const cv::Mat& mask = cv::Mat(),
                                 const std::vector<ObjectId>& objects = {});

    // ─── Object Management ──────────────────────────────────────────

    /**
     * @brief Stop tracking specified objects.
     *
     * Removes objects from active tracking and frees associated memory.
     * Subsequent frames will not include these objects in output.
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
     * Resets processor state as if starting a new video.
     * Call between video sequences.
     */
    void clear_memory();

    /**
     * @brief Clear non-permanent memory (working + long-term).
     *
     * Keeps sensory memory (per-object visual context).
     * Useful for scene changes within a video.
     */
    void clear_non_permanent_memory();

    /**
     * @brief Clear sensory memory (per-object visual context).
     *
     * Resets short-term visual context for all objects.
     * Useful when objects undergo significant appearance changes.
     */
    void clear_sensory_memory();

    // ─── Configuration ──────────────────────────────────────────────

    /**
     * @brief Update configuration (some parameters can be changed mid-sequence).
     *
     * @param config New configuration
     *
     * @note Not all parameters can be changed safely. Changing model-related
     *       parameters (variant, model_dir, model_prefix) requires reloading
     *       models and is not recommended mid-sequence.
     */
    void update_config(const CutieConfig& config);

    /**
     * @brief Get current configuration.
     * @return Reference to current CutieConfig
     */
    const CutieConfig& config() const;

private:
    /// Implementation (PIMPL pattern)
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace core
}  // namespace cutie

#endif  // CUTIE_CORE_PROCESSOR_H
