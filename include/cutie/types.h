/**
 * @file types.h
 * @brief Core data types for Cutie video object segmentation.
 *
 * Defines fundamental types used throughout the Cutie inference pipeline,
 * including device selection, model variants, and segmentation output formats.
 */

#ifndef CUTIE_TYPES_H
#define CUTIE_TYPES_H

#include <cstdint>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

#include "cutie/ort/core/ort_config.h"  // Ort::Value

namespace cutie
{

/// Unique identifier for tracked objects (user-assigned, immutable)
using ObjectId = int32_t;

/**
 * @enum Device
 * @brief Compute device selection.
 */
enum class Device
{
    kCPU,   ///< CPU inference
    kCUDA   ///< GPU inference (CUDA)
};

/**
 * @enum ModelVariant
 * @brief Model architecture variant.
 */
enum class ModelVariant
{
    kBase,   ///< Base variant (higher accuracy)
    kSmall   ///< Small variant (faster inference)
};

namespace types
{

/**
 * @struct CutieMask
 * @brief CPU-resident segmentation output.
 *
 * Contains per-pixel object assignments and optional probability maps.
 * All data is stored on CPU (cv::Mat).
 */
struct CutieMask
{
    cv::Mat index_mask;                ///< H×W, CV_32SC1, pixel value = ObjectId (0=background)
    std::vector<ObjectId> object_ids;  ///< List of active object IDs
    cv::Mat prob;                      ///< [num_objects+1, H, W], CV_32FC1, optional probability map
    bool flag = false;                 ///< Validity flag

    CutieMask() = default;

    /**
     * @brief Construct with index mask and object list.
     * @param idx_mask Index mask (H×W, CV_32SC1)
     * @param ids Active object IDs
     */
    CutieMask(cv::Mat idx_mask, std::vector<ObjectId> ids)
        : index_mask(std::move(idx_mask)), object_ids(std::move(ids)), flag(true)
    {
    }
};

/**
 * @struct GpuCutieMask
 * @brief GPU-resident segmentation output (zero-copy).
 *
 * All data remains on GPU (cv::cuda::GpuMat, Ort::Value).
 * Call `download()` to transfer to CPU when needed.
 * 所有数据保持在 GPU 上，避免不必要的数据传输。
 */
struct GpuCutieMask
{
    cv::cuda::GpuMat index_mask;       ///< GPU: H×W, CV_32SC1, pixel value = ObjectId (0=background)
    std::vector<ObjectId> object_ids;  ///< CPU: List of active object IDs
    Ort::Value gpu_prob{nullptr};      ///< GPU: [num_obj+1, H, W] float32 probability map
    bool flag = false;                 ///< Validity flag

    GpuCutieMask() = default;

    /**
     * @brief Download GPU data to CPU CutieMask.
     *
     * Transfers index_mask and gpu_prob from GPU to CPU.
     * 将 GPU 上的数据下载到 CPU，用于后续处理或保存。
     *
     * @return CPU-resident CutieMask
     */
    CutieMask download() const;
};

}  // namespace types
}  // namespace cutie

#endif  // CUTIE_TYPES_H
