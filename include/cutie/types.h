#ifndef CUTIE_TYPES_H
#define CUTIE_TYPES_H

#include <cstdint>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

#include "cutie/ort/core/ort_config.h"  // Ort::Value

namespace cutie
{

using ObjectId = int32_t;

enum class Device
{
    kCPU,
    kCUDA
};
enum class ModelVariant
{
    kBase,
    kSmall
};

namespace types
{

struct CutieMask
{
    cv::Mat index_mask;                // H×W, CV_32SC1, pixel=ObjectId (0=bg)
    std::vector<ObjectId> object_ids;  // active object list
    cv::Mat prob;                      // [num_objects+1, H, W], CV_32FC1 (optional)
    bool flag = false;

    CutieMask() = default;

    CutieMask(cv::Mat idx_mask, std::vector<ObjectId> ids)
        : index_mask(std::move(idx_mask)), object_ids(std::move(ids)), flag(true)
    {
    }
};

/// GPU 推理输出：全部数据留在 GPU 上，按需 download 到 CPU。
struct GpuCutieMask
{
    cv::cuda::GpuMat index_mask;       ///< GPU: H×W, CV_32SC1, pixel=ObjectId (0=bg)
    std::vector<ObjectId> object_ids;  ///< CPU: active object list
    Ort::Value gpu_prob{nullptr};      ///< GPU: [num_obj+1, H, W] float32 概率图
    bool flag = false;

    GpuCutieMask() = default;

    /// 按需下载到 CPU CutieMask。
    CutieMask download() const;
};

}  // namespace types
}  // namespace cutie

#endif  // CUTIE_TYPES_H
