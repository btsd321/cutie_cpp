#ifndef CUTIE_TYPES_H
#define CUTIE_TYPES_H

#include <cstdint>
#include <vector>

#include <opencv2/core.hpp>

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

}  // namespace types
}  // namespace cutie

#endif  // CUTIE_TYPES_H
