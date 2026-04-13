#ifndef CUTIE_UTILS_H
#define CUTIE_UTILS_H

#include <array>
#include <vector>

#include <opencv2/core.hpp>

#include "cutie/types.h"

namespace cutie
{
namespace utils
{

/// BGR cv::Mat (CV_8UC3) → RGB float32 [1, 3, H', W'] with ImageNet normalization.
/// Resizes long edge to max_internal_size, pads to multiple of `divisor`.
/// Returns the processed float blob and the [top, bottom, left, right] padding.
std::pair<cv::Mat, std::array<int, 4>> preprocess_image(const cv::Mat& bgr_image,
                                                        int max_internal_size, int divisor = 16);

/// index mask (CV_8UC1 or CV_32SC1) → one-hot float32 [num_objects, 1, H, W]
/// objects: list of ObjectIds present in mask. 0 = background (excluded).
cv::Mat index_mask_to_one_hot(const cv::Mat& mask, const std::vector<ObjectId>& objects);

/// prob [num_objects+1, H, W] float32 → index mask [H, W] CV_32SC1 (argmax per pixel)
/// Channel 0 = background. Channels 1..N correspond to objects[0..N-1].
cv::Mat prob_to_index_mask(const cv::Mat& prob, const std::vector<ObjectId>& objects);

/// Resize mask to target size with INTER_NEAREST.
cv::Mat resize_mask(const cv::Mat& mask, int target_h, int target_w);

/// Remove padding from an image.
cv::Mat unpad(const cv::Mat& img, const std::array<int, 4>& pad);

}  // namespace utils
}  // namespace cutie

#endif  // CUTIE_UTILS_H
