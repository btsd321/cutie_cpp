#pragma once

#include <utility>
#include <vector>

#include <opencv2/core.hpp>

namespace cutie
{
namespace core
{

/// Anisotropic L2 similarity (reference: cutie/model/utils/memory_utils.py)
///
/// mk: [B, CK, N]     — memory keys (flattened)
/// ms: [B, 1, N]       — memory shrinkage (nullable → pass empty)
/// qk: [B, CK, HW]    — query keys (flattened)
/// qe: [B, CK, HW]    — query selection (nullable → pass empty)
///
/// Returns: [B, N, HW] — similarity scores
///
/// All inputs/outputs are cv::Mat with 3 dimensions, CV_32FC1, row-major.
cv::Mat get_similarity(const cv::Mat& mk, const cv::Mat& ms, const cv::Mat& qk, const cv::Mat& qe);

/// Top-k softmax normalization along dim=1 (the N dimension).
///
/// similarity: [B, N, HW]
/// top_k: if > 0, only keep top_k values per column
/// return_usage: if true, also return usage [B, N] (sum of affinity along HW)
///
/// Returns: (affinity [B, N, HW], usage [B, N] or empty)
std::pair<cv::Mat, cv::Mat> do_softmax(const cv::Mat& similarity, int top_k = -1,
                                       bool return_usage = false);

/// Attention-weighted readout.
///
/// affinity: [B, N, HW]
/// mv: [B, CV, N] — memory values
///
/// Returns: [B, CV, HW]
cv::Mat readout(const cv::Mat& affinity, const cv::Mat& mv);

/// Aggregate object probabilities with a background channel.
/// Input:  prob [num_objects, H, W] — per-object logits (no background)
/// Output: [num_objects+1, H, W] — softmax probabilities (channel 0 = background)
cv::Mat aggregate(const cv::Mat& prob);

/// Pad image/tensor so that H and W are multiples of `d`.
/// Returns: (padded_mat, {pad_top, pad_bottom, pad_left, pad_right})
std::pair<cv::Mat, std::array<int, 4>> pad_divide_by(const cv::Mat& img, int d);

/// Remove padding.
cv::Mat unpad(const cv::Mat& img, const std::array<int, 4>& pad);

}  // namespace core
}  // namespace cutie
