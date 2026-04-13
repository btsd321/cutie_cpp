#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

#include "cutie/core/memory_utils.h"

namespace cutie
{
namespace core
{

// Helper: access element in a 3D Mat [d0, d1, d2] stored contiguously
static inline float& at3(cv::Mat& m, int d0, int d1, int d2)
{
    return m.ptr<float>()[d0 * m.size[1] * m.size[2] + d1 * m.size[2] + d2];
}

static inline float at3c(const cv::Mat& m, int d0, int d1, int d2)
{
    return m.ptr<float>()[d0 * m.size[1] * m.size[2] + d1 * m.size[2] + d2];
}

cv::Mat get_similarity(const cv::Mat& mk, const cv::Mat& ms, const cv::Mat& qk, const cv::Mat& qe)
{
    // mk: [B, CK, N], ms: [B, 1, N], qk: [B, CK, HW], qe: [B, CK, HW]
    // Output: [B, N, HW]
    int B = mk.size[0];
    int CK = mk.size[1];
    int N = mk.size[2];
    int HW = qk.size[2];

    bool has_qe = !qe.empty();
    bool has_ms = !ms.empty();

    int out_sizes[] = {B, N, HW};
    cv::Mat similarity(3, out_sizes, CV_32FC1, cv::Scalar(0.0f));

    float scale = 1.0f / std::sqrt(static_cast<float>(CK));

    for (int b = 0; b < B; ++b)
    {
        const float* mk_b = mk.ptr<float>() + b * CK * N;
        const float* qk_b = qk.ptr<float>() + b * CK * HW;
        const float* qe_b = has_qe ? qe.ptr<float>() + b * CK * HW : nullptr;
        const float* ms_b = has_ms ? ms.ptr<float>() + b * 1 * N : nullptr;
        float* sim_b = similarity.ptr<float>() + b * N * HW;

        if (has_qe)
        {
            // a_sq = mk^T @ qe : [N, HW]   (mk^T: [N, CK], qe: [CK, HW])
            // two_ab = 2 * mk^T @ (qk * qe) : [N, HW]
            // b_sq = sum_over_CK(qe * qk^2) : [1, HW] broadcast to [N, HW]

            // Compute b_sq: [HW]
            std::vector<float> b_sq(HW, 0.0f);
            for (int ck = 0; ck < CK; ++ck)
            {
                for (int hw = 0; hw < HW; ++hw)
                {
                    float q = qk_b[ck * HW + hw];
                    float e = qe_b[ck * HW + hw];
                    b_sq[hw] += e * q * q;
                }
            }

            for (int n = 0; n < N; ++n)
            {
                float a_sq_accum;
                for (int hw = 0; hw < HW; ++hw)
                {
                    float a_sq_val = 0.0f;
                    float two_ab_val = 0.0f;
                    for (int ck = 0; ck < CK; ++ck)
                    {
                        float m = mk_b[ck * N + n];
                        float e = qe_b[ck * HW + hw];
                        float k = qk_b[ck * HW + hw];
                        a_sq_val += m * m * e;
                        two_ab_val += m * k * e;
                    }
                    float val = -a_sq_val + 2.0f * two_ab_val - b_sq[hw];
                    // Apply shrinkage
                    if (has_ms)
                    {
                        val *= ms_b[n] * scale;
                    }
                    else
                    {
                        val *= scale;
                    }
                    sim_b[n * HW + hw] = val;
                }
            }
        }
        else
        {
            // Without selection: standard L2 similarity
            // a_sq = sum_CK(mk^2) : [N]
            // two_ab = 2 * mk^T @ qk : [N, HW]
            std::vector<float> a_sq(N, 0.0f);
            for (int ck = 0; ck < CK; ++ck)
            {
                for (int n = 0; n < N; ++n)
                {
                    float m = mk_b[ck * N + n];
                    a_sq[n] += m * m;
                }
            }

            for (int n = 0; n < N; ++n)
            {
                for (int hw = 0; hw < HW; ++hw)
                {
                    float dot = 0.0f;
                    for (int ck = 0; ck < CK; ++ck)
                    {
                        dot += mk_b[ck * N + n] * qk_b[ck * HW + hw];
                    }
                    float val = -a_sq[n] + 2.0f * dot;
                    if (has_ms)
                    {
                        val *= ms_b[n] * scale;
                    }
                    else
                    {
                        val *= scale;
                    }
                    sim_b[n * HW + hw] = val;
                }
            }
        }
    }

    return similarity;
}

std::pair<cv::Mat, cv::Mat> do_softmax(const cv::Mat& similarity, int top_k, bool return_usage)
{
    // similarity: [B, N, HW]
    // softmax along dim=1 (N dimension)
    int B = similarity.size[0];
    int N = similarity.size[1];
    int HW = similarity.size[2];

    int out_sizes[] = {B, N, HW};
    cv::Mat affinity(3, out_sizes, CV_32FC1, cv::Scalar(0.0f));

    cv::Mat usage;
    if (return_usage)
    {
        int usage_sizes[] = {B, N};
        usage = cv::Mat(2, usage_sizes, CV_32FC1, cv::Scalar(0.0f));
    }

    for (int b = 0; b < B; ++b)
    {
        const float* sim_b = similarity.ptr<float>() + b * N * HW;
        float* aff_b = affinity.ptr<float>() + b * N * HW;

        if (top_k > 0 && top_k < N)
        {
            // For each HW position, find top_k along N, do softmax over those
            std::vector<std::pair<float, int>> vals(N);

            for (int hw = 0; hw < HW; ++hw)
            {
                // Gather values for this HW position across N
                for (int n = 0; n < N; ++n)
                {
                    vals[n] = {sim_b[n * HW + hw], n};
                }
                // Partial sort to get top_k
                std::partial_sort(vals.begin(), vals.begin() + top_k, vals.end(),
                                  [](const auto& a, const auto& b)
                                  {
                                      return a.first > b.first;
                                  });

                // Softmax over top_k
                float max_val = vals[0].first;
                float sum_exp = 0.0f;
                for (int i = 0; i < top_k; ++i)
                {
                    float e = std::exp(vals[i].first - max_val);
                    vals[i].first = e;
                    sum_exp += e;
                }
                for (int i = 0; i < top_k; ++i)
                {
                    int n = vals[i].second;
                    aff_b[n * HW + hw] = vals[i].first / sum_exp;
                }
            }
        }
        else
        {
            // Full softmax along N for each HW
            for (int hw = 0; hw < HW; ++hw)
            {
                float max_val = -1e30f;
                for (int n = 0; n < N; ++n)
                {
                    max_val = std::max(max_val, sim_b[n * HW + hw]);
                }
                float sum_exp = 0.0f;
                for (int n = 0; n < N; ++n)
                {
                    float e = std::exp(sim_b[n * HW + hw] - max_val);
                    aff_b[n * HW + hw] = e;
                    sum_exp += e;
                }
                for (int n = 0; n < N; ++n)
                {
                    aff_b[n * HW + hw] /= sum_exp;
                }
            }
        }

        // Compute usage: sum of affinity along HW → [B, N]
        if (return_usage)
        {
            float* usage_b = usage.ptr<float>() + b * N;
            for (int n = 0; n < N; ++n)
            {
                float sum = 0.0f;
                for (int hw = 0; hw < HW; ++hw)
                {
                    sum += aff_b[n * HW + hw];
                }
                usage_b[n] = sum;
            }
        }
    }

    return {affinity, usage};
}

cv::Mat readout(const cv::Mat& affinity, const cv::Mat& mv)
{
    // affinity: [B, N, HW]
    // mv: [B, CV, N]
    // output: [B, CV, HW] = mv @ affinity (batched matmul)
    int B = mv.size[0];
    int CV = mv.size[1];
    int N = mv.size[2];
    int HW = affinity.size[2];

    int out_sizes[] = {B, CV, HW};
    cv::Mat result(3, out_sizes, CV_32FC1, cv::Scalar(0.0f));

    for (int b = 0; b < B; ++b)
    {
        const float* mv_b = mv.ptr<float>() + b * CV * N;
        const float* aff_b = affinity.ptr<float>() + b * N * HW;
        float* out_b = result.ptr<float>() + b * CV * HW;

        // out[cv, hw] = sum_n( mv[cv, n] * aff[n, hw] )
        for (int cv_i = 0; cv_i < CV; ++cv_i)
        {
            for (int hw = 0; hw < HW; ++hw)
            {
                float sum = 0.0f;
                for (int n = 0; n < N; ++n)
                {
                    sum += mv_b[cv_i * N + n] * aff_b[n * HW + hw];
                }
                out_b[cv_i * HW + hw] = sum;
            }
        }
    }

    return result;
}

cv::Mat aggregate(const cv::Mat& prob)
{
    // prob: [num_objects, H, W] — logits (no background)
    // output: [num_objects+1, H, W] — softmax with bg channel at index 0
    int num_obj = prob.size[0];
    int H = prob.size[1];
    int W = prob.size[2];

    int out_sizes[] = {num_obj + 1, H, W};
    cv::Mat result(3, out_sizes, CV_32FC1);

    const float* prob_data = prob.ptr<float>();
    float* out_data = result.ptr<float>();

    for (int r = 0; r < H; ++r)
    {
        for (int c = 0; c < W; ++c)
        {
            int pixel = r * W + c;
            // Background logit = 0 (aggregate adds a zero-logit bg channel)
            float max_val = 0.0f;
            for (int o = 0; o < num_obj; ++o)
            {
                float v = prob_data[o * H * W + pixel];
                max_val = std::max(max_val, v);
            }

            float sum_exp = std::exp(0.0f - max_val);  // background
            for (int o = 0; o < num_obj; ++o)
            {
                sum_exp += std::exp(prob_data[o * H * W + pixel] - max_val);
            }

            // Channel 0 = background
            out_data[pixel] = std::exp(0.0f - max_val) / sum_exp;
            // Channels 1..num_obj = objects
            for (int o = 0; o < num_obj; ++o)
            {
                out_data[(o + 1) * H * W + pixel] =
                    std::exp(prob_data[o * H * W + pixel] - max_val) / sum_exp;
            }
        }
    }

    return result;
}

std::pair<cv::Mat, std::array<int, 4>> pad_divide_by(const cv::Mat& img, int d)
{
    int h, w;
    if (img.dims <= 2)
    {
        h = img.rows;
        w = img.cols;
    }
    else
    {
        // For multi-dim: last two dims are H, W
        h = img.size[img.dims - 2];
        w = img.size[img.dims - 1];
    }

    int pad_h = (d - h % d) % d;
    int pad_w = (d - w % d) % d;
    int pad_top = pad_h / 2;
    int pad_bottom = pad_h - pad_top;
    int pad_left = pad_w / 2;
    int pad_right = pad_w - pad_left;

    std::array<int, 4> pad = {pad_top, pad_bottom, pad_left, pad_right};

    if (pad_h == 0 && pad_w == 0)
    {
        return {img, pad};
    }

    if (img.dims <= 2)
    {
        cv::Mat padded;
        cv::copyMakeBorder(img, padded, pad_top, pad_bottom, pad_left, pad_right,
                           cv::BORDER_CONSTANT, cv::Scalar(0));
        return {padded, pad};
    }

    // For higher-dim tensors (e.g., [C, H, W] or [N, C, H, W]):
    // Pad the last two dimensions manually
    int ndim = img.dims;
    std::vector<int> new_sizes;
    for (int i = 0; i < ndim - 2; ++i) new_sizes.push_back(img.size[i]);
    new_sizes.push_back(h + pad_h);
    new_sizes.push_back(w + pad_w);

    cv::Mat padded(ndim, new_sizes.data(), img.type(), cv::Scalar(0));

    // Number of slices (product of all dims except last 2)
    int num_slices = 1;
    for (int i = 0; i < ndim - 2; ++i) num_slices *= img.size[i];

    int new_h = h + pad_h;
    int new_w = w + pad_w;
    const float* src = img.ptr<float>();
    float* dst = padded.ptr<float>();

    for (int s = 0; s < num_slices; ++s)
    {
        for (int r = 0; r < h; ++r)
        {
            std::memcpy(dst + s * new_h * new_w + (r + pad_top) * new_w + pad_left,
                        src + s * h * w + r * w, w * sizeof(float));
        }
    }

    return {padded, pad};
}

cv::Mat unpad(const cv::Mat& img, const std::array<int, 4>& pad)
{
    int top = pad[0], bottom = pad[1], left = pad[2], right = pad[3];
    if (top == 0 && bottom == 0 && left == 0 && right == 0)
    {
        return img;
    }

    if (img.dims <= 2)
    {
        int h = img.rows - top - bottom;
        int w = img.cols - left - right;
        if (h <= 0 || w <= 0)
            return img;
        return img(cv::Rect(left, top, w, h)).clone();
    }

    // Higher-dim: unpad last two dimensions
    int ndim = img.dims;
    int H = img.size[ndim - 2];
    int W = img.size[ndim - 1];
    int new_h = H - top - bottom;
    int new_w = W - left - right;
    if (new_h <= 0 || new_w <= 0)
        return img;

    std::vector<int> new_sizes;
    for (int i = 0; i < ndim - 2; ++i) new_sizes.push_back(img.size[i]);
    new_sizes.push_back(new_h);
    new_sizes.push_back(new_w);

    cv::Mat result(ndim, new_sizes.data(), img.type());

    int num_slices = 1;
    for (int i = 0; i < ndim - 2; ++i) num_slices *= img.size[i];

    const float* src = img.ptr<float>();
    float* dst = result.ptr<float>();

    for (int s = 0; s < num_slices; ++s)
    {
        for (int r = 0; r < new_h; ++r)
        {
            std::memcpy(dst + s * new_h * new_w + r * new_w, src + s * H * W + (r + top) * W + left,
                        new_w * sizeof(float));
        }
    }

    return result;
}

}  // namespace core
}  // namespace cutie
