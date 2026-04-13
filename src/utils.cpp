#include <algorithm>
#include <cstring>
#include <stdexcept>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include "cutie/utils.h"

namespace cutie
{
namespace utils
{

// ImageNet normalization constants
static const float kMean[] = {0.485f, 0.456f, 0.406f};
static const float kStd[] = {0.229f, 0.224f, 0.225f};

std::pair<cv::Mat, std::array<int, 4>> preprocess_image(const cv::Mat& bgr_image,
                                                        int max_internal_size, int divisor)
{
    if (bgr_image.empty())
    {
        throw std::runtime_error("preprocess_image: input image is empty");
    }

    // BGR → RGB
    cv::Mat rgb;
    cv::cvtColor(bgr_image, rgb, cv::COLOR_BGR2RGB);

    // Resize: min side → max_internal_size (matching Python reference)
    int h = rgb.rows, w = rgb.cols;
    if (max_internal_size > 0)
    {
        int min_side = std::min(h, w);
        if (min_side > max_internal_size)
        {
            float scale = static_cast<float>(max_internal_size) / min_side;
            int new_h = static_cast<int>(h * scale);
            int new_w = static_cast<int>(w * scale);
            cv::resize(rgb, rgb, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
            h = new_h;
            w = new_w;
        }
    }

    // Convert to float32 [0, 1]
    cv::Mat float_img;
    rgb.convertTo(float_img, CV_32FC3, 1.0 / 255.0);

    // ImageNet normalize per channel
    std::vector<cv::Mat> channels(3);
    cv::split(float_img, channels);
    for (int c = 0; c < 3; ++c)
    {
        channels[c] = (channels[c] - kMean[c]) / kStd[c];
    }
    cv::merge(channels, float_img);

    // Pad to multiple of divisor
    int pad_h = (divisor - h % divisor) % divisor;
    int pad_w = (divisor - w % divisor) % divisor;
    int pad_top = pad_h / 2;
    int pad_bottom = pad_h - pad_top;
    int pad_left = pad_w / 2;
    int pad_right = pad_w - pad_left;

    if (pad_h > 0 || pad_w > 0)
    {
        cv::copyMakeBorder(float_img, float_img, pad_top, pad_bottom, pad_left, pad_right,
                           cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    }

    // HWC → NCHW blob [1, 3, H', W']
    cv::Mat blob = cv::dnn::blobFromImage(float_img);

    std::array<int, 4> pad = {pad_top, pad_bottom, pad_left, pad_right};
    return {blob, pad};
}

cv::Mat index_mask_to_one_hot(const cv::Mat& mask, const std::vector<ObjectId>& objects)
{
    if (mask.empty() || objects.empty())
    {
        return cv::Mat();
    }

    cv::Mat mask32;
    if (mask.type() == CV_8UC1)
    {
        mask.convertTo(mask32, CV_32SC1);
    }
    else if (mask.type() == CV_32SC1)
    {
        mask32 = mask;
    }
    else
    {
        throw std::runtime_error("index_mask_to_one_hot: unsupported mask type");
    }

    int h = mask32.rows, w = mask32.cols;
    int num_objects = static_cast<int>(objects.size());

    // Output: [num_objects, 1, H, W] as 4D Mat
    int sizes[] = {num_objects, 1, h, w};
    cv::Mat one_hot(4, sizes, CV_32FC1, cv::Scalar(0.0f));

    for (int oi = 0; oi < num_objects; ++oi)
    {
        ObjectId target_id = objects[oi];
        // Pointer to the slice [oi, 0, :, :]
        float* dst = one_hot.ptr<float>() + oi * h * w;
        for (int r = 0; r < h; ++r)
        {
            const int32_t* src = mask32.ptr<int32_t>(r);
            for (int c = 0; c < w; ++c)
            {
                dst[r * w + c] = (src[c] == target_id) ? 1.0f : 0.0f;
            }
        }
    }

    return one_hot;
}

cv::Mat prob_to_index_mask(const cv::Mat& prob, const std::vector<ObjectId>& objects)
{
    // prob: [num_objects+1, H, W] stored as 3D Mat or contiguous buffer
    // Channel 0 = background, channels 1..N = objects[0..N-1]
    if (prob.empty())
    {
        return cv::Mat();
    }

    int num_channels = prob.size[0];  // num_objects + 1
    int h = prob.size[1];
    int w = prob.size[2];

    cv::Mat result(h, w, CV_32SC1, cv::Scalar(0));
    const float* data = prob.ptr<float>();

    for (int r = 0; r < h; ++r)
    {
        int32_t* dst = result.ptr<int32_t>(r);
        for (int c = 0; c < w; ++c)
        {
            int pixel = r * w + c;
            float max_val = data[pixel];  // background channel
            int max_idx = 0;
            for (int ch = 1; ch < num_channels; ++ch)
            {
                float val = data[ch * h * w + pixel];
                if (val > max_val)
                {
                    max_val = val;
                    max_idx = ch;
                }
            }
            // max_idx 0 = background → ObjectId 0
            // max_idx i (i>=1) → objects[i-1]
            dst[c] = (max_idx == 0) ? 0 : objects[max_idx - 1];
        }
    }

    return result;
}

cv::Mat resize_mask(const cv::Mat& mask, int target_h, int target_w)
{
    cv::Mat resized;
    cv::resize(mask, resized, cv::Size(target_w, target_h), 0, 0, cv::INTER_NEAREST);
    return resized;
}

cv::Mat unpad(const cv::Mat& img, const std::array<int, 4>& pad)
{
    int top = pad[0], bottom = pad[1], left = pad[2], right = pad[3];

    if (top == 0 && bottom == 0 && left == 0 && right == 0)
    {
        return img.clone();
    }

    if (img.dims == 2)
    {
        // 2D: simple ROI
        int h = img.rows - top - bottom;
        int w = img.cols - left - right;
        if (h <= 0 || w <= 0)
            return img;
        return img(cv::Rect(left, top, w, h)).clone();
    }

    if (img.dims == 3)
    {
        // 3D: [C, H, W] — unpad spatial dims
        int C = img.size[0];
        int H = img.size[1];
        int W = img.size[2];
        int new_h = H - top - bottom;
        int new_w = W - left - right;
        if (new_h <= 0 || new_w <= 0)
            return img;

        int out_sizes[] = {C, new_h, new_w};
        cv::Mat result(3, out_sizes, CV_32FC1);
        for (int c = 0; c < C; ++c)
        {
            for (int r = 0; r < new_h; ++r)
            {
                std::memcpy(result.ptr<float>() + c * new_h * new_w + r * new_w,
                            img.ptr<float>() + c * H * W + (r + top) * W + left,
                            new_w * sizeof(float));
            }
        }
        return result;
    }

    // Unsupported dims — return as-is
    return img.clone();
}

}  // namespace utils
}  // namespace cutie
