/**
 * @file processor.cpp
 * @brief CutieProcessor implementation (public API entry point).
 *
 * Implements the main processor class that manages video object segmentation
 * inference. Provides both CPU and GPU inference paths with memory management.
 */

#include <cstring>
#include <stdexcept>

#include <cuda_runtime.h>

#include "cutie/core/inference_core.h"
#include "cutie/core/processor.h"
#include "cutie/ort/core/gpu_memory.h"
#include "cutie/utils.h"

namespace cutie
{

// ── GpuCutieMask::download ──────────────────────────────────────────
// GPU 结果下载到 CPU，用于后续处理或保存。

types::CutieMask types::GpuCutieMask::download() const
{
    using GA = ortcore::GpuMemoryAllocator;

    types::CutieMask result;
    result.object_ids = object_ids;
    result.flag = flag;

    if (!index_mask.empty())
    {
        index_mask.download(result.index_mask);
    }

    if (gpu_prob.IsTensor())
    {
        auto s = GA::shape(gpu_prob);
        int64_t total = GA::numel(s);
        std::vector<int> cv_sizes(s.begin(), s.end());
        result.prob = cv::Mat(static_cast<int>(cv_sizes.size()), cv_sizes.data(), CV_32FC1);
        cudaMemcpy(result.prob.ptr<float>(), GA::data_ptr(gpu_prob), total * sizeof(float),
                   cudaMemcpyDeviceToHost);
    }

    return result;
}

namespace core
{

// ── CutieConfig factory methods ─────────────────────────────────────

CutieConfig CutieConfig::base_default(const std::string& model_dir)
{
    CutieConfig c;
    c.variant = ModelVariant::kBase;
    c.model_dir = model_dir;

    c.model.key_dim = 64;
    c.model.value_dim = 256;
    c.model.sensory_dim = 256;
    c.model.pixel_dim = 256;
    c.model.f16_dim = 1024;
    c.model.f8_dim = 512;
    c.model.f4_dim = 256;
    c.model.num_queries = 16;

    return c;
}

CutieConfig CutieConfig::small_default(const std::string& model_dir)
{
    CutieConfig c;
    c.variant = ModelVariant::kSmall;
    c.model_dir = model_dir;

    c.model.key_dim = 64;
    c.model.value_dim = 256;
    c.model.sensory_dim = 256;
    c.model.pixel_dim = 256;
    c.model.f16_dim = 256;
    c.model.f8_dim = 128;
    c.model.f4_dim = 64;
    c.model.num_queries = 16;

    return c;
}

// ── CutieProcessor pimpl ────────────────────────────────────────────

struct CutieProcessor::Impl
{
    CutieConfig config;
    InferenceCore core;

    explicit Impl(const CutieConfig& cfg, std::shared_ptr<linden::log::ILogger> logger)
        : config(cfg), core(cfg, std::move(logger))
    {
    }
};

CutieProcessor::CutieProcessor(const CutieConfig& config,
                               std::shared_ptr<linden::log::ILogger> logger)
    : impl_(std::make_unique<Impl>(config, std::move(logger)))
{
}

CutieProcessor::~CutieProcessor() = default;

CutieProcessor::CutieProcessor(CutieProcessor&&) noexcept = default;
CutieProcessor& CutieProcessor::operator=(CutieProcessor&&) noexcept = default;

types::CutieMask CutieProcessor::step(const cv::Mat& image, const cv::Mat& mask,
                                      const std::vector<ObjectId>& objects)
{
    StepOptions opts;
    return step(image, mask, objects, opts);
}

types::CutieMask CutieProcessor::step(const cv::Mat& image, const cv::Mat& mask,
                                      const std::vector<ObjectId>& objects,
                                      const StepOptions& options)
{
    // Convert index mask to the format expected by InferenceCore
    cv::Mat input_mask;
    std::vector<ObjectId> obj_ids = objects;

    if (!mask.empty() && options.idx_mask)
    {
        input_mask = mask;
        if (input_mask.type() != CV_32SC1)
        {
            mask.convertTo(input_mask, CV_32SC1);
        }
    }
    else if (!mask.empty())
    {
        input_mask = mask;
    }

    cv::Mat prob =
        impl_->core.step(image, input_mask, obj_ids, options.end, options.force_permanent);

    // Convert prob → CutieMask
    types::CutieMask result;
    result.object_ids = impl_->core.active_objects();

    if (!prob.empty())
    {
        result.prob = prob.clone();
        result.index_mask = utils::prob_to_index_mask(prob, result.object_ids);
    }

    return result;
}

// ── GPU step ────────────────────────────────────────────────────────

types::GpuCutieMask CutieProcessor::step_gpu(const cv::cuda::GpuMat& gpu_image,
                                             const cv::cuda::GpuMat& gpu_mask,
                                             const std::vector<ObjectId>& objects)
{
    StepOptions opts;
    return step_gpu(gpu_image, gpu_mask, objects, opts);
}

types::GpuCutieMask CutieProcessor::step_gpu(const cv::cuda::GpuMat& gpu_image,
                                             const cv::cuda::GpuMat& gpu_mask,
                                             const std::vector<ObjectId>& objects,
                                             const StepOptions& options)
{
    cv::cuda::GpuMat input_mask = gpu_mask;
    std::vector<ObjectId> obj_ids = objects;

    // GPU mask type 转换（如需要）
    if (!gpu_mask.empty() && options.idx_mask && gpu_mask.type() != CV_32SC1)
    {
        gpu_mask.convertTo(input_mask, CV_32SC1);
    }

    return impl_->core.step_gpu(gpu_image, input_mask, obj_ids, options.end,
                                options.force_permanent);
}

types::GpuCutieMask CutieProcessor::step_gpu(const cv::Mat& image, const cv::Mat& mask,
                                             const std::vector<ObjectId>& objects)
{
    // 自动上传到 GPU
    cv::cuda::GpuMat gpu_image;
    gpu_image.upload(image);

    cv::cuda::GpuMat gpu_mask;
    if (!mask.empty())
    {
        cv::Mat mask_i32 = mask;
        if (mask_i32.type() != CV_32SC1) mask.convertTo(mask_i32, CV_32SC1);
        gpu_mask.upload(mask_i32);
    }

    StepOptions opts;
    return step_gpu(gpu_image, gpu_mask, objects, opts);
}

void CutieProcessor::delete_objects(const std::vector<ObjectId>& objects)
{
    impl_->core.delete_objects(objects);
}

std::vector<ObjectId> CutieProcessor::active_objects() const
{
    return impl_->core.active_objects();
}

int CutieProcessor::num_objects() const
{
    return impl_->core.num_objects();
}

void CutieProcessor::clear_memory()
{
    impl_->core.clear_memory();
}

void CutieProcessor::clear_non_permanent_memory()
{
    impl_->core.clear_non_permanent_memory();
}

void CutieProcessor::clear_sensory_memory()
{
    impl_->core.clear_sensory_memory();
}

void CutieProcessor::update_config(const CutieConfig& config)
{
    impl_->config = config;
    impl_->core.update_config(config);
}

const CutieConfig& CutieProcessor::config() const
{
    return impl_->config;
}

}  // namespace core
}  // namespace cutie
