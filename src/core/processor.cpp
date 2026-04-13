#include <cstring>
#include <stdexcept>

#include "cutie/core/inference_core.h"
#include "cutie/core/processor.h"
#include "cutie/utils.h"

namespace cutie
{
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
