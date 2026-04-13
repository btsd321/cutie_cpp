#include <linden_logger/logger_interface.hpp>

#include "cutie/core/image_feature_store.h"
#include "cutie/core/inference_core.h"
#include "cutie/core/memory_manager.h"
#include "cutie/core/object_manager.h"
#include "cutie/core/processor.h"
#include "cutie/ort/core/cuda_kernels.h"
#include "cutie/ort/core/gpu_tensor_ops.h"
#include "cutie/utils.h"

#ifdef ENABLE_ONNXRUNTIME
#include "cutie/ort/cv/ort_cutie.h"
#endif

#include <cuda_runtime.h>

#include <cmath>
#include <cstring>
#include <numeric>
#include <set>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

namespace cutie
{
namespace core
{

using GA = ortcore::GpuMemoryAllocator;

// ── Impl ────────────────────────────────────────────────────────────

struct InferenceCore::Impl
{
    CutieConfig cfg;
    std::shared_ptr<linden::log::ILogger> logger;
    int mem_every;
    int chunk_size;
    int model_h_ = 0;
    int model_w_ = 0;

    int curr_ti = -1;
    int last_mem_ti = 0;
    std::set<int> stagger_ti;
    Ort::Value last_mask{nullptr};  // GPU: [1, num_objects, H, W]
    std::array<int, 4> pad{};
    Ort::Value cached_image_gpu{nullptr};  // GPU: 当前帧的预处理图像 [1,3,H,W]

    ObjectManager object_manager;
    std::unique_ptr<MemoryManager> memory;
    ImageFeatureStore feature_store;

#ifdef ENABLE_ONNXRUNTIME
    std::unique_ptr<ortcv::OrtCutie> network;
#endif

    explicit Impl(const CutieConfig& config, std::shared_ptr<linden::log::ILogger> logger_ptr);

    cv::Mat step(const cv::Mat& image, const cv::Mat& mask, const std::vector<ObjectId>& objects,
                 bool end, bool force_permanent);

    void ensure_features(int ti);
    Ort::Value segment(ImageFeatureStore::CachedFeatures& feat, bool update_sensory);
    void add_memory(ImageFeatureStore::CachedFeatures& feat, Ort::Value& prob,
                    bool force_permanent);
    NetworkCallbacks make_callbacks();
};

// ── Impl construction ───────────────────────────────────────────────

InferenceCore::Impl::Impl(const CutieConfig& config,
                          std::shared_ptr<linden::log::ILogger> logger_ptr)
    : cfg(config), logger(logger_ptr ? std::move(logger_ptr) : linden::log::StdLogger::instance())
{
    mem_every = cfg.mem_every;
    chunk_size = cfg.chunk_size;

    int stagger_updates = cfg.stagger_updates;
    if (stagger_updates >= mem_every)
    {
        for (int i = 1; i <= mem_every; ++i) stagger_ti.insert(i);
    }
    else
    {
        for (int i = 0; i < stagger_updates; ++i)
        {
            int val =
                static_cast<int>(std::round(1.0 + i * (mem_every - 1.0) / (stagger_updates - 1.0)));
            stagger_ti.insert(val);
        }
    }

#ifdef ENABLE_ONNXRUNTIME
    network = std::make_unique<ortcv::OrtCutie>(cfg, logger);
    model_h_ = network->model_input_h();
    model_w_ = network->model_input_w();

    memory = std::make_unique<MemoryManager>(cfg, &object_manager, &network->gpu_alloc());

    if (model_h_ > 0 && model_w_ > 0)
        logger->info("InferenceCore: model input size {}x{} (auto-resize enabled)", model_h_,
                     model_w_);
    logger->info("InferenceCore: GPU inference initialized (model_dir={})", cfg.model_dir);
#else
    logger->error("InferenceCore: no backend enabled. Build with -DENABLE_ONNXRUNTIME=ON");
    throw std::runtime_error("InferenceCore: no backend enabled.");
#endif
}

// ── ensure_features ─────────────────────────────────────────────────

void InferenceCore::Impl::ensure_features(int ti)
{
    if (feature_store.has(ti))
        return;

#ifdef ENABLE_ONNXRUNTIME
    // cached_image_gpu 已在 step() 中通过 GPU 预处理生成
    auto img_feat = network->encode_image(cached_image_gpu);
    auto key_feat = network->transform_key(img_feat.f16);

    ImageFeatureStore::CachedFeatures cached;
    cached.f16 = std::move(img_feat.f16);
    cached.f8 = std::move(img_feat.f8);
    cached.f4 = std::move(img_feat.f4);
    cached.pix_feat = std::move(img_feat.pix_feat);
    cached.key = std::move(key_feat.key);
    cached.shrinkage = std::move(key_feat.shrinkage);
    cached.selection = std::move(key_feat.selection);

    feature_store.put(ti, std::move(cached));
#endif
}

// ── make_callbacks ──────────────────────────────────────────────────

NetworkCallbacks InferenceCore::Impl::make_callbacks()
{
    NetworkCallbacks cb;

#ifdef ENABLE_ONNXRUNTIME
    cb.pixel_fusion = [this](Ort::Value& pix_feat, Ort::Value& visual_readout,
                             Ort::Value& sensory, Ort::Value& last_mask_arg) -> Ort::Value
    {
        return network->pixel_fusion(pix_feat, visual_readout, sensory, last_mask_arg);
    };

    cb.readout_query = [this](Ort::Value& pixel_readout, Ort::Value& obj_memory) -> Ort::Value
    {
        return network->readout_query(pixel_readout, obj_memory);
    };
#endif

    return cb;
}

// ── segment ─────────────────────────────────────────────────────────

Ort::Value InferenceCore::Impl::segment(ImageFeatureStore::CachedFeatures& feat,
                                         bool update_sensory)
{
    if (!memory->engaged())
    {
        logger->warn("InferenceCore: trying to segment without any memory!");
        auto key_shape = GA::shape(feat.key);
        int H = static_cast<int>(key_shape[2]) * 16;
        int W = static_cast<int>(key_shape[3]) * 16;
#ifdef ENABLE_ONNXRUNTIME
        return network->gpu_alloc().zeros({1, H, W});
#else
        return Ort::Value{nullptr};
#endif
    }

#ifdef ENABLE_ONNXRUNTIME
    auto& alloc = network->gpu_alloc();

    // Flatten key/selection: [1, C, H, W] → [1, C, HW]
    auto flat_key = ortcore::gpu_flatten_spatial(alloc, feat.key);
    auto flat_sel = ortcore::gpu_flatten_spatial(alloc, feat.selection);

    auto callbacks = make_callbacks();
    auto memory_readout_map =
        memory->read(feat.pix_feat, flat_key, flat_sel, last_mask, callbacks);

    // realize dict → stacked tensor [B, num_objects, C, H, W]
    auto memory_readout = object_manager.realize_dict_gpu(memory_readout_map, alloc);

    // Segment using mask decoder
    auto sensory_in = memory->get_sensory(object_manager.all_obj_ids());

    auto seg_result = network->segment(feat.f8, feat.f4, memory_readout, sensory_in);

    if (update_sensory)
    {
        memory->update_sensory(seg_result.new_sensory, object_manager.all_obj_ids());
    }

    // logits: [num_objects, 1, H/4, W/4] → sigmoid → aggregate
    auto logits_shape = GA::shape(seg_result.logits);
    int num_obj = static_cast<int>(logits_shape[0]);
    int lh = static_cast<int>(logits_shape[2]);
    int lw = static_cast<int>(logits_shape[3]);

    // Remove singleton dim: [num_obj, 1, lh, lw] → [num_obj, lh, lw]
    auto logits_3d = Ort::Value::CreateTensor<float>(
        alloc.memory_info(), GA::data_ptr(seg_result.logits), num_obj * lh * lw,
        std::vector<int64_t>{num_obj, lh, lw}.data(), 3);
    auto logits_clone = alloc.clone(logits_3d);

    // Sigmoid → prob
    auto prob_no_bg = ortcore::gpu_sigmoid(alloc, logits_clone);

    // Aggregate: add background, softmax
    auto prob_with_bg = ortcore::gpu_aggregate(alloc, prob_no_bg);

    // Upsample 4× to full resolution（GPU）
    int full_h = lh * 4;
    int full_w = lw * 4;

    return alloc.resize_channels(prob_with_bg, full_h, full_w);
#else
    return Ort::Value{nullptr};
#endif
}

// ── add_memory ──────────────────────────────────────────────────────

void InferenceCore::Impl::add_memory(ImageFeatureStore::CachedFeatures& feat, Ort::Value& prob,
                                     bool force_permanent)
{
    if (!prob.IsTensor() || GA::shape(prob)[1] == 0)
    {
        logger->warn("InferenceCore: trying to add empty object mask to memory, skipping.");
        return;
    }

#ifdef ENABLE_ONNXRUNTIME
    auto& alloc = network->gpu_alloc();
    auto all_ids = object_manager.all_obj_ids();
    memory->initialize_sensory_if_needed(feat.key, all_ids);

    auto sensory_in = memory->get_sensory(all_ids);

    // 使用缓存的 GPU 图像（无需再次上传）
    auto enc = network->encode_mask(cached_image_gpu, feat.pix_feat, sensory_in, prob);

    // Flatten key/shrinkage: [1, C, H, W] → [1, C, HW]
    auto flat_key = ortcore::gpu_flatten_spatial(alloc, feat.key);
    auto flat_shrinkage = ortcore::gpu_flatten_spatial(alloc, feat.shrinkage);
    auto flat_selection = ortcore::gpu_flatten_spatial(alloc, feat.selection);

    // mask_value: [B, N, C, H, W] → flatten spatial → [B, N, C, HW]
    auto mv_shape = GA::shape(enc.mask_value);
    int B = static_cast<int>(mv_shape[0]);
    int N = static_cast<int>(mv_shape[1]);
    int C = static_cast<int>(mv_shape[2]);
    int mh = static_cast<int>(mv_shape[3]);
    int mw = static_cast<int>(mv_shape[4]);
    int mhw = mh * mw;
    auto flat_mask_value = Ort::Value::CreateTensor<float>(
        alloc.memory_info(), GA::data_ptr(enc.mask_value), B * N * C * mhw,
        std::vector<int64_t>{B, N, C, (int64_t)mhw}.data(), 4);
    auto flat_mv = alloc.clone(flat_mask_value);

    memory->add_memory(flat_key, flat_shrinkage, flat_mv, enc.obj_summaries, all_ids,
                       flat_selection, force_permanent);

    last_mem_ti = curr_ti;
    memory->update_sensory(enc.new_sensory, all_ids);
#endif
}

// ── step ────────────────────────────────────────────────────────────

cv::Mat InferenceCore::Impl::step(const cv::Mat& image, const cv::Mat& mask,
                                  const std::vector<ObjectId>& objects, bool end,
                                  bool force_permanent)
{
    curr_ti++;

    int orig_h = image.rows;
    int orig_w = image.cols;

    // 当 ONNX 为动态分辨率时，model_h_/model_w_ 为 0（dim 值 -1）。
    // 此时使用 max_internal_size 计算目标尺寸（与 Python 版行为一致：短边超出则等比缩放）。
    // 当 ONNX 为静态分辨率时，model_h_/model_w_ > 0，直接使用固定尺寸。
    int target_h = model_h_;
    int target_w = model_w_;
    if (target_h <= 0 || target_w <= 0)
    {
        // 动态分辨率 ONNX：按 max_internal_size 等比缩放
        target_h = orig_h;
        target_w = orig_w;
        if (cfg.max_internal_size > 0)
        {
            int min_side = std::min(orig_h, orig_w);
            if (min_side > cfg.max_internal_size)
            {
                float scale = static_cast<float>(cfg.max_internal_size) / min_side;
                target_h = static_cast<int>(orig_h * scale);
                target_w = static_cast<int>(orig_w * scale);
            }
        }
    }

    bool resize_needed = (orig_h != target_h || orig_w != target_w);

#ifdef ENABLE_ONNXRUNTIME
    auto& alloc = network->gpu_alloc();

    // GPU 预处理：BGR uint8 → RGB float32 [1,3,padH,padW] + pad
    auto [image_gpu, pad_out] = alloc.preprocess_image_gpu(image, target_h, target_w);
    pad = pad_out;
    cached_image_gpu = std::move(image_gpu);
#endif

    // Mask 预处理（CPU，因为 mask 是稀疏整数标签）
    cv::Mat proc_mask = mask;
    if (!mask.empty() && resize_needed)
    {
        cv::resize(mask, proc_mask, cv::Size(target_w, target_h), 0, 0, cv::INTER_NEAREST);
    }

    bool is_mem_frame = ((curr_ti - last_mem_ti >= mem_every) || !mask.empty()) && !end;
    bool need_segment =
        mask.empty() || (object_manager.num_obj() > 0 && !object_manager.has_all(objects));
    int stagger_delta = curr_ti - last_mem_ti;
    bool update_sensory = stagger_ti.count(stagger_delta) > 0 && !end;

    ensure_features(curr_ti);
    auto& feat = feature_store.get(curr_ti);

    Ort::Value pred_prob_with_bg{nullptr};

    if (need_segment)
    {
        pred_prob_with_bg = segment(feat, update_sensory);
    }

    // Handle input mask
    if (!proc_mask.empty())
    {
        auto [tmp_ids, new_ids] = object_manager.add_new_objects(objects);

        // Pad mask
        cv::Mat padded_mask = proc_mask;
        if (padded_mask.type() != CV_32SC1)
            proc_mask.convertTo(padded_mask, CV_32SC1);
        if (pad[0] > 0 || pad[1] > 0 || pad[2] > 0 || pad[3] > 0)
        {
            cv::copyMakeBorder(padded_mask, padded_mask, pad[0], pad[1], pad[2], pad[3],
                               cv::BORDER_CONSTANT, cv::Scalar(0));
        }

        int H = padded_mask.rows;
        int W = padded_mask.cols;
        int HW = H * W;

#ifdef ENABLE_ONNXRUNTIME
        // 上传 padded_mask 和 objects 到 GPU
        int32_t* d_mask = nullptr;
        cudaMalloc(&d_mask, HW * sizeof(int32_t));
        cudaMemcpy(d_mask, padded_mask.ptr<int32_t>(), HW * sizeof(int32_t),
                   cudaMemcpyHostToDevice);

        // 上传 objects 数组到 GPU（用于 one_hot_encode kernel）
        std::vector<int32_t> obj_i32(objects.begin(), objects.end());
        int32_t* d_objects = nullptr;
        cudaMalloc(&d_objects, obj_i32.size() * sizeof(int32_t));
        cudaMemcpy(d_objects, obj_i32.data(), obj_i32.size() * sizeof(int32_t),
                   cudaMemcpyHostToDevice);

        if (need_segment && pred_prob_with_bg.IsTensor())
        {
            // pred_prob_with_bg: [num_ch, H, W] (GPU)
            auto pred_shape = GA::shape(pred_prob_with_bg);
            int num_existing = static_cast<int>(pred_shape[0]) - 1;
            int total_obj = object_manager.num_obj();

            // 1. 在 pred 中将 input mask > 0 的像素对应通道置零（GPU kernel）
            // pred_no_bg = pred[1:, :, :] → [num_existing, HW]
            cuda::mask_merge_zero(
                GA::data_ptr(pred_prob_with_bg) + HW,  // skip bg channel
                d_mask, num_existing, HW);

            // 2. 构建 merged_no_bg: [total_obj, H, W]（GPU）
            auto merged = alloc.zeros({total_obj, H, W});

            // 拷贝已有预测（跳过 bg）
            int copy_obj = std::min(num_existing, total_obj);
            if (copy_obj > 0)
            {
                cuda::copy_d2d(GA::data_ptr(merged),
                               GA::data_ptr(pred_prob_with_bg) + HW,
                               copy_obj * HW);
            }

            // 3. 新对象 one-hot 写入对应通道
            for (size_t mi = 0; mi < tmp_ids.size(); ++mi)
            {
                int ch = tmp_ids[mi] - 1;
                // one_hot_encode 写单个 object 到 merged 的 ch 通道
                cuda::one_hot_encode(d_mask, d_objects + mi,
                                    GA::data_ptr(merged) + ch * HW, 1, HW);
            }

            pred_prob_with_bg = ortcore::gpu_aggregate(alloc, merged);
        }
        else
        {
            int total_obj = object_manager.num_obj();
            auto one_hot_gpu = alloc.zeros({total_obj, H, W});

            for (size_t mi = 0; mi < tmp_ids.size(); ++mi)
            {
                int ch = tmp_ids[mi] - 1;
                cuda::one_hot_encode(d_mask, d_objects + mi,
                                    GA::data_ptr(one_hot_gpu) + ch * HW, 1, HW);
            }

            pred_prob_with_bg = ortcore::gpu_aggregate(alloc, one_hot_gpu);
        }

        cudaFree(d_mask);
        cudaFree(d_objects);
#endif
    }

    // Update last_mask: [1, num_objects, H, W]
    if (pred_prob_with_bg.IsTensor())
    {
        auto shape = GA::shape(pred_prob_with_bg);
        int num_ch = static_cast<int>(shape[0]);
        int num_obj = num_ch - 1;
        int H = static_cast<int>(shape[1]);
        int W = static_cast<int>(shape[2]);
        int HW = H * W;

        if (num_obj > 0)
        {
#ifdef ENABLE_ONNXRUNTIME
            auto lm = alloc.allocate({1, num_obj, H, W});
            cuda::copy_d2d(GA::data_ptr(lm),
                           GA::data_ptr(pred_prob_with_bg) + HW,  // skip bg
                           num_obj * HW);
            last_mask = std::move(lm);
#endif
        }
    }

    // Save as memory
    if (is_mem_frame || force_permanent)
    {
        add_memory(feat, last_mask, force_permanent);
    }

    // Clean up feature cache
    feature_store.keep_only(-1);

    // 下载到 CPU 做后处理：unpad + resize
    cv::Mat output;
    if (pred_prob_with_bg.IsTensor())
    {
#ifdef ENABLE_ONNXRUNTIME
        cv::Mat output_cpu = alloc.download(pred_prob_with_bg);
        output = utils::unpad(output_cpu, pad);

        if (resize_needed)
        {
            int num_ch = output.size[0];
            int oh = output.size[1];
            int ow = output.size[2];
            int out_sizes[] = {num_ch, orig_h, orig_w};
            cv::Mat resized(3, out_sizes, CV_32FC1);
            for (int c = 0; c < num_ch; ++c)
            {
                cv::Mat slice(oh, ow, CV_32FC1, output.ptr<float>() + c * oh * ow);
                cv::Mat rs;
                cv::resize(slice, rs, cv::Size(orig_w, orig_h), 0, 0, cv::INTER_LINEAR);
                std::memcpy(resized.ptr<float>() + c * orig_h * orig_w, rs.ptr<float>(),
                            orig_h * orig_w * sizeof(float));
            }
            output = resized;
        }
#endif
    }

    return output;
}

// ── Public InferenceCore ────────────────────────────────────────────

InferenceCore::InferenceCore(const CutieConfig& config,
                             std::shared_ptr<linden::log::ILogger> logger)
    : impl_(std::make_unique<Impl>(config, std::move(logger)))
{
}

InferenceCore::~InferenceCore() = default;

InferenceCore::InferenceCore(InferenceCore&&) noexcept = default;
InferenceCore& InferenceCore::operator=(InferenceCore&&) noexcept = default;

cv::Mat InferenceCore::step(const cv::Mat& image, const cv::Mat& mask,
                            const std::vector<ObjectId>& objects, bool end, bool force_permanent)
{
    return impl_->step(image, mask, objects, end, force_permanent);
}

void InferenceCore::delete_objects(const std::vector<ObjectId>& objects)
{
    impl_->object_manager.delete_objects(objects);
    impl_->memory->purge_except(impl_->object_manager.all_obj_ids());
}

std::vector<ObjectId> InferenceCore::active_objects() const
{
    return impl_->object_manager.all_obj_ids();
}

int InferenceCore::num_objects() const
{
    return impl_->object_manager.num_obj();
}

void InferenceCore::clear_memory()
{
    impl_->curr_ti = -1;
    impl_->last_mem_ti = 0;
#ifdef ENABLE_ONNXRUNTIME
    impl_->memory =
        std::make_unique<MemoryManager>(impl_->cfg, &impl_->object_manager,
                                        &impl_->network->gpu_alloc());
#endif
}

void InferenceCore::clear_non_permanent_memory()
{
    impl_->curr_ti = -1;
    impl_->last_mem_ti = 0;
    impl_->memory->clear_non_permanent_memory();
}

void InferenceCore::clear_sensory_memory()
{
    impl_->curr_ti = -1;
    impl_->last_mem_ti = 0;
    impl_->memory->clear_sensory_memory();
}

void InferenceCore::update_config(const CutieConfig& config)
{
    impl_->cfg = config;
    impl_->mem_every = config.mem_every;
    impl_->memory->update_config(config);
}

}  // namespace core
}  // namespace cutie
