/**
 * @file inference_core.cpp
 * @brief InferenceCore implementation (main inference pipeline).
 *
 * Implements the core inference loop that orchestrates the complete Cutie
 * pipeline: image encoding, memory read, segmentation, and memory write.
 * Manages frame-level state and memory scheduling.
 */

#include <linden_logger/logger_interface.hpp>

#include "cutie/core/image_feature_store.h"
#include "cutie/core/inference_core.h"
#include "cutie/core/memory_manager.h"
#include "cutie/core/object_manager.h"
#include "cutie/core/processor.h"
#include "cutie/common/cuda_kernels.h"
#include "cutie/common/gpu_image_preprocess.h"
#include "cutie/common/gpu_mask_preprocess.h"
#include "cutie/common/gpu_postprocess.h"
#include "cutie/common/gpu_tensor_ops.h"
#include "cutie/utils.h"

#ifdef ENABLE_ONNXRUNTIME
#include "cutie/ort/cv/ort_cutie.h"
#endif

#ifdef ENABLE_TENSORRT
#include "cutie/trt/cv/trt_cutie.h"
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
namespace
{
// 诊断：把 GPU mask（CV_32SC1，值=ObjectId）download 到 CPU，打印每个 id 的中心+面积
void diag_log_id_centers(const cv::cuda::GpuMat& gpu_mask,
                         const std::vector<ObjectId>& ids,
                         linden::log::ILogger* logger, const std::string& tag)
{
    if (!logger || gpu_mask.empty() || ids.empty()) return;
    cv::Mat cpu;
    gpu_mask.download(cpu);
    if (cpu.type() != CV_32SC1) cpu.convertTo(cpu, CV_32SC1);

    // [DIAG] 打印 mask 尺寸和实际内存布局
    logger->info("[CUTIE-DIAG] {} mask: rows={} cols={} (rows=H, cols=W), step={}, continuous={}",
                 tag, cpu.rows, cpu.cols, cpu.step[0], cpu.isContinuous());

    for (auto oid : ids)
    {
        cv::Mat eq;
        cv::compare(cpu, static_cast<int>(oid), eq, cv::CMP_EQ);
        int area = cv::countNonZero(eq);
        if (area == 0)
        {
            logger->info("[CUTIE-DIAG] {} ID={} 像素=0", tag, static_cast<int>(oid));
            continue;
        }
        cv::Moments m = cv::moments(eq, true);
        float cx = static_cast<float>(m.m10 / m.m00);
        float cy = static_cast<float>(m.m01 / m.m00);

        // [DIAG] 找前5个非零像素的实际 (row, col) 位置验证内存布局
        std::vector<std::pair<int,int>> samples;
        for (int r = 0; r < cpu.rows && samples.size() < 5; ++r) {
            for (int c = 0; c < cpu.cols && samples.size() < 5; ++c) {
                if (cpu.at<int32_t>(r, c) == static_cast<int>(oid)) {
                    samples.push_back({r, c});
                }
            }
        }
        std::string samples_str;
        for (auto [r, c] : samples) {
            samples_str += "(" + std::to_string(c) + "," + std::to_string(r) + ") ";
        }

        logger->info("[CUTIE-DIAG] {} ID={} center=({:.1f},{:.1f}) area={}px | "
                     "前5个像素(x,y): {}",
                     tag, static_cast<int>(oid), cx, cy, area, samples_str);
    }
}

// 诊断：把 prob tensor [C,H,W] 的指定通道 argmax 中心提取出来（仅诊断用，CPU 路径）
void diag_log_prob_centers(const Ort::Value& prob, const std::vector<ObjectId>& obj_ids,
                           linden::log::ILogger* logger, const std::string& tag)
{
    if (!logger || !prob.IsTensor()) return;
#ifdef ENABLE_ONNXRUNTIME
    auto shape = ortcore::GpuMemoryAllocator::shape(prob);
    if (shape.size() < 3) return;
    int C = static_cast<int>(shape[0]);
    int H = static_cast<int>(shape[1]);
    int W = static_cast<int>(shape[2]);
    int HW = H * W;
    // 下载到 CPU（仅诊断）
    std::vector<float> host(C * HW);
    cudaMemcpy(host.data(), ortcore::GpuMemoryAllocator::data_ptr(prob),
               C * HW * sizeof(float), cudaMemcpyDeviceToHost);
    // bg=ch0；ch i+1 对应 obj_ids[i]
    for (size_t i = 0; i < obj_ids.size() && static_cast<int>(i + 1) < C; ++i)
    {
        const float* ch = host.data() + (i + 1) * HW;
        // argmax over channel: 取 prob > 0.5 视为前景，算 moments
        cv::Mat fg(H, W, CV_8UC1);
        for (int p = 0; p < HW; ++p) fg.data[p] = (ch[p] > 0.5f) ? 255 : 0;
        int area = cv::countNonZero(fg);
        if (area == 0)
        {
            logger->info("[CUTIE-DIAG] {} ID={} prob>0.5 像素=0 (tensor rows={} cols={})", tag,
                         static_cast<int>(obj_ids[i]), H, W);
            continue;
        }
        cv::Moments m = cv::moments(fg, true);
        float cx = static_cast<float>(m.m10 / m.m00);
        float cy = static_cast<float>(m.m01 / m.m00);
        logger->info("[CUTIE-DIAG] {} ID={} center=({:.1f},{:.1f}) area={}px (tensor rows={} cols={})",
                     tag, static_cast<int>(obj_ids[i]), cx, cy, area, H, W);
    }
#else
    (void)prob; (void)obj_ids; (void)tag;
#endif
}
}  // namespace

namespace core
{

using GA = ortcore::GpuMemoryAllocator;

// ── Impl ────────────────────────────────────────────────────────────
// InferenceCore 的内部实现结构，管理推理状态和内存。
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
    ortcore::GpuImagePreprocessor gpu_preprocessor;

#ifdef ENABLE_ONNXRUNTIME
    std::unique_ptr<ortcv::OrtCutie> network;
#elif defined(ENABLE_TENSORRT)
    std::unique_ptr<trtcv::TrtCutie> network;
#endif

    explicit Impl(const CutieConfig& config, std::shared_ptr<linden::log::ILogger> logger_ptr);

    /// GPU 推理核心路径（step 和 step_gpu 共用）。
    types::GpuCutieMask step_gpu_impl(const cv::cuda::GpuMat& gpu_image,
                                      const cv::cuda::GpuMat& gpu_mask,
                                      const std::vector<ObjectId>& objects, bool end,
                                      bool force_permanent);

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
    logger->info("InferenceCore: GPU inference initialized with ONNX Runtime (model_dir={})", cfg.model_dir);
#elif defined(ENABLE_TENSORRT)
    network = std::make_unique<trtcv::TrtCutie>(cfg, logger);
    model_h_ = network->model_input_h();
    model_w_ = network->model_input_w();

    memory = std::make_unique<MemoryManager>(cfg, &object_manager, &network->gpu_alloc());

    if (model_h_ > 0 && model_w_ > 0)
        logger->info("InferenceCore: model input size {}x{} (auto-resize enabled)", model_h_,
                     model_w_);
    logger->info("InferenceCore: GPU inference initialized with TensorRT (model_dir={})", cfg.model_dir);
#else
    logger->error("InferenceCore: no backend enabled. Build with -DENABLE_ONNXRUNTIME=ON or -DENABLE_TENSORRT=ON");
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

    // logits: [B, N_static, H/4, W/4] → slice to [B, N_actual, H/4, W/4] → sigmoid → aggregate → upsample → softmax
    auto logits_shape = GA::shape(seg_result.logits);
    int B = static_cast<int>(logits_shape[0]);
    int N_static = static_cast<int>(logits_shape[1]);
    int lh = static_cast<int>(logits_shape[2]);
    int lw = static_cast<int>(logits_shape[3]);

    // Get actual number of objects
    int N_actual = object_manager.num_obj();

    // Slice to actual number of objects: [B, N_static, lh, lw] → [B, N_actual, lh, lw]
    auto logits_sliced = alloc.slice_dim(seg_result.logits, 1, N_actual);

    // Reshape: [B, N_actual, lh, lw] → [N_actual, lh, lw] (remove batch dim)
    auto logits_3d = Ort::Value::CreateTensor<float>(
        alloc.memory_info(), GA::data_ptr(logits_sliced), N_actual * lh * lw,
        std::vector<int64_t>{N_actual, lh, lw}.data(), 3);
    auto logits_clone = alloc.clone(logits_3d);

    // Sigmoid → prob
    auto prob_no_bg = ortcore::gpu_sigmoid(alloc, logits_clone);

    // Aggregate: add background, convert to logits (no softmax yet)
    auto logits_with_bg = ortcore::gpu_aggregate_logits(alloc, prob_no_bg);

    // Upsample 4× to full resolution in logit space（GPU）
    int full_h = lh * 4;
    int full_w = lw * 4;
    auto logits_upsampled = alloc.resize_channels(logits_with_bg, full_h, full_w);

    // Softmax along channel dimension
    return ortcore::gpu_softmax_channels(alloc, logits_upsampled);
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

// ── step_gpu_impl（GPU 推理核心路径）──────────────────────────────────

types::GpuCutieMask InferenceCore::Impl::step_gpu_impl(const cv::cuda::GpuMat& gpu_image,
                                                        const cv::cuda::GpuMat& gpu_mask,
                                                        const std::vector<ObjectId>& objects,
                                                        bool end, bool force_permanent)
{
    curr_ti++;

    int orig_h = gpu_image.rows;
    int orig_w = gpu_image.cols;

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

    // [CUTIE-DIAG] 入口尺寸 + target 尺寸
    logger->info("[CUTIE-DIAG] step_gpu_impl 入口: orig rows={} cols={} (rows=H,cols=W), "
                 "target rows={} cols={}, resize_needed={}, max_internal_size={}",
                 orig_h, orig_w, target_h, target_w, resize_needed, cfg.max_internal_size);
    if (!gpu_mask.empty())
    {
        logger->info("[CUTIE-DIAG] gpu_mask 入口: rows={} cols={}, type={}",
                     gpu_mask.rows, gpu_mask.cols, gpu_mask.type());
        diag_log_id_centers(gpu_mask, objects, logger.get(), "①入口gpu_mask(orig坐标)");
    }

#ifdef ENABLE_ONNXRUNTIME
    auto& alloc = network->gpu_alloc();

    // ① 融合 GPU 预处理：单 kernel 完成 resize + pad + BGR→RGB + norm + CHW
    auto [image_gpu_val, pad_out] = gpu_preprocessor.preprocess(gpu_image, target_h, target_w,
                                                                 alloc);
    pad = pad_out;
    cached_image_gpu = std::move(image_gpu_val);

    // [CUTIE-DIAG] 图像预处理后 pad 信息
    logger->info("[CUTIE-DIAG] 图像 preprocess 后 pad=[top={}, bottom={}, left={}, right={}], "
                 "padded={}x{} (期望 target={}x{})",
                 pad[0], pad[1], pad[2], pad[3], target_h + pad[0] + pad[1],
                 target_w + pad[2] + pad[3], target_h, target_w);
#endif

    bool is_mem_frame = ((curr_ti - last_mem_ti >= mem_every) || !gpu_mask.empty()) && !end;
    bool need_segment =
        gpu_mask.empty() || (object_manager.num_obj() > 0 && !object_manager.has_all(objects));
    int stagger_delta = curr_ti - last_mem_ti;
    bool update_sensory = stagger_ti.count(stagger_delta) > 0 && !end;

    ensure_features(curr_ti);
    auto& feat = feature_store.get(curr_ti);

    Ort::Value pred_prob_with_bg{nullptr};

    if (need_segment)
    {
        pred_prob_with_bg = segment(feat, update_sensory);
    }

    // ② Handle input mask（全 GPU 路径）
    if (!gpu_mask.empty())
    {
        auto [tmp_ids, new_ids] = object_manager.add_new_objects(objects);

        // GPU resize mask (nearest)
        cv::cuda::GpuMat resized_mask = gpu_mask;
        if (resize_needed)
        {
            logger->info("[CUTIE-DIAG] 即将调用 gpu_resize_mask_nearest: gpu_mask.rows={} cols={}, "
                         "target_h={} target_w={}",
                         gpu_mask.rows, gpu_mask.cols, target_h, target_w);
            resized_mask = ortcore::gpu_resize_mask_nearest(gpu_mask, target_h, target_w);
            logger->info("[CUTIE-DIAG] gpu_resize_mask_nearest 返回: resized_mask.rows={} cols={}",
                         resized_mask.rows, resized_mask.cols);
        }

        // [CUTIE-DIAG] mask resize 后
        logger->info("[CUTIE-DIAG] resized_mask: {}x{} (rows=H, cols=W), resize_needed={}",
                     resized_mask.rows, resized_mask.cols, resize_needed);

        // [DIAG] 验证 resize 映射关系：下载前后对比
        if (resize_needed && logger)
        {
            cv::Mat src_cpu, dst_cpu;
            gpu_mask.download(src_cpu);
            resized_mask.download(dst_cpu);
            if (src_cpu.type() != CV_32SC1) src_cpu.convertTo(src_cpu, CV_32SC1);
            if (dst_cpu.type() != CV_32SC1) dst_cpu.convertTo(dst_cpu, CV_32SC1);

            for (auto oid : objects)
            {
                if (oid == 0) continue;
                // 在源 mask 找第一个非零像素
                int src_r = -1, src_c = -1;
                for (int r = 0; r < src_cpu.rows && src_r < 0; ++r) {
                    for (int c = 0; c < src_cpu.cols && src_r < 0; ++c) {
                        if (src_cpu.at<int32_t>(r, c) == static_cast<int>(oid)) {
                            src_r = r; src_c = c;
                        }
                    }
                }
                if (src_r < 0) continue;

                // 计算该源像素在目标中的预期位置
                int exp_dst_c = src_c * dst_cpu.cols / src_cpu.cols;
                int exp_dst_r = src_r * dst_cpu.rows / src_cpu.rows;

                // 在目标 mask 找第一个非零像素
                int dst_r = -1, dst_c = -1;
                for (int r = 0; r < dst_cpu.rows && dst_r < 0; ++r) {
                    for (int c = 0; c < dst_cpu.cols && dst_r < 0; ++c) {
                        if (dst_cpu.at<int32_t>(r, c) == static_cast<int>(oid)) {
                            dst_r = r; dst_c = c;
                        }
                    }
                }

                // [DIAG] 额外验证：反向查询 - 目标首像素对应源的哪个位置
                int back_src_c = -1, back_src_r = -1;
                if (dst_r >= 0 && dst_c >= 0) {
                    // kernel 逻辑：sx = x * src_w / dst_w, sy = y * src_h / dst_h
                    back_src_c = dst_c * src_cpu.cols / dst_cpu.cols;
                    back_src_r = dst_r * src_cpu.rows / dst_cpu.rows;
                    // 检查源的该位置是否真的是这个 ID
                    int32_t src_val_at_back = src_cpu.at<int32_t>(back_src_r, back_src_c);

                    // [DIAG] 再多采样几个目标像素验证
                    std::string sample_str;
                    for (int offset = 0; offset < 3 && (dst_c + offset) < dst_cpu.cols; ++offset) {
                        int test_dst_c = dst_c + offset;
                        int test_src_c = test_dst_c * src_cpu.cols / dst_cpu.cols;
                        int test_src_r = dst_r * src_cpu.rows / dst_cpu.rows;
                        int32_t src_val = src_cpu.at<int32_t>(test_src_r, test_src_c);
                        int32_t dst_val = dst_cpu.at<int32_t>(dst_r, test_dst_c);
                        sample_str += "dst(" + std::to_string(test_dst_c) + "," +
                                      std::to_string(dst_r) + ")→src(" +
                                      std::to_string(test_src_c) + "," +
                                      std::to_string(test_src_r) + "):" +
                                      std::to_string(dst_val) + "/" +
                                      std::to_string(src_val) + "; ";
                    }

                    logger->info("[CUTIE-DIAG] resize映射验证 ID={}: "
                                 "源首像素(c,r)=({},{}) → 预期目标({},{}) vs 实际目标({},{}), "
                                 "反查: 目标({},{})应采样源({},{}), 源该位置值={} | 采样验证: {}",
                                 static_cast<int>(oid), src_c, src_r, exp_dst_c, exp_dst_r,
                                 dst_c, dst_r, dst_c, dst_r, back_src_c, back_src_r,
                                 src_val_at_back, sample_str);
                }
                else {
                    logger->info("[CUTIE-DIAG] resize映射验证 ID={}: 源首像素(c,r)=({},{}) → "
                                 "预期目标({},{}) vs 实际目标({},{})",
                                 static_cast<int>(oid), src_c, src_r, exp_dst_c, exp_dst_r,
                                 dst_c, dst_r);
                }
            }
        }

        diag_log_id_centers(resized_mask, objects, logger.get(), "②mask resize后(target坐标)");

        // GPU pad mask
        cv::cuda::GpuMat padded_mask = ortcore::gpu_pad_mask(resized_mask, pad);

        // [CUTIE-DIAG] mask pad 后
        logger->info("[CUTIE-DIAG] padded_mask: rows={} cols={}, pad=[top={},left={}]",
                     padded_mask.rows, padded_mask.cols, pad[0], pad[2]);
        diag_log_id_centers(padded_mask, objects, logger.get(), "③mask pad后(padded坐标)");

        int H = padded_mask.rows;
        int W = padded_mask.cols;
        int HW = H * W;

#ifdef ENABLE_ONNXRUNTIME
        // GPU mask data pointer（已在 GPU 上）
        const int32_t* d_mask = reinterpret_cast<const int32_t*>(padded_mask.data);
        int mask_pitch = static_cast<int>(padded_mask.step / sizeof(int32_t));

        // [DIAG] 检查 padded_mask 的 pitch
        if (padded_mask.step != padded_mask.cols * sizeof(int32_t)) {
            logger->info("[CUTIE-DIAG] padded_mask 有 padding: step={} vs cols*4={}, pitch={}",
                         padded_mask.step, padded_mask.cols * sizeof(int32_t), mask_pitch);
        }

        // 上传 objects 数组到 GPU（通常 <20 个 int32，开销可忽略）
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
            cuda::mask_merge_zero(GA::data_ptr(pred_prob_with_bg) + HW,  // skip bg channel
                                  d_mask, mask_pitch, H, W, num_existing);

            // 2. 构建 merged_no_bg: [total_obj, H, W]（GPU）
            auto merged = alloc.zeros({total_obj, H, W});

            // 拷贝已有预测（跳过 bg）
            int copy_obj = std::min(num_existing, total_obj);
            if (copy_obj > 0)
            {
                cuda::copy_d2d(GA::data_ptr(merged), GA::data_ptr(pred_prob_with_bg) + HW,
                               copy_obj * HW);
            }

            // 3. 新对象 one-hot 写入对应通道
            for (size_t mi = 0; mi < tmp_ids.size(); ++mi)
            {
                int ch = tmp_ids[mi] - 1;
                cuda::one_hot_encode(d_mask, mask_pitch, H, W, d_objects + mi,
                                     GA::data_ptr(merged) + ch * HW, 1);
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
                cuda::one_hot_encode(d_mask, mask_pitch, H, W, d_objects + mi,
                                     GA::data_ptr(one_hot_gpu) + ch * HW, 1);
            }

            pred_prob_with_bg = ortcore::gpu_aggregate(alloc, one_hot_gpu);
        }

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

    // ③ GPU 后处理：unpad + resize + argmax
    types::GpuCutieMask result;
    if (pred_prob_with_bg.IsTensor())
    {
#ifdef ENABLE_ONNXRUNTIME
        // [CUTIE-DIAG] unpad 前 prob shape
        {
            auto s = ortcore::GpuMemoryAllocator::shape(pred_prob_with_bg);
            logger->info("[CUTIE-DIAG] pred_prob_with_bg shape=[C={}, rows={}, cols={}] (unpad前)",
                         s.size() > 0 ? s[0] : -1, s.size() > 1 ? s[1] : -1,
                         s.size() > 2 ? s[2] : -1);
            diag_log_prob_centers(pred_prob_with_bg, object_manager.all_obj_ids(), logger.get(),
                                  "④prob unpad前(padded坐标)");
        }

        auto prob_unpadded = ortcore::gpu_unpad(alloc, pred_prob_with_bg, pad);

        // [CUTIE-DIAG] unpad 后 prob shape + 中心
        {
            auto s = ortcore::GpuMemoryAllocator::shape(prob_unpadded);
            logger->info("[CUTIE-DIAG] prob_unpadded shape=[C={}, rows={}, cols={}] (unpad后, "
                         "期望 rows={} cols={})",
                         s.size() > 0 ? s[0] : -1, s.size() > 1 ? s[1] : -1,
                         s.size() > 2 ? s[2] : -1, target_h, target_w);
            diag_log_prob_centers(prob_unpadded, object_manager.all_obj_ids(), logger.get(),
                                  "⑤prob unpad后(target坐标)");
        }

        if (resize_needed)
        {
            prob_unpadded = alloc.resize_channels(prob_unpadded, orig_h, orig_w);

            // [CUTIE-DIAG] resize_channels 后
            auto s = ortcore::GpuMemoryAllocator::shape(prob_unpadded);
            logger->info("[CUTIE-DIAG] prob resize_channels(rows={},cols={}) 后 "
                         "shape=[C={}, rows={}, cols={}]",
                         orig_h, orig_w, s.size() > 0 ? s[0] : -1, s.size() > 1 ? s[1] : -1,
                         s.size() > 2 ? s[2] : -1);
            diag_log_prob_centers(prob_unpadded, object_manager.all_obj_ids(), logger.get(),
                                  "⑥prob resize回orig后(orig坐标)");
        }

        result.index_mask =
            ortcore::gpu_prob_to_index_mask(alloc, prob_unpadded, object_manager.all_obj_ids());
        result.gpu_prob = std::move(prob_unpadded);

        // [CUTIE-DIAG] 最终 index_mask 中心
        diag_log_id_centers(result.index_mask, object_manager.all_obj_ids(), logger.get(),
                            "⑦最终index_mask(应=orig坐标)");
#endif
    }

    result.object_ids = object_manager.all_obj_ids();
    result.flag = true;
    return result;
}

// ── step（CPU 包装：upload → step_gpu_impl → download）──────────────

cv::Mat InferenceCore::Impl::step(const cv::Mat& image, const cv::Mat& mask,
                                  const std::vector<ObjectId>& objects, bool end,
                                  bool force_permanent)
{
    // 上传图像到 GPU
    cv::cuda::GpuMat gpu_image;
    gpu_image.upload(image);

    // 上传 mask 到 GPU（如果有）
    cv::cuda::GpuMat gpu_mask;
    if (!mask.empty())
    {
        cv::Mat mask_i32 = mask;
        if (mask_i32.type() != CV_32SC1) mask.convertTo(mask_i32, CV_32SC1);
        gpu_mask.upload(mask_i32);
    }

    // 调用 GPU 路径
    auto gpu_result = step_gpu_impl(gpu_image, gpu_mask, objects, end, force_permanent);

    // 下载概率图到 CPU
    if (gpu_result.gpu_prob.IsTensor())
    {
#ifdef ENABLE_ONNXRUNTIME
        return network->gpu_alloc().download(gpu_result.gpu_prob);
#endif
    }

    return cv::Mat();
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

types::GpuCutieMask InferenceCore::step_gpu(const cv::cuda::GpuMat& image_gpu,
                                            const cv::cuda::GpuMat& mask_gpu,
                                            const std::vector<ObjectId>& objects, bool end,
                                            bool force_permanent)
{
    return impl_->step_gpu_impl(image_gpu, mask_gpu, objects, end, force_permanent);
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
