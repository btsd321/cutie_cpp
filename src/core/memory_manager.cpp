/**
 * @file memory_manager.cpp
 * @brief MemoryManager implementation (three-tier memory system).
 *
 * Implements memory management for working memory (recent frames),
 * long-term memory (compressed prototypes), and sensory memory (per-object GRU state).
 * Coordinates memory read/write operations across all three tiers.
 */

#include <algorithm>
#include <cstring>
#include <set>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

#include "cutie/core/memory_manager.h"
#include "cutie/core/processor.h"
#include "cutie/common/cuda_kernels.h"

#include <cuda_runtime.h>

namespace cutie
{
namespace core
{

using GA = ortcore::GpuMemoryAllocator;

// ── 构造 / 配置 ────────────────────────────────────────────────────
// 初始化三层内存系统，配置参数来自 CutieConfig。
    : alloc_(alloc),
      object_manager_(object_manager),
      sensory_dim_(cfg.model.sensory_dim),
      top_k_(cfg.top_k),
      chunk_size_(cfg.chunk_size),
      use_long_term_(cfg.use_long_term),
      count_long_term_usage_(cfg.long_term.count_usage)
{
    if (use_long_term_)
    {
        max_mem_frames_ = cfg.long_term.max_mem_frames - 1;
        min_mem_frames_ = cfg.long_term.min_mem_frames - 1;
        num_prototypes_ = cfg.long_term.num_prototypes;
        max_long_tokens_ = cfg.long_term.max_num_tokens;
        buffer_tokens_ = cfg.long_term.buffer_tokens;
    }
    else
    {
        max_mem_frames_ = cfg.max_mem_frames - 1;
    }

    work_mem_ = std::make_unique<KeyValueMemoryStore>(alloc, use_long_term_, use_long_term_);
    if (use_long_term_)
    {
        long_mem_ = std::make_unique<KeyValueMemoryStore>(alloc, false, count_long_term_usage_);
    }
}

void MemoryManager::update_config(const CutieConfig& cfg)
{
    config_stale_ = true;
    top_k_ = cfg.top_k;

    if (use_long_term_)
    {
        max_mem_frames_ = cfg.long_term.max_mem_frames - 1;
        min_mem_frames_ = cfg.long_term.min_mem_frames - 1;
        num_prototypes_ = cfg.long_term.num_prototypes;
        max_long_tokens_ = cfg.long_term.max_num_tokens;
        buffer_tokens_ = cfg.long_term.buffer_tokens;
    }
    else
    {
        max_mem_frames_ = cfg.max_mem_frames - 1;
    }
}

// ── 辅助：按 ID 获取数据 ───────────────────────────────────────────

Ort::Value MemoryManager::get_mask_by_ids(const Ort::Value& mask,
                                           const std::vector<ObjectId>& obj_ids)
{
    // mask: [B, num_all_objects, H, W], 取指定 obj_ids 的通道
    auto shape = GA::shape(mask);
    int B = static_cast<int>(shape[0]);
    int H = static_cast<int>(shape[2]);
    int W = static_cast<int>(shape[3]);
    int num_all = static_cast<int>(shape[1]);
    int num_sel = static_cast<int>(obj_ids.size());

    auto result = alloc_->allocate({B, num_sel, H, W});
    int HW = H * W;

    for (int n = 0; n < num_sel; ++n)
    {
        int tmp_id = object_manager_->find_tmp_by_id(obj_ids[n]);
        int src_idx = tmp_id - 1;
        for (int b = 0; b < B; ++b)
        {
            cuda::copy_d2d(
                GA::data_ptr(result) + (b * num_sel + n) * HW,
                GA::data_ptr(mask) + (b * num_all + src_idx) * HW, HW);
        }
    }
    return result;
}

Ort::Value MemoryManager::get_sensory_by_ids(const std::vector<ObjectId>& obj_ids)
{
    if (obj_ids.empty()) return Ort::Value{nullptr};

    auto it = sensory_.find(obj_ids[0]);
    if (it == sensory_.end()) return Ort::Value{nullptr};

    auto first_shape = GA::shape(it->second);
    int B = static_cast<int>(first_shape[0]);
    int C = static_cast<int>(first_shape[1]);
    int H = static_cast<int>(first_shape[2]);
    int W = static_cast<int>(first_shape[3]);
    int num_obj = static_cast<int>(obj_ids.size());
    int slice = C * H * W;

    auto result = alloc_->allocate({B, num_obj, C, H, W});
    for (int n = 0; n < num_obj; ++n)
    {
        auto sit = sensory_.find(obj_ids[n]);
        if (sit == sensory_.end())
            throw std::runtime_error("MemoryManager: sensory not found for obj " +
                                     std::to_string(obj_ids[n]));
        for (int b = 0; b < B; ++b)
        {
            cuda::copy_d2d(
                GA::data_ptr(result) + b * num_obj * slice + n * slice,
                GA::data_ptr(sit->second) + b * slice, slice);
        }
    }
    return result;
}

Ort::Value MemoryManager::get_object_mem_by_ids(const std::vector<ObjectId>& obj_ids)
{
    if (obj_ids.empty()) return Ort::Value{nullptr};
    if (obj_v_.count(obj_ids[0]) == 0) return Ort::Value{nullptr};

    auto first_shape = GA::shape(obj_v_.at(obj_ids[0]));
    int B = static_cast<int>(first_shape[0]);
    int Q = static_cast<int>(first_shape[1]);
    int C = static_cast<int>(first_shape[2]);
    int num_obj = static_cast<int>(obj_ids.size());
    int slice = Q * C;

    auto result = alloc_->allocate({B, num_obj, Q, C});
    for (int n = 0; n < num_obj; ++n)
    {
        auto oit = obj_v_.find(obj_ids[n]);
        if (oit == obj_v_.end())
            throw std::runtime_error("MemoryManager: obj_v not found for obj " +
                                     std::to_string(obj_ids[n]));
        for (int b = 0; b < B; ++b)
        {
            cuda::copy_d2d(
                GA::data_ptr(result) + b * num_obj * slice + n * slice,
                GA::data_ptr(oit->second) + b * slice, slice);
        }
    }
    return result;
}

Ort::Value MemoryManager::get_visual_values_by_ids(const std::vector<ObjectId>& obj_ids)
{
    if (obj_ids.empty()) return Ort::Value{nullptr};

    int num_obj = static_cast<int>(obj_ids.size());
    const auto& wv = work_mem_->value();
    auto it0 = wv.find(obj_ids[0]);
    if (it0 == wv.end()) return Ort::Value{nullptr};

    auto ws = GA::shape(it0->second);
    int B = static_cast<int>(ws[0]);
    int CV = static_cast<int>(ws[1]);
    int N_work = static_cast<int>(ws[2]);

    int N_long = 0;
    bool has_long = use_long_term_ && long_mem_ && long_mem_->contains(obj_ids[0]);
    if (has_long)
    {
        N_long = static_cast<int>(GA::shape(long_mem_->value().at(obj_ids[0]))[2]);
    }
    int N_total = N_long + N_work;

    auto result = alloc_->allocate({B, num_obj, CV, N_total});

    for (int n = 0; n < num_obj; ++n)
    {
        auto obj_id = obj_ids[n];
        const auto& work_v = wv.at(obj_id);

        for (int b = 0; b < B; ++b)
        {
            float* dst = GA::data_ptr(result) + b * num_obj * CV * N_total + n * CV * N_total;

            if (has_long)
            {
                const auto& long_v = long_mem_->value().at(obj_id);
                for (int cv_i = 0; cv_i < CV; ++cv_i)
                {
                    cuda::copy_d2d(
                        dst + cv_i * N_total,
                        GA::data_ptr(long_v) + b * CV * N_long + cv_i * N_long, N_long);
                    cuda::copy_d2d(
                        dst + cv_i * N_total + N_long,
                        GA::data_ptr(work_v) + b * CV * N_work + cv_i * N_work, N_work);
                }
            }
            else
            {
                cuda::copy_d2d(dst, GA::data_ptr(work_v) + b * CV * N_work, CV * N_work);
            }
        }
    }
    return result;
}

// ── read ────────────────────────────────────────────────────────────

std::unordered_map<ObjectId, Ort::Value> MemoryManager::read(
    Ort::Value& pix_feat, Ort::Value& query_key, Ort::Value& selection, Ort::Value& last_mask,
    NetworkCallbacks& network)
{
    auto pf_shape = GA::shape(pix_feat);
    int h = static_cast<int>(pf_shape[2]);
    int w = static_cast<int>(pf_shape[3]);
    int B = static_cast<int>(pf_shape[0]);

    std::unordered_map<ObjectId, Ort::Value> all_readout_mem;

    const auto& buckets = work_mem_->buckets();
    for (auto& [bucket_id, bucket_objs] : buckets)
    {
        Ort::Value affinity{nullptr};

        if (use_long_term_ && long_mem_ && long_mem_->engaged(bucket_id))
        {
            int long_mem_size = long_mem_->size(bucket_id);
            auto memory_key = alloc_->concat(
                long_mem_->key().at(bucket_id), work_mem_->key().at(bucket_id), -1);
            auto shrink = alloc_->concat(
                long_mem_->shrinkage().at(bucket_id), work_mem_->shrinkage().at(bucket_id), -1);

            auto similarity = ortcore::gpu_get_similarity(*alloc_, memory_key, shrink,
                                                           query_key, selection);
            auto [aff, usg] = ortcore::gpu_do_softmax(*alloc_, similarity, top_k_, true);
            affinity = std::move(aff);

            // 下载 usage 到 CPU 做 update
            if (usg.IsTensor())
            {
                cv::Mat usg_cpu = alloc_->download(usg);
                int total_n = static_cast<int>(GA::shape(usg)[1]);
                int work_n = total_n - long_mem_size;
                work_mem_->update_bucket_usage(
                    bucket_id, usg_cpu.ptr<float>() + long_mem_size, B, work_n);
                if (count_long_term_usage_)
                {
                    long_mem_->update_bucket_usage(
                        bucket_id, usg_cpu.ptr<float>(), B, long_mem_size);
                }
            }
        }
        else
        {
            auto& memory_key = work_mem_->key().at(bucket_id);
            auto& shrink = work_mem_->shrinkage().at(bucket_id);
            auto similarity = ortcore::gpu_get_similarity(*alloc_, memory_key, shrink,
                                                           query_key, selection);
            if (use_long_term_)
            {
                auto [aff, usg] = ortcore::gpu_do_softmax(*alloc_, similarity, top_k_, true);
                affinity = std::move(aff);
                if (usg.IsTensor())
                {
                    cv::Mat usg_cpu = alloc_->download(usg);
                    int N = static_cast<int>(GA::shape(usg)[1]);
                    work_mem_->update_bucket_usage(bucket_id, usg_cpu.ptr<float>(), B, N);
                }
            }
            else
            {
                auto [aff, _] = ortcore::gpu_do_softmax(*alloc_, similarity, top_k_);
                affinity = std::move(aff);
            }
        }

        // 处理对象块
        std::vector<std::vector<ObjectId>> object_chunks;
        if (chunk_size_ < 1)
        {
            object_chunks.push_back(bucket_objs);
        }
        else
        {
            for (size_t i = 0; i < bucket_objs.size(); i += chunk_size_)
            {
                size_t end = std::min(i + static_cast<size_t>(chunk_size_), bucket_objs.size());
                object_chunks.emplace_back(bucket_objs.begin() + i, bucket_objs.begin() + end);
            }
        }

        for (auto& objects : object_chunks)
        {
            auto this_sensory = get_sensory_by_ids(objects);
            auto this_last_mask = get_mask_by_ids(last_mask, objects);
            auto this_msk_value = get_visual_values_by_ids(objects);

            // readout: this_msk_value [B, num_obj, CV, N] @ affinity [B, N, HW]
            auto visual_readout = ortcore::gpu_readout_4d(*alloc_, affinity, this_msk_value);

            // reshape: [B, num_obj, CV, HW] → [B, num_obj, CV, h, w]
            int num_obj_chunk = static_cast<int>(objects.size());
            int CV = static_cast<int>(GA::shape(visual_readout)[2]);
            auto vr_5d = Ort::Value::CreateTensor<float>(
                alloc_->memory_info(),
                GA::data_ptr(visual_readout),
                B * num_obj_chunk * CV * h * w,
                std::vector<int64_t>{B, num_obj_chunk, CV, h, w}.data(), 5);
            // clone 以持有独立内存
            auto visual_readout_5d = alloc_->clone(vr_5d);

            // 下采样 last_mask 到 stride-16 (h×w)（GPU）
            auto lm_shape = GA::shape(this_last_mask);
            int lm_H = static_cast<int>(lm_shape[2]);
            int lm_W = static_cast<int>(lm_shape[3]);
            Ort::Value last_mask_small{nullptr};
            if (lm_H != h || lm_W != w)
            {
                // this_last_mask: [B, num_objs, lm_H, lm_W]
                // 展平为 [B*num_objs, lm_H, lm_W]，用 resize_channels，再 reshape 回来
                int lm_num_objs = static_cast<int>(lm_shape[1]);
                int C = B * lm_num_objs;

                // 零拷贝 reshape [B, N, H, W] → [B*N, H, W]
                auto flat = Ort::Value::CreateTensor<float>(
                    alloc_->memory_info(),
                    const_cast<float*>(GA::data_ptr(this_last_mask)),
                    C * lm_H * lm_W,
                    std::vector<int64_t>{C, lm_H, lm_W}.data(), 3);
                auto flat_clone = alloc_->clone(flat);

                auto resized = alloc_->resize_channels(flat_clone, h, w, cv::INTER_AREA);

                // reshape 回 [B, num_objs, h, w]
                auto result_4d = Ort::Value::CreateTensor<float>(
                    alloc_->memory_info(),
                    GA::data_ptr(resized), C * h * w,
                    std::vector<int64_t>{B, lm_num_objs, h, w}.data(), 4);
                last_mask_small = alloc_->clone(result_4d);
            }
            else
            {
                last_mask_small = alloc_->clone(this_last_mask);
            }

            auto pixel_readout = network.pixel_fusion(pix_feat, visual_readout_5d,
                                                       this_sensory, last_mask_small);

            auto this_obj_mem_raw = get_object_mem_by_ids(objects);

            // unsqueeze(2): [B, N, Q, C] → [B, N, 1, Q, C]
            Ort::Value this_obj_mem{nullptr};
            if (this_obj_mem_raw.IsTensor() && GA::shape(this_obj_mem_raw).size() == 4)
            {
                auto om_shape = GA::shape(this_obj_mem_raw);
                int64_t total = GA::numel(om_shape);
                auto om_5d = Ort::Value::CreateTensor<float>(
                    alloc_->memory_info(),
                    GA::data_ptr(this_obj_mem_raw), total,
                    std::vector<int64_t>{om_shape[0], om_shape[1], 1, om_shape[2], om_shape[3]}
                        .data(),
                    5);
                this_obj_mem = alloc_->clone(om_5d);
            }
            else
            {
                this_obj_mem = this_obj_mem_raw.IsTensor()
                                   ? alloc_->clone(this_obj_mem_raw)
                                   : Ort::Value{nullptr};
            }

            Ort::Value readout_memory{nullptr};
            if (this_obj_mem.IsTensor())
            {
                readout_memory = network.readout_query(pixel_readout, this_obj_mem);
            }
            else
            {
                readout_memory = std::move(pixel_readout);
            }

            // 按对象拆分 readout_memory
            int num_obj = static_cast<int>(objects.size());
            auto rm_shape = GA::shape(readout_memory);
            if (rm_shape.size() >= 2 && rm_shape[1] == num_obj)
            {
                auto splits = ortcore::gpu_split(*alloc_, readout_memory, 1);
                for (int i = 0; i < num_obj; ++i)
                {
                    all_readout_mem.insert_or_assign(objects[i], std::move(splits[i]));
                }
            }
            else if (num_obj == 1)
            {
                all_readout_mem.insert_or_assign(objects[0], std::move(readout_memory));
            }
        }
    }

    return all_readout_mem;
}

// ── add_memory ──────────────────────────────────────────────────────

void MemoryManager::add_memory(Ort::Value& key, Ort::Value& shrinkage, Ort::Value& msk_value,
                                Ort::Value& obj_value, const std::vector<ObjectId>& objects,
                                Ort::Value& selection, bool as_permanent)
{
    engaged_ = true;

    int64_t ne = GA::shape(key).back();

    if (HW_ == 0 || config_stale_)
    {
        config_stale_ = false;
        HW_ = static_cast<int>(ne);
        max_work_tokens_ = max_mem_frames_ * HW_;
        if (use_long_term_) min_work_tokens_ = min_mem_frames_ * HW_;
    }

    // 处理 obj_value 累积: [B, num_objects, Q, C]
    if (obj_value.IsTensor() && GA::shape(obj_value).size() == 4)
    {
        auto ov_shape = GA::shape(obj_value);
        int B_ov = static_cast<int>(ov_shape[0]);
        int num_objects = static_cast<int>(objects.size());
        int Q_ov = static_cast<int>(ov_shape[2]);
        int C_ov = static_cast<int>(ov_shape[3]);
        int slice_ov = Q_ov * C_ov;

        for (int oi = 0; oi < num_objects; ++oi)
        {
            ObjectId obj_id = objects[oi];
            // 提取 [:, oi, :, :] → [B, Q, C]
            auto per_obj = alloc_->allocate({B_ov, Q_ov, C_ov});
            for (int b = 0; b < B_ov; ++b)
            {
                cuda::copy_d2d(
                    GA::data_ptr(per_obj) + b * slice_ov,
                    GA::data_ptr(obj_value) + b * num_objects * slice_ov + oi * slice_ov,
                    slice_ov);
            }

            if (obj_v_.count(obj_id) > 0)
            {
                // GPU 上原地累加
                int64_t total = B_ov * Q_ov * C_ov;
                cuda::add_inplace(GA::data_ptr(obj_v_[obj_id]),
                                  GA::data_ptr(per_obj), total);
            }
            else
            {
                obj_v_.insert_or_assign(obj_id, std::move(per_obj));
            }
        }
    }

    // 拆分 msk_value 为 per-object values
    std::unordered_map<ObjectId, Ort::Value> msk_values;
    auto mv_shape = GA::shape(msk_value);

    if (mv_shape.size() == 4)
    {
        // [B, num_obj, CV, N]
        int B = static_cast<int>(mv_shape[0]);
        int num_obj = static_cast<int>(mv_shape[1]);
        int CV = static_cast<int>(mv_shape[2]);
        int N = static_cast<int>(mv_shape[3]);
        int slice = CV * N;
        for (int oi = 0; oi < num_obj; ++oi)
        {
            auto val = alloc_->allocate({B, CV, N});
            for (int b = 0; b < B; ++b)
            {
                cuda::copy_d2d(
                    GA::data_ptr(val) + b * slice,
                    GA::data_ptr(msk_value) + b * num_obj * slice + oi * slice,
                    slice);
            }
            msk_values.insert_or_assign(objects[oi], std::move(val));
        }
    }
    else if (mv_shape.size() == 3 && objects.size() == 1)
    {
        msk_values.insert_or_assign(objects[0], alloc_->clone(msk_value));
    }

    std::string perm_str = as_permanent ? "all" : "first";
    work_mem_->add(alloc_->clone(key), msk_values, alloc_->clone(shrinkage),
                   selection.IsTensor() ? alloc_->clone(selection) : Ort::Value{nullptr},
                   -1, perm_str);

    // 内存管理
    for (auto& [bucket_id, _] : work_mem_->buckets())
    {
        if (use_long_term_)
        {
            if (work_mem_->non_perm_size(bucket_id) >= max_work_tokens_)
            {
                if (long_mem_->non_perm_size(bucket_id) >= (max_long_tokens_ - num_prototypes_))
                {
                    long_mem_->remove_obsolete_features(
                        bucket_id, max_long_tokens_ - num_prototypes_ - buffer_tokens_);
                }
                compress_features(bucket_id);
            }
        }
        else
        {
            work_mem_->remove_old_memory(bucket_id, max_work_tokens_);
        }
    }
}

// ── compress / consolidation ────────────────────────────────────────

void MemoryManager::compress_features(int bucket_id)
{
    auto sliced = work_mem_->get_all_sliced(bucket_id, 0, -min_work_tokens_);

    auto [proto_key, proto_values, proto_shrinkage] =
        consolidation(sliced.key, sliced.shrinkage, sliced.selection, sliced.values, sliced.usage);

    work_mem_->sieve_by_range(bucket_id, 0, -min_work_tokens_, min_work_tokens_);

    Ort::Value empty_sel{nullptr};
    long_mem_->add(std::move(proto_key), proto_values, std::move(proto_shrinkage),
                   std::move(empty_sel), bucket_id, "no");
}

MemoryManager::ConsolidationResult MemoryManager::consolidation(
    Ort::Value& candidate_key, Ort::Value& candidate_shrinkage,
    Ort::Value& candidate_selection,
    std::unordered_map<ObjectId, Ort::Value>& candidate_value, Ort::Value& usage)
{
    auto ck_shape = GA::shape(candidate_key);
    int B = static_cast<int>(ck_shape[0]);
    int CK = static_cast<int>(ck_shape[1]);
    int N = static_cast<int>(ck_shape[2]);

    // 下载 usage 到 CPU 做 top-k 排序
    cv::Mat usage_cpu;
    if (usage.IsTensor())
    {
        usage_cpu = alloc_->download(usage);
    }
    else
    {
        // 无 usage 信息，均匀选取
        int u_sizes[] = {B, N};
        usage_cpu = cv::Mat(2, u_sizes, CV_32FC1, cv::Scalar(1.0f));
    }

    int k = std::min(num_prototypes_, N);

    // 找 top-k 索引（CPU）
    std::vector<int> proto_indices(k);
    {
        const float* u = usage_cpu.ptr<float>();
        std::vector<std::pair<float, int>> scored(N);
        for (int i = 0; i < N; ++i) scored[i] = {u[i], i};
        std::partial_sort(scored.begin(), scored.begin() + k, scored.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });
        for (int i = 0; i < k; ++i) proto_indices[i] = scored[i].second;
    }

    // 下载 candidate_key 到 CPU，gather，上传 (小张量)
    cv::Mat ck_cpu = alloc_->download(candidate_key);
    int pk_sizes[] = {B, CK, k};
    cv::Mat pk_cpu(3, pk_sizes, CV_32FC1);
    for (int b = 0; b < B; ++b)
    {
        for (int ck = 0; ck < CK; ++ck)
        {
            for (int i = 0; i < k; ++i)
            {
                pk_cpu.ptr<float>()[b * CK * k + ck * k + i] =
                    ck_cpu.ptr<float>()[b * CK * N + ck * N + proto_indices[i]];
            }
        }
    }
    auto prototype_key = alloc_->upload(pk_cpu);

    Ort::Value prototype_selection{nullptr};
    if (candidate_selection.IsTensor())
    {
        cv::Mat cs_cpu = alloc_->download(candidate_selection);
        cv::Mat ps_cpu(3, pk_sizes, CV_32FC1);
        for (int b = 0; b < B; ++b)
        {
            for (int ck = 0; ck < CK; ++ck)
            {
                for (int i = 0; i < k; ++i)
                {
                    ps_cpu.ptr<float>()[b * CK * k + ck * k + i] =
                        cs_cpu.ptr<float>()[b * CK * N + ck * N + proto_indices[i]];
                }
            }
        }
        prototype_selection = alloc_->upload(ps_cpu);
    }

    // 计算 similarity 和 affinity (GPU)
    auto similarity = ortcore::gpu_get_similarity(
        *alloc_, candidate_key, candidate_shrinkage, prototype_key, prototype_selection);
    auto [affinity, _] = ortcore::gpu_do_softmax(*alloc_, similarity);

    // Readout values (GPU)
    std::unordered_map<ObjectId, Ort::Value> proto_values;
    for (auto& [obj_id, val] : candidate_value)
    {
        proto_values.insert_or_assign(obj_id, ortcore::gpu_readout(*alloc_, affinity, val));
    }

    // Readout shrinkage (GPU)
    auto proto_shrinkage = ortcore::gpu_readout(*alloc_, affinity, candidate_shrinkage);

    return {std::move(prototype_key), std::move(proto_values), std::move(proto_shrinkage)};
}

// ── sensory 管理 ────────────────────────────────────────────────────

void MemoryManager::initialize_sensory_if_needed(const Ort::Value& sample_key,
                                                  const std::vector<ObjectId>& ids)
{
    auto shape = GA::shape(sample_key);
    int B = static_cast<int>(shape[0]);
    int h = static_cast<int>(shape[2]);
    int w = static_cast<int>(shape[3]);

    for (auto obj : ids)
    {
        if (sensory_.count(obj) == 0)
        {
            sensory_.insert_or_assign(obj, alloc_->zeros({B, sensory_dim_, h, w}));
        }
    }
}

void MemoryManager::update_sensory(Ort::Value& sensory, const std::vector<ObjectId>& ids)
{
    auto shape = GA::shape(sensory);
    int B = static_cast<int>(shape[0]);
    int num_obj = static_cast<int>(ids.size());
    int C = static_cast<int>(shape[2]);
    int H = static_cast<int>(shape[3]);
    int W = static_cast<int>(shape[4]);
    int slice = C * H * W;

    for (int oi = 0; oi < num_obj; ++oi)
    {
        auto obj_sensory = alloc_->allocate({B, C, H, W});
        for (int b = 0; b < B; ++b)
        {
            cuda::copy_d2d(
                GA::data_ptr(obj_sensory) + b * slice,
                GA::data_ptr(sensory) + b * num_obj * slice + oi * slice, slice);
        }
        sensory_.insert_or_assign(ids[oi], std::move(obj_sensory));
    }
}

Ort::Value MemoryManager::get_sensory(const std::vector<ObjectId>& ids)
{
    return get_sensory_by_ids(ids);
}

void MemoryManager::purge_except(const std::vector<ObjectId>& keep_ids)
{
    std::set<ObjectId> keep_set(keep_ids.begin(), keep_ids.end());

    work_mem_->purge_except(keep_ids);
    if (use_long_term_ && long_mem_ && long_mem_->engaged())
        long_mem_->purge_except(keep_ids);

    for (auto it = sensory_.begin(); it != sensory_.end();)
    {
        if (keep_set.count(it->first) == 0) it = sensory_.erase(it);
        else ++it;
    }
    for (auto it = obj_v_.begin(); it != obj_v_.end();)
    {
        if (keep_set.count(it->first) == 0) it = obj_v_.erase(it);
        else ++it;
    }

    if (!work_mem_->engaged()) engaged_ = false;
}

void MemoryManager::clear_non_permanent_memory()
{
    work_mem_->clear_non_permanent_memory();
    if (use_long_term_ && long_mem_) long_mem_->clear_non_permanent_memory();
}

void MemoryManager::clear_sensory_memory()
{
    sensory_.clear();
}

}  // namespace core
}  // namespace cutie
