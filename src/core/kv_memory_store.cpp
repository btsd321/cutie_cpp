#include <algorithm>
#include <cassert>
#include <numeric>
#include <set>
#include <stdexcept>

#include "cutie/core/kv_memory_store.h"
#include "cutie/ort/core/cuda_kernels.h"

namespace cutie
{
namespace core
{

using GA = ortcore::GpuMemoryAllocator;

KeyValueMemoryStore::KeyValueMemoryStore(ortcore::GpuMemoryAllocator* alloc, bool save_selection,
                                         bool save_usage)
    : alloc_(alloc), save_selection_(save_selection), save_usage_(save_usage)
{
}

// ── 内部工具 ────────────────────────────────────────────────────────

int64_t KeyValueMemoryStore::last_dim(const Ort::Value& t) const
{
    auto shape = GA::shape(t);
    return shape.back();
}

Ort::Value KeyValueMemoryStore::cat_last(Ort::Value& existing, Ort::Value& new_val, bool prepend)
{
    if (!existing.IsTensor())
        return alloc_->clone(new_val);
    if (!new_val.IsTensor())
        return alloc_->clone(existing);

    auto se = GA::shape(existing);
    int dim = static_cast<int>(se.size()) - 1;

    if (prepend)
        return alloc_->concat(new_val, existing, dim);
    else
        return alloc_->concat(existing, new_val, dim);
}

Ort::Value KeyValueMemoryStore::slice_last(const Ort::Value& tensor, int64_t start, int64_t end)
{
    if (!tensor.IsTensor())
        return Ort::Value{nullptr};

    auto shape = GA::shape(tensor);
    int64_t total = shape.back();

    if (start < 0) start = 0;
    if (end <= 0) end = total + end;
    if (end > total) end = total;
    if (start >= end) return Ort::Value{nullptr};

    return alloc_->slice_last(tensor, start, end - start);
}

// ── add ─────────────────────────────────────────────────────────────

void KeyValueMemoryStore::add(Ort::Value key,
                              std::unordered_map<ObjectId, Ort::Value>& values,
                              Ort::Value shrinkage, Ort::Value selection,
                              int supposed_bucket_id, const std::string& as_permanent)
{
    int64_t ne = last_dim(key);
    int B = static_cast<int>(GA::shape(key)[0]);

    if (supposed_bucket_id >= 0)
    {
        bool bucket_exist = buckets_.count(supposed_bucket_id) > 0;
        bool prepend = (as_permanent == "all");

        for (auto& [obj, value] : values)
        {
            if (bucket_exist && v_.count(obj) > 0)
            {
                v_[obj] = cat_last(v_[obj], value, prepend);
            }
            else
            {
                v_.insert_or_assign(obj, alloc_->clone(value));
            }
        }

        std::vector<ObjectId> obj_ids;
        for (auto& [obj, _] : values) obj_ids.push_back(obj);
        buckets_[supposed_bucket_id] = obj_ids;

        if (k_.count(supposed_bucket_id) > 0)
        {
            k_[supposed_bucket_id] = cat_last(k_[supposed_bucket_id], key, prepend);
            s_[supposed_bucket_id] = cat_last(s_[supposed_bucket_id], shrinkage, prepend);
        }
        else
        {
            k_.insert_or_assign(supposed_bucket_id, alloc_->clone(key));
            s_.insert_or_assign(supposed_bucket_id, alloc_->clone(shrinkage));
        }

        if (as_permanent == "all")
            perm_end_pt_[supposed_bucket_id] += static_cast<int>(ne);
        else if (as_permanent == "first" && perm_end_pt_[supposed_bucket_id] == 0)
            perm_end_pt_[supposed_bucket_id] = static_cast<int>(ne);

        if (!prepend)
        {
            if (save_selection_ && selection.IsTensor())
            {
                if (e_.count(supposed_bucket_id) > 0)
                    e_[supposed_bucket_id] = cat_last(e_[supposed_bucket_id], selection, false);
                else
                    e_.insert_or_assign(supposed_bucket_id, alloc_->clone(selection));
            }
            if (save_usage_ && as_permanent != "all")
            {
                auto& uc = use_cnt_[supposed_bucket_id];
                auto& lc = life_cnt_[supposed_bucket_id];
                uc.resize(uc.size() + B * ne, 0.0f);
                lc.resize(lc.size() + B * ne, 1e-7f);
            }
        }
        return;
    }

    // Auto bucket assignment
    int new_bucket_id = -1;
    std::set<int> enabled_buckets;

    for (auto& [obj, value] : values)
    {
        if (v_.count(obj) > 0)
        {
            bool prepend = (as_permanent == "all");
            v_[obj] = cat_last(v_[obj], value, prepend);
            for (auto& [bid, obj_ids] : buckets_)
            {
                for (auto oid : obj_ids)
                {
                    if (oid == obj) { enabled_buckets.insert(bid); break; }
                }
            }
        }
        else
        {
            v_.insert_or_assign(obj, alloc_->clone(value));
            if (new_bucket_id < 0)
            {
                new_bucket_id = global_bucket_id_++;
                buckets_[new_bucket_id] = {};
            }
            buckets_[new_bucket_id].push_back(obj);
            enabled_buckets.insert(new_bucket_id);
        }
    }

    for (int bucket_id : enabled_buckets)
    {
        bool prepend = false;
        if (as_permanent == "all")
        {
            perm_end_pt_[bucket_id] += static_cast<int>(ne);
            prepend = true;
        }
        else if (as_permanent == "first" && perm_end_pt_[bucket_id] == 0)
        {
            perm_end_pt_[bucket_id] = static_cast<int>(ne);
            prepend = true;
        }

        if (k_.count(bucket_id) > 0)
        {
            k_[bucket_id] = cat_last(k_[bucket_id], key, prepend);
            s_[bucket_id] = cat_last(s_[bucket_id], shrinkage, prepend);
        }
        else
        {
            k_.insert_or_assign(bucket_id, alloc_->clone(key));
            s_.insert_or_assign(bucket_id, alloc_->clone(shrinkage));
        }

        if (!prepend)
        {
            if (save_selection_ && selection.IsTensor())
            {
                if (e_.count(bucket_id) > 0)
                    e_[bucket_id] = cat_last(e_[bucket_id], selection, false);
                else
                    e_.insert_or_assign(bucket_id, alloc_->clone(selection));
            }
            if (save_usage_ && as_permanent != "all")
            {
                auto& uc = use_cnt_[bucket_id];
                auto& lc = life_cnt_[bucket_id];
                uc.resize(uc.size() + B * ne, 0.0f);
                lc.resize(lc.size() + B * ne, 1e-7f);
            }
        }
    }
}

// ── update_bucket_usage ─────────────────────────────────────────────

void KeyValueMemoryStore::update_bucket_usage(int bucket_id, const float* usage_data, int B, int N)
{
    if (!save_usage_) return;
    auto it = use_cnt_.find(bucket_id);
    if (it == use_cnt_.end()) return;

    auto& cnt = it->second;
    auto& life = life_cnt_[bucket_id];
    int total = B * N;
    if (static_cast<int>(cnt.size()) < total) return;

    for (int i = 0; i < total; ++i)
    {
        cnt[i] += usage_data[i];
        life[i] += 1.0f;
    }
}

// ── sieve_by_range ──────────────────────────────────────────────────

void KeyValueMemoryStore::sieve_by_range(int bucket_id, int start, int end, int min_size)
{
    if (buckets_.count(bucket_id) == 0) return;

    int p_size = perm_end_pt_[bucket_id];
    int64_t total_n = last_dim(k_[bucket_id]);
    int non_perm = static_cast<int>(total_n) - p_size;
    if (non_perm <= min_size) return;

    int abs_start = start + p_size;
    int abs_end = (end < 0) ? static_cast<int>(total_n) + end : (end == 0 ? static_cast<int>(total_n) : end);

    // 保留 [:abs_start] 和 [abs_end:]
    auto k_keep1 = slice_last(k_[bucket_id], 0, abs_start);
    auto k_keep2 = slice_last(k_[bucket_id], abs_end, total_n);
    k_[bucket_id] = cat_last(k_keep1, k_keep2);

    auto s_keep1 = slice_last(s_[bucket_id], 0, abs_start);
    auto s_keep2 = slice_last(s_[bucket_id], abs_end, total_n);
    s_[bucket_id] = cat_last(s_keep1, s_keep2);

    if (save_selection_ && e_.count(bucket_id) > 0)
    {
        int e_start = start;
        int e_end = (end <= 0) ? non_perm + end + 1 : end - p_size;
        auto e_keep1 = slice_last(e_[bucket_id], 0, e_start);
        auto e_keep2 = slice_last(e_[bucket_id], e_end, non_perm);
        e_[bucket_id] = cat_last(e_keep1, e_keep2);
    }

    if (save_usage_ && use_cnt_.count(bucket_id) > 0)
    {
        int u_start = start;
        int u_end = (end <= 0) ? non_perm + end + 1 : end - p_size;
        int u_total = static_cast<int>(use_cnt_[bucket_id].size());
        // CPU vector erase
        auto& uc = use_cnt_[bucket_id];
        auto& lc = life_cnt_[bucket_id];
        if (u_end > u_start && u_end <= u_total)
        {
            uc.erase(uc.begin() + u_start, uc.begin() + u_end);
            lc.erase(lc.begin() + u_start, lc.begin() + u_end);
        }
    }

    for (auto obj_id : buckets_[bucket_id])
    {
        if (v_.count(obj_id) == 0) continue;
        int64_t v_total = last_dim(v_[obj_id]);
        auto v_keep1 = slice_last(v_[obj_id], 0, abs_start);
        auto v_keep2 = slice_last(v_[obj_id], abs_end, v_total);
        v_[obj_id] = cat_last(v_keep1, v_keep2);
    }
}

void KeyValueMemoryStore::remove_old_memory(int bucket_id, int max_len)
{
    sieve_by_range(bucket_id, 0, -max_len, max_len);
}

void KeyValueMemoryStore::remove_obsolete_features(int bucket_id, int max_size)
{
    if (!save_usage_ || use_cnt_.count(bucket_id) == 0) return;

    auto& uc = use_cnt_[bucket_id];
    auto& lc = life_cnt_[bucket_id];
    int N = static_cast<int>(uc.size());
    if (N <= max_size) return;

    // 计算 usage = use / life
    std::vector<std::pair<float, int>> scored(N);
    for (int i = 0; i < N; ++i)
    {
        scored[i] = {uc[i] / lc[i], i};
    }
    std::partial_sort(scored.begin(), scored.begin() + max_size, scored.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });

    std::vector<int> keep_indices(max_size);
    for (int i = 0; i < max_size; ++i) keep_indices[i] = scored[i].second;
    std::sort(keep_indices.begin(), keep_indices.end());

    // Gather on GPU: 下载 → CPU gather index → 上传
    // 对于 key, shrinkage, selection, values 使用 GPU slice 和 concat
    // 简化实现：下载到 CPU，gather，重新上传
    auto gather_gpu = [&](Ort::Value& tensor)
    {
        if (!tensor.IsTensor()) return;
        auto shape = GA::shape(tensor);
        int ndim = static_cast<int>(shape.size());
        int64_t last = shape[ndim - 1];
        int64_t rows = GA::numel(shape) / last;

        // 下载到 CPU
        cv::Mat cpu = alloc_->download(tensor);

        // gather
        std::vector<int64_t> new_shape = shape;
        new_shape[ndim - 1] = max_size;
        int64_t new_total = GA::numel(new_shape);
        std::vector<int> cv_sizes(new_shape.begin(), new_shape.end());
        cv::Mat gathered(ndim, cv_sizes.data(), CV_32FC1);

        const float* src = cpu.ptr<float>();
        float* dst = gathered.ptr<float>();
        for (int64_t r = 0; r < rows; ++r)
        {
            for (int ki = 0; ki < max_size; ++ki)
            {
                dst[r * max_size + ki] = src[r * last + keep_indices[ki]];
            }
        }

        // 重新上传
        tensor = alloc_->upload(gathered);
    };

    gather_gpu(k_[bucket_id]);
    gather_gpu(s_[bucket_id]);
    if (save_selection_ && e_.count(bucket_id) > 0) gather_gpu(e_[bucket_id]);
    for (auto obj_id : buckets_[bucket_id])
    {
        if (v_.count(obj_id) > 0) gather_gpu(v_[obj_id]);
    }

    // Gather usage (CPU)
    std::vector<float> new_uc(max_size), new_lc(max_size);
    for (int i = 0; i < max_size; ++i)
    {
        new_uc[i] = uc[keep_indices[i]];
        new_lc[i] = lc[keep_indices[i]];
    }
    uc = std::move(new_uc);
    lc = std::move(new_lc);
}

KeyValueMemoryStore::SlicedData KeyValueMemoryStore::get_all_sliced(int bucket_id, int start,
                                                                    int end)
{
    SlicedData data;

    int p_size = 0;
    auto it = perm_end_pt_.find(bucket_id);
    if (it != perm_end_pt_.end()) p_size = it->second;

    int abs_start = start + p_size;
    int64_t total = last_dim(k_[bucket_id]);
    int abs_end = (end <= 0) ? static_cast<int>(total) + end + 1 : end;

    data.key = slice_last(k_[bucket_id], abs_start, abs_end);
    data.shrinkage = slice_last(s_[bucket_id], abs_start, abs_end);

    if (save_selection_ && e_.count(bucket_id) > 0)
    {
        int64_t e_n = last_dim(e_[bucket_id]);
        int e_start = start;
        int e_end = (end <= 0) ? static_cast<int>(e_n) + end + 1 : end - p_size;
        data.selection = slice_last(e_[bucket_id], e_start, e_end);
    }

    if (buckets_.count(bucket_id) > 0)
    {
        for (auto obj_id : buckets_[bucket_id])
        {
            if (v_.count(obj_id) > 0)
            {
                data.values[obj_id] = slice_last(v_[obj_id], abs_start, abs_end);
            }
        }
    }

    // Usage: compute on CPU and return as CPU Ort::Value
    if (save_usage_ && use_cnt_.count(bucket_id) > 0)
    {
        auto& uc = use_cnt_[bucket_id];
        auto& lc = life_cnt_[bucket_id];
        int u_start = start;
        int u_n = static_cast<int>(uc.size());
        int u_end = (end <= 0) ? u_n + end + 1 : end - p_size;
        int slice_len = u_end - u_start;
        if (slice_len > 0)
        {
            // 返回 CPU Ort::Value (usage 只用于 CPU 排序)
            std::vector<float> usage_data(slice_len);
            for (int i = 0; i < slice_len; ++i)
            {
                usage_data[i] = uc[u_start + i] / lc[u_start + i];
            }
            // 使用 CPU MemoryInfo 创建
            auto cpu_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            std::vector<int64_t> u_shape = {1, static_cast<int64_t>(slice_len)};
            data.usage = Ort::Value::CreateTensor<float>(
                cpu_info, usage_data.data(), usage_data.size(), u_shape.data(), u_shape.size());
            // 由于 usage_data 会销毁，需要拷贝
            Ort::AllocatorWithDefaultOptions cpu_alloc;
            auto owned = Ort::Value::CreateTensor<float>(cpu_alloc, u_shape.data(), u_shape.size());
            std::memcpy(owned.GetTensorMutableData<float>(), usage_data.data(),
                        slice_len * sizeof(float));
            data.usage = std::move(owned);
        }
    }

    return data;
}

void KeyValueMemoryStore::purge_except(const std::vector<ObjectId>& keep_ids)
{
    std::set<ObjectId> keep_set(keep_ids.begin(), keep_ids.end());

    for (auto it = v_.begin(); it != v_.end();)
    {
        if (keep_set.count(it->first) == 0) it = v_.erase(it);
        else ++it;
    }

    std::vector<int> to_remove;
    for (auto& [bid, obj_ids] : buckets_)
    {
        std::vector<ObjectId> filtered;
        for (auto id : obj_ids)
            if (keep_set.count(id) > 0) filtered.push_back(id);
        obj_ids = filtered;
        if (obj_ids.empty()) to_remove.push_back(bid);
    }

    for (int bid : to_remove)
    {
        buckets_.erase(bid);
        k_.erase(bid);
        s_.erase(bid);
        e_.erase(bid);
        use_cnt_.erase(bid);
        life_cnt_.erase(bid);
        perm_end_pt_.erase(bid);
    }
}

void KeyValueMemoryStore::clear_non_permanent_memory()
{
    for (auto& [bid, _] : buckets_)
    {
        sieve_by_range(bid, 0, 0, 0);
    }
}

bool KeyValueMemoryStore::engaged(int bucket_id) const
{
    if (bucket_id < 0) return !buckets_.empty();
    return buckets_.count(bucket_id) > 0;
}

int KeyValueMemoryStore::size(int bucket_id) const
{
    auto it = k_.find(bucket_id);
    if (it == k_.end()) return 0;
    return static_cast<int>(last_dim(it->second));
}

int KeyValueMemoryStore::perm_size(int bucket_id) const
{
    auto it = perm_end_pt_.find(bucket_id);
    return (it != perm_end_pt_.end()) ? it->second : 0;
}

int KeyValueMemoryStore::non_perm_size(int bucket_id) const
{
    return size(bucket_id) - perm_size(bucket_id);
}

int KeyValueMemoryStore::num_objects() const
{
    return static_cast<int>(v_.size());
}

}  // namespace core
}  // namespace cutie
