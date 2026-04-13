#ifndef CUTIE_CORE_KV_MEMORY_STORE_H
#define CUTIE_CORE_KV_MEMORY_STORE_H

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "cutie/ort/core/gpu_memory.h"
#include "cutie/types.h"

namespace cutie
{
namespace core
{

/// Bucket-based key-value memory store (GPU version).
/// All tensors stored as GPU Ort::Value (float32).
/// Reference: cutie/inference/kv_memory_store.py
class KeyValueMemoryStore
{
public:
    explicit KeyValueMemoryStore(ortcore::GpuMemoryAllocator* alloc, bool save_selection = false,
                                 bool save_usage = false);
    ~KeyValueMemoryStore() = default;

    KeyValueMemoryStore(const KeyValueMemoryStore&) = delete;
    KeyValueMemoryStore& operator=(const KeyValueMemoryStore&) = delete;
    KeyValueMemoryStore(KeyValueMemoryStore&&) = default;
    KeyValueMemoryStore& operator=(KeyValueMemoryStore&&) = default;

    /// Add key-value pairs (all GPU tensors).
    /// key: [B, CK, N_new], shrinkage: [B, 1, N_new]
    /// values: {ObjectId → [B, CV, N_new]}
    /// selection: [B, CK, N_new] (nullable)
    void add(Ort::Value key, std::unordered_map<ObjectId, Ort::Value>& values,
             Ort::Value shrinkage, Ort::Value selection, int supposed_bucket_id = -1,
             const std::string& as_permanent = "no");

    /// Update usage for a bucket (CPU cv::Mat for simplicity).
    void update_bucket_usage(int bucket_id, const float* usage_data, int B, int N);

    /// Remove oldest memory from a bucket (FIFO).
    void remove_old_memory(int bucket_id, int max_len);

    /// Remove features based on usage score (downloads usage to CPU for sorting).
    void remove_obsolete_features(int bucket_id, int max_size);

    /// Remove range of features.
    void sieve_by_range(int bucket_id, int start, int end, int min_size);

    /// Get all sliced data for consolidation.
    struct SlicedData
    {
        Ort::Value key{nullptr};
        Ort::Value shrinkage{nullptr};
        Ort::Value selection{nullptr};
        std::unordered_map<ObjectId, Ort::Value> values;
        Ort::Value usage{nullptr};  // CPU tensor for sorting
    };
    SlicedData get_all_sliced(int bucket_id, int start, int end);

    /// Remove all data for objects not in keep_ids.
    void purge_except(const std::vector<ObjectId>& keep_ids);

    /// Clear non-permanent memory from all buckets.
    void clear_non_permanent_memory();

    bool engaged(int bucket_id = -1) const;
    int size(int bucket_id) const;
    int perm_size(int bucket_id) const;
    int non_perm_size(int bucket_id) const;
    int num_objects() const;

    // Direct access (const references to GPU tensors)
    const std::map<int, std::vector<ObjectId>>& buckets() const { return buckets_; }
    const std::map<int, Ort::Value>& key() const { return k_; }
    const std::unordered_map<ObjectId, Ort::Value>& value() const { return v_; }
    const std::map<int, Ort::Value>& shrinkage() const { return s_; }
    const std::map<int, Ort::Value>& selection() const { return e_; }

    bool contains(ObjectId obj_id) const { return v_.count(obj_id) > 0; }

private:
    /// GPU concat along last dim.
    Ort::Value cat_last(Ort::Value& existing, Ort::Value& new_val, bool prepend = false);

    /// GPU slice along last dim.
    Ort::Value slice_last(const Ort::Value& tensor, int64_t start, int64_t end);

    /// 获取张量最后一维的大小。
    int64_t last_dim(const Ort::Value& t) const;

    ortcore::GpuMemoryAllocator* alloc_;
    bool save_selection_;
    bool save_usage_;

    int global_bucket_id_ = 0;
    std::map<int, std::vector<ObjectId>> buckets_;
    std::map<int, Ort::Value> k_;                       // bucket_id → [B, CK, N]
    std::unordered_map<ObjectId, Ort::Value> v_;        // obj_id → [B, CV, N]
    std::map<int, Ort::Value> s_;                       // bucket_id → [B, 1, N]
    std::map<int, Ort::Value> e_;                       // bucket_id → [B, CK, N]

    // Usage tracking stays on CPU (small tensors, need sorting)
    std::map<int, std::vector<float>> use_cnt_;         // bucket_id → [B*N]
    std::map<int, std::vector<float>> life_cnt_;        // bucket_id → [B*N]
    std::map<int, int> perm_end_pt_;
};

}  // namespace core
}  // namespace cutie

#endif  // CUTIE_CORE_KV_MEMORY_STORE_H
