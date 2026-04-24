/**
 * @file kv_memory_store.h
 * @brief Bucket-based key-value memory store for GPU tensors.
 *
 * Manages working memory as a collection of buckets, each containing
 * key, value, shrinkage, and selection tensors. All data is GPU-resident
 * (Ort::Value float32). Supports FIFO removal, usage-based pruning, and
 * consolidation for long-term memory.
 */

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

/**
 * @class KeyValueMemoryStore
 * @brief Bucket-based key-value memory store (GPU version).
 *
 * Stores frame features in buckets for efficient memory management.
 * Each bucket contains key, value, shrinkage, and optional selection tensors.
 * All tensors are GPU-resident (Ort::Value, float32).
 *
 * 基于桶的 KV 内存存储，支持多对象、多帧的特征管理。每个桶独立维护
 * 一组 KV 对，支持 FIFO 移除、基于使用频率的剪枝、以及永久/非永久内存分离。
 *
 * Reference: cutie/inference/kv_memory_store.py
 */
class KeyValueMemoryStore
{
public:
    /**
     * @brief Construct memory store.
     *
     * @param alloc GPU memory allocator
     * @param save_selection Whether to save selection tensors (default false)
     * @param save_usage Whether to track usage statistics (default false)
     */
    explicit KeyValueMemoryStore(ortcore::GpuMemoryAllocator* alloc, bool save_selection = false,
                                 bool save_usage = false);
    ~KeyValueMemoryStore() = default;

    KeyValueMemoryStore(const KeyValueMemoryStore&) = delete;
    KeyValueMemoryStore& operator=(const KeyValueMemoryStore&) = delete;
    KeyValueMemoryStore(KeyValueMemoryStore&&) = default;
    KeyValueMemoryStore& operator=(KeyValueMemoryStore&&) = default;

    /**
     * @brief Add key-value pairs to memory.
     *
     * Concatenates new key-value pairs to existing bucket data.
     * 将新的 KV 对添加到指定桶，支持永久/非永久标记。
     *
     * @param key GPU tensor [B, CK, N_new]
     * @param values ObjectId → GPU tensor [B, CV, N_new] map
     * @param shrinkage GPU tensor [B, 1, N_new]
     * @param selection GPU tensor [B, CK, N_new] (nullable)
     * @param supposed_bucket_id Target bucket ID (auto-assign if -1)
     * @param as_permanent Mark as permanent memory ("yes"/"no", default "no")
     */
    void add(Ort::Value key, std::unordered_map<ObjectId, Ort::Value>& values,
             Ort::Value shrinkage, Ort::Value selection, int supposed_bucket_id = -1,
             const std::string& as_permanent = "no");

    /**
     * @brief Update usage statistics for a bucket.
     *
     * 更新桶的使用频率统计，用于后续的特征剪枝。
     *
     * @param bucket_id Target bucket
     * @param usage_data CPU float array [B*N]
     * @param B Batch size
     * @param N Number of features
     */
    void update_bucket_usage(int bucket_id, const float* usage_data, int B, int N);

    /**
     * @brief Remove oldest memory from bucket (FIFO).
     *
     * 从桶中移除最旧的特征，保持 FIFO 顺序。
     *
     * @param bucket_id Target bucket
     * @param max_len Maximum features to keep
     */
    void remove_old_memory(int bucket_id, int max_len);

    /**
     * @brief Remove features based on usage score.
     *
     * 根据使用频率移除不常用的特征，保持内存大小。
     *
     * @param bucket_id Target bucket
     * @param max_size Maximum features to keep
     */
    void remove_obsolete_features(int bucket_id, int max_size);

    /**
     * @brief Remove range of features from bucket.
     *
     * 移除指定范围的特征，保持最小大小。
     *
     * @param bucket_id Target bucket
     * @param start Start index
     * @param end End index (exclusive)
     * @param min_size Minimum features to keep
     */
    void sieve_by_range(int bucket_id, int start, int end, int min_size);

    /**
     * @struct SlicedData
     * @brief Sliced data for consolidation.
     *
     * Contains key, value, shrinkage, selection, and usage tensors
     * extracted from a bucket range.
     */
    struct SlicedData
    {
        Ort::Value key{nullptr};
        Ort::Value shrinkage{nullptr};
        Ort::Value selection{nullptr};
        std::unordered_map<ObjectId, Ort::Value> values;
        Ort::Value usage{nullptr};  ///< CPU tensor for sorting
    };

    /**
     * @brief Get sliced data for consolidation.
     *
     * 提取指定范围的数据用于长期内存的压缩和整合。
     *
     * @param bucket_id Target bucket
     * @param start Start index
     * @param end End index (exclusive)
     * @return SlicedData containing extracted tensors
     */
    SlicedData get_all_sliced(int bucket_id, int start, int end);

    /**
     * @brief Remove all data for objects not in keep_ids.
     *
     * 清理不在保留列表中的对象的所有数据。
     *
     * @param keep_ids List of ObjectIds to keep
     */
    void purge_except(const std::vector<ObjectId>& keep_ids);

    /**
     * @brief Clear non-permanent memory from all buckets.
     *
     * 清除所有非永久内存，保留永久标记的特征。
     */
    void clear_non_permanent_memory();

    /**
     * @brief Check if bucket has data.
     * @param bucket_id Bucket ID (-1 = any bucket)
     * @return True if bucket contains data
     */
    bool engaged(int bucket_id = -1) const;

    /**
     * @brief Get total features in bucket.
     * @param bucket_id Target bucket
     * @return Feature count
     */
    int size(int bucket_id) const;

    /**
     * @brief Get permanent features in bucket.
     * @param bucket_id Target bucket
     * @return Permanent feature count
     */
    int perm_size(int bucket_id) const;

    /**
     * @brief Get non-permanent features in bucket.
     * @param bucket_id Target bucket
     * @return Non-permanent feature count
     */
    int non_perm_size(int bucket_id) const;

    /**
     * @brief Get number of tracked objects.
     * @return Object count
     */
    int num_objects() const;

    // Direct access to internal data (const references)

    /**
     * @brief Get bucket → ObjectId mapping.
     * @return Reference to buckets map
     */
    const std::map<int, std::vector<ObjectId>>& buckets() const { return buckets_; }

    /**
     * @brief Get key tensors.
     * @return Reference to bucket_id → key map
     */
    const std::map<int, Ort::Value>& key() const { return k_; }

    /**
     * @brief Get value tensors.
     * @return Reference to ObjectId → value map
     */
    const std::unordered_map<ObjectId, Ort::Value>& value() const { return v_; }

    /**
     * @brief Get shrinkage tensors.
     * @return Reference to bucket_id → shrinkage map
     */
    const std::map<int, Ort::Value>& shrinkage() const { return s_; }

    /**
     * @brief Get selection tensors.
     * @return Reference to bucket_id → selection map
     */
    const std::map<int, Ort::Value>& selection() const { return e_; }

    /**
     * @brief Check if object is tracked.
     * @param obj_id ObjectId to check
     * @return True if object has stored values
     */
    bool contains(ObjectId obj_id) const { return v_.count(obj_id) > 0; }

private:
    /**
     * @brief Concatenate tensors along last dimension.
     *
     * GPU 端沿最后一维拼接两个张量。
     *
     * @param existing Existing tensor
     * @param new_val New tensor to append
     * @param prepend If true, prepend instead of append
     * @return Concatenated tensor
     */
    Ort::Value cat_last(Ort::Value& existing, Ort::Value& new_val, bool prepend = false);

    /**
     * @brief Slice tensor along last dimension.
     *
     * GPU 端沿最后一维切片张量。
     *
     * @param tensor Input tensor
     * @param start Start index
     * @param end End index (exclusive)
     * @return Sliced tensor
     */
    Ort::Value slice_last(const Ort::Value& tensor, int64_t start, int64_t end);

    /**
     * @brief Get size of last dimension.
     *
     * 获取张量最后一维的大小。
     *
     * @param t Input tensor
     * @return Last dimension size
     */
    int64_t last_dim(const Ort::Value& t) const;

    ortcore::GpuMemoryAllocator* alloc_;
    bool save_selection_;
    bool save_usage_;

    int global_bucket_id_ = 0;
    std::map<int, std::vector<ObjectId>> buckets_;  ///< bucket_id → [ObjectIds in bucket]
    std::map<int, Ort::Value> k_;                   ///< bucket_id → [B, CK, N]
    std::unordered_map<ObjectId, Ort::Value> v_;    ///< obj_id → [B, CV, N]
    std::map<int, Ort::Value> s_;                   ///< bucket_id → [B, 1, N]
    std::map<int, Ort::Value> e_;                   ///< bucket_id → [B, CK, N]

    // Usage tracking stays on CPU (small tensors, need sorting)
    std::map<int, std::vector<float>> use_cnt_;     ///< bucket_id → [B*N]
    std::map<int, std::vector<float>> life_cnt_;    ///< bucket_id → [B*N]
    std::map<int, int> perm_end_pt_;                ///< bucket_id → permanent/non-permanent boundary
};

}  // namespace core
}  // namespace cutie

#endif  // CUTIE_CORE_KV_MEMORY_STORE_H
