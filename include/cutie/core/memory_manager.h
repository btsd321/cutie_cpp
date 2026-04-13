#ifndef CUTIE_CORE_MEMORY_MANAGER_H
#define CUTIE_CORE_MEMORY_MANAGER_H

#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#include "cutie/core/kv_memory_store.h"
#include "cutie/core/object_manager.h"
#include "cutie/ort/core/gpu_memory.h"
#include "cutie/ort/core/gpu_tensor_ops.h"
#include "cutie/types.h"

namespace cutie
{
namespace core
{

struct CutieConfig;

/// GPU 版本的网络回调：所有参数和返回值均为 GPU Ort::Value。
struct NetworkCallbacks
{
    /// pixel_fusion(pix_feat, visual_readout, sensory, last_mask) → pixel_readout
    std::function<Ort::Value(Ort::Value&, Ort::Value&, Ort::Value&, Ort::Value&)> pixel_fusion;

    /// readout_query(pixel_readout, obj_memory) → memory_readout
    std::function<Ort::Value(Ort::Value&, Ort::Value&)> readout_query;
};

/// Manages three memory types: working, long-term, and sensory (GPU version).
/// All tensors stored as GPU Ort::Value.
class MemoryManager
{
public:
    MemoryManager(const CutieConfig& cfg, ObjectManager* object_manager,
                  ortcore::GpuMemoryAllocator* alloc);
    ~MemoryManager() = default;

    MemoryManager(const MemoryManager&) = delete;
    MemoryManager& operator=(const MemoryManager&) = delete;
    MemoryManager(MemoryManager&&) = default;
    MemoryManager& operator=(MemoryManager&&) = default;

    /// Read memory and produce per-object readout (all GPU).
    /// Returns: ObjectId → readout tensor [B, C, H, W] (GPU)
    std::unordered_map<ObjectId, Ort::Value> read(Ort::Value& pix_feat, Ort::Value& query_key,
                                                   Ort::Value& selection, Ort::Value& last_mask,
                                                   NetworkCallbacks& network);

    /// Add memory for the current frame (all GPU tensors).
    void add_memory(Ort::Value& key, Ort::Value& shrinkage, Ort::Value& msk_value,
                    Ort::Value& obj_value, const std::vector<ObjectId>& objects,
                    Ort::Value& selection, bool as_permanent = false);

    /// Initialize sensory state for new objects.
    void initialize_sensory_if_needed(const Ort::Value& sample_key,
                                      const std::vector<ObjectId>& ids);

    /// Get sensory for given objects, stacked. Returns: [B, num_objects, C, H, W] (GPU)
    Ort::Value get_sensory(const std::vector<ObjectId>& ids);

    /// Update sensory: sensory tensor [B, num_objects, C, H, W] (GPU)
    void update_sensory(Ort::Value& sensory, const std::vector<ObjectId>& ids);

    void purge_except(const std::vector<ObjectId>& keep_ids);
    void clear_non_permanent_memory();
    void clear_sensory_memory();
    void update_config(const CutieConfig& cfg);

    bool engaged() const { return engaged_; }
    KeyValueMemoryStore& work_mem() { return *work_mem_; }
    const KeyValueMemoryStore& work_mem() const { return *work_mem_; }

private:
    Ort::Value get_visual_values_by_ids(const std::vector<ObjectId>& obj_ids);
    Ort::Value get_mask_by_ids(const Ort::Value& mask, const std::vector<ObjectId>& obj_ids);
    Ort::Value get_sensory_by_ids(const std::vector<ObjectId>& obj_ids);
    Ort::Value get_object_mem_by_ids(const std::vector<ObjectId>& obj_ids);

    void compress_features(int bucket_id);

    struct ConsolidationResult
    {
        Ort::Value key{nullptr};
        std::unordered_map<ObjectId, Ort::Value> values;
        Ort::Value shrinkage{nullptr};
    };
    ConsolidationResult consolidation(Ort::Value& candidate_key, Ort::Value& candidate_shrinkage,
                                      Ort::Value& candidate_selection,
                                      std::unordered_map<ObjectId, Ort::Value>& candidate_value,
                                      Ort::Value& usage);

    ortcore::GpuMemoryAllocator* alloc_;
    ObjectManager* object_manager_;
    std::unique_ptr<KeyValueMemoryStore> work_mem_;
    std::unique_ptr<KeyValueMemoryStore> long_mem_;

    // Per-object sensory: ObjectId → [B, C, H, W] (GPU)
    std::unordered_map<ObjectId, Ort::Value> sensory_;
    // Per-object summaries: ObjectId → [B, Q, C] (GPU)
    std::unordered_map<ObjectId, Ort::Value> obj_v_;

    int sensory_dim_ = 0;
    int top_k_ = 30;
    int chunk_size_ = -1;
    bool use_long_term_ = false;
    bool count_long_term_usage_ = true;
    int max_mem_frames_ = 5;
    int min_mem_frames_ = 2;
    int num_prototypes_ = 128;
    int max_long_tokens_ = 10000;
    int buffer_tokens_ = 2000;

    int HW_ = 0;
    int max_work_tokens_ = 0;
    int min_work_tokens_ = 0;

    bool config_stale_ = true;
    bool engaged_ = false;
};

}  // namespace core
}  // namespace cutie

#endif  // CUTIE_CORE_MEMORY_MANAGER_H
