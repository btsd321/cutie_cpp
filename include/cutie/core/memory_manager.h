/**
 * @file memory_manager.h
 * @brief Three-tier memory management (working, long-term, sensory).
 *
 * Orchestrates memory read/write operations across working memory (recent frames),
 * long-term memory (compressed prototypes), and sensory memory (per-object GRU state).
 * All tensors are GPU-resident (Ort::Value).
 */

#ifndef CUTIE_CORE_MEMORY_MANAGER_H
#define CUTIE_CORE_MEMORY_MANAGER_H

#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#include "cutie/core/kv_memory_store.h"
#include "cutie/core/object_manager.h"
#include "cutie/common/gpu_memory.h"
#include "cutie/common/gpu_tensor_ops.h"
#include "cutie/types.h"

namespace cutie
{
namespace core
{

struct CutieConfig;

/**
 * @struct NetworkCallbacks
 * @brief GPU network callbacks for memory operations.
 *
 * Callbacks for pixel fusion and readout query operations.
 * All parameters and return values are GPU Ort::Value tensors.
 * 网络回调函数，用于内存读取过程中的特征融合和查询。
 */
struct NetworkCallbacks
{
    /**
     * @brief Pixel fusion callback.
     *
     * Fuses pixel features with visual and sensory readouts.
     * 融合像素特征、视觉记忆和感知记忆。
     *
     * @param pix_feat Pixel features [B, C, H, W]
     * @param visual_readout Visual memory readout [B, C, H, W]
     * @param sensory Sensory memory [B, C, H, W]
     * @param last_mask Last frame mask [B, num_obj, H, W]
     * @return Fused pixel readout [B, C, H, W]
     */
    std::function<Ort::Value(Ort::Value&, Ort::Value&, Ort::Value&, Ort::Value&)> pixel_fusion;

    /**
     * @brief Readout query callback.
     *
     * Queries memory using pixel readout to produce per-object memory readout.
     * 使用像素查询从记忆中读取每个对象的特征。
     *
     * @param pixel_readout Pixel readout [B, C, H, W]
     * @param obj_memory Object memory [B, num_obj, Q, C]
     * @return Memory readout [B, num_obj, C, H, W]
     */
    std::function<Ort::Value(Ort::Value&, Ort::Value&)> readout_query;
};

/**
 * @class MemoryManager
 * @brief Manages three memory types: working, long-term, and sensory (GPU version).
 *
 * Coordinates memory operations across three tiers:
 * - **Working memory**: Recent N frames (FIFO buffer, fast access)
 * - **Long-term memory**: Compressed prototypes (optional, for long videos)
 * - **Sensory memory**: Per-object GRU hidden state (updated each frame)
 *
 * 三层内存管理系统：工作内存存储最近帧的特征，长期内存存储压缩的原型，
 * 感知内存存储每个对象的 GRU 隐藏状态。支持内存读取、写入、初始化和清理。
 *
 * All tensors are GPU-resident (Ort::Value).
 */
class MemoryManager
{
public:
    /**
     * @brief Construct memory manager.
     *
     * @param cfg Configuration
     * @param object_manager Object ID manager
     * @param alloc GPU memory allocator
     */
    MemoryManager(const CutieConfig& cfg, ObjectManager* object_manager,
                  ortcore::GpuMemoryAllocator* alloc);
    ~MemoryManager() = default;

    MemoryManager(const MemoryManager&) = delete;
    MemoryManager& operator=(const MemoryManager&) = delete;
    MemoryManager(MemoryManager&&) = default;
    MemoryManager& operator=(MemoryManager&&) = default;

    /**
     * @brief Read memory and produce per-object readout.
     *
     * Queries working and long-term memory using pixel features and query keys,
     * producing per-object readout tensors for the transformer.
     * 从工作内存和长期内存中读取特征，返回每个对象的记忆读出。
     *
     * @param pix_feat Pixel features [B, C, H, W]
     * @param query_key Query key for memory matching [B, CK, H, W]
     * @param selection Selection mask [B, CK, H, W]
     * @param last_mask Last frame mask [B, num_obj, H, W]
     * @param network Network callbacks for fusion
     * @return ObjectId → readout tensor [B, C, H, W] map
     */
    std::unordered_map<ObjectId, Ort::Value> read(Ort::Value& pix_feat, Ort::Value& query_key,
                                                   Ort::Value& selection, Ort::Value& last_mask,
                                                   NetworkCallbacks& network);

    /**
     * @brief Add memory for current frame.
     *
     * Stores key, value, and shrinkage tensors in working memory.
     * 将当前帧的特征添加到工作内存。
     *
     * @param key Key tensor [B, CK, H, W]
     * @param shrinkage Shrinkage tensor [B, 1, H, W]
     * @param msk_value Mask value tensor [B, num_obj, H, W]
     * @param obj_value Object value tensor [B, num_obj, C, H, W]
     * @param objects Active object IDs
     * @param selection Selection tensor [B, CK, H, W]
     * @param as_permanent Mark as permanent memory (default false)
     */
    void add_memory(Ort::Value& key, Ort::Value& shrinkage, Ort::Value& msk_value,
                    Ort::Value& obj_value, const std::vector<ObjectId>& objects,
                    Ort::Value& selection, bool as_permanent = false);

    /**
     * @brief Initialize sensory state for new objects.
     *
     * Creates initial GRU hidden state for newly registered objects.
     * 为新对象初始化感知记忆（GRU 隐藏状态）。
     *
     * @param sample_key Sample key tensor (for shape inference)
     * @param ids Object IDs to initialize
     */
    void initialize_sensory_if_needed(const Ort::Value& sample_key,
                                      const std::vector<ObjectId>& ids);

    /**
     * @brief Get sensory memory for objects (stacked).
     *
     * 获取指定对象的感知记忆，按 tmp_id 顺序堆叠。
     *
     * @param ids Object IDs to retrieve
     * @return Stacked sensory tensor [B, num_objects, C, H, W]
     */
    Ort::Value get_sensory(const std::vector<ObjectId>& ids);

    /**
     * @brief Update sensory memory for objects.
     *
     * 更新指定对象的感知记忆。
     *
     * @param sensory Updated sensory tensor [B, num_objects, C, H, W]
     * @param ids Object IDs to update
     */
    void update_sensory(Ort::Value& sensory, const std::vector<ObjectId>& ids);

    /**
     * @brief Remove all data for objects not in keep_ids.
     * @param keep_ids List of ObjectIds to keep
     */
    void purge_except(const std::vector<ObjectId>& keep_ids);

    /**
     * @brief Clear non-permanent memory (working + long-term).
     */
    void clear_non_permanent_memory();

    /**
     * @brief Clear sensory memory (per-object visual context).
     */
    void clear_sensory_memory();

    /**
     * @brief Update configuration.
     * @param cfg New configuration
     */
    void update_config(const CutieConfig& cfg);

    /**
     * @brief Check if memory is engaged (has data).
     * @return True if memory contains data
     */
    bool engaged() const { return engaged_; }

    /**
     * @brief Get working memory store.
     * @return Reference to working memory
     */
    KeyValueMemoryStore& work_mem() { return *work_mem_; }

    /**
     * @brief Get working memory store (const).
     * @return Const reference to working memory
     */
    const KeyValueMemoryStore& work_mem() const { return *work_mem_; }

private:
    /**
     * @brief Get visual value tensors for objects.
     *
     * 从内存中提取指定对象的视觉特征。
     *
     * @param obj_ids Object IDs
     * @return Stacked value tensor
     */
    Ort::Value get_visual_values_by_ids(const std::vector<ObjectId>& obj_ids);

    /**
     * @brief Get mask for objects.
     *
     * 从掩码中提取指定对象的掩码。
     *
     * @param mask Full mask tensor
     * @param obj_ids Object IDs
     * @return Per-object mask tensor
     */
    Ort::Value get_mask_by_ids(const Ort::Value& mask, const std::vector<ObjectId>& obj_ids);

    /**
     * @brief Get sensory memory for objects.
     * @param obj_ids Object IDs
     * @return Stacked sensory tensor
     */
    Ort::Value get_sensory_by_ids(const std::vector<ObjectId>& obj_ids);

    /**
     * @brief Get object memory summaries for objects.
     * @param obj_ids Object IDs
     * @return Stacked object memory tensor
     */
    Ort::Value get_object_mem_by_ids(const std::vector<ObjectId>& obj_ids);

    /**
     * @brief Compress features in bucket (long-term memory consolidation).
     *
     * 压缩桶中的特征，用于长期内存的整合。
     *
     * @param bucket_id Target bucket
     */
    void compress_features(int bucket_id);

    /**
     * @struct ConsolidationResult
     * @brief Result of memory consolidation.
     */
    struct ConsolidationResult
    {
        Ort::Value key{nullptr};
        std::unordered_map<ObjectId, Ort::Value> values;
        Ort::Value shrinkage{nullptr};
    };

    /**
     * @brief Consolidate memory (compress and merge).
     *
     * 整合内存，将多个特征压缩为原型。
     *
     * @param candidate_key Candidate key tensor
     * @param candidate_shrinkage Candidate shrinkage tensor
     * @param candidate_selection Candidate selection tensor
     * @param candidate_value Candidate value tensors
     * @param usage Usage statistics
     * @return Consolidated result
     */
    ConsolidationResult consolidation(Ort::Value& candidate_key, Ort::Value& candidate_shrinkage,
                                      Ort::Value& candidate_selection,
                                      std::unordered_map<ObjectId, Ort::Value>& candidate_value,
                                      Ort::Value& usage);

    ortcore::GpuMemoryAllocator* alloc_;
    ObjectManager* object_manager_;
    std::unique_ptr<KeyValueMemoryStore> work_mem_;   ///< Working memory (recent frames)
    std::unique_ptr<KeyValueMemoryStore> long_mem_;   ///< Long-term memory (compressed)

    // Per-object sensory: ObjectId → [B, C, H, W] (GPU)
    std::unordered_map<ObjectId, Ort::Value> sensory_;
    // Per-object summaries: ObjectId → [B, Q, C] (GPU)
    std::unordered_map<ObjectId, Ort::Value> obj_v_;

    // Configuration parameters
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
