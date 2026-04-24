/**
 * @file object_manager.h
 * @brief Object ID tracking and tensor index mapping.
 *
 * Manages bidirectional mapping between stable user-assigned ObjectIds
 * and temporary tensor indices used in inference. Provides utilities for
 * mask conversion and per-object tensor stacking.
 */

#ifndef CUTIE_CORE_OBJECT_MANAGER_H
#define CUTIE_CORE_OBJECT_MANAGER_H

#include <cstdint>
#include <map>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

#include "cutie/common/gpu_memory.h"
#include "cutie/types.h"

namespace cutie
{
namespace core
{

/**
 * @class ObjectManager
 * @brief Manages ObjectId ↔ temporary tensor index mapping.
 *
 * Maintains stable ObjectIds (user-assigned) and maps them to temporary
 * indices (1, 2, 3, ...) used in tensor operations. Provides utilities for
 * mask conversion and per-object tensor aggregation.
 *
 * 对象管理器维护 ObjectId 和临时索引的映射关系，支持对象的添加、删除和
 * 张量索引的重新紧凑化。所有操作都是纯 STL，无推理框架依赖。
 *
 * Reference: cutie/inference/object_manager.py
 */
class ObjectManager
{
public:
    ObjectManager() = default;

    /**
     * @brief Register new objects and get their tensor indices.
     *
     * Existing objects return their current tmp_id; new objects get
     * the next available tmp_id (starting from 1).
     *
     * @param objects List of ObjectIds to register
     * @return Pair of (tmp_ids, obj_ids) for the given objects
     */
    std::pair<std::vector<int>, std::vector<ObjectId>> add_new_objects(
        const std::vector<ObjectId>& objects);

    /**
     * @brief Remove objects and recompact temporary indices.
     *
     * Deletes objects and reassigns tmp_ids to maintain contiguous
     * sequence (1, 2, 3, ...).
     * 删除对象后重新紧凑化临时索引，确保张量操作的连续性。
     *
     * @param obj_ids_to_remove List of ObjectIds to remove
     */
    void delete_objects(const std::vector<ObjectId>& obj_ids_to_remove);

    /**
     * @brief Map temporary index to ObjectId.
     * @param tmp_id Temporary tensor index
     * @return Corresponding ObjectId
     */
    ObjectId tmp_to_obj_id(int tmp_id) const;

    /**
     * @brief Map ObjectId to temporary index.
     * @param obj_id Object identifier
     * @return Corresponding temporary tensor index
     */
    int find_tmp_by_id(ObjectId obj_id) const;

    /**
     * @brief Check if all given objects are registered.
     * @param objects List of ObjectIds to check
     * @return True if all objects are active
     */
    bool has_all(const std::vector<ObjectId>& objects) const;

    /**
     * @brief Get all currently active ObjectIds.
     * @return Vector of active ObjectIds
     */
    std::vector<ObjectId> all_obj_ids() const;

    /**
     * @brief Get count of currently active objects.
     * @return Number of active objects
     */
    int num_obj() const;

    /**
     * @brief Get all historical ObjectIds (including deleted).
     * @return Reference to historical object list
     */
    const std::vector<ObjectId>& all_historical_ids() const
    {
        return all_historical_object_ids_;
    }

    /**
     * @brief Convert index mask with tmp_ids to ObjectIds.
     *
     * Converts pixel values from temporary indices to ObjectIds.
     * 将掩码中的临时索引转换为 ObjectId，用于输出结果。
     *
     * @param mask H×W CV_32SC1 with pixel values = tmp_id (0=background)
     * @return H×W CV_32SC1 with pixel values = ObjectId
     */
    cv::Mat tmp_to_obj_cls(const cv::Mat& mask) const;

    /**
     * @brief Create one-hot encoding from index mask.
     *
     * Converts per-pixel ObjectIds to one-hot tensor for network input.
     * 从分类掩码生成 one-hot 编码，用于网络输入。
     *
     * @param cls_mask H×W CV_32SC1 with pixel values = ObjectId (0=background)
     * @return [num_obj, H, W] CV_32FC1 one-hot tensor (num_obj = active object count)
     */
    cv::Mat make_one_hot(const cv::Mat& cls_mask) const;

    /**
     * @brief Stack per-object tensors into single tensor.
     *
     * Concatenates per-object tensors (stored in map) along specified axis,
     * ordered by tmp_id (1, 2, 3, ...).
     * 按 tmp_id 顺序堆叠每个对象的张量，用于批处理。
     *
     * @param obj_dict ObjectId → tensor map
     * @param dim Concatenation axis (default 1)
     * @return Stacked tensor
     */
    cv::Mat realize_dict(const std::unordered_map<ObjectId, cv::Mat>& obj_dict, int dim = 1) const;

    /**
     * @brief Stack GPU tensors into single GPU tensor.
     *
     * GPU version of realize_dict. Concatenates Ort::Value tensors
     * along specified axis, ordered by tmp_id.
     * GPU 版本的张量堆叠，用于 GPU 推理路径。
     *
     * @param obj_dict ObjectId → GPU Ort::Value map
     * @param alloc GPU memory allocator
     * @param dim Concatenation axis (default 1)
     * @return Stacked GPU Ort::Value
     */
    Ort::Value realize_dict_gpu(const std::unordered_map<ObjectId, Ort::Value>& obj_dict,
                                ortcore::GpuMemoryAllocator& alloc, int dim = 1) const;

    /**
     * @brief Get tmp_id → ObjectId mapping.
     * @return Reference to mapping
     */
    const std::map<int, ObjectId>& tmp_id_to_obj() const
    {
        return tmp_id_to_obj_;
    }

private:
    std::map<int, ObjectId> tmp_id_to_obj_;     ///< tmp_id → ObjectId
    std::map<ObjectId, int> obj_id_to_tmp_id_;  ///< ObjectId → tmp_id
    std::vector<ObjectId> all_historical_object_ids_;  ///< All ObjectIds ever registered
};

}  // namespace core
}  // namespace cutie

#endif  // CUTIE_CORE_OBJECT_MANAGER_H
