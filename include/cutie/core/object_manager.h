#ifndef CUTIE_CORE_OBJECT_MANAGER_H
#define CUTIE_CORE_OBJECT_MANAGER_H

#include <cstdint>
#include <map>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

#include "cutie/ort/core/gpu_memory.h"
#include "cutie/types.h"

namespace cutie
{
namespace core
{

/// Manages the mapping between stable ObjectIds and temporary tensor indices.
/// Temporary IDs start from 1. ObjectIds are immutable and user-assigned.
/// Pure STL, no inference framework dependency.
/// Reference: cutie/inference/object_manager.py
class ObjectManager
{
public:
    ObjectManager() = default;

    /// Register objects. Returns (tmp_ids, obj_ids) for the given objects.
    /// Existing objects return their current tmp_id; new objects get the next available tmp_id.
    std::pair<std::vector<int>, std::vector<ObjectId>> add_new_objects(
        const std::vector<ObjectId>& objects);

    /// Remove objects and recompact tmp_ids.
    void delete_objects(const std::vector<ObjectId>& obj_ids_to_remove);

    /// Map tmp_id → ObjectId.
    ObjectId tmp_to_obj_id(int tmp_id) const;

    /// Map ObjectId → tmp_id.
    int find_tmp_by_id(ObjectId obj_id) const;

    /// Check if all given objects are registered.
    bool has_all(const std::vector<ObjectId>& objects) const;

    /// All currently active object IDs.
    std::vector<ObjectId> all_obj_ids() const;

    /// Number of currently active objects.
    int num_obj() const;

    /// All historical object IDs (including deleted).
    const std::vector<ObjectId>& all_historical_ids() const
    {
        return all_historical_object_ids_;
    }

    /// Convert an index mask with tmp_ids to ObjectIds.
    /// mask: H×W CV_32SC1 with pixel values = tmp_id. 0 = background.
    cv::Mat tmp_to_obj_cls(const cv::Mat& mask) const;

    /// Create one-hot from index mask.
    /// cls_mask: H×W CV_32SC1 with pixel values = ObjectId. 0 = background.
    /// Returns: [num_obj, H, W] CV_32FC1 (num_obj = number of active objects, ordered by tmp_id)
    cv::Mat make_one_hot(const cv::Mat& cls_mask) const;

    /// Stack per-object tensors (stored in a map) into a single tensor along a given axis.
    /// The stacking order follows tmp_id order (1, 2, 3, ...).
    /// T should be cv::Mat or similar.
    /// obj_dict: ObjectId → tensor
    /// Returns concatenated tensor.
    cv::Mat realize_dict(const std::unordered_map<ObjectId, cv::Mat>& obj_dict, int dim = 1) const;

    /// GPU 版本：按 tmp_id 顺序堆叠 GPU Ort::Value 张量。
    /// obj_dict: ObjectId → GPU Ort::Value
    /// 返回: 沿 dim=1 堆叠的 GPU Ort::Value
    Ort::Value realize_dict_gpu(const std::unordered_map<ObjectId, Ort::Value>& obj_dict,
                                ortcore::GpuMemoryAllocator& alloc, int dim = 1) const;

    // Access the tmp→obj mapping
    const std::map<int, ObjectId>& tmp_id_to_obj() const
    {
        return tmp_id_to_obj_;
    }

private:
    std::map<int, ObjectId> tmp_id_to_obj_;     // tmp_id → ObjectId
    std::map<ObjectId, int> obj_id_to_tmp_id_;  // ObjectId → tmp_id
    std::vector<ObjectId> all_historical_object_ids_;
};

}  // namespace core
}  // namespace cutie

#endif  // CUTIE_CORE_OBJECT_MANAGER_H
