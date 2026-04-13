#include <algorithm>
#include <set>
#include <stdexcept>

#include "cutie/core/object_manager.h"
#include "cutie/ort/core/gpu_tensor_ops.h"

namespace cutie
{
namespace core
{

std::pair<std::vector<int>, std::vector<ObjectId>> ObjectManager::add_new_objects(
    const std::vector<ObjectId>& objects)
{
    std::vector<int> corresponding_tmp_ids;
    std::vector<ObjectId> corresponding_obj_ids;

    for (auto obj_id : objects)
    {
        auto it = obj_id_to_tmp_id_.find(obj_id);
        if (it != obj_id_to_tmp_id_.end())
        {
            // Existing object
            corresponding_tmp_ids.push_back(it->second);
            corresponding_obj_ids.push_back(obj_id);
        }
        else
        {
            // New object: assign next tmp_id
            int new_tmp_id = static_cast<int>(obj_id_to_tmp_id_.size()) + 1;
            obj_id_to_tmp_id_[obj_id] = new_tmp_id;
            tmp_id_to_obj_[new_tmp_id] = obj_id;
            all_historical_object_ids_.push_back(obj_id);
            corresponding_tmp_ids.push_back(new_tmp_id);
            corresponding_obj_ids.push_back(obj_id);
        }
    }

    return {corresponding_tmp_ids, corresponding_obj_ids};
}

void ObjectManager::delete_objects(const std::vector<ObjectId>& obj_ids_to_remove)
{
    std::set<ObjectId> to_remove(obj_ids_to_remove.begin(), obj_ids_to_remove.end());

    int total = static_cast<int>(tmp_id_to_obj_.size());

    std::map<int, ObjectId> new_tmp_to_obj;
    std::map<ObjectId, int> new_obj_to_tmp;

    int new_tmp_id = 1;
    for (int t = 1; t <= total; ++t)
    {
        auto it = tmp_id_to_obj_.find(t);
        if (it == tmp_id_to_obj_.end())
            continue;
        ObjectId obj_id = it->second;
        if (to_remove.count(obj_id) == 0)
        {
            new_tmp_to_obj[new_tmp_id] = obj_id;
            new_obj_to_tmp[obj_id] = new_tmp_id;
            ++new_tmp_id;
        }
    }

    tmp_id_to_obj_ = std::move(new_tmp_to_obj);
    obj_id_to_tmp_id_ = std::move(new_obj_to_tmp);
}

ObjectId ObjectManager::tmp_to_obj_id(int tmp_id) const
{
    auto it = tmp_id_to_obj_.find(tmp_id);
    if (it == tmp_id_to_obj_.end())
    {
        throw std::runtime_error("ObjectManager: tmp_id not found: " + std::to_string(tmp_id));
    }
    return it->second;
}

int ObjectManager::find_tmp_by_id(ObjectId obj_id) const
{
    auto it = obj_id_to_tmp_id_.find(obj_id);
    if (it == obj_id_to_tmp_id_.end())
    {
        throw std::runtime_error("ObjectManager: obj_id not found: " + std::to_string(obj_id));
    }
    return it->second;
}

bool ObjectManager::has_all(const std::vector<ObjectId>& objects) const
{
    for (auto obj_id : objects)
    {
        if (obj_id_to_tmp_id_.count(obj_id) == 0)
        {
            return false;
        }
    }
    return true;
}

std::vector<ObjectId> ObjectManager::all_obj_ids() const
{
    std::vector<ObjectId> result;
    result.reserve(tmp_id_to_obj_.size());
    // Return in tmp_id order
    for (auto& [tmp_id, obj_id] : tmp_id_to_obj_)
    {
        result.push_back(obj_id);
    }
    return result;
}

int ObjectManager::num_obj() const
{
    return static_cast<int>(tmp_id_to_obj_.size());
}

cv::Mat ObjectManager::tmp_to_obj_cls(const cv::Mat& mask) const
{
    cv::Mat result = cv::Mat::zeros(mask.size(), CV_32SC1);
    for (int r = 0; r < mask.rows; ++r)
    {
        const int32_t* src = mask.ptr<int32_t>(r);
        int32_t* dst = result.ptr<int32_t>(r);
        for (int c = 0; c < mask.cols; ++c)
        {
            int tmp_id = src[c];
            if (tmp_id == 0)
                continue;
            auto it = tmp_id_to_obj_.find(tmp_id);
            if (it != tmp_id_to_obj_.end())
            {
                dst[c] = it->second;
            }
        }
    }
    return result;
}

cv::Mat ObjectManager::make_one_hot(const cv::Mat& cls_mask) const
{
    int h = cls_mask.rows, w = cls_mask.cols;
    int n = num_obj();

    if (n == 0)
    {
        int sizes[] = {0, h, w};
        return cv::Mat(3, sizes, CV_32FC1);
    }

    int sizes[] = {n, h, w};
    cv::Mat result(3, sizes, CV_32FC1, cv::Scalar(0.0f));

    // Iterate in tmp_id order
    int idx = 0;
    for (auto& [tmp_id, obj_id] : tmp_id_to_obj_)
    {
        float* dst = result.ptr<float>() + idx * h * w;
        for (int r = 0; r < h; ++r)
        {
            const int32_t* src = cls_mask.ptr<int32_t>(r);
            for (int c = 0; c < w; ++c)
            {
                dst[r * w + c] = (src[c] == obj_id) ? 1.0f : 0.0f;
            }
        }
        ++idx;
    }

    return result;
}

cv::Mat ObjectManager::realize_dict(const std::unordered_map<ObjectId, cv::Mat>& obj_dict,
                                    int dim) const
{
    // Stack tensors from obj_dict in tmp_id order along dimension `dim`.
    // For simplicity, only support dim=1 stacking for now (the common case in Cutie).
    std::vector<cv::Mat> ordered;
    ordered.reserve(tmp_id_to_obj_.size());

    for (auto& [tmp_id, obj_id] : tmp_id_to_obj_)
    {
        auto it = obj_dict.find(obj_id);
        if (it == obj_dict.end())
        {
            throw std::runtime_error("realize_dict: missing ObjectId " + std::to_string(obj_id));
        }
        ordered.push_back(it->second);
    }

    if (ordered.empty())
    {
        return cv::Mat();
    }

    // Simple vertical concat for dim=0, channel concat otherwise
    // For 2D Mats: vconcat; for higher-dim, we do manual stacking
    if (dim == 0 && ordered[0].dims <= 2)
    {
        cv::Mat result;
        cv::vconcat(ordered, result);
        return result;
    }

    // For multi-dimensional Mats, create a new Mat with an extra dimension
    const cv::Mat& first = ordered[0];
    std::vector<int> new_sizes;
    for (int i = 0; i < dim; ++i)
    {
        new_sizes.push_back(first.size[i]);
    }
    new_sizes.push_back(static_cast<int>(ordered.size()));
    for (int i = dim; i < first.dims; ++i)
    {
        new_sizes.push_back(first.size[i]);
    }

    int total_per_item = 1;
    for (int i = 0; i < first.dims; ++i)
    {
        total_per_item *= first.size[i];
    }

    cv::Mat result(static_cast<int>(new_sizes.size()), new_sizes.data(), CV_32FC1);
    float* dst = result.ptr<float>();

    // Compute strides for inserting at `dim`
    int outer = 1;
    for (int i = 0; i < dim; ++i) outer *= first.size[i];
    int inner = total_per_item / outer;

    for (int o = 0; o < outer; ++o)
    {
        for (int n = 0; n < static_cast<int>(ordered.size()); ++n)
        {
            const float* src = ordered[n].ptr<float>() + o * inner;
            std::copy(src, src + inner, dst);
            dst += inner;
        }
    }

    return result;
}

Ort::Value ObjectManager::realize_dict_gpu(const std::unordered_map<ObjectId, Ort::Value>& obj_dict,
                                            ortcore::GpuMemoryAllocator& alloc, int dim) const
{
    std::vector<Ort::Value*> ordered;
    ordered.reserve(tmp_id_to_obj_.size());

    for (auto& [tmp_id, obj_id] : tmp_id_to_obj_)
    {
        auto it = obj_dict.find(obj_id);
        if (it == obj_dict.end())
        {
            throw std::runtime_error("realize_dict_gpu: missing ObjectId " +
                                     std::to_string(obj_id));
        }
        ordered.push_back(const_cast<Ort::Value*>(&it->second));
    }

    if (ordered.empty())
    {
        return Ort::Value{nullptr};
    }

    return ortcore::gpu_stack(alloc, ordered, dim);
}

}  // namespace core
}  // namespace cutie
