#ifndef CUTIE_CORE_IMAGE_FEATURE_STORE_H
#define CUTIE_CORE_IMAGE_FEATURE_STORE_H

#include <unordered_map>

#include "cutie/ort/core/ort_config.h"

namespace cutie
{

namespace ortcore
{
class GpuMemoryAllocator;
}

namespace core
{

/// Caches encoded image features (GPU Ort::Value) to avoid recomputation
/// when a frame is processed multiple times (e.g., mask + propagation on same frame).
class ImageFeatureStore
{
public:
    ImageFeatureStore() = default;

    struct CachedFeatures
    {
        Ort::Value f16{nullptr};        // GPU: [1, C16, H/16, W/16]
        Ort::Value f8{nullptr};         // GPU: [1, C8, H/8, W/8]
        Ort::Value f4{nullptr};         // GPU: [1, C4, H/4, W/4]
        Ort::Value pix_feat{nullptr};   // GPU: [1, pixel_dim, H/16, W/16]
        Ort::Value key{nullptr};        // GPU: [1, key_dim, H/16, W/16]
        Ort::Value shrinkage{nullptr};  // GPU: [1, 1, H/16, W/16]
        Ort::Value selection{nullptr};  // GPU: [1, key_dim, H/16, W/16]

        bool valid() const { return f16.IsTensor(); }
    };

    bool has(int frame_idx) const
    {
        return cache_.count(frame_idx) > 0;
    }

    CachedFeatures& get(int frame_idx)
    {
        return cache_.at(frame_idx);
    }

    const CachedFeatures& get(int frame_idx) const
    {
        return cache_.at(frame_idx);
    }

    void put(int frame_idx, CachedFeatures features)
    {
        cache_.erase(frame_idx);
        cache_.emplace(frame_idx, std::move(features));
    }

    void clear()
    {
        cache_.clear();
    }

    /// Remove all entries except for the given frame index.
    void keep_only(int frame_idx)
    {
        auto it = cache_.find(frame_idx);
        if (it != cache_.end())
        {
            CachedFeatures saved = std::move(it->second);
            cache_.clear();
            if (saved.valid())
            {
                cache_.emplace(frame_idx, std::move(saved));
            }
        }
        else
        {
            cache_.clear();
        }
    }

private:
    std::unordered_map<int, CachedFeatures> cache_;
};

}  // namespace core
}  // namespace cutie

#endif  // CUTIE_CORE_IMAGE_FEATURE_STORE_H
