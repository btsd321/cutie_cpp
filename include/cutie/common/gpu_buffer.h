#pragma once

#include <cuda_runtime.h>
#include <cstddef>

/// @file gpu_buffer.h
/// 持久化 GPU 内存块，避免 per-frame cudaMalloc/cudaFree。
/// 参考 trtgpu::GpuBuffer（lite.ai.toolkit）的 RAII 设计，独立维护。

namespace cutie
{
namespace ortcore
{

/// 持久化 GPU 内存块，自动扩容，RAII 析构释放。
/// 用于 GpuImagePreprocessor 等需要 per-frame GPU 缓冲的场景，
/// 避免高频 cudaMalloc/cudaFree 开销。
class GpuBuffer
{
public:
    GpuBuffer() = default;
    ~GpuBuffer();

    GpuBuffer(const GpuBuffer&) = delete;
    GpuBuffer& operator=(const GpuBuffer&) = delete;
    GpuBuffer(GpuBuffer&& other) noexcept;
    GpuBuffer& operator=(GpuBuffer&& other) noexcept;

    /// 确保至少 size 字节可用（容量不足时重新分配，旧内容不保留）。
    void reserve(size_t size);

    void* data() { return ptr_; }
    const void* data() const { return ptr_; }
    size_t capacity() const { return capacity_; }

    void release();

    template <typename T>
    T* as()
    {
        return static_cast<T*>(ptr_);
    }

    template <typename T>
    const T* as() const
    {
        return static_cast<const T*>(ptr_);
    }

private:
    void* ptr_ = nullptr;
    size_t capacity_ = 0;
};

}  // namespace ortcore
}  // namespace cutie
