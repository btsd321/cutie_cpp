/// @file gpu_buffer.cpp
/// GpuBuffer 实现：RAII 持久化 GPU 内存块。

#include "cutie/ort/core/gpu_buffer.h"

#include <cstdio>
#include <utility>

namespace cutie
{
namespace ortcore
{

GpuBuffer::~GpuBuffer()
{
    release();
}

GpuBuffer::GpuBuffer(GpuBuffer&& other) noexcept
    : ptr_(other.ptr_), capacity_(other.capacity_)
{
    other.ptr_ = nullptr;
    other.capacity_ = 0;
}

GpuBuffer& GpuBuffer::operator=(GpuBuffer&& other) noexcept
{
    if (this != &other)
    {
        release();
        ptr_ = other.ptr_;
        capacity_ = other.capacity_;
        other.ptr_ = nullptr;
        other.capacity_ = 0;
    }
    return *this;
}

void GpuBuffer::reserve(size_t size)
{
    if (size <= capacity_) return;
    release();
    cudaError_t err = cudaMalloc(&ptr_, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "GpuBuffer::reserve cudaMalloc failed (%zu bytes): %s\n", size,
                cudaGetErrorString(err));
        ptr_ = nullptr;
        capacity_ = 0;
        return;
    }
    capacity_ = size;
}

void GpuBuffer::release()
{
    if (ptr_)
    {
        cudaFree(ptr_);
        ptr_ = nullptr;
        capacity_ = 0;
    }
}

}  // namespace ortcore
}  // namespace cutie
