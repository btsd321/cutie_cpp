#include <cstring>
#include <numeric>
#include <stdexcept>

#include <linden_logger/logger_interface.hpp>

#include "cutie/ort/core/ort_utils.h"

// ── ONNX Runtime 日志回调 ────────────────────────────────────────
// 将 ONNX Runtime 的日志转发到 linden_logger（进程级 Env 共用此回调）。
// 日志输出固定走 StdLogger 单例：Env 是进程级永生对象，无法绑定每个实例
// 各自的 logger；ORT 自身的运行时日志较少，固定级别（WARNING）足够。
namespace {
void ORT_API_CALL ort_logging_callback(void* param, OrtLoggingLevel severity,
                                       const char* category, const char* /*logid*/,
                                       const char* code_location, const char* message)
{
    auto* logger = static_cast<linden::log::ILogger*>(param);
    if (!logger) return;

    // 将 ORT 日志级别映射到 linden_logger 级别
    linden::log::LogLevel level;
    switch (severity)
    {
        case ORT_LOGGING_LEVEL_VERBOSE:
        case ORT_LOGGING_LEVEL_INFO:
            level = linden::log::LogLevel::DEBUG;
            break;
        case ORT_LOGGING_LEVEL_WARNING:
            level = linden::log::LogLevel::WARN;
            break;
        case ORT_LOGGING_LEVEL_ERROR:
        case ORT_LOGGING_LEVEL_FATAL:
            level = linden::log::LogLevel::ERROR;
            break;
        default:
            level = linden::log::LogLevel::INFO;
    }

    // 格式化输出：[ORT][category] message (location)
    if (code_location && code_location[0] != '\0')
    {
        logger->logf(level, "[ORT][{}] {} ({})", fmt::make_format_args(category, message, code_location));
    }
    else
    {
        logger->logf(level, "[ORT][{}] {}", fmt::make_format_args(category, message));
    }
}
}  // namespace

namespace cutie
{
namespace ortcore
{

Ort::Env& ort_global_env()
{
    // 进程级永生 Ort::Env 单例：堆分配且永不 delete，确保其析构函数永不执行。
    // 详见头文件 ort_utils.h 中 ort_global_env() 的说明。
    // 静态指针本身的初始化是线程安全的（C++11 函数局部 static）；析构时仅
    // 释放指针、不释放所指向的 Env 对象，故 Env 全程不被析构、provider 不被卸载。
    static Ort::Env* env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "cutie",
                                        ort_logging_callback,
                                        linden::log::StdLogger::instance().get());
    return *env;
}

Ort::Value create_tensor(const float* data, const std::vector<int64_t>& shape,
                         const Ort::MemoryInfo& /*memory_info*/)
{
    int64_t total = shape_numel(shape);
    Ort::AllocatorWithDefaultOptions allocator;
    auto value = Ort::Value::CreateTensor<float>(allocator, shape.data(), shape.size());
    std::memcpy(value.GetTensorMutableData<float>(), data, total * sizeof(float));
    return value;
}

Ort::Value mat_to_tensor(const cv::Mat& blob, const Ort::MemoryInfo& memory_info)
{
    // blob is assumed NCHW float32 from cv::dnn::blobFromImage
    if (blob.empty())
    {
        throw std::runtime_error("mat_to_tensor: input blob is empty");
    }

    // Determine shape from blob dimensions
    std::vector<int64_t> shape;
    for (int i = 0; i < blob.dims; ++i)
    {
        shape.push_back(static_cast<int64_t>(blob.size[i]));
    }

    int64_t total = shape_numel(shape);
    float* data = const_cast<float*>(blob.ptr<float>());

    return Ort::Value::CreateTensor<float>(memory_info, data, total, shape.data(), shape.size());
}

std::vector<int64_t> get_tensor_shape(const Ort::Value& tensor)
{
    auto info = tensor.GetTensorTypeAndShapeInfo();
    return info.GetShape();
}

int64_t shape_numel(const std::vector<int64_t>& shape)
{
    if (shape.empty())
        return 0;
    return std::accumulate(shape.begin(), shape.end(), int64_t(1), std::multiplies<int64_t>());
}

Ort::Value clone_tensor(const Ort::Value& src, const Ort::MemoryInfo& /*memory_info*/)
{
    auto shape = get_tensor_shape(src);
    int64_t total = shape_numel(shape);
    Ort::AllocatorWithDefaultOptions allocator;
    auto value = Ort::Value::CreateTensor<float>(allocator, shape.data(), shape.size());
    std::memcpy(value.GetTensorMutableData<float>(), src.GetTensorData<float>(),
                total * sizeof(float));
    return value;
}

Ort::Value zeros(const std::vector<int64_t>& shape, const Ort::MemoryInfo& /*memory_info*/)
{
    int64_t total = shape_numel(shape);
    Ort::AllocatorWithDefaultOptions allocator;
    auto value = Ort::Value::CreateTensor<float>(allocator, shape.data(), shape.size());
    std::memset(value.GetTensorMutableData<float>(), 0, total * sizeof(float));
    return value;
}

Ort::Value concat_tensors(const std::vector<Ort::Value>& tensors, int dim,
                          const Ort::MemoryInfo& memory_info)
{
    if (tensors.empty())
    {
        throw std::runtime_error("concat_tensors: empty input");
    }
    if (tensors.size() == 1)
    {
        return clone_tensor(tensors[0], memory_info);
    }

    auto base_shape = get_tensor_shape(tensors[0]);
    int ndim = static_cast<int>(base_shape.size());
    if (dim < 0)
        dim += ndim;
    if (dim < 0 || dim >= ndim)
    {
        throw std::runtime_error("concat_tensors: invalid dim");
    }

    // Calculate output shape
    std::vector<int64_t> out_shape = base_shape;
    out_shape[dim] = 0;
    for (auto& t : tensors)
    {
        auto s = get_tensor_shape(t);
        out_shape[dim] += s[dim];
    }

    int64_t total = shape_numel(out_shape);
    std::vector<float> buffer(total);

    // Compute strides for the output
    // Copy slices along dim
    int64_t outer = 1, inner = 1;
    for (int i = 0; i < dim; ++i) outer *= out_shape[i];
    for (int i = dim + 1; i < ndim; ++i) inner *= out_shape[i];

    int64_t out_dim_stride = out_shape[dim] * inner;
    int64_t offset_in_dim = 0;

    for (auto& t : tensors)
    {
        auto s = get_tensor_shape(t);
        int64_t t_dim = s[dim];
        const float* src = t.GetTensorData<float>();
        int64_t t_dim_stride = t_dim * inner;

        for (int64_t o = 0; o < outer; ++o)
        {
            std::memcpy(buffer.data() + o * out_dim_stride + offset_in_dim * inner,
                        src + o * t_dim_stride, t_dim * inner * sizeof(float));
        }
        offset_in_dim += t_dim;
    }

    Ort::AllocatorWithDefaultOptions allocator;
    auto value = Ort::Value::CreateTensor<float>(allocator, out_shape.data(), out_shape.size());
    std::memcpy(value.GetTensorMutableData<float>(), buffer.data(), total * sizeof(float));
    return value;
}

cv::Mat tensor_to_mat(const Ort::Value& tensor)
{
    auto shape = get_tensor_shape(tensor);
    std::vector<int> cv_sizes(shape.begin(), shape.end());
    int64_t total = shape_numel(shape);

    cv::Mat result(static_cast<int>(cv_sizes.size()), cv_sizes.data(), CV_32FC1);
    const float* src = tensor.GetTensorData<float>();
    std::memcpy(result.ptr<float>(), src, total * sizeof(float));
    return result;
}

Ort::Value mat_to_tensor_nd(const cv::Mat& mat, const Ort::MemoryInfo& memory_info)
{
    if (mat.empty())
    {
        throw std::runtime_error("mat_to_tensor_nd: empty input");
    }
    if (mat.type() != CV_32FC1)
    {
        throw std::runtime_error("mat_to_tensor_nd: expected CV_32FC1");
    }

    std::vector<int64_t> shape;
    for (int i = 0; i < mat.dims; ++i)
    {
        shape.push_back(static_cast<int64_t>(mat.size[i]));
    }
    int64_t total = shape_numel(shape);

    Ort::AllocatorWithDefaultOptions allocator;
    auto value = Ort::Value::CreateTensor<float>(allocator, shape.data(), shape.size());
    std::memcpy(value.GetTensorMutableData<float>(), mat.ptr<float>(), total * sizeof(float));
    return value;
}

Ort::Value pad_tensor_dim(const Ort::Value& src, int axis, int64_t target_size)
{
    auto shape = get_tensor_shape(src);
    int ndim = static_cast<int>(shape.size());
    if (axis < 0) axis += ndim;

    int64_t actual_size = shape[axis];
    if (actual_size >= target_size)
    {
        // No padding needed – return a plain clone
        Ort::AllocatorWithDefaultOptions alloc;
        auto value = Ort::Value::CreateTensor<float>(alloc, shape.data(), shape.size());
        std::memcpy(value.GetTensorMutableData<float>(), src.GetTensorData<float>(),
                    shape_numel(shape) * sizeof(float));
        return value;
    }

    std::vector<int64_t> out_shape = shape;
    out_shape[axis] = target_size;

    Ort::AllocatorWithDefaultOptions alloc;
    auto value = Ort::Value::CreateTensor<float>(alloc, out_shape.data(), out_shape.size());
    float* dst = value.GetTensorMutableData<float>();
    std::memset(dst, 0, shape_numel(out_shape) * sizeof(float));

    // Compute outer / inner strides based on output shape
    int64_t outer = 1;
    for (int i = 0; i < axis; ++i) outer *= shape[i];
    int64_t inner = 1;
    for (int i = axis + 1; i < ndim; ++i) inner *= shape[i];

    const float* src_ptr = src.GetTensorData<float>();
    int64_t src_stride = actual_size * inner;
    int64_t dst_stride = target_size * inner;

    for (int64_t o = 0; o < outer; ++o)
    {
        std::memcpy(dst + o * dst_stride, src_ptr + o * src_stride,
                    actual_size * inner * sizeof(float));
    }
    return value;
}

Ort::Value slice_tensor_dim(const Ort::Value& src, int axis, int64_t actual_size)
{
    auto shape = get_tensor_shape(src);
    int ndim = static_cast<int>(shape.size());
    if (axis < 0) axis += ndim;

    int64_t source_size = shape[axis];
    if (actual_size >= source_size)
    {
        Ort::AllocatorWithDefaultOptions alloc;
        auto value = Ort::Value::CreateTensor<float>(alloc, shape.data(), shape.size());
        std::memcpy(value.GetTensorMutableData<float>(), src.GetTensorData<float>(),
                    shape_numel(shape) * sizeof(float));
        return value;
    }

    std::vector<int64_t> out_shape = shape;
    out_shape[axis] = actual_size;

    Ort::AllocatorWithDefaultOptions alloc;
    auto value = Ort::Value::CreateTensor<float>(alloc, out_shape.data(), out_shape.size());
    float* dst = value.GetTensorMutableData<float>();

    int64_t outer = 1;
    for (int i = 0; i < axis; ++i) outer *= shape[i];
    int64_t inner = 1;
    for (int i = axis + 1; i < ndim; ++i) inner *= shape[i];

    const float* src_ptr = src.GetTensorData<float>();
    int64_t src_stride = source_size * inner;
    int64_t dst_stride = actual_size * inner;

    for (int64_t o = 0; o < outer; ++o)
    {
        std::memcpy(dst + o * dst_stride, src_ptr + o * src_stride,
                    actual_size * inner * sizeof(float));
    }
    return value;
}

}  // namespace ortcore
}  // namespace cutie
