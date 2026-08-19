/**
 * @file py_processor.cpp
 * @brief CutieProcessor 的 Python 绑定。
 *
 * 设计要点：
 * - **释放 GIL**：单帧推理耗时 15-30 ms，若不释放 GIL 会阻塞整个 Python 解释器，
 *   使多线程取帧/写盘完全失效。所有 step 调用都用 `py::gil_scoped_release` 包裹。
 * - **走 GPU 路径**：只暴露 `step_gpu(cv::Mat, ...)` 重载。它内部自动上传到 GPU
 *   并全程在 GPU 完成推理，仅最终掩码需要 D2H，比 CPU `step()` 少一次概率图落地。
 * - **单次 D2H**：故意不调用 `GpuCutieMask::download()`（它会自行分配 cv::Mat），
 *   而是用 py_convert.h 的函数直接把 GPU 数据写入 numpy 缓冲区。
 * - **概率图按需下载**：prob 张量在 1080p / 3 目标下约 25 MB/帧，默认不下载。
 */

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cutie/core/processor.h"
#include "cutie/types.h"

#include "py_convert.h"
#include "py_logger.h"

namespace cutie
{
namespace python
{

namespace py = pybind11;
using core::CutieConfig;
using core::CutieProcessor;
using core::StepOptions;

namespace
{

/**
 * @struct StepResult
 * @brief 单帧推理结果的 Python 侧载体。
 *
 * 只持有已转换为 numpy 的数据，不保留任何 GPU 资源，
 * 因此可以安全地跨帧长期持有。
 */
struct StepResult
{
    py::array index_mask;              ///< (H, W) int32，像素值 = 目标 ID
    std::vector<ObjectId> object_ids;  ///< 当前活跃目标 ID
    py::array prob;                    ///< [N+1, H, W] float32，未请求时为空数组
    bool valid = false;                ///< 结果有效性标志
};

/**
 * @brief 执行单帧推理并把结果转换为 numpy。
 *
 * @param processor   处理器实例
 * @param image       输入帧（BGR，(H, W, 3) uint8）
 * @param mask        首帧掩码（(H, W) int32），后续帧传空数组
 * @param object_ids  要跟踪的目标 ID；为空且提供了 mask 时自动从 mask 推导
 * @param options     单帧选项
 * @param return_prob 是否下载概率图（开销较大）
 * @return 转换完成的结果
 */
StepResult do_step(CutieProcessor& processor, const ImageArray& image, const MaskArray& mask,
                   const std::vector<ObjectId>& object_ids, const StepOptions& options,
                   bool return_prob)
{
    // 零拷贝包装：wrapped_* 借用 numpy 缓冲区，其生命周期不超出本函数。
    const cv::Mat wrapped_image = image_from_numpy(image);

    cv::Mat wrapped_mask;
    if (mask.size() > 0)
    {
        wrapped_mask = mask_from_numpy(mask);
        if (wrapped_mask.rows != wrapped_image.rows || wrapped_mask.cols != wrapped_image.cols)
        {
            throw std::invalid_argument(
                "掩码尺寸 (" + std::to_string(wrapped_mask.rows) + ", " +
                std::to_string(wrapped_mask.cols) + ") 与图像尺寸 (" +
                std::to_string(wrapped_image.rows) + ", " + std::to_string(wrapped_image.cols) +
                ") 不一致");
        }
    }

    // object_ids 省略时从掩码推导，让 Python 侧首帧调用可以只传 mask。
    std::vector<ObjectId> ids = object_ids;
    if (ids.empty() && !wrapped_mask.empty())
    {
        ids = unique_object_ids(wrapped_mask);
        if (ids.empty())
        {
            throw std::invalid_argument("掩码中没有非零像素，无法推导目标 ID");
        }
    }

    types::GpuCutieMask gpu_result;
    {
        // 推理期间释放 GIL，让 Python 侧其它线程（取帧、编码、IO）能并行推进。
        py::gil_scoped_release release;

        // 自行上传而不用 step_gpu 的 cv::Mat 便捷重载：那个重载不接受 StepOptions
        // （见 processor.h），且会对掩码做一次多余的类型转换——这里的掩码来自
        // numpy int32，已经是 CV_32SC1，可直接上传。H2D 因此只发生一次。
        cv::cuda::GpuMat gpu_image;
        gpu_image.upload(wrapped_image);

        cv::cuda::GpuMat gpu_mask;
        if (!wrapped_mask.empty())
        {
            gpu_mask.upload(wrapped_mask);
        }

        gpu_result = processor.step_gpu(gpu_image, gpu_mask, ids, options);
    }

    // 此处已重新持有 GIL，可以创建 numpy 对象。
    StepResult result;
    result.valid = gpu_result.flag;
    result.object_ids = gpu_result.object_ids;
    result.index_mask = index_mask_to_numpy(gpu_result.index_mask);
    if (return_prob)
    {
        result.prob = prob_to_numpy(gpu_result.gpu_prob);
    }
    else
    {
        result.prob = py::array_t<float>(std::vector<py::ssize_t>{0});
    }

    return result;
}

/**
 * @brief 构造 CutieProcessor，按需接入 Python logging 桥接。
 *
 * @param config             C++ 配置
 * @param use_native_logger  true 使用 C++ 原生 stdout 日志；false 转发到 Python logging
 * @param log_level          日志等级
 * @return 处理器实例
 */
std::unique_ptr<CutieProcessor> make_processor(const CutieConfig& config, bool use_native_logger,
                                              linden::log::LogLevel log_level)
{
    std::shared_ptr<linden::log::ILogger> logger;
    if (!use_native_logger)
    {
        logger = make_python_logger("cutie_cpp.native", log_level);
    }
    // logger 为 nullptr 时 CutieProcessor 会退回 StdLogger 单例。
    return std::make_unique<CutieProcessor>(config, std::move(logger));
}

}  // namespace

/**
 * @brief 注册处理器与结果类型的 Python 绑定。
 * @param m 目标模块
 */
void register_processor(py::module_& m)
{
    py::class_<StepResult>(m, "NativeStepResult", "单帧推理结果（数据已在 CPU）")
        .def_readonly("index_mask", &StepResult::index_mask)
        .def_readonly("object_ids", &StepResult::object_ids)
        .def_readonly("prob", &StepResult::prob)
        .def_readonly("valid", &StepResult::valid);

    py::class_<CutieProcessor>(m, "NativeProcessor", "有状态的视频目标分割处理器")
        .def(py::init(&make_processor), py::arg("config"), py::arg("use_native_logger") = false,
             py::arg("log_level") = linden::log::LogLevel::INFO,
             "构造处理器；模型文件缺失或解析失败时抛出 RuntimeError")
        .def("step", &do_step, py::arg("image"), py::arg("mask"), py::arg("object_ids"),
             py::arg("options"), py::arg("return_prob") = false,
             "执行单帧推理（内部走全 GPU 路径，推理期间释放 GIL）")
        .def("delete_objects", &CutieProcessor::delete_objects, py::arg("object_ids"),
             "停止跟踪指定目标并释放其内存")
        .def("active_objects", &CutieProcessor::active_objects, "获取当前活跃目标 ID 列表")
        .def("num_objects", &CutieProcessor::num_objects, "获取当前活跃目标数量")
        .def("clear_memory", &CutieProcessor::clear_memory, "清空全部内存（工作/长期/感知）")
        .def("clear_non_permanent_memory", &CutieProcessor::clear_non_permanent_memory,
             "清空非永久内存（工作 + 长期），保留感知内存")
        .def("clear_sensory_memory", &CutieProcessor::clear_sensory_memory, "清空感知内存")
        .def("config", &CutieProcessor::config, py::return_value_policy::copy,
             "获取当前配置的副本");
}

}  // namespace python
}  // namespace cutie
