/**
 * @file py_config.cpp
 * @brief CutieConfig / StepOptions 及相关枚举的 Python 绑定。
 *
 * 这一层是 C++ 结构体的**薄映射**，不做参数校验——校验逻辑放在 Python 侧的
 * `cutie_cpp.config.CutieConfig.validate()`，那里能给出更友好的错误信息，
 * 也便于在不进入 C++ 之前就拦截错误配置（例如 CPU device 不受支持）。
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cutie/core/processor.h"
#include "cutie/types.h"

namespace cutie
{
namespace python
{

namespace py = pybind11;
using core::CutieConfig;
using core::StepOptions;

/**
 * @brief 注册配置相关的 Python 类型。
 * @param m 目标模块
 */
void register_config(py::module_& m)
{
    // ─── 枚举 ───────────────────────────────────────────────────────
    py::enum_<Device>(m, "Device", "计算设备")
        .value("CPU", Device::kCPU, "CPU 推理（当前后端不支持，仅为完整性保留）")
        .value("CUDA", Device::kCUDA, "CUDA GPU 推理");

    py::enum_<ModelVariant>(m, "ModelVariant", "模型架构变体")
        .value("BASE", ModelVariant::kBase, "base 变体（精度更高）")
        .value("SMALL", ModelVariant::kSmall, "small 变体（速度更快）");

    // ─── 长期内存配置（嵌套结构体）───────────────────────────────────
    py::class_<CutieConfig::LongTermConfig>(m, "NativeLongTermConfig", "长期内存整合参数")
        .def(py::init<>())
        .def_readwrite("count_usage", &CutieConfig::LongTermConfig::count_usage)
        .def_readwrite("max_mem_frames", &CutieConfig::LongTermConfig::max_mem_frames)
        .def_readwrite("min_mem_frames", &CutieConfig::LongTermConfig::min_mem_frames)
        .def_readwrite("num_prototypes", &CutieConfig::LongTermConfig::num_prototypes)
        .def_readwrite("max_num_tokens", &CutieConfig::LongTermConfig::max_num_tokens)
        .def_readwrite("buffer_tokens", &CutieConfig::LongTermConfig::buffer_tokens);

    // ─── 模型维度（嵌套结构体）─────────────────────────────────────
    py::class_<CutieConfig::ModelDims>(m, "NativeModelDims", "模型架构维度")
        .def(py::init<>())
        .def_readwrite("key_dim", &CutieConfig::ModelDims::key_dim)
        .def_readwrite("value_dim", &CutieConfig::ModelDims::value_dim)
        .def_readwrite("sensory_dim", &CutieConfig::ModelDims::sensory_dim)
        .def_readwrite("pixel_dim", &CutieConfig::ModelDims::pixel_dim)
        .def_readwrite("f16_dim", &CutieConfig::ModelDims::f16_dim)
        .def_readwrite("f8_dim", &CutieConfig::ModelDims::f8_dim)
        .def_readwrite("f4_dim", &CutieConfig::ModelDims::f4_dim)
        .def_readwrite("num_queries", &CutieConfig::ModelDims::num_queries);

    // ─── 主配置 ─────────────────────────────────────────────────────
    py::class_<CutieConfig>(m, "NativeConfig", "C++ 侧完整推理配置")
        .def(py::init<>())
        .def_static("base_default", &CutieConfig::base_default, py::arg("model_dir"),
                    "创建 base 变体的默认配置")
        .def_static("small_default", &CutieConfig::small_default, py::arg("model_dir"),
                    "创建 small 变体的默认配置")
        // 模型选择
        .def_readwrite("variant", &CutieConfig::variant)
        .def_readwrite("model_dir", &CutieConfig::model_dir)
        .def_readwrite("model_prefix", &CutieConfig::model_prefix)
        // 设备
        .def_readwrite("device", &CutieConfig::device)
        .def_readwrite("device_id", &CutieConfig::device_id)
        .def_readwrite("single_object", &CutieConfig::single_object)
        // 推理参数
        .def_readwrite("max_internal_size", &CutieConfig::max_internal_size)
        .def_readwrite("mem_every", &CutieConfig::mem_every)
        .def_readwrite("top_k", &CutieConfig::top_k)
        .def_readwrite("chunk_size", &CutieConfig::chunk_size)
        .def_readwrite("stagger_updates", &CutieConfig::stagger_updates)
        // 内存配置
        .def_readwrite("max_mem_frames", &CutieConfig::max_mem_frames)
        .def_readwrite("use_long_term", &CutieConfig::use_long_term)
        .def_readwrite("long_term", &CutieConfig::long_term)
        .def_readwrite("model", &CutieConfig::model);

    // ─── 单帧选项 ───────────────────────────────────────────────────
    py::class_<StepOptions>(m, "NativeStepOptions", "单帧推理选项")
        .def(py::init<>())
        .def_readwrite("idx_mask", &StepOptions::idx_mask)
        .def_readwrite("end", &StepOptions::end)
        .def_readwrite("force_permanent", &StepOptions::force_permanent);
}

}  // namespace python
}  // namespace cutie
