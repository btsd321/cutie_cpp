/**
 * @file module.cpp
 * @brief cutie_cpp._core 扩展模块入口。
 *
 * 本模块是**内部实现**，不面向最终用户。用户应通过 `cutie_cpp` 包的
 * Python API（`VideoSegmenter` / `CutieConfig`）使用本库——那一层提供了
 * dataclass 配置、参数校验、YAML 加载和更友好的错误信息。
 *
 * 命名约定：本模块暴露的类型统一带 `Native` 前缀，与 Python 侧同名封装区分。
 */

#include <pybind11/pybind11.h>

// config.h 会 #define ENABLE_ONNXRUNTIME，而 cutie target 又通过 PUBLIC
// compile definition 传递了同名宏，直接包含会触发 redefined 警告。
#ifdef ENABLE_ONNXRUNTIME
#undef ENABLE_ONNXRUNTIME
#endif
#include "cutie/config.h"

namespace cutie
{
namespace python
{

namespace py = pybind11;

// 各子模块的注册函数（定义于对应 .cpp）
void register_config(py::module_& m);
void register_logger(py::module_& m);
void register_processor(py::module_& m);

}  // namespace python
}  // namespace cutie

PYBIND11_MODULE(_core, m)
{
    m.doc() = "cutie_cpp 的 C++ 扩展模块（内部实现，请使用 cutie_cpp 包的公开 API）";

    // 与 C++ 库版本保持一致，由 CMake 通过 config.h 注入。
    m.attr("__version__") = CUTIE_VERSION;

    cutie::python::register_logger(m);
    cutie::python::register_config(m);
    cutie::python::register_processor(m);
}
