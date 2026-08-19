/**
 * @file py_logger.h
 * @brief 把 C++ 侧 linden_logger 的日志转发到 Python 标准库 logging。
 *
 * Python 编码规范要求统一用 `logging` 模块输出运行信息，因此 C++ 库的日志
 * 也应汇入同一套 handler，避免终端出现两种格式、两个来源的输出。
 *
 * 注意事项：
 * - `linden::log::StdLogger` 使用独立工作线程异步输出；本桥接类的 `logf()`
 *   可能在**非 Python 线程**被调用，因此每次转发都必须 `py::gil_scoped_acquire`。
 * - 获取 GIL 会让日志调用串行化。高频 debug 日志场景下开销明显，
 *   因此 Python 侧提供 `use_native_logger=True` 回退到 C++ 原生 stdout 日志。
 * - 若 Python 解释器已开始终结（`Py_IsInitialized()` 为假），转发会被静默丢弃，
 *   避免在进程退出阶段崩溃。
 */

#ifndef CUTIE_PYTHON_PY_LOGGER_H
#define CUTIE_PYTHON_PY_LOGGER_H

#include <memory>
#include <string>

#include <pybind11/pybind11.h>

#include <linden_logger/logger_interface.hpp>

namespace cutie
{
namespace python
{

namespace py = pybind11;

/**
 * @class PyLoggerBridge
 * @brief ILogger 实现，将日志记录转发给 Python 的 logging.Logger。
 *
 * 每条日志按等级映射到 Python logging 的对应方法：
 * DEBUG→debug、INFO→info、WARN→warning、ERROR→error。
 */
class PyLoggerBridge : public linden::log::ILogger
{
public:
    /**
     * @brief 构造桥接 logger。
     * @param logger_name Python logger 名称（如 "cutie_cpp.native"）
     */
    explicit PyLoggerBridge(const std::string& logger_name);

    ~PyLoggerBridge() override;

    /**
     * @brief 格式化日志并转发到 Python logging。
     *
     * @param level 日志等级
     * @param fmt   fmt 格式字符串
     * @param args  格式化参数
     *
     * @note 内部获取 GIL；调用方不应持有 GIL 以外的 Python 状态假设。
     *       格式化或转发过程中的任何异常都会被吞掉——日志失败不应中断推理。
     */
    void logf(linden::log::LogLevel level, fmt::string_view fmt, fmt::format_args args) override;

private:
    /// 缓存的 Python logging.Logger 对象
    py::object py_logger_;
};

/**
 * @brief 创建转发到 Python logging 的 logger 实例。
 *
 * @param logger_name Python logger 名称
 * @param level       初始日志等级
 * @return 可传给 CutieProcessor 的 ILogger 共享指针
 */
std::shared_ptr<linden::log::ILogger> make_python_logger(const std::string& logger_name,
                                                         linden::log::LogLevel level);

/**
 * @brief 注册日志相关的 Python 绑定（LogLevel 枚举）。
 * @param m 目标模块
 */
void register_logger(py::module_& m);

}  // namespace python
}  // namespace cutie

#endif  // CUTIE_PYTHON_PY_LOGGER_H
