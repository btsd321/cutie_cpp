/**
 * @file py_logger.cpp
 * @brief py_logger.h 的实现。
 */

#include "py_logger.h"

#include <fmt/format.h>

namespace cutie
{
namespace python
{

namespace
{

/// LogLevel → Python logging.Logger 的方法名
const char* level_to_method(linden::log::LogLevel level)
{
    switch (level)
    {
        case linden::log::LogLevel::DEBUG:
            return "debug";
        case linden::log::LogLevel::INFO:
            return "info";
        case linden::log::LogLevel::WARN:
            return "warning";  // Python 侧是 warning，不是 warn（warn 已弃用）
        case linden::log::LogLevel::ERROR:
            return "error";
    }
    return "info";
}

}  // namespace

PyLoggerBridge::PyLoggerBridge(const std::string& logger_name)
{
    py::gil_scoped_acquire gil;
    py_logger_ = py::module_::import("logging").attr("getLogger")(logger_name);
}

PyLoggerBridge::~PyLoggerBridge()
{
    // py::object 析构需要 GIL；解释器可能已在终结流程中，此时直接释放引用即可。
    py::gil_scoped_acquire gil;
    py_logger_.release().dec_ref();
}

void PyLoggerBridge::logf(linden::log::LogLevel level, fmt::string_view fmt,
                          fmt::format_args args)
{
    // 快速级别过滤：避免为被禁用的等级做格式化和 GIL 获取。
    if (static_cast<int>(level) < static_cast<int>(current_level_))
    {
        return;
    }

    std::string message;
    try
    {
        message = fmt::vformat(fmt, args);
    }
    catch (const std::exception&)
    {
        // 格式化失败不应影响推理，退化为原始格式串。
        message = std::string(fmt.data(), fmt.size());
    }

    // logf 可能来自 C++ 工作线程，必须先取 GIL 才能碰 Python 对象。
    py::gil_scoped_acquire gil;
    try
    {
        py_logger_.attr(level_to_method(level))("%s", message);
    }
    catch (const py::error_already_set&)
    {
        // 日志转发失败（如解释器终结中）时静默丢弃，不向上传播。
        PyErr_Clear();
    }
}

std::shared_ptr<linden::log::ILogger> make_python_logger(const std::string& logger_name,
                                                         linden::log::LogLevel level)
{
    auto logger = std::make_shared<PyLoggerBridge>(logger_name);
    logger->set_level(level);
    return logger;
}

void register_logger(py::module_& m)
{
    py::enum_<linden::log::LogLevel>(m, "LogLevel", "C++ 侧日志等级")
        .value("DEBUG", linden::log::LogLevel::DEBUG)
        .value("INFO", linden::log::LogLevel::INFO)
        .value("WARN", linden::log::LogLevel::WARN)
        .value("ERROR", linden::log::LogLevel::ERROR);
}

}  // namespace python
}  // namespace cutie
