cmake_minimum_required(VERSION 3.18)
project(cutie VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# === CUDA 设置 ===
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# 支持 Turing(75), Ampere(80,86), Ada Lovelace(89), Hopper(90)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

find_package(CUDAToolkit REQUIRED)
if(CUDAToolkit_VERSION VERSION_LESS "11.8")
    message(FATAL_ERROR "CUDA 11.8 or higher is required, found ${CUDAToolkit_VERSION}")
endif()
message(STATUS "CUDA Toolkit: ${CUDAToolkit_VERSION}")

# === 后端选项（参考 lite.ai.toolkit 模式） ===
option(ENABLE_ONNXRUNTIME "Build with ONNXRuntime backend" ON)
option(ENABLE_TENSORRT    "Build with TensorRT backend"    OFF)
option(BUILD_EXAMPLES     "Build example programs"          ON)
option(DOWNLOAD_MODELS    "Download .pth model weights at build time" ON)

# === uninstall target ===
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
    @ONLY
)
add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
    COMMENT "Uninstalling cutie..."
)

# === CMake 模块路径 ===
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# === 必需依赖 ===
find_package(OpenCV REQUIRED)

# === linden_logger 依赖 ===
find_package(linden_logger QUIET)
if(NOT linden_logger_FOUND)
    add_subdirectory(
        "${CMAKE_CURRENT_SOURCE_DIR}/.ref_project/linden_logger"
        "${CMAKE_CURRENT_BINARY_DIR}/linden_logger"
    )
endif()

# === 条件依赖 ===
if(ENABLE_ONNXRUNTIME)
    # 优先尝试 config 模式（vcpkg 或标准安装提供 onnxruntimeConfig.cmake）
    find_package(onnxruntime CONFIG QUIET)
    if(onnxruntime_FOUND)
        message(STATUS "Found ONNXRuntime via config: onnxruntime::onnxruntime")

        # vcpkg 静态编译的 onnxruntime 会传递依赖 CUDA::cufft_static_nocallback，
        # 该 target 在 CMake < 3.23 不存在，需手动补充 stub。
        # 此处 find_package(CUDAToolkit) 仅在 config 模式下才需要，对非 vcpkg 无影响。
        find_package(CUDAToolkit QUIET)
        if(NOT TARGET CUDA::cufft_static_nocallback)
            add_library(CUDA::cufft_static_nocallback INTERFACE IMPORTED)
            if(TARGET CUDA::cufft_static)
                target_link_libraries(CUDA::cufft_static_nocallback INTERFACE CUDA::cufft_static)
            elseif(TARGET CUDA::cufft)
                target_link_libraries(CUDA::cufft_static_nocallback INTERFACE CUDA::cufft)
            endif()
        endif()

        # 优先从 target 属性读取 include 目录（标准安装会正确导出）
        get_target_property(_ort_inc onnxruntime::onnxruntime INTERFACE_INCLUDE_DIRECTORIES)
        if(_ort_inc)
            set(ONNXRuntime_INCLUDE_DIRS "${_ort_inc}")
        else()
            # vcpkg 静态构建不导出 INTERFACE_INCLUDE_DIRECTORIES，从安装树路径推导：
            # onnxruntime_DIR = <triplet>/share/onnxruntime → triplet root = ../../
            get_filename_component(_ort_prefix "${onnxruntime_DIR}/../.." ABSOLUTE)
            if(EXISTS "${_ort_prefix}/include/onnxruntime/onnxruntime_cxx_api.h")
                set(ONNXRuntime_INCLUDE_DIRS "${_ort_prefix}/include/onnxruntime")
            elseif(EXISTS "${_ort_prefix}/include/onnxruntime_cxx_api.h")
                set(ONNXRuntime_INCLUDE_DIRS "${_ort_prefix}/include")
            else()
                message(WARNING "ONNXRuntime headers not found under ${_ort_prefix}/include")
            endif()
        endif()
        set(ONNXRuntime_LIBRARIES onnxruntime::onnxruntime)
    else()
        # 回退：自定义 FindONNXRuntime.cmake 查找单体 libonnxruntime.so（非 vcpkg 环境）
        find_package(ONNXRuntime REQUIRED)
    endif()
endif()

if(ENABLE_TENSORRT)
    find_package(CUDA REQUIRED)
    find_package(TensorRT REQUIRED)
endif()

# === 配置头文件生成 ===
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/cutie/config.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/cutie/config.h"
    @ONLY
)

# === 源文件 ===
set(CUTIE_CORE_SOURCES
    src/core/object_manager.cpp
    src/core/memory_utils.cpp
    src/core/kv_memory_store.cpp
    src/core/memory_manager.cpp
    src/core/inference_core.cpp
    src/core/processor.cpp
    src/utils.cpp
)

set(CUTIE_ORT_SOURCES "")
if(ENABLE_ONNXRUNTIME)
    list(APPEND CUTIE_ORT_SOURCES
        src/ort/core/ort_handler.cpp
        src/ort/core/ort_utils.cpp
        src/ort/core/gpu_memory.cpp
        src/ort/core/gpu_tensor_ops.cpp
        src/ort/core/cuda_kernels.cu
        src/ort/cv/ort_cutie.cpp
    )
endif()

set(CUTIE_TRT_SOURCES "")
if(ENABLE_TENSORRT)
    list(APPEND CUTIE_TRT_SOURCES
        src/trt/core/trt_handler.cpp
        src/trt/core/trt_utils.cpp
        src/trt/cv/trt_cutie.cpp
    )
endif()

# === 库 ===
add_library(cutie SHARED
    ${CUTIE_CORE_SOURCES}
    ${CUTIE_ORT_SOURCES}
    ${CUTIE_TRT_SOURCES}
)

target_include_directories(cutie
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(cutie PUBLIC ${OpenCV_LIBS} linden::linden_logger)
target_link_libraries(cutie PRIVATE CUDA::cudart CUDA::cublas)

set_target_properties(cutie PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

if(ENABLE_ONNXRUNTIME)
    target_include_directories(cutie PRIVATE ${ONNXRuntime_INCLUDE_DIRS})
    target_link_libraries(cutie PRIVATE ${ONNXRuntime_LIBRARIES})
    target_compile_definitions(cutie PRIVATE ENABLE_ONNXRUNTIME)
endif()

if(ENABLE_TENSORRT)
    target_include_directories(cutie PRIVATE ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS})
    target_link_libraries(cutie PRIVATE
        ${TensorRT_LIBRARIES}
        ${CUDA_LIBRARIES}
    )
    target_compile_definitions(cutie PRIVATE ENABLE_TENSORRT)
endif()

# === 模型下载 custom target ===
# 使用 stamp 文件避免每次 build 都重复调用脚本
set(CUTIE_MODEL_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/share/model")
set(CUTIE_MODEL_STAMP      "${CUTIE_MODEL_SOURCE_DIR}/.download_complete")

if(DOWNLOAD_MODELS)
    add_custom_command(
        OUTPUT  "${CUTIE_MODEL_STAMP}"
        COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/share/scripts/download_models.sh"
        COMMAND ${CMAKE_COMMAND} -E touch "${CUTIE_MODEL_STAMP}"
        COMMENT "Checking / downloading missing Cutie .pth models to share/model/..."
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    add_custom_target(download_models
        DEPENDS "${CUTIE_MODEL_STAMP}"
    )
    add_dependencies(cutie download_models)
else()
    add_custom_target(download_models
        COMMENT "Model download skipped (DOWNLOAD_MODELS=OFF)"
    )
endif()

# === 示例 ===
if(BUILD_EXAMPLES)
    add_executable(demo_basic examples/demo_basic.cpp)
    target_link_libraries(demo_basic cutie ${OpenCV_LIBS})
    # 编译期嵌入两个候选模型目录，供 demo 运行时自动探测
    target_compile_definitions(demo_basic PRIVATE
        CUTIE_BUILD_MODEL_DIR="${CUTIE_MODEL_SOURCE_DIR}"
        CUTIE_INSTALL_MODEL_DIR="${CMAKE_INSTALL_PREFIX}/share/cutie/model"
    )
endif()

# === 安装 ===
include(GNUInstallDirs)

install(TARGETS cutie
    EXPORT cutieTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/cutie
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h"
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/include/cutie/config.h"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cutie
)

install(EXPORT cutieTargets
    FILE cutieTargets.cmake
    NAMESPACE cutie::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cutie
)

# === CMake 包配置文件生成 ===
include(CMakePackageConfigHelpers)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cutieConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cutieConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/cutie"
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/cutieConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/cutieConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/cutieConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/cutie"
)

# === 模型文件安装（.pth 权重 + 任意已导出的 .onnx 子模块）===
# OPTIONAL: 若 share/model/ 为空或不存在也不报错
install(DIRECTORY "${CUTIE_MODEL_SOURCE_DIR}/"
    DESTINATION "share/cutie/model"
    OPTIONAL
    FILES_MATCHING
        PATTERN "*.pth"
        PATTERN "*.onnx"
)
