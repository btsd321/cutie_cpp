cmake_minimum_required(VERSION 3.18)

# === nvcc 自动探测 ===
# 必须在 project() 之前：CUDA 语言的编译器检测在 project() 时就会发生。
# 很多环境（含 pip 构建）不会把 nvcc 加入 PATH，此处从常见安装位置回退查找。
if(NOT CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX})
    find_program(_cutie_nvcc nvcc
        HINTS
            ENV CUDA_HOME
            ENV CUDA_PATH
            ENV CUDA_TOOLKIT_ROOT_DIR
            /usr/local/cuda
            /opt/cuda
        PATH_SUFFIXES bin
    )
    if(_cutie_nvcc)
        set(CMAKE_CUDA_COMPILER "${_cutie_nvcc}")
        message(STATUS "Auto-detected nvcc: ${_cutie_nvcc}")
    else()
        message(FATAL_ERROR
            "未找到 nvcc。请安装 CUDA Toolkit (>= 11.8) 并确保其在 PATH 中，"
            "或显式指定 -DCMAKE_CUDA_COMPILER=/path/to/nvcc。")
    endif()
endif()

project(cutie VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# === CUDA 设置 ===
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

find_package(CUDAToolkit REQUIRED)
if(CUDAToolkit_VERSION VERSION_LESS "11.8")
    message(FATAL_ERROR "CUDA 11.8 or higher is required, found ${CUDAToolkit_VERSION}")
endif()
message(STATUS "CUDA Toolkit: ${CUDAToolkit_VERSION}")

# === CUDA 架构选择 ===
# 可通过 -DCUTIE_CUDA_ARCHS="89;120" 覆盖，以缩短编译时间。
# 默认覆盖 Turing(75) / Ampere(80,86) / Ada(89) / Hopper(90)；
# Blackwell(120) 需要 CUDA >= 12.8，故按版本条件加入。
set(CUTIE_CUDA_ARCHS "" CACHE STRING "Target CUDA architectures (e.g. \"89;120\")")

if(CUTIE_CUDA_ARCHS)
    set(CMAKE_CUDA_ARCHITECTURES ${CUTIE_CUDA_ARCHS})
else()
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)
    if(NOT CUDAToolkit_VERSION VERSION_LESS "12.8")
        list(APPEND CMAKE_CUDA_ARCHITECTURES 120)
    else()
        message(WARNING
            "CUDA ${CUDAToolkit_VERSION} 不支持 sm_120 (Blackwell)，已跳过。"
            "如需支持 RTX 50 系列显卡，请升级到 CUDA >= 12.8。")
    endif()
endif()
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# === 后端选项（参考 lite.ai.toolkit 模式） ===
option(ENABLE_ONNXRUNTIME "Build with ONNXRuntime backend" ON)
option(ENABLE_TENSORRT    "Build with TensorRT backend"    OFF)
option(ENABLE_PYTHON      "Build Python bindings (cutie_cpp._core)" OFF)
option(BUILD_EXAMPLES     "Build example programs"          ON)
option(DOWNLOAD_MODELS    "Download .pth model weights at build time" ON)
option(EXPORT_ONNX        "Auto-export ONNX submodules at build time if missing" OFF)
option(EXPORT_TENSORRT    "Auto-export TensorRT engines at build time if missing"  OFF)
set(PYTHON_EXECUTABLE     "" CACHE STRING "Python interpreter path (default: auto-detect)")
set(NUM_OBJECTS           "2" CACHE STRING "Number of objects for TensorRT engine export")

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

# === Python 解释器查找 ===
if(PYTHON_EXECUTABLE)
    set(_python_exe "${PYTHON_EXECUTABLE}")
else()
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_FOUND)
        set(_python_exe "${Python3_EXECUTABLE}")
    else()
        set(_python_exe "python3")
    endif()
endif()
message(STATUS "Python interpreter: ${_python_exe}")

# === 必需依赖 ===
find_package(OpenCV REQUIRED)

# vcpkg 的 OpenCV（OpenCVModules.cmake）与静态 onnxruntime 都会传递依赖
# CUDA::cufft_static_nocallback，而该 target 在 CMake < 3.23 中不存在，需补 stub。
#
# 注意：此处**必须**放在 ONNXRuntime 查找之外。曾经它只建在 ORT 的 config 分支内，
# 导致改用 module 模式（官方预编译 ORT）时链接期报
# "cannot find -lCUDA::cufft_static_nocallback"——真正需要它的是 OpenCV，不是 ORT。
if(NOT TARGET CUDA::cufft_static_nocallback)
    add_library(CUDA::cufft_static_nocallback INTERFACE IMPORTED)
    if(TARGET CUDA::cufft_static)
        target_link_libraries(CUDA::cufft_static_nocallback INTERFACE CUDA::cufft_static)
    elseif(TARGET CUDA::cufft)
        target_link_libraries(CUDA::cufft_static_nocallback INTERFACE CUDA::cufft)
    endif()
endif()

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
    # vcpkg 的 onnxruntime 是**静态**库，ORT 核心会被链入 libcutie.so。
    # 其 CUDA provider 插件（运行时 dlopen）必须来自**同一份** vcpkg 构建，
    # 否则进程里会出现两个 ORT 实例，provider bridge 失效并段错误。
    # Python wheel 因此也打包 vcpkg 的 provider，见下方 SKBUILD 分支。
    #
    # 优先尝试 config 模式（vcpkg 或标准安装提供 onnxruntimeConfig.cmake）
    find_package(onnxruntime CONFIG QUIET)
    if(onnxruntime_FOUND)
        message(STATUS "Found ONNXRuntime via config: onnxruntime::onnxruntime")

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
    # 展开 TENSORRT_ROOT 中的 ~ 符号
    if(DEFINED TENSORRT_ROOT)
        string(REGEX REPLACE "^~" "$ENV{HOME}" TENSORRT_ROOT "${TENSORRT_ROOT}")
    endif()

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

# GPU 公共源文件（ORT 和 TRT 共享）
set(CUTIE_COMMON_GPU_SOURCES
    src/common/cuda_kernels.cu
    src/common/gpu_buffer.cpp
    src/common/gpu_memory.cpp
    src/common/gpu_tensor_ops.cpp
    src/common/gpu_image_preprocess.cu
    src/common/gpu_postprocess.cu
    src/common/gpu_mask_preprocess.cu
)

set(CUTIE_ORT_SOURCES "")
if(ENABLE_ONNXRUNTIME)
    list(APPEND CUTIE_ORT_SOURCES
        src/ort/core/ort_handler.cpp
        src/ort/core/ort_utils.cpp
        src/ort/cv/ort_cutie.cpp
    )
endif()

set(CUTIE_TRT_SOURCES "")
if(ENABLE_TENSORRT)
    list(APPEND CUTIE_TRT_SOURCES
        src/trt/core/trt_engine_builder.cpp
        src/trt/core/trt_handler.cpp
        src/trt/cv/trt_cutie.cpp
    )
endif()

# === 库 ===
add_library(cutie SHARED
    ${CUTIE_CORE_SOURCES}
    ${CUTIE_COMMON_GPU_SOURCES}
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
    # 公共头文件 (cutie/types.h) 暴露了 Ort::Value,因此 include 必须 PUBLIC
    target_include_directories(cutie PUBLIC ${ONNXRuntime_INCLUDE_DIRS})
    target_link_libraries(cutie PUBLIC ${ONNXRuntime_LIBRARIES})
    target_compile_definitions(cutie PUBLIC ENABLE_ONNXRUNTIME)
endif()

if(ENABLE_TENSORRT)
    target_include_directories(cutie PRIVATE ${TensorRT_INCLUDE_DIRS})
    target_link_libraries(cutie PRIVATE
        TensorRT::nvinfer
        TensorRT::nvonnxparser
    )
    target_compile_definitions(cutie PRIVATE ENABLE_TENSORRT)
endif()

# === 模型下载 custom target ===
set(CUTIE_MODEL_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/share/model")
set(CUTIE_MODEL_STAMP      "${CUTIE_MODEL_SOURCE_DIR}/.download_complete")
set(CUTIE_MODEL_PREFIX     "cutie-base-mega")

if(DOWNLOAD_MODELS)
    add_custom_command(
        OUTPUT  "${CUTIE_MODEL_STAMP}"
        COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/share/scripts/download_models.sh"
        COMMAND ${CMAKE_COMMAND} -E touch "${CUTIE_MODEL_STAMP}"
        COMMENT "Checking / downloading missing Cutie .pth models to share/model/..."
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    add_custom_target(download_models DEPENDS "${CUTIE_MODEL_STAMP}")
    add_dependencies(cutie download_models)
else()
    add_custom_target(download_models
        COMMENT "Model download skipped (DOWNLOAD_MODELS=OFF)"
    )
endif()

# === ONNX 导出 custom target ===
set(CUTIE_ONNX_STAMP       "${CUTIE_MODEL_SOURCE_DIR}/.onnx_export_complete")
set(CUTIE_EXPORT_ONNX_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/share/scripts/export_onnx.py")

if(EXPORT_ONNX)
    add_custom_command(
        OUTPUT  "${CUTIE_ONNX_STAMP}"
        COMMAND ${CMAKE_COMMAND}
            -DMODEL_DIR="${CUTIE_MODEL_SOURCE_DIR}"
            -DMODEL_PREFIX="${CUTIE_MODEL_PREFIX}"
            -DPYTHON_EXE="${_python_exe}"
            -DEXPORT_SCRIPT="${CUTIE_EXPORT_ONNX_SCRIPT}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/export_onnx_if_needed.cmake"
        COMMAND ${CMAKE_COMMAND} -E touch "${CUTIE_ONNX_STAMP}"
        DEPENDS download_models
        COMMENT "Checking / exporting ONNX submodules..."
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    add_custom_target(export_onnx DEPENDS "${CUTIE_ONNX_STAMP}")
    add_dependencies(cutie export_onnx)
else()
    add_custom_target(export_onnx
        COMMENT "ONNX export skipped (EXPORT_ONNX=OFF)"
    )
endif()

# === TensorRT 导出 custom target ===
if(ENABLE_TENSORRT)
    set(CUTIE_TRT_STAMP         "${CUTIE_MODEL_SOURCE_DIR}/.trt_export_complete")
    set(CUTIE_EXPORT_TRT_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/share/scripts/export_tensorrt.py")
    if(EXPORT_TENSORRT)
        add_custom_command(
            OUTPUT  "${CUTIE_TRT_STAMP}"
            COMMAND ${CMAKE_COMMAND}
                -DMODEL_DIR="${CUTIE_MODEL_SOURCE_DIR}"
                -DMODEL_PREFIX="${CUTIE_MODEL_PREFIX}"
                -DNUM_OBJECTS="${NUM_OBJECTS}"
                -DPYTHON_EXE="${_python_exe}"
                -DEXPORT_SCRIPT="${CUTIE_EXPORT_TRT_SCRIPT}"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/export_trt_if_needed.cmake"
            COMMAND ${CMAKE_COMMAND} -E touch "${CUTIE_TRT_STAMP}"
            DEPENDS export_onnx
            COMMENT "Checking / exporting TensorRT engines..."
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        )
        add_custom_target(export_tensorrt DEPENDS "${CUTIE_TRT_STAMP}")
        add_dependencies(cutie export_tensorrt)
    else()
        add_custom_target(export_tensorrt
            COMMENT "TensorRT export skipped (EXPORT_TENSORRT=OFF)"
        )
    endif()
endif()

# === 示例 ===
if(BUILD_EXAMPLES)
    find_package(argparse CONFIG REQUIRED)
    add_executable(demo_basic examples/demo_basic.cpp)
    target_link_libraries(demo_basic cutie ${OpenCV_LIBS} argparse::argparse)
    # 编译期嵌入两个候选模型目录，供 demo 运行时自动探测
    target_compile_definitions(demo_basic PRIVATE
        CUTIE_BUILD_MODEL_DIR="${CUTIE_MODEL_SOURCE_DIR}"
        CUTIE_INSTALL_MODEL_DIR="${CMAKE_INSTALL_PREFIX}/share/cutie_cpp/model"
    )
endif()

# === Python 绑定（pybind11）===
if(ENABLE_PYTHON)
    # scikit-build-core 会注入 Python_EXECUTABLE 等变量；独立构建时自行查找。
    find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
    find_package(pybind11 CONFIG REQUIRED)
    message(STATUS "Python bindings: enabled (pybind11 ${pybind11_VERSION})")

    pybind11_add_module(_core
        src/python/module.cpp
        src/python/py_config.cpp
        src/python/py_convert.cpp
        src/python/py_logger.cpp
        src/python/py_processor.cpp
    )

    target_link_libraries(_core PRIVATE cutie CUDA::cudart)
    target_include_directories(_core PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/python")

    if(SKBUILD)
        # wheel 构建：所有运行时库都装进 cutie_cpp 包目录，
        # 用 $ORIGIN 让它们互相在同目录解析，免设 LD_LIBRARY_PATH。
        #
        # 随包分发：libcutie.so（含静态链入的 OpenCV）+ _core 扩展
        #           + ONNXRuntime 动态库与 CUDA provider 插件
        # 不随包分发：CUDA 运行时（约 800 MB）与 cuDNN（约 600 MB），
        #             由 cutie_cpp/_loader.py 在加载失败时给出安装提示。
        set_target_properties(_core cutie PROPERTIES
            INSTALL_RPATH "$ORIGIN"
            BUILD_WITH_INSTALL_RPATH TRUE
        )
        install(TARGETS _core LIBRARY DESTINATION cutie_cpp)
        install(TARGETS cutie LIBRARY DESTINATION cutie_cpp)

        # ONNXRuntime 的 CUDA execution provider 插件。
        #
        # ORT 核心已静态链入 libcutie.so，但 provider 是运行时 dlopen 的插件，
        # 不是链接期依赖，必须显式随包安装，否则 wheel 装好后推理会报
        # "Failed to load library libonnxruntime_providers_cuda.so"。
        #
        # 必须用**与静态核心同一份 vcpkg 构建**的 provider：混用其它来源
        # （如 pip 的 onnxruntime-gpu）会让进程里出现两个 ORT 实例并段错误。
        set(CUTIE_ORT_PROVIDER_DIR "" CACHE STRING
            "含 libonnxruntime_providers_*.so 的目录（留空则从 vcpkg 自动推导）")

        set(_ort_provider_dir "${CUTIE_ORT_PROVIDER_DIR}")
        if(NOT _ort_provider_dir)
            # vcpkg 的 provider 与其它库同在 <triplet>/lib 下，
            # 由 onnxruntime_DIR（<triplet>/share/onnxruntime）向上两级推导。
            if(onnxruntime_DIR)
                get_filename_component(_vcpkg_triplet "${onnxruntime_DIR}/../.." ABSOLUTE)
                set(_ort_provider_dir "${_vcpkg_triplet}/lib")
            endif()
        endif()

        set(_ort_provider_libs "")
        foreach(_lib
            "libonnxruntime_providers_shared.so"
            "libonnxruntime_providers_cuda.so")
            if(EXISTS "${_ort_provider_dir}/${_lib}")
                list(APPEND _ort_provider_libs "${_ort_provider_dir}/${_lib}")
            endif()
        endforeach()

        if(NOT _ort_provider_libs)
            message(FATAL_ERROR
                "未找到 ONNXRuntime CUDA provider 插件"
                "（查找目录: ${_ort_provider_dir}）。\n"
                "wheel 缺少它将无法推理。请用 -DCUTIE_ORT_PROVIDER_DIR=<dir> 指定"
                "含 libonnxruntime_providers_cuda.so 的目录。")
        endif()

        install(FILES ${_ort_provider_libs} DESTINATION cutie_cpp)
        message(STATUS "Wheel 将打包 ONNXRuntime provider: ${_ort_provider_dir}")
    else()
        # 开发构建：扩展模块就地输出到源码包目录 python/cutie_cpp/，
        # 这样 PYTHONPATH=python 即可 import，无需 pip install 或软链接。
        # BUILD_RPATH 指向 build/ 以便运行时找到 libcutie.so。
        set_target_properties(_core PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python/cutie_cpp"
            BUILD_RPATH "${CMAKE_BINARY_DIR}"
        )
    endif()
endif()

# === 安装 ===
# wheel 构建（SKBUILD）只安装 Python 包本身，跳过 C++ 开发文件：
# 头文件、CMake 导出配置、.pth 权重都不属于 Python 轮子，
# 否则 wheel 会膨胀到近 1 GB（权重约 786 MB）。
if(NOT SKBUILD)

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
        # Find 模块与 cutieConfig.cmake 同目录安装，供 find_dependency 回退使用
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindTensorRT.cmake"
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindONNXRuntime.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/cutie"
    )

    # === 模型文件安装（.pth 权重 + 任意已导出的 .onnx/.engine 文件）===
    # OPTIONAL: 若 share/model/ 为空或不存在也不报错
    install(DIRECTORY "${CUTIE_MODEL_SOURCE_DIR}/"
        DESTINATION "share/cutie_cpp/model"
        OPTIONAL
        FILES_MATCHING
            PATTERN "*.pth"
            PATTERN "*.onnx"
            PATTERN "*.engine"
    )

endif()  # NOT SKBUILD
