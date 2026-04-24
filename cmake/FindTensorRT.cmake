# FindTensorRT.cmake
# Locate TensorRT installation
#
# Sets:
#   TensorRT_FOUND
#   TensorRT_INCLUDE_DIRS
#   TensorRT_LIBRARIES
#   TensorRT::nvinfer (imported target)
#   TensorRT::nvonnxparser (imported target)

# 展开 ~ 符号
if(DEFINED ENV{TENSORRT_ROOT})
    file(TO_CMAKE_PATH "$ENV{TENSORRT_ROOT}" _tensorrt_root_env)
    string(REGEX REPLACE "^~" "$ENV{HOME}" _tensorrt_root_env "${_tensorrt_root_env}")
else()
    set(_tensorrt_root_env "")
endif()

if(DEFINED TENSORRT_ROOT)
    file(TO_CMAKE_PATH "${TENSORRT_ROOT}" _tensorrt_root_var)
    string(REGEX REPLACE "^~" "$ENV{HOME}" _tensorrt_root_var "${_tensorrt_root_var}")
else()
    set(_tensorrt_root_var "")
endif()

find_path(TensorRT_INCLUDE_DIR
    NAMES NvInfer.h
    PATH_SUFFIXES include
    PATHS
        ${_tensorrt_root_var}
        ${_tensorrt_root_env}
        /usr/local
        /usr
    NO_DEFAULT_PATH
)

# 如果上面没找到，使用默认搜索路径
if(NOT TensorRT_INCLUDE_DIR)
    find_path(TensorRT_INCLUDE_DIR
        NAMES NvInfer.h
        PATH_SUFFIXES include
    )
endif()

find_library(TensorRT_nvinfer_LIBRARY
    NAMES nvinfer
    PATH_SUFFIXES lib lib64
    PATHS
        ${_tensorrt_root_var}
        ${_tensorrt_root_env}
        /usr/local
        /usr
    NO_DEFAULT_PATH
)

if(NOT TensorRT_nvinfer_LIBRARY)
    find_library(TensorRT_nvinfer_LIBRARY
        NAMES nvinfer
        PATH_SUFFIXES lib lib64
    )
endif()

find_library(TensorRT_nvonnxparser_LIBRARY
    NAMES nvonnxparser
    PATH_SUFFIXES lib lib64
    PATHS
        ${_tensorrt_root_var}
        ${_tensorrt_root_env}
        /usr/local
        /usr
    NO_DEFAULT_PATH
)

if(NOT TensorRT_nvonnxparser_LIBRARY)
    find_library(TensorRT_nvonnxparser_LIBRARY
        NAMES nvonnxparser
        PATH_SUFFIXES lib lib64
    )
endif()

find_library(TensorRT_nvinfer_plugin_LIBRARY
    NAMES nvinfer_plugin
    PATH_SUFFIXES lib lib64
    PATHS
        ${_tensorrt_root_var}
        ${_tensorrt_root_env}
        /usr/local
        /usr
    NO_DEFAULT_PATH
)

if(NOT TensorRT_nvinfer_plugin_LIBRARY)
    find_library(TensorRT_nvinfer_plugin_LIBRARY
        NAMES nvinfer_plugin
        PATH_SUFFIXES lib lib64
    )
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorRT
    REQUIRED_VARS TensorRT_nvinfer_LIBRARY TensorRT_nvonnxparser_LIBRARY TensorRT_INCLUDE_DIR
)

if(TensorRT_FOUND)
    set(TensorRT_INCLUDE_DIRS ${TensorRT_INCLUDE_DIR})
    set(TensorRT_LIBRARIES
        ${TensorRT_nvinfer_LIBRARY}
        ${TensorRT_nvonnxparser_LIBRARY}
        ${TensorRT_nvinfer_plugin_LIBRARY}
    )

    # 创建 imported targets
    if(NOT TARGET TensorRT::nvinfer)
        add_library(TensorRT::nvinfer UNKNOWN IMPORTED)
        set_target_properties(TensorRT::nvinfer PROPERTIES
            IMPORTED_LOCATION "${TensorRT_nvinfer_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}")
    endif()

    if(NOT TARGET TensorRT::nvonnxparser)
        add_library(TensorRT::nvonnxparser UNKNOWN IMPORTED)
        set_target_properties(TensorRT::nvonnxparser PROPERTIES
            IMPORTED_LOCATION "${TensorRT_nvonnxparser_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}")
    endif()

    message(STATUS "Found TensorRT: ${TensorRT_INCLUDE_DIR}")
    message(STATUS "  - nvinfer: ${TensorRT_nvinfer_LIBRARY}")
    message(STATUS "  - nvonnxparser: ${TensorRT_nvonnxparser_LIBRARY}")

    mark_as_advanced(
        TensorRT_INCLUDE_DIR
        TensorRT_nvinfer_LIBRARY
        TensorRT_nvonnxparser_LIBRARY
        TensorRT_nvinfer_plugin_LIBRARY
    )
endif()
