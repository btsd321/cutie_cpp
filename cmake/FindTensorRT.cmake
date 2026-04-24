# FindTensorRT.cmake
# Locate TensorRT installation
#
# Sets:
#   TensorRT_FOUND
#   TensorRT_INCLUDE_DIRS
#   TensorRT_LIBRARIES
#   TensorRT::nvinfer (imported target)
#   TensorRT::nvonnxparser (imported target)

find_path(TensorRT_INCLUDE_DIR
    NAMES NvInfer.h
    PATH_SUFFIXES include
    PATHS
        ${TENSORRT_ROOT}
        $ENV{TENSORRT_ROOT}
        /usr/local
        /usr
)

find_library(TensorRT_nvinfer_LIBRARY
    NAMES nvinfer
    PATH_SUFFIXES lib lib64
    PATHS
        ${TENSORRT_ROOT}
        $ENV{TENSORRT_ROOT}
        /usr/local
        /usr
)

find_library(TensorRT_nvonnxparser_LIBRARY
    NAMES nvonnxparser
    PATH_SUFFIXES lib lib64
    PATHS
        ${TENSORRT_ROOT}
        $ENV{TENSORRT_ROOT}
        /usr/local
        /usr
)

find_library(TensorRT_nvinfer_plugin_LIBRARY
    NAMES nvinfer_plugin
    PATH_SUFFIXES lib lib64
    PATHS
        ${TENSORRT_ROOT}
        $ENV{TENSORRT_ROOT}
        /usr/local
        /usr
)

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
