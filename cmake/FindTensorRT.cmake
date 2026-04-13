# FindTensorRT.cmake
# Locate TensorRT installation
#
# Sets:
#   TensorRT_FOUND
#   TensorRT_INCLUDE_DIRS
#   TensorRT_LIBRARIES

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
    REQUIRED_VARS TensorRT_nvinfer_LIBRARY TensorRT_INCLUDE_DIR
)

if(TensorRT_FOUND)
    set(TensorRT_INCLUDE_DIRS ${TensorRT_INCLUDE_DIR})
    set(TensorRT_LIBRARIES
        ${TensorRT_nvinfer_LIBRARY}
        ${TensorRT_nvinfer_plugin_LIBRARY}
    )
    mark_as_advanced(
        TensorRT_INCLUDE_DIR
        TensorRT_nvinfer_LIBRARY
        TensorRT_nvinfer_plugin_LIBRARY
    )
endif()
