# FindONNXRuntime.cmake
# Locate ONNX Runtime installation
#
# Sets:
#   ONNXRuntime_FOUND
#   ONNXRuntime_INCLUDE_DIRS
#   ONNXRuntime_LIBRARIES

# Try environment variable first, then common paths
find_path(ONNXRuntime_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    PATH_SUFFIXES include include/onnxruntime
    PATHS
        ${ONNXRUNTIME_ROOT}
        $ENV{ONNXRUNTIME_ROOT}
        /usr/local
        /usr
)

find_library(ONNXRuntime_LIBRARY
    NAMES onnxruntime
    PATH_SUFFIXES lib lib64
    PATHS
        ${ONNXRUNTIME_ROOT}
        $ENV{ONNXRUNTIME_ROOT}
        /usr/local
        /usr
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ONNXRuntime
    REQUIRED_VARS ONNXRuntime_LIBRARY ONNXRuntime_INCLUDE_DIR
)

if(ONNXRuntime_FOUND)
    set(ONNXRuntime_INCLUDE_DIRS ${ONNXRuntime_INCLUDE_DIR})
    set(ONNXRuntime_LIBRARIES ${ONNXRuntime_LIBRARY})
    mark_as_advanced(ONNXRuntime_INCLUDE_DIR ONNXRuntime_LIBRARY)
endif()
