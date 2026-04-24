# cmake/export_trt_if_needed.cmake
# 检查 TensorRT 引擎文件是否存在，缺失则调用 export_tensorrt.py

# 以 pixel_encoder 引擎为代表检查是否已导出
set(_sentinel "${MODEL_DIR}/${MODEL_PREFIX}_N${NUM_OBJECTS}_pixel_encoder.engine")

if(NOT EXISTS "${_sentinel}")
    message(STATUS "Exporting TensorRT engines for ${MODEL_PREFIX} N=${NUM_OBJECTS}...")
    execute_process(
        COMMAND "${PYTHON_EXE}" "${EXPORT_SCRIPT}"
            --model-dir "${MODEL_DIR}"
            --model-prefix "${MODEL_PREFIX}"
            --num-objects "${NUM_OBJECTS}"
            --output "${MODEL_DIR}"
        RESULT_VARIABLE _ret
        OUTPUT_VARIABLE _output
        ERROR_VARIABLE _error
    )
    if(NOT _ret EQUAL 0)
        message(FATAL_ERROR "TensorRT export failed (exit code ${_ret})\nOutput: ${_output}\nError: ${_error}")
    endif()
    message(STATUS "TensorRT export completed successfully")
else()
    message(STATUS "TensorRT engines already exist, skipping export")
endif()
