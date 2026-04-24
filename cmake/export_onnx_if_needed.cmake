# cmake/export_onnx_if_needed.cmake
# 检查 6 个 ONNX 子模块文件是否存在，缺失则调用 export_onnx.py

set(_submodules pixel_encoder key_projection mask_encoder pixel_fuser object_transformer mask_decoder)
set(_all_exist TRUE)

foreach(_mod ${_submodules})
    if(NOT EXISTS "${MODEL_DIR}/${MODEL_PREFIX}_${_mod}.onnx")
        set(_all_exist FALSE)
        message(STATUS "Missing ONNX file: ${MODEL_PREFIX}_${_mod}.onnx")
        break()
    endif()
endforeach()

if(NOT _all_exist)
    message(STATUS "Exporting ONNX submodules for ${MODEL_PREFIX}...")
    execute_process(
        COMMAND "${PYTHON_EXE}" "${EXPORT_SCRIPT}"
            --variant base
            --weights "${MODEL_DIR}/${MODEL_PREFIX}.pth"
            --output "${MODEL_DIR}"
        RESULT_VARIABLE _ret
        OUTPUT_VARIABLE _output
        ERROR_VARIABLE _error
    )
    if(NOT _ret EQUAL 0)
        message(FATAL_ERROR "ONNX export failed (exit code ${_ret})\nOutput: ${_output}\nError: ${_error}")
    endif()
    message(STATUS "ONNX export completed successfully")
else()
    message(STATUS "All ONNX submodules already exist, skipping export")
endif()
