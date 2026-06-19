cmake_minimum_required(VERSION 3.18)
project(kvision_inference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── Find ONNX Runtime ──
# Try pkg-config first (works on Arch/EndeavourOS)
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(ONNXRUNTIME QUIET libonnxruntime)
endif()

if(NOT ONNXRUNTIME_FOUND)
    # Fallback: manual search
    find_path(ONNXRUNTIME_INCLUDE_DIRS
        NAMES onnxruntime_cxx_api.h
        PATHS /usr/include /usr/local/include /usr/include/onnxruntime
        PATH_SUFFIXES onnxruntime onnxruntime/core/session
    )
    find_library(ONNXRUNTIME_LIBRARIES
        NAMES onnxruntime
        PATHS /usr/lib /usr/local/lib /usr/lib64
    )
    if(ONNXRUNTIME_INCLUDE_DIRS AND ONNXRUNTIME_LIBRARIES)
        set(ONNXRUNTIME_FOUND TRUE)
    else()
        message(FATAL_ERROR "ONNX Runtime not found. Install with: sudo pacman -S onnxruntime-cpu  (or onnxruntime-cuda)")
    endif()
endif()

message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIRS}")
message(STATUS "ONNX Runtime libraries: ${ONNXRUNTIME_LIBRARIES}")

# ── Inference library ──
add_library(kvision_infer STATIC
    kvision_inference.cpp
)
target_include_directories(kvision_infer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ONNXRUNTIME_INCLUDE_DIRS}
)
target_link_libraries(kvision_infer PUBLIC
    ${ONNXRUNTIME_LIBRARIES}
)

# ── Test binary ──
add_executable(test_inference test_inference.cpp)
target_link_libraries(test_inference PRIVATE kvision_infer)

# ── Install ──
install(TARGETS kvision_infer ARCHIVE DESTINATION lib)
install(FILES kvision_inference.h DESTINATION include)
