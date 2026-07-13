cmake_minimum_required(VERSION 3.22.1)
project(offlineai_native)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- llama.cpp is vendored, not committed to this repo (it's large and has
# its own license/release cadence). See README.md "Getting llama.cpp" for
# the one-time clone step. This expects it at app/src/main/cpp/llama.cpp ---
set(LLAMA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp)

if(NOT EXISTS ${LLAMA_DIR}/CMakeLists.txt)
    message(FATAL_ERROR
        "llama.cpp not found at ${LLAMA_DIR}. Run the clone step in "
        "README.md ('Getting llama.cpp') before building.")
endif()

# Build llama.cpp for on-device (CPU) inference. GPU delegate backends are
# a Phase 2 item (see HardwareProfiler notes in README).
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

add_subdirectory(${LLAMA_DIR} llama_build)

add_library(offlineai_native SHARED
    native-lib.cpp
)

target_include_directories(offlineai_native PRIVATE
    ${LLAMA_DIR}
    ${LLAMA_DIR}/include
    ${LLAMA_DIR}/ggml/include
)

target_link_libraries(offlineai_native
    llama
    android
    log
)
