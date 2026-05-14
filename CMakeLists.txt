cmake_minimum_required(VERSION 3.14)
project(qwentts-ggml LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# version.h: embed git commit hash into all binaries.
# runs on every build, only rewrites if the hash changed.
set(VERSION_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/version.h")
add_custom_target(version ALL
    COMMAND "${CMAKE_COMMAND}" "-DSRC_DIR=${CMAKE_CURRENT_SOURCE_DIR}" "-DOUTPUT=${VERSION_OUTPUT}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/tools/version.cmake"
    BYPRODUCTS "${VERSION_OUTPUT}"
    COMMENT "Checking git version"
)

# pthread: required explicitly on older glibc (< 2.34) where libpthread
# is not merged into libc. Modern distros link it implicitly but aarch64
# and older x86_64 toolchains need the explicit dependency.
find_package(Threads REQUIRED)

# Suppress MSVC fopen/sprintf deprecation warnings, force UTF-8 source and
# execution charsets so non-ASCII string literals (CJK in text-chunker.h,
# prompt-builder.cpp) survive the compile without a BOM.
if(MSVC)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
    add_compile_options(/utf-8)
endif()

# Put executables and backend .so in the same directory (build root).
# Without this, ggml defaults to bin/ for .so but executables stay in root,
# and ggml_backend_load_all() can't find the backends at runtime.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Audio tokenizer tensor names can exceed default GGML_MAX_NAME of 64
add_compile_definitions(GGML_MAX_NAME=128)

# Harden: mark fread/fwrite/etc with warn_unused_result on all platforms
if(NOT MSVC)
    add_compile_definitions(_FORTIFY_SOURCE=2)
endif()

# CUDA architectures: cover Turing to Blackwell for distributed binaries.
# Users can override with -DCMAKE_CUDA_ARCHITECTURES=native for local builds.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND AND CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.8")
        set(CMAKE_CUDA_ARCHITECTURES "75-virtual;80-virtual;86-real;89-real;120a-real;121a-real")
    else()
        set(CMAKE_CUDA_ARCHITECTURES "75-virtual;80-virtual;86-real;89-real")
    endif()
endif()

# ggml as subdirectory, inherits GGML_CUDA, GGML_METAL, etc. from cmake flags
add_subdirectory(ggml)

# Shared compile options and ggml linkage
macro(link_ggml_backends target)
    target_include_directories(${target} PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_BINARY_DIR}
    )
    target_include_directories(${target} SYSTEM PRIVATE
        ${CMAKE_SOURCE_DIR}/ggml/include
    )
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /wd4100 /wd4505)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wshadow -Wconversion
                              -Wno-unused-parameter -Wno-unused-function -Wno-sign-conversion)
    endif()
    target_link_libraries(${target} PRIVATE ggml Threads::Threads)
    if(TARGET ggml-base)
        target_link_libraries(${target} PRIVATE ggml-base)
    endif()
    foreach(backend cpu blas cuda metal vulkan)
        if(TARGET ggml-${backend})
            get_target_property(CURRENT_BACKEND_TYPE ggml-${backend} TYPE)
            if (CURRENT_BACKEND_TYPE STREQUAL "MODULE_LIBRARY")
                # DL mode: backend is loaded at runtime via dlopen,
                # skip all link-time deps.
                continue()
            endif()
            target_link_libraries(${target} PRIVATE ggml-${backend})
        endif()
    endforeach()
    add_dependencies(${target} version)
endmacro()

# Core library always STATIC : the bundled CLI tools include
# pipeline-tts.h / pipeline-codec.h / backend.h directly for the tests
# paths that need every pipeline_* / backend_* symbol resolved without
# going through the public ABI. QWENTTS_STATIC is propagated PUBLIC :
# the lib's own .cpp files see it (so QWEN_API resolves to empty when
# compiling qwen.cpp on Windows), and every consumer that links
# qwen-core inherits it too (same effect on their side, no spurious
# dllimport on a static archive). The shared library for ABI consumers
# is a separate, opt-in target below.
add_library(qwen-core STATIC
    src/qwen.cpp
    src/pipeline-tts.cpp
    src/pipeline-codec.cpp
)
target_compile_definitions(qwen-core PUBLIC QWENTTS_STATIC)
target_include_directories(qwen-core PUBLIC src)
target_link_libraries(qwen-core PUBLIC ggml)
link_ggml_backends(qwen-core)

# Public shared library for ABI consumers (Python ctypes, Rust bindgen,
# Go cgo). Opt-in : -DQWENTTS_SHARED=ON at configure time. Exports only
# the QWEN_API-marked symbols ; every internal pipeline_* / backend_*
# stays hidden inside the .so. Intentionally a different target name
# from qwen-core so the static path used by the tools is never affected.
option(QWENTTS_SHARED "Build the shared qwen library for ABI consumers" OFF)
if(QWENTTS_SHARED)
    add_library(qwen SHARED
        src/qwen.cpp
        src/pipeline-tts.cpp
        src/pipeline-codec.cpp
    )
    target_compile_definitions(qwen PRIVATE QWENTTS_BUILD)
    set_target_properties(qwen PROPERTIES
        C_VISIBILITY_PRESET       hidden
        CXX_VISIBILITY_PRESET     hidden
        VISIBILITY_INLINES_HIDDEN ON
    )
    link_ggml_backends(qwen)
endif()

# quantize: GGUF requantizer (BF16 -> K-quants), shared policy with
# omnivoice.cpp / acestep.cpp.
add_executable(quantize tools/quantize.cpp)
link_ggml_backends(quantize)

# qwen-codec : standalone codec CLI (codes <-> WAV via 12Hz tokenizer)
add_executable(qwen-codec tools/qwen-codec.cpp)
target_link_libraries(qwen-codec PRIVATE qwen-core)
link_ggml_backends(qwen-codec)

# qwen-tts : full TTS pipeline (Talker LM + 12Hz tokenizer decoder).
add_executable(qwen-tts tools/qwen-tts.cpp)
target_link_libraries(qwen-tts PRIVATE qwen-core)
link_ggml_backends(qwen-tts)

# test-abi-c : pure C99 smoke test that locks in the public ABI contract.
# Compiles qwen.h with a C compiler under -Wall -Werror -pedantic and
# links against the static lib. The test never loads a model ; failure
# means the public API regressed. Built by default so a regression breaks
# the main build, not just an opt-in target.
add_executable(test-abi-c tests/abi-c.c)
set_target_properties(test-abi-c PROPERTIES
    C_STANDARD 99
    C_STANDARD_REQUIRED ON
    C_EXTENSIONS OFF
)
if(MSVC)
    target_compile_options(test-abi-c PRIVATE /W4 /WX)
else()
    target_compile_options(test-abi-c PRIVATE -Wall -Werror -pedantic)
endif()
target_include_directories(test-abi-c PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_BINARY_DIR}
)
target_link_libraries(test-abi-c PRIVATE qwen-core)
link_ggml_backends(test-abi-c)
