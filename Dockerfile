# syntax=docker/dockerfile:1
#
# Build context must have the `ggml` submodule checked out already
# (`git clone --recurse-submodules`, or `actions/checkout` with
# `submodules: recursive` in CI) -- this Dockerfile does not fetch it.
#
# Usage:
#   docker build --target cpu    -t qwentts.cpp:cpu    .
#   docker build --target cuda   -t qwentts.cpp:cuda   .
#   docker build --target vulkan -t qwentts.cpp:vulkan .
#
# Pascal (sm_61) and other GPUs older than this project's default
# distributed arch list (Turing+) need an explicit override, since
# `docker build` has no GPU device to auto-detect against:
#   docker build --target cuda -t qwentts.cpp:cuda \
#       --build-arg CMAKE_CUDA_ARCHITECTURES=61 .
# See docs/DOCKER.md for details.

ARG CUDA_BUILD_IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04
ARG CUDA_RUNTIME_IMAGE=nvidia/cuda:12.4.1-runtime-ubuntu22.04

# ---------------------------------------------------------------- CPU build
FROM ubuntu:22.04 AS build-cpu
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
        git ca-certificates cmake g++ make pkg-config libopenblas-dev \
    > /dev/null && rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY . .
RUN cmake -B build -DGGML_BLAS=ON -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build --config Release -j"$(nproc)"

FROM ubuntu:22.04 AS cpu
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
        libgomp1 libopenblas0 curl ca-certificates \
    > /dev/null && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=build-cpu /build/build/tts-server /build/build/qwen-tts /build/build/qwen-codec /build/build/*.so* ./
COPY docker/entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh
# Binaries are copied out of the build tree their RPATH points at, so the
# ggml shared libraries (copied alongside, above) need an explicit search path.
ENV LD_LIBRARY_PATH=/app
ENTRYPOINT ["./entrypoint.sh"]

# --------------------------------------------------------------- CUDA build
FROM ${CUDA_BUILD_IMAGE} AS build-cuda
ARG CMAKE_CUDA_ARCHITECTURES
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
        git ca-certificates cmake g++ make \
    > /dev/null && rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY . .
# `docker build` never has GPU device access, unlike `docker run --gpus`, so:
#  - CMAKE_CUDA_ARCHITECTURES must be set explicitly when targeting a GPU
#    generation outside this project's own default arch list (see
#    docs/DOCKER.md for the Pascal/sm_61 example); when unset here, CMake's
#    own project default (Turing and newer) is used unchanged.
#  - ggml's CUDA VMM pool allocator needs driver-API symbols (cuMemCreate,
#    cuMemMap, ...) at link time. The real libcuda.so isn't present without
#    a GPU, but the devel image ships a link-time-only stub at
#    lib64/stubs/libcuda.so for exactly this case; it isn't on the default
#    linker search path so both -L and -lcuda are needed explicitly.
RUN cmake -B build -DGGML_CUDA=ON \
        -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
        ${CMAKE_CUDA_ARCHITECTURES:+-DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}} \
        -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -lcuda" \
        -DCMAKE_SHARED_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -lcuda" \
        -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build --config Release -j"$(nproc)"

FROM ${CUDA_RUNTIME_IMAGE} AS cuda
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
        libgomp1 curl ca-certificates \
    > /dev/null && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=build-cuda /build/build/tts-server /build/build/qwen-tts /build/build/qwen-codec /build/build/*.so* ./
COPY docker/entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh
ENV LD_LIBRARY_PATH=/app
ENTRYPOINT ["./entrypoint.sh"]

# ------------------------------------------------------------- Vulkan build
# AMD/Intel GPUs (and NVIDIA via its Vulkan ICD). glslc (shader compiler) is
# only packaged by the LunarG SDK repo on Ubuntu 22.04, not apt's universe.
FROM ubuntu:22.04 AS build-vulkan
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
        git ca-certificates cmake g++ make wget gnupg \
    > /dev/null && rm -rf /var/lib/apt/lists/*
RUN wget -qO- https://packages.lunarg.com/lunarg-signing-key-pub.asc | gpg --dearmor -o /usr/share/keyrings/lunarg.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/lunarg.gpg] https://packages.lunarg.com/vulkan/1.3.296 jammy main" \
        > /etc/apt/sources.list.d/lunarg-vulkan.list && \
    apt-get update -qq && apt-get install -y -qq --no-install-recommends vulkan-sdk \
    > /dev/null && rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY . .
RUN cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build --config Release -j"$(nproc)"

FROM ubuntu:22.04 AS vulkan
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
        libgomp1 libvulkan1 mesa-vulkan-drivers curl ca-certificates \
    > /dev/null && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=build-vulkan /build/build/tts-server /build/build/qwen-tts /build/build/qwen-codec /build/build/*.so* ./
COPY docker/entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh
ENV LD_LIBRARY_PATH=/app
ENTRYPOINT ["./entrypoint.sh"]
