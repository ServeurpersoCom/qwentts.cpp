#!/bin/bash

rm -rf build
mkdir build
cd build

cmake .. -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc
cmake --build . --config Release -j "$(nproc)"
