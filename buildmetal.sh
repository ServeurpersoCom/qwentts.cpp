#!/bin/bash
# Apple GPU build. The GGML flags are the macOS platform defaults, pinned
# so the recipe stands on its own. QWEN_SHARED=ON is the addition : it also
# emits libqwen.dylib, the shared target binding consumers (ctypes, bindgen,
# cgo) dlopen. The other recipes keep the static-only default.

rm -rf build
mkdir build
cd build

cmake .. -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DGGML_BLAS=ON -DQWEN_SHARED=ON
cmake --build . --config Release -j "$(getconf _NPROCESSORS_ONLN)"
