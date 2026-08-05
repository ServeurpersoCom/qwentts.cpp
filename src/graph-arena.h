#pragma once
// graph-arena.h: persistent no_alloc ggml context reused across graph
// builds. Rebuilding an identical graph into the same arena lands every
// node at a stable address, so the backend CUDA graph cache (keyed on
// the first node pointer) resolves to the same executable instance at
// every decode step instead of thrashing on fresh allocations.

#include "ggml.h"

#include <cstddef>
#include <cstdio>

struct GraphArena {
    struct ggml_context * ctx = nullptr;
};

// Allocate the arena once, sized for max_nodes tensors plus one graph.
static bool graph_arena_init(GraphArena * a, int max_nodes) {
    const size_t bytes =
        ggml_tensor_overhead() * (size_t) max_nodes + ggml_graph_overhead_custom((size_t) max_nodes, false);
    struct ggml_init_params gp = { bytes, NULL, true };
    a->ctx                     = ggml_init(gp);
    if (!a->ctx) {
        fprintf(stderr, "[GraphArena] FATAL: ggml_init failed (%zu bytes)\n", bytes);
        return false;
    }
    return true;
}

// Rewind the arena: the next build sequence reuses the same addresses.
static struct ggml_context * graph_arena_begin(GraphArena * a) {
    ggml_reset(a->ctx);
    return a->ctx;
}

static void graph_arena_free(GraphArena * a) {
    if (a->ctx) {
        ggml_free(a->ctx);
        a->ctx = NULL;
    }
}
