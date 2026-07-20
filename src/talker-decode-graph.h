#pragma once
// talker-decode-graph.h: one static batched talker decode graph, built
// per (attention window class, batch width N) and replayed directly on
// the backend. The kv window rounds up in 256 step spans; every input
// re-uploads before each replay: positions, kv rows, and masks carry
// the moving per-slot n_past, the frame code ids and the overlay rows
// carry each slot's previous frame, so nothing bakes and the allocator
// contract holds. The graph always covers KV sets [0, N).

#include "ggml-alloc.h"
#include "ggml.h"

#include <vector>

struct TalkerDecodeGraph {
    struct ggml_context *    ctx     = nullptr;
    struct ggml_cgraph *     gf      = nullptr;
    ggml_gallocr_t           galloc  = nullptr;
    struct ggml_tensor *     ids_in  = nullptr;  // [(1 + n_acoustic) * N] i32, group major: g * N + slot
    struct ggml_tensor *     overlay = nullptr;  // [hidden, N] f32
    struct ggml_tensor *     pos_in  = nullptr;  // [N] i32
    struct ggml_tensor *     rows_in = nullptr;  // [1, 1, N] i64
    struct ggml_tensor *     mask_in = nullptr;  // [n_kv_pad, 1, 1, N] f16
    struct ggml_tensor *     logits  = nullptr;  // [vocab, N] f32
    std::vector<ggml_fp16_t> mask;               // [n_kv_pad * N] f16
    std::vector<int32_t>     pos_data;           // [N] host staging
    std::vector<int64_t>     rows_data;          // [N] host staging
    int                      n_kv_pad = 0;       // window class width, 0 marks an empty slot
    int                      N        = 0;       // batch width this build covers
};

static void talker_decode_graph_free(TalkerDecodeGraph * tg) {
    if (tg->galloc) {
        ggml_gallocr_free(tg->galloc);
        tg->galloc = nullptr;
    }
    if (tg->ctx) {
        ggml_free(tg->ctx);
        tg->ctx = nullptr;
    }
    tg->gf      = nullptr;
    tg->ids_in  = nullptr;
    tg->overlay = nullptr;
    tg->pos_in  = nullptr;
    tg->rows_in = nullptr;
    tg->mask_in = nullptr;
    tg->logits  = nullptr;
    tg->mask.clear();
    tg->pos_data.clear();
    tg->rows_data.clear();
    tg->n_kv_pad = 0;
    tg->N        = 0;
}
