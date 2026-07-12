#pragma once
// talker-decode-graph.h: one static talker decode graph, built once
// per attention window class (the kv window rounds up in 256 step
// spans) and replayed directly on the backend. Every input re-uploads
// before each replay: positions, kv row, and mask carry the moving
// n_past, the frame code ids and the overlay row carry the previous
// frame, so nothing bakes and the allocator contract holds.

#include "ggml-alloc.h"
#include "ggml.h"

#include <vector>

struct TalkerDecodeGraph {
    struct ggml_context *    ctx     = nullptr;
    struct ggml_cgraph *     gf      = nullptr;
    ggml_gallocr_t           galloc  = nullptr;
    struct ggml_tensor *     ids_in  = nullptr;  // [1 + n_acoustic] i32
    struct ggml_tensor *     overlay = nullptr;  // [hidden, 1] f32
    struct ggml_tensor *     pos_in  = nullptr;  // [1] i32
    struct ggml_tensor *     rows_in = nullptr;  // [1] i64
    struct ggml_tensor *     mask_in = nullptr;  // [n_kv_pad, 1] f16
    struct ggml_tensor *     logits  = nullptr;  // [vocab, 1] f32
    std::vector<ggml_fp16_t> mask;               // [n_kv_pad] f16
    int                      n_kv_pad = 0;       // window class width, 0 marks an empty slot
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
    tg->n_kv_pad = 0;
}
