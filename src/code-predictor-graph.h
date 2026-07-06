#pragma once
// code-predictor-graph.h: one static predictor graph, built once at
// load, allocated once, and replayed with a 4 byte code id upload per
// call. Positions, kv rows, and the causal mask bake in at build time
// because the frame cache layout repeats identically: the prefill
// always writes rows 0..1 and step g always appends at row g + 1.

#include "ggml-alloc.h"
#include "ggml.h"

struct CodePredGraph {
    struct ggml_context * ctx    = nullptr;
    struct ggml_cgraph *  gf     = nullptr;
    ggml_gallocr_t        galloc = nullptr;
    struct ggml_tensor *  ids_in = nullptr;
    struct ggml_tensor *  logits = nullptr;
};

static void code_predictor_graph_free(CodePredGraph * cp) {
    if (cp->galloc) {
        ggml_gallocr_free(cp->galloc);
        cp->galloc = nullptr;
    }
    if (cp->ctx) {
        ggml_free(cp->ctx);
        cp->ctx = nullptr;
    }
    cp->gf     = nullptr;
    cp->ids_in = nullptr;
    cp->logits = nullptr;
}
