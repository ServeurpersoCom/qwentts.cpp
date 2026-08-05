#pragma once
// sampling-graph.h: the predictor sampling tail in standard ops, so
// the whole frame decodes on the backend without per step logits
// readbacks. Each tail applies the per step temperature, keeps the
// top_k candidates, then draws one token where the cdf crosses the
// per step uniform u. top_k bakes from the generation defaults at
// build; nucleus filtering is not applied. Greedy slots upload u = 0,
// which lands the draw on each slot's first (highest) candidate.
//
// Sampler inputs and the codes accumulator live in a caller owned
// persistent context, never in gallocr input buffers. The uniform
// draws stay on the host (philox depends only on seed and subsequence)
// and upload once per frame inside the state tensor.

#include "ggml-backend.h"
#include "ggml.h"
#include "philox.h"

#include <vector>

struct SamplerInputs {
    struct ggml_tensor * state   = nullptr;  // [2, N, n_steps] f32, per slot (temperature, u)
    struct ggml_tensor * codes   = nullptr;  // [N, n_codes] i32, row g holds code g of every slot
    int                  n_steps = 0;        // sampled codes per frame (semantic + acoustic)
    int                  N       = 0;
    int                  top_k   = 0;        // candidate count baked into every tail
};

// Create the sampler tensors inside pctx. The caller allocates pctx
// into a persistent backend buffer afterwards.
static inline void sampler_inputs_build(struct ggml_context * pctx, SamplerInputs * sp, int N, int n_steps, int top_k) {
    sp->n_steps = n_steps;
    sp->N       = N;
    sp->top_k   = top_k;
    sp->state   = ggml_new_tensor_3d(pctx, GGML_TYPE_F32, 2, N, n_steps);
    sp->codes   = ggml_new_tensor_2d(pctx, GGML_TYPE_I32, N, n_steps + 1);
    ggml_set_name(sp->state, "sampler.state");
    ggml_set_name(sp->codes, "sampler.codes");
}

// Upload the per frame sampler state. Greedy slots (temperature <= 0)
// carry temperature 1 and u 0, which selects the argmax through the
// descending candidate order. subseq_base[i] indexes slot i's philox
// stream: draw g uses subsequence subseq_base[i] + 1 + g.
static inline void sampler_inputs_upload(SamplerInputs * sp,
                                         const float *   temperature,
                                         const int64_t * seed,
                                         const int64_t * subseq_base,
                                         int             N) {
    std::vector<float> st((size_t) 2 * (size_t) N * (size_t) sp->n_steps);

    for (int g = 0; g < sp->n_steps; g++) {
        for (int i = 0; i < N; i++) {
            const bool greedy = temperature[i] <= 0.0f;
            float      u      = 0.0f;
            if (!greedy) {
                philox_uniform_fill(seed[i], subseq_base[i] + 1 + g, 0u, &u, 1);
            }
            float * row = st.data() + ((size_t) g * (size_t) N + (size_t) i) * 2;
            row[0]      = greedy ? 1.0f : temperature[i];
            row[1]      = u;
        }
    }
    ggml_backend_tensor_set(sp->state, st.data(), 0, st.size() * sizeof(float));
}

// One sampling tail: reads this step's per slot (temperature, u) from
// the state, draws one token id per slot from logits [n_vocab, N] and
// writes the N ids to row step_idx + 1 of the codes accumulator. Every
// gather batches over the slot dim through 3D get_rows.
static inline struct ggml_tensor * sampler_tail_build(struct ggml_context * gctx,
                                                      struct ggml_tensor *  logits,
                                                      SamplerInputs *       sp,
                                                      int                   step_idx) {
    const int64_t n_vocab = logits->ne[0];
    const int64_t N       = logits->ne[1];

    struct ggml_tensor * temp =
        ggml_view_2d(gctx, sp->state, 1, N, sp->state->nb[1], (size_t) step_idx * sp->state->nb[2]);
    struct ggml_tensor * u =
        ggml_view_2d(gctx, sp->state, 1, N, sp->state->nb[1], sp->state->nb[0] + (size_t) step_idx * sp->state->nb[2]);

    struct ggml_tensor * cur = ggml_div(gctx, logits, temp);

    // keep each slot's top_k candidates, logits and ids in descending
    // order. argsort guarantees the order on every backend (top_k does
    // not), and the descending layout is what makes the u = 0 draw an
    // argmax.
    struct ggml_tensor * candidates = NULL;
    if (sp->top_k > 0 && sp->top_k < n_vocab) {
        struct ggml_tensor * order = ggml_argsort(gctx, cur, GGML_SORT_ORDER_DESC);
        struct ggml_tensor * idx =
            ggml_cont(gctx, ggml_view_2d(gctx, order, sp->top_k, N, order->nb[1], 0));  // [top_k, N] i32
        struct ggml_tensor * a3d = ggml_reshape_3d(gctx, cur, 1, n_vocab, N);
        cur                      = ggml_reshape_2d(gctx, ggml_get_rows(gctx, a3d, idx), sp->top_k, N);
        candidates               = idx;
    }

    // draw one token per slot: find where the cdf crosses u
    struct ggml_tensor * probs  = ggml_soft_max(gctx, cur);
    struct ggml_tensor * cumsum = ggml_cumsum(gctx, probs);

    struct ggml_tensor * diff       = ggml_sub(gctx, cumsum, u);
    struct ggml_tensor * cross_mask = ggml_step(gctx, diff);
    struct ggml_tensor * idxf       = ggml_sum_rows(gctx, cross_mask);  // [1, N]
    struct ggml_tensor * idx =
        ggml_cast(gctx, ggml_scale_bias(gctx, idxf, -1.0f, (float) cross_mask->ne[0]), GGML_TYPE_I32);

    if (candidates) {
        struct ggml_tensor * cand_3d = ggml_reshape_3d(gctx, candidates, 1, candidates->ne[0], N);
        idx                          = ggml_get_rows(gctx, cand_3d, idx);  // [1, 1, N]
    }

    struct ggml_tensor * ids = ggml_reshape_1d(gctx, idx, N);
    struct ggml_tensor * dst = ggml_view_1d(gctx, sp->codes, N, (size_t) (step_idx + 1) * sp->codes->nb[1]);
    return ggml_cpy(gctx, ids, dst);
}
