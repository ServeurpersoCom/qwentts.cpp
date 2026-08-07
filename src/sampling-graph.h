#pragma once
// sampling-graph.h: the predictor sampling tail in standard ops, so
// the whole frame decodes on the backend without per step logits
// readbacks. Each tail applies the per step temperature, masks
// everything below the top_k cutoff, then walks the cdf in vocabulary
// order and draws the first token that crosses u * total, which is the
// multinomial sample_top_k_p in sampling.h expressed as a graph: for a
// given philox u both pick the same token, so the sub-talker sequence
// stays aligned with the reference the cossim harnesses check against.
// top_k bakes from the generation defaults at build; nucleus filtering
// is not applied. Greedy slots upload u = 0, which pins the cutoff at
// rank 0 so only the argmax survives the mask and any draw lands on it.
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
// carry temperature 1 and u 0, which the tail reads as the argmax
// request. subseq_base[i] indexes slot i's philox stream: draw g uses
// subsequence subseq_base[i] + 1 + g.
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
    // u feeds unary ops and a broadcast multiply, both of which want a
    // packed row: the state view strides over the (temperature, u)
    // pairs, so it materialises once here.
    struct ggml_tensor * u = ggml_cont(gctx, ggml_view_2d(gctx, sp->state, 1, N, sp->state->nb[1],
                                                          sp->state->nb[0] + (size_t) step_idx * sp->state->nb[2]));

    struct ggml_tensor * cur = ggml_div(gctx, logits, temp);

    // Rank ordered token ids, highest logit first. argsort guarantees
    // that order on every backend (top_k does not), and only two ranks
    // are ever read back: rank 0 is the row max, rank k - 1 is the
    // top_k cutoff. A k outside [1, n_vocab) falls back to the last
    // rank, which keeps the whole row.
    struct ggml_tensor * order = ggml_argsort(gctx, cur, GGML_SORT_ORDER_DESC);
    struct ggml_tensor * cur3d = ggml_reshape_3d(gctx, cur, 1, n_vocab, N);
    const int64_t k_rank = (sp->top_k > 0 && (int64_t) sp->top_k < n_vocab) ? (int64_t) sp->top_k - 1 : n_vocab - 1;

    // Logit sitting at a given rank, per slot: gather the token id out
    // of the rank order, then that token's logit.
    auto rank_logit = [&](int64_t rank) {
        struct ggml_tensor * id =
            ggml_cont(gctx, ggml_view_2d(gctx, order, 1, N, order->nb[1], (size_t) rank * order->nb[0]));
        return ggml_reshape_2d(gctx, ggml_get_rows(gctx, cur3d, id), 1, N);  // [1, N]
    };
    struct ggml_tensor * max_logit = rank_logit(0);
    struct ggml_tensor * kth_logit = rank_logit(k_rank);

    // Greedy slots upload u = 0 and no philox draw, so the cutoff moves
    // up to the row max and the mask keeps the argmax alone; every
    // other slot cuts at the top_k rank.
    struct ggml_tensor * stochastic = ggml_step(gctx, u);  // 1 where u > 0
    struct ggml_tensor * cutoff =
        ggml_add(gctx, max_logit, ggml_mul(gctx, stochastic, ggml_sub(gctx, kth_logit, max_logit)));

    // 1 on the kept tokens: step is a strict compare, so the mask is
    // built from its complement to keep the ties at the cutoff, the
    // same way the host sampler masks logits below the threshold.
    struct ggml_tensor * dropped = ggml_step(gctx, ggml_neg(gctx, ggml_sub(gctx, cur, cutoff)));
    struct ggml_tensor * keep    = ggml_scale_bias(gctx, dropped, -1.0f, 1.0f);

    // Draw in vocabulary order over the unnormalised exponentials, the
    // multinomial of sampling.h: the token where the running sum first
    // passes u * total. Masked tokens contribute nothing, so the walk
    // can never land on one.
    struct ggml_tensor * weights = ggml_mul(gctx, ggml_exp(gctx, ggml_sub(gctx, cur, max_logit)), keep);
    struct ggml_tensor * cdf     = ggml_cumsum(gctx, weights);
    struct ggml_tensor * total =
        ggml_cont(gctx, ggml_view_2d(gctx, cdf, 1, N, cdf->nb[1], (size_t) (n_vocab - 1) * cdf->nb[0]));
    struct ggml_tensor * cross_mask = ggml_step(gctx, ggml_sub(gctx, cdf, ggml_mul(gctx, total, u)));
    struct ggml_tensor * idxf       = ggml_sum_rows(gctx, cross_mask);  // [1, N]
    struct ggml_tensor * idx = ggml_cast(gctx, ggml_scale_bias(gctx, idxf, -1.0f, (float) n_vocab), GGML_TYPE_I32);

    struct ggml_tensor * ids = ggml_reshape_1d(gctx, idx, N);
    struct ggml_tensor * dst = ggml_view_1d(gctx, sp->codes, N, (size_t) (step_idx + 1) * sp->codes->nb[1]);
    return ggml_cpy(gctx, ids, dst);
}
