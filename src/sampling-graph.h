#pragma once
// sampling-graph.h: the predictor sampling tail in standard ops, so
// the whole frame decodes on the backend without per step logits
// readbacks. Each tail applies the per step temperature, sorts the
// vocabulary in descending order, restricts it to the candidates, then
// draws one token where the cdf crosses the per step uniform u. Greedy
// slots upload u = 0, which lands the draw on the first (highest)
// entry.
//
// Two tail flavors share the inputs. The fixed tail keeps the first
// top_k candidates through a static view, the width baked at build.
// The full tail masks the whole vocabulary to the per slot top_k and
// top_p nucleus, mirroring sample_top_k_p in sampling.h: top_k keeps
// every logit at or above the k-th largest, top_p keeps an entry while
// the cumulative probability before it stays below top_p, the first
// entry always survives.
//
// Sampler inputs and the codes accumulator live in a caller owned
// persistent context, never in gallocr input buffers. The uniform
// draws stay on the host (philox depends only on seed and subsequence)
// and upload once per frame inside the state tensors.

#include "ggml-backend.h"
#include "ggml.h"
#include "philox.h"

#include <vector>

// Masked entries drop by this much, which zeroes them under softmax.
#define SAMPLER_MASK_DROP 1.0e30f

enum SamplerTail {
    SAMPLER_TAIL_FIXED,  // first top_k candidates through a static view
    SAMPLER_TAIL_FULL,   // per slot top_k and top_p masks over the vocabulary
};

struct SamplerInputs {
    struct ggml_tensor * state   = nullptr;  // [3, N, n_steps] f32, per slot (temperature, u, top_p)
    struct ggml_tensor * kidx    = nullptr;  // [N, n_steps] i32, per slot rank of the top_k cutoff
    struct ggml_tensor * codes   = nullptr;  // [N, n_codes] i32, row g holds code g of every slot
    int                  n_steps = 0;        // sampled codes per frame (semantic + acoustic)
    int                  N       = 0;
    int                  n_vocab = 0;        // sub-talker vocabulary width
    int                  top_k   = 0;        // candidate width of the fixed tail
};

// Per slot sampling controls uploaded once per frame.
struct SamplerSlot {
    float   temperature;  // <= 0 selects greedy
    int     top_k;        // <= 0 or >= n_vocab disables the cutoff
    float   top_p;        // >= 1 disables the nucleus
    int64_t seed;
    int64_t subseq_base;  // draw g uses subsequence subseq_base + 1 + g
};

// Create the sampler tensors inside pctx. The caller allocates pctx
// into a persistent backend buffer afterwards.
static inline void sampler_inputs_build(struct ggml_context * pctx,
                                        SamplerInputs *       sp,
                                        int                   N,
                                        int                   n_steps,
                                        int                   n_vocab,
                                        int                   top_k) {
    sp->n_steps = n_steps;
    sp->N       = N;
    sp->n_vocab = n_vocab;
    sp->top_k   = top_k;
    sp->state   = ggml_new_tensor_3d(pctx, GGML_TYPE_F32, 3, N, n_steps);
    sp->kidx    = ggml_new_tensor_2d(pctx, GGML_TYPE_I32, N, n_steps);
    sp->codes   = ggml_new_tensor_2d(pctx, GGML_TYPE_I32, N, n_steps + 1);
    ggml_set_name(sp->state, "sampler.state");
    ggml_set_name(sp->kidx, "sampler.kidx");
    ggml_set_name(sp->codes, "sampler.codes");
}

// Upload the per frame sampler state. Greedy slots carry temperature 1
// and u 0, which selects the argmax through the descending order. A
// disabled top_k points the cutoff at the last rank, a disabled top_p
// uploads 2 so no cumulative probability ever reaches it.
static inline void sampler_inputs_upload(SamplerInputs * sp, const SamplerSlot * slots, int N) {
    std::vector<float>   st((size_t) 3 * (size_t) N * (size_t) sp->n_steps);
    std::vector<int32_t> ki((size_t) N * (size_t) sp->n_steps);

    for (int g = 0; g < sp->n_steps; g++) {
        for (int i = 0; i < N; i++) {
            const SamplerSlot & s      = slots[i];
            const bool          greedy = s.temperature <= 0.0f;
            float               u      = 0.0f;
            if (!greedy) {
                philox_uniform_fill(s.seed, s.subseq_base + 1 + g, 0u, &u, 1);
            }
            const bool k_on = s.top_k > 0 && s.top_k < sp->n_vocab;
            const bool p_on = s.top_p > 0.0f && s.top_p < 1.0f;

            float * row = st.data() + ((size_t) g * (size_t) N + (size_t) i) * 3;
            row[0]      = greedy ? 1.0f : s.temperature;
            row[1]      = u;
            row[2]      = p_on ? s.top_p : 2.0f;

            ki[(size_t) g * (size_t) N + (size_t) i] = k_on ? s.top_k - 1 : sp->n_vocab - 1;
        }
    }
    ggml_backend_tensor_set(sp->state, st.data(), 0, st.size() * sizeof(float));
    ggml_backend_tensor_set(sp->kidx, ki.data(), 0, ki.size() * sizeof(int32_t));
}

// Drop the entries whose keep flag is 0: keep 1 leaves x untouched,
// keep 0 subtracts SAMPLER_MASK_DROP.
static inline struct ggml_tensor * sampler_mask_apply(struct ggml_context * gctx,
                                                      struct ggml_tensor *  x,
                                                      struct ggml_tensor *  keep) {
    return ggml_add(gctx, x, ggml_scale_bias(gctx, keep, SAMPLER_MASK_DROP, -SAMPLER_MASK_DROP));
}

// One sampling tail: reads this step's per slot controls from the
// state, draws one token id per slot from logits [n_vocab, N] and
// writes the N ids to row step_idx + 1 of the codes accumulator. Every
// gather batches over the slot dim through 3D get_rows.
static inline struct ggml_tensor * sampler_tail_build(struct ggml_context * gctx,
                                                      struct ggml_tensor *  logits,
                                                      SamplerInputs *       sp,
                                                      int                   step_idx,
                                                      enum SamplerTail      tail) {
    const int64_t n_vocab = logits->ne[0];
    const int64_t N       = logits->ne[1];

    const size_t         step_off = (size_t) step_idx * sp->state->nb[2];
    struct ggml_tensor * temp     = ggml_view_2d(gctx, sp->state, 1, N, sp->state->nb[1], step_off);
    struct ggml_tensor * u        = ggml_view_2d(gctx, sp->state, 1, N, sp->state->nb[1], step_off + sp->state->nb[0]);

    struct ggml_tensor * cur = ggml_div(gctx, logits, temp);

    // Descending order of the whole vocabulary: argsort guarantees the
    // order on every backend, and the layout is what makes the u = 0
    // draw an argmax.
    struct ggml_tensor * order = ggml_argsort(gctx, cur, GGML_SORT_ORDER_DESC);
    struct ggml_tensor * cur3d = ggml_reshape_3d(gctx, cur, 1, n_vocab, N);

    struct ggml_tensor * candidates;  // [n_cand, N] logits in descending order
    struct ggml_tensor * cand_ids;    // [n_cand, N] their token ids
    if (tail == SAMPLER_TAIL_FIXED) {
        cand_ids   = ggml_cont(gctx, ggml_view_2d(gctx, order, sp->top_k, N, order->nb[1], 0));
        candidates = ggml_reshape_2d(gctx, ggml_get_rows(gctx, cur3d, cand_ids), sp->top_k, N);
    } else {
        struct ggml_tensor * top_p =
            ggml_view_2d(gctx, sp->state, 1, N, sp->state->nb[1], step_off + 2 * sp->state->nb[0]);
        struct ggml_tensor * kidx =
            ggml_view_2d(gctx, sp->kidx, 1, N, sp->kidx->nb[0], (size_t) step_idx * sp->kidx->nb[1]);
        struct ggml_tensor * sorted = ggml_reshape_2d(gctx, ggml_get_rows(gctx, cur3d, order), n_vocab, N);

        // top_k: the k-th largest logit is the cutoff, an entry survives
        // when it is not strictly below it (ties at the cutoff survive).
        struct ggml_tensor * sorted3d = ggml_reshape_3d(gctx, sorted, 1, n_vocab, N);
        struct ggml_tensor * kth      = ggml_reshape_2d(gctx, ggml_get_rows(gctx, sorted3d, kidx), 1, N);
        struct ggml_tensor * below_k  = ggml_step(gctx, ggml_neg(gctx, ggml_sub(gctx, sorted, kth)));
        struct ggml_tensor * keep_k   = ggml_scale_bias(gctx, below_k, -1.0f, 1.0f);
        struct ggml_tensor * topk     = sampler_mask_apply(gctx, sorted, keep_k);

        // top_p: an entry survives while the cumulative probability
        // before it stays strictly below the nucleus bound.
        struct ggml_tensor * probs_k = ggml_soft_max(gctx, topk);
        struct ggml_tensor * before  = ggml_sub(gctx, ggml_cumsum(gctx, probs_k), probs_k);
        struct ggml_tensor * keep_p  = ggml_step(gctx, ggml_neg(gctx, ggml_sub(gctx, before, top_p)));

        cand_ids   = order;
        candidates = sampler_mask_apply(gctx, topk, keep_p);
    }

    // draw one token per slot: find where the cdf crosses u
    struct ggml_tensor * probs      = ggml_soft_max(gctx, candidates);
    struct ggml_tensor * cumsum     = ggml_cumsum(gctx, probs);
    struct ggml_tensor * cross_mask = ggml_step(gctx, ggml_sub(gctx, cumsum, u));
    struct ggml_tensor * idxf       = ggml_sum_rows(gctx, cross_mask);  // [1, N]
    struct ggml_tensor * rank =
        ggml_cast(gctx, ggml_scale_bias(gctx, idxf, -1.0f, (float) candidates->ne[0]), GGML_TYPE_I32);

    struct ggml_tensor * ids3d = ggml_reshape_3d(gctx, cand_ids, 1, cand_ids->ne[0], N);
    struct ggml_tensor * idx   = ggml_get_rows(gctx, ids3d, rank);  // [1, 1, N]

    struct ggml_tensor * ids = ggml_reshape_1d(gctx, idx, N);
    struct ggml_tensor * dst = ggml_view_1d(gctx, sp->codes, N, (size_t) (step_idx + 1) * sp->codes->nb[1]);
    return ggml_cpy(gctx, ids, dst);
}
