#pragma once
// code-predictor-forward.h: run the 5-layer Qwen3 code predictor over a
// growing context to produce the 15 acoustic codes of one audio frame
// for every active slot in one batched pass, KV cached per slot.
//
// Input:
//   hidden_bridge [hidden, max_batch] f32 -- persistent backend tensor
//                                        holding the talker last
//                                        position hidden per slot (post
//                                        final norm), written on device
//                                        by the talker graph and read
//                                        here as a graph leaf
//   c0[N]                             -- semantic codes sampled from the
//                                        Talker codec_head (codebook 0),
//                                        one per slot
// Output:
//   codes[N * 16]                     -- per slot [c0, c1, ..., c15],
//                                        slot major, ready for decode
//                                        through the codec
//
// The predictor cache is local to a single frame and holds one set per
// slot: the prefill writes the first two positions (talker_hidden +
// embed(c0)) of every slot, then 14 batched single-token steps follow.
// Every slot runs the same sub-step sequence every frame, so the batch
// stays in perfect lockstep and the graphs bake positions, kv rows and
// masks at build time.
//
// Architecture mirrors the Talker block, only differences are:
//   - 5 layers instead of 28
//   - plain 1D RoPE (no multimodal sections)
//   - one private embedding table and one private linear head per
//     acoustic codebook (1..15)
//
// Graph metadata lives in a caller owned static frame graph per batch
// width N, built lazily on the first frame at a given N, then replayed
// directly on the backend with an N * 4 byte code id upload per call.

#include "code-predictor-graph.h"
#include "code-predictor-weights.h"
#include "debug.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "kv-cache.h"
#include "qt-error.h"
#include "sampling-graph.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

struct CodePredictorOutput {
    // Per slot sixteen codes, slot major: codes[slot * 16 + g] holds c0
    // from the talker plus c1..c15 from the predictor.
    std::vector<int32_t> codes;
};

// Manual F32 attention chain for the code predictor block. Same shape
// contract as talker_attn_f32: q [hd, T, n_q_heads, N], k/v
// [hd, T_full, n_kv, N], output [hd, n_q_heads, T, N]; the mul_mat
// broadcasts over dims 2 and 3. Used when use_flash_attn is false.
static struct ggml_tensor * code_predictor_attn_f32(struct ggml_context * ctx,
                                                    struct ggml_tensor *  q,
                                                    struct ggml_tensor *  k,
                                                    struct ggml_tensor *  v,
                                                    struct ggml_tensor *  mask,
                                                    float                 scale) {
    struct ggml_tensor * scores = ggml_mul_mat(ctx, k, q);
    scores                      = ggml_soft_max_ext(ctx, scores, mask, scale, 0.0f);
    struct ggml_tensor * vt     = ggml_cont(ctx, ggml_transpose(ctx, v));
    struct ggml_tensor * out    = ggml_mul_mat(ctx, vt, scores);
    return ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));
}

// Node budget for one predictor graph, same accounting as the talker.
static int code_predictor_graph_max_nodes(int n_layers) {
    return 48 * n_layers + 96;
}

// Node budget for the unrolled frame graph: the prefill and every
// acoustic step chained in one cgraph.
static int code_predictor_frame_graph_max_nodes(int n_layers, int n_passes) {
    return n_passes * code_predictor_graph_max_nodes(n_layers);
}

// One batched Qwen3 decoder block, KV cached over sets [0, N). x holds
// the N slots' token columns flattened slot major: column j = n * T + t
// is position t of slot n. K and V for the fresh positions are written
// into each slot's set at the rows carried by kv_rows; the attention
// reads the fixed [0, n_kv_pad) window per set with the mask carrying
// neg inf beyond n_past + T. Returns the layer output [hidden, T * N].
// use_flash_attn and clamp_fp16 follow the same contract as in
// talker-forward.h.
static struct ggml_tensor * code_predictor_layer_forward(struct ggml_context *        ctx,
                                                         const CodePredictorWeights * cw,
                                                         const TalkerLayer &          layer,
                                                         struct ggml_tensor *         x,
                                                         struct ggml_tensor *         positions,
                                                         struct ggml_tensor *         mask,
                                                         struct ggml_tensor *         kv_rows,
                                                         struct ggml_tensor *         k4,
                                                         struct ggml_tensor *         v4,
                                                         int                          T,
                                                         int                          N,
                                                         int                          n_kv_pad,
                                                         bool                         use_flash_attn,
                                                         bool                         clamp_fp16,
                                                         struct ggml_cgraph *         gf) {
    const int   n_q_heads = cw->num_attention_heads;
    const int   n_kv      = cw->num_key_value_heads;
    const int   hd        = cw->head_dim;
    const float eps       = cw->rms_norm_eps;
    const int   TN        = T * N;

    struct ggml_tensor * h = ggml_rms_norm(ctx, x, eps);
    h                      = ggml_mul(ctx, h, layer.input_norm_w);

    struct ggml_tensor * q = ggml_mul_mat(ctx, layer.attn.q_proj_w, h);
    struct ggml_tensor * k = ggml_mul_mat(ctx, layer.attn.k_proj_w, h);
    struct ggml_tensor * v = ggml_mul_mat(ctx, layer.attn.v_proj_w, h);

    q = ggml_reshape_3d(ctx, q, hd, n_q_heads, TN);
    k = ggml_reshape_3d(ctx, k, hd, n_kv, TN);
    v = ggml_reshape_3d(ctx, v, hd, n_kv, TN);

    q = ggml_rms_norm(ctx, q, eps);
    q = ggml_mul(ctx, q, layer.attn.q_norm_w);
    k = ggml_rms_norm(ctx, k, eps);
    k = ggml_mul(ctx, k, layer.attn.k_norm_w);

    // RoPE over the flattened token axis: positions [T * N] repeat the
    // same in-frame offsets for every slot.
    q = ggml_rope_ext(ctx, q, positions, NULL, hd, GGML_ROPE_TYPE_NEOX, 0, cw->rope_theta, 1.0f, 0.0f, 1.0f, 0.0f,
                      0.0f);
    k = ggml_rope_ext(ctx, k, positions, NULL, hd, GGML_ROPE_TYPE_NEOX, 0, cw->rope_theta, 1.0f, 0.0f, 1.0f, 0.0f,
                      0.0f);

    // Write the fresh positions into each slot's set via set_rows:
    // [hd, heads, T*N] unflattens to [hd, heads, T, N], permutes to
    // [hd, T, heads, N] and lands at the kv_rows [T, 1, N] destinations,
    // broadcast across the n_kv head dim.
    struct ggml_tensor * k_perm =
        ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, k, hd, n_kv, T, N), 0, 2, 1, 3));
    struct ggml_tensor * v_perm =
        ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, v, hd, n_kv, T, N), 0, 2, 1, 3));

    struct ggml_tensor * k_sets = ggml_view_4d(ctx, k4, hd, k4->ne[1], n_kv, N, k4->nb[1], k4->nb[2], k4->nb[3], 0);
    struct ggml_tensor * v_sets = ggml_view_4d(ctx, v4, hd, v4->ne[1], n_kv, N, v4->nb[1], v4->nb[2], v4->nb[3], 0);

    ggml_build_forward_expand(gf, ggml_set_rows(ctx, k_sets, k_perm, kv_rows));
    ggml_build_forward_expand(gf, ggml_set_rows(ctx, v_sets, v_perm, kv_rows));

    struct ggml_tensor * k_full = ggml_view_4d(ctx, k4, hd, n_kv_pad, n_kv, N, k4->nb[1], k4->nb[2], k4->nb[3], 0);
    struct ggml_tensor * v_full = ggml_view_4d(ctx, v4, hd, n_kv_pad, n_kv, N, v4->nb[1], v4->nb[2], v4->nb[3], 0);

    // Q [hd, n_q_heads, T, N] -> [hd, T, n_q_heads, N] for
    // flash_attn_ext, taken as a view like the talker path does.
    struct ggml_tensor * q_p = ggml_permute(ctx, ggml_reshape_4d(ctx, q, hd, n_q_heads, T, N), 0, 2, 1, 3);

    // Clamp V before attention when clamp_fp16 is set, same rationale
    // as the talker block: sub Ampere CUDA tensor cores accumulate in
    // FP16 and a V projection overflow corrupts everything downstream.
    if (clamp_fp16) {
        v_full = ggml_clamp(ctx, v_full, -65504.0f, 65504.0f);
    }

    // Attention: fused flash kernel or manual F32 chain.
    float                scale = 1.0f / sqrtf((float) hd);
    struct ggml_tensor * attn;
    if (use_flash_attn) {
        attn = ggml_flash_attn_ext(ctx, q_p, k_full, v_full, mask, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
    } else {
        attn = code_predictor_attn_f32(ctx, q_p, k_full, v_full, mask, scale);
    }

    // [hd, n_q_heads, T, N] -> [n_q_heads*hd, T*N], flatten heads for
    // o_proj, token order matching x.
    attn = ggml_reshape_2d(ctx, attn, n_q_heads * hd, TN);

    struct ggml_tensor * o = ggml_mul_mat(ctx, layer.attn.o_proj_w, attn);
    x                      = ggml_add(ctx, x, o);
    if (clamp_fp16) {
        x = ggml_clamp(ctx, x, -65504.0f, 65504.0f);
    }

    struct ggml_tensor * h2 = ggml_rms_norm(ctx, x, eps);
    h2                      = ggml_mul(ctx, h2, layer.post_attn_norm_w);

    // gate and up mul_mats adjacent to the GLU node: the CUDA backend
    // fuses the three into one kernel (ggml_cuda_should_fuse_mul_mat).
    struct ggml_tensor * gate = ggml_mul_mat(ctx, layer.mlp.gate_proj_w, h2);
    struct ggml_tensor * up   = ggml_mul_mat(ctx, layer.mlp.up_proj_w, h2);
    struct ggml_tensor * gu   = ggml_swiglu_split(ctx, gate, up);
    struct ggml_tensor * mlp  = ggml_mul_mat(ctx, layer.mlp.down_proj_w, gu);

    x = ggml_add(ctx, x, mlp);
    if (clamp_fp16) {
        x = ggml_clamp(ctx, x, -65504.0f, 65504.0f);
    }
    return x;
}

// Baked inputs of one predictor pass: positions, kv rows and the
// causal mask upload once after allocation, then every replay reuses
// them.
struct CodePredPassBake {
    struct ggml_tensor * pos;
    struct ggml_tensor * rows;
    struct ggml_tensor * mask;
    int                  T;
    int                  n_past;
};

// Upload the baked inputs of every recorded pass. n_kv_pad is the
// constant mask width shared by all flavors.
static void code_predictor_bake_upload(const std::vector<CodePredPassBake> & bake, int N, int n_kv_pad) {
    for (const CodePredPassBake & b : bake) {
        std::vector<int32_t> pos((size_t) b.T * (size_t) N);
        for (int n = 0; n < N; n++) {
            for (int t = 0; t < b.T; t++) {
                pos[(size_t) n * (size_t) b.T + (size_t) t] = b.n_past + t;
            }
        }
        ggml_backend_tensor_set(b.pos, pos.data(), 0, pos.size() * sizeof(int32_t));

        std::vector<int64_t> rows((size_t) b.T * (size_t) N);
        for (int n = 0; n < N; n++) {
            for (int t = 0; t < b.T; t++) {
                rows[(size_t) n * (size_t) b.T + (size_t) t] = (int64_t) (b.n_past + t);
            }
        }
        ggml_backend_tensor_set(b.rows, rows.data(), 0, rows.size() * sizeof(int64_t));

        std::vector<ggml_fp16_t> mask((size_t) n_kv_pad * (size_t) b.T * (size_t) N);
        const ggml_fp16_t        zero    = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t        neg_inf = ggml_fp32_to_fp16(-INFINITY);
        for (size_t i = 0; i < mask.size(); i++) {
            mask[i] = neg_inf;
        }
        for (int n = 0; n < N; n++) {
            for (int q = 0; q < b.T; q++) {
                const int q_pos = b.n_past + q;
                for (int k = 0; k <= q_pos; k++) {
                    mask[((size_t) n * (size_t) b.T + (size_t) q) * (size_t) n_kv_pad + (size_t) k] = zero;
                }
            }
        }
        ggml_backend_tensor_set(b.mask, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    }
}

// Append one predictor pass to an existing graph. A non NULL
// hidden_bridge selects the T=2 prefill flavor reading [talker_hidden,
// embed(c0)] per slot through lm_head[0]; otherwise the pass is the
// single token step for g_head, appending at the fixed cache row
// g_head + 1. The pass ends in its sampling tail: it gathers its input
// ids from row g_head of the persistent sp->codes accumulator (row 0
// is the host written c0) and writes the ids it samples from
// lm_head[g_head] into row g_head + 1, so passes replay with no logits
// readback. Passes chained in one graph execute in node insertion
// order on the direct backend compute path, so each pass reads the
// codes row and the kv rows its predecessors wrote. use_flash_attn /
// clamp_fp16 apply to every layer. logits_out receives the pass
// logits, bake records the inputs to upload after allocation.
static void code_predictor_pass_append(struct ggml_context *           gctx,
                                       struct ggml_cgraph *            gf,
                                       const CodePredictorWeights *    cw,
                                       KVCache *                       kv,
                                       struct ggml_tensor *            embd_table,
                                       struct ggml_tensor *            hidden_bridge,
                                       SamplerInputs *                 sp,
                                       int                             g_head,
                                       int                             N,
                                       bool                            use_flash_attn,
                                       bool                            clamp_fp16,
                                       struct ggml_tensor **           logits_out,
                                       std::vector<CodePredPassBake> & bake) {
    const int T        = hidden_bridge ? 2 : 1;
    const int n_past   = hidden_bridge ? 0 : g_head + 1;
    const int n_layers = cw->num_hidden_layers;

    // The attention window spans the whole frame cache (16 slots): a
    // constant width keeps every flavor at the same mask shape.
    const int n_kv_pad = kv->max_seq_len;

    // Inputs: one code id per slot gathered in graph from embd_table,
    // positions, kv rows and the attention mask. The ids come from row
    // g_head of the persistent codes accumulator, written either by
    // the host (row 0, c0) or by the previous graph's sampling tail,
    // so replays upload nothing per step. The prefill path (T == 2)
    // concats each slot's resident talker hidden ahead of embed(c0),
    // both on device: the per slot sequence is [talker_hidden,
    // embed(c0)] with zero row upload.
    struct ggml_tensor * pos_in  = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, T * N);
    struct ggml_tensor * mask_in = ggml_new_tensor_4d(gctx, GGML_TYPE_F16, n_kv_pad, T, 1, N);
    struct ggml_tensor * rows_in = ggml_new_tensor_3d(gctx, GGML_TYPE_I64, T, 1, N);
    ggml_set_name(pos_in, "positions");
    ggml_set_name(mask_in, "causal_mask");
    ggml_set_name(rows_in, "kv_rows");
    // pos, rows, and mask bake once, so they also carry the output
    // flag: the allocator never frees an output, which keeps their
    // slots out of the intermediate reuse pool across replays.
    ggml_set_input(pos_in);
    ggml_set_output(pos_in);
    ggml_set_input(mask_in);
    ggml_set_output(mask_in);
    ggml_set_input(rows_in);
    ggml_set_output(rows_in);

    struct ggml_tensor * ids_in = ggml_view_1d(gctx, sp->codes, N, (size_t) g_head * sp->codes->nb[1]);
    ggml_set_name(ids_in, "sub_code_ids");

    struct ggml_tensor * x_in = ggml_get_rows(gctx, embd_table, ids_in);  // [hidden, N]
    if (T == 2) {
        struct ggml_tensor * bridge_cols =
            ggml_view_2d(gctx, hidden_bridge, hidden_bridge->ne[0], N, hidden_bridge->nb[1], 0);
        struct ggml_tensor * b3 = ggml_reshape_3d(gctx, bridge_cols, hidden_bridge->ne[0], 1, N);
        struct ggml_tensor * e3 = ggml_reshape_3d(gctx, x_in, x_in->ne[0], 1, N);
        x_in                    = ggml_concat(gctx, b3, e3, 1);  // [in_dim, 2, N]
        x_in                    = ggml_reshape_2d(gctx, x_in, x_in->ne[0], T * N);
    }
    ggml_set_name(x_in, "sub_input");

    // small_to_mtp projection: Linear(talker_hidden -> hidden) with bias.
    // When absent (Identity case) the input is already at predictor hidden.
    struct ggml_tensor * h = x_in;
    if (cw->mtp_proj_w) {
        h = ggml_mul_mat(gctx, cw->mtp_proj_w, h);
        if (cw->mtp_proj_b) {
            h = ggml_add(gctx, h, cw->mtp_proj_b);
        }
        ggml_set_name(h, "mtp_proj_out");
    }

    for (int l = 0; l < n_layers; l++) {
        h = code_predictor_layer_forward(gctx, cw, cw->layers[(size_t) l], h, pos_in, mask_in, rows_in,
                                         kv->k4[(size_t) l], kv->v4[(size_t) l], T, N, n_kv_pad, use_flash_attn,
                                         clamp_fp16, gf);
    }

    struct ggml_tensor * h_final = ggml_rms_norm(gctx, h, cw->rms_norm_eps);
    h_final                      = ggml_mul(gctx, h_final, cw->norm_w);
    if (T > 1) {
        // Last position of every slot: columns (T - 1) + n * T, one
        // strided view so the prefill pays N lm_head rows.
        h_final = ggml_cont(gctx, ggml_view_2d(gctx, h_final, h_final->ne[0], N, (size_t) T * h_final->nb[1],
                                               (size_t) (T - 1) * h_final->nb[1]));
    }

    struct ggml_tensor * logits = ggml_mul_mat(gctx, cw->lm_head[(size_t) g_head], h_final);
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);
    ggml_build_forward_expand(gf, logits);

    ggml_build_forward_expand(gf, sampler_tail_build(gctx, logits, sp, g_head));

    bake.push_back({ pos_in, rows_in, mask_in, T, n_past });
    *logits_out = logits;
}

// Build the unrolled frame graph over sets [0, N): the T=2 prefill and
// the n_acoustic - 1 steps chained in one static cgraph, so a frame
// replays in a single backend compute. talker_embd_table feeds the
// prefill c0 embedding, each step embeds through its own codebook
// table. Pass order in the node list carries the data dependencies:
// every step reads the codes row and the kv rows its predecessors
// wrote.
static bool code_predictor_frame_graph_build(const CodePredictorWeights * cw,
                                             KVCache *                    kv,
                                             ggml_backend_t               backend,
                                             struct ggml_tensor *         talker_embd_table,
                                             struct ggml_tensor *         hidden_bridge,
                                             SamplerInputs *              sp,
                                             int                          N,
                                             bool                         use_flash_attn,
                                             bool                         clamp_fp16,
                                             CodePredGraph *              cp) {
    const int n_layers   = cw->num_hidden_layers;
    const int n_acoustic = cw->num_acoustic_codebooks;
    const int max_nodes  = code_predictor_frame_graph_max_nodes(n_layers, n_acoustic);

    const size_t bytes =
        ggml_tensor_overhead() * (size_t) max_nodes + ggml_graph_overhead_custom((size_t) max_nodes, false);
    struct ggml_init_params gp = { bytes, NULL, true };
    cp->ctx                    = ggml_init(gp);
    if (!cp->ctx) {
        fprintf(stderr, "[CodePredictor] FATAL: frame graph ctx allocation failed\n");
        return false;
    }
    struct ggml_cgraph * gf = ggml_new_graph_custom(cp->ctx, max_nodes, false);

    std::vector<CodePredPassBake> bake;
    bake.reserve((size_t) n_acoustic);
    struct ggml_tensor * logits = NULL;

    code_predictor_pass_append(cp->ctx, gf, cw, kv, talker_embd_table, hidden_bridge, sp, 0, N, use_flash_attn,
                               clamp_fp16, &logits, bake);
    for (int g = 1; g < n_acoustic; g++) {
        code_predictor_pass_append(cp->ctx, gf, cw, kv, cw->codec_embedding[(size_t) (g - 1)], NULL, sp, g, N,
                                   use_flash_attn, clamp_fp16, &logits, bake);
    }

    cp->galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!cp->galloc || !ggml_gallocr_alloc_graph(cp->galloc, gf)) {
        fprintf(stderr, "[CodePredictor] FATAL: frame graph allocation failed\n");
        code_predictor_graph_free(cp);
        return false;
    }
    code_predictor_bake_upload(bake, N, kv->max_seq_len);

    cp->gf     = gf;
    cp->logits = logits;
    cp->N      = N;
    return true;
}

// Run the predictor for one audio frame through the unrolled frame
// graph, all N slots in lockstep: the host writes c0 into row 0 of the
// codes accumulator, uploads the per frame sampler state, replays one
// graph, then reads the [N, 16] accumulator back in one transfer. Per
// slot temperature controls greedy (temperature <= 0) vs stochastic;
// seed and subseq_base index each slot's Philox stream, subseq_base[i]
// being the subsequence of slot i's c0 sample (the 15 acoustic samples
// consume subseq_base[i] + 1 .. subseq_base[i] + 15). Fills out->codes
// as [N * 16] slot major. dump_dir may be NULL and applies to slot 0.
static bool code_predictor_frame_step(const CodePredictorWeights * cw,
                                      ggml_backend_t               backend,
                                      CodePredGraph *              frame_graph,
                                      SamplerInputs *              sp,
                                      const int32_t *              c0,
                                      int                          N,
                                      const float *                temperature,
                                      const int64_t *              seed,
                                      const int64_t *              subseq_base,
                                      const char *                 dump_dir,
                                      CodePredictorOutput *        out) {
    const int n_acoustic = cw->num_acoustic_codebooks;
    const int n_codes    = n_acoustic + 1;

    ggml_backend_tensor_set(sp->codes, c0, 0, (size_t) N * sizeof(int32_t));
    sampler_inputs_upload(sp, temperature, seed, subseq_base, N);

    if (ggml_backend_graph_compute(backend, frame_graph->gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[CodePredictor] FATAL: frame graph compute failed\n");
        return false;
    }

    // One readback per frame: the accumulator is row major over code
    // groups, out->codes is slot major.
    std::vector<int32_t> acc((size_t) N * (size_t) n_codes);
    ggml_backend_tensor_get(sp->codes, acc.data(), 0, acc.size() * sizeof(int32_t));
    out->codes.resize((size_t) N * (size_t) n_codes);
    for (int i = 0; i < N; i++) {
        for (int g = 0; g < n_codes; g++) {
            out->codes[(size_t) i * (size_t) n_codes + (size_t) g] = acc[(size_t) g * (size_t) N + (size_t) i];
        }
    }

    if (dump_dir) {
        DebugDumper d;
        debug_init(&d, dump_dir);
        std::vector<int32_t> codes32(out->codes.begin(), out->codes.begin() + n_codes);
        int                  n = (int) codes32.size();
        debug_dump_i32_as_f32(&d, "codes-step0", codes32.data(), &n, 1);
    }

    return true;
}
