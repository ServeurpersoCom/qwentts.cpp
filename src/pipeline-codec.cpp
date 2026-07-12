// pipeline-codec.cpp: load + decode for the Qwen3-TTS 12Hz codec.
//
// load chains the four module loaders (quantizer, transformer, upsample,
// DAC) and then loads the two pre_conv tensors into a dedicated wctx.
// decode builds the full forward graph in a per-call context, lets the
// scheduler allocate intermediates, uploads codes/positions/mask, runs
// graph_compute, and pulls the audio buffer back to host.

#include "pipeline-codec.h"

#include "causal-trans-conv.h"
#include "debug.h"
#include "qt-error.h"
#include "timer.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

bool pipeline_codec_load(PipelineCodec * pc, const char * gguf_path, BackendPair bp) {
    pc->bp              = bp;
    pc->backend         = bp.backend;
    pc->qenc_host_ready = false;
    pc->stream_ready    = false;
    pc->stream_ctx      = NULL;
    pc->stream_buf      = NULL;
    pc->stream_pos      = 0;
    for (int i = 0; i < CODEC_STREAM_CLASSES; i++) {
        pc->stream_graphs[i] = {};
    }
    for (int i = 0; i < CODEC_SNAP_SLOTS; i++) {
        pc->snaps[i] = {};
    }
    pc->snap_stamp = 0;

    if (!gf_load(&pc->gguf, gguf_path)) {
        qt_log(QT_LOG_ERROR, "[Pipeline] failed to load %s", gguf_path);
        return false;
    }

    if (!quant_decoder_load(&pc->qdec, pc->gguf, pc->backend)) {
        gf_close(&pc->gguf);
        return false;
    }

    if (!tok_trans_load(&pc->transformer, pc->gguf, pc->backend)) {
        quant_decoder_free(&pc->qdec);
        gf_close(&pc->gguf);
        return false;
    }

    if (!upsample_stage_load(&pc->upsample, pc->gguf, pc->backend)) {
        tok_trans_free(&pc->transformer);
        quant_decoder_free(&pc->qdec);
        gf_close(&pc->gguf);
        return false;
    }

    if (!dac_decoder_load(&pc->dac, pc->gguf, pc->backend)) {
        upsample_stage_free(&pc->upsample);
        tok_trans_free(&pc->transformer);
        quant_decoder_free(&pc->qdec);
        gf_close(&pc->gguf);
        return false;
    }

    // pre_conv: 2 tensors, dedicated wctx
    {
        WeightCtx wctx;
        wctx_init(&wctx, 4);
        pc->pre_conv_w = gf_load_conv(&wctx, pc->gguf, "tok_dec.pre_conv.weight");
        pc->pre_conv_b = gf_load_tensor(&wctx, pc->gguf, "tok_dec.pre_conv.bias");
        if (!wctx_alloc(&wctx, pc->backend)) {
            qt_log(QT_LOG_ERROR, "[Pipeline] pre_conv backend allocation failed");
            dac_decoder_free(&pc->dac);
            upsample_stage_free(&pc->upsample);
            tok_trans_free(&pc->transformer);
            quant_decoder_free(&pc->qdec);
            gf_close(&pc->gguf);
            return false;
        }
        pc->pre_conv_wctx = std::move(wctx);
    }

    // Encoder half (seanet, enc_transformer, enc_downsample, qenc)
    // stays on disk until the first encode request.
    pc->enc_loaded = false;

    if (!graph_arena_init(&pc->dec_arena, 4096)) {
        wctx_free(&pc->pre_conv_wctx);
        dac_decoder_free(&pc->dac);
        upsample_stage_free(&pc->upsample);
        tok_trans_free(&pc->transformer);
        quant_decoder_free(&pc->qdec);
        gf_close(&pc->gguf);
        return false;
    }

    pc->sched = backend_sched_new(bp, 4096);

    qt_log(QT_LOG_INFO, "[Pipeline] Ready: hop %d samples @ %d Hz mono, %d codebooks @ 12.5 Hz", TOKENIZER_HOP_LENGTH,
           TOKENIZER_SAMPLE_RATE, TOKENIZER_NUM_CODEBOOKS);
    return true;
}

std::vector<float> pipeline_codec_decode(PipelineCodec * pc, const int32_t * codes, int K, int T) {
    if (K != TOKENIZER_NUM_CODEBOOKS) {
        qt_log(QT_LOG_ERROR, "[Pipeline] codes have %d codebooks, expected %d", K, TOKENIZER_NUM_CODEBOOKS);
        return {};
    }
    if (T <= 0) {
        qt_log(QT_LOG_ERROR, "[Pipeline] T must be > 0 (got %d)", T);
        return {};
    }

    // Persistent arena: identical slice widths rebuild every node at
    // the same address with identical shapes, so the backend CUDA graph
    // cache replays the captured executable without an update. Steady
    // state streaming decodes constant size slices and hits this path
    // on every emit.
    const int             n_max_nodes = 4096;
    struct ggml_context * gctx        = graph_arena_begin(&pc->dec_arena);

    // Inputs: codes [T, K] i32, positions [T] i32, mask [T, T] f32.
    struct ggml_tensor * codes_in = ggml_new_tensor_2d(gctx, GGML_TYPE_I32, T, K);
    ggml_set_name(codes_in, "codes_in");
    ggml_set_input(codes_in);

    struct ggml_tensor * positions = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, T);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    struct ggml_tensor * mask = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, T, T);
    ggml_set_name(mask, "mask");
    ggml_set_input(mask);

    // Build forward graph. Layout transitions are explicit ggml_cont(ggml_transpose(...))
    // calls: 3 transposes total at the natural module boundaries.
    struct ggml_tensor * h = quant_decode(gctx, &pc->qdec, codes_in);                            // [512, T] C-first
    h                      = ggml_cont(gctx, ggml_transpose(gctx, h));                           // [T, 512] T-first
    h                      = qwen_causal_conv1d(gctx, pc->pre_conv_w, pc->pre_conv_b, h, 3, 1);  // [T, 1024] T-first
    h                      = ggml_cont(gctx, ggml_transpose(gctx, h));                           // [1024, T] C-first
    h                      = tok_trans_forward(gctx, &pc->transformer, h, positions, mask);      // [1024, T]
    h                      = ggml_cont(gctx, ggml_transpose(gctx, h));                           // [T, 1024] T-first
    h                      = upsample_stage_forward(gctx, &pc->upsample, h);                     // [T*4, 1024]
    h                      = dac_decoder_forward(gctx, &pc->dac, h);                             // [T*1920, 1]
    h                      = ggml_clamp(gctx, h, -1.0f, 1.0f);

    ggml_set_name(h, "audio_out");
    ggml_set_output(h);

    struct ggml_cgraph * graph = ggml_new_graph_custom(gctx, n_max_nodes, false);
    ggml_build_forward_expand(graph, h);

    ggml_backend_sched_reset(pc->sched);
    if (!ggml_backend_sched_alloc_graph(pc->sched, graph)) {
        qt_log(QT_LOG_ERROR, "[Pipeline] sched_alloc_graph failed");
        ggml_backend_sched_reset(pc->sched);
        return {};
    }

    // Upload inputs
    ggml_backend_tensor_set(codes_in, codes, 0, (size_t) T * (size_t) K * sizeof(int32_t));

    std::vector<int32_t> pos_buf;
    tok_trans_build_positions(T, pos_buf);
    ggml_backend_tensor_set(positions, pos_buf.data(), 0, pos_buf.size() * sizeof(int32_t));

    std::vector<float> mask_buf;
    tok_trans_build_causal_sliding_mask(T, pc->transformer.sliding_window, mask_buf);
    ggml_backend_tensor_set(mask, mask_buf.data(), 0, mask_buf.size() * sizeof(float));

    // Compute
    enum ggml_status st = ggml_backend_sched_graph_compute(pc->sched, graph);
    if (st != GGML_STATUS_SUCCESS) {
        qt_log(QT_LOG_ERROR, "[Pipeline] graph_compute status=%d", (int) st);
        ggml_backend_sched_reset(pc->sched);
        return {};
    }

    // Fetch audio output. The arena and the sched allocation persist
    // into the next decode.
    const int          n_samples = T * TOKENIZER_HOP_LENGTH;
    std::vector<float> audio((size_t) n_samples);
    ggml_backend_tensor_get(h, audio.data(), 0, (size_t) n_samples * sizeof(float));
    return audio;
}

// Ring width of the streaming transformer KV cache: covers the 72 frame
// sliding window plus the fresh frame with headroom, constant so the
// step graph shape never changes.
static const int CODEC_STREAM_RING = 128;

// Allocate the streaming state on first use: one tensor per causal conv
// left context and per transposed conv carry, plus the transformer KV
// ring and the dedicated graph arena. Idempotent.
static bool pipeline_codec_stream_ensure(PipelineCodec * pc) {
    if (pc->stream_ready) {
        return true;
    }

    const int n_state = 2 + UPSAMPLE_MAX_BLOCKS + DAC_NUM_BLOCKS * (1 + DAC_RES_UNITS) + 4;

    struct ggml_init_params gp = { ggml_tensor_overhead() * (size_t) n_state, NULL, true };
    pc->stream_ctx             = ggml_init(gp);
    if (!pc->stream_ctx) {
        qt_log(QT_LOG_ERROR, "[Pipeline] stream state ggml_init failed");
        return false;
    }

    char name[64];
    auto tensor2d = [&](int t, int c, const char * n) {
        struct ggml_tensor * x = ggml_new_tensor_2d(pc->stream_ctx, GGML_TYPE_F32, t, c);
        ggml_set_name(x, n);
        return x;
    };

    // pre_conv k=3 over the 512 wide quantizer latents
    pc->stream_pre_conv = tensor2d(2, 512, "stream_pre_conv");

    // upsample ConvNeXt depthwise contexts, k=7 over the stage width
    for (int i = 0; i < pc->upsample.num_blocks; i++) {
        snprintf(name, sizeof(name), "stream_up_dw_%d", i);
        pc->stream_up.dw[i] = tensor2d(pc->upsample.dwconv_kernel - 1, pc->upsample.channels, name);
    }

    // DAC contexts: conv_pre, per block transconv carry + res unit conv1, conv_post.
    // conv_pre reads the 1024 wide upsample output.
    pc->stream_dac.pre = tensor2d(6, 1024, "stream_dac_pre");
    for (int i = 0; i < DAC_NUM_BLOCKS; i++) {
        const QwenDACBlock & b = pc->dac.blk[i];
        snprintf(name, sizeof(name), "stream_dac_carry_%d", i);
        pc->stream_dac.carry[i] = tensor2d(b.kernel - b.stride, b.out_ch, name);
        for (int r = 0; r < DAC_RES_UNITS; r++) {
            snprintf(name, sizeof(name), "stream_dac_ru_%d_%d", i, r);
            pc->stream_dac.ru[i][r] = tensor2d(6 * b.ru[r].dilation, b.out_ch, name);
        }
    }
    pc->stream_dac.post = tensor2d(6, pc->dac.channels[DAC_NUM_BLOCKS], "stream_dac_post");

    pc->stream_buf = ggml_backend_alloc_ctx_tensors(pc->stream_ctx, pc->backend);
    if (!pc->stream_buf) {
        qt_log(QT_LOG_ERROR, "[Pipeline] stream state backend allocation failed");
        ggml_free(pc->stream_ctx);
        pc->stream_ctx = NULL;
        return false;
    }

    // The ring must hold the whole sliding window plus the fresh frame,
    // otherwise window slots alias through the modulo and corrupt the
    // attention silently.
    if (pc->transformer.sliding_window + 1 > CODEC_STREAM_RING) {
        qt_log(QT_LOG_ERROR, "[Pipeline] sliding window %d exceeds KV ring %d", pc->transformer.sliding_window,
               CODEC_STREAM_RING);
        return false;
    }

    if (!kv_cache_init(&pc->stream_kv, pc->transformer.num_layers, pc->transformer.num_kv_heads,
                       pc->transformer.head_dim, CODEC_STREAM_RING, pc->backend)) {
        ggml_backend_buffer_free(pc->stream_buf);
        pc->stream_buf = NULL;
        ggml_free(pc->stream_ctx);
        pc->stream_ctx = NULL;
        return false;
    }

    pc->stream_ready = true;
    qt_log(QT_LOG_INFO, "[Pipeline] Codec stream state ready: %d conv contexts, KV ring %d", n_state - 4,
           CODEC_STREAM_RING);
    return true;
}

bool pipeline_codec_stream_reset(PipelineCodec * pc) {
    if (!pipeline_codec_stream_ensure(pc)) {
        return false;
    }
    // Zero contexts reproduce the offline zero left pads bit for bit;
    // the zeroed KV ring stays hidden behind the sliding window mask.
    ggml_backend_buffer_clear(pc->stream_buf, 0);
    ggml_backend_buffer_clear(pc->stream_kv.buffer, 0);
    kv_cache_reset(&pc->stream_kv);
    pc->stream_pos = 0;
    return true;
}

// FNV-1a 64 over the raw code bytes with T folded in, so a reference
// sharing a prefix with a longer one cannot alias its key.
uint64_t pipeline_codec_ref_key(const int32_t * codes, int K, int T) {
    const uint8_t * p = (const uint8_t *) codes;
    const size_t    n = (size_t) K * (size_t) T * sizeof(int32_t);
    uint64_t        h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) {
        h = (h ^ p[i]) * 1099511628211ULL;
    }
    return (h ^ (uint64_t) T) * 1099511628211ULL;
}

// Allocate the mirror tensors of a snapshot slot: one duplicate per
// stream state and KV ring tensor. Creation order is preserved so the
// copy walker pairs source and mirror positionally.
static bool codec_snap_ensure(PipelineCodec * pc, CodecStateSnap * s) {
    if (s->ctx) {
        return true;
    }
    int n = 0;
    for (struct ggml_tensor * t = ggml_get_first_tensor(pc->stream_ctx); t;
         t                      = ggml_get_next_tensor(pc->stream_ctx, t)) {
        n++;
    }
    for (struct ggml_tensor * t = ggml_get_first_tensor(pc->stream_kv.ctx); t;
         t                      = ggml_get_next_tensor(pc->stream_kv.ctx, t)) {
        n++;
    }
    struct ggml_init_params gp = { ggml_tensor_overhead() * (size_t) n, NULL, true };
    s->ctx                     = ggml_init(gp);
    if (!s->ctx) {
        qt_log(QT_LOG_ERROR, "[Pipeline] snapshot ggml_init failed");
        return false;
    }
    for (struct ggml_tensor * t = ggml_get_first_tensor(pc->stream_ctx); t;
         t                      = ggml_get_next_tensor(pc->stream_ctx, t)) {
        ggml_dup_tensor(s->ctx, t);
    }
    for (struct ggml_tensor * t = ggml_get_first_tensor(pc->stream_kv.ctx); t;
         t                      = ggml_get_next_tensor(pc->stream_kv.ctx, t)) {
        ggml_dup_tensor(s->ctx, t);
    }
    s->buf = ggml_backend_alloc_ctx_tensors(s->ctx, pc->backend);
    if (!s->buf) {
        qt_log(QT_LOG_ERROR, "[Pipeline] snapshot backend allocation failed");
        ggml_free(s->ctx);
        s->ctx = NULL;
        return false;
    }
    return true;
}

// Copy every stream state and KV ring tensor to (save) or from its
// slot mirror, device to device on a shared backend.
static void codec_snap_copy(PipelineCodec * pc, CodecStateSnap * s, bool save) {
    struct ggml_tensor * m = ggml_get_first_tensor(s->ctx);
    for (struct ggml_tensor * t = ggml_get_first_tensor(pc->stream_ctx); t;
         t = ggml_get_next_tensor(pc->stream_ctx, t), m = ggml_get_next_tensor(s->ctx, m)) {
        ggml_backend_tensor_copy(save ? t : m, save ? m : t);
    }
    for (struct ggml_tensor * t = ggml_get_first_tensor(pc->stream_kv.ctx); t;
         t = ggml_get_next_tensor(pc->stream_kv.ctx, t), m = ggml_get_next_tensor(s->ctx, m)) {
        ggml_backend_tensor_copy(save ? t : m, save ? m : t);
    }
}

bool pipeline_codec_stream_restore(PipelineCodec * pc, uint64_t key) {
    if (!pc->stream_ready) {
        return false;
    }
    for (int i = 0; i < CODEC_SNAP_SLOTS; i++) {
        CodecStateSnap * s = &pc->snaps[i];
        if (s->stamp == 0 || s->key != key) {
            continue;
        }
        Timer t;
        codec_snap_copy(pc, s, false);
        pc->stream_pos = s->pos;
        s->stamp       = ++pc->snap_stamp;
        qt_log(QT_LOG_INFO, "[Pipeline] Codec state restored from snapshot in %.1f ms (%d frames)", t.ms(), s->pos);
        return true;
    }
    return false;
}

bool pipeline_codec_stream_snapshot(PipelineCodec * pc, uint64_t key) {
    CodecStateSnap * lru = &pc->snaps[0];
    for (int i = 1; i < CODEC_SNAP_SLOTS; i++) {
        if (pc->snaps[i].stamp < lru->stamp) {
            lru = &pc->snaps[i];
        }
    }
    if (!codec_snap_ensure(pc, lru)) {
        return false;
    }
    codec_snap_copy(pc, lru, true);
    lru->key   = key;
    lru->pos   = pc->stream_pos;
    lru->stamp = ++pc->snap_stamp;
    qt_log(QT_LOG_INFO, "[Pipeline] Codec state snapshot saved (%d frames)", lru->pos);
    return true;
}

// Build the static stream graph of chunk width T = 1 << cls: the
// same module chain as the T=1 frame graph, every stream state and
// KV ring tensor shared across classes, inputs and intermediates
// sized for T.
static bool codec_stream_graph_ensure(PipelineCodec * pc, int cls) {
    CodecStreamGraph * sg = &pc->stream_graphs[cls];
    if (sg->ctx) {
        return true;
    }
    const int T    = 1 << cls;
    const int ring = CODEC_STREAM_RING;
    const int K    = TOKENIZER_NUM_CODEBOOKS;

    // The ring must hold the sliding window plus the whole fresh chunk.
    if (pc->transformer.sliding_window + T > ring) {
        qt_log(QT_LOG_ERROR, "[Pipeline] sliding window %d plus chunk %d exceeds KV ring %d",
               pc->transformer.sliding_window, T, ring);
        return false;
    }

    const int               max_nodes = 4096;
    struct ggml_init_params gp        = {
        ggml_tensor_overhead() * (size_t) max_nodes + ggml_graph_overhead_custom(max_nodes, false),
        NULL,
        true,
    };
    sg->ctx = ggml_init(gp);
    if (!sg->ctx) {
        qt_log(QT_LOG_ERROR, "[Pipeline] stream graph ggml_init failed (T=%d)", T);
        return false;
    }
    struct ggml_context * gctx = sg->ctx;
    struct ggml_cgraph *  gf   = ggml_new_graph_custom(gctx, max_nodes, false);

    struct ggml_tensor * codes_in = ggml_new_tensor_2d(gctx, GGML_TYPE_I32, T, K);
    struct ggml_tensor * pos_in   = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, T);
    struct ggml_tensor * rows_in  = ggml_new_tensor_1d(gctx, GGML_TYPE_I64, T);
    struct ggml_tensor * mask_in  = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, ring, T);
    ggml_set_name(codes_in, "codes_in");
    ggml_set_name(pos_in, "positions");
    ggml_set_name(rows_in, "kv_rows");
    ggml_set_name(mask_in, "ring_mask");
    ggml_set_input(codes_in);
    ggml_set_input(pos_in);
    ggml_set_input(rows_in);
    ggml_set_input(mask_in);

    // Same module chain as pipeline_codec_decode with the stateful
    // variants threaded through the persistent stream tensors. The
    // quantizer uses the alignment safe variant: no scheduler input
    // duplication happens on the direct backend compute path.
    struct ggml_tensor * h = quant_decode_stream(gctx, &pc->qdec, codes_in);  // [512, T] C-first
    h                      = ggml_cont(gctx, ggml_transpose(gctx, h));        // [T, 512] T-first
    h = qwen_causal_conv1d_stream(gctx, gf, pc->pre_conv_w, pc->pre_conv_b, h, 3, 1, pc->stream_pre_conv);
    h = ggml_cont(gctx, ggml_transpose(gctx, h));                             // [1024, T] C-first
    h = tok_trans_forward_stream(gctx, gf, &pc->transformer, h, pos_in, mask_in, rows_in, &pc->stream_kv);
    h = ggml_cont(gctx, ggml_transpose(gctx, h));                             // [T, 1024] T-first
    h = upsample_stage_forward_stream(gctx, gf, &pc->upsample, h, &pc->stream_up);
    h = dac_decoder_forward_stream(gctx, gf, &pc->dac, h, &pc->stream_dac);
    h = ggml_clamp(gctx, h, -1.0f, 1.0f);
    ggml_set_name(h, "audio_out");
    ggml_set_output(h);
    ggml_build_forward_expand(gf, h);

    sg->galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(pc->backend));
    if (!sg->galloc || !ggml_gallocr_alloc_graph(sg->galloc, gf)) {
        qt_log(QT_LOG_ERROR, "[Pipeline] stream graph allocation failed (T=%d)", T);
        if (sg->galloc) {
            ggml_gallocr_free(sg->galloc);
        }
        ggml_free(sg->ctx);
        *sg = {};
        return false;
    }

    sg->gf    = gf;
    sg->codes = codes_in;
    sg->pos   = pos_in;
    sg->rows  = rows_in;
    sg->mask  = mask_in;
    sg->out   = h;
    return true;
}

bool pipeline_codec_decode_stream(PipelineCodec * pc, const int32_t * codes, int T, float * audio_out) {
    const int K    = TOKENIZER_NUM_CODEBOOKS;
    const int ring = CODEC_STREAM_RING;

    int cls = 0;
    while ((1 << cls) < T) {
        cls++;
    }
    if (cls >= CODEC_STREAM_CLASSES || (1 << cls) != T) {
        qt_log(QT_LOG_ERROR, "[Pipeline] invalid stream chunk width %d", T);
        return false;
    }
    if (!codec_stream_graph_ensure(pc, cls)) {
        return false;
    }
    CodecStreamGraph * sg = &pc->stream_graphs[cls];

    ggml_backend_tensor_set(sg->codes, codes, 0, (size_t) T * (size_t) K * sizeof(int32_t));

    sg->pos_buf.resize((size_t) T);
    sg->rows_buf.resize((size_t) T);
    for (int t = 0; t < T; t++) {
        sg->pos_buf[(size_t) t]  = pc->stream_pos + t;
        sg->rows_buf[(size_t) t] = (int64_t) ((pc->stream_pos + t) % ring);
    }
    ggml_backend_tensor_set(sg->pos, sg->pos_buf.data(), 0, (size_t) T * sizeof(int32_t));
    ggml_backend_tensor_set(sg->rows, sg->rows_buf.data(), 0, (size_t) T * sizeof(int64_t));

    tok_trans_build_stream_mask(pc->stream_pos, T, ring, pc->transformer.sliding_window, sg->mask_buf);
    ggml_backend_tensor_set(sg->mask, sg->mask_buf.data(), 0, sg->mask_buf.size() * sizeof(float));

    enum ggml_status st = ggml_backend_graph_compute(pc->backend, sg->gf);
    if (st != GGML_STATUS_SUCCESS) {
        qt_log(QT_LOG_ERROR, "[Pipeline] stream graph_compute status=%d", (int) st);
        return false;
    }

    if (audio_out) {
        ggml_backend_tensor_get(sg->out, audio_out, 0, (size_t) T * (size_t) TOKENIZER_HOP_LENGTH * sizeof(float));
    }

    // The state tensors and every class allocation persist into the
    // next chunk.
    pc->stream_pos += T;
    return true;
}

bool pipeline_codec_ensure_encoder(PipelineCodec * pc) {
    if (pc->enc_loaded) {
        return true;
    }

    Timer t_load;
    if (!seanet_encoder_load(&pc->seanet, pc->gguf, pc->backend)) {
        return false;
    }
    if (!enc_trans_load(&pc->enc_transformer, pc->gguf, pc->backend)) {
        seanet_encoder_free(&pc->seanet);
        return false;
    }
    if (!enc_down_load(&pc->enc_downsample, pc->gguf, pc->backend)) {
        enc_trans_free(&pc->enc_transformer);
        seanet_encoder_free(&pc->seanet);
        return false;
    }
    if (!quant_encode_load(&pc->qenc, pc->gguf, pc->backend)) {
        enc_down_free(&pc->enc_downsample);
        enc_trans_free(&pc->enc_transformer);
        seanet_encoder_free(&pc->seanet);
        return false;
    }

    pc->enc_loaded = true;
    qt_log(QT_LOG_INFO, "[Pipeline] Codec encoder lazy loaded in %.0f ms", t_load.ms());
    return true;
}

std::vector<int32_t> pipeline_codec_encode(PipelineCodec * pc,
                                           const float *   audio,
                                           int             n_samples,
                                           const char *    dump_dir) {
    if (!pipeline_codec_ensure_encoder(pc)) {
        qt_log(QT_LOG_ERROR, "[Pipeline] codec encoder load failed");
        return {};
    }
    if (n_samples <= 0 || (n_samples % TOKENIZER_HOP_LENGTH) != 0) {
        qt_log(QT_LOG_ERROR, "[Pipeline] n_samples must be a positive multiple of %d (got %d)", TOKENIZER_HOP_LENGTH,
               n_samples);
        return {};
    }
    int T = n_samples / TOKENIZER_HOP_LENGTH;

    // Lazy-load CPU mirror of the RVQ encode codebooks on first call.
    if (!pc->qenc_host_ready) {
        quant_encode_host_load(&pc->qenc_sem_host, pc->qenc.semantic, pc->qenc.codebook_size, pc->qenc.codebook_dim,
                               pc->qenc.hidden_size);
        quant_encode_host_load(&pc->qenc_aco_host, pc->qenc.acoustic, pc->qenc.codebook_size, pc->qenc.codebook_dim,
                               pc->qenc.hidden_size);
        pc->qenc_host_ready = true;
    }

    const int    n_max_nodes = 4096;
    const size_t graph_ctx_size =
        ggml_tensor_overhead() * (size_t) n_max_nodes + ggml_graph_overhead_custom((size_t) n_max_nodes, false);

    struct ggml_init_params gp   = { graph_ctx_size, NULL, /*no_alloc=*/true };
    struct ggml_context *   gctx = ggml_init(gp);
    if (!gctx) {
        qt_log(QT_LOG_ERROR, "[Pipeline] ggml_init failed for encode graph ctx");
        return {};
    }

    // SEANet input shape: [T_audio, 1] f32 T-first (mono waveform).
    struct ggml_tensor * audio_in = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, n_samples, 1);
    ggml_set_name(audio_in, "audio_in");
    ggml_set_input(audio_in);

    // Encoder transformer mask is built on the post-SEANet T = n_samples / 960.
    int                  T_emb     = n_samples / 960;
    struct ggml_tensor * positions = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, T_emb);
    ggml_set_name(positions, "enc_positions");
    ggml_set_input(positions);

    struct ggml_tensor * mask = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, T_emb, T_emb);
    ggml_set_name(mask, "enc_mask");
    ggml_set_input(mask);

    // Forward chain.
    struct ggml_tensor * sn_init_t    = NULL;
    struct ggml_tensor * sn_resnet0_t = NULL;
    struct ggml_tensor * sn_stage0_t  = NULL;
    struct ggml_tensor * sn_stage1_t  = NULL;
    struct ggml_tensor * sn_stage3_t  = NULL;
    struct ggml_tensor * h_seanet =
        seanet_encoder_forward(gctx, &pc->seanet, audio_in, &sn_init_t, &sn_resnet0_t, &sn_stage0_t, &sn_stage1_t,
                               &sn_stage3_t);                                                       // [T_emb, 512]
    struct ggml_tensor * h    = ggml_cont(gctx, ggml_transpose(gctx, h_seanet));                    // [512, T_emb]
    struct ggml_tensor * h_et = enc_trans_forward(gctx, &pc->enc_transformer, h, positions, mask);  // [512, T_emb]
    h                         = ggml_cont(gctx, ggml_transpose(gctx, h_et));                        // [T_emb, 512]
    h                         = enc_down_forward(gctx, &pc->enc_downsample, h);                     // [T, 512]

    // The CPU RVQ encode loop expects the hidden buffer as [T, hidden]
    // row-major (hidden fast in memory). The downsample output ne=(T, 512)
    // walks T fast in ggml memory, which is [hidden, T] in numpy terms.
    // Transpose to get the buffer layout we want once read back to host.
    h = ggml_cont(gctx, ggml_transpose(gctx, h));  // ne=(512, T)

    const char *         dump            = dump_dir;
    struct ggml_tensor * h_seanet_dump   = NULL;
    struct ggml_tensor * sn_init_dump    = NULL;
    struct ggml_tensor * sn_resnet0_dump = NULL;
    struct ggml_tensor * sn_stage0_dump  = NULL;
    struct ggml_tensor * sn_stage1_dump  = NULL;
    struct ggml_tensor * sn_stage3_dump  = NULL;
    if (dump) {
        // SEANet output naturally lands as ggml ne=(T, hidden) (T innermost).
        // The encoder_transformer and downsample dumps further down are
        // T-first numpy [T, hidden], so we transpose+cont to bring hidden
        // innermost before pinning as a graph output. The dump_2d then
        // emits shape (ne[1], ne[0]) = (T, hidden) on the numpy side.
        h_seanet_dump = ggml_cont(gctx, ggml_transpose(gctx, h_seanet));
        ggml_set_output(h_seanet_dump);
        ggml_set_name(h_seanet_dump, "seanet_out_dump");
        ggml_set_output(h_et);
        ggml_set_name(h_et, "enc_transformer_out");

        // SEANet bisection points. Same transpose convention as h_seanet.
        if (sn_init_t) {
            sn_init_dump = ggml_cont(gctx, ggml_transpose(gctx, sn_init_t));
            ggml_set_output(sn_init_dump);
            ggml_set_name(sn_init_dump, "seanet_init_dump");
        }
        if (sn_resnet0_t) {
            sn_resnet0_dump = ggml_cont(gctx, ggml_transpose(gctx, sn_resnet0_t));
            ggml_set_output(sn_resnet0_dump);
            ggml_set_name(sn_resnet0_dump, "seanet_resnet0_dump");
        }
        if (sn_stage0_t) {
            sn_stage0_dump = ggml_cont(gctx, ggml_transpose(gctx, sn_stage0_t));
            ggml_set_output(sn_stage0_dump);
            ggml_set_name(sn_stage0_dump, "seanet_stage0_dump");
        }
        if (sn_stage1_t) {
            sn_stage1_dump = ggml_cont(gctx, ggml_transpose(gctx, sn_stage1_t));
            ggml_set_output(sn_stage1_dump);
            ggml_set_name(sn_stage1_dump, "seanet_stage1_dump");
        }
        if (sn_stage3_t) {
            sn_stage3_dump = ggml_cont(gctx, ggml_transpose(gctx, sn_stage3_t));
            ggml_set_output(sn_stage3_dump);
            ggml_set_name(sn_stage3_dump, "seanet_stage3_dump");
        }
    }

    ggml_set_name(h, "enc_hidden_out");
    ggml_set_output(h);

    struct ggml_cgraph * graph = ggml_new_graph_custom(gctx, n_max_nodes, false);
    ggml_build_forward_expand(graph, h);
    if (h_seanet_dump) {
        ggml_build_forward_expand(graph, h_seanet_dump);
    }
    if (sn_init_dump) {
        ggml_build_forward_expand(graph, sn_init_dump);
    }
    if (sn_resnet0_dump) {
        ggml_build_forward_expand(graph, sn_resnet0_dump);
    }
    if (sn_stage0_dump) {
        ggml_build_forward_expand(graph, sn_stage0_dump);
    }
    if (sn_stage1_dump) {
        ggml_build_forward_expand(graph, sn_stage1_dump);
    }
    if (sn_stage3_dump) {
        ggml_build_forward_expand(graph, sn_stage3_dump);
    }

    // Reset before alloc: a prior decode/synthesis may have left the shared
    // codec sched allocated, which trips GGML_ASSERT(!sched->is_alloc). Mirrors
    // the decode path (top of this file) and speaker_encoder_extract.
    ggml_backend_sched_reset(pc->sched);
    if (!ggml_backend_sched_alloc_graph(pc->sched, graph)) {
        qt_log(QT_LOG_ERROR, "[Pipeline] encode sched_alloc_graph failed");
        ggml_backend_sched_reset(pc->sched);
        ggml_free(gctx);
        return {};
    }

    ggml_backend_tensor_set(audio_in, audio, 0, (size_t) n_samples * sizeof(float));

    std::vector<int32_t> pos_buf;
    enc_trans_build_positions(T_emb, pos_buf);
    ggml_backend_tensor_set(positions, pos_buf.data(), 0, pos_buf.size() * sizeof(int32_t));

    std::vector<float> mask_buf;
    enc_trans_build_causal_mask(T_emb, mask_buf);
    ggml_backend_tensor_set(mask, mask_buf.data(), 0, mask_buf.size() * sizeof(float));

    enum ggml_status st = ggml_backend_sched_graph_compute(pc->sched, graph);
    if (st != GGML_STATUS_SUCCESS) {
        qt_log(QT_LOG_ERROR, "[Pipeline] encode graph_compute status=%d", (int) st);
        ggml_backend_sched_reset(pc->sched);
        ggml_free(gctx);
        return {};
    }

    if (dump) {
        DebugDumper d;
        debug_init(&d, dump);
        // Raw audio input dump: the SEANet sees this, and any divergence
        // in the resampler (torchaudio reimpl C++ vs librosa Python) shows
        // up here as a phase or amplitude drift.
        debug_dump_1d(&d, "audio-input", audio, n_samples);
        // ggml ne layout matches numpy's last-dim-fastest, so a [d0, d1]
        // tensor in ggml dumps as a [d1, d0] numpy array. We emit the
        // shape ggml-side (ne[1], ne[0]) so numpy reshapes it correctly
        // on read. Values themselves are the same memory order.
        auto dump2 = [&](const char * name, struct ggml_tensor * t) {
            size_t             n = ggml_nelements(t);
            std::vector<float> buf(n);
            ggml_backend_tensor_get(t, buf.data(), 0, n * sizeof(float));
            debug_dump_2d(&d, name, buf.data(), (int) t->ne[1], (int) t->ne[0]);
        };
        dump2("seanet-out", h_seanet_dump);
        dump2("enc-transformer-out", h_et);
        dump2("codec-pre-fsq", h);
        if (sn_init_dump) {
            dump2("seanet-init", sn_init_dump);
        }
        if (sn_resnet0_dump) {
            dump2("seanet-resnet0", sn_resnet0_dump);
        }
        if (sn_stage0_dump) {
            dump2("seanet-stage0", sn_stage0_dump);
        }
        if (sn_stage1_dump) {
            dump2("seanet-stage1", sn_stage1_dump);
        }
        if (sn_stage3_dump) {
            dump2("seanet-stage3", sn_stage3_dump);
        }
    }

    // Read back the post-downsample hidden buffer for CPU-side RVQ encode.
    // Layout in ggml is [T, hidden] with T on ne[0]. The contiguous memory
    // walks T fast, hidden slow, which matches the `[T, hidden] row-major
    // index = t*hidden + c` convention expected by quant_encode_cpu.
    std::vector<float> hidden_host((size_t) T * (size_t) pc->qenc.hidden_size);
    ggml_backend_tensor_get(h, hidden_host.data(), 0, hidden_host.size() * sizeof(float));

    ggml_backend_sched_reset(pc->sched);
    ggml_free(gctx);

    return quant_encode_cpu(&pc->qenc_sem_host, &pc->qenc_aco_host, hidden_host.data(), T);
}

void pipeline_codec_free(PipelineCodec * pc) {
    if (pc->sched) {
        ggml_backend_sched_free(pc->sched);
        pc->sched = NULL;
    }
    graph_arena_free(&pc->dec_arena);
    if (pc->stream_ready) {
        for (int i = 0; i < CODEC_STREAM_CLASSES; i++) {
            CodecStreamGraph * sg = &pc->stream_graphs[i];
            if (sg->galloc) {
                ggml_gallocr_free(sg->galloc);
            }
            if (sg->ctx) {
                ggml_free(sg->ctx);
            }
            *sg = {};
        }
        kv_cache_free(&pc->stream_kv);
        ggml_backend_buffer_free(pc->stream_buf);
        pc->stream_buf = NULL;
        ggml_free(pc->stream_ctx);
        pc->stream_ctx   = NULL;
        pc->stream_ready = false;
    }
    for (int i = 0; i < CODEC_SNAP_SLOTS; i++) {
        CodecStateSnap * s = &pc->snaps[i];
        if (s->buf) {
            ggml_backend_buffer_free(s->buf);
        }
        if (s->ctx) {
            ggml_free(s->ctx);
        }
        *s = {};
    }
    pc->snap_stamp = 0;
    if (pc->enc_loaded) {
        quant_encode_free(&pc->qenc);
        enc_down_free(&pc->enc_downsample);
        enc_trans_free(&pc->enc_transformer);
        seanet_encoder_free(&pc->seanet);
        pc->enc_loaded = false;
    }
    wctx_free(&pc->pre_conv_wctx);
    dac_decoder_free(&pc->dac);
    upsample_stage_free(&pc->upsample);
    tok_trans_free(&pc->transformer);
    quant_decoder_free(&pc->qdec);
    if (pc->gguf.gguf) {
        gf_close(&pc->gguf);
    }
}
