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

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

bool pipeline_codec_load(PipelineCodec * pc, const char * gguf_path, BackendPair bp) {
    pc->bp      = bp;
    pc->backend = bp.backend;

    if (!gf_load(&pc->gguf, gguf_path)) {
        qt_log(QT_LOG_ERROR, "[Pipeline] failed to load %s", gguf_path);
        return false;
    }

    if (!qwen_quantizer_decoder_load(&pc->qdec, pc->gguf, pc->backend)) {
        gf_close(&pc->gguf);
        return false;
    }

    if (!qwen_tokenizer_transformer_load(&pc->transformer, pc->gguf, pc->backend)) {
        qwen_quantizer_decoder_free(&pc->qdec);
        gf_close(&pc->gguf);
        return false;
    }

    if (!qwen_upsample_stage_load(&pc->upsample, pc->gguf, pc->backend)) {
        qwen_tokenizer_transformer_free(&pc->transformer);
        qwen_quantizer_decoder_free(&pc->qdec);
        gf_close(&pc->gguf);
        return false;
    }

    if (!qwen_dac_decoder_load(&pc->dac, pc->gguf, pc->backend)) {
        qwen_upsample_stage_free(&pc->upsample);
        qwen_tokenizer_transformer_free(&pc->transformer);
        qwen_quantizer_decoder_free(&pc->qdec);
        gf_close(&pc->gguf);
        return false;
    }

    // pre_conv: 2 tensors, dedicated wctx
    {
        WeightCtx wctx;
        wctx_init(&wctx, 4);
        pc->pre_conv_w = gf_load_tensor(&wctx, pc->gguf, "tok_dec.pre_conv.weight");
        pc->pre_conv_b = gf_load_tensor(&wctx, pc->gguf, "tok_dec.pre_conv.bias");
        if (!wctx_alloc(&wctx, pc->backend)) {
            qt_log(QT_LOG_ERROR, "[Pipeline] pre_conv backend allocation failed");
            qwen_dac_decoder_free(&pc->dac);
            qwen_upsample_stage_free(&pc->upsample);
            qwen_tokenizer_transformer_free(&pc->transformer);
            qwen_quantizer_decoder_free(&pc->qdec);
            gf_close(&pc->gguf);
            return false;
        }
        pc->pre_conv_ctx = wctx.ctx;
        pc->pre_conv_buf = wctx.buffer;
    }

    if (!qwen_seanet_encoder_load(&pc->seanet, pc->gguf, pc->backend)) {
        ggml_backend_buffer_free(pc->pre_conv_buf);
        ggml_free(pc->pre_conv_ctx);
        qwen_dac_decoder_free(&pc->dac);
        qwen_upsample_stage_free(&pc->upsample);
        qwen_tokenizer_transformer_free(&pc->transformer);
        qwen_quantizer_decoder_free(&pc->qdec);
        gf_close(&pc->gguf);
        return false;
    }

    if (!qwen_encoder_transformer_load(&pc->enc_transformer, pc->gguf, pc->backend)) {
        qwen_seanet_encoder_free(&pc->seanet);
        ggml_backend_buffer_free(pc->pre_conv_buf);
        ggml_free(pc->pre_conv_ctx);
        qwen_dac_decoder_free(&pc->dac);
        qwen_upsample_stage_free(&pc->upsample);
        qwen_tokenizer_transformer_free(&pc->transformer);
        qwen_quantizer_decoder_free(&pc->qdec);
        gf_close(&pc->gguf);
        return false;
    }

    if (!qwen_encoder_downsample_load(&pc->enc_downsample, pc->gguf, pc->backend)) {
        qwen_encoder_transformer_free(&pc->enc_transformer);
        qwen_seanet_encoder_free(&pc->seanet);
        ggml_backend_buffer_free(pc->pre_conv_buf);
        ggml_free(pc->pre_conv_ctx);
        qwen_dac_decoder_free(&pc->dac);
        qwen_upsample_stage_free(&pc->upsample);
        qwen_tokenizer_transformer_free(&pc->transformer);
        qwen_quantizer_decoder_free(&pc->qdec);
        gf_close(&pc->gguf);
        return false;
    }

    if (!qwen_quantizer_encode_load(&pc->qenc, pc->gguf, pc->backend)) {
        qwen_encoder_downsample_free(&pc->enc_downsample);
        qwen_encoder_transformer_free(&pc->enc_transformer);
        qwen_seanet_encoder_free(&pc->seanet);
        ggml_backend_buffer_free(pc->pre_conv_buf);
        ggml_free(pc->pre_conv_ctx);
        qwen_dac_decoder_free(&pc->dac);
        qwen_upsample_stage_free(&pc->upsample);
        qwen_tokenizer_transformer_free(&pc->transformer);
        qwen_quantizer_decoder_free(&pc->qdec);
        gf_close(&pc->gguf);
        return false;
    }

    pc->sched = backend_sched_new(bp, 4096);

    qt_log(QT_LOG_INFO, "[Pipeline] Ready: hop %d samples @ %d Hz mono, %d codebooks @ 12.5 Hz",
           QWEN_TOKENIZER_HOP_LENGTH, QWEN_TOKENIZER_SAMPLE_RATE, QWEN_TOKENIZER_NUM_CODEBOOKS);
    return true;
}

std::vector<float> pipeline_codec_decode(PipelineCodec * pc, const int32_t * codes, int K, int T) {
    if (K != QWEN_TOKENIZER_NUM_CODEBOOKS) {
        qt_log(QT_LOG_ERROR, "[Pipeline] codes have %d codebooks, expected %d", K, QWEN_TOKENIZER_NUM_CODEBOOKS);
        return {};
    }
    if (T <= 0) {
        qt_log(QT_LOG_ERROR, "[Pipeline] T must be > 0 (got %d)", T);
        return {};
    }

    // Per-call graph context: tensor descriptors only, allocation is
    // delegated to the scheduler.
    const int    n_max_nodes = 4096;
    const size_t graph_ctx_size =
        ggml_tensor_overhead() * (size_t) n_max_nodes + ggml_graph_overhead_custom((size_t) n_max_nodes, false);

    struct ggml_init_params gp   = { graph_ctx_size, NULL, /*no_alloc=*/true };
    struct ggml_context *   gctx = ggml_init(gp);
    if (!gctx) {
        qt_log(QT_LOG_ERROR, "[Pipeline] ggml_init failed for graph ctx");
        return {};
    }

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
    struct ggml_tensor * h = qwen_quantizer_decode(gctx, &pc->qdec, codes_in);                   // [512, T] C-first
    h                      = ggml_cont(gctx, ggml_transpose(gctx, h));                           // [T, 512] T-first
    h                      = qwen_causal_conv1d(gctx, pc->pre_conv_w, pc->pre_conv_b, h, 3, 1);  // [T, 1024] T-first
    h                      = ggml_cont(gctx, ggml_transpose(gctx, h));                           // [1024, T] C-first
    h = qwen_tokenizer_transformer_forward(gctx, &pc->transformer, h, positions, mask);          // [1024, T]
    h = ggml_cont(gctx, ggml_transpose(gctx, h));                                                // [T, 1024] T-first
    h = qwen_upsample_stage_forward(gctx, &pc->upsample, h);                                     // [T*4, 1024]
    h = qwen_dac_decoder_forward(gctx, &pc->dac, h);                                             // [T*1920, 1]
    h = ggml_clamp(gctx, h, -1.0f, 1.0f);

    ggml_set_name(h, "audio_out");
    ggml_set_output(h);

    struct ggml_cgraph * graph = ggml_new_graph_custom(gctx, n_max_nodes, false);
    ggml_build_forward_expand(graph, h);

    if (!ggml_backend_sched_alloc_graph(pc->sched, graph)) {
        qt_log(QT_LOG_ERROR, "[Pipeline] sched_alloc_graph failed");
        ggml_backend_sched_reset(pc->sched);
        ggml_free(gctx);
        return {};
    }

    // Upload inputs
    ggml_backend_tensor_set(codes_in, codes, 0, (size_t) T * (size_t) K * sizeof(int32_t));

    std::vector<int32_t> pos_buf;
    qwen_build_positions(T, pos_buf);
    ggml_backend_tensor_set(positions, pos_buf.data(), 0, pos_buf.size() * sizeof(int32_t));

    std::vector<float> mask_buf;
    qwen_build_causal_sliding_mask(T, pc->transformer.sliding_window, mask_buf);
    ggml_backend_tensor_set(mask, mask_buf.data(), 0, mask_buf.size() * sizeof(float));

    // Compute
    enum ggml_status st = ggml_backend_sched_graph_compute(pc->sched, graph);
    if (st != GGML_STATUS_SUCCESS) {
        qt_log(QT_LOG_ERROR, "[Pipeline] graph_compute status=%d", (int) st);
        ggml_backend_sched_reset(pc->sched);
        ggml_free(gctx);
        return {};
    }

    // Fetch audio output
    const int          n_samples = T * QWEN_TOKENIZER_HOP_LENGTH;
    std::vector<float> audio((size_t) n_samples);
    ggml_backend_tensor_get(h, audio.data(), 0, (size_t) n_samples * sizeof(float));

    ggml_backend_sched_reset(pc->sched);
    ggml_free(gctx);
    return audio;
}

std::vector<int32_t> pipeline_codec_encode(PipelineCodec * pc, const float * audio, int n_samples,
                                           const char * dump_dir) {
    if (n_samples <= 0 || (n_samples % QWEN_TOKENIZER_HOP_LENGTH) != 0) {
        qt_log(QT_LOG_ERROR, "[Pipeline] n_samples must be a positive multiple of %d (got %d)",
               QWEN_TOKENIZER_HOP_LENGTH, n_samples);
        return {};
    }
    int T = n_samples / QWEN_TOKENIZER_HOP_LENGTH;

    // Lazy-load CPU mirror of the RVQ encode codebooks on first call.
    if (!pc->qenc_host_ready) {
        qwen_quantizer_encode_host_load(&pc->qenc_sem_host, pc->qenc.semantic, pc->qenc.codebook_size,
                                        pc->qenc.codebook_dim, pc->qenc.hidden_size);
        qwen_quantizer_encode_host_load(&pc->qenc_aco_host, pc->qenc.acoustic, pc->qenc.codebook_size,
                                        pc->qenc.codebook_dim, pc->qenc.hidden_size);
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
    struct ggml_tensor * h_seanet = qwen_seanet_encoder_forward(gctx, &pc->seanet, audio_in);  // [T_emb, 512]
    struct ggml_tensor * h        = ggml_cont(gctx, ggml_transpose(gctx, h_seanet));           // [512, T_emb]
    struct ggml_tensor * h_et =
        qwen_encoder_transformer_forward(gctx, &pc->enc_transformer, h, positions, mask);      // [512, T_emb]
    h = ggml_cont(gctx, ggml_transpose(gctx, h_et));                                           // [T_emb, 512]
    h = qwen_encoder_downsample_forward(gctx, &pc->enc_downsample, h);                         // [T, 512]

    // The CPU RVQ encode loop expects the hidden buffer as [T, hidden]
    // row-major (hidden fast in memory). The downsample output ne=(T, 512)
    // walks T fast in ggml memory, which is [hidden, T] in numpy terms.
    // Transpose to get the buffer layout we want once read back to host.
    h = ggml_cont(gctx, ggml_transpose(gctx, h));  // ne=(512, T)

    const char * dump = dump_dir;
    struct ggml_tensor * h_seanet_dump = NULL;
    if (dump) {
        // SEANet output naturally lands as channel-first ggml ne=(T, hidden).
        // The encoder_transformer and downsample dumps further down are
        // T-first numpy [T, hidden], so we transpose the SEANet view to
        // match before pinning it as a graph output.
        h_seanet_dump = ggml_cont(gctx, ggml_transpose(gctx, h_seanet));
        ggml_set_output(h_seanet_dump);
        ggml_set_name(h_seanet_dump, "seanet_out_dump");
        ggml_set_output(h_et);
        ggml_set_name(h_et, "enc_transformer_out");
    }

    ggml_set_name(h, "enc_hidden_out");
    ggml_set_output(h);

    struct ggml_cgraph * graph = ggml_new_graph_custom(gctx, n_max_nodes, false);
    ggml_build_forward_expand(graph, h);
    if (h_seanet_dump) {
        ggml_build_forward_expand(graph, h_seanet_dump);
    }

    if (!ggml_backend_sched_alloc_graph(pc->sched, graph)) {
        qt_log(QT_LOG_ERROR, "[Pipeline] encode sched_alloc_graph failed");
        ggml_backend_sched_reset(pc->sched);
        ggml_free(gctx);
        return {};
    }

    ggml_backend_tensor_set(audio_in, audio, 0, (size_t) n_samples * sizeof(float));

    std::vector<int32_t> pos_buf;
    qwen_encoder_build_positions(T_emb, pos_buf);
    ggml_backend_tensor_set(positions, pos_buf.data(), 0, pos_buf.size() * sizeof(int32_t));

    std::vector<float> mask_buf;
    qwen_encoder_build_causal_sliding_mask(T_emb, pc->enc_transformer.sliding_window, mask_buf);
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
        dump2("seanet-out",          h_seanet_dump);
        dump2("enc-transformer-out", h_et);
        dump2("codec-pre-fsq",       h);
    }

    // Read back the post-downsample hidden buffer for CPU-side RVQ encode.
    // Layout in ggml is [T, hidden] with T on ne[0]. The contiguous memory
    // walks T fast, hidden slow, which matches the `[T, hidden] row-major
    // index = t*hidden + c` convention expected by qwen_quantizer_encode_cpu.
    std::vector<float> hidden_host((size_t) T * (size_t) pc->qenc.hidden_size);
    ggml_backend_tensor_get(h, hidden_host.data(), 0, hidden_host.size() * sizeof(float));

    ggml_backend_sched_reset(pc->sched);
    ggml_free(gctx);

    return qwen_quantizer_encode_cpu(&pc->qenc_sem_host, &pc->qenc_aco_host, hidden_host.data(), T);
}

void pipeline_codec_free(PipelineCodec * pc) {
    if (pc->sched) {
        ggml_backend_sched_free(pc->sched);
        pc->sched = NULL;
    }
    qwen_quantizer_encode_free(&pc->qenc);
    qwen_encoder_downsample_free(&pc->enc_downsample);
    qwen_encoder_transformer_free(&pc->enc_transformer);
    qwen_seanet_encoder_free(&pc->seanet);
    if (pc->pre_conv_buf) {
        ggml_backend_buffer_free(pc->pre_conv_buf);
        pc->pre_conv_buf = NULL;
    }
    if (pc->pre_conv_ctx) {
        ggml_free(pc->pre_conv_ctx);
        pc->pre_conv_ctx = NULL;
    }
    qwen_dac_decoder_free(&pc->dac);
    qwen_upsample_stage_free(&pc->upsample);
    qwen_tokenizer_transformer_free(&pc->transformer);
    qwen_quantizer_decoder_free(&pc->qdec);
    if (pc->gguf.gguf) {
        gf_close(&pc->gguf);
    }
}
