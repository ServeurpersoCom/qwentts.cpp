#pragma once
// talker-lora.h: a PEFT LoRA adapter on the Talker's attention and MLP
// projections, read from its safetensors file and kept beside the base
// weights. Each projection becomes W x + B (A x); the quantized base is
// never touched, the rank-r factors live in F32 with alpha / r folded
// into B at load.
//
// Tensor naming follows PEFT on talker.model :
//   layers.{l}.self_attn.{q,k,v,o}_proj.lora_{A,B}.weight
//   layers.{l}.mlp.{gate,up,down}_proj.lora_{A,B}.weight
// A is [r, in] and B is [out, r] row-major, which ggml reads as
// ne = [in, r] and ne = [r, out]. Every layer must carry all seven
// pairs; a missing, extra or misshapen tensor fails the load.

#include "ggml-backend.h"
#include "ggml.h"
#include "qt-error.h"
#include "talker-weights.h"
#include "yyjson.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static bool talker_lora_read_file(const char * path, std::vector<uint8_t> & out) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        return false;
    }
    bool ok = fseek(f, 0, SEEK_END) == 0;
    long n  = ok ? ftell(f) : -1;
    ok      = ok && n > 8 && fseek(f, 0, SEEK_SET) == 0;
    if (ok) {
        out.resize((size_t) n);
        ok = fread(out.data(), 1, out.size(), f) == out.size();
    }
    fclose(f);
    return ok;
}

static TalkerLora * talker_lora_slot(TalkerWeights * tw, int l, const std::string & proj) {
    if (l < 0 || l >= tw->num_hidden_layers) {
        return nullptr;
    }
    TalkerLayer & layer = tw->layers[(size_t) l];
    if (proj == "self_attn.q_proj") return &layer.attn.q_lora;
    if (proj == "self_attn.k_proj") return &layer.attn.k_lora;
    if (proj == "self_attn.v_proj") return &layer.attn.v_lora;
    if (proj == "self_attn.o_proj") return &layer.attn.o_lora;
    if (proj == "mlp.gate_proj") return &layer.mlp.gate_lora;
    if (proj == "mlp.up_proj") return &layer.mlp.up_lora;
    if (proj == "mlp.down_proj") return &layer.mlp.down_lora;
    return nullptr;
}

static const struct ggml_tensor * talker_lora_base(const TalkerWeights * tw, int l, const TalkerLora * slot) {
    const TalkerLayer & layer = tw->layers[(size_t) l];
    if (slot == &layer.attn.q_lora) return layer.attn.q_proj_w;
    if (slot == &layer.attn.k_lora) return layer.attn.k_proj_w;
    if (slot == &layer.attn.v_lora) return layer.attn.v_proj_w;
    if (slot == &layer.attn.o_lora) return layer.attn.o_proj_w;
    if (slot == &layer.mlp.gate_lora) return layer.mlp.gate_proj_w;
    if (slot == &layer.mlp.up_lora) return layer.mlp.up_proj_w;
    return layer.mlp.down_proj_w;
}

static bool talker_lora_load(TalkerWeights * tw, const char * path, float scale, ggml_backend_t backend) {
    std::vector<uint8_t> file;
    if (!talker_lora_read_file(path, file)) {
        qt_log(QT_LOG_ERROR, "[TalkerLoRA] FATAL: cannot read %s", path);
        return false;
    }
    uint64_t header_len = 0;
    memcpy(&header_len, file.data(), 8);
    if (header_len == 0 || header_len > file.size() - 8) {
        qt_log(QT_LOG_ERROR, "[TalkerLoRA] FATAL: %s is not a safetensors file", path);
        return false;
    }
    const uint8_t * payload = file.data() + 8 + header_len;
    const size_t    payload_len = file.size() - 8 - (size_t) header_len;

    yyjson_doc * doc = yyjson_read((const char *) file.data() + 8, (size_t) header_len, 0);
    if (!doc) {
        qt_log(QT_LOG_ERROR, "[TalkerLoRA] FATAL: %s has an unreadable header", path);
        return false;
    }

    struct Entry {
        TalkerLora *  slot;
        bool          is_b;
        int           layer;
        int64_t       rows, cols;
        const void *  src;
        std::string   dtype;
    };
    std::vector<Entry> entries;
    bool               ok = true;

    yyjson_val * root = yyjson_doc_get_root(doc);
    size_t       idx, max;
    yyjson_val * key, *val;
    yyjson_obj_foreach(root, idx, max, key, val) {
        const char * name = yyjson_get_str(key);
        if (std::strcmp(name, "__metadata__") == 0) {
            continue;
        }
        int  l = -1;
        char proj[64];
        char ab = 0;
        if (sscanf(name, "layers.%d.%63[A-Za-z_.]", &l, proj) != 2) {
            qt_log(QT_LOG_ERROR, "[TalkerLoRA] FATAL: unexpected tensor %s", name);
            ok = false;
            break;
        }
        std::string p(proj);
        const std::string tail_a = ".lora_A.weight", tail_b = ".lora_B.weight";
        if (p.size() > tail_a.size() && p.compare(p.size() - tail_a.size(), tail_a.size(), tail_a) == 0) {
            ab = 'A';
        } else if (p.size() > tail_b.size() && p.compare(p.size() - tail_b.size(), tail_b.size(), tail_b) == 0) {
            ab = 'B';
        }
        TalkerLora * slot = ab ? talker_lora_slot(tw, l, p.substr(0, p.size() - tail_a.size())) : nullptr;
        yyjson_val * shape = yyjson_obj_get(val, "shape");
        yyjson_val * offs  = yyjson_obj_get(val, "data_offsets");
        const char * dtype = yyjson_get_str(yyjson_obj_get(val, "dtype"));
        if (!slot || yyjson_arr_size(shape) != 2 || yyjson_arr_size(offs) != 2 || !dtype) {
            qt_log(QT_LOG_ERROR, "[TalkerLoRA] FATAL: unexpected tensor %s", name);
            ok = false;
            break;
        }
        const uint64_t begin = yyjson_get_uint(yyjson_arr_get(offs, 0));
        const uint64_t end   = yyjson_get_uint(yyjson_arr_get(offs, 1));
        Entry          e{ slot, ab == 'B', l, (int64_t) yyjson_get_uint(yyjson_arr_get(shape, 0)),
                 (int64_t) yyjson_get_uint(yyjson_arr_get(shape, 1)), payload + begin, dtype };
        const size_t   elem = e.dtype == "F32" ? 4 : 2;
        if ((e.dtype != "F32" && e.dtype != "BF16" && e.dtype != "F16") || end > payload_len || begin > end ||
            end - begin != (uint64_t) (e.rows * e.cols) * elem) {
            qt_log(QT_LOG_ERROR, "[TalkerLoRA] FATAL: tensor %s has dtype %s or a size that does not match", name,
                   dtype);
            ok = false;
            break;
        }
        entries.push_back(e);
    }
    yyjson_doc_free(doc);
    if (!ok) {
        return false;
    }
    if (entries.size() != (size_t) tw->num_hidden_layers * 14) {
        qt_log(QT_LOG_ERROR, "[TalkerLoRA] FATAL: %zu tensors, expected %d (7 A/B pairs per layer)", entries.size(),
               tw->num_hidden_layers * 14);
        return false;
    }

    WeightCtx wctx;
    wctx_init(&wctx, (int) entries.size());
    int rank = 0;
    for (const Entry & e : entries) {
        const struct ggml_tensor * w   = talker_lora_base(tw, e.layer, e.slot);
        const int64_t              in  = w->ne[0];
        const int64_t              out = w->ne[1];
        const int64_t              r   = e.is_b ? e.cols : e.rows;
        const bool fits = e.is_b ? (e.rows == out) : (e.cols == in);
        if (!fits || r <= 0 || (rank && r != rank)) {
            qt_log(QT_LOG_ERROR, "[TalkerLoRA] FATAL: layer %d factor %c is [%lld, %lld], base is [%lld, %lld]",
                   e.layer, e.is_b ? 'B' : 'A', (long long) e.rows, (long long) e.cols, (long long) out,
                   (long long) in);
            ggml_free(wctx.ctx);
            return false;
        }
        rank = (int) r;

        struct ggml_tensor * t = ggml_new_tensor_2d(wctx.ctx, GGML_TYPE_F32, e.cols, e.rows);
        const size_t         n = (size_t) (e.rows * e.cols);
        std::unique_ptr<float[]> buf(new float[n]);
        const float              mul = e.is_b ? scale : 1.0f;
        for (size_t i = 0; i < n; i++) {
            float v;
            if (e.dtype == "F32") {
                memcpy(&v, (const uint8_t *) e.src + 4 * i, 4);
            } else {
                uint16_t h;
                memcpy(&h, (const uint8_t *) e.src + 2 * i, 2);
                v = e.dtype == "BF16" ? ggml_bf16_to_fp32(*(const ggml_bf16_t *) &h) : ggml_fp16_to_fp32(h);
            }
            buf[i] = v * mul;
        }
        wctx.pending.push_back({ t, buf.get(), n * sizeof(float), 0 });
        wctx.staging.push_back(std::move(buf));
        (e.is_b ? e.slot->b : e.slot->a) = t;
    }
    for (const TalkerLayer & layer : tw->layers) {
        const TalkerLora * all[] = { &layer.attn.q_lora,  &layer.attn.k_lora,  &layer.attn.v_lora, &layer.attn.o_lora,
                                     &layer.mlp.gate_lora, &layer.mlp.up_lora, &layer.mlp.down_lora };
        for (const TalkerLora * s : all) {
            if (!s->a || !s->b) {
                qt_log(QT_LOG_ERROR, "[TalkerLoRA] FATAL: a projection has only one of its two factors");
                ggml_free(wctx.ctx);
                return false;
            }
        }
    }
    if (!wctx_alloc(&wctx, backend)) {
        ggml_free(wctx.ctx);
        return false;
    }
    tw->lora_ctx = wctx.ctx;
    tw->lora_buf = wctx.buffer;
    qt_log(QT_LOG_INFO, "[TalkerLoRA] Loaded %s: rank %d, scale %.3f, %zu tensors", path, rank, (double) scale,
           entries.size());
    return true;
}
