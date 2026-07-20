#pragma once
// kv-cache.h: persistent per-layer KV cache for the Talker LM and the
// Code Predictor, batched over n_sets independent sequences. Kept
// minimal: the cache is sized at init for a fixed max sequence length
// and set count and never reallocates. Reset just rewinds cur_len.
//
// Layout per layer, both K and V :
//   4D ggml_tensor [hd, max_seq_len, n_kv, n_sets] f32, contiguous on hd
// plus one 3D view [hd, max_seq_len, n_kv] per set for the per-sequence
// prefill path and device side set copies. The batched decode graph
// views the 4D tensor directly with ne3 = N over sets [0, N).
//
// The 3D layout matches what the attention block uses for K, so the
// write path is a set_rows of the freshly RoPE'd K into the set view,
// and the read path is a view spanning the padded causal window on
// dim 1. V uses the same layout.

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

struct KVCache {
    int n_layers;
    int n_kv_heads;
    int head_dim;
    int max_seq_len;
    int n_sets;

    // Write head per set.
    std::vector<int> cur_len;

    // Per layer 4D tensors, both in `buffer` allocated below.
    std::vector<struct ggml_tensor *> k4;
    std::vector<struct ggml_tensor *> v4;

    // Per set 3D views into the 4D tensors, indexed [set * n_layers + layer].
    std::vector<struct ggml_tensor *> k;
    std::vector<struct ggml_tensor *> v;

    struct ggml_context * ctx;
    ggml_backend_buffer_t buffer;
};

static struct ggml_tensor * kv_cache_k(const KVCache * kv, int set, int layer) {
    return kv->k[(size_t) set * (size_t) kv->n_layers + (size_t) layer];
}

static struct ggml_tensor * kv_cache_v(const KVCache * kv, int set, int layer) {
    return kv->v[(size_t) set * (size_t) kv->n_layers + (size_t) layer];
}

// Allocate a fresh KV cache backed by a dedicated buffer on `backend`.
// Tensors are zero initialised: the padded attention window reads past
// cur_len and the masked tail must see finite values.
static bool kv_cache_init(KVCache *      kv,
                          int            n_layers,
                          int            n_kv_heads,
                          int            head_dim,
                          int            max_seq_len,
                          int            n_sets,
                          ggml_backend_t backend) {
    kv->n_layers    = n_layers;
    kv->n_kv_heads  = n_kv_heads;
    kv->head_dim    = head_dim;
    kv->max_seq_len = max_seq_len;
    kv->n_sets      = n_sets;
    kv->cur_len.assign((size_t) n_sets, 0);
    kv->k4.assign((size_t) n_layers, NULL);
    kv->v4.assign((size_t) n_layers, NULL);
    kv->k.assign((size_t) n_sets * (size_t) n_layers, NULL);
    kv->v.assign((size_t) n_sets * (size_t) n_layers, NULL);

    const size_t            n_tensors = (size_t) (2 * n_layers) * (size_t) (1 + n_sets) + 4;
    struct ggml_init_params gp        = {
        ggml_tensor_overhead() * n_tensors,
        NULL,
        true,
    };
    kv->ctx = ggml_init(gp);
    if (!kv->ctx) {
        fprintf(stderr, "[KVCache] FATAL: ggml_init failed\n");
        return false;
    }

    for (int l = 0; l < n_layers; l++) {
        kv->k4[(size_t) l] = ggml_new_tensor_4d(kv->ctx, GGML_TYPE_F32, head_dim, max_seq_len, n_kv_heads, n_sets);
        kv->v4[(size_t) l] = ggml_new_tensor_4d(kv->ctx, GGML_TYPE_F32, head_dim, max_seq_len, n_kv_heads, n_sets);
        char name[64];
        snprintf(name, sizeof(name), "kv_k_l%d", l);
        ggml_set_name(kv->k4[(size_t) l], name);
        snprintf(name, sizeof(name), "kv_v_l%d", l);
        ggml_set_name(kv->v4[(size_t) l], name);

        // Per set 3D views, created before the buffer allocation so
        // ggml_backend_alloc_ctx_tensors runs its view init on them
        // (buffer and data resolve against the owning 4D tensor).
        struct ggml_tensor * k4 = kv->k4[(size_t) l];
        struct ggml_tensor * v4 = kv->v4[(size_t) l];
        for (int s = 0; s < n_sets; s++) {
            size_t off = (size_t) s * k4->nb[3];
            kv->k[(size_t) s * (size_t) n_layers + (size_t) l] =
                ggml_view_3d(kv->ctx, k4, head_dim, max_seq_len, n_kv_heads, k4->nb[1], k4->nb[2], off);
            kv->v[(size_t) s * (size_t) n_layers + (size_t) l] =
                ggml_view_3d(kv->ctx, v4, head_dim, max_seq_len, n_kv_heads, v4->nb[1], v4->nb[2], off);
        }
    }

    kv->buffer = ggml_backend_alloc_ctx_tensors(kv->ctx, backend);
    if (!kv->buffer) {
        fprintf(stderr, "[KVCache] FATAL: backend allocation failed\n");
        ggml_free(kv->ctx);
        kv->ctx = NULL;
        return false;
    }

    // Zero-init the buffer so any out of bounds read returns a known value.
    ggml_backend_buffer_clear(kv->buffer, 0);

    size_t bytes_per_layer =
        (size_t) head_dim * (size_t) max_seq_len * (size_t) n_kv_heads * (size_t) n_sets * sizeof(float);
    size_t total_mb = (size_t) (2 * n_layers) * bytes_per_layer / (1024 * 1024);
    fprintf(stderr, "[KVCache] Allocated: %d layers, %d KV heads, head_dim %d, max_seq_len %d, %d sets -> %zu MB\n",
            n_layers, n_kv_heads, head_dim, max_seq_len, n_sets, total_mb);
    return true;
}

// Rewind one set so its next forward starts a fresh sequence.
static void kv_cache_reset(KVCache * kv, int set) {
    kv->cur_len[(size_t) set] = 0;
}

// Device side copy of one whole set into another through the
// persistent 3D views. Used by the batch engine to compact the active
// slot range after a retirement: the tail set moves into the freed one
// so the batched decode keeps viewing a consecutive [0, N) span.
static void kv_cache_copy_set(KVCache * kv, int src, int dst) {
    for (int l = 0; l < kv->n_layers; l++) {
        ggml_backend_tensor_copy(kv_cache_k(kv, src, l), kv_cache_k(kv, dst, l));
        ggml_backend_tensor_copy(kv_cache_v(kv, src, l), kv_cache_v(kv, dst, l));
    }
    kv->cur_len[(size_t) dst] = kv->cur_len[(size_t) src];
}

static void kv_cache_free(KVCache * kv) {
    if (kv->buffer) {
        ggml_backend_buffer_free(kv->buffer);
        kv->buffer = NULL;
    }
    if (kv->ctx) {
        ggml_free(kv->ctx);
        kv->ctx = NULL;
    }
    kv->k4.clear();
    kv->v4.clear();
    kv->k.clear();
    kv->v.clear();
    kv->cur_len.clear();
}
