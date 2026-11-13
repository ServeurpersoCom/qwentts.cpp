#!/usr/bin/env python3
# convert.py: Qwen3-TTS HF checkpoint -> GGUF.
#
# Reads a HF safetensors checkpoint and writes a GGUF F32 file with a
# flat tensor naming scheme and arch-specific KV metadata. Two converter
# paths live here :
#   tokenizer  Mimi-style 12Hz audio codec (encoder + decoder + RVQ) with
#              precomputed codebook embeddings (embedding_sum / clamp(
#              cluster_usage, min=eps)) for single-shot F.embedding lookup.
#   talker     Qwen3-style autoregressive LM (talker + 5-layer code
#              predictor MTP head) plus the optional ECAPA-TDNN speaker
#              encoder shipped with Base checkpoints. BPE vocab and
#              merges of the Qwen2 tokenizer are embedded in the GGUF.

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open

import gguf

# RVQ codebook epsilon clamp, matches Qwen3TTS EuclideanCodebook.epsilon
RVQ_EPS = 1e-5

# Tokenizer 12Hz fixed shape: 4 DAC blocks (one per upsample stride) and
# 3 residual units per block (dilations 1, 3, 9).
DAC_NUM_BLOCKS = 4
DAC_RU_PER_BLOCK = 3


def rename_pre_transformer(name: str) -> str:
    # decoder.pre_transformer.layers.{i}.input_layernorm.weight             -> tok_dec.pre_tfm.blk.{i}.attn_norm.weight
    # decoder.pre_transformer.layers.{i}.post_attention_layernorm.weight    -> tok_dec.pre_tfm.blk.{i}.ffn_norm.weight
    # decoder.pre_transformer.layers.{i}.self_attn.q_proj.weight            -> tok_dec.pre_tfm.blk.{i}.attn_q.weight
    # decoder.pre_transformer.layers.{i}.self_attn.{k,v,o}_proj.weight      -> tok_dec.pre_tfm.blk.{i}.attn_{k,v,output}.weight
    # decoder.pre_transformer.layers.{i}.self_attn_layer_scale.scale        -> tok_dec.pre_tfm.blk.{i}.attn_scale
    # decoder.pre_transformer.layers.{i}.mlp.{gate,up,down}_proj.weight     -> tok_dec.pre_tfm.blk.{i}.ffn_{gate,up,down}.weight
    # decoder.pre_transformer.layers.{i}.mlp_layer_scale.scale              -> tok_dec.pre_tfm.blk.{i}.ffn_scale
    assert name.startswith("decoder.pre_transformer.layers.")
    parts = name.split(".")
    idx   = parts[3]
    rest  = parts[4:]
    if rest == ["input_layernorm", "weight"]:
        return f"tok_dec.pre_tfm.blk.{idx}.attn_norm.weight"
    if rest == ["post_attention_layernorm", "weight"]:
        return f"tok_dec.pre_tfm.blk.{idx}.ffn_norm.weight"
    if rest[:2] == ["self_attn", "q_proj"]:
        return f"tok_dec.pre_tfm.blk.{idx}.attn_q.{rest[-1]}"
    if rest[:2] == ["self_attn", "k_proj"]:
        return f"tok_dec.pre_tfm.blk.{idx}.attn_k.{rest[-1]}"
    if rest[:2] == ["self_attn", "v_proj"]:
        return f"tok_dec.pre_tfm.blk.{idx}.attn_v.{rest[-1]}"
    if rest[:2] == ["self_attn", "o_proj"]:
        return f"tok_dec.pre_tfm.blk.{idx}.attn_output.{rest[-1]}"
    if rest == ["self_attn_layer_scale", "scale"]:
        return f"tok_dec.pre_tfm.blk.{idx}.attn_scale"
    if rest[:2] == ["mlp", "gate_proj"]:
        return f"tok_dec.pre_tfm.blk.{idx}.ffn_gate.{rest[-1]}"
    if rest[:2] == ["mlp", "up_proj"]:
        return f"tok_dec.pre_tfm.blk.{idx}.ffn_up.{rest[-1]}"
    if rest[:2] == ["mlp", "down_proj"]:
        return f"tok_dec.pre_tfm.blk.{idx}.ffn_down.{rest[-1]}"
    if rest == ["mlp_layer_scale", "scale"]:
        return f"tok_dec.pre_tfm.blk.{idx}.ffn_scale"
    raise ValueError(f"Unhandled pre_transformer tensor : {name}")


def rename_pre_conv(name: str) -> str:
    # decoder.pre_conv.conv.{weight,bias}  ->  tok_dec.pre_conv.{weight,bias}
    suffix = name.rsplit(".", 1)[-1]
    return f"tok_dec.pre_conv.{suffix}"


def rename_upsample(name: str) -> str:
    # Two ModuleList per stage: index 0 is the CausalTransConv, index 1
    # is the ConvNeXt block.
    # decoder.upsample.{i}.0.conv.{weight,bias}            -> tok_dec.upsample.{i}.conv.{weight,bias}
    # decoder.upsample.{i}.1.dwconv.conv.{weight,bias}     -> tok_dec.upsample.{i}.dwconv.{weight,bias}
    # decoder.upsample.{i}.1.norm.{weight,bias}            -> tok_dec.upsample.{i}.norm.{weight,bias}
    # decoder.upsample.{i}.1.pwconv{1,2}.{weight,bias}     -> tok_dec.upsample.{i}.pwconv{1,2}.{weight,bias}
    # decoder.upsample.{i}.1.gamma                         -> tok_dec.upsample.{i}.gamma
    parts = name.split(".")
    assert parts[0] == "decoder" and parts[1] == "upsample"
    block_idx = parts[2]
    sub       = parts[3]
    if sub == "0":
        suffix = parts[-1]
        return f"tok_dec.upsample.{block_idx}.conv.{suffix}"
    if sub == "1":
        if parts[4] == "dwconv":
            suffix = parts[-1]
            return f"tok_dec.upsample.{block_idx}.dwconv.{suffix}"
        if parts[4] in ("norm", "pwconv1", "pwconv2"):
            suffix = parts[-1]
            return f"tok_dec.upsample.{block_idx}.{parts[4]}.{suffix}"
        if parts[4] == "gamma":
            return f"tok_dec.upsample.{block_idx}.gamma"
    raise ValueError(f"Unhandled upsample tensor : {name}")


def precompute_codebook(embedding_sum: np.ndarray, cluster_usage: np.ndarray) -> np.ndarray:
    # Qwen3TTS EuclideanCodebook.decode :
    #   embedding = embedding_sum / cluster_usage.clamp(min=epsilon)[:, None]
    # Stored in F32 to match runtime precision of the codebook lookup. We
    # pre-divide at convert time so the runtime can read a ready to use
    # F.embedding table straight from the GGUF.
    usage = np.clip(cluster_usage, RVQ_EPS, None).astype(np.float32)
    sums  = embedding_sum.astype(np.float32)
    return sums / usage[:, None]


def rename_decoder_chain(name: str) -> str:
    # Direct {i} preservation per the koboldcpp tok_dec convention :
    # decoder.decoder.{i}.block.0.alpha                       -> tok_dec.dec.{i}.snake.alpha
    # decoder.decoder.{i}.block.0.beta                        -> tok_dec.dec.{i}.snake.beta
    # decoder.decoder.{i}.block.1.conv.{weight,bias}          -> tok_dec.dec.{i}.conv_t.{weight,bias}
    # decoder.decoder.{i}.block.{j}.act{1,2}.{alpha,beta}     -> tok_dec.dec.{i}.res.{j-2}.act{1,2}.{alpha,beta}
    # decoder.decoder.{i}.block.{j}.conv{1,2}.conv.{w,b}      -> tok_dec.dec.{i}.res.{j-2}.conv{1,2}.{w,b}
    # decoder.decoder.0.conv.{weight,bias}                    -> tok_dec.dec.0.conv.{weight,bias}
    # decoder.decoder.5.{alpha,beta}                          -> tok_dec.dec.5.snake.{alpha,beta}
    # decoder.decoder.6.conv.{weight,bias}                    -> tok_dec.dec.6.conv.{weight,bias}
    parts = name.split(".")
    assert parts[0] == "decoder" and parts[1] == "decoder"
    idx = int(parts[2])
    if idx == 0:
        return f"tok_dec.dec.0.conv.{parts[-1]}"
    if idx == 5:
        return f"tok_dec.dec.5.snake.{parts[-1]}"
    if idx == 6:
        return f"tok_dec.dec.6.conv.{parts[-1]}"
    sub = int(parts[4])
    if sub == 0:
        return f"tok_dec.dec.{idx}.snake.{parts[-1]}"
    if sub == 1:
        return f"tok_dec.dec.{idx}.conv_t.{parts[-1]}"
    if sub in (2, 3, 4):
        ru   = sub - 2
        rest = parts[5]
        if rest in ("act1", "act2"):
            return f"tok_dec.dec.{idx}.res.{ru}.{rest}.{parts[-1]}"
        if rest in ("conv1", "conv2"):
            return f"tok_dec.dec.{idx}.res.{ru}.{rest}.{parts[-1]}"
    raise ValueError(f"Unhandled decoder chain tensor : {name}")


def rename_seanet(name: str) -> str:
    # encoder.encoder.layers.{idx}.conv.{weight,bias}                 -> tok_enc.conv.{idx}.{weight,bias}
    # encoder.encoder.layers.{idx}.block.{j}.conv.{weight,bias}       -> tok_enc.res.{idx}.blk.{j}.{weight,bias}
    # We pass the raw Python ModuleList index through so the loader
    # reconstructs the SEANet topology from the upsampling_ratios array.
    parts = name.split(".")
    assert parts[0] == "encoder" and parts[1] == "encoder" and parts[2] == "layers"
    idx    = parts[3]
    suffix = parts[-1]
    if len(parts) == 6 and parts[4] == "conv":
        return f"tok_enc.conv.{idx}.{suffix}"
    if len(parts) == 8 and parts[4] == "block" and parts[6] == "conv":
        sub = parts[5]
        return f"tok_enc.res.{idx}.blk.{sub}.{suffix}"
    raise ValueError(f"Unhandled SEANet tensor : {name}")


def rename_encoder_transformer(name: str) -> str:
    # encoder.encoder_transformer.layers.{i}.input_layernorm.{weight,bias}            -> tok_enc.blk.{i}.attn_norm.{weight,bias}
    # encoder.encoder_transformer.layers.{i}.post_attention_layernorm.{weight,bias}   -> tok_enc.blk.{i}.ffn_norm.{weight,bias}
    # encoder.encoder_transformer.layers.{i}.self_attn.{q,k,v,o}_proj.weight          -> tok_enc.blk.{i}.attn_{q,k,v,output}.weight
    # encoder.encoder_transformer.layers.{i}.self_attn_layer_scale.scale              -> tok_enc.blk.{i}.attn_scale
    # encoder.encoder_transformer.layers.{i}.mlp.{fc1,fc2}.weight                     -> tok_enc.blk.{i}.{ffn_up,ffn_down}.weight
    # encoder.encoder_transformer.layers.{i}.mlp_layer_scale.scale                    -> tok_enc.blk.{i}.ffn_scale
    assert name.startswith("encoder.encoder_transformer.layers.")
    parts = name.split(".")
    idx   = parts[3]
    rest  = parts[4:]
    if rest[:1] == ["input_layernorm"]:
        return f"tok_enc.blk.{idx}.attn_norm.{rest[-1]}"
    if rest[:1] == ["post_attention_layernorm"]:
        return f"tok_enc.blk.{idx}.ffn_norm.{rest[-1]}"
    if rest[:2] == ["self_attn", "q_proj"]:
        return f"tok_enc.blk.{idx}.attn_q.{rest[-1]}"
    if rest[:2] == ["self_attn", "k_proj"]:
        return f"tok_enc.blk.{idx}.attn_k.{rest[-1]}"
    if rest[:2] == ["self_attn", "v_proj"]:
        return f"tok_enc.blk.{idx}.attn_v.{rest[-1]}"
    if rest[:2] == ["self_attn", "o_proj"]:
        return f"tok_enc.blk.{idx}.attn_output.{rest[-1]}"
    if rest == ["self_attn_layer_scale", "scale"]:
        return f"tok_enc.blk.{idx}.attn_scale"
    if rest[:2] == ["mlp", "fc1"]:
        return f"tok_enc.blk.{idx}.ffn_up.{rest[-1]}"
    if rest[:2] == ["mlp", "fc2"]:
        return f"tok_enc.blk.{idx}.ffn_down.{rest[-1]}"
    if rest == ["mlp_layer_scale", "scale"]:
        return f"tok_enc.blk.{idx}.ffn_scale"
    raise ValueError(f"Unhandled encoder transformer tensor : {name}")


def rename_encoder_downsample(name: str) -> str:
    # encoder.downsample.conv.{weight,bias}  ->  tok_enc.downsample.{weight,bias}
    suffix = name.rsplit(".", 1)[-1]
    return f"tok_enc.downsample.{suffix}"


def rename_encoder_quantizer_proj(name: str) -> str:
    # encoder.quantizer.{semantic|acoustic}_residual_vector_quantizer.{input_proj,output_proj}.weight ->
    #   tok_enc.vq_{semantic|acoustic}.{input_proj,output_proj}.weight
    if "input_proj" in name:
        proj = "input_proj"
    elif "output_proj" in name:
        proj = "output_proj"
    else:
        raise ValueError(f"Unknown encoder quantizer proj : {name}")
    if "semantic_residual_vector_quantizer" in name:
        return f"tok_enc.vq_semantic.{proj}.weight"
    if "acoustic_residual_vector_quantizer" in name:
        return f"tok_enc.vq_acoustic.{proj}.weight"
    raise ValueError(f"Unhandled encoder quantizer proj : {name}")


def convert_tokenizer_12hz(checkpoint_dir: Path, out_path: Path) -> int:
    cfg_path = checkpoint_dir / "config.json"
    st_path = checkpoint_dir / "model.safetensors"
    if not cfg_path.is_file() or not st_path.is_file():
        print(f"[Convert] FATAL: missing checkpoint files in {checkpoint_dir}")
        return 1

    cfg = json.loads(cfg_path.read_text())
    dec = cfg["decoder_config"]
    enc = cfg["encoder_config"]

    arch = "qwen3-tts-tokenizer"
    writer = gguf.GGUFWriter(str(out_path), arch)

    writer.add_string("general.name", "Qwen3-TTS-Tokenizer-12Hz")

    # Tokenizer-level metadata
    writer.add_uint32("qwen3-tts-tokenizer.input_sample_rate", cfg["input_sample_rate"])
    writer.add_uint32("qwen3-tts-tokenizer.output_sample_rate", cfg["output_sample_rate"])
    writer.add_uint32("qwen3-tts-tokenizer.decode_upsample_rate", cfg["decode_upsample_rate"])
    writer.add_uint32("qwen3-tts-tokenizer.encode_downsample_rate", cfg["encode_downsample_rate"])
    writer.add_uint32("qwen3-tts-tokenizer.encoder_valid_num_quantizers", cfg["encoder_valid_num_quantizers"])

    # Decoder-level metadata
    writer.add_uint32("qwen3-tts-tokenizer.decoder.latent_dim", dec["latent_dim"])
    writer.add_uint32("qwen3-tts-tokenizer.decoder.codebook_dim", dec["codebook_dim"])
    writer.add_uint32("qwen3-tts-tokenizer.decoder.codebook_size", dec["codebook_size"])
    writer.add_uint32("qwen3-tts-tokenizer.decoder.decoder_dim", dec["decoder_dim"])
    writer.add_uint32("qwen3-tts-tokenizer.decoder.hidden_size", dec["hidden_size"])
    writer.add_uint32("qwen3-tts-tokenizer.decoder.intermediate_size", dec["intermediate_size"])
    writer.add_uint32("qwen3-tts-tokenizer.decoder.head_dim", dec["head_dim"])
    writer.add_uint32("qwen3-tts-tokenizer.decoder.num_attention_heads", dec["num_attention_heads"])
    writer.add_uint32("qwen3-tts-tokenizer.decoder.num_key_value_heads", dec["num_key_value_heads"])
    writer.add_uint32("qwen3-tts-tokenizer.decoder.num_hidden_layers", dec["num_hidden_layers"])
    writer.add_uint32("qwen3-tts-tokenizer.decoder.num_quantizers", dec["num_quantizers"])
    writer.add_uint32("qwen3-tts-tokenizer.decoder.num_semantic_quantizers", dec["num_semantic_quantizers"])
    writer.add_float32("qwen3-tts-tokenizer.decoder.rms_norm_eps", dec["rms_norm_eps"])
    writer.add_float32("qwen3-tts-tokenizer.decoder.rope_theta", float(dec["rope_theta"]))
    writer.add_uint32("qwen3-tts-tokenizer.decoder.sliding_window", dec["sliding_window"])
    writer.add_float32("qwen3-tts-tokenizer.decoder.layer_scale_initial_scale", dec["layer_scale_initial_scale"])
    writer.add_array("qwen3-tts-tokenizer.decoder.upsample_rates", dec["upsample_rates"])
    writer.add_array("qwen3-tts-tokenizer.decoder.upsampling_ratios", dec["upsampling_ratios"])
    writer.add_uint32("qwen3-tts-tokenizer.decoder.vector_quantization_hidden_dim", dec["vector_quantization_hidden_dimension"])
    writer.add_uint32("qwen3-tts-tokenizer.decoder.codebook_dim_internal", 256)  # the actual codebook vector dim before output_proj

    # Encoder-level metadata. The encoder is a Mimi-style stack: SEANet conv
    # downsampler -> 8-layer Mimi transformer -> 1 conv downsample -> RVQ.
    writer.add_uint32("qwen3-tts-tokenizer.encoder.num_filters", enc["num_filters"])
    writer.add_uint32("qwen3-tts-tokenizer.encoder.kernel_size", enc["kernel_size"])
    writer.add_uint32("qwen3-tts-tokenizer.encoder.last_kernel_size", enc["last_kernel_size"])
    writer.add_uint32("qwen3-tts-tokenizer.encoder.residual_kernel_size", enc["residual_kernel_size"])
    writer.add_uint32("qwen3-tts-tokenizer.encoder.num_residual_layers", enc["num_residual_layers"])
    writer.add_uint32("qwen3-tts-tokenizer.encoder.dilation_growth_rate", enc["dilation_growth_rate"])
    writer.add_uint32("qwen3-tts-tokenizer.encoder.compress", enc["compress"])
    writer.add_array("qwen3-tts-tokenizer.encoder.upsampling_ratios", enc["upsampling_ratios"])
    writer.add_uint32("qwen3-tts-tokenizer.encoder.hidden_size", enc["hidden_size"])
    writer.add_uint32("qwen3-tts-tokenizer.encoder.intermediate_size", enc["intermediate_size"])
    writer.add_uint32("qwen3-tts-tokenizer.encoder.head_dim", enc["head_dim"])
    writer.add_uint32("qwen3-tts-tokenizer.encoder.num_attention_heads", enc["num_attention_heads"])
    writer.add_uint32("qwen3-tts-tokenizer.encoder.num_key_value_heads", enc["num_key_value_heads"])
    writer.add_uint32("qwen3-tts-tokenizer.encoder.num_hidden_layers", enc["num_hidden_layers"])
    writer.add_float32("qwen3-tts-tokenizer.encoder.norm_eps", enc["norm_eps"])
    writer.add_float32("qwen3-tts-tokenizer.encoder.rope_theta", float(enc["rope_theta"]))
    writer.add_float32("qwen3-tts-tokenizer.encoder.layer_scale_initial_scale", enc["layer_scale_initial_scale"])
    writer.add_uint32("qwen3-tts-tokenizer.encoder.codebook_dim", enc["codebook_dim"])
    writer.add_uint32("qwen3-tts-tokenizer.encoder.codebook_size", enc["codebook_size"])
    writer.add_uint32("qwen3-tts-tokenizer.encoder.num_quantizers", enc["num_quantizers"])
    writer.add_uint32("qwen3-tts-tokenizer.encoder.num_semantic_quantizers", enc["num_semantic_quantizers"])
    writer.add_uint32("qwen3-tts-tokenizer.encoder.vector_quantization_hidden_dim", enc["vector_quantization_hidden_dimension"])

    # Walk safetensors
    n_added = 0
    n_skipped_encoder_extra_acoustic = 0
    # Truncation policy: the encoder ships 32 acoustic codebooks (semantic 1
    # + acoustic 31), but encoder_valid_num_quantizers=16 means only the first
    # 16 (1 semantic + 15 acoustic) are consumed at encode time and matched
    # by the decoder. We drop the unused acoustic 15..30 to stay aligned with
    # the decoder side and shrink the GGUF.
    encoder_valid = cfg["encoder_valid_num_quantizers"]
    encoder_acoustic_kept = encoder_valid - cfg["encoder_config"]["num_semantic_quantizers"]

    # Pair embed_sum and cluster_usage entries by (origin, side, layer)
    # for fusion at the end of the walk.
    rvq_buffers = {}

    with safe_open(str(st_path), framework="pt") as f:
        all_keys = list(f.keys())
        for k in all_keys:
            t = f.get_tensor(k)
            arr = t.numpy().astype(np.float32)

            # Encoder SEANet conv stack
            if k.startswith("encoder.encoder.layers."):
                writer.add_tensor(rename_seanet(k), arr)
                n_added += 1
                continue

            # Encoder Mimi-style transformer
            if k.startswith("encoder.encoder_transformer."):
                writer.add_tensor(rename_encoder_transformer(k), arr)
                n_added += 1
                continue

            # Encoder downsample (final conv k=4 stride=2 between transformer
            # output and the RVQ)
            if k.startswith("encoder.downsample."):
                writer.add_tensor(rename_encoder_downsample(k), arr)
                n_added += 1
                continue

            # Encoder quantizer
            if k.startswith("encoder.quantizer."):
                # input_proj projects 512 -> 256 before the codebook lookup,
                # output_proj projects 256 -> 512 to reconstruct the residual
                # during the RVQ encode loop. Both are needed at encode time.
                if k.endswith(".input_proj.weight") or k.endswith(".output_proj.weight"):
                    writer.add_tensor(rename_encoder_quantizer_proj(k), arr)
                    n_added += 1
                    continue
                # Codebook tensors : pair embed_sum and cluster_usage and
                # emit a single pre-fused codebook tensor (koboldcpp name,
                # our pre-fusion semantics so the runtime can read F.embedding
                # straight from disk with no extra division).
                if "._codebook." in k or ".codebook." in k:
                    parts     = k.split(".")
                    side_full = parts[2]
                    if side_full.startswith("semantic"):
                        side = "semantic"
                    elif side_full.startswith("acoustic"):
                        side = "acoustic"
                    else:
                        raise ValueError(f"Unknown encoder quantizer side : {k}")
                    layer_idx = int(parts[4])
                    field_raw = parts[-1]

                    # Apply truncation : only keep the first encoder_acoustic_kept
                    # acoustic layers ; semantic always has 1 layer.
                    if side == "acoustic" and layer_idx >= encoder_acoustic_kept:
                        n_skipped_encoder_extra_acoustic += 1
                        continue

                    if field_raw == "initialized":
                        # Boolean flag, not used at runtime
                        continue
                    field = "embedding_sum" if field_raw == "embed_sum" else field_raw
                    rvq_buffers.setdefault(("encoder", side, layer_idx), {})[field] = arr
                    continue
                raise ValueError(f"Unhandled encoder quantizer tensor : {k}")

            # Decoder pre-conv
            if k.startswith("decoder.pre_conv."):
                writer.add_tensor(rename_pre_conv(k), arr)
                n_added += 1
                continue

            # Decoder pre-transformer
            if k.startswith("decoder.pre_transformer."):
                # Top level pre transformer projections and norm.
                if k == "decoder.pre_transformer.input_proj.weight":
                    writer.add_tensor("tok_dec.pre_tfm.input_proj.weight", arr)
                    n_added += 1
                    continue
                if k == "decoder.pre_transformer.input_proj.bias":
                    writer.add_tensor("tok_dec.pre_tfm.input_proj.bias", arr)
                    n_added += 1
                    continue
                if k == "decoder.pre_transformer.output_proj.weight":
                    writer.add_tensor("tok_dec.pre_tfm.output_proj.weight", arr)
                    n_added += 1
                    continue
                if k == "decoder.pre_transformer.output_proj.bias":
                    writer.add_tensor("tok_dec.pre_tfm.output_proj.bias", arr)
                    n_added += 1
                    continue
                if k == "decoder.pre_transformer.norm.weight":
                    writer.add_tensor("tok_dec.pre_tfm.norm.weight", arr)
                    n_added += 1
                    continue
                # Per layer transformer block.
                if k.startswith("decoder.pre_transformer.layers."):
                    writer.add_tensor(rename_pre_transformer(k), arr)
                    n_added += 1
                    continue
                raise ValueError(f"Unhandled decoder.pre_transformer tensor : {k}")

            # Decoder upsample stage
            if k.startswith("decoder.upsample."):
                writer.add_tensor(rename_upsample(k), arr)
                n_added += 1
                continue

            # Decoder DAC chain (decoder.decoder.{0..6}.*)
            if k.startswith("decoder.decoder."):
                writer.add_tensor(rename_decoder_chain(k), arr)
                n_added += 1
                continue

            # Decoder quantizer side
            if k.startswith("decoder.quantizer."):
                # output_proj is needed at decode time
                if k.endswith(".output_proj.weight"):
                    if "rvq_first" in k:
                        writer.add_tensor("tok_dec.vq_first.output_proj.weight", arr)
                    elif "rvq_rest" in k:
                        writer.add_tensor("tok_dec.vq_rest.output_proj.weight", arr)
                    else:
                        raise ValueError(f"Unknown quantizer output_proj : {k}")
                    n_added += 1
                    continue
                # input_proj on the decoder side has the same value as the
                # encoder side and is unused at decode time : skip rather
                # than carry a duplicate.
                if k.endswith(".input_proj.weight"):
                    continue
                # Codebook tensors : pair embed_sum and cluster_usage and
                # emit a single pre-fused codebook tensor under the
                # koboldcpp tok_dec.vq_{first,rest} namespace.
                if "._codebook." in k:
                    parts     = k.split(".")
                    side      = parts[2]
                    layer_idx = int(parts[5])
                    field     = parts[-1]
                    rvq_buffers.setdefault(("decoder", side, layer_idx), {})[field] = arr
                    continue
                raise ValueError(f"Unhandled quantizer tensor : {k}")

            raise ValueError(f"Unhandled top-level tensor : {k}")

    # Fuse paired embed_sum and cluster_usage into a single pre-divided
    # codebook tensor per layer per side. Output names follow koboldcpp.
    n_codebooks_emitted = 0
    for (origin, side, layer_idx), buf in sorted(rvq_buffers.items()):
        if "cluster_usage" not in buf or "embedding_sum" not in buf:
            print(f"[Convert] WARNING: incomplete codebook ({origin}, {side}, {layer_idx}): {list(buf.keys())}")
            continue
        emb = precompute_codebook(buf["embedding_sum"], buf["cluster_usage"])
        if origin == "decoder":
            # HF source uses rvq_first / rvq_rest, koboldcpp emits vq_first
            # / vq_rest. Strip the leading "r" so the output matches the
            # tok_dec.vq_{first,rest}.{layer}.codebook convention used by
            # the runtime loaders and the koboldcpp HF release.
            assert side in ("rvq_first", "rvq_rest"), f"Unexpected decoder side : {side}"
            out_side = side[1:]
            writer.add_tensor(f"tok_dec.{out_side}.{layer_idx}.codebook", emb)
        else:
            writer.add_tensor(f"tok_enc.vq_{side}.{layer_idx}.codebook", emb)
        n_codebooks_emitted += 1

    print(f"[Convert] Tensors: {n_added} written, {n_codebooks_emitted} codebooks fused")
    print(f"[Convert] Truncate: {n_skipped_encoder_extra_acoustic} encoder acoustic codebook tensors dropped (kept {encoder_acoustic_kept}/{cfg['encoder_config']['num_quantizers'] - cfg['encoder_config']['num_semantic_quantizers']})")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"[Convert] Wrote {out_path}")
    return 0


def rename_talker_layer(name: str) -> str:
    # talker.model.layers.{i}.input_layernorm.weight             -> talker.blk.{i}.attn_norm.weight
    # talker.model.layers.{i}.post_attention_layernorm.weight    -> talker.blk.{i}.ffn_norm.weight
    # talker.model.layers.{i}.self_attn.{q,k,v,o}_proj.weight    -> talker.blk.{i}.attn_{q,k,v,output}.weight
    # talker.model.layers.{i}.self_attn.{q,k}_norm.weight        -> talker.blk.{i}.attn_{q,k}_norm.weight
    # talker.model.layers.{i}.mlp.{gate,up,down}_proj.weight     -> talker.blk.{i}.ffn_{gate,up,down}.weight
    assert name.startswith("talker.model.layers.")
    return _xlate_lm_layer(name, prefix="talker.model.layers.", out_prefix="talker.blk")


def rename_code_predictor_layer(name: str) -> str:
    # talker.code_predictor.model.layers.{i}.<role>  ->  code_pred.blk.{i}.<role>
    assert name.startswith("talker.code_predictor.model.layers.")
    return _xlate_lm_layer(name, prefix="talker.code_predictor.model.layers.", out_prefix="code_pred.blk")


def rename_text_projection(name: str) -> str:
    # talker.text_projection.linear_fc{1,2}.{weight,bias}  ->  talker.text_proj.fc{1,2}.{weight,bias}
    assert name.startswith("talker.text_projection.")
    return name.replace("talker.text_projection.linear_fc", "talker.text_proj.fc")


# Common HF -> llama.cpp suffix table for the Qwen3 backbone, shared between
# the Talker and the Code Predictor since both follow the same architecture.
_LM_LAYER_SUFFIX = {
    "input_layernorm.weight":           "attn_norm.weight",
    "post_attention_layernorm.weight":  "ffn_norm.weight",
    "self_attn.q_proj.weight":          "attn_q.weight",
    "self_attn.k_proj.weight":          "attn_k.weight",
    "self_attn.v_proj.weight":          "attn_v.weight",
    "self_attn.o_proj.weight":          "attn_output.weight",
    "self_attn.q_norm.weight":          "attn_q_norm.weight",
    "self_attn.k_norm.weight":          "attn_k_norm.weight",
    "mlp.gate_proj.weight":             "ffn_gate.weight",
    "mlp.up_proj.weight":               "ffn_up.weight",
    "mlp.down_proj.weight":             "ffn_down.weight",
}


def _xlate_lm_layer(name: str, prefix: str, out_prefix: str) -> str:
    rest      = name[len(prefix):]
    dot       = rest.find(".")
    layer_idx = rest[:dot]
    suffix    = rest[dot + 1:]
    new_suffix = _LM_LAYER_SUFFIX.get(suffix)
    if new_suffix is None:
        raise ValueError(f"Unhandled LM layer suffix : {suffix} (full name : {name})")
    return f"{out_prefix}.{layer_idx}.{new_suffix}"


def load_bpe_vocab(checkpoint_dir: Path):
    # Load Qwen2 BPE vocab + merges from a HF checkpoint directory and
    # produce the (tokens, token_types, merges) triple expected by the
    # GGUF tokenizer convention. Special tokens listed in
    # tokenizer_config.json:added_tokens_decoder are tagged as user-defined
    # (token_type 4) so the runtime can recognise them as verbatim chunks.
    vocab = json.loads((checkpoint_dir / "vocab.json").read_text())
    tok_cfg = json.loads((checkpoint_dir / "tokenizer_config.json").read_text())
    added = tok_cfg.get("added_tokens_decoder", {})

    max_id = max(vocab.values())
    for sid in added.keys():
        max_id = max(max_id, int(sid))

    tokens = [None] * (max_id + 1)
    token_types = [1] * (max_id + 1)  # 1 = normal

    for tok, tid in vocab.items():
        tokens[tid] = tok

    for sid_str, info in added.items():
        sid = int(sid_str)
        tokens[sid] = info["content"]
        token_types[sid] = 4  # 4 = user-defined / special

    # Empty slots (gaps in id space) get a placeholder so the GGUF array
    # has no None entries.
    for i, tok in enumerate(tokens):
        if tok is None:
            tokens[i] = f"<|unused-{i}|>"
            token_types[i] = 5  # 5 = unused

    # merges.txt : first line may be a "#version" comment, skip it. Only that
    # line: real merges start with "#" too ("# #", "## #", ...), 96 in Qwen2.
    merges_lines = (checkpoint_dir / "merges.txt").read_text(encoding="utf-8").splitlines()
    merges = [ln for ln in merges_lines if ln and not ln.startswith("#version")]

    return tokens, token_types, merges


def convert_talker_base(checkpoint_dir: Path, out_path: Path, model_size: str) -> int:
    cfg_path = checkpoint_dir / "config.json"
    st_path = checkpoint_dir / "model.safetensors"
    gen_path = checkpoint_dir / "generation_config.json"
    if not cfg_path.is_file() or not st_path.is_file():
        print(f"[Convert] FATAL: missing checkpoint files in {checkpoint_dir}")
        return 1

    cfg = json.loads(cfg_path.read_text())
    talker_cfg = cfg["talker_config"]
    cp_cfg = talker_cfg["code_predictor_config"]
    spk_cfg = cfg.get("speaker_encoder_config")
    gen_cfg = json.loads(gen_path.read_text()) if gen_path.is_file() else {}

    arch = "qwen3-tts"
    writer = gguf.GGUFWriter(str(out_path), arch)

    writer.add_string("general.name", f"Qwen3-TTS-12Hz-{model_size}-{cfg['tts_model_type']}")

    # Top-level TTS metadata
    writer.add_string("qwen3-tts.tokenizer_type", cfg["tokenizer_type"])
    writer.add_string("qwen3-tts.model_size", cfg["tts_model_size"])
    writer.add_string("qwen3-tts.model_type", cfg["tts_model_type"])
    writer.add_uint32("qwen3-tts.num_code_groups", talker_cfg["num_code_groups"])

    # Talker LM hyperparameters
    writer.add_uint32("qwen3-tts.talker.embedding_length", talker_cfg["hidden_size"])
    writer.add_uint32("qwen3-tts.talker.feed_forward_length", talker_cfg["intermediate_size"])
    writer.add_uint32("qwen3-tts.talker.block_count", talker_cfg["num_hidden_layers"])
    writer.add_uint32("qwen3-tts.talker.attention.head_count", talker_cfg["num_attention_heads"])
    writer.add_uint32("qwen3-tts.talker.attention.head_count_kv", talker_cfg["num_key_value_heads"])
    writer.add_uint32("qwen3-tts.talker.attention.key_length", talker_cfg["head_dim"])
    writer.add_uint32("qwen3-tts.talker.vocab_size", talker_cfg["vocab_size"])
    writer.add_uint32("qwen3-tts.talker.text_vocab_size", talker_cfg["text_vocab_size"])
    writer.add_uint32("qwen3-tts.talker.text_hidden_size", talker_cfg["text_hidden_size"])
    writer.add_uint32("qwen3-tts.talker.context_length", talker_cfg["max_position_embeddings"])
    writer.add_float32("qwen3-tts.talker.rope.freq_base", float(talker_cfg["rope_theta"]))
    writer.add_float32("qwen3-tts.talker.attention.layer_norm_rms_epsilon", float(talker_cfg["rms_norm_eps"]))
    writer.add_uint32("qwen3-tts.talker.position_id_per_seconds", talker_cfg["position_id_per_seconds"])
    rope_scaling = talker_cfg.get("rope_scaling") or {}
    if "mrope_section" in rope_scaling:
        writer.add_array("qwen3-tts.talker.rope.mrope_section", rope_scaling["mrope_section"])
        writer.add_bool("qwen3-tts.talker.mrope_interleaved", bool(rope_scaling.get("interleaved", False)))

    # Code predictor (subtalker) hyperparameters
    writer.add_uint32("qwen3-tts.code_pred.embedding_length", cp_cfg["hidden_size"])
    writer.add_uint32("qwen3-tts.code_pred.feed_forward_length", cp_cfg["intermediate_size"])
    writer.add_uint32("qwen3-tts.code_pred.block_count", cp_cfg["num_hidden_layers"])
    writer.add_uint32("qwen3-tts.code_pred.attention.head_count", cp_cfg["num_attention_heads"])
    writer.add_uint32("qwen3-tts.code_pred.attention.head_count_kv", cp_cfg["num_key_value_heads"])
    writer.add_uint32("qwen3-tts.code_pred.attention.key_length", cp_cfg["head_dim"])
    writer.add_uint32("qwen3-tts.code_pred.vocab_size", cp_cfg["vocab_size"])
    writer.add_uint32("qwen3-tts.code_pred.context_length", cp_cfg["max_position_embeddings"])
    writer.add_float32("qwen3-tts.code_pred.rope.freq_base", float(cp_cfg["rope_theta"]))
    writer.add_float32("qwen3-tts.code_pred.attention.layer_norm_rms_epsilon", float(cp_cfg["rms_norm_eps"]))

    # Speaker encoder hyperparameters (Base checkpoints only). CustomVoice
    # and VoiceDesign carry no speaker encoder so the keys are skipped
    # entirely, the runtime detects the absence via tensor lookup.
    if spk_cfg is not None:
        writer.add_uint32("qwen3-tts.spk_enc.embedding_length", spk_cfg["enc_dim"])
        writer.add_uint32("qwen3-tts.spk_enc.sample_rate", spk_cfg["sample_rate"])

    # Codec stream special tokens
    writer.add_uint32("qwen3-tts.codec.pad_id", talker_cfg["codec_pad_id"])
    writer.add_uint32("qwen3-tts.codec.bos_id", talker_cfg["codec_bos_id"])
    writer.add_uint32("qwen3-tts.codec.eos_id", talker_cfg["codec_eos_token_id"])
    writer.add_uint32("qwen3-tts.codec.think_id", talker_cfg["codec_think_id"])
    writer.add_uint32("qwen3-tts.codec.nothink_id", talker_cfg["codec_nothink_id"])
    writer.add_uint32("qwen3-tts.codec.think_bos_id", talker_cfg["codec_think_bos_id"])
    writer.add_uint32("qwen3-tts.codec.think_eos_id", talker_cfg["codec_think_eos_id"])

    # Language id table flattened to two parallel arrays. Names stay as in
    # the upstream config so the runtime can pass --lang chinese verbatim.
    lang_map = talker_cfg.get("codec_language_id") or {}
    lang_names = list(lang_map.keys())
    lang_ids = [int(lang_map[k]) for k in lang_names]
    writer.add_array("qwen3-tts.codec.language_names", lang_names)
    writer.add_array("qwen3-tts.codec.language_ids", lang_ids)

    # Speaker table for CustomVoice variants. Three parallel arrays indexed
    # by speaker position : name, codec embedding id, and optional dialect
    # name pulled from codec_language_id. Empty dialect string means the
    # speaker keeps the user supplied language. Skipped entirely for Base
    # and VoiceDesign which have no spk_id map.
    spk_map = talker_cfg.get("spk_id") or {}
    if spk_map:
        dialect_map = talker_cfg.get("spk_is_dialect") or {}
        spk_names    = list(spk_map.keys())
        spk_ids      = [int(spk_map[k]) for k in spk_names]
        spk_dialects = [dialect_map.get(k) or "" for k in spk_names]
        spk_dialects = [d if isinstance(d, str) else "" for d in spk_dialects]
        writer.add_array("qwen3-tts.codec.speaker_names", spk_names)
        writer.add_array("qwen3-tts.codec.speaker_ids", spk_ids)
        writer.add_array("qwen3-tts.codec.speaker_dialects", spk_dialects)

    # Text-side special tokens (Qwen2 BPE)
    writer.add_uint32("qwen3-tts.text.im_start_id", cfg["im_start_token_id"])
    writer.add_uint32("qwen3-tts.text.im_end_id", cfg["im_end_token_id"])
    writer.add_uint32("qwen3-tts.text.tts_pad_id", cfg["tts_pad_token_id"])
    writer.add_uint32("qwen3-tts.text.tts_bos_id", cfg["tts_bos_token_id"])
    writer.add_uint32("qwen3-tts.text.tts_eos_id", cfg["tts_eos_token_id"])

    # Default sampling parameters from generation_config.json
    if gen_cfg:
        if "do_sample" in gen_cfg:
            writer.add_bool("generation.do_sample", bool(gen_cfg["do_sample"]))
        if "top_k" in gen_cfg:
            writer.add_uint32("generation.top_k", int(gen_cfg["top_k"]))
        if "top_p" in gen_cfg:
            writer.add_float32("generation.top_p", float(gen_cfg["top_p"]))
        if "temperature" in gen_cfg:
            writer.add_float32("generation.temperature", float(gen_cfg["temperature"]))
        if "repetition_penalty" in gen_cfg:
            writer.add_float32("generation.repetition_penalty", float(gen_cfg["repetition_penalty"]))
        if "subtalker_dosample" in gen_cfg:
            writer.add_bool("generation.subtalker_do_sample", bool(gen_cfg["subtalker_dosample"]))
        if "subtalker_top_k" in gen_cfg:
            writer.add_uint32("generation.subtalker_top_k", int(gen_cfg["subtalker_top_k"]))
        if "subtalker_top_p" in gen_cfg:
            writer.add_float32("generation.subtalker_top_p", float(gen_cfg["subtalker_top_p"]))
        if "subtalker_temperature" in gen_cfg:
            writer.add_float32("generation.subtalker_temperature", float(gen_cfg["subtalker_temperature"]))
        if "max_new_tokens" in gen_cfg:
            writer.add_uint32("generation.max_new_tokens", int(gen_cfg["max_new_tokens"]))

    # BPE tokenizer payload
    bpe_tokens, bpe_token_types, bpe_merges = load_bpe_vocab(checkpoint_dir)
    writer.add_string("tokenizer.ggml.model", "gpt2")
    writer.add_array("tokenizer.ggml.tokens", bpe_tokens)
    writer.add_array("tokenizer.ggml.token_type", bpe_token_types)
    writer.add_array("tokenizer.ggml.merges", bpe_merges)
    writer.add_uint32("tokenizer.ggml.eos_token_id", 151643)  # <|endoftext|>

    # Top level talker tensors. Renames mirror the koboldcpp TENSOR_MAP.
    TALKER_TOP = {
        "talker.model.codec_embedding.weight":         "talker.codec_embd.weight",
        "talker.model.text_embedding.weight":          "talker.text_embd.weight",
        "talker.model.norm.weight":                    "talker.output_norm.weight",
        "talker.codec_head.weight":                    "talker.codec_head.weight",
        "talker.text_projection.linear_fc1.weight":    "talker.text_proj.fc1.weight",
        "talker.text_projection.linear_fc1.bias":      "talker.text_proj.fc1.bias",
        "talker.text_projection.linear_fc2.weight":    "talker.text_proj.fc2.weight",
        "talker.text_projection.linear_fc2.bias":      "talker.text_proj.fc2.bias",
    }

    # Top level code predictor tensors.
    CP_TOP = {
        "talker.code_predictor.model.norm.weight":                       "code_pred.output_norm.weight",
        "talker.code_predictor.small_to_mtp_projection.weight":          "code_pred.mtp_proj.weight",
        "talker.code_predictor.small_to_mtp_projection.bias":            "code_pred.mtp_proj.bias",
    }

    # Speaker encoder rename table, koboldcpp SPEAKER_ENCODER_PATTERNS plus
    # standalone tensors. block 0 is the entry conv, blocks 1 to 3 hold the
    # Res2Net + SE + TDNN stack.
    SPK_TOP = {
        "speaker_encoder.blocks.0.conv.weight":     "spk_enc.conv0.weight",
        "speaker_encoder.blocks.0.conv.bias":       "spk_enc.conv0.bias",
        "speaker_encoder.asp.conv.weight":          "spk_enc.asp.conv.weight",
        "speaker_encoder.asp.conv.bias":            "spk_enc.asp.conv.bias",
        "speaker_encoder.asp.tdnn.conv.weight":     "spk_enc.asp.tdnn.weight",
        "speaker_encoder.asp.tdnn.conv.bias":       "spk_enc.asp.tdnn.bias",
        "speaker_encoder.mfa.conv.weight":          "spk_enc.mfa.weight",
        "speaker_encoder.mfa.conv.bias":            "spk_enc.mfa.bias",
        "speaker_encoder.fc.weight":                "spk_enc.fc.weight",
        "speaker_encoder.fc.bias":                  "spk_enc.fc.bias",
    }

    def rename_speaker_encoder_block(k: str) -> str:
        # speaker_encoder.blocks.{i}.res2net_block.blocks.{j}.conv.{weight,bias}  ->  spk_enc.blk.{i}.res2net.{j}.{weight,bias}
        # speaker_encoder.blocks.{i}.se_block.conv{1,2}.{weight,bias}             ->  spk_enc.blk.{i}.se.conv{1,2}.{weight,bias}
        # speaker_encoder.blocks.{i}.tdnn{1,2}.conv.{weight,bias}                 ->  spk_enc.blk.{i}.tdnn{1,2}.{weight,bias}
        parts = k.split(".")
        idx   = parts[2]
        if parts[3] == "res2net_block":
            sub    = parts[5]
            suffix = parts[-1]
            return f"spk_enc.blk.{idx}.res2net.{sub}.{suffix}"
        if parts[3] == "se_block":
            which  = parts[4]  # conv1 or conv2
            suffix = parts[-1]
            return f"spk_enc.blk.{idx}.se.{which}.{suffix}"
        if parts[3] in ("tdnn1", "tdnn2"):
            return f"spk_enc.blk.{idx}.{parts[3]}.{parts[-1]}"
        raise ValueError(f"Unhandled speaker encoder block tensor : {k}")

    # Walk safetensors
    n_added = 0
    n_unhandled = 0
    with safe_open(str(st_path), framework="pt") as f:
        all_keys = sorted(list(f.keys()))
        for k in all_keys:
            t = f.get_tensor(k)
            arr = t.float().numpy()

            # Talker transformer layers
            if k.startswith("talker.model.layers."):
                writer.add_tensor(rename_talker_layer(k), arr)
                n_added += 1
                continue

            # Talker top level (codec_embedding, text_embedding, norm,
            # codec_head, text_projection)
            if k in TALKER_TOP:
                writer.add_tensor(TALKER_TOP[k], arr)
                n_added += 1
                continue

            # Code predictor transformer layers
            if k.startswith("talker.code_predictor.model.layers."):
                writer.add_tensor(rename_code_predictor_layer(k), arr)
                n_added += 1
                continue

            # Code predictor codec embeddings (one per acoustic codebook)
            if k.startswith("talker.code_predictor.model.codec_embedding."):
                idx = k.split(".")[4]
                writer.add_tensor(f"code_pred.codec_embd.{idx}.weight", arr)
                n_added += 1
                continue

            # Code predictor lm heads (one per acoustic codebook)
            if k.startswith("talker.code_predictor.lm_head."):
                idx = k.split(".")[3]
                writer.add_tensor(f"code_pred.lm_head.{idx}.weight", arr)
                n_added += 1
                continue

            # Code predictor top level (final norm, MTP projection)
            if k in CP_TOP:
                writer.add_tensor(CP_TOP[k], arr)
                n_added += 1
                continue

            # Speaker encoder, only present in Base checkpoints
            if k in SPK_TOP:
                writer.add_tensor(SPK_TOP[k], arr)
                n_added += 1
                continue
            if k.startswith("speaker_encoder.blocks.") and not k.startswith("speaker_encoder.blocks.0."):
                writer.add_tensor(rename_speaker_encoder_block(k), arr)
                n_added += 1
                continue

            print(f"[Convert] WARNING: unhandled tensor : {k}  shape={tuple(t.shape)}")
            n_unhandled += 1

    print(f"[Convert] Tensors: {n_added} written, {n_unhandled} unhandled")
    print(f"[Convert] BPE: {len(bpe_tokens)} tokens, {len(bpe_merges)} merges")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"[Convert] Wrote {out_path}")
    return 0


CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR     = "models"

# Talker checkpoints follow the upstream pattern Qwen3-TTS-12Hz-{size}-{kind}
# where size is 0.6B or 1.7B and kind is Base, CustomVoice or VoiceDesign.
# The compiled regex captures both groups for the GGUF filename suffix and
# the model_size argument fed to convert_talker_base.
TALKER_RE = re.compile(r"^Qwen3-TTS-12Hz-([0-9.]+B)-(\w+)$")


def classify(dir_name: str):
    """Return (kind, model_size) for a known checkpoint directory or None.
    kind is 'tokenizer' or 'talker'. model_size is '0.6B' / '1.7B' for
    talker, None for tokenizer."""
    if "Tokenizer" in dir_name:
        return ("tokenizer", None)
    m = TALKER_RE.match(dir_name)
    if m:
        return ("talker", m.group(1))
    return None


def output_path_for(out_dir: Path, kind: str, dir_name: str) -> Path:
    """Map a checkpoint directory name to its F32 GGUF output path."""
    if kind == "tokenizer":
        return out_dir / "qwen-tokenizer-12hz-F32.gguf"
    m = TALKER_RE.match(dir_name)
    short = f"{m.group(1).lower()}-{m.group(2).lower()}"
    return out_dir / f"qwen-talker-{short}-F32.gguf"


def main() -> int:
    ckpt_root = Path(CHECKPOINT_DIR)
    out_dir   = Path(OUTPUT_DIR)

    if not ckpt_root.is_dir():
        print(f"[Convert] FATAL: {ckpt_root}/ not found")
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped_unknown: list[str] = []
    rc = 0

    for name in sorted(os.listdir(ckpt_root)):
        ckpt = ckpt_root / name
        if not ckpt.is_dir():
            continue

        info = classify(name)
        if info is None:
            skipped_unknown.append(name)
            continue

        kind, model_size = info
        out = output_path_for(out_dir, kind, name)

        if out.exists():
            print(f"[Convert] skip {out.name}: exists")
            converted += 1
            continue

        if kind == "tokenizer":
            rc |= convert_tokenizer_12hz(ckpt, out)
        else:
            rc |= convert_talker_base(ckpt, out, model_size)
        converted += 1

    if skipped_unknown:
        print(f"[Convert] skipped (unknown): {', '.join(skipped_unknown)}")
    print(f"[Convert] done : {converted} model(s) in {out_dir}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
