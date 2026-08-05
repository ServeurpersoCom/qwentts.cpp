#pragma once
// rvq-file.h: packed RVQ code stream file IO (.rvq).
//
// Flat code stream packed at code_bits per code, LSB-first, no header.
// Layout is [K, T] row-major. K and code_bits are fixed by the codec
// config in the GGUF; T is derived from the file size:
// T = (filesize * 8) / (K * code_bits).

#include "utf8.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

// Pack a flat code stream into code_bits-per-code, LSB-first. Output size
// is ceil(N * code_bits / 8) bytes.
static std::vector<uint8_t> rvq_pack_codes(const std::vector<int32_t> & codes, int code_bits) {
    const uint32_t       mask       = (1u << code_bits) - 1u;
    const size_t         total_bits = codes.size() * (size_t) code_bits;
    std::vector<uint8_t> out((total_bits + 7) / 8, 0);

    uint64_t acc         = 0;
    int      bits_in_acc = 0;
    size_t   out_pos     = 0;
    for (size_t i = 0; i < codes.size(); i++) {
        acc |= ((uint64_t) ((uint32_t) codes[i] & mask)) << bits_in_acc;
        bits_in_acc += code_bits;
        while (bits_in_acc >= 8) {
            out[out_pos++] = (uint8_t) (acc & 0xFF);
            acc >>= 8;
            bits_in_acc -= 8;
        }
    }
    if (bits_in_acc > 0) {
        out[out_pos++] = (uint8_t) (acc & 0xFF);
    }
    return out;
}

// Symmetric unpack: reads N codes from packed bytes.
static std::vector<int32_t> rvq_unpack_codes(const std::vector<uint8_t> & in, size_t n_codes, int code_bits) {
    const uint32_t       mask = (1u << code_bits) - 1u;
    std::vector<int32_t> out(n_codes);

    uint64_t acc         = 0;
    int      bits_in_acc = 0;
    size_t   in_pos      = 0;
    for (size_t i = 0; i < n_codes; i++) {
        while (bits_in_acc < code_bits && in_pos < in.size()) {
            acc |= ((uint64_t) in[in_pos++]) << bits_in_acc;
            bits_in_acc += 8;
        }
        out[i] = (int32_t) (acc & mask);
        acc >>= code_bits;
        bits_in_acc -= code_bits;
    }
    return out;
}

// Read a .rvq file and unpack it into K*T codes. T is inferred from the
// file size.
// Unpack a raw .rvq byte stream: K codebooks at code_bits per code,
// LSB-first, [K, T] row-major. T derives from the byte count.
static bool rvq_read_buf(const uint8_t *        data,
                         size_t                 size,
                         int                    K,
                         int                    code_bits,
                         std::vector<int32_t> & codes,
                         int *                  n_frames) {
    if (size == 0) {
        fprintf(stderr, "[RVQ] FATAL: empty code stream\n");
        return false;
    }
    const size_t total_bits = size * 8;
    const size_t n_codes    = total_bits / (size_t) code_bits;
    if (n_codes == 0 || (n_codes % (size_t) K) != 0) {
        fprintf(stderr, "[RVQ] FATAL: stream yields %zu codes, not a multiple of K=%d\n", n_codes, K);
        return false;
    }
    std::vector<uint8_t> buf(data, data + size);
    codes     = rvq_unpack_codes(buf, n_codes, code_bits);
    *n_frames = (int) (n_codes / (size_t) K);
    return true;
}

static bool rvq_read_file(const char * path, int K, int code_bits, std::vector<int32_t> & codes, int * n_frames) {
    FILE * f = utf8_fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[RVQ] FATAL: cannot open %s\n", path);
        return false;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) {
        fprintf(stderr, "[RVQ] FATAL: %s is empty\n", path);
        fclose(f);
        return false;
    }
    std::vector<uint8_t> buf((size_t) sz);
    if (fread(buf.data(), 1, buf.size(), f) != buf.size()) {
        fprintf(stderr, "[RVQ] FATAL: short read on %s\n", path);
        fclose(f);
        return false;
    }
    fclose(f);
    return rvq_read_buf(buf.data(), buf.size(), K, code_bits, codes, n_frames);
}

// Pack and write a .rvq file.
static bool rvq_write_file(const char * path, const std::vector<int32_t> & codes, int code_bits) {
    std::vector<uint8_t> packed = rvq_pack_codes(codes, code_bits);
    FILE *               f      = utf8_fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "[RVQ] FATAL: cannot open %s for write\n", path);
        return false;
    }
    if (fwrite(packed.data(), 1, packed.size(), f) != packed.size()) {
        fprintf(stderr, "[RVQ] FATAL: short write on %s\n", path);
        fclose(f);
        return false;
    }
    fclose(f);
    return true;
}
