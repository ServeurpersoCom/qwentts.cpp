#pragma once
// bpe.h, Qwen3/GPT-2 byte-level BPE tokenizer (CPU-only, no dependencies)
//
// Parses the tokenizer.json fields stored in a GGUF model and produces a
// byte-level BPE encoder/decoder. Arch-specific special tokens (text
// markers, language tags, audio sentinels) are loaded through
// bpe_load_specials_from_keys with a caller-provided list of GGUF KV keys.
// Loads vocab + merges from a GGUF tokenizer payload. Handles byte-level
// encoding, GPT-2 regex pre-tokenizer, BPE merges, and a registry of
// verbatim special tokens (endoftext plus any caller-registered tokens
// such as TTS style markers and language tags).

#include "gguf.h"
#include "qt-error.h"
#include "unicode_ranges.h"

#include <cassert>
#include <climits>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

// GPT-2 byte-level encoding table
// Maps byte [0..255] -> Unicode char for BPE vocab keys.
// Printable ASCII stays as-is, control/space bytes get remapped.
static void build_byte_encoder(std::string byte2str[256]) {
    // Standard GPT-2 byte encoder
    int bs[256], cs[256], n = 0, total = 0;
    // Printable ranges that map to themselves
    for (int b = '!'; b <= '~'; b++) {
        bs[total] = b;
        cs[total] = b;
        total++;
    }
    for (int b = 0xA1; b <= 0xAC; b++) {
        bs[total] = b;
        cs[total] = b;
        total++;
    }
    for (int b = 0xAE; b <= 0xFF; b++) {
        bs[total] = b;
        cs[total] = b;
        total++;
    }
    // Remaining bytes get mapped to 256+
    bool used[256] = {};
    for (int i = 0; i < total; i++) {
        used[bs[i]] = true;
    }
    for (int b = 0; b < 256; b++) {
        if (!used[b]) {
            bs[total] = b;
            cs[total] = 256 + n;
            n++;
            total++;
        }
    }
    assert(total == 256);
    // Convert codepoints to UTF-8 strings
    for (int i = 0; i < 256; i++) {
        int  cp = cs[i];
        char buf[4];
        int  len;
        if (cp < 0x80) {
            buf[0] = (char) cp;
            len    = 1;
        } else if (cp < 0x800) {
            buf[0] = (char) (0xC0 | (cp >> 6));
            buf[1] = (char) (0x80 | (cp & 0x3F));
            len    = 2;
        } else {
            buf[0] = (char) (0xE0 | (cp >> 12));
            buf[1] = (char) (0x80 | ((cp >> 6) & 0x3F));
            buf[2] = (char) (0x80 | (cp & 0x3F));
            len    = 3;
        }
        byte2str[bs[i]] = std::string(buf, len);
    }
}

// UTF-8 helpers
static int utf8_codepoint(const char * s, int * advance) {
    unsigned char c = s[0];
    if (c < 0x80) {
        *advance = 1;
        return c;
    }
    if ((c & 0xE0) == 0xC0) {
        *advance = 2;
        return ((c & 0x1F) << 6) | (s[1] & 0x3F);
    }
    if ((c & 0xF0) == 0xE0) {
        *advance = 3;
        return ((c & 0x0F) << 12) | ((s[1] & 0x3F) << 6) | (s[2] & 0x3F);
    }
    if ((c & 0xF8) == 0xF0) {
        *advance = 4;
        return ((c & 0x07) << 18) | ((s[1] & 0x3F) << 12) | ((s[2] & 0x3F) << 6) | (s[3] & 0x3F);
    }
    // invalid lead byte: advance one and return the raw byte to avoid an
    // infinite loop. The Python tokenizer never reaches this path since Python
    // str guarantees valid UTF 8; in C++ the std::string input has no such
    // guarantee, so this branch handles malformed input defensively.
    *advance = 1;
    return c;
}

// Unicode category checks: \p{L} and \p{N} exactly, from the generated tables.
// Combining marks (Mn/Mc, e.g. Thai vowel and tone marks, Hebrew niqqud) are
// neither, so the regex splits around them; treating them as letters gave
// different chunks than the HF tokenizer and garbled speech.
static bool in_ranges(int cp, const UnicodeRange * r, int n) {
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (cp < r[mid].lo) {
            hi = mid - 1;
        } else if (cp > r[mid].hi) {
            lo = mid + 1;
        } else {
            return true;
        }
    }
    return false;
}

static bool is_letter(int cp) {
    if (cp < 0x80) {
        return (cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z');
    }
    return in_ranges(cp, UNICODE_LETTER_RANGES, (int) (sizeof(UNICODE_LETTER_RANGES) / sizeof(UNICODE_LETTER_RANGES[0])));
}

// \p{N}: any number, not only ASCII digits (Thai ๑, Devanagari १, ½, ², Ⅻ, ...)
static bool is_digit(int cp) {
    if (cp < 0x80) {
        return cp >= '0' && cp <= '9';
    }
    return in_ranges(cp, UNICODE_NUMBER_RANGES, (int) (sizeof(UNICODE_NUMBER_RANGES) / sizeof(UNICODE_NUMBER_RANGES[0])));
}

// \s: the Unicode White_Space property (what the HF tokenizer's regex uses).
static bool is_whitespace(int cp) {
    return (cp >= 0x09 && cp <= 0x0D) || cp == 0x20 || cp == 0x85 || cp == 0xA0 || cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F ||
           cp == 0x3000;
}

static bool is_newline(int cp) {
    return cp == '\n' || cp == '\r';
}

static void utf8_append(std::string & out, int cp) {
    if (cp < 0x80) {
        out += (char) cp;
    } else if (cp < 0x800) {
        out += (char) (0xC0 | (cp >> 6));
        out += (char) (0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out += (char) (0xE0 | (cp >> 12));
        out += (char) (0x80 | ((cp >> 6) & 0x3F));
        out += (char) (0x80 | (cp & 0x3F));
    } else {
        out += (char) (0xF0 | (cp >> 18));
        out += (char) (0x80 | ((cp >> 12) & 0x3F));
        out += (char) (0x80 | ((cp >> 6) & 0x3F));
        out += (char) (0x80 | (cp & 0x3F));
    }
}

// NFC (UAX #15), the normalizer the HF Qwen2 tokenizer runs before splitting:
// canonical decomposition, canonical ordering, canonical composition. Without
// it, text that is equivalent but encoded differently (a precomposed nukta
// letter, Thai marks typed in another order, decomposed accents) tokenizes
// differently from what the model was trained on.
static const int HANGUL_S = 0xAC00, HANGUL_L = 0x1100, HANGUL_V = 0x1161, HANGUL_T = 0x11A7;
static const int HANGUL_L_COUNT = 19, HANGUL_V_COUNT = 21, HANGUL_T_COUNT = 28;
static const int HANGUL_N_COUNT = HANGUL_V_COUNT * HANGUL_T_COUNT, HANGUL_S_COUNT = HANGUL_L_COUNT * HANGUL_N_COUNT;

static int unicode_ccc(int cp) {
    if (cp < 0x300) {
        return 0;
    }
    int lo = 0, hi = (int) (sizeof(UNICODE_COMBINING_CLASSES) / sizeof(UNICODE_COMBINING_CLASSES[0])) - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (cp < UNICODE_COMBINING_CLASSES[mid].lo) {
            hi = mid - 1;
        } else if (cp > UNICODE_COMBINING_CLASSES[mid].hi) {
            lo = mid + 1;
        } else {
            return UNICODE_COMBINING_CLASSES[mid].ccc;
        }
    }
    return 0;
}

static void nfd_append(std::vector<int> & out, int cp) {
    if (cp >= HANGUL_S && cp < HANGUL_S + HANGUL_S_COUNT) {
        int s = cp - HANGUL_S;
        out.push_back(HANGUL_L + s / HANGUL_N_COUNT);
        out.push_back(HANGUL_V + (s % HANGUL_N_COUNT) / HANGUL_T_COUNT);
        if (s % HANGUL_T_COUNT) {
            out.push_back(HANGUL_T + s % HANGUL_T_COUNT);
        }
        return;
    }
    if (cp >= 0xC0) {
        int lo = 0, hi = (int) (sizeof(UNICODE_DECOMPOSITIONS) / sizeof(UNICODE_DECOMPOSITIONS[0])) - 1;
        while (lo <= hi) {
            int mid = (lo + hi) / 2;
            if (cp < UNICODE_DECOMPOSITIONS[mid].cp) {
                hi = mid - 1;
            } else if (cp > UNICODE_DECOMPOSITIONS[mid].cp) {
                lo = mid + 1;
            } else {
                const UnicodeDecomposition & d = UNICODE_DECOMPOSITIONS[mid];
                out.insert(out.end(), UNICODE_DECOMPOSITION_CODES + d.offset,
                           UNICODE_DECOMPOSITION_CODES + d.offset + d.length);
                return;
            }
        }
    }
    out.push_back(cp);
}

// The primary composite of (a, b), or -1.
static int nfc_compose_pair(int a, int b) {
    if (a >= HANGUL_L && a < HANGUL_L + HANGUL_L_COUNT && b >= HANGUL_V && b < HANGUL_V + HANGUL_V_COUNT) {
        return HANGUL_S + ((a - HANGUL_L) * HANGUL_V_COUNT + (b - HANGUL_V)) * HANGUL_T_COUNT;
    }
    if (a >= HANGUL_S && a < HANGUL_S + HANGUL_S_COUNT && (a - HANGUL_S) % HANGUL_T_COUNT == 0 && b > HANGUL_T &&
        b < HANGUL_T + HANGUL_T_COUNT) {
        return a + (b - HANGUL_T);
    }
    int lo = 0, hi = (int) (sizeof(UNICODE_COMPOSITIONS) / sizeof(UNICODE_COMPOSITIONS[0])) - 1;
    while (lo <= hi) {
        int                        mid = (lo + hi) / 2;
        const UnicodeComposition & c   = UNICODE_COMPOSITIONS[mid];
        if (a < c.first || (a == c.first && b < c.second)) {
            hi = mid - 1;
        } else if (a > c.first || b > c.second) {
            lo = mid + 1;
        } else {
            return c.composite;
        }
    }
    return -1;
}

static std::string nfc_normalize(const std::string & text) {
    bool ascii = true;
    for (unsigned char c : text) {
        if (c >= 0x80) {
            ascii = false;
            break;
        }
    }
    if (ascii) {
        return text;
    }

    std::vector<int> cps;
    const char *     s   = text.c_str();
    int              len = (int) text.size();
    for (int i = 0; i < len;) {
        int adv;
        int cp = utf8_codepoint(s + i, &adv);
        if (adv == 1 && (unsigned char) s[i] >= 0x80) {
            return text;  // malformed UTF-8: leave the bytes as they are
        }
        nfd_append(cps, cp);
        i += adv;
    }

    // Canonical ordering: stable sort of each run of non-starters by class.
    for (size_t i = 1; i < cps.size(); i++) {
        int c = unicode_ccc(cps[i]);
        for (size_t j = i; c && j > 0 && unicode_ccc(cps[j - 1]) > c; j--) {
            std::swap(cps[j], cps[j - 1]);
        }
    }

    // Canonical composition: join each character to the last starter unless a
    // character in between blocks it (class 0, or a class >= its own).
    std::vector<int> out;
    int              starter  = -1;
    int              last_ccc = 0;
    for (int cp : cps) {
        int c = unicode_ccc(cp);
        if (starter >= 0) {
            bool adjacent = (int) out.size() - 1 == starter;
            if (adjacent || (last_ccc != 0 && last_ccc < c)) {
                int composite = nfc_compose_pair(out[starter], cp);
                if (composite >= 0) {
                    out[starter] = composite;
                    continue;
                }
            }
        }
        if (c == 0) {
            starter  = (int) out.size();
            last_ccc = 0;
        } else {
            last_ccc = c;
        }
        out.push_back(cp);
    }

    std::string result;
    result.reserve(text.size());
    for (int cp : out) {
        utf8_append(result, cp);
    }
    return result;
}

// Qwen2 pre-tokenizer regex (the HF tokenizer's Split pattern):
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|
//    ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
// At each position the alternatives are tried in that order and the first
// match is taken, exactly as the regex engine would. Works on code points, so
// a chunk never ends inside a multi-byte character.
static std::vector<std::string> gpt2_pre_tokenize(const std::string & text) {
    std::vector<int> cps, off;
    const char *     s   = text.c_str();
    int              len = (int) text.size();
    for (int i = 0; i < len;) {
        int adv;
        cps.push_back(utf8_codepoint(s + i, &adv));
        off.push_back(i);
        i += adv;
    }
    off.push_back(len);
    const int n = (int) cps.size();

    auto L  = [&](int k) { return k < n && is_letter(cps[k]); };
    auto N  = [&](int k) { return k < n && is_digit(cps[k]); };
    auto S  = [&](int k) { return k < n && is_whitespace(cps[k]); };
    auto NL = [&](int k) { return k < n && is_newline(cps[k]); };
    auto P  = [&](int k) { return k < n && !is_whitespace(cps[k]) && !is_letter(cps[k]) && !is_digit(cps[k]); };
    auto lower = [&](int k) {
        int c = k < n ? cps[k] : 0;
        return (c >= 'A' && c <= 'Z') ? c + 32 : c;
    };

    std::vector<std::string> chunks;
    int                      k = 0;
    while (k < n) {
        int e = -1;
        // (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if (cps[k] == '\'') {
            int a = lower(k + 1), b = lower(k + 2);
            if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) {
                e = k + 3;
            } else if (a == 's' || a == 't' || a == 'm' || a == 'd') {
                e = k + 2;
            }
        }
        // [^\r\n\p{L}\p{N}]?\p{L}+
        if (e < 0) {
            int j = (L(k) || N(k) || NL(k)) ? k : k + 1;
            if (L(j)) {
                while (L(j)) {
                    j++;
                }
                e = j;
            }
        }
        // \p{N}
        if (e < 0 && N(k)) {
            e = k + 1;
        }
        //  ?[^\s\p{L}\p{N}]+[\r\n]*
        if (e < 0) {
            int j = cps[k] == ' ' ? k + 1 : k;
            if (P(j)) {
                while (P(j)) {
                    j++;
                }
                while (NL(j)) {
                    j++;
                }
                e = j;
            }
        }
        // \s*[\r\n]+  |  \s+(?!\S)  |  \s+
        if (e < 0 && S(k)) {
            int j = k, last_nl = -1;
            while (S(j)) {
                if (NL(j)) {
                    last_nl = j;
                }
                j++;
            }
            if (last_nl >= 0) {
                e = last_nl + 1;
            } else if (j < n && j - k > 1) {
                e = j - 1;  // leave the last space to prefix the next chunk
            } else {
                e = j;
            }
        }
        if (e < 0) {
            e = k + 1;  // unreachable for valid input; never stall
        }
        chunks.push_back(text.substr(off[k], off[e] - off[k]));
        k = e;
    }
    return chunks;
}

// BPE tokenizer struct
struct BPETokenizer {
    std::unordered_map<std::string, int> vocab;          // token_str -> id
    std::unordered_map<std::string, int> merges;         // "a b" -> rank
    std::string                          byte2str[256];  // byte -> GPT-2 UTF-8 string
    int                                  eos_id;         // <|endoftext|>
    int                                  n_vocab;
    std::vector<std::string>             id_to_str;      // id -> token_str (reverse vocab)

    // Registered special tokens. Each (str, id) pair is matched verbatim in
    // bpe_encode and emitted as a single id, bypassing the BPE merge passes.
    std::vector<std::pair<std::string, int>> specials;
};

// Register a special token. Strings already registered are skipped (no dup).
static void bpe_add_special(BPETokenizer * tok, const std::string & str, int id) {
    for (const auto & sp : tok->specials) {
        if (sp.first == str) {
            return;
        }
    }
    tok->specials.emplace_back(str, id);
}

// Load tokenizer from GGUF KV (tokenizer.ggml.tokens + tokenizer.ggml.merges)
static bool load_bpe_from_gguf(BPETokenizer * tok, const char * gguf_path) {
    build_byte_encoder(tok->byte2str);

    struct gguf_init_params gp  = { true, NULL };
    struct gguf_context *   ctx = gguf_init_from_file(gguf_path, gp);
    if (!ctx) {
        qt_log(QT_LOG_ERROR, "[BPE] Failed to open %s", gguf_path);
        return false;
    }

    int64_t tok_key = gguf_find_key(ctx, "tokenizer.ggml.tokens");
    int64_t mrg_key = gguf_find_key(ctx, "tokenizer.ggml.merges");
    if (tok_key < 0 || mrg_key < 0) {
        qt_log(QT_LOG_INFO, "[BPE] Tokenizer not found in %s", gguf_path);
        gguf_free(ctx);
        return false;
    }

    int n_tokens = (int) gguf_get_arr_n(ctx, tok_key);
    int n_merges = (int) gguf_get_arr_n(ctx, mrg_key);

    for (int i = 0; i < n_tokens; i++) {
        const char * s             = gguf_get_arr_str(ctx, tok_key, (size_t) i);
        tok->vocab[std::string(s)] = i;
    }

    for (int i = 0; i < n_merges; i++) {
        const char * s              = gguf_get_arr_str(ctx, mrg_key, (size_t) i);
        tok->merges[std::string(s)] = i;
    }

    gguf_free(ctx);

    tok->n_vocab = (int) tok->vocab.size();

    tok->id_to_str.resize(tok->n_vocab);
    for (auto & kv : tok->vocab) {
        if (kv.second >= 0 && kv.second < tok->n_vocab) {
            tok->id_to_str[kv.second] = kv.first;
        }
    }

    // Resolve eos_id from the vocab itself rather than hard-coding 151643.
    // Falls back to -1 if the standard sentinel is absent.
    auto eos_it = tok->vocab.find("<|endoftext|>");
    tok->eos_id = (eos_it != tok->vocab.end()) ? eos_it->second : -1;
    if (tok->eos_id >= 0) {
        bpe_add_special(tok, "<|endoftext|>", tok->eos_id);
    }

    qt_log(QT_LOG_INFO, "[BPE] Loaded from GGUF: %d vocab, %d merges, eos_id=%d", tok->n_vocab, n_merges, tok->eos_id);
    return true;
}

// Read arch-specific special tokens from a caller-provided list of GGUF KV
// keys. Each key holds a u32 vocab id, mapped back to its vocab string and
// registered through bpe_add_special. The endoftext sentinel is already
// registered by load_bpe_from_gguf, so callers should not list it here.
static bool bpe_load_specials_from_keys(BPETokenizer *       tok,
                                        const char *         gguf_path,
                                        const char * const * keys,
                                        int                  n_keys) {
    struct gguf_init_params gp  = { true, NULL };
    struct gguf_context *   ctx = gguf_init_from_file(gguf_path, gp);
    if (!ctx) {
        qt_log(QT_LOG_ERROR, "[BPE] Failed to open %s for specials", gguf_path);
        return false;
    }

    int n_added = 0;
    for (int i = 0; i < n_keys; i++) {
        int64_t k = gguf_find_key(ctx, keys[i]);
        if (k < 0) {
            qt_log(QT_LOG_WARN, "[BPE] WARNING: missing %s in GGUF", keys[i]);
            continue;
        }
        int id = (int) gguf_get_val_u32(ctx, k);
        if (id < 0 || id >= tok->n_vocab) {
            qt_log(QT_LOG_WARN, "[BPE] WARNING: %s id=%d out of vocab range", keys[i], id);
            continue;
        }
        const std::string & s = tok->id_to_str[id];
        if (s.empty()) {
            qt_log(QT_LOG_WARN, "[BPE] WARNING: %s id=%d has empty vocab string", keys[i], id);
            continue;
        }
        bpe_add_special(tok, s, id);
        n_added++;
    }

    gguf_free(ctx);
    qt_log(QT_LOG_INFO, "[BPE] Registered %d arch special tokens (total specials=%zu)", n_added, tok->specials.size());
    return true;
}

// Byte-level encode: raw text bytes -> GPT-2 BPE string
static std::string byte_level_encode(const BPETokenizer * tok, const std::string & text) {
    std::string out;
    for (unsigned char c : text) {
        out += tok->byte2str[c];
    }
    return out;
}

// BPE merge algorithm
// Input: list of symbols (strings). Merges pairs by priority.
static std::vector<std::string> bpe_merge(const std::unordered_map<std::string, int> & merge_rank,
                                          const std::vector<std::string> &             symbols) {
    if (symbols.size() <= 1) {
        return symbols;
    }

    std::vector<std::string> work = symbols;

    while (work.size() > 1) {
        // Find the pair with lowest rank (highest priority)
        int best_rank = INT_MAX;
        int best_pos  = -1;
        for (int i = 0; i < (int) work.size() - 1; i++) {
            std::string key = work[i] + " " + work[i + 1];
            auto        it  = merge_rank.find(key);
            if (it != merge_rank.end() && it->second < best_rank) {
                best_rank = it->second;
                best_pos  = i;
            }
        }
        if (best_pos < 0) {
            break;  // no more merges
        }

        // Merge the pair
        std::string merged = work[best_pos] + work[best_pos + 1];
        work[best_pos]     = merged;
        work.erase(work.begin() + best_pos + 1);
    }
    return work;
}

// Encode a single pre-tokenized chunk -> token ids
static void encode_chunk(const BPETokenizer * tok, const std::string & chunk, std::vector<int> & ids) {
    // Byte-level encode
    std::string encoded = byte_level_encode(tok, chunk);

    // Split into individual UTF-8 characters (each is a BPE symbol)
    std::vector<std::string> symbols;
    const char *             s   = encoded.c_str();
    int                      len = (int) encoded.size();
    int                      i   = 0;
    while (i < len) {
        int adv;
        utf8_codepoint(s + i, &adv);
        symbols.push_back(std::string(s + i, adv));
        i += adv;
    }

    // Apply BPE merges
    std::vector<std::string> merged = bpe_merge(tok->merges, symbols);

    // Look up in vocab
    for (const auto & piece : merged) {
        auto it = tok->vocab.find(piece);
        if (it != tok->vocab.end()) {
            ids.push_back(it->second);
        } else {
            // Fallback: encode each byte individually (should not happen with byte-level BPE)
            qt_log(QT_LOG_WARN, "[BPE] WARNING: unknown token '%s'", piece.c_str());
            for (unsigned char c : piece) {
                auto it2 = tok->vocab.find(std::string(1, c));
                if (it2 != tok->vocab.end()) {
                    ids.push_back(it2->second);
                }
            }
        }
    }
}

// Full encode: text -> token ids.
// Walks the text from left to right, matching any registered special token
// verbatim. For each segment between specials, runs the GPT-2 byte-level
// pre-tokenizer + BPE merges. The endoftext sentinel is auto-registered as
// a special by load_bpe_from_gguf, so existing call sites that embed
// "<|endoftext|>" in the input text keep working.
// add_eos = true appends the eos_id at the end (post-processor behavior).
static std::vector<int> bpe_encode(const BPETokenizer * tok, const std::string & text, bool add_eos = true) {
    std::vector<int> ids;

    auto encode_segment = [&](const std::string & seg) {
        if (seg.empty()) {
            return;
        }
        auto chunks = gpt2_pre_tokenize(nfc_normalize(seg));
        for (const auto & chunk : chunks) {
            encode_chunk(tok, chunk, ids);
        }
    };

    size_t pos = 0;
    while (pos < text.size()) {
        // Find the leftmost occurrence of any registered special token.
        size_t best_pos = std::string::npos;
        int    best_idx = -1;
        for (size_t i = 0; i < tok->specials.size(); i++) {
            size_t p = text.find(tok->specials[i].first, pos);
            if (p != std::string::npos && p < best_pos) {
                best_pos = p;
                best_idx = (int) i;
            }
        }

        if (best_idx < 0) {
            encode_segment(text.substr(pos));
            break;
        }

        if (best_pos > pos) {
            encode_segment(text.substr(pos, best_pos - pos));
        }
        const auto & sp = tok->specials[(size_t) best_idx];
        ids.push_back(sp.second);
        pos = best_pos + sp.first.size();
    }

    if (add_eos && tok->eos_id >= 0) {
        ids.push_back(tok->eos_id);
    }
    return ids;
}
