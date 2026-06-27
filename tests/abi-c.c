/* tests/abi-c.c: link-only ABI smoke test for qwen.h.
 *
 * Compiled in pure C99 with -Wall -Werror -pedantic. The purpose of this
 * test is NOT to run a full synthesis (no GGUF loaded, no model required);
 * it is to guarantee at every build that :
 *
 *   1. qwen.h parses with a C compiler (no <cstdio>, no std::*, no
 *      C++-only forward declarations leak in).
 *   2. Every public qwen_* symbol has C linkage and links from a C
 *      translation unit.
 *   3. The structs are POD and zero-initialisable with `{0}` from C.
 *   4. The qt_log_set callback routes formatted messages from the lib
 *      to the user, and abi_version validation rejects future structs.
 *
 * If this test stops compiling or stops linking, the public ABI has
 * regressed and the build breaks before anything else.
 */

#include "qwen.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool stub_cancel(void * ud) {
    (void) ud;
    return false;
}

static bool stub_on_chunk(const float * samples, int n_samples, void * ud) {
    (void) samples;
    (void) n_samples;
    (void) ud;
    return true;
}

/* Counter incremented by the stub log callback. The probe checks that at
 * least one log line was routed through the callback by triggering a
 * qt_init failure (which emits a [Qwen] ERROR line via qt_log). */
static int               g_log_lines         = 0;
static enum qt_log_level g_last_log_level    = QT_LOG_DEBUG;
static char              g_last_log_msg[512] = { 0 };

static void stub_log(enum qt_log_level level, const char * msg, void * user_data) {
    (void) user_data;
    g_log_lines++;
    g_last_log_level = level;
    if (msg) {
        size_t n = strlen(msg);
        if (n >= sizeof(g_last_log_msg)) {
            n = sizeof(g_last_log_msg) - 1;
        }
        memcpy(g_last_log_msg, msg, n);
        g_last_log_msg[n] = '\0';
    }
}

int main(void) {
    /* Static version string, always reachable. */
    const char * version = qt_version();
    printf("[Probe] %s\n", version);

    /* Default-initialise the public structs from C. */
    struct qt_init_params iparams;
    qt_init_default_params(&iparams);

    struct qt_tts_params params;
    qt_tts_default_params(&params);

    /* Sanity-check a few default values, including the abi_version and
     * the new use_fa / clamp_fp16 / on_chunk / codec_chunk_sec /
     * codec_left_context_sec slots. */
    if (params.max_new_tokens != 2048 || params.codec_chunk_sec <= 0.0f || params.codec_left_context_sec < 0.0f) {
        fprintf(stderr, "[Probe] default values do not match\n");
        return 1;
    }
    if (iparams.abi_version != QT_ABI_VERSION || params.abi_version != QT_ABI_VERSION) {
        fprintf(stderr, "[Probe] abi_version not set by qt_*_default_params\n");
        return 1;
    }
    if (!iparams.use_fa || iparams.clamp_fp16) {
        fprintf(stderr, "[Probe] init_params defaults do not match (use_fa=true, clamp_fp16=false)\n");
        return 1;
    }

    /* Touch every reference-pointer field, every callback typedef and
     * every output struct field so the compiler validates the layout
     * end-to-end without ever needing a model. */
    params.cancel             = stub_cancel;
    params.cancel_user_data   = NULL;
    params.on_chunk           = stub_on_chunk;
    params.on_chunk_user_data = NULL;

    struct qt_audio audio = { 0 };
    qt_audio_free(&audio);

    struct qt_voice_ref voice_ref = { 0 };
    qt_voice_ref_free(&voice_ref);

    /* Install the log callback before the failing init so the [Qwen]
     * ERROR line lands on stub_log instead of stderr. */
    qt_log_set(stub_log, NULL);

    /* Call every entry through its early-return path. qt_init returns
     * NULL on missing paths, qt_synthesize / qt_duration_sec_to_tokens
     * fail on NULL handle, qt_free is safe on NULL. None of these load a
     * model, but the linker must resolve every name to satisfy the call. */
    struct qt_context * dummy = qt_init(NULL);
    if (dummy != NULL) {
        fprintf(stderr, "[Probe] qt_init(NULL) was supposed to return NULL\n");
        qt_free(dummy);
        return 2;
    }

    /* qt_init(NULL) just failed -> qt_last_error() must point to a
     * non-empty thread-local string. Pointer is always valid (c_str on
     * an empty std::string still gives a NUL byte), so we only need to
     * check the first byte to confirm an error was actually recorded. */
    const char * err = qt_last_error();
    if (err == NULL || err[0] == '\0') {
        fprintf(stderr, "[Probe] qt_last_error() empty after a known failure\n");
        return 5;
    }

    /* The same failure must have surfaced through the log callback at
     * ERROR level. */
    if (g_log_lines == 0) {
        fprintf(stderr, "[Probe] qt_log_set callback never invoked\n");
        return 6;
    }
    if (g_last_log_level != QT_LOG_ERROR) {
        fprintf(stderr, "[Probe] last log level was %d, expected %d\n", (int) g_last_log_level, (int) QT_LOG_ERROR);
        return 7;
    }
    printf("[Probe] qt_log_set routed %d line(s), last: '%s'\n", g_log_lines, g_last_log_msg);
    printf("[Probe] qt_last_error reads '%s'\n", err);

    /* abi_version validation : a struct claiming a future ABI must be
     * rejected up front, before any allocation. */
    struct qt_init_params future_iparams;
    qt_init_default_params(&future_iparams);
    future_iparams.talker_path   = "irrelevant.gguf";
    future_iparams.codec_path    = "irrelevant.gguf";
    future_iparams.abi_version   = QT_ABI_VERSION + 1;
    struct qt_context * rejected = qt_init(&future_iparams);
    if (rejected != NULL) {
        fprintf(stderr, "[Probe] qt_init accepted a future abi_version\n");
        qt_free(rejected);
        return 8;
    }

    enum qt_status rc = qt_synthesize(NULL, &params, &audio);
    if (rc != QT_STATUS_INVALID_PARAMS) {
        fprintf(stderr, "[Probe] qt_synthesize(NULL) returned %d, expected %d\n", (int) rc,
                (int) QT_STATUS_INVALID_PARAMS);
        return 3;
    }

    rc = qt_extract_voice_ref(NULL, NULL, 0, &voice_ref);
    if (rc != QT_STATUS_INVALID_PARAMS) {
        fprintf(stderr, "[Probe] qt_extract_voice_ref(NULL) returned %d, expected %d\n", (int) rc,
                (int) QT_STATUS_INVALID_PARAMS);
        return 11;
    }

    int frames = qt_duration_sec_to_tokens(NULL, 1.0f);
    if (frames < 1) {
        fprintf(stderr, "[Probe] qt_duration_sec_to_tokens returned %d, expected >= 1\n", frames);
        return 4;
    }

    /* Speaker getters resolve and behave on a NULL handle : count is 0 and
     * any index returns NULL. No model is needed to lock in the contract. */
    if (qt_n_speakers(NULL) != 0) {
        fprintf(stderr, "[Probe] qt_n_speakers(NULL) must be 0\n");
        return 9;
    }
    if (qt_speaker_name(NULL, 0) != NULL) {
        fprintf(stderr, "[Probe] qt_speaker_name(NULL, 0) must be NULL\n");
        return 10;
    }

    /* Restore the default stderr fallback before exit so the trailing
     * [Qwen] log lines from the cleanup paths land where the user
     * expects them. */
    qt_log_set(NULL, NULL);

    qt_free(NULL);
    qt_audio_free(&audio);
    qt_voice_ref_free(&voice_ref);

    return 0;
}
