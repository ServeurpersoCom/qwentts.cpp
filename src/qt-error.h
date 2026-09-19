#pragma once
// qt-error.h: internal helpers backing the public qt_last_error() entry
// and the qt_log_set callback routing.
//
// Not part of the public ABI. Translation units that emit user-facing
// errors include this header to record a diagnostic on the calling thread
// before they return a negative qt_status (or NULL). The actual storage
// and the public qt_last_error() reader live in qwen.cpp.
//
// Storage is thread_local so concurrent qt_synthesize calls on different
// threads never race on each other's messages. The setter is variadic with
// printf semantics; messages longer than the internal buffer are
// truncated, never split. Passing NULL as fmt clears the slot.
//
// qt_throw is the load-path counterpart: functions deep inside the GGUF
// reader and the codec load chain cannot return false up dozens of call
// sites without a massive cascade. They throw a std::runtime_error
// instead, which the ABI boundary entries (qt_init, qt_synthesize) catch
// and convert into qt_set_error + a negative qt_status. Exceptions never
// cross the extern "C" boundary, so the public API stays pure C.
//
// qt_log routes a formatted message to the user-installed qt_log_cb, or
// to stderr when no callback is installed. Used by every translation unit
// in the lib that wants its diagnostics to be redirectable from a wrapper
// (Python logging, Rust tracing, ...). The level enum lives in qwen.h.

#include "qwen.h"

#include <cstdarg>

void qt_set_error(const char * fmt, ...)
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((format(printf, 1, 2)))
#endif
    ;

void qt_set_error_v(const char * fmt, va_list ap);

// Throws std::runtime_error formatted with printf semantics. Tagged
// noreturn so the compiler can prune unreachable branches at the call
// site. Designed for the GGUF / codec load path where any failure means
// the model is unusable and unwinding to the ABI boundary is the only
// sane recovery.
[[noreturn]] void qt_throw(const char * fmt, ...)
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((format(printf, 1, 2)))
#endif
    ;

// Routes a formatted message at the requested level to the installed
// qt_log_cb, or to stderr with a trailing newline when no callback is set.
// Every diagnostic of the library goes through here, ggml's included, so a
// host that installs a callback gets all of them. printf semantics.
void qt_log(enum qt_log_level level, const char * fmt, ...)
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((format(printf, 2, 3)))
#endif
    ;
