#pragma once
// sampling-defaults.h: the generation defaults of the talker and the
// sub-talker, matching the upstream Python reference. qt_tts_default_params
// seeds every qt_tts_params from these values; tools override only the
// fields the caller sets explicitly.

#define QT_DEFAULT_MAX_NEW_TOKENS        2048
#define QT_DEFAULT_TEMPERATURE           0.9f
#define QT_DEFAULT_TOP_K                 50
#define QT_DEFAULT_TOP_P                 1.0f
#define QT_DEFAULT_REPETITION_PENALTY    1.05f
#define QT_DEFAULT_SUBTALKER_TEMPERATURE 0.9f
#define QT_DEFAULT_SUBTALKER_TOP_K       50
#define QT_DEFAULT_SUBTALKER_TOP_P       1.0f
