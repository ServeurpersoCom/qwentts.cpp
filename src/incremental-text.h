#pragma once

#include "qwen.h"
#include <cctype>
#include <stdexcept>
#include <string>

// The input reducer owns only the mutable UTF-8 tail. Its shell requests
// chunks when take() returns WAIT; no model or I/O is part of the reducer.
class IncrementalText {
public:
    enum class State { waiting, ready, draining, done, cancelled, failed };
    enum class Event { append, finish, consume, cancel, fail };
    static constexpr size_t limit = 65536;
    State state = State::waiting;
    std::string tail;
    const char * rule = "R0";

    bool apply(Event event, const std::string & bytes = {}) {
        if (state == State::cancelled || state == State::failed || state == State::done) {
            rule = "R0-terminal";
            return false;
        }
        switch (event) {
            case Event::append:
                if (state == State::draining || bytes.empty() || bytes.size() > limit || tail.size() > 2 * limit - bytes.size()) {
                    state = State::failed; rule = "R6-invalid-input"; return false;
                }
                tail += bytes;
                // One incoming chunk can complete an existing tail. Validate
                // each canonical unit and final suffix before rejecting it.
                for (size_t pos = 0; pos < tail.size();) {
                    size_t n = boundary(pos);
                    if (!n) n = tail.size() - pos;
                    if (n > limit) { state = State::failed; rule = "R6-invalid-input"; return false; }
                    pos += n;
                }
                state = boundary() ? State::ready : State::waiting;
                rule = "R1-append"; return true;
            case Event::finish:
                state = tail.empty() ? State::done : State::draining;
                rule = "R2-finish"; return true;
            case Event::consume:
                if (state == State::draining) {
                    size_t n = boundary(); tail.erase(0, n ? n : tail.size());
                    state = tail.empty() ? State::done : State::draining;
                }
                else if (state == State::ready) {
                    tail.erase(0, boundary()); state = boundary() ? State::ready : State::waiting;
                }
                else { rule = "R0-not-ready"; return false; }
                rule = "R3-consume"; return true;
            case Event::cancel:
                state = State::cancelled; tail.clear(); rule = "R4-cancel"; return true;
            case Event::fail:
                state = State::failed; tail.clear(); rule = "R5-fail"; return true;
        }
        return false;
    }

    std::string take() {
        size_t n = state == State::draining ? (boundary() ? boundary() : tail.size()) : state == State::ready ? boundary() : 0;
        std::string result = tail.substr(0, n);
        if (n) apply(Event::consume);
        return result;
    }

    static bool valid_utf8(const std::string & s) {
        for (size_t i = 0; i < s.size();) {
            unsigned char c = (unsigned char)s[i++];
            if (c < 0x80) { if (c == 0) return false; continue; }
            int n = c >= 0xc2 && c <= 0xdf ? 1 : c >= 0xe0 && c <= 0xef ? 2 : c >= 0xf0 && c <= 0xf4 ? 3 : -1;
            if (n < 0 || i + (size_t)n > s.size()) return false;
            unsigned char first = (unsigned char)s[i];
            if ((c == 0xe0 && first < 0xa0) || (c == 0xed && first >= 0xa0) ||
                (c == 0xf0 && first < 0x90) || (c == 0xf4 && first >= 0x90)) return false;
            while (n--) { if (((unsigned char)s[i++] & 0xc0) != 0x80) return false; }
        }
        return true;
    }

private:
    size_t boundary(size_t start = 0) const {
        // Canonical units do not depend on producer chunk boundaries:
        // whitespace-prefixed words, or one CJK/Hangul/Kana codepoint.
        // CJK units deliberately prevent cross-character BPE merges.
        bool content = false;
        for (size_t i = start; i < tail.size();) {
            unsigned char c = (unsigned char)tail[i];
            if (c < 128 && std::isspace(c)) {
                if (content) return i - start;
                ++i; continue;
            }
            size_t n = c < 128 ? 1 : c >= 0xc2 && c <= 0xdf ? 2 : c >= 0xe0 && c <= 0xef ? 3 : c >= 0xf0 && c <= 0xf4 ? 4 : 1;
            if (i + n > tail.size()) return 0; // split UTF-8 tail
            unsigned cp = c & (n == 1 ? 0x7f : n == 2 ? 0x1f : n == 3 ? 0x0f : 0x07);
            for (size_t j = 1; j < n; ++j) cp = (cp << 6) | ((unsigned char)tail[i+j] & 0x3f);
            bool script = (cp >= 0x3000 && cp <= 0x30ff) || (cp >= 0x3400 && cp <= 0x9fff) ||
                          (cp >= 0xac00 && cp <= 0xd7af) || (cp >= 0xf900 && cp <= 0xfaff) ||
                          (cp >= 0xff00 && cp <= 0xffef) || (cp >= 0x20000 && cp <= 0x323af);
            if (script) return content ? i - start : i + n - start;
            content = true;
            i += n;
        }
        return 0;
    }
};
