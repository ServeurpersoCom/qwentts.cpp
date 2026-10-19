#include "incremental-text.h"
#include <cassert>
#include <iostream>
#include <vector>

int main() {
    using S = IncrementalText::State;
    using E = IncrementalText::Event;
    IncrementalText x;
    assert(x.apply(E::append, "Hel") && x.state == S::waiting);
    assert(std::string(x.rule) == "R1-append");
    assert(x.apply(E::append, "lo wor") && x.state == S::ready);
    assert(x.take() == "Hello" && x.tail == " wor");
    assert(std::string(x.rule) == "R3-consume");
    assert(x.apply(E::append, "ld ") && x.take() == " world");
    assert(x.apply(E::finish) && x.state == S::draining);
    assert(std::string(x.rule) == "R2-finish");
    assert(x.take() == " " && x.state == S::done);
    assert(!x.apply(E::append, "late"));
    assert(std::string(x.rule) == "R0-terminal");
    IncrementalText cancel;
    assert(cancel.apply(E::cancel) && cancel.state == S::cancelled);
    assert(std::string(cancel.rule) == "R4-cancel");
    assert(!cancel.apply(E::finish));
    IncrementalText overflow;
    assert(!overflow.apply(E::append, std::string(IncrementalText::limit + 1, 'x')));
    assert(overflow.state == S::failed);
    assert(std::string(overflow.rule) == "R6-invalid-input");
    assert(IncrementalText::valid_utf8("\xe4\xbd\xa0\xe5\xa5\xbd"));
    assert(!IncrementalText::valid_utf8("\xe4\xbd"));
    assert(!IncrementalText::valid_utf8("\xc0\xaf"));
    assert(!IncrementalText::valid_utf8(std::string("a\0b", 3)));
    // Every byte split (including within UTF-8) yields identical committed
    // units; EOF must not change segmentation of already available units.
    for (const std::string & sample : {std::string("Hello world "), std::string("你好，世界。"),
                                     std::string("日本語です。"), std::string("한국어 테스트"),
                                     std::string("English 你好 world ")}) {
        auto collect = [](const std::vector<std::string> & parts) {
            IncrementalText input;
            std::vector<std::string> units;
            for (const auto & part : parts) {
                if (!part.empty()) assert(input.apply(E::append, part));
                for (;;) { auto unit = input.take(); if (unit.empty()) break; units.push_back(unit); }
            }
            assert(input.apply(E::finish));
            for (;;) { auto unit = input.take(); if (unit.empty()) break; units.push_back(unit); }
            assert(input.state == S::done);
            return units;
        };
        const auto expected = collect({sample});
        for (size_t split = 0; split <= sample.size(); ++split) {
            assert(collect({sample.substr(0, split), sample.substr(split)}) == expected);
        }
        std::vector<std::string> bytes;
        for (char byte : sample) bytes.emplace_back(1, byte);
        assert(collect(bytes) == expected);
    }
    // A legal producer chunk may complete a previous tail before supplying
    // many small words. The bound applies to unfinished units, not their sum.
    IncrementalText large;
    assert(large.apply(E::append, "x"));
    assert(large.apply(E::append, std::string(" ") + std::string(65535, ' ')));
    assert(large.take() == "x");
    assert(large.tail.size() == IncrementalText::limit);
    assert(large.apply(E::finish));
    assert(large.take().size() == IncrementalText::limit);
    IncrementalText many_words;
    assert(many_words.apply(E::append, "x"));
    std::string full_chunk = " ";
    for (size_t i = 0; i < 32767; ++i) full_chunk += "a ";
    full_chunk += "z";
    assert(full_chunk.size() == IncrementalText::limit);
    assert(many_words.apply(E::append, full_chunk));
    size_t committed = 0;
    for (;;) { auto unit = many_words.take(); if (unit.empty()) break; committed += unit.size(); }
    assert(committed + many_words.tail.size() == 65537);
    assert(many_words.tail == " z");
    IncrementalText oversized_word;
    assert(oversized_word.apply(E::append, std::string(65536, 'x')));
    assert(!oversized_word.apply(E::append, "x "));
    IncrementalText cjk;
    assert(cjk.apply(E::append, "你") && cjk.state == S::ready);
    assert(cjk.take() == "你" && cjk.state == S::waiting); // no EOF needed
    for (int a = 0; a < 5; ++a) for (int b = 0; b < 5; ++b) for (int c = 0; c < 5; ++c) {
        IncrementalText trace;
        for (int v : {a, b, c}) {
            trace.apply(static_cast<E>(v), "word ");
            assert(trace.tail.size() <= IncrementalText::limit);
            // Any nonterminal state can finish or cancel; no dead end.
            auto probe = trace;
            probe.apply(E::cancel);
            assert(probe.state == S::cancelled || probe.state == S::done || probe.state == S::failed);
        }
    }
    std::cout << "incremental input traces passed\n";
}
