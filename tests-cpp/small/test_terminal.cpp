#include <doctest/doctest.h>

#include "../../modules/dasTerminal/src/terminal.h"

#include <algorithm>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

using das_terminal::Snapshot;
using das_terminal::Terminal;

namespace {

std::string rowText(const Snapshot & snapshot, int row, bool alternate = false) {
    const auto & cells = (alternate ? snapshot.alternate : snapshot.normal)
        .rows[static_cast<size_t>(row)];
    std::string result;
    for (const auto & cell : cells) {
        if (cell.width != 0) result += cell.grapheme;
    }
    while (!result.empty() && result.back() == ' ') result.pop_back();
    return result;
}

Snapshot feedInChunks(const std::string & bytes, const std::vector<size_t> & chunks) {
    Terminal terminal(12, 4);
    size_t offset = 0;
    for (size_t chunk : chunks) {
        const size_t count = std::min(chunk, bytes.size() - offset);
        terminal.feed(reinterpret_cast<const uint8_t *>(bytes.data() + offset), count);
        offset += count;
        if (offset == bytes.size()) break;
    }
    if (offset != bytes.size())
        terminal.feed(reinterpret_cast<const uint8_t *>(bytes.data() + offset), bytes.size() - offset);
    return terminal.snapshot();
}

} // namespace

TEST_CASE("terminal: text controls, wrapping, and scrollback") {
    Terminal terminal(5, 2);
    terminal.feed("abc\rZ\r\n12345X");
    const Snapshot snapshot = terminal.snapshot();

    CHECK(rowText(snapshot, 0) == "12345");
    CHECK(rowText(snapshot, 1) == "X");
    REQUIRE(snapshot.normal.scrollback.size() == 1);
    CHECK(snapshot.normal.scrollback[0][0].grapheme == "Z");
    CHECK(snapshot.normal.cursor.row == 1);
    CHECK(snapshot.normal.cursor.column == 1);
}

TEST_CASE("terminal: cursor movement and erase are semantic") {
    Terminal terminal(8, 3);
    terminal.feed("abcdefgh\x1b[2;3HXY\x1b[1D\x1b[K");
    const Snapshot snapshot = terminal.snapshot();

    CHECK(rowText(snapshot, 0) == "abcdefgh");
    CHECK(rowText(snapshot, 1) == "  X");
    CHECK(snapshot.normal.cursor.row == 1);
    CHECK(snapshot.normal.cursor.column == 3);
}

TEST_CASE("terminal: SGR snapshot preserves indexed, RGB, and attributes") {
    Terminal terminal(8, 2);
    terminal.feed("\x1b[1;31mA\x1b[38;2;1;2;3;48;5;200mB\x1b[0mC");
    const Snapshot snapshot = terminal.snapshot();
    const auto & row = snapshot.normal.rows[0];

    CHECK((row[0].attributes & das_terminal::attr_bold) != 0);
    CHECK(row[0].foreground.kind == das_terminal::ColorKind::indexed);
    CHECK(row[0].foreground.index == 1);
    CHECK(row[1].foreground.kind == das_terminal::ColorKind::rgb);
    CHECK(row[1].foreground.red == 1);
    CHECK(row[1].foreground.green == 2);
    CHECK(row[1].foreground.blue == 3);
    CHECK(row[1].background.kind == das_terminal::ColorKind::indexed);
    CHECK(row[1].background.index == 200);
    CHECK(row[2].attributes == das_terminal::attr_none);
    CHECK(row[2].foreground.kind == das_terminal::ColorKind::default_color);
}

TEST_CASE("terminal: title, CWD, alternate screen, modes, and replies are observable") {
    Terminal terminal(10, 3);
    terminal.feed("main\x1b]2;PowerShell\x07\x1b]7;file:///repo\x1b\\"
                  "\x1b[?25l\x1b[?2004h\x1b[?1049halt\x1b[6n");
    Snapshot snapshot = terminal.snapshot();

    CHECK(snapshot.title == "PowerShell");
    CHECK(snapshot.current_directory == "file:///repo");
    CHECK(snapshot.alternate_active);
    CHECK(snapshot.modes.bracketed_paste);
    CHECK_FALSE(snapshot.alternate.cursor.visible);
    CHECK(rowText(snapshot, 0, true) == "alt");
    CHECK(rowText(snapshot, 0) == "main");
    CHECK(terminal.drainReplies() == std::vector<std::string>{"\x1b[1;4R"});

    terminal.feed("\x1b[?1049l");
    snapshot = terminal.snapshot();
    CHECK_FALSE(snapshot.alternate_active);
    CHECK(rowText(snapshot, 0) == "main");
}

TEST_CASE("terminal: UTF-8 wide and combining cells are explicit") {
    Terminal terminal(8, 2);
    terminal.feed(std::string("A") + "\xcc\x81" + "\xe7\x95\x8c" + "B");
    const Snapshot snapshot = terminal.snapshot();
    const auto & row = snapshot.normal.rows[0];

    CHECK(row[0].grapheme == std::string("A") + "\xcc\x81");
    CHECK(row[0].width == 1);
    CHECK(row[1].grapheme == "\xe7\x95\x8c");
    CHECK(row[1].width == 2);
    CHECK(row[2].width == 0);
    CHECK(row[3].grapheme == "B");
}

TEST_CASE("terminal: parsing is invariant under arbitrary byte chunks") {
    const std::string stream =
        "hello\r\n\x1b[1;34mblue\x1b[0m \xe7\x95\x8c\r\n"
        "\x1b]2;chunk test\x1b\\\x1b[?2004h\x1b[2;5H!";
    const Snapshot expected = feedInChunks(stream, {stream.size()});

    CHECK(feedInChunks(stream, std::vector<size_t>(stream.size(), 1)) == expected);

    std::mt19937 random(0x5eedu);
    for (int run = 0; run != 100; ++run) {
        std::vector<size_t> chunks;
        size_t remaining = stream.size();
        while (remaining) {
            const size_t chunk = std::min<size_t>(remaining, 1u + random() % 11u);
            chunks.push_back(chunk);
            remaining -= chunk;
        }
        CHECK(feedInChunks(stream, chunks) == expected);
    }
}
