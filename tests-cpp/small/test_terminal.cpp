#include <doctest/doctest.h>

#include "daScript/ast/ast.h"
#include "daScript/ast/ast_handle.h"
#include "daScript/ast/ast_interop.h"
#include "daScript/misc/sysos.h"
#include "../../modules/dasTerminal/src/aot_builtin_terminal.h"
#include "../../modules/dasTerminal/src/terminal.h"
#include "../../modules/dasTerminal/src/pty.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <random>
#include <string>
#include <thread>
#include <vector>

using das_terminal::Snapshot;
using das_terminal::Key;
using das_terminal::KeyEvent;
using das_terminal::PtyProcess;
using das_terminal::PtyProcessOptions;
using das_terminal::PtyReadStatus;
using das_terminal::Terminal;

namespace {

std::string cellRowText(const std::vector<das_terminal::Cell> & cells) {
    std::string result;
    for (const auto & cell : cells) {
        if (cell.width != 0) result += cell.grapheme;
    }
    while (!result.empty() && result.back() == ' ') result.pop_back();
    return result;
}

std::string rowText(const Snapshot & snapshot, int row, bool alternate = false) {
    const auto & cells = (alternate ? snapshot.alternate : snapshot.normal)
        .rows[static_cast<size_t>(row)];
    return cellRowText(cells);
}

Snapshot feedInChunks(const std::string & bytes, const std::vector<size_t> & chunks,
                      int columns = 12, int rows = 4) {
    Terminal terminal(columns, rows);
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

struct SemanticCase {
    const char * name;
    int columns;
    int rows;
    std::string bytes;
    std::vector<std::string> expected_rows;
    int cursor_row;
    int cursor_column;
};

void checkSemanticCase(const SemanticCase & test) {
    CAPTURE(test.name);
    const Snapshot expected = feedInChunks(test.bytes, {test.bytes.size()}, test.columns, test.rows);
    REQUIRE(expected.normal.rows.size() == test.expected_rows.size());
    for (size_t row = 0; row != test.expected_rows.size(); ++row)
        CHECK(rowText(expected, static_cast<int>(row)) == test.expected_rows[row]);
    CHECK(expected.normal.cursor.row == test.cursor_row);
    CHECK(expected.normal.cursor.column == test.cursor_column);
    CHECK(expected.unknown_sequences.empty());

    CHECK(feedInChunks(test.bytes, std::vector<size_t>(test.bytes.size(), 1),
                       test.columns, test.rows) == expected);

    std::mt19937 random(0x5eedu);
    for (int run = 0; run != 20; ++run) {
        std::vector<size_t> chunks;
        size_t remaining = test.bytes.size();
        while (remaining) {
            const size_t chunk = std::min<size_t>(remaining, 1u + random() % 7u);
            chunks.push_back(chunk);
            remaining -= chunk;
        }
        CHECK(feedInChunks(test.bytes, chunks, test.columns, test.rows) == expected);
    }
}

bool snapshotContains(const Snapshot & snapshot, const std::string & needle) {
    std::string text;
    const auto append_rows = [&text](const std::vector<std::vector<das_terminal::Cell>> & rows) {
        for (const auto & row : rows) {
            for (const auto & cell : row)
                if (cell.width != 0) text += cell.grapheme;
            text.push_back('\n');
        }
    };
    append_rows(snapshot.normal.scrollback);
    append_rows(snapshot.normal.rows);
    append_rows(snapshot.alternate.scrollback);
    append_rows(snapshot.alternate.rows);
    return text.find(needle) != std::string::npos;
}

bool waitForTerminalText(PtyProcess & process, Terminal & terminal,
                         const std::string & needle, uint32_t timeout_ms,
                         std::string & error) {
    const auto deadline = std::chrono::steady_clock::now() +
        std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        std::string bytes;
        const PtyReadStatus status = process.read(bytes, error);
        if (status == PtyReadStatus::data) {
            terminal.feed(bytes);
            if (snapshotContains(terminal.snapshot(), needle)) return true;
            continue;
        }
        if (status == PtyReadStatus::error) return false;
        if (status == PtyReadStatus::closed) {
            if (snapshotContains(terminal.snapshot(), needle)) return true;
            error = "ConPTY closed before terminal snapshot contained '" + needle + "'";
            return false;
        }
        if (snapshotContains(terminal.snapshot(), needle)) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    error = "timed out waiting for terminal snapshot to contain '" + needle + "'";
    return false;
}

struct PtyProcessGuard {
    explicit PtyProcessGuard(PtyProcess * value) : process(value) {}

    ~PtyProcessGuard() {
        if (!process) return;
        uint32_t exit_code = 0;
        std::string error;
        if (!process->wait(0, exit_code, error)) {
            error.clear();
            process->terminate(99, error);
            error.clear();
            process->wait(5000, exit_code, error);
        }
    }

    PtyProcess * process;
};

struct ResizeProbePty final : PtyProcess {
    uint32_t processId() const override { return 1; }
    PtyReadStatus read(std::string &, std::string &, size_t) override {
        return PtyReadStatus::idle;
    }
    bool write(const uint8_t *, size_t, std::string &) override { return true; }
    bool resize(int columns, int rows, std::string & error) override {
        resize_called = true;
        resize_columns = columns;
        resize_rows = rows;
        if (resize_succeeds) return true;
        error = "injected resize failure";
        return false;
    }
    bool wait(uint32_t, uint32_t &, std::string &) override { return false; }
    bool terminate(uint32_t, std::string &) override { return true; }

    bool resize_succeeds = false;
    bool resize_called = false;
    int resize_columns = 0;
    int resize_rows = 0;
};

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

TEST_CASE("terminal: scrollback keeps the newest 10000 rows") {
    Terminal terminal(8, 1);
    std::string stream;
    for (int line = 0; line != 10005; ++line)
        stream += "L" + std::to_string(line) + "\r\n";
    terminal.feed(stream);
    const Snapshot snapshot = terminal.snapshot();

    REQUIRE(snapshot.normal.scrollback.size() == 10000);
    CHECK(cellRowText(snapshot.normal.scrollback.front()) == "L5");
    CHECK(cellRowText(snapshot.normal.scrollback.back()) == "L10004");
}

TEST_CASE("terminal: resize keeps screen and scrollback widths consistent") {
    Terminal terminal(5, 1);
    terminal.feed("abc\r\n");
    terminal.resize(7, 1);
    Snapshot snapshot = terminal.snapshot();

    REQUIRE(snapshot.normal.scrollback.size() == 1);
    CHECK(snapshot.normal.rows[0].size() == 7);
    CHECK(snapshot.normal.scrollback[0].size() == 7);
    CHECK(cellRowText(snapshot.normal.scrollback[0]) == "abc");

    terminal.resize(2, 1);
    snapshot = terminal.snapshot();
    CHECK(snapshot.normal.rows[0].size() == 2);
    CHECK(snapshot.normal.scrollback[0].size() == 2);
    CHECK(cellRowText(snapshot.normal.scrollback[0]) == "ab");
}

TEST_CASE("terminal: resize does not leave a truncated wide cell") {
    Terminal terminal(4, 1);
    terminal.feed(std::string("A") + "\xe7\x95\x8c\r\n");
    terminal.resize(2, 1);
    const Snapshot snapshot = terminal.snapshot();

    REQUIRE(snapshot.normal.scrollback.size() == 1);
    REQUIRE(snapshot.normal.scrollback[0].size() == 2);
    CHECK(cellRowText(snapshot.normal.scrollback[0]) == "A");
    CHECK(snapshot.normal.scrollback[0][1].width == 1);
}

TEST_CASE("terminal: failed PTY resize preserves emulator dimensions") {
    auto terminal = das::make_smart<das_terminal::TerminalHandle>(8, 2);
    auto * process = new ResizeProbePty();
    terminal->process.reset(process);

    das_terminal::builtin_terminal_resize(terminal, 20, 5);
    CHECK(process->resize_called);
    CHECK(process->resize_columns == 20);
    CHECK(process->resize_rows == 5);
    CHECK(terminal->state.columns() == 8);
    CHECK(terminal->state.rows() == 2);
    CHECK(terminal->transport_error == "injected resize failure");

    process->resize_succeeds = true;
    das_terminal::builtin_terminal_resize(terminal, 20, 5);
    CHECK(terminal->state.columns() == 20);
    CHECK(terminal->state.rows() == 5);
    CHECK(terminal->transport_error.empty());
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

TEST_CASE("terminal: Core cursor controls have exact chunk-invariant semantics") {
    const std::vector<SemanticCase> cases = {
        {"CUU defaults and clamps", 8, 4, "0\r\n1\r\n2\x1b[A\x1b[99A@",
            {"0@", "1", "2", ""}, 0, 2},
        {"CUD CUF and CUB clamp", 8, 4, "\x1b[2;2HA\x1b[99B\x1b[99C\x1b[2DB",
            {"", " A", "", "     B"}, 3, 6},
        {"CNL and CPL reset the column", 8, 4, "a\r\nb\x1b[Ec\x1b[Fd",
            {"a", "d", "c", ""}, 1, 1},
        {"CHA VPA and HVP are one-based", 8, 4, "\x1b[2;3HA\x1b[6GB\x1b[4dC",
            {"", "  A  B", "", "      C"}, 3, 7},
        {"HPR and VPR move relative and clamp", 8, 4, "\x1b[2;2H\x1b[3a\x1b[9eQ",
            {"", "", "", "    Q"}, 3, 5},
        {"CUP defaults home", 8, 4, "\x1b[4;8H!\x1b[H@",
            {"@", "", "", "       !"}, 0, 1},
    };
    for (const SemanticCase & test : cases) checkSemanticCase(test);
}

TEST_CASE("terminal: Core erase controls have exact chunk-invariant semantics") {
    const std::vector<SemanticCase> cases = {
        {"EL erases through the end", 6, 3, "abcdef\x1b[1;3H\x1b[K",
            {"ab", "", ""}, 0, 2},
        {"EL 1 erases through the cursor", 6, 3, "abcdef\x1b[1;3H\x1b[1K",
            {"   def", "", ""}, 0, 2},
        {"EL 2 erases the whole line", 6, 3, "abcdef\x1b[1;3H\x1b[2K",
            {"", "", ""}, 0, 2},
        {"ED erases through the end", 6, 3, "abcdef\r\nghijkl\r\nmnopqr\x1b[2;3H\x1b[J",
            {"abcdef", "gh", ""}, 1, 2},
        {"ED 1 erases through the cursor", 6, 3, "abcdef\r\nghijkl\r\nmnopqr\x1b[2;3H\x1b[1J",
            {"", "   jkl", "mnopqr"}, 1, 2},
        {"ED 2 erases the display", 6, 3, "abcdef\r\nghijkl\r\nmnopqr\x1b[2;3H\x1b[2J",
            {"", "", ""}, 1, 2},
        {"ECH erases without moving", 6, 3, "abcdef\x1b[1;3H\x1b[2X",
            {"ab  ef", "", ""}, 0, 2},
        {"ECH preserves wide-cell invariants", 6, 3, "A\xe7\x95\x8c" "B\x1b[1;3H\x1b[X",
            {"A  B", "", ""}, 0, 2},
    };
    for (const SemanticCase & test : cases) checkSemanticCase(test);
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

TEST_CASE("terminal: SGR reset preserves an active OSC 8 hyperlink") {
    Terminal terminal(8, 2);
    terminal.feed("\x1b]8;;https://example.test\x1b\\"
                  "\x1b[31mA\x1b[0mB"
                  "\x1b]8;;\x1b\\C");
    const Snapshot snapshot = terminal.snapshot();
    const auto & row = snapshot.normal.rows[0];

    CHECK(row[0].hyperlink == "https://example.test");
    CHECK(row[1].hyperlink == "https://example.test");
    CHECK(row[1].foreground.kind == das_terminal::ColorKind::default_color);
    CHECK(row[2].hyperlink.empty());
}

TEST_CASE("terminal: oversized CSI parameters clamp without overflow") {
    const std::string huge = "999999999999999999999999999999999999999999999999";
    Terminal terminal(4, 3);
    terminal.feed("X\x1b[" + huge + "C@\x1b[2;2H\x1b[" + huge + "B#");
    Snapshot snapshot = terminal.snapshot();

    CHECK(rowText(snapshot, 0) == "X  @");
    CHECK(rowText(snapshot, 2) == " #");
    CHECK(snapshot.normal.cursor.row == 2);
    CHECK(snapshot.normal.cursor.column == 2);
    CHECK(snapshot.unknown_sequences.empty());

    terminal.feed("\x1b[1;1Habcd\x1b[1;2H\x1b[" + huge + "X");
    snapshot = terminal.snapshot();
    CHECK(rowText(snapshot, 0) == "a");
    CHECK(snapshot.normal.cursor.column == 1);
    CHECK(snapshot.unknown_sequences.empty());
}

TEST_CASE("terminal: ESC cancels an incomplete CSI sequence") {
    Terminal terminal(8, 2);
    terminal.feed("\x1b[12;\x1bZ");
    const Snapshot snapshot = terminal.snapshot();

    REQUIRE(snapshot.unknown_sequences.size() == 1);
    CHECK(snapshot.unknown_sequences[0] == "\x1bZ");
}

TEST_CASE("terminal: overlong control strings are bounded and ignored") {
    Terminal terminal(8, 2);
    terminal.feed(std::string("\x1b[") + std::string(5000, ';') + "mX");
    Snapshot snapshot = terminal.snapshot();
    CHECK(rowText(snapshot, 0) == "X");
    CHECK(snapshot.unknown_sequences.empty());

    terminal.feed(std::string("\x1b]0;") + std::string(70 * 1024, 'a') + "\x1b\\Y");
    snapshot = terminal.snapshot();
    CHECK(rowText(snapshot, 0) == "XY");
    CHECK(snapshot.title.empty());

    terminal.feed(std::string("\x1b]999;") + std::string(60 * 1024, 'z') + "\x07");
    snapshot = terminal.snapshot();
    REQUIRE(snapshot.unknown_sequences.size() == 1);
    CHECK(snapshot.unknown_sequences[0].size() == 4096);
    CHECK(snapshot.unknown_sequences[0].substr(4082) == "...[truncated]");
}

TEST_CASE("terminal: empty DEC private mode parameters are ignored") {
    Terminal terminal(8, 2);
    terminal.feed("\x1b[?h\x1b[?;25l");
    Snapshot snapshot = terminal.snapshot();
    CHECK_FALSE(snapshot.normal.cursor.visible);
    CHECK(snapshot.unknown_sequences.empty());

    terminal.feed("\x1b[?25;h");
    snapshot = terminal.snapshot();
    CHECK(snapshot.normal.cursor.visible);
    CHECK(snapshot.unknown_sequences.empty());
}

TEST_CASE("terminal: Core SGR set and reset forms are explicit") {
    struct SgrCase {
        const char * name;
        std::string bytes;
        uint16_t attributes;
        das_terminal::ColorKind foreground_kind;
        int foreground_index;
        das_terminal::ColorKind background_kind;
        int background_index;
    };
    const uint16_t all_attributes = das_terminal::attr_bold | das_terminal::attr_faint |
        das_terminal::attr_italic | das_terminal::attr_underline | das_terminal::attr_blink |
        das_terminal::attr_inverse | das_terminal::attr_hidden | das_terminal::attr_strike;
    const std::vector<SgrCase> cases = {
        {"set attributes", "\x1b[1;2;3;4;5;7;8;9mX", all_attributes,
            das_terminal::ColorKind::default_color, 0, das_terminal::ColorKind::default_color, 0},
        {"rapid blink and double underline map to canonical flags", "\x1b[6;21mX",
            das_terminal::attr_blink | das_terminal::attr_underline,
            das_terminal::ColorKind::default_color, 0, das_terminal::ColorKind::default_color, 0},
        {"reset attributes", "\x1b[1;2;3;4;5;7;8;9m\x1b[22;23;24;25;27;28;29mX",
            das_terminal::attr_none, das_terminal::ColorKind::default_color, 0,
            das_terminal::ColorKind::default_color, 0},
        {"bright indexed colors", "\x1b[91;104mX", das_terminal::attr_none,
            das_terminal::ColorKind::indexed, 9, das_terminal::ColorKind::indexed, 12},
        {"default colors", "\x1b[31;44m\x1b[39;49mX", das_terminal::attr_none,
            das_terminal::ColorKind::default_color, 0, das_terminal::ColorKind::default_color, 0},
    };

    for (const SgrCase & test : cases) {
        CAPTURE(test.name);
        const Snapshot expected = feedInChunks(test.bytes, {test.bytes.size()});
        const auto & cell = expected.normal.rows[0][0];
        CHECK(cell.attributes == test.attributes);
        CHECK(cell.foreground.kind == test.foreground_kind);
        CHECK(cell.foreground.index == test.foreground_index);
        CHECK(cell.background.kind == test.background_kind);
        CHECK(cell.background.index == test.background_index);
        CHECK(expected.unknown_sequences.empty());
        CHECK(feedInChunks(test.bytes, std::vector<size_t>(test.bytes.size(), 1)) == expected);
    }
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

TEST_CASE("terminal: full reset drops queued replies and events") {
    Terminal terminal(8, 2);
    terminal.feed("before\x07\x1b]2;title\x07\x1b[5n\x1b" "c");
    const Snapshot snapshot = terminal.snapshot();

    CHECK(rowText(snapshot, 0).empty());
    CHECK(snapshot.title.empty());
    CHECK(terminal.drainReplies().empty());
    CHECK(terminal.takeEvents().empty());

    terminal.feed("\x07");
    CHECK(terminal.takeEvents() == std::vector<std::string>{"bell"});
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

TEST_CASE("terminal: overwriting a wide-cell continuation clears the whole grapheme") {
    Terminal terminal(6, 2);
    terminal.feed(std::string("A") + "\xe7\x95\x8c" + "B\x1b[1;3H@");
    const Snapshot snapshot = terminal.snapshot();
    const auto & row = snapshot.normal.rows[0];

    CHECK(rowText(snapshot, 0) == "A @B");
    CHECK(row[1].grapheme == " ");
    CHECK(row[1].width == 1);
    CHECK(row[2].grapheme == "@");
    CHECK(row[2].width == 1);
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

TEST_CASE("terminal: wide glyph at an unwrapped right edge stays in-grid") {
    Terminal terminal(3, 2);
    terminal.feed("\x1b[?7l" "AB\xe7\x95\x8c");
    Snapshot snapshot = terminal.snapshot();
    CHECK(rowText(snapshot, 0) == "AB\xe7\x95\x8c");
    CHECK(snapshot.normal.rows[0][2].width == 1);
    CHECK(snapshot.normal.cursor.column == 2);

    Terminal narrow(1, 2);
    narrow.feed("\xe7\x95\x8c");
    snapshot = narrow.snapshot();
    CHECK(rowText(snapshot, 0) == "\xe7\x95\x8c");
    CHECK(snapshot.normal.rows[0][0].width == 1);
}

TEST_CASE("terminal: normalized key encoding follows VT modes and modifiers") {
    struct KeyCase {
        const char * name;
        KeyEvent event;
        std::string expected;
    };
    Terminal terminal(8, 2);
    const std::vector<KeyCase> normal_cases = {
        {"text", KeyEvent(Key::text, "x"), "x"},
        {"Alt text", KeyEvent(Key::text, "x", false, true), "\x1bx"},
        {"Control text", KeyEvent(Key::text, "c", false, false, true), "\x03"},
        {"Control space", KeyEvent(Key::text, " ", false, false, true), std::string(1, '\0')},
        {"Enter", KeyEvent(Key::enter), "\r"},
        {"Alt Enter", KeyEvent(Key::enter, "", false, true), "\x1b\r"},
        {"Shift Tab", KeyEvent(Key::tab, "", true), "\x1b[Z"},
        {"Backspace", KeyEvent(Key::backspace), "\x7f"},
        {"Up", KeyEvent(Key::up), "\x1b[A"},
        {"Shift Left", KeyEvent(Key::left, "", true), "\x1b[1;2D"},
        {"Control Alt Right", KeyEvent(Key::right, "", false, true, true), "\x1b[1;7C"},
        {"Control Delete", KeyEvent(Key::delete_key, "", false, false, true), "\x1b[3;5~"},
        {"F1", KeyEvent(Key::f1), "\x1bOP"},
        {"Shift F4", KeyEvent(Key::f4, "", true), "\x1b[1;2S"},
        {"Alt F12", KeyEvent(Key::f12, "", false, true), "\x1b[24;3~"},
    };
    for (const KeyCase & test : normal_cases) {
        CAPTURE(test.name);
        CHECK(terminal.encodeKey(test.event) == test.expected);
    }

    terminal.feed("\x1b[?1h");
    CHECK(terminal.encodeKey(KeyEvent(Key::up)) == "\x1bOA");
    CHECK(terminal.encodeKey(KeyEvent(Key::home)) == "\x1bOH");
    CHECK(terminal.encodeKey(KeyEvent(Key::down, "", false, false, true)) == "\x1b[1;5B");
    terminal.feed("\x1b[?1l");
    CHECK(terminal.encodeKey(KeyEvent(Key::home)) == "\x1b[H");
}

TEST_CASE("terminal: paste encoding and terminal replies remain separate") {
    Terminal terminal(8, 2);
    const std::string paste = std::string("one\r\n") + "\xe7\x95\x8c";

    CHECK(terminal.encodePaste(paste) == paste);
    terminal.feed("\x1b[?2004h");
    CHECK(terminal.encodePaste(paste) == std::string("\x1b[200~") + paste + "\x1b[201~");

    terminal.feed("\x1b[5n");
    CHECK(terminal.drainReplies() == std::vector<std::string>{"\x1b[0n"});
    CHECK(terminal.drainReplies().empty());
    CHECK(terminal.encodePaste(paste) == std::string("\x1b[200~") + paste + "\x1b[201~");

    terminal.feed("\x1b[?2004l");
    CHECK(terminal.encodePaste(paste) == paste);
}

TEST_CASE("terminal: ConPTY PowerShell process uses semantic handshakes") {
#if defined(_WIN32)
    // PowerShell startup can exceed ten seconds on cold, heavily loaded CI
    // runners. Successful handshakes still return immediately.
    const int integration_timeout_ms = 30000;
    PtyProcessOptions options;
    options.columns = 40;
    options.rows = 10;
    const std::string probe_path =
        das::getDasRoot() + "/modules/dasTerminal/tests/conpty_probe.ps1";
    const std::string arguments =
        " -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass"
        " -File \"" + probe_path + "\"";
    options.command_line = "pwsh.exe" + arguments;

    std::string error;
    std::unique_ptr<PtyProcess> process = das_terminal::launchPtyProcess(options, error);
    if (!process) {
        INFO("pwsh.exe unavailable, exercising the same fixture with Windows PowerShell: " << error);
        error.clear();
        options.command_line = "powershell.exe" + arguments;
        process = das_terminal::launchPtyProcess(options, error);
    }
    INFO(error);
    REQUIRE(process);
    PtyProcessGuard guard{process.get()};
    CHECK(process->processId() != 0);

    Terminal terminal(options.columns, options.rows);
    REQUIRE(waitForTerminalText(
        *process, terminal, "DASHERD_READY:", integration_timeout_ms, error));
    INFO(error);

    REQUIRE(process->resize(100, 30, error));
    terminal.resize(100, 30);
    REQUIRE(process->write(std::string("semantic-probe") +
        terminal.encodeKey(KeyEvent(Key::enter)), error));
    REQUIRE(waitForTerminalText(
        *process, terminal, "DASHERD_ECHO:semantic-probe", integration_timeout_ms, error));
    INFO(error);

    uint32_t exit_code = 0;
    REQUIRE(process->wait(integration_timeout_ms, exit_code, error));
    INFO(error);
    CHECK(exit_code == 23);
#else
    MESSAGE("PTY transport is not implemented on this platform yet");
#endif
}
