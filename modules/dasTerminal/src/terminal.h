#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>
#include <vector>

namespace das_terminal {

enum CellAttribute : uint16_t {
    attr_none = 0,
    attr_bold = 1u << 0u,
    attr_faint = 1u << 1u,
    attr_italic = 1u << 2u,
    attr_underline = 1u << 3u,
    attr_blink = 1u << 4u,
    attr_inverse = 1u << 5u,
    attr_hidden = 1u << 6u,
    attr_strike = 1u << 7u,
};

enum class ColorKind : uint8_t {
    default_color,
    indexed,
    rgb,
};

struct Color {
    ColorKind kind = ColorKind::default_color;
    uint8_t index = 0;
    uint8_t red = 0;
    uint8_t green = 0;
    uint8_t blue = 0;

    bool operator==(const Color & other) const;
    bool operator!=(const Color & other) const { return !(*this == other); }
};

struct Cell {
    std::string grapheme = " ";
    uint8_t width = 1;
    uint16_t attributes = attr_none;
    Color foreground;
    Color background;
    std::string hyperlink;

    bool operator==(const Cell & other) const;
    bool operator!=(const Cell & other) const { return !(*this == other); }
};

struct Cursor {
    int column = 0;
    int row = 0;
    bool visible = true;
    bool blinking = true;
    int style = 0;

    bool operator==(const Cursor & other) const;
};

struct BufferSnapshot {
    std::vector<std::vector<Cell>> rows;
    std::vector<std::vector<Cell>> scrollback;
    Cursor cursor;

    bool operator==(const BufferSnapshot & other) const;
};

struct Modes {
    bool auto_wrap = true;
    bool application_cursor_keys = false;
    bool bracketed_paste = false;
    bool focus_reporting = false;
    bool mouse_button_reporting = false;
    bool mouse_any_reporting = false;
    bool sgr_mouse_encoding = false;

    bool operator==(const Modes & other) const;
};

struct Snapshot {
    int columns = 0;
    int rows = 0;
    bool alternate_active = false;
    BufferSnapshot normal;
    BufferSnapshot alternate;
    Modes modes;
    std::string title;
    std::string current_directory;
    std::vector<std::string> unknown_sequences;

    bool operator==(const Snapshot & other) const;
    bool operator!=(const Snapshot & other) const { return !(*this == other); }
};

enum class Key : uint8_t {
    text,
    enter,
    tab,
    backspace,
    escape,
    up,
    down,
    right,
    left,
    home,
    end,
    insert,
    delete_key,
    page_up,
    page_down,
    f1,
    f2,
    f3,
    f4,
    f5,
    f6,
    f7,
    f8,
    f9,
    f10,
    f11,
    f12,
};

struct KeyEvent {
    Key key;
    std::string text;
    bool shift;
    bool alt;
    bool control;

    explicit KeyEvent(Key code = Key::text, const std::string & value = std::string(),
                      bool shift_down = false, bool alt_down = false,
                      bool control_down = false)
        : key(code), text(value), shift(shift_down), alt(alt_down), control(control_down) {}
};

class Terminal {
public:
    using CellRows = std::deque<std::vector<Cell>>;

    Terminal(int columns, int rows);
    ~Terminal();
    Terminal(const Terminal &) = delete;
    Terminal & operator=(const Terminal &) = delete;

    void feed(const uint8_t * bytes, size_t count);
    void feed(const std::string & bytes) {
        feed(reinterpret_cast<const uint8_t *>(bytes.data()), bytes.size());
    }
    void resize(int columns, int rows);

    std::string encodeKey(const KeyEvent & event) const;
    std::string encodePaste(const std::string & text) const;

    int columns() const { return columns_; }
    int rows() const { return rows_; }
    bool alternateActive() const { return alternate_active_; }
    const Modes & modes() const { return modes_; }
    const std::string & title() const { return title_; }
    const std::string & currentDirectory() const { return current_directory_; }
    const std::vector<std::string> & unknownSequences() const { return unknown_sequences_; }
    uint64_t revision() const { return revision_; }
    const CellRows * cellRows(int screen, bool scrollback) const;
    const Cursor * cursor(int screen) const;

    Snapshot snapshot() const;
    std::vector<std::string> drainReplies();
    std::vector<std::string> takeEvents();

private:
    struct Buffer;
    enum class ParseState : uint8_t;

    Buffer & activeBuffer();
    const Buffer & activeBuffer() const;
    void reset();
    void resetBuffer(Buffer & buffer);
    void processByte(uint8_t byte);
    void processGroundByte(uint8_t byte);
    void processControl(uint8_t byte);
    void processEscape(uint8_t byte);
    void processCsi(uint8_t final_byte);
    void processOsc();
    void processCodepoint(uint32_t codepoint);
    void finishUtf8(bool valid);
    void putGrapheme(const std::string & grapheme, int width, bool combining);
    void carriageReturn();
    void lineFeed();
    void reverseIndex();
    void tab();
    void scrollUp(Buffer & buffer);
    void eraseDisplay(int mode);
    void eraseLine(int mode);
    void eraseCells(int count);
    void clearCell(Buffer & buffer, int row, int column);
    void setCursor(int row, int column);
    void setPrivateMode(int mode, bool enabled);
    void selectGraphicRendition(const std::vector<int> & parameters);
    void recordUnknown(const std::string & sequence);

    int columns_ = 0;
    int rows_ = 0;
    Buffer * normal_ = nullptr;
    Buffer * alternate_ = nullptr;
    bool alternate_active_ = false;
    Modes modes_;
    std::string title_;
    std::string current_directory_;
    std::vector<std::string> unknown_sequences_;
    std::vector<std::string> replies_;
    std::vector<std::string> events_;

    ParseState parse_state_;
    std::string sequence_;
    std::string csi_parameters_;
    std::string osc_data_;
    uint32_t utf8_codepoint_ = 0;
    uint32_t utf8_minimum_ = 0;
    int utf8_remaining_ = 0;
    uint64_t revision_ = 0;
};

} // namespace das_terminal
