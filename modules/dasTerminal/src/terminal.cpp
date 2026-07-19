#include "terminal.h"

#include <algorithm>
#include <cstdlib>
#include <sstream>

namespace das_terminal {
namespace {

Cell blankCell() { return Cell(); }

std::vector<Cell> blankRow(int columns) {
    return std::vector<Cell>(static_cast<size_t>(columns), blankCell());
}

std::string encodeUtf8(uint32_t value) {
    std::string result;
    if (value <= 0x7fu) {
        result.push_back(static_cast<char>(value));
    } else if (value <= 0x7ffu) {
        result.push_back(static_cast<char>(0xc0u | (value >> 6u)));
        result.push_back(static_cast<char>(0x80u | (value & 0x3fu)));
    } else if (value <= 0xffffu) {
        result.push_back(static_cast<char>(0xe0u | (value >> 12u)));
        result.push_back(static_cast<char>(0x80u | ((value >> 6u) & 0x3fu)));
        result.push_back(static_cast<char>(0x80u | (value & 0x3fu)));
    } else {
        result.push_back(static_cast<char>(0xf0u | (value >> 18u)));
        result.push_back(static_cast<char>(0x80u | ((value >> 12u) & 0x3fu)));
        result.push_back(static_cast<char>(0x80u | ((value >> 6u) & 0x3fu)));
        result.push_back(static_cast<char>(0x80u | (value & 0x3fu)));
    }
    return result;
}

bool inRange(uint32_t value, uint32_t first, uint32_t last) {
    return value >= first && value <= last;
}

bool isCombining(uint32_t value) {
    return inRange(value, 0x0300, 0x036f) || inRange(value, 0x0483, 0x0489) ||
        inRange(value, 0x0591, 0x05bd) || value == 0x05bf ||
        inRange(value, 0x05c1, 0x05c2) || inRange(value, 0x0610, 0x061a) ||
        inRange(value, 0x064b, 0x065f) || inRange(value, 0x1ab0, 0x1aff) ||
        inRange(value, 0x1dc0, 0x1dff) || inRange(value, 0x20d0, 0x20ff) ||
        inRange(value, 0xfe00, 0xfe0f) || inRange(value, 0xfe20, 0xfe2f) ||
        inRange(value, 0xe0100, 0xe01ef);
}

bool isWide(uint32_t value) {
    return inRange(value, 0x1100, 0x115f) || value == 0x2329 || value == 0x232a ||
        inRange(value, 0x2e80, 0xa4cf) || inRange(value, 0xac00, 0xd7a3) ||
        inRange(value, 0xf900, 0xfaff) || inRange(value, 0xfe10, 0xfe19) ||
        inRange(value, 0xfe30, 0xfe6f) || inRange(value, 0xff00, 0xff60) ||
        inRange(value, 0xffe0, 0xffe6) || inRange(value, 0x1f300, 0x1faff) ||
        inRange(value, 0x20000, 0x3fffd);
}

int parameterOr(const std::vector<int> & parameters, size_t index, int fallback) {
    if (index >= parameters.size() || parameters[index] <= 0) return fallback;
    return parameters[index];
}

std::vector<int> parseParameters(const std::string & text, char & private_marker) {
    std::vector<int> result;
    size_t begin = 0;
    private_marker = 0;
    if (!text.empty() && (text[0] == '?' || text[0] == '>' || text[0] == '!')) {
        private_marker = text[0];
        begin = 1;
    }
    size_t item = begin;
    while (item <= text.size()) {
        const size_t separator = text.find(';', item);
        const size_t end = separator == std::string::npos ? text.size() : separator;
        if (end == item) result.push_back(-1);
        else result.push_back(std::atoi(text.substr(item, end - item).c_str()));
        if (separator == std::string::npos) break;
        item = separator + 1;
    }
    if (result.empty()) result.push_back(-1);
    return result;
}

Color indexedColor(int index) {
    Color color;
    color.kind = ColorKind::indexed;
    color.index = static_cast<uint8_t>(std::max(0, std::min(255, index)));
    return color;
}

Color rgbColor(int red, int green, int blue) {
    Color color;
    color.kind = ColorKind::rgb;
    color.red = static_cast<uint8_t>(std::max(0, std::min(255, red)));
    color.green = static_cast<uint8_t>(std::max(0, std::min(255, green)));
    color.blue = static_cast<uint8_t>(std::max(0, std::min(255, blue)));
    return color;
}

int keyModifierParameter(const KeyEvent & event) {
    return 1 + (event.shift ? 1 : 0) + (event.alt ? 2 : 0) + (event.control ? 4 : 0);
}

std::string csiKey(char final_byte, int modifier) {
    std::ostringstream result;
    result << "\x1b[1;" << modifier << final_byte;
    return result.str();
}

std::string tildeKey(int code, int modifier) {
    std::ostringstream result;
    result << "\x1b[" << code;
    if (modifier != 1) result << ';' << modifier;
    result << '~';
    return result.str();
}

std::string altPrefix(const std::string & bytes, bool alt) {
    return alt && !bytes.empty() ? std::string("\x1b") + bytes : bytes;
}

std::string controlText(const std::string & text) {
    if (text.size() != 1) return text;
    const unsigned char value = static_cast<unsigned char>(text[0]);
    if (value >= 'a' && value <= 'z') return std::string(1, static_cast<char>(value - 'a' + 1));
    if (value >= 'A' && value <= 'Z') return std::string(1, static_cast<char>(value - 'A' + 1));
    switch (value) {
    case ' ':
    case '@': return std::string(1, '\0');
    case '[': return std::string(1, '\x1b');
    case '\\': return std::string(1, '\x1c');
    case ']': return std::string(1, '\x1d');
    case '^': return std::string(1, '\x1e');
    case '_': return std::string(1, '\x1f');
    case '?': return std::string(1, '\x7f');
    default: return text;
    }
}

} // namespace

bool Color::operator==(const Color & other) const {
    return kind == other.kind && index == other.index && red == other.red &&
        green == other.green && blue == other.blue;
}

bool Cell::operator==(const Cell & other) const {
    return grapheme == other.grapheme && width == other.width &&
        attributes == other.attributes && foreground == other.foreground &&
        background == other.background && hyperlink == other.hyperlink;
}

bool Cursor::operator==(const Cursor & other) const {
    return column == other.column && row == other.row && visible == other.visible &&
        blinking == other.blinking && style == other.style;
}

bool BufferSnapshot::operator==(const BufferSnapshot & other) const {
    return rows == other.rows && scrollback == other.scrollback && cursor == other.cursor;
}

bool Modes::operator==(const Modes & other) const {
    return auto_wrap == other.auto_wrap &&
        application_cursor_keys == other.application_cursor_keys &&
        bracketed_paste == other.bracketed_paste &&
        focus_reporting == other.focus_reporting &&
        mouse_button_reporting == other.mouse_button_reporting &&
        mouse_any_reporting == other.mouse_any_reporting &&
        sgr_mouse_encoding == other.sgr_mouse_encoding;
}

bool Snapshot::operator==(const Snapshot & other) const {
    return columns == other.columns && rows == other.rows &&
        alternate_active == other.alternate_active && normal == other.normal &&
        alternate == other.alternate && modes == other.modes && title == other.title &&
        current_directory == other.current_directory &&
        unknown_sequences == other.unknown_sequences;
}

struct Terminal::Buffer {
    std::vector<std::vector<Cell>> rows;
    std::vector<std::vector<Cell>> scrollback;
    Cursor cursor;
    Cursor saved_cursor;
    Cell pen;
    bool wrap_pending = false;
};

enum class Terminal::ParseState : uint8_t {
    ground,
    escape,
    csi,
    osc,
    osc_escape,
};

Terminal::Terminal(int columns, int rows)
    : columns_(std::max(1, columns)), rows_(std::max(1, rows)),
      normal_(new Buffer()), alternate_(new Buffer()), parse_state_(ParseState::ground) {
    resetBuffer(*normal_);
    resetBuffer(*alternate_);
}

Terminal::~Terminal() {
    delete normal_;
    delete alternate_;
}

Terminal::Buffer & Terminal::activeBuffer() {
    return alternate_active_ ? *alternate_ : *normal_;
}

const Terminal::Buffer & Terminal::activeBuffer() const {
    return alternate_active_ ? *alternate_ : *normal_;
}

void Terminal::resetBuffer(Buffer & buffer) {
    buffer.rows.assign(static_cast<size_t>(rows_), blankRow(columns_));
    buffer.scrollback.clear();
    buffer.cursor = Cursor();
    buffer.saved_cursor = Cursor();
    buffer.pen = Cell();
    buffer.wrap_pending = false;
}

void Terminal::reset() {
    resetBuffer(*normal_);
    resetBuffer(*alternate_);
    alternate_active_ = false;
    modes_ = Modes();
    title_.clear();
    current_directory_.clear();
    unknown_sequences_.clear();
}

void Terminal::feed(const uint8_t * bytes, size_t count) {
    if (!bytes || count == 0) return;
    ++revision_;
    for (size_t index = 0; index != count; ++index) processByte(bytes[index]);
}

void Terminal::processByte(uint8_t byte) {
    if (utf8_remaining_ > 0) {
        if ((byte & 0xc0u) == 0x80u) {
            utf8_codepoint_ = (utf8_codepoint_ << 6u) | (byte & 0x3fu);
            if (--utf8_remaining_ == 0) finishUtf8(true);
            return;
        }
        finishUtf8(false);
        processByte(byte);
        return;
    }

    switch (parse_state_) {
    case ParseState::ground: processGroundByte(byte); break;
    case ParseState::escape: processEscape(byte); break;
    case ParseState::csi:
        sequence_.push_back(static_cast<char>(byte));
        if (byte >= 0x40u && byte <= 0x7eu) processCsi(byte);
        else if (byte >= 0x20u && byte <= 0x3fu) csi_parameters_.push_back(static_cast<char>(byte));
        else if (byte == 0x1bu) parse_state_ = ParseState::escape;
        else recordUnknown(sequence_);
        break;
    case ParseState::osc:
        if (byte == 0x07u) processOsc();
        else if (byte == 0x1bu) parse_state_ = ParseState::osc_escape;
        else osc_data_.push_back(static_cast<char>(byte));
        break;
    case ParseState::osc_escape:
        if (byte == '\\') processOsc();
        else {
            osc_data_.push_back('\x1b');
            osc_data_.push_back(static_cast<char>(byte));
            parse_state_ = ParseState::osc;
        }
        break;
    }
}

void Terminal::processGroundByte(uint8_t byte) {
    if (byte == 0x1bu) {
        parse_state_ = ParseState::escape;
        sequence_ = "\x1b";
    } else if (byte < 0x20u || byte == 0x7fu) {
        processControl(byte);
    } else if (byte < 0x80u) {
        processCodepoint(byte);
    } else {
        if ((byte & 0xe0u) == 0xc0u) {
            utf8_codepoint_ = byte & 0x1fu;
            utf8_minimum_ = 0x80u;
            utf8_remaining_ = 1;
        } else if ((byte & 0xf0u) == 0xe0u) {
            utf8_codepoint_ = byte & 0x0fu;
            utf8_minimum_ = 0x800u;
            utf8_remaining_ = 2;
        } else if ((byte & 0xf8u) == 0xf0u) {
            utf8_codepoint_ = byte & 0x07u;
            utf8_minimum_ = 0x10000u;
            utf8_remaining_ = 3;
        } else {
            processCodepoint(0xfffdu);
        }
    }
}

void Terminal::finishUtf8(bool valid) {
    const uint32_t codepoint = utf8_codepoint_;
    valid = valid && codepoint >= utf8_minimum_ && codepoint <= 0x10ffffu &&
        !inRange(codepoint, 0xd800u, 0xdfffu);
    utf8_codepoint_ = 0;
    utf8_minimum_ = 0;
    utf8_remaining_ = 0;
    processCodepoint(valid ? codepoint : 0xfffdu);
}

void Terminal::processControl(uint8_t byte) {
    switch (byte) {
    case 0x07: events_.push_back("bell"); break;
    case 0x08:
        activeBuffer().cursor.column = std::max(0, activeBuffer().cursor.column - 1);
        activeBuffer().wrap_pending = false;
        break;
    case 0x09: tab(); break;
    case 0x0a:
    case 0x0b:
    case 0x0c: lineFeed(); break;
    case 0x0d: carriageReturn(); break;
    default: break;
    }
}

void Terminal::processEscape(uint8_t byte) {
    sequence_.push_back(static_cast<char>(byte));
    parse_state_ = ParseState::ground;
    switch (byte) {
    case '[':
        parse_state_ = ParseState::csi;
        csi_parameters_.clear();
        break;
    case ']':
        parse_state_ = ParseState::osc;
        osc_data_.clear();
        break;
    case '7': activeBuffer().saved_cursor = activeBuffer().cursor; break;
    case '8': activeBuffer().cursor = activeBuffer().saved_cursor; break;
    case 'D': lineFeed(); break;
    case 'E': carriageReturn(); lineFeed(); break;
    case 'M': reverseIndex(); break;
    case 'c': reset(); break;
    case '\\': break;
    default: recordUnknown(sequence_); break;
    }
}

void Terminal::processCsi(uint8_t final_byte) {
    parse_state_ = ParseState::ground;
    char private_marker = 0;
    const std::vector<int> parameters = parseParameters(csi_parameters_, private_marker);
    Buffer & buffer = activeBuffer();
    const int first = parameterOr(parameters, 0, 1);

    if (private_marker == '?' && (final_byte == 'h' || final_byte == 'l')) {
        for (size_t index = 0; index != parameters.size(); ++index)
            setPrivateMode(parameters[index], final_byte == 'h');
        return;
    }
    switch (final_byte) {
    case 'A': buffer.cursor.row = std::max(0, buffer.cursor.row - first); break;
    case 'B': buffer.cursor.row = std::min(rows_ - 1, buffer.cursor.row + first); break;
    case 'C': buffer.cursor.column = std::min(columns_ - 1, buffer.cursor.column + first); break;
    case 'D': buffer.cursor.column = std::max(0, buffer.cursor.column - first); break;
    case 'E': setCursor(buffer.cursor.row + first, 0); break;
    case 'F': setCursor(buffer.cursor.row - first, 0); break;
    case 'G': setCursor(buffer.cursor.row, first - 1); break;
    case 'H':
    case 'f': setCursor(parameterOr(parameters, 0, 1) - 1, parameterOr(parameters, 1, 1) - 1); break;
    case 'J': eraseDisplay(parameters[0] < 0 ? 0 : parameters[0]); break;
    case 'K': eraseLine(parameters[0] < 0 ? 0 : parameters[0]); break;
    case 'X': eraseCells(first); break;
    case 'a': buffer.cursor.column = std::min(columns_ - 1, buffer.cursor.column + first); break;
    case 'd': setCursor(first - 1, buffer.cursor.column); break;
    case 'e': buffer.cursor.row = std::min(rows_ - 1, buffer.cursor.row + first); break;
    case 'm': selectGraphicRendition(parameters); break;
    case 'n':
        if (private_marker == 0 && parameters[0] == 5) replies_.push_back("\x1b[0n");
        else if (private_marker == 0 && parameters[0] == 6) {
            std::ostringstream reply;
            reply << "\x1b[" << buffer.cursor.row + 1 << ';' << buffer.cursor.column + 1 << 'R';
            replies_.push_back(reply.str());
        } else recordUnknown(sequence_);
        break;
    case 'c':
        if (private_marker == 0) replies_.push_back("\x1b[?1;2c");
        else recordUnknown(sequence_);
        break;
    case 's': buffer.saved_cursor = buffer.cursor; break;
    case 'u': buffer.cursor = buffer.saved_cursor; break;
    case 'q':
        if (private_marker == 0) {
            buffer.cursor.style = parameters[0] < 0 ? 0 : parameters[0];
            buffer.cursor.blinking = buffer.cursor.style == 0 || buffer.cursor.style == 1 ||
                buffer.cursor.style == 3 || buffer.cursor.style == 5;
        } else recordUnknown(sequence_);
        break;
    default: recordUnknown(sequence_); break;
    }
    buffer.wrap_pending = false;
}

void Terminal::processOsc() {
    parse_state_ = ParseState::ground;
    const size_t separator = osc_data_.find(';');
    const std::string command = separator == std::string::npos ? osc_data_ : osc_data_.substr(0, separator);
    const std::string value = separator == std::string::npos ? std::string() : osc_data_.substr(separator + 1);
    if (command == "0" || command == "2") {
        title_ = value;
        events_.push_back("title:" + value);
    } else if (command == "7") {
        current_directory_ = value;
        events_.push_back("cwd:" + value);
    } else if (command == "8") {
        const size_t uri_separator = value.find(';');
        activeBuffer().pen.hyperlink = uri_separator == std::string::npos
            ? std::string() : value.substr(uri_separator + 1);
    } else {
        recordUnknown("OSC " + osc_data_);
    }
    osc_data_.clear();
}

void Terminal::processCodepoint(uint32_t codepoint) {
    const bool combining = isCombining(codepoint);
    putGrapheme(encodeUtf8(codepoint), combining ? 0 : (isWide(codepoint) ? 2 : 1), combining);
}

void Terminal::putGrapheme(const std::string & grapheme, int width, bool combining) {
    Buffer & buffer = activeBuffer();
    if (combining) {
        int row = buffer.cursor.row;
        int column = buffer.cursor.column - 1;
        if (buffer.wrap_pending) column = columns_ - 1;
        if (column < 0 && row > 0) { --row; column = columns_ - 1; }
        while (column > 0 && buffer.rows[static_cast<size_t>(row)][static_cast<size_t>(column)].width == 0) --column;
        if (column >= 0) buffer.rows[static_cast<size_t>(row)][static_cast<size_t>(column)].grapheme += grapheme;
        return;
    }
    if (buffer.wrap_pending && modes_.auto_wrap) {
        carriageReturn();
        lineFeed();
    }
    if (width == 2 && buffer.cursor.column == columns_ - 1 && modes_.auto_wrap) {
        carriageReturn();
        lineFeed();
    }
    const int row = buffer.cursor.row;
    const int column = buffer.cursor.column;
    clearCell(buffer, row, column);
    if (width == 2 && column + 1 < columns_) clearCell(buffer, row, column + 1);
    Cell cell = buffer.pen;
    cell.grapheme = grapheme;
    cell.width = static_cast<uint8_t>(width);
    buffer.rows[static_cast<size_t>(row)][static_cast<size_t>(column)] = cell;
    if (width == 2 && column + 1 < columns_) {
        Cell continuation = buffer.pen;
        continuation.grapheme.clear();
        continuation.width = 0;
        buffer.rows[static_cast<size_t>(row)][static_cast<size_t>(column + 1)] = continuation;
    }
    const int next = column + width;
    if (next >= columns_) {
        buffer.cursor.column = columns_ - 1;
        buffer.wrap_pending = modes_.auto_wrap;
    } else {
        buffer.cursor.column = next;
        buffer.wrap_pending = false;
    }
}

void Terminal::carriageReturn() {
    activeBuffer().cursor.column = 0;
    activeBuffer().wrap_pending = false;
}

void Terminal::lineFeed() {
    Buffer & buffer = activeBuffer();
    buffer.wrap_pending = false;
    if (buffer.cursor.row == rows_ - 1) scrollUp(buffer);
    else ++buffer.cursor.row;
}

void Terminal::reverseIndex() {
    Buffer & buffer = activeBuffer();
    buffer.wrap_pending = false;
    if (buffer.cursor.row > 0) --buffer.cursor.row;
    else {
        buffer.rows.insert(buffer.rows.begin(), blankRow(columns_));
        buffer.rows.pop_back();
    }
}

void Terminal::tab() {
    Buffer & buffer = activeBuffer();
    buffer.cursor.column = std::min(columns_ - 1, ((buffer.cursor.column / 8) + 1) * 8);
    buffer.wrap_pending = false;
}

void Terminal::scrollUp(Buffer & buffer) {
    if (&buffer == normal_) {
        buffer.scrollback.push_back(buffer.rows.front());
        if (buffer.scrollback.size() > 10000) buffer.scrollback.erase(buffer.scrollback.begin());
    }
    buffer.rows.erase(buffer.rows.begin());
    buffer.rows.push_back(blankRow(columns_));
}

void Terminal::eraseDisplay(int mode) {
    Buffer & buffer = activeBuffer();
    if (mode == 2 || mode == 3) {
        for (int row = 0; row != rows_; ++row) buffer.rows[static_cast<size_t>(row)] = blankRow(columns_);
        if (mode == 3) buffer.scrollback.clear();
    } else if (mode == 0) {
        eraseLine(0);
        for (int row = buffer.cursor.row + 1; row != rows_; ++row) buffer.rows[static_cast<size_t>(row)] = blankRow(columns_);
    } else if (mode == 1) {
        eraseLine(1);
        for (int row = 0; row < buffer.cursor.row; ++row) buffer.rows[static_cast<size_t>(row)] = blankRow(columns_);
    }
}

void Terminal::eraseLine(int mode) {
    Buffer & buffer = activeBuffer();
    int first = mode == 0 ? buffer.cursor.column : 0;
    int last = mode == 1 ? buffer.cursor.column : columns_ - 1;
    if (mode == 2) { first = 0; last = columns_ - 1; }
    for (int column = first; column <= last; ++column)
        clearCell(buffer, buffer.cursor.row, column);
}

void Terminal::eraseCells(int count) {
    Buffer & buffer = activeBuffer();
    const int last = std::min(columns_, buffer.cursor.column + std::max(1, count));
    for (int column = buffer.cursor.column; column < last; ++column)
        clearCell(buffer, buffer.cursor.row, column);
}

void Terminal::clearCell(Buffer & buffer, int row, int column) {
    if (row < 0 || row >= rows_ || column < 0 || column >= columns_) return;
    std::vector<Cell> & cells = buffer.rows[static_cast<size_t>(row)];
    int first = column;
    while (first > 0 && cells[static_cast<size_t>(first)].width == 0) --first;
    if (cells[static_cast<size_t>(first)].width == 2 && first + 1 >= column) {
        cells[static_cast<size_t>(first)] = blankCell();
        if (first + 1 < columns_) cells[static_cast<size_t>(first + 1)] = blankCell();
    } else {
        cells[static_cast<size_t>(column)] = blankCell();
    }
}

void Terminal::setCursor(int row, int column) {
    Buffer & buffer = activeBuffer();
    buffer.cursor.row = std::max(0, std::min(rows_ - 1, row));
    buffer.cursor.column = std::max(0, std::min(columns_ - 1, column));
    buffer.wrap_pending = false;
}

void Terminal::setPrivateMode(int mode, bool enabled) {
    switch (mode) {
    case 1: modes_.application_cursor_keys = enabled; break;
    case 7: modes_.auto_wrap = enabled; break;
    case 25:
        normal_->cursor.visible = enabled;
        alternate_->cursor.visible = enabled;
        break;
    case 1000: modes_.mouse_button_reporting = enabled; break;
    case 1003: modes_.mouse_any_reporting = enabled; break;
    case 1004: modes_.focus_reporting = enabled; break;
    case 1006: modes_.sgr_mouse_encoding = enabled; break;
    case 1049:
        if (enabled && !alternate_active_) {
            normal_->saved_cursor = normal_->cursor;
            resetBuffer(*alternate_);
            alternate_->cursor.visible = normal_->cursor.visible;
            alternate_->cursor.style = normal_->cursor.style;
            alternate_->cursor.blinking = normal_->cursor.blinking;
            alternate_active_ = true;
        } else if (!enabled && alternate_active_) {
            alternate_active_ = false;
            normal_->cursor = normal_->saved_cursor;
        }
        break;
    case 2004: modes_.bracketed_paste = enabled; break;
    default: recordUnknown(std::string("DEC mode ") + (enabled ? "set " : "reset ") + std::to_string(mode)); break;
    }
}

void Terminal::selectGraphicRendition(const std::vector<int> & parameters) {
    Cell & pen = activeBuffer().pen;
    for (size_t index = 0; index < parameters.size(); ++index) {
        const int value = parameters[index] < 0 ? 0 : parameters[index];
        switch (value) {
        case 0: pen = Cell(); break;
        case 1: pen.attributes |= attr_bold; break;
        case 2: pen.attributes |= attr_faint; break;
        case 3: pen.attributes |= attr_italic; break;
        case 4: pen.attributes |= attr_underline; break;
        case 5: pen.attributes |= attr_blink; break;
        case 6: pen.attributes |= attr_blink; break;
        case 7: pen.attributes |= attr_inverse; break;
        case 8: pen.attributes |= attr_hidden; break;
        case 9: pen.attributes |= attr_strike; break;
        case 21: pen.attributes |= attr_underline; break;
        case 22: pen.attributes &= static_cast<uint16_t>(~(attr_bold | attr_faint)); break;
        case 23: pen.attributes &= static_cast<uint16_t>(~attr_italic); break;
        case 24: pen.attributes &= static_cast<uint16_t>(~attr_underline); break;
        case 25: pen.attributes &= static_cast<uint16_t>(~attr_blink); break;
        case 27: pen.attributes &= static_cast<uint16_t>(~attr_inverse); break;
        case 28: pen.attributes &= static_cast<uint16_t>(~attr_hidden); break;
        case 29: pen.attributes &= static_cast<uint16_t>(~attr_strike); break;
        case 39: pen.foreground = Color(); break;
        case 49: pen.background = Color(); break;
        default:
            if (value >= 30 && value <= 37) pen.foreground = indexedColor(value - 30);
            else if (value >= 40 && value <= 47) pen.background = indexedColor(value - 40);
            else if (value >= 90 && value <= 97) pen.foreground = indexedColor(value - 90 + 8);
            else if (value >= 100 && value <= 107) pen.background = indexedColor(value - 100 + 8);
            else if ((value == 38 || value == 48) && index + 2 < parameters.size() && parameters[index + 1] == 5) {
                (value == 38 ? pen.foreground : pen.background) = indexedColor(parameters[index + 2]);
                index += 2;
            } else if ((value == 38 || value == 48) && index + 4 < parameters.size() && parameters[index + 1] == 2) {
                (value == 38 ? pen.foreground : pen.background) = rgbColor(parameters[index + 2], parameters[index + 3], parameters[index + 4]);
                index += 4;
            }
            break;
        }
    }
}

void Terminal::recordUnknown(const std::string & sequence) {
    if (unknown_sequences_.size() < 256) unknown_sequences_.push_back(sequence);
    parse_state_ = ParseState::ground;
}

void Terminal::resize(int columns, int rows) {
    columns = std::max(1, columns);
    rows = std::max(1, rows);
    if (columns == columns_ && rows == rows_) return;
    ++revision_;
    columns_ = columns;
    rows_ = rows;
    Buffer * buffers[] = {normal_, alternate_};
    for (size_t item = 0; item != 2; ++item) {
        Buffer & buffer = *buffers[item];
        buffer.rows.resize(static_cast<size_t>(rows_), blankRow(columns_));
        for (size_t row = 0; row != buffer.rows.size(); ++row)
            buffer.rows[row].resize(static_cast<size_t>(columns_), blankCell());
        buffer.cursor.row = std::min(buffer.cursor.row, rows_ - 1);
        buffer.cursor.column = std::min(buffer.cursor.column, columns_ - 1);
        buffer.saved_cursor.row = std::min(buffer.saved_cursor.row, rows_ - 1);
        buffer.saved_cursor.column = std::min(buffer.saved_cursor.column, columns_ - 1);
        buffer.wrap_pending = false;
    }
}

std::string Terminal::encodeKey(const KeyEvent & event) const {
    const int modifier = keyModifierParameter(event);
    switch (event.key) {
    case Key::text:
        return altPrefix(event.control ? controlText(event.text) : event.text, event.alt);
    case Key::enter: return altPrefix("\r", event.alt);
    case Key::tab:
        if (event.shift) return event.alt
            ? std::string("\x1b\x1b[Z") : std::string("\x1b[Z");
        return altPrefix("\t", event.alt);
    case Key::backspace: return altPrefix("\x7f", event.alt);
    case Key::escape: return altPrefix("\x1b", event.alt);
    case Key::up:
    case Key::down:
    case Key::right:
    case Key::left:
    case Key::home:
    case Key::end: {
        const char finals[] = {'A', 'B', 'C', 'D', 'H', 'F'};
        const size_t index = static_cast<size_t>(event.key) - static_cast<size_t>(Key::up);
        const char final_byte = finals[index];
        if (modifier != 1) return csiKey(final_byte, modifier);
        return std::string(modes_.application_cursor_keys ? "\x1bO" : "\x1b[") + final_byte;
    }
    case Key::insert: return tildeKey(2, modifier);
    case Key::delete_key: return tildeKey(3, modifier);
    case Key::page_up: return tildeKey(5, modifier);
    case Key::page_down: return tildeKey(6, modifier);
    case Key::f1:
    case Key::f2:
    case Key::f3:
    case Key::f4: {
        const char final_byte = static_cast<char>('P' +
            static_cast<int>(event.key) - static_cast<int>(Key::f1));
        if (modifier != 1) return csiKey(final_byte, modifier);
        return std::string("\x1bO") + final_byte;
    }
    case Key::f5: return tildeKey(15, modifier);
    case Key::f6: return tildeKey(17, modifier);
    case Key::f7: return tildeKey(18, modifier);
    case Key::f8: return tildeKey(19, modifier);
    case Key::f9: return tildeKey(20, modifier);
    case Key::f10: return tildeKey(21, modifier);
    case Key::f11: return tildeKey(23, modifier);
    case Key::f12: return tildeKey(24, modifier);
    }
    return std::string();
}

std::string Terminal::encodePaste(const std::string & text) const {
    if (!modes_.bracketed_paste) return text;
    return std::string("\x1b[200~") + text + "\x1b[201~";
}

const Terminal::CellRows * Terminal::cellRows(int screen, bool scrollback) const {
    const Buffer * buffer = screen == 0 ? normal_ : screen == 1 ? alternate_ : nullptr;
    if (!buffer) return nullptr;
    return scrollback ? &buffer->scrollback : &buffer->rows;
}

const Cursor * Terminal::cursor(int screen) const {
    const Buffer * buffer = screen == 0 ? normal_ : screen == 1 ? alternate_ : nullptr;
    return buffer ? &buffer->cursor : nullptr;
}

Snapshot Terminal::snapshot() const {
    Snapshot result;
    result.columns = columns_;
    result.rows = rows_;
    result.alternate_active = alternate_active_;
    result.normal.rows = normal_->rows;
    result.normal.scrollback = normal_->scrollback;
    result.normal.cursor = normal_->cursor;
    result.alternate.rows = alternate_->rows;
    result.alternate.scrollback = alternate_->scrollback;
    result.alternate.cursor = alternate_->cursor;
    result.modes = modes_;
    result.title = title_;
    result.current_directory = current_directory_;
    result.unknown_sequences = unknown_sequences_;
    return result;
}

std::vector<std::string> Terminal::drainReplies() {
    std::vector<std::string> result;
    result.swap(replies_);
    return result;
}

std::vector<std::string> Terminal::takeEvents() {
    std::vector<std::string> result;
    result.swap(events_);
    return result;
}

} // namespace das_terminal
