options gen2
options gc
options persistent_heap
options indenting = 4

require imgui/imgui_harness
require imgui/imgui_file_dialog
require imgui/imgui_fonts
require imgui/imgui_icons
require imgui/imgui_markdown_view
require imgui/imgui_text_source_view
require imgui/imgui_theme_daslang
require daslib/clargs
require daslib/fio
require daslib/json
require daslib/json_boost
require daslib/safe_addr
require daslib/stringify
require daslib/utf8_utils
require live/live_commands
require math
require strings

//! Small user-facing Unicode text and Markdown viewer.
//!
//! Interpreted:
//!   daslang examples/text/main.das -- README.md
//!   daslang examples/text/main.das -- --plain LICENSE
//!
//! A standalone executable accepts the same path without the `--` separator.

enum TextViewMode {
    markdown
    plain_text
}

struct TextViewerOpenArgs {
    path : string
}

struct TextViewerModeArgs {
    mode : string
}

struct TextViewerZoomArgs {
    percent : int
}

struct TextViewerSourceRangeArgs {
    text : string
}

var private g_fonts = FontRoleSet()
var private g_font_sources : array<string>
var private g_document = MdDocument()
var private g_markdown_view = MarkdownViewState()
var private g_source_document = TextSourceDocument()
var private g_source_view = TextSourceViewState()
var private g_source_result = TextSourceViewResult()
var private g_path : string
var private g_status = "Open a UTF-8 text or Markdown file."
var private g_mode = TextViewMode.markdown
var private g_mode_override = -1
var private g_document_revision = 0
var private g_zoom_percent = 100
var private g_open_clicks = 0
var private g_reload_clicks = 0
var private g_quit_clicks = 0
var private g_markdown_clicks = 0
var private g_plain_clicks = 0

def private mode_name() : string {
    return g_mode == TextViewMode.markdown ? "markdown" : "plain"
}

def private markdown_path(path : string) : bool {
    let lower = to_lower(path)
    return ends_with(lower, ".md") || ends_with(lower, ".markdown")
}

def private set_mode(mode : TextViewMode) {
    if (g_mode == mode) return
    g_mode = mode
    markdown_view_clear_selection(g_markdown_view)
    text_source_view_clear_selection(g_source_view)
}

def private release_document() {
    markdown_view_invalidate(g_markdown_view)
    delete g_document.nodes
    g_document = MdDocument()
    text_source_view_invalidate(g_source_view)
    text_source_document_dispose(g_source_document)
}

def private install_source(path, source : string) {
    let next_revision = g_document_revision + 1
    var next_document <- md_parse(source)
    var next_source_document <- markdown_source_document(source, next_revision)
    release_document()
    g_document <- next_document
    g_source_document <- next_source_document
    g_path := path
    g_document_revision = next_revision

    if (g_mode_override == int(TextViewMode.markdown)) {
        g_mode = TextViewMode.markdown
    } elif (g_mode_override == int(TextViewMode.plain_text)) {
        g_mode = TextViewMode.plain_text
    } else {
        g_mode = markdown_path(path) ? TextViewMode.markdown : TextViewMode.plain_text
    }
    g_status = (g_document.ok
        ? "Loaded {length(g_source_document.source)} UTF-8 bytes."
        : g_document.error)
}

def private load_path(path : string) : bool {
    if (empty(path)) {
        g_status = "No file path was provided."
        return false
    }
    if (!fexist(path)) {
        g_status = "File does not exist: {path}"
        return false
    }
    let file_info = stat(path)
    if (!file_info.is_valid || file_info.is_dir) {
        g_status = "Path is not a regular file: {path}"
        return false
    }
    var source = fread(path)
    if (!is_utf8_string_valid(source)) {
        g_status = "File is not valid UTF-8: {path}"
        return false
    }
    if (length(source) >= 3 && contains_utf8_bom(source)) {
        source = slice(source, 3)
    }
    install_source(path, source)
    return true
}

def private open_file_dialog() {
    let start_dir = empty(g_path) ? "" : dir_name(g_path)
    file_dialog_open(FileDialogConfig(
        title = "Open text or Markdown",
        start_dir = start_dir,
        filters <- file_filters(
            "Text and Markdown|md;markdown;txt;rst;das|Markdown|md;markdown|All files|"),
        mode = FileDialogMode.open,
        max_selection = 1))
}

def private setup_default_layout(dock_id : uint) {
    DockBuilderRemoveNode(dock_id)
    DockBuilderAddDockSpaceNode(dock_id, ImGuiDockNodeFlags.None)
    let viewport = GetMainViewport()
    DockBuilderSetNodeSize(dock_id, viewport.Size)
    DockBuilderDockWindow("Document", dock_id)
    DockBuilderFinish(dock_id)
}

def private parse_command_line() {
    let args <- get_user_args()
    for (arg in args) {
        if (arg == "--plain") {
            g_mode_override = int(TextViewMode.plain_text)
        } elif (arg == "--markdown") {
            g_mode_override = int(TextViewMode.markdown)
        } elif (!starts_with(arg, "--") && empty(g_path)) {
            g_path := arg
        }
    }
}

def private install_welcome_document() {
    let welcome = %stringify~# Text viewer

Open a UTF-8 file from **File / Open...**, or pass its path on the command line.

- `.md` and `.markdown` files open as rendered Markdown.
- Other files open as regular text.
- Switch modes at any time with **View** or the buttons below.
- Use the bottom-right slider or Ctrl+mouse-wheel to zoom.
%%
    install_source("", welcome)
    g_mode = TextViewMode.markdown
    g_status = "No file selected."
}

def private draw_menu_bar() {
    main_menu_bar(MAIN_MENU) {
        menu(FILE_MENU, (text = "File", enabled = true)) {
            menu_label(FILE_OPEN, (text = "Open...", shortcut = "Ctrl+O"))
            menu_label(FILE_RELOAD, (text = "Reload", shortcut = "Ctrl+R",
                                     enabled = !empty(g_path)))
            separator(FILE_SEPARATOR)
            menu_label(FILE_QUIT, (text = "Quit", shortcut = "Ctrl+Q"))
        }
        menu(VIEW_MENU, (text = "View", enabled = true)) {
            menu_label(VIEW_MARKDOWN, (text = "Markdown", shortcut = "",
                                      selected = g_mode == TextViewMode.markdown))
            menu_label(VIEW_PLAIN, (text = "Plain text", shortcut = "",
                                   selected = g_mode == TextViewMode.plain_text))
        }
        if (!empty(g_path)) {
            separator(MENU_PATH_SEPARATOR)
            text_disabled(MENU_PATH, (text = g_path))
        }
    }

    if (FILE_OPEN.click_count > g_open_clicks) {
        g_open_clicks = FILE_OPEN.click_count
        open_file_dialog()
    }
    if (FILE_RELOAD.click_count > g_reload_clicks) {
        g_reload_clicks = FILE_RELOAD.click_count
        let _ = load_path(g_path)
    }
    if (FILE_QUIT.click_count > g_quit_clicks) {
        g_quit_clicks = FILE_QUIT.click_count
        request_exit()
    }
    if (VIEW_MARKDOWN.click_count > g_markdown_clicks) {
        g_markdown_clicks = VIEW_MARKDOWN.click_count
        set_mode(TextViewMode.markdown)
    }
    if (VIEW_PLAIN.click_count > g_plain_clicks) {
        g_plain_clicks = VIEW_PLAIN.click_count
        set_mode(TextViewMode.plain_text)
    }
}

def private draw_document() {
    let footer_height = GetFrameHeightWithSpacing()
    child(DOCUMENT_CONTENT,
          (text = "document content",
           size = float2(0.0f, -footer_height),
           child_flags = ImGuiChildFlags.None,
           window_flags = ImGuiWindowFlags.HorizontalScrollbar)) {
        if (g_mode == TextViewMode.markdown) {
            var style = MarkdownViewStyle(
                fonts = g_fonts,
                zoom = float(g_zoom_percent) / 100.0f,
                write_system_clipboard = !harness_is_headless())
            let _ = markdown_view("text viewer", g_document, g_markdown_view,
                                  style, g_document_revision)
        } else {
            var source_style = TextSourceViewStyle(
                fonts = g_fonts,
                zoom = float(g_zoom_percent) / 100.0f,
                wrap = false,
                write_system_clipboard = !harness_is_headless())
            g_source_result <- text_source_view("text source", g_source_document,
                                                g_source_view, source_style)
        }
    }

    let rendered = g_mode == TextViewMode.markdown
    if (icon_button(MODE_TOGGLE,
            (glyph = rendered ? "edit" : "visible",
             tip = rendered ? "Show Markdown source" : "Render Markdown",
             size = 18.0f, weight = 1.0f))) {
        set_mode(rendered ? TextViewMode.plain_text : TextViewMode.markdown)
    }
    same_line()
    text_disabled(MODE_LABEL,
        (text = rendered ? "Rendered Markdown" : "Markdown source"))
    same_line()
    text_disabled(STATUS, (text = g_status))

    let right_edge = GetCursorPosX() + GetContentRegionAvail().x
    let zoom_width = 245.0f
    same_line()
    SetCursorPosX(max(GetCursorPosX(), right_edge - zoom_width))
    text(ZOOM_LABEL, (text = "Zoom"))
    same_line()
    SetNextItemWidth(150.0f)
    if (edit_slider_int(safe_addr(g_zoom_percent),
            (id = "TEXT_VIEWER_ZOOM", text = "##zoom", v_min = 50,
             v_max = 300, format = "%d%%"))) {
        g_zoom_percent = clamp(((g_zoom_percent + 2) / 5) * 5, 50, 300)
    }
    same_line()
    if (small_button(ZOOM_RESET, (text = "100%"))) {
        g_zoom_percent = 100
    }
}

def private handle_shortcuts_and_zoom() {
    let io & = unsafe(GetIO())
    let command_down = io.ConfigMacOSXBehaviors ? io.KeySuper : io.KeyCtrl
    if (command_down && IsKeyPressed(ImGuiKey.O, false)) {
        open_file_dialog()
    }
    if (command_down && IsKeyPressed(ImGuiKey.R, false) && !empty(g_path)) {
        let _ = load_path(g_path)
    }
    if (command_down && IsKeyPressed(ImGuiKey.Q, false)) {
        request_exit()
    }
    if (IsWindowHovered(ImGuiHoveredFlags.RootAndChildWindows) && command_down) {
        if (io.MouseWheel > 0.0f) {
            g_zoom_percent = min(g_zoom_percent + 5, 300)
        } elif (io.MouseWheel < 0.0f) {
            g_zoom_percent = max(g_zoom_percent - 5, 50)
        }
    }
}

[live_command(description = "Inspect the text viewer without a screenshot.")]
def text_viewer_state(_input : JsonValue?) : JsonValue? {
    return JV((
        ok = g_document.ok,
        path = g_path,
        byte_count = length(g_source_document.source),
        mode = mode_name(),
        zoom_percent = g_zoom_percent,
        document_revision = g_document_revision,
        render_revision = g_markdown_view.render_revision,
        source_render_revision = g_source_view.render_revision,
        source_layout_hits = g_source_view.layout_hits,
        source_layout_misses = g_source_view.layout_misses,
        source_marker_count = g_source_document.marker_count,
        source_cell_count = length(g_source_view.cells),
        source_selection_revision = g_source_view.selection_revision,
        source_selection_commit_revision = g_source_view.selection_commit_revision,
        source_selection_dragging = g_source_view.selection_dragging,
        source_selection_start = g_source_result.selection.source.start_byte,
        source_selection_end = g_source_result.selection.source.end_byte,
        source_selection_has_markers = g_source_result.selection.styles.has_markers,
        status = g_status))
}

[live_command(description = "Find source text and inspect its exact cell-aligned screen range.")]
def text_viewer_source_range(input : JsonValue?) : JsonValue? {
    let args = from_JV(input, type<TextViewerSourceRangeArgs>)
    let start_byte = find(g_source_document.source, args.text)
    if (start_byte < 0) {
        return JV((ok = false, error = "source text not found", text = args.text))
    }
    let end_byte = start_byte + length(args.text)
    var found = false
    var all_markers = true
    var columns = 0
    var matched_cells = 0
    var marker_cells = 0
    var first_cell_start = -1
    var last_cell_end = -1
    var rect = float4()
    for (cell in g_source_view.cells) {
        if (cell.source.end_byte <= start_byte || cell.source.start_byte >= end_byte) continue
        let screen = float4(
            g_source_view.screen_origin.x + cell.rect.x,
            g_source_view.screen_origin.y + cell.rect.y,
            g_source_view.screen_origin.x + cell.rect.z,
            g_source_view.screen_origin.y + cell.rect.w)
        if (!found) {
            rect = screen
            found = true
        } else {
            rect.x = min(rect.x, screen.x)
            rect.y = min(rect.y, screen.y)
            rect.z = max(rect.z, screen.z)
            rect.w = max(rect.w, screen.w)
        }
        columns += cell.columns
        matched_cells++
        marker_cells += cell.style_id == 1 ? 1 : 0
        if (first_cell_start < 0) {
            first_cell_start = cell.source.start_byte
        }
        last_cell_end = cell.source.end_byte
        all_markers = all_markers && cell.style_id == 1
    }
    return JV((ok = found, text = args.text,
        source_start = start_byte, source_end = end_byte,
        columns = columns, matched_cells = matched_cells,
        marker_cells = marker_cells, first_cell_start = first_cell_start,
        last_cell_end = last_cell_end,
        all_markers = found && all_markers,
        cell_width = g_source_view.key.cell_width,
        x0 = rect.x, y0 = rect.y, x1 = rect.z, y1 = rect.w))
}

[live_command(description = "Open a UTF-8 file in the text viewer.")]
def text_viewer_open(input : JsonValue?) : JsonValue? {
    let args = from_JV(input, type<TextViewerOpenArgs>)
    let ok = load_path(args.path)
    return JV((ok = ok, path = g_path, status = g_status,
               document_revision = g_document_revision))
}

[live_command(description = "Switch between markdown and plain text modes.")]
def text_viewer_set_mode(input : JsonValue?) : JsonValue? {
    let args = from_JV(input, type<TextViewerModeArgs>)
    let requested = to_lower(args.mode)
    if (requested == "markdown") {
        set_mode(TextViewMode.markdown)
    } elif (requested == "plain" || requested == "text") {
        set_mode(TextViewMode.plain_text)
    } else {
        return JV((ok = false, error = "mode must be 'markdown' or 'plain'"))
    }
    return JV((ok = true, mode = mode_name()))
}

[live_command(description = "Set the fixed-step viewer zoom percentage.")]
def text_viewer_set_zoom(input : JsonValue?) : JsonValue? {
    let args = from_JV(input, type<TextViewerZoomArgs>)
    g_zoom_percent = clamp(((args.percent + 2) / 5) * 5, 50, 300)
    return JV((ok = true, zoom_percent = g_zoom_percent))
}

[export]
def init() {
    harness_init("daScript text viewer", 1100, 780)
    var io & = unsafe(GetIO())
    io.ConfigFlags |= ImGuiConfigFlags.DockingEnable
    g_fonts = load_unicode_font_roles(18.0f, g_font_sources)
    apply_daslang_theme(GetStyle())
    parse_command_line()
    if (empty(g_path)) {
        install_welcome_document()
    } elif (!load_path(g_path)) {
        let load_error = g_status
        install_welcome_document()
        g_status = load_error
    }
}

[export]
def update() {
    if (!harness_begin_frame()) return
    harness_new_frame()

    draw_menu_bar()
    dockspace(DOCK_ROOT, (flags = ImGuiDockNodeFlags.PassthruCentralNode)) {
        if (!DOCK_ROOT.has_initial_layout && DOCK_ROOT.dock_id != 0u) {
            setup_default_layout(DOCK_ROOT.dock_id)
            DOCK_ROOT.has_initial_layout = true
        }
        dock_window(DOCUMENT_WINDOW,
                    (text = "Document", closable = false,
                     flags = ImGuiWindowFlags.NoCollapse)) {
            handle_shortcuts_and_zoom()
            draw_document()
        }
    }

    let dialog_result = file_dialog_draw()
    if (dialog_result == FileDialogResult.confirmed) {
        let _ = load_path(file_dialog_selected_path())
    } elif (dialog_result == FileDialogResult.cancelled) {
        g_status = "Open cancelled."
    }

    harness_end_frame()
}

[export]
def shutdown() {
    markdown_view_dispose(g_markdown_view)
    text_source_view_dispose(g_source_view)
    release_document()
    harness_shutdown()
    delete g_font_sources
}

[export]
def main() {
    init()
    while (!exit_requested()) {
        update()
        harness_maybe_collect_gc()
    }
    shutdown()
}
