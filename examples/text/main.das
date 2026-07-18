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
require daslib/command_line
require daslib/fio
require daslib/json
require daslib/json_boost
require daslib/module_path
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

struct TextViewerDocument {
    id : int
    pending_dock : bool = true
    document : MdDocument = MdDocument()
    markdown_view : MarkdownViewState = MarkdownViewState()
    source_document : TextSourceDocument = TextSourceDocument()
    source_view : TextSourceViewState = TextSourceViewState()
    source_result : TextSourceViewResult = TextSourceViewResult()
    path : string
    status : string = "Open a UTF-8 text or Markdown file."
    mode : TextViewMode = TextViewMode.markdown
    mode_override : int = -1
    document_revision : int = 0
    zoom_step : int = 20       // 10..60 maps to 50%..300% in exact 5% steps
}

var private g_fonts = FontRoleSet()
var private g_font_sources : array<string>
var private g_documents : array<TextViewerDocument>
var private g_document_windows : table<int; DockWindowState>
var private g_active_document_id = 0
var private g_next_document_id = 1
var private g_initial_path : string
var private g_initial_mode_override = -1
var private g_last_open_error : string
var private g_open_clicks = 0
var private g_reload_clicks = 0
var private g_quit_clicks = 0
var private g_markdown_clicks = 0
var private g_plain_clicks = 0

def private mode_name(doc : TextViewerDocument const) : string {
    return doc.mode == TextViewMode.markdown ? "markdown" : "plain"
}

def private zoom_percent(doc : TextViewerDocument const) : int {
    return doc.zoom_step * 5
}

def private set_zoom_percent(var doc : TextViewerDocument; percent : int) {
    doc.zoom_step = clamp((percent + 2) / 5, 10, 60)
}

def private active_document_index() : int {
    let index = find_index_if(g_documents) $(doc) => doc.id == g_active_document_id
    return index >= 0 ? index : (length(g_documents) > 0 ? 0 : -1)
}

def private markdown_path(path : string) : bool {
    let lower = to_lower(path)
    return ends_with(lower, ".md") || ends_with(lower, ".markdown")
}

def private set_mode(var doc : TextViewerDocument; mode : TextViewMode) {
    if (doc.mode == mode) return
    doc.mode = mode
    markdown_view_clear_selection(doc.markdown_view)
    text_source_view_clear_selection(doc.source_view)
}

def private release_document_content(var doc : TextViewerDocument) {
    markdown_view_invalidate(doc.markdown_view)
    delete doc.document.nodes
    doc.document = MdDocument()
    text_source_view_invalidate(doc.source_view)
    text_source_document_dispose(doc.source_document)
}

def private dispose_document(var doc : TextViewerDocument) {
    markdown_view_dispose(doc.markdown_view)
    text_source_view_dispose(doc.source_view)
    release_document_content(doc)
}

def private install_source(var viewer : TextViewerDocument; path, source : string) {
    let next_revision = viewer.document_revision + 1
    var next_document <- md_parse(source)
    var next_source_document <- markdown_source_document(source, next_revision)
    release_document_content(viewer)
    viewer.document <- next_document
    viewer.source_document <- next_source_document
    viewer.path := path
    viewer.document_revision = next_revision

    if (viewer.mode_override == int(TextViewMode.markdown)) {
        viewer.mode = TextViewMode.markdown
    } elif (viewer.mode_override == int(TextViewMode.plain_text)) {
        viewer.mode = TextViewMode.plain_text
    } else {
        viewer.mode = markdown_path(path) ? TextViewMode.markdown : TextViewMode.plain_text
    }
    viewer.status = (viewer.document.ok
        ? "Loaded {length(viewer.source_document.source)} UTF-8 bytes."
        : viewer.document.error)
}

def private load_path(var viewer : TextViewerDocument; path : string) : bool {
    if (empty(path)) {
        viewer.status = "No file path was provided."
        return false
    }
    if (!fexist(path)) {
        viewer.status = "File does not exist: {path}"
        return false
    }
    let file_info = stat(path)
    if (!file_info.is_valid || file_info.is_dir) {
        viewer.status = "Path is not a regular file: {path}"
        return false
    }
    var source = fread(path)
    if (!is_utf8_string_valid(source)) {
        viewer.status = "File is not valid UTF-8: {path}"
        return false
    }
    if (length(source) >= 3 && contains_utf8_bom(source)) {
        source = slice(source, 3)
    }
    install_source(viewer, path, source)
    return true
}

def private install_welcome_document(var viewer : TextViewerDocument) {
    let welcome = %stringify~# Text viewer

Open a UTF-8 file from **File / Open...**, or pass its path on the command line.

- `.md` and `.markdown` files open as rendered Markdown.
- Other files open as regular text.
- Switch modes at any time with **View** or the buttons below.
- Use the bottom-right slider or Ctrl+mouse-wheel to zoom.
%%
    install_source(viewer, "", welcome)
    viewer.mode = TextViewMode.markdown
    viewer.status = "No file selected."
}

def private add_document(path : string; mode_override : int = -1;
                         welcome_on_failure : bool = false) : bool {
    var viewer = TextViewerDocument(id = g_next_document_id,
                                    mode_override = mode_override)
    g_next_document_id++
    if (empty(path)) {
        install_welcome_document(viewer)
    } elif (!load_path(viewer, path)) {
        g_last_open_error = viewer.status
        if (!welcome_on_failure) {
            dispose_document(viewer)
            return false
        }
        let error = viewer.status
        install_welcome_document(viewer)
        viewer.status = error
    }
    g_active_document_id = viewer.id
    g_document_windows[viewer.id] = DockWindowState(open = true)
    g_documents |> emplace(viewer)
    return true
}

def private open_file_dialog() {
    let active = active_document_index()
    var start_dir : string
    if (active >= 0 && !empty(g_documents[active].path)) {
        start_dir = dir_name(g_documents[active].path)
    }
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
            g_initial_mode_override = int(TextViewMode.plain_text)
        } elif (arg == "--markdown") {
            g_initial_mode_override = int(TextViewMode.markdown)
        } elif (!starts_with(arg, "--") && empty(g_initial_path)) {
            g_initial_path := arg
        }
    }
}

def private draw_menu_bar() {
    let active = active_document_index()
    let has_active = active >= 0
    main_menu_bar(MAIN_MENU) {
        menu(FILE_MENU, (text = "File", enabled = true)) {
            menu_label(FILE_OPEN, (text = "Open...", shortcut = "Ctrl+O"))
            menu_label(FILE_RELOAD, (text = "Reload", shortcut = "Ctrl+R",
                                     enabled = has_active && !empty(g_documents[active].path)))
            separator(FILE_SEPARATOR)
            menu_label(FILE_QUIT, (text = "Quit", shortcut = "Ctrl+Q"))
        }
        menu(VIEW_MENU, (text = "View", enabled = true)) {
            menu_label(VIEW_MARKDOWN, (text = "Markdown", shortcut = "",
                                      selected = has_active &&
                                          g_documents[active].mode == TextViewMode.markdown))
            menu_label(VIEW_PLAIN, (text = "Plain text", shortcut = "",
                                   selected = has_active &&
                                       g_documents[active].mode == TextViewMode.plain_text))
        }
        if (has_active && !empty(g_documents[active].path)) {
            separator(MENU_PATH_SEPARATOR)
            text_disabled(MENU_PATH, (text = g_documents[active].path))
        }
    }

    if (FILE_OPEN.click_count > g_open_clicks) {
        g_open_clicks = FILE_OPEN.click_count
        open_file_dialog()
    }
    if (FILE_RELOAD.click_count > g_reload_clicks) {
        g_reload_clicks = FILE_RELOAD.click_count
        if (has_active && !empty(g_documents[active].path)) {
            let path = g_documents[active].path
            let _ = load_path(g_documents[active], path)
        }
    }
    if (FILE_QUIT.click_count > g_quit_clicks) {
        g_quit_clicks = FILE_QUIT.click_count
        request_exit()
    }
    if (VIEW_MARKDOWN.click_count > g_markdown_clicks) {
        g_markdown_clicks = VIEW_MARKDOWN.click_count
        if (has_active) set_mode(g_documents[active], TextViewMode.markdown)
    }
    if (VIEW_PLAIN.click_count > g_plain_clicks) {
        g_plain_clicks = VIEW_PLAIN.click_count
        if (has_active) set_mode(g_documents[active], TextViewMode.plain_text)
    }
}

def private draw_document(var doc : TextViewerDocument) {
    let footer_height = GetFrameHeightWithSpacing()
    child(DOCUMENT_CONTENT,
          (text = "document content",
           size = float2(0.0f, -footer_height),
           child_flags = ImGuiChildFlags.None,
           window_flags = ImGuiWindowFlags.HorizontalScrollbar)) {
        if (doc.mode == TextViewMode.markdown) {
            var style = MarkdownViewStyle(
                fonts = g_fonts,
                zoom = float(zoom_percent(doc)) / 100.0f,
                write_system_clipboard = !harness_is_headless())
            let _ = markdown_view("text viewer", doc.document, doc.markdown_view,
                                  style, doc.document_revision)
        } else {
            var source_style = TextSourceViewStyle(
                fonts = g_fonts,
                zoom = float(zoom_percent(doc)) / 100.0f,
                wrap = false,
                write_system_clipboard = !harness_is_headless())
            doc.source_result <- text_source_view("text source", doc.source_document,
                                                  doc.source_view, source_style)
        }
    }

    let rendered = doc.mode == TextViewMode.markdown
    if (icon_button(MODE_TOGGLE,
            (glyph = rendered ? "edit" : "visible",
             tip = rendered ? "Show Markdown source" : "Render Markdown",
             size = 18.0f, weight = 1.0f))) {
        set_mode(doc, rendered ? TextViewMode.plain_text : TextViewMode.markdown)
    }
    same_line()
    text_disabled(MODE_LABEL,
        (text = rendered ? "Rendered Markdown" : "Markdown source"))
    same_line()
    text_disabled(STATUS, (text = doc.status))

    let right_edge = GetCursorPosX() + GetContentRegionAvail().x
    let zoom_width = 245.0f
    same_line()
    SetCursorPosX(max(GetCursorPosX(), right_edge - zoom_width))
    text(ZOOM_LABEL, (text = "Zoom"))
    same_line()
    SetNextItemWidth(150.0f)
    edit_slider_int(unsafe(addr(doc.zoom_step)),
        (id = "TEXT_VIEWER_ZOOM", text = "##zoom", v_min = 10,
         v_max = 60, format = ""))
    same_line()
    text_disabled(ZOOM_VALUE, (text = "{zoom_percent(doc)}%"))
    same_line()
    if (small_button(ZOOM_RESET, (text = "100%"))) {
        set_zoom_percent(doc, 100)
    }
}

def private handle_global_shortcuts() {
    let io & = unsafe(GetIO())
    let command_down = io.ConfigMacOSXBehaviors ? io.KeySuper : io.KeyCtrl
    if (command_down && IsKeyPressed(ImGuiKey.O, false)) {
        open_file_dialog()
    }
    if (command_down && IsKeyPressed(ImGuiKey.Q, false)) {
        request_exit()
    }
}

def private handle_document_shortcuts(var doc : TextViewerDocument) {
    let io & = unsafe(GetIO())
    let command_down = io.ConfigMacOSXBehaviors ? io.KeySuper : io.KeyCtrl
    if (IsWindowFocused(ImGuiFocusedFlags.RootAndChildWindows)
        && command_down && IsKeyPressed(ImGuiKey.R, false) && !empty(doc.path)) {
        let path = doc.path
        let _ = load_path(doc, path)
    }
    if (IsWindowHovered(ImGuiHoveredFlags.RootAndChildWindows) && command_down) {
        if (io.MouseWheel > 0.0f) {
            doc.zoom_step = min(doc.zoom_step + 1, 60)
        } elif (io.MouseWheel < 0.0f) {
            doc.zoom_step = max(doc.zoom_step - 1, 10)
        }
    }
}

def private draw_primary_document(var doc : TextViewerDocument) {
    dock_window(DOCUMENT_WINDOW,
                (text = "Document", closable = false,
                 flags = ImGuiWindowFlags.NoCollapse)) {
        if (IsWindowFocused(ImGuiFocusedFlags.RootAndChildWindows)) {
            g_active_document_id = doc.id
        }
        handle_document_shortcuts(doc)
        draw_document(doc)
    }
}

def private draw_additional_document(var doc : TextViewerDocument; dock_id : uint) {
    if (doc.pending_dock && dock_id != 0u) {
        unsafe {
            g_document_windows[doc.id].pending_dock_to = dock_id
        }
        doc.pending_dock = false
    }
    let visible_title = empty(doc.path) ? "Text viewer" : base_name(doc.path)
    let title = "{visible_title}##text_document_{doc.id}"
    dock_window(g_document_windows[doc.id],
                (text = title, closable = true,
                 flags = ImGuiWindowFlags.NoCollapse)) {
        if (IsWindowFocused(ImGuiFocusedFlags.RootAndChildWindows)) {
            g_active_document_id = doc.id
        }
        handle_document_shortcuts(doc)
        draw_document(doc)
    }
}

def private remove_closed_documents() {
    var index = length(g_documents) - 1
    while (index > 0) {
        let id = g_documents[index].id
        if (!g_document_windows[id].open) {
            let removed_active = g_documents[index].id == g_active_document_id
            dispose_document(g_documents[index])
            g_documents |> erase(index)
            g_document_windows |> erase(id)
            if (removed_active) {
                g_active_document_id = g_documents[0].id
            }
        }
        index--
    }
}

[live_command(description = "Inspect the text viewer without a screenshot.")]
def text_viewer_state(_input : JsonValue?) : JsonValue? {
    let active = active_document_index()
    if (active < 0) return JV((ok = false, error = "no document is open"))
    var doc & = unsafe(g_documents[active])
    return JV((
        ok = doc.document.ok,
        document_count = length(g_documents),
        active_document_id = doc.id,
        path = doc.path,
        byte_count = length(doc.source_document.source),
        mode = mode_name(doc),
        zoom_percent = zoom_percent(doc),
        zoom_step = doc.zoom_step,
        document_revision = doc.document_revision,
        render_revision = doc.markdown_view.render_revision,
        source_render_revision = doc.source_view.render_revision,
        source_layout_hits = doc.source_view.layout_hits,
        source_layout_misses = doc.source_view.layout_misses,
        source_marker_count = doc.source_document.marker_count,
        source_cell_count = length(doc.source_view.cells),
        source_selection_revision = doc.source_view.selection_revision,
        source_selection_commit_revision = doc.source_view.selection_commit_revision,
        source_selection_dragging = doc.source_view.selection_dragging,
        source_selection_start = doc.source_result.selection.source.start_byte,
        source_selection_end = doc.source_result.selection.source.end_byte,
        source_selection_has_markers = doc.source_result.selection.styles.has_markers,
        status = doc.status))
}

[live_command(description = "Find source text and inspect its exact cell-aligned screen range.")]
def text_viewer_source_range(input : JsonValue?) : JsonValue? {
    let active = active_document_index()
    if (active < 0) return JV((ok = false, error = "no document is open"))
    var doc & = unsafe(g_documents[active])
    let args = from_JV(input, type<TextViewerSourceRangeArgs>)
    let start_byte = find(doc.source_document.source, args.text)
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
    for (cell in doc.source_view.cells) {
        if (cell.source.end_byte <= start_byte || cell.source.start_byte >= end_byte) continue
        let screen = float4(
            doc.source_view.screen_origin.x + cell.rect.x,
            doc.source_view.screen_origin.y + cell.rect.y,
            doc.source_view.screen_origin.x + cell.rect.z,
            doc.source_view.screen_origin.y + cell.rect.w)
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
        cell_width = doc.source_view.key.cell_width,
        x0 = rect.x, y0 = rect.y, x1 = rect.z, y1 = rect.w))
}

[live_command(description = "Open a UTF-8 file in the text viewer.")]
def text_viewer_open(input : JsonValue?) : JsonValue? {
    let args = from_JV(input, type<TextViewerOpenArgs>)
    let active = active_document_index()
    if (active < 0) {
        let ok = add_document(args.path)
        let created = active_document_index()
        return JV((ok = ok, path = created >= 0 ? g_documents[created].path : "",
                   status = created >= 0 ? g_documents[created].status : g_last_open_error,
                   document_revision = created >= 0
                       ? g_documents[created].document_revision : 0))
    }
    let ok = load_path(g_documents[active], args.path)
    return JV((ok = ok, path = g_documents[active].path,
               status = g_documents[active].status,
               document_revision = g_documents[active].document_revision))
}

[live_command(description = "Open a UTF-8 file as another docked document.")]
def text_viewer_open_window(input : JsonValue?) : JsonValue? {
    let args = from_JV(input, type<TextViewerOpenArgs>)
    let ok = add_document(args.path)
    let active = active_document_index()
    return JV((ok = ok, document_count = length(g_documents),
               active_document_id = active >= 0 ? g_documents[active].id : 0,
               path = active >= 0 ? g_documents[active].path : "",
               status = ok && active >= 0
                   ? g_documents[active].status : g_last_open_error))
}

[live_command(description = "Switch between markdown and plain text modes.")]
def text_viewer_set_mode(input : JsonValue?) : JsonValue? {
    let active = active_document_index()
    if (active < 0) return JV((ok = false, error = "no document is open"))
    let args = from_JV(input, type<TextViewerModeArgs>)
    let requested = to_lower(args.mode)
    if (requested == "markdown") {
        set_mode(g_documents[active], TextViewMode.markdown)
    } elif (requested == "plain" || requested == "text") {
        set_mode(g_documents[active], TextViewMode.plain_text)
    } else {
        return JV((ok = false, error = "mode must be 'markdown' or 'plain'"))
    }
    return JV((ok = true, mode = mode_name(g_documents[active])))
}

[live_command(description = "Set the fixed-step viewer zoom percentage.")]
def text_viewer_set_zoom(input : JsonValue?) : JsonValue? {
    let active = active_document_index()
    if (active < 0) return JV((ok = false, error = "no document is open"))
    let args = from_JV(input, type<TextViewerZoomArgs>)
    set_zoom_percent(g_documents[active], args.percent)
    return JV((ok = true, zoom_percent = zoom_percent(g_documents[active])))
}

[export]
def init() {
    harness_init("daScript text viewer", 1100, 780)
    var io & = unsafe(GetIO())
    io.ConfigFlags |= ImGuiConfigFlags.DockingEnable
    g_fonts = load_unicode_font_roles(18.0f, g_font_sources)
    apply_daslang_theme(GetStyle())
    parse_command_line()
    let _ = add_document(g_initial_path, g_initial_mode_override, true)
}

[export]
def update() {
    if (!harness_begin_frame()) return
    harness_new_frame()

    draw_menu_bar()
    handle_global_shortcuts()
    dockspace(DOCK_ROOT, (flags = ImGuiDockNodeFlags.PassthruCentralNode)) {
        if (!DOCK_ROOT.has_initial_layout && DOCK_ROOT.dock_id != 0u) {
            setup_default_layout(DOCK_ROOT.dock_id)
            DOCK_ROOT.has_initial_layout = true
        }
        if (length(g_documents) > 0) {
            draw_primary_document(g_documents[0])
        }
        for (index in range(1, length(g_documents))) {
            unsafe {
                draw_additional_document(g_documents[index], DOCK_ROOT.dock_id)
            }
        }
    }

    let dialog_result = file_dialog_draw()
    if (dialog_result == FileDialogResult.confirmed) {
        let selected_path = file_dialog_selected_path()
        let _ = add_document(selected_path)
    } elif (dialog_result == FileDialogResult.cancelled) {
        let active = active_document_index()
        if (active >= 0) {
            g_documents[active].status = "Open cancelled."
        }
    }

    remove_closed_documents()

    harness_end_frame()
}

[export]
def shutdown() {
    for (doc in g_documents) {
        dispose_document(doc)
    }
    delete g_documents
    delete g_document_windows
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
