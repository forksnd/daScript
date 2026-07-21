options gen2
options gc
options persistent_heap
options no_unused_function_arguments = false

require imgui/imgui_harness
require imgui/imgui_icons
require imgui/imgui_drawlist_builtin
require imgui/imgui_terminal
require imgui/imgui_fonts
require imgui/imgui_docking_builtin
require imgui/imgui_table_builtin
require imgui/imgui_markdown_tree_sitter
require imgui/imgui_text_source_view
require imgui/imgui_text_tree_sitter
require dashv/dashv_boost
require clipboard/clipboard
require daslib/safe_addr
require daslib/json
require daslib/json_boost
require daslib/jobque_boost
require daslib/strings_boost
require daslib/utf8_utils
require live/live_commands
require ./watcher_core.das
require ./repository_core.das
require ./file_inspection.das
require ./inspection_worker.das
require ./image_preview_gl.das
require math
require strings

struct private WatcherEnvelope {
    @rename = "type" _type : string
}

struct private WatcherConnectedMessage {
    @rename = "type" _type : string
    client_id : int
}

struct private WatcherAttachedMessage {
    @rename = "type" _type : string
    session_id : string
    controls : bool
}

struct private WatcherSessionMessage {
    @rename = "type" _type : string
    session_id : string
}

struct private WatcherErrorMessage {
    @rename = "type" _type : string
    error : string
}

struct private WatcherSessionsMessage {
    @rename = "type" _type : string
    data : array<WatcherSessionInfo>
}

struct private WatcherRepositoriesMessage {
    @rename = "type" _type : string
    data : array<RepositorySnapshot>
}

struct private WatcherFileInspectionMessage {
    @rename = "type" _type : string
    request_id : int
    repository_id : string
    worktree_path : string
    file_path : string
    comparison : string
    revision : string
    diff_context : string
    task_session_id : string
    loading : bool
    patch : string
    view_text : string
    view_base64 : string
    view_byte_count : int
    view_available : bool
    status : string
    error : string
}

struct private WatcherGitFileActionMessage {
    @rename = "type" _type : string
    request_id : int
    repository_id : string
    worktree_path : string
    file_path : string
    action : string
    task_session_id : string
    loading : bool
    status : string
    error : string
}

struct private WatcherRawResetMessage {
    @rename = "type" _type : string
    offset : int64
}

struct private HerderInspectorTextArgs {
    side : string = "view"
}

struct private HerderInspectorOpenArgs {
    repository_id : string
    worktree_path : string
    file_path : string
    comparison : string = "working"
    revision : string
    diff_context : string = "compact"
}

struct private HerderInspectorModeArgs {
    mode : string
}

struct private HerderGitWorktreeArgs {
    repository_id : string
    worktree_path : string
}

struct private HerderGitPerspectiveArgs {
    perspective : string
}

struct private HerderGitFileActionArgs {
    repository_id : string
    worktree_path : string
    file_path : string
    action : string
}

struct private HerderGitShaArgs {
    sha : string
}

enum private GitInspectorMode {
    diff
    view
}

let INSPECTOR_SYNTAX_LABELS : array<string> <- [
    "Auto", "Plain text", "daScript", "C", "C++", "Markdown", "JSON"]
let INSPECTOR_SYNTAX_VALUES : array<string> <- [
    "auto", "plain", "daslang", "c", "cpp", "markdown", "json"]
let INSPECTOR_CONTEXT_LABELS : array<string> <- [
    "Compact (3)", "Expanded (20)", "Full file"]
let INSPECTOR_CONTEXT_VALUES : array<string> <- [
    "compact", "expanded", "full"]

struct private GitInspectorState {
    request_id : int
    repository_id : string
    worktree_path : string
    file_path : string
    comparison : string
    git_revision : string
    diff_context : string = "compact"
    task_session_id : string
    loading : bool
    status : string
    error : string
    revision : int
    prepare : InspectorLoadState = InspectorLoadState()
    prepare_elapsed_usec : int64
    mode : GitInspectorMode = GitInspectorMode.diff
    parsed : GitParsedPatch = GitParsedPatch()
    old_document : TextSourceDocument = TextSourceDocument()
    new_document : TextSourceDocument = TextSourceDocument()
    view_document : TextSourceDocument = TextSourceDocument()
    old_view : TextSourceViewState = TextSourceViewState()
    new_view : TextSourceViewState = TextSourceViewState()
    view : TextSourceViewState = TextSourceViewState()
    markdown_document : RichDocument = RichDocument()
    markdown_view : MarkdownViewState = MarkdownViewState()
    changed_source_ranges : array<TextSourceRange>
    view_source : string
    view_prepared : bool
    markdown_gutter_segments : int
    source_patch : string
    source_view_text : string
    source_view_base64 : string
    source_view_byte_count : int
    source_view_available : bool
    syntax_choice : int
    diff_context_choice : int
    resolved_language : string
    preview_texture : uint
    preview_width : int
    preview_height : int
    preview_byte_count : int
    preview_error : string
    diff_scroll_y : float
    diff_scroll_jump_pending : bool
    diff_split_fraction : float = 0.5f
    diff_current_hunk : int
    diff_left_rect : float4
    diff_right_rect : float4
    diff_scrollbar_rect : float4
}

var private g_terminal = Terminal()
var private g_terminal_view = ImGuiTerminalState()
var private g_terminal_cache = ImGuiTerminalCache()
var private g_terminal_font : ImFont?
var private g_fonts = FontRoleSet()
var private g_font_sources : array<string>
var private g_zoom_percent = 150
var private g_status = "Starting"
var private g_url : string
var private g_token : string
var private g_selected_worktree_path : string
var private g_selected_repository_id : string
var private g_focus_terminal_pending = false
var private g_requested_history_commit : string
var private g_pending_clipboard_sha : string
var private g_last_copied_sha : string
var private g_clipboard_copy_status = ClipboardStatus.no_content
var private g_clipboard_copy_attempts = 0
var private g_clipboard_copy_verified = false
var private g_inspector = GitInspectorState()
var private g_prepare_worker = InspectorPrepareWorker()
var private g_jobque_owned = false
var private SESSION_ROW : table<int; ToggleState>
var private WORKTREE_ROW : table<int; ToggleState>
var private GIT_FILE_ROW : table<int; ToggleState>
var private GIT_COMMIT_TITLE_TEXT : table<int; NarrativeState>
var private GIT_COMMIT_AUTHOR_TEXT : table<int; NarrativeState>
var private GIT_PR_COMMIT_HOVER : table<int; EmptyMarkerState>
var private GIT_HISTORY_COMMIT_ROW : table<int; ClickState>
var private GIT_COMMIT_FILE_ROW : table<int; ToggleState>
var private REPOSITORY_ERROR : table<int; NarrativeState>
var private REPOSITORY_TITLE : table<int; NarrativeState>
var private REPOSITORY_EMPTY : table<int; NarrativeState>
var private GIT_GROUP_TITLE : table<int; NarrativeState>
var private REPOSITORY_SEPARATOR : table<int; EmptyMarkerState>
var private WORKTREE_SEPARATOR : table<int; EmptyMarkerState>
var private GIT_GROUP_SEPARATOR : table<int; EmptyMarkerState>
var private GIT_GROUP_SPACING : table<int; EmptyMarkerState>
var private GIT_HOVER_TOOLTIP : NarrativeState

def private setup_default_layout(dock_id : uint) {
    DockBuilderRemoveNode(dock_id)
    DockBuilderAddDockSpaceNode(dock_id, ImGuiDockNodeFlags.None)
    let viewport = GetMainViewport()
    DockBuilderSetNodeSize(dock_id, viewport.Size)
    var left_id : uint = 0u
    var right_id : uint = 0u
    DockBuilderSplitNode(dock_id, ImGuiDir.Left, 0.24f, left_id, right_id)
    var upper_id : uint = 0u
    var terminal_id : uint = 0u
    DockBuilderSplitNode(right_id, ImGuiDir.Down, 0.34f, terminal_id, upper_id)
    var git_id : uint = 0u
    var inspector_id : uint = 0u
    DockBuilderSplitNode(upper_id, ImGuiDir.Left, 0.34f, git_id, inspector_id)
    var changelist_id : uint = 0u
    var activity_id : uint = 0u
    DockBuilderSplitNode(git_id, ImGuiDir.Up, 0.35f, changelist_id, activity_id)
    DockBuilderDockWindow("Sessions & Activity", left_id)
    DockBuilderDockWindow("Repositories & Worktrees", left_id)
    DockBuilderDockWindow("Git Changelist", changelist_id)
    DockBuilderDockWindow("Git Activity", activity_id)
    DockBuilderDockWindow("File Inspector", inspector_id)
    DockBuilderDockWindow("Terminal", terminal_id)
    DockBuilderDockWindow("Settings", activity_id)
    DockBuilderFinish(dock_id)
}

def private option_value(prefix : string; fallback : string = "") : string {
    for (argument in get_command_line_arguments()) {
        if (starts_with(argument, prefix)) return slice(argument, length(prefix))
    }
    return fallback
}

def private reset_terminal(columns : int = 100; rows : int = 30) {
    terminal_dispose(g_terminal)
    terminal_viewport_snapshot_dispose(g_terminal_cache.snapshot)
    var inscope created <- terminal_create(columns, rows)
    g_terminal := created
    g_terminal_view = ImGuiTerminalState()
    g_terminal_cache = ImGuiTerminalCache()
}

def private feed_terminal_frame(message : string#; byte_count : int) {
    var inscope bytes : array<uint8>
    bytes |> resize(byte_count)
    if (byte_count > 0) {
        unsafe { memcpy(addr(bytes[0]), reinterpret<void?>(message), byte_count) }
        terminal_feed_bytes(g_terminal, bytes)
    }
}

def private upload_inspector_preview(event : InspectorPrepareEvent#) {
    g_inspector.preview_width = event.preview_width
    g_inspector.preview_height = event.preview_height
    g_inspector.preview_byte_count = event.preview_byte_count
    g_inspector.preview_error = clone_string(event.preview_error)
    if (empty(event.preview_pixels)) return
    inspector_preview_texture_upload(event.preview_pixels,
        event.preview_width, event.preview_height,
        g_inspector.preview_texture)
}

def private release_inspector_content() {
    if (g_inspector.preview_texture != 0u) {
        inspector_preview_texture_release(g_inspector.preview_texture)
    }
    git_parsed_patch_dispose(g_inspector.parsed)
    text_source_view_invalidate(g_inspector.old_view)
    text_source_view_invalidate(g_inspector.new_view)
    text_source_view_invalidate(g_inspector.view)
    text_source_document_dispose(g_inspector.old_document)
    text_source_document_dispose(g_inspector.new_document)
    text_source_document_dispose(g_inspector.view_document)
    markdown_view_dispose(g_inspector.markdown_view)
    rich_document_dispose(g_inspector.markdown_document)
    delete g_inspector.changed_source_ranges
    g_inspector.view_source = ""
    g_inspector.view_prepared = false
    g_inspector.markdown_gutter_segments = 0
    g_inspector.preview_width = 0
    g_inspector.preview_height = 0
    g_inspector.preview_byte_count = 0
    g_inspector.preview_error = ""
    g_inspector.diff_scroll_y = 0.0f
    g_inspector.diff_scroll_jump_pending = false
    g_inspector.diff_current_hunk = 0
    g_inspector.diff_left_rect = float4()
    g_inspector.diff_right_rect = float4()
    g_inspector.diff_scrollbar_rect = float4()
}

def private queue_inspector_prepare(status : string) {
    let choice = clamp(g_inspector.syntax_choice, 0,
        length(INSPECTOR_SYNTAX_VALUES) - 1)
    g_inspector.resolved_language = inspector_language_for_path(
        g_inspector.file_path, INSPECTOR_SYNTAX_VALUES[choice])
    let generation = inspector_load_queue(g_inspector.prepare, status)
    g_inspector.error = ""
    if (!inspector_prepare_worker_queue(g_prepare_worker, generation,
            g_inspector.source_patch, g_inspector.source_view_text,
            g_inspector.resolved_language, g_inspector.source_view_base64,
            g_inspector.source_view_byte_count,
            g_inspector.source_view_available)) {
        let failure = "File inspection preparation worker is unavailable"
        inspector_load_fail(g_inspector.prepare, generation,
            failure)
        g_inspector.error = failure
    }
}

def private install_file_inspection(message : WatcherFileInspectionMessage const) {
    if (message.request_id != g_inspector.request_id) return
    g_inspector.task_session_id = message.task_session_id
    g_inspector.loading = message.loading
    g_inspector.status = message.status
    if (message.loading) return
    g_inspector.error = message.error
    if (!empty(message.error)) {
        inspector_load_fail(g_inspector.prepare,
            g_inspector.prepare.generation, message.error)
        return
    }
    g_inspector.source_patch = clone_string(message.patch)
    g_inspector.source_view_text = clone_string(message.view_text)
    g_inspector.source_view_base64 = clone_string(message.view_base64)
    g_inspector.source_view_byte_count = message.view_byte_count
    g_inspector.source_view_available = message.view_available
    queue_inspector_prepare("Preparing diff...")
}

def private drain_inspector_prepare_events() {
    while (true) {
        var event : InspectorPrepareEvent
        if (!inspector_prepare_worker_take_event(g_prepare_worker, event)) break
        if (event.kind == InspectorPrepareEventKind.started) {
            inspector_load_start(g_inspector.prepare, event.generation,
                "Preparing diff...")
        } elif (event.kind == InspectorPrepareEventKind.failed) {
            if (inspector_load_fail(g_inspector.prepare,
                    event.generation, clone_string(event.error))) {
                g_inspector.error = clone_string(event.error)
                g_inspector.prepare_elapsed_usec = event.elapsed_usec
            }
        } elif (event.kind == InspectorPrepareEventKind.completed) {
            if (!inspector_load_succeed(g_inspector.prepare,
                    event.generation)) continue
            release_inspector_content()
            g_inspector.revision = event.generation
            g_inspector.parsed <- event.parsed
            g_inspector.old_document <- event.old_document
            g_inspector.new_document <- event.new_document
            g_inspector.view_document <- event.view_document
            g_inspector.markdown_document <- event.markdown_document
            g_inspector.markdown_view <- event.markdown_view
            g_inspector.changed_source_ranges <- event.changed_source_ranges
            g_inspector.view_source = clone_string(event.view_source)
            g_inspector.view_prepared = !g_inspector.parsed.binary
            upload_inspector_preview(
                unsafe(reinterpret<InspectorPrepareEvent const#>(event)))
            g_inspector.error = ""
            g_inspector.prepare_elapsed_usec = event.elapsed_usec
        }
    }
}

def private begin_file_inspection(repository_id, worktree_path, file_path,
                                  comparison, diff_context : string;
                                  git_revision : string = "") {
    let same_identity = g_inspector.repository_id == repository_id && g_inspector.worktree_path == worktree_path && g_inspector.file_path == file_path && g_inspector.comparison == comparison && g_inspector.git_revision == git_revision
    let same_content = g_inspector.prepare.has_retained_content && same_identity
    g_inspector.request_id++
    g_inspector.repository_id = repository_id
    g_inspector.worktree_path = worktree_path
    g_inspector.file_path = file_path
    g_inspector.comparison = comparison
    g_inspector.git_revision = git_revision
    g_inspector.diff_context = diff_context
    g_inspector.diff_context_choice = 0
    if (diff_context == "expanded") {
        g_inspector.diff_context_choice = 1
    } elif (diff_context == "full") {
        g_inspector.diff_context_choice = 2
    }
    DIFF_CONTEXT.value = g_inspector.diff_context_choice
    g_inspector.task_session_id = ""
    g_inspector.loading = true
    g_inspector.status = "Requesting diff..."
    g_inspector.error = ""
    if (!same_identity) {
        g_inspector.source_patch = ""
        g_inspector.source_view_text = ""
        g_inspector.source_view_base64 = ""
        g_inspector.source_view_byte_count = 0
        g_inspector.source_view_available = false
        g_inspector.syntax_choice = 0
        g_inspector.resolved_language = inspector_language_for_path(file_path)
        DIFF_SYNTAX.value = 0
    }
    if (same_content) {
        inspector_load_supersede(g_inspector.prepare, "Requesting diff...")
    } else {
        release_inspector_content()
        inspector_load_reset(g_inspector.prepare)
    }
    g_inspector.mode = GitInspectorMode.diff
    INSPECTOR_DIFF_TAB.pending_select = true
}

class private WatcherRichClient : HvWebSocketClient {
    connected : bool
    attached : bool
    controls : bool
    client_id : int
    selected_id : string
    attached_id : string
    sessions : array<WatcherSessionInfo>
    repositories : array<RepositorySnapshot>
    error : string
    last_heartbeat : int64
    last_list : int64
    last_reconnect : int64
    sent_input_bytes : int64
    desired_columns : int
    desired_rows : int
    sent_columns : int
    sent_rows : int
    last_review_identity : string
    last_history_identity : string
    last_refs_identity : string
    git_action_request_id : int
    git_action_loading : bool
    git_action_file_path : string
    git_action_status : string
    git_failure_revisions : table<string; int64>
    git_failure_baseline_ready : bool

    def override onOpen {
        connected = true
        error = ""
        last_reconnect = ref_time_ticks()
        last_heartbeat = ref_time_ticks()
        last_list = ref_time_ticks()
        send("\{\"op\":\"list\"\}")
    }

    def override onClose {
        connected = false
        attached = false
        controls = false
        sent_columns = 0
        sent_rows = 0
        error = "watcher connection closed"
        last_reconnect = ref_time_ticks()
    }

    def override onMessage(message : string#) {
        var envelope : WatcherEnvelope
        let body = string(message)
        if (!sscan_json(body, envelope)) {
            error = "invalid watcher message"
            return
        }
        if (envelope._type == "connected") {
            var incoming : WatcherConnectedMessage
            if (sscan_json(body, incoming)) {
                client_id = incoming.client_id
            }
        } elif (envelope._type == "sessions") {
            var incoming : WatcherSessionsMessage
            if (sscan_json(body, incoming)) {
                delete sessions
                sessions <- incoming.data
                if (empty(selected_id)) {
                    for (session in sessions) {
                        if (session.state == "running") {
                            attach_session(session.id)
                            break
                        }
                    }
                }
            }
        } elif (envelope._type == "launched") {
            var incoming : WatcherSessionMessage
            if (sscan_json(body, incoming)) {
                attach_session(incoming.session_id)
                TERMINAL_WIN.open = true
                TERMINAL_WIN.pending_raise = true
                g_focus_terminal_pending = true
            }
        } elif (envelope._type == "repositories") {
            var incoming : WatcherRepositoriesMessage
            if (sscan_json(body, incoming)) {
                delete repositories
                repositories <- incoming.data
                var selection_found = false
                for (repository in repositories) {
                    if (repository.record.id != g_selected_repository_id) continue
                    if (empty(repository.worktrees)) {
                        selection_found = true
                    }
                    for (worktree in repository.worktrees) {
                        if (worktree.path == g_selected_worktree_path) {
                            selection_found = true
                        }
                    }
                }
                if (!selection_found && !empty(repositories)) {
                    g_selected_repository_id = repositories[0].record.id
                    if (!empty(repositories[0].worktrees)) {
                        g_selected_worktree_path = repositories[0].worktrees[0].path
                    } else {
                        g_selected_worktree_path = repositories[0].record.checkout_path
                    }
                }
                let review_identity = "{g_selected_repository_id}\n{g_selected_worktree_path}"
                if (!empty(g_selected_repository_id)
                        && !empty(g_selected_worktree_path)
                        && review_identity != last_review_identity) {
                    request_repository_review(
                        g_selected_repository_id, g_selected_worktree_path)
                    request_repository_history(
                        g_selected_repository_id, g_selected_worktree_path)
                    request_repository_refs(
                        g_selected_repository_id, g_selected_worktree_path)
                }
                for (repository in repositories) {
                    let id = repository.record.id
                    let seen = key_exists(git_failure_revisions, id) ? git_failure_revisions[id] : 0l
                    if (repository.failure_revision <= seen) continue
                    git_failure_revisions[id] = repository.failure_revision
                    if (git_failure_baseline_ready
                            && !empty(repository.failure_session_id)) {
                        attach_session(repository.failure_session_id)
                        TERMINAL_WIN.open = true
                        TERMINAL_WIN.pending_raise = true
                        g_focus_terminal_pending = true
                    }
                    if (git_failure_baseline_ready) {
                        error = repository.failure_message
                    }
                }
                git_failure_baseline_ready = true
            } else {
                error = "invalid repository snapshot message"
            }
        } elif (envelope._type == "file_inspection") {
            var incoming : WatcherFileInspectionMessage
            if (sscan_json(body, incoming)) {
                install_file_inspection(incoming)
            }
        } elif (envelope._type == "git_file_action") {
            var incoming : WatcherGitFileActionMessage
            if (sscan_json(body, incoming)
                    && incoming.request_id == git_action_request_id) {
                git_action_loading = incoming.loading
                git_action_file_path = incoming.file_path
                git_action_status = incoming.status
                if (!empty(incoming.error)) {
                    error = incoming.error
                }
                if (incoming.status == "failed"
                        && !empty(incoming.task_session_id)) {
                    attach_session(incoming.task_session_id)
                    TERMINAL_WIN.open = true
                    TERMINAL_WIN.pending_raise = true
                    g_focus_terminal_pending = true
                }
            }
        } elif (envelope._type == "attached") {
            var incoming : WatcherAttachedMessage
            if (sscan_json(body, incoming)) {
                attached_id = incoming.session_id
                attached = true
                controls = incoming.controls
                g_status = incoming.controls ? "Attached with control" : "Attached read-only"
                if (controls) {
                    if (desired_columns <= 0 || desired_rows <= 0) {
                        desired_columns = g_terminal.columns
                        desired_rows = g_terminal.rows
                    }
                    flush_resize()
                }
            }
        } elif (envelope._type == "detached") {
            attached = false
            controls = false
            attached_id = ""
            sent_columns = 0
            sent_rows = 0
        } elif (envelope._type == "raw_reset") {
            var incoming : WatcherRawResetMessage
            if (sscan_json(body, incoming)) {
                reset_terminal(g_terminal.columns, g_terminal.rows)
                g_status = "Raw replay reset at byte {incoming.offset}"
            }
        } elif (envelope._type == "error") {
            var incoming : WatcherErrorMessage
            if (sscan_json(body, incoming)) {
                error = incoming.error
            }
        }
    }

    def override onMessageFrame(message : string#; byte_count : int; opcode : ws_opcode) {
        if (opcode == ws_opcode.WS_OPCODE_TEXT) {
            onMessage(message)
        } elif (opcode == ws_opcode.WS_OPCODE_BINARY) {
            feed_terminal_frame(message, byte_count)
        }
    }

    def attach_session(session_id : string) {
        if (!connected || empty(session_id)) return
        var columns = 100
        var rows = 30
        for (session in sessions) {
            if (session.id == session_id) {
                columns = max(20, session.columns)
                rows = max(5, session.rows)
                break
            }
        }
        selected_id = session_id
        attached = false
        controls = false
        sent_columns = 0
        sent_rows = 0
        error = ""
        reset_terminal(columns, rows)
        send("\{\"op\":\"attach\",\"session_id\":{sprint_json(session_id, false)}," +
            "\"control\":true,\"raw\":true\}")
    }

    def launch(command_line, cwd : string; repository_id : string = "";
               worktree_path : string = "") {
        if (!connected) {
            error = "watcher is not connected"
            return
        }
        send("\{\"op\":\"launch\",\"command_line\":{sprint_json(command_line, false)}," +
            "\"cwd\":{sprint_json(cwd, false)},\"columns\":100,\"rows\":30," +
            "\"timeout_ms\":0,\"display_name\":\"PowerShell\",\"kind\":\"interactive-shell\"," +
            "\"target_id\":\"local\",\"repository_id\":{sprint_json(repository_id, false)}," +
            "\"worktree_path\":{sprint_json(worktree_path, false)}\}")
    }

    def add_repository(path : string) {
        if (!connected || empty(path)) return
        send("\{\"op\":\"repository_add\",\"repository_path\":{sprint_json(path, false)}," +
            "\"target_id\":\"local\"\}")
    }

    def refresh_repository(repository_id : string) {
        if (!connected || empty(repository_id)) return
        send("\{\"op\":\"repository_refresh\",\"repository_id\":" +
            "{sprint_json(repository_id, false)}\}")
    }

    def request_repository_review(repository_id : string; worktree_path : string) {
        if (!connected || empty(repository_id) || empty(worktree_path)) return
        last_review_identity = "{repository_id}\n{worktree_path}"
        send("\{\"op\":\"repository_review\"," +
            "\"repository_id\":{sprint_json(repository_id, false)}," +
            "\"worktree_path\":{sprint_json(worktree_path, false)}\}")
    }

    def request_repository_history(repository_id : string;
                                   worktree_path : string;
                                   ref_name : string = "") {
        if (!connected || empty(repository_id) || empty(worktree_path)) return
        last_history_identity = "{repository_id}\n{worktree_path}\n{ref_name}"
        send("\{\"op\":\"repository_history\"," +
            "\"repository_id\":{sprint_json(repository_id, false)}," +
            "\"worktree_path\":{sprint_json(worktree_path, false)}," +
            "\"revision\":{sprint_json(ref_name, false)}\}")
    }

    def request_repository_commit_files(repository_id : string;
                                        worktree_path : string;
                                        commit_hash : string) {
        if (!connected || empty(repository_id) || empty(worktree_path)
                || empty(commit_hash)) return
        send("\{\"op\":\"repository_commit_files\"," +
            "\"repository_id\":{sprint_json(repository_id, false)}," +
            "\"worktree_path\":{sprint_json(worktree_path, false)}," +
            "\"revision\":{sprint_json(commit_hash, false)}\}")
    }

    def request_repository_refs(repository_id : string; worktree_path : string) {
        if (!connected || empty(repository_id) || empty(worktree_path)) return
        last_refs_identity = "{repository_id}\n{worktree_path}"
        send("\{\"op\":\"repository_refs\"," +
            "\"repository_id\":{sprint_json(repository_id, false)}," +
            "\"worktree_path\":{sprint_json(worktree_path, false)}\}")
    }

    def git_file_action(repository_id : string; worktree_path : string;
                        file_path : string; action : string) {
        if (!connected || git_action_loading || empty(repository_id)
                || empty(worktree_path) || empty(file_path)) return
        git_action_request_id++
        git_action_loading = true
        git_action_file_path = file_path
        git_action_status = "running"
        send("\{\"op\":\"git_file_action\"," +
            "\"request_id\":{git_action_request_id}," +
            "\"repository_id\":{sprint_json(repository_id, false)}," +
            "\"worktree_path\":{sprint_json(worktree_path, false)}," +
            "\"file_path\":{sprint_json(file_path, false)}," +
            "\"action\":{sprint_json(action, false)}\}")
    }

    def inspect_file(repository_id, worktree_path, file_path,
                     comparison : string;
                     diff_context : string = "compact";
                     git_revision : string = "") {
        if (!connected || empty(repository_id) || empty(worktree_path)
                || empty(file_path)) return
        begin_file_inspection(repository_id, worktree_path, file_path,
            comparison, diff_context, git_revision)
        send("\{\"op\":\"file_inspect\",\"request_id\":{g_inspector.request_id}," +
            "\"repository_id\":{sprint_json(repository_id, false)}," +
            "\"worktree_path\":{sprint_json(worktree_path, false)}," +
            "\"file_path\":{sprint_json(file_path, false)}," +
            "\"comparison\":{sprint_json(comparison, false)}," +
            "\"revision\":{sprint_json(git_revision, false)}," +
            "\"diff_context\":{sprint_json(diff_context, false)}\}")
    }

    def send_input(text : string) {
        if (!controls || empty(text)) return
        send("\{\"op\":\"input\",\"text\":{sprint_json(text, false)}\}")
        sent_input_bytes += int64(length(text))
    }

    def request_resize(columns, rows : int) {
        desired_columns = columns
        desired_rows = rows
        flush_resize()
    }

    def flush_resize() {
        if (!controls || desired_columns <= 0 || desired_rows <= 0 ||
            (desired_columns == sent_columns && desired_rows == sent_rows)) return
        send("\{\"op\":\"resize\",\"columns\":{desired_columns},\"rows\":{desired_rows}\}")
        sent_columns = desired_columns
        sent_rows = desired_rows
    }

    def terminate_selected() {
        if (!connected || empty(selected_id)) return
        send("\{\"op\":\"terminate\",\"session_id\":{sprint_json(selected_id, false)}\}")
    }

    def refresh() {
        if (connected) send("\{\"op\":\"list\"\}")
    }

    def pump() {
        process_event_que()
        if (!connected) {
            if (get_time_nsec(last_reconnect) / 1000000l >= 1000l) {
                cleanup()
                let result = init(g_url)
                last_reconnect = ref_time_ticks()
                if (result != 0) {
                    error = "watcher reconnect failed ({result})"
                }
            }
            return
        }
        if (get_time_nsec(last_heartbeat) / 1000000l >= 1000l) {
            send("\{\"op\":\"heartbeat\"\}")
            last_heartbeat = ref_time_ticks()
        }
        if (get_time_nsec(last_list) / 1000000l >= 1000l) {
            refresh()
            last_list = ref_time_ticks()
        }
    }
}

var private g_client : WatcherRichClient?

def private zoom_by(delta : int) {
    g_zoom_percent = clamp(g_zoom_percent + delta, 50, 300)
}

def private selected_session_label() : string {
    if (g_client == null || empty(g_client.selected_id)) return "No session selected"
    for (session in g_client.sessions) {
        if (session.id == g_client.selected_id) {
            return "{session.state}  pid {session.pid}  {session.command_line}"
        }
    }
    return g_client.selected_id
}

def private draw_session_list() {
    let connection = g_client != null && g_client.connected ? "connected" : "disconnected"
    text("Watcher: {connection}")
    if (small_button(REFRESH_SESSIONS, (text = "Refresh")) && g_client != null) {
        g_client->refresh()
    }
    same_line()
    if (small_button(TERMINATE_SESSION, (text = "Terminate")) && g_client != null) {
        g_client->terminate_selected()
    }
    separator()
    child(SESSIONS_CHILD, (text = "sessions", size = float2(0.0f, -150.0f),
            child_flags = ImGuiChildFlags.Borders, window_flags = ImGuiWindowFlags.None)) {
        var visible_sessions = 0
        if (g_client != null) {
            for (index in range(length(g_client.sessions))) {
                let session = g_client.sessions[index]
                if (!SHOW_EXITED.value && session.state == "exited") continue
                visible_sessions++
                let row_text = "[{session.state}] {session.command_line}##{session.id}"
                if (selectable(SESSION_ROW[index], (text = row_text))) {
                    g_client->attach_session(session.id)
                    TERMINAL_WIN.open = true
                    TERMINAL_WIN.pending_raise = true
                    g_focus_terminal_pending = true
                }
                SESSION_ROW[index].value = session.id == g_client.selected_id
            }
        }
        if (visible_sessions == 0) {
            text_disabled(NO_SESSION_LABEL,
                (text = SHOW_EXITED.value ? "No sessions" : "No active sessions"))
        }
    }
    text_disabled("Exited sessions are retained and hidden by default.")
}

def private draw_terminal_footer() {
    if (small_button(TERMINAL_ZOOM_OUT_BUTTON, (text = "-5%"))) zoom_by(-5)
    same_line()
    if (small_button(TERMINAL_ZOOM_IN_BUTTON, (text = "+5%"))) zoom_by(5)
    same_line()
    text("Zoom {g_zoom_percent}%")
    same_line()
    text_disabled("{g_terminal_view.scrollback_rows} history lines - {g_status}")
    if (g_client != null) {
        same_line()
        text_disabled("input sent {g_client.sent_input_bytes} bytes")
    }
}

def private copy_terminal_selection() {
    g_status = (imgui_terminal_copy_selection(g_terminal, g_terminal_view)
        ? "Selection copied" : "Nothing selected")
}

def private paste_terminal_clipboard() {
    var pasted = ""
    if (clipboard_get_text(pasted) == ClipboardStatus.ok && g_client != null) {
        g_client->send_input(terminal_encode_paste(g_terminal, pasted))
        g_status = "Clipboard pasted"
    } else {
        g_status = "Clipboard unavailable"
    }
}

def private open_dock_window(var state : DockWindowState) {
    state.open = true
    state.pending_raise = true
}

def private reset_layout() {
    REPOSITORIES_WIN.open = true
    SESSIONS_WIN.open = true
    GIT_CHANGELIST_WIN.open = true
    GIT_ACTIVITY_WIN.open = true
    FILE_INSPECTOR_WIN.open = true
    TERMINAL_WIN.open = true
    SETTINGS_WIN.open = false
    DOCK_ROOT.pending_reset = true
}

def private draw_main_menu() {
    main_menu_bar(MAIN_MENU) {
        menu(FILE_MENU, (text = "File", enabled = true)) {
            if (menu_label(FILE_EXIT, (text = "Exit", shortcut = "Alt+F4"))) {
                request_exit()
            }
        }
        menu(COMMANDS_MENU, (text = "Commands", enabled = true)) {
            if (menu_label(COMMAND_POWERSHELL,
                    (text = "PowerShell", enabled = g_client != null && g_client.connected))) {
                g_client->launch("powershell.exe -NoLogo -NoProfile", g_selected_worktree_path,
                    g_selected_repository_id, g_selected_worktree_path)
            }
        }
        menu(EDIT_MENU, (text = "Edit", enabled = true)) {
            if (menu_label(EDIT_COPY, (text = "Copy", shortcut = "Ctrl+C",
                    enabled = g_terminal_view.selection_active))) copy_terminal_selection()
            if (menu_label(EDIT_PASTE, (text = "Paste", shortcut = "Ctrl+V"))) {
                paste_terminal_clipboard()
            }
            separator(EDIT_SELECTION_SEPARATOR)
            if (menu_label(EDIT_SELECT_ALL, (text = "Select all"))) {
                imgui_terminal_select_all(g_terminal, g_terminal_view)
            }
            if (menu_label(EDIT_CLEAR_SELECTION,
                    (text = "Clear selection", enabled = g_terminal_view.selection_active))) {
                imgui_terminal_clear_selection(g_terminal_view)
            }
        }
        menu(VIEW_MENU, (text = "View", enabled = true)) {
            if (menu_label(VIEW_REPOSITORIES,
                    (text = "Repositories & Worktrees", selected = REPOSITORIES_WIN.open))) {
                open_dock_window(REPOSITORIES_WIN)
            }
            if (menu_label(VIEW_SESSIONS,
                    (text = "Sessions & Activity", selected = SESSIONS_WIN.open))) {
                open_dock_window(SESSIONS_WIN)
            }
            if (menu_label(VIEW_GIT_CHANGELIST,
                    (text = "Git Changelist", selected = GIT_CHANGELIST_WIN.open))) {
                open_dock_window(GIT_CHANGELIST_WIN)
            }
            if (menu_label(VIEW_GIT_ACTIVITY,
                    (text = "Git Activity", selected = GIT_ACTIVITY_WIN.open))) {
                open_dock_window(GIT_ACTIVITY_WIN)
            }
            if (menu_label(VIEW_FILE_INSPECTOR,
                    (text = "File Inspector", selected = FILE_INSPECTOR_WIN.open))) {
                open_dock_window(FILE_INSPECTOR_WIN)
            }
            if (menu_label(VIEW_TERMINAL,
                    (text = "Terminal", selected = TERMINAL_WIN.open))) {
                open_dock_window(TERMINAL_WIN)
            }
            separator(VIEW_WINDOWS_SEPARATOR)
            menu_item(SHOW_EXITED, (text = "Show exited sessions"))
            separator(VIEW_ZOOM_SEPARATOR)
            if (menu_label(VIEW_ZOOM_IN, (text = "Zoom in", shortcut = "Ctrl++"))) zoom_by(5)
            if (menu_label(VIEW_ZOOM_OUT, (text = "Zoom out", shortcut = "Ctrl+-"))) zoom_by(-5)
            if (menu_label(VIEW_ZOOM_RESET, (text = "Reset zoom", shortcut = "Ctrl+0"))) {
                g_zoom_percent = 100
            }
            if (menu_label(VIEW_SCROLL_BOTTOM, (text = "Scroll terminal to bottom"))) {
                g_terminal_view.scroll_offset = 0
            }
            separator(VIEW_LAYOUT_SEPARATOR)
            if (menu_label(VIEW_RESET_LAYOUT, (text = "Reset layout"))) reset_layout()
        }
        menu(SETTINGS_MENU, (text = "Settings", enabled = true)) {
            if (menu_label(OPEN_SETTINGS, (text = "Open Settings"))) {
                open_dock_window(SETTINGS_WIN)
            }
        }
    }
}

def private draw_repositories_window() {
    text_disabled("Target: Local")
    separator()
    input_text_growable(ADD_REPOSITORY_PATH, (text = "Repository checkout"))
    same_line()
    if (button(ADD_REPOSITORY, (text = "Add")) && g_client != null) {
        g_client->add_repository(ADD_REPOSITORY_PATH.value)
    }
    separator()
    var row = 0
    var repository_row = 0
    if (g_client != null) {
        for (repository in g_client.repositories) {
            let refreshing = repository.refreshing ? "  refreshing..." : ""
            text(REPOSITORY_TITLE[repository_row],
                (text = "{repository.record.display_name}  [Local]{refreshing}"))
            if (!empty(repository.error)) {
                text_colored(REPOSITORY_ERROR[repository_row], (color = float4(1.0f, 0.35f, 0.30f, 1.0f),
                    text = repository.error))
            }
            for (worktree in repository.worktrees) {
                var branch = worktree.branch
                if (worktree.detached) {
                    branch = "detached {slice(worktree.head, 0, min(8, length(worktree.head)))}"
                }
                let prefix = worktree.is_main ? "MAIN  " : "      "
                let dirty = worktree.staged + worktree.unstaged + worktree.untracked + worktree.conflicted
                let dirty_label = dirty > 0 ? "  *{dirty}" : ""
                let row_label = "{prefix}{branch}{dirty_label}##{repository.record.id}:{worktree.path}"
                if (selectable(WORKTREE_ROW[row], (text = row_label))) {
                    g_selected_repository_id = repository.record.id
                    g_selected_worktree_path = worktree.path
                    g_client->refresh_repository(repository.record.id)
                    g_client->request_repository_review(
                        repository.record.id, worktree.path)
                    GIT_CHANGELIST_WIN.open = true
                    GIT_CHANGELIST_WIN.pending_raise = true
                    GIT_ACTIVITY_WIN.open = true
                }
                WORKTREE_ROW[row].value = (repository.record.id == g_selected_repository_id && worktree.path == g_selected_worktree_path)
                if (worktree.is_main) separator(WORKTREE_SEPARATOR[row])
                row++
            }
            if (empty(repository.worktrees)) {
                text_disabled(REPOSITORY_EMPTY[repository_row],
                    (text = "Waiting for Git worktree inventory..."))
            }
            separator(REPOSITORY_SEPARATOR[repository_row])
            repository_row++
        }
    }
    if (g_client == null || empty(g_client.repositories)) text_disabled("No repositories registered")
}

def private git_status_byte(code : string; index : int) : int {
    if (index < 0 || index >= length(code)) return ' '
    var result = ' '
    peek_data(code) $(data) {
        result = int(data[index])
    }
    return result
}

def private git_conflict(code : string) : bool {
    return (code == "DD" || code == "AU" || code == "UD" || code == "UA"
        || code == "DU" || code == "AA" || code == "UU")
}

def private git_file_in_group(file : GitFileState const; group : int) : bool {
    if (group == 0) return git_conflict(file.code)
    if (git_conflict(file.code)) return false
    if (group == 1) return file.code != "??" && git_status_byte(file.code, 0) != ' '
    if (group == 2) return file.code != "??" && git_status_byte(file.code, 1) != ' '
    return file.code == "??"
}

def private git_group_scope(group : int) : string {
    return group == 1 ? "staged" : "working"
}

def private draw_git_file_group(repository : RepositorySnapshot const;
                                worktree : WorktreeState const;
                                group : int; var file_row : int&) : int {
    var count = 0
    for (file in repository.files) {
        if (file.worktree_path == worktree.path && git_file_in_group(file, group)) count++
    }
    if (count == 0) return 0
    let title = group == 0 ? "CONFLICTS" : (group == 1 ? "STAGED"
        : (group == 2 ? "MODIFIED" : "UNTRACKED"))
    text(GIT_GROUP_TITLE[group], (text = "{title}  {count}"))
    separator(GIT_GROUP_SEPARATOR[group])
    let comparison = git_group_scope(group)
    for (file in repository.files) {
        if (file.worktree_path != worktree.path || !git_file_in_group(file, group)) continue
        GIT_FILE_ROW[file_row].value = (g_inspector.repository_id == repository.record.id
            && g_inspector.worktree_path == worktree.path
            && g_inspector.file_path == file.path
            && g_inspector.comparison == comparison)
        let old_label = !empty(file.old_path) ? "  (from {file.old_path})" : ""
        if (selectable(GIT_FILE_ROW[file_row],
                (text = "{file.code}  {file.path}{old_label}##{comparison}:{file.path}"))) {
            if (g_client != null) {
                g_client->inspect_file(repository.record.id, worktree.path,
                    file.path, comparison)
                FILE_INSPECTOR_WIN.open = true
                FILE_INSPECTOR_WIN.pending_raise = true
            }
        }
        file_row++
    }
    dummy(GIT_GROUP_SPACING[group], (size = float2(0.0f, GetStyle().ItemSpacing.y)))
    return count
}

def private git_status_file_code(repository : RepositorySnapshot const;
                                 worktree_path : string;
                                 file_path : string) : string {
    for (file in repository.files) {
        if (file.worktree_path == worktree_path && file.path == file_path) {
            return file.code
        }
    }
    return ""
}

def private git_line_stats(added_lines, deleted_lines : int;
                           stats_known, binary : bool) : string {
    if (binary) return "binary"
    if (!stats_known) return "+?  -?"
    return "+{added_lines}  -{deleted_lines}"
}

def private service_sha_copy() {
    if (empty(g_pending_clipboard_sha)) return
    if (g_clipboard_copy_attempts >= 8) {
        g_pending_clipboard_sha = ""
        return
    }
    g_clipboard_copy_attempts++
    g_clipboard_copy_status = clipboard_set_text(g_pending_clipboard_sha)
    if (g_clipboard_copy_status != ClipboardStatus.ok) return
    var copied_text = ""
    g_clipboard_copy_status = clipboard_get_text(copied_text)
    if (g_clipboard_copy_status == ClipboardStatus.ok
            && copied_text == g_pending_clipboard_sha) {
        g_last_copied_sha = g_pending_clipboard_sha
        g_pending_clipboard_sha = ""
        g_clipboard_copy_verified = true
    }
}

def private queue_sha_copy(hash : string) {
    g_pending_clipboard_sha = hash
    g_clipboard_copy_attempts = 0
    g_clipboard_copy_verified = false
    service_sha_copy()
}

def private draw_git_path_text(path : string; p0, p1 : float2;
                               subdued : bool) {
    let alpha = subdued ? 0.55f : 1.0f
    let directory_color = GetColorU32(float4(0.43f, 0.64f, 0.82f, alpha))
    let separator_color = GetColorU32(float4(0.78f, 0.55f, 0.30f, alpha))
    let name_color = GetColorU32(float4(0.90f, 0.91f, 0.93f, alpha))
    var inscope components <- split(replace(path, "\\", "/"), "/")
    var position = p0
    with_window_drawlist() $(var dl) {
        dl |> with_drawlist_clip_rect(p0, p1, true) {
            for (index in range(length(components))) {
                let component = components[index]
                let color = index + 1 == length(components) ? name_color : directory_color
                dl |> add_text(position, color, component)
                position.x += CalcTextSize(component).x
                if (index + 1 < length(components)) {
                    dl |> add_text(position, separator_color, "/")
                    position.x += CalcTextSize("/").x
                }
            }
        }
    }
}

def private draw_git_stats_text(position : float2;
                                added_lines, deleted_lines : int;
                                stats_known, binary, subdued : bool) {
    let alpha = subdued ? 0.55f : 1.0f
    with_window_drawlist() $(var dl) {
        if (binary || !stats_known) {
            let stats = binary ? "binary" : "+?  -?"
            dl |> add_text(position, GetColorU32(ImGuiCol.TextDisabled), stats)
        } else {
            let added = "+{added_lines}"
            let deleted = "-{deleted_lines}"
            dl |> add_text(position,
                GetColorU32(float4(0.42f, 0.82f, 0.42f, alpha)), added)
            dl |> add_text(float2(position.x + CalcTextSize(added).x
                    + GetStyle().ItemSpacing.x, position.y),
                GetColorU32(float4(0.37f, 0.65f, 0.96f, alpha)), deleted)
        }
    }
}

def private git_visible_row_right(row_right : float) : float {
    let style & = GetStyle()
    let window_right = GetWindowPos().x + GetWindowSize().x - style.WindowPadding.x - style.ScrollbarSize
    return min(row_right, window_right)
}

def private decorate_git_file_row(row_id, path : string;
                                  added_lines, deleted_lines : int;
                                  stats_known, binary, subdued : bool;
                                  action_glyph, action_tip : string;
                                  var action_clicked : bool&) {
    let stats = git_line_stats(added_lines, deleted_lines, stats_known, binary)
    let row_min = GetItemRectMin()
    let row_max = GetItemRectMax()
    let row_right = git_visible_row_right(row_max.x)
    let hovered = IsItemHovered()
    let icon_size = max(12.0f, GetTextLineHeight() - 3.0f)
    let icon_width = icon_size + GetStyle().FramePadding.x * 2.0f
    let action_x = row_right - icon_width
    let stats_size = CalcTextSize(stats)
    let stats_x = max(row_min.x, action_x - stats_size.x - GetStyle().ItemSpacing.x)
    let text_y = row_min.y + max(0.0f,
        (row_max.y - row_min.y - GetTextLineHeight()) * 0.5f)
    draw_git_path_text(path,
        float2(row_min.x + GetStyle().FramePadding.x, text_y),
        float2(stats_x - GetStyle().ItemSpacing.x, row_max.y), subdued)
    draw_git_stats_text(float2(stats_x, text_y), added_lines, deleted_lines,
        stats_known, binary, subdued)
    action_clicked = false
    if (!empty(action_glyph)) {
        action_clicked = icon_button_draw_at("##{row_id}:action", action_glyph,
            float2(action_x, row_min.y), icon_size, 1.0f, hovered)
        if (hovered && IsItemHovered()) {
            set_tooltip(GIT_HOVER_TOOLTIP, action_tip)
        }
    }
}

def private draw_pr_included_files(repository : RepositorySnapshot const;
                                   worktree : WorktreeState const;
                                   var file_row : int&) : int {
    var count = 0
    for (file in repository.review_files) {
        if (file.worktree_path == worktree.path) count++
    }
    for (file in repository.review_files) {
        if (file.worktree_path != worktree.path) continue
        let status_code = git_status_file_code(
            repository, worktree.path, file.path)
        let staged = (!empty(status_code) && status_code != "??"
            && git_status_byte(status_code, 0) != ' ')
        GIT_FILE_ROW[file_row].value = (g_inspector.repository_id == repository.record.id
            && g_inspector.worktree_path == worktree.path
            && g_inspector.file_path == file.path
            && g_inspector.comparison == "pr")
        let busy = g_client != null && g_client.git_action_loading
        var action_clicked = false
        let stats = git_line_stats(file.added_lines, file.deleted_lines,
            file.stats_known, file.binary)
        var selected = false
        with_item_flag(ImGuiItemFlags.None | ImGuiItemFlagsPrivate.AllowOverlap, true) {
            with_style((ImGuiCol.Text, float4(0.0f, 0.0f, 0.0f, 0.0f))) {
                selected = selectable(GIT_FILE_ROW[file_row],
                    (text = "{file.path}  {stats}##pr:{file.path}"))
            }
        }
        decorate_git_file_row("pr:{file.path}", file.path, file.added_lines,
            file.deleted_lines, file.stats_known, file.binary, false,
            staged && !busy ? "minus" : "",
            "Remove from the staged PR result", action_clicked)
        if (action_clicked && g_client != null) {
            g_client->git_file_action(repository.record.id,
                worktree.path, file.path, "unstage")
        } elif (selected) {
            if (g_client != null) {
                g_client->inspect_file(repository.record.id, worktree.path,
                    file.path, "pr")
                FILE_INSPECTOR_WIN.open = true
                FILE_INSPECTOR_WIN.pending_raise = true
            }
        }
        file_row++
    }
    return count
}

def private draw_pr_not_included_files(repository : RepositorySnapshot const;
                                       worktree : WorktreeState const;
                                       var file_row : int&) : int {
    var count = 0
    for (file in repository.files) {
        if (file.worktree_path != worktree.path) continue
        if (file.code == "??" || git_status_byte(file.code, 1) != ' ') count++
    }
    for (file in repository.files) {
        if (file.worktree_path != worktree.path
                || (file.code != "??" && git_status_byte(file.code, 1) == ' ')) continue
        GIT_FILE_ROW[file_row].value = (g_inspector.repository_id == repository.record.id
            && g_inspector.worktree_path == worktree.path
            && g_inspector.file_path == file.path
            && g_inspector.comparison == "working")
        let busy = g_client != null && g_client.git_action_loading
        var action_clicked = false
        let stats = git_line_stats(file.added_lines, file.deleted_lines,
            file.stats_known, file.binary)
        var selected = false
        with_item_flag(ImGuiItemFlags.None | ImGuiItemFlagsPrivate.AllowOverlap, true) {
            with_style((ImGuiCol.Text, float4(0.0f, 0.0f, 0.0f, 0.0f))) {
                selected = selectable(GIT_FILE_ROW[file_row],
                    (text = "{file.path}  {stats}##working:{file.path}"))
            }
        }
        decorate_git_file_row("working:{file.path}", file.path, file.added_lines,
            file.deleted_lines, file.stats_known, file.binary, true,
            busy ? "" : "check", "Add to the staged PR result",
            action_clicked)
        if (action_clicked && g_client != null) {
            g_client->git_file_action(repository.record.id,
                worktree.path, file.path, "stage")
        } elif (selected && g_client != null) {
            g_client->inspect_file(repository.record.id, worktree.path,
                file.path, "working")
            FILE_INSPECTOR_WIN.open = true
            FILE_INSPECTOR_WIN.pending_raise = true
        }
        file_row++
    }
    return count
}

def private draw_shared_pr_files(repository : RepositorySnapshot const;
                                 worktree : WorktreeState const) {
    var file_row = 0
    let not_included = draw_pr_not_included_files(
        repository, worktree, file_row)
    if (not_included > 0) {
        separator(GIT_GROUP_SEPARATOR[11])
    }
    let included = draw_pr_included_files(
        repository, worktree, file_row)
    if (included == 0 && not_included == 0 && !worktree.review_loading
            && empty(worktree.review_error)) {
        text_disabled("Nothing to send or stage")
    }
    if (worktree.review_loading) text_disabled("Refreshing worktree PR...")
    if (!empty(worktree.review_error)) {
        text_colored(GIT_ERROR,
            (color = float4(1.0f, 0.35f, 0.30f, 1.0f),
            text = worktree.review_error))
    }
}

def private draw_sha_copy_icon(hash, row_id : string) {
    let row_min = GetItemRectMin()
    let row_max = GetItemRectMax()
    let row_right = git_visible_row_right(row_max.x)
    let hovered = IsItemHovered()
    let icon_size = max(12.0f, GetTextLineHeight() - 3.0f)
    let icon_width = icon_size + GetStyle().FramePadding.x * 2.0f
    if (icon_button_draw_at("##{row_id}:copy-sha", "duplicate",
            float2(row_right - icon_width, row_min.y), icon_size, 1.0f,
            hovered)) {
        queue_sha_copy(hash)
    }
    if (hovered && IsItemHovered()) {
        set_tooltip(GIT_HOVER_TOOLTIP, "Copy full SHA")
    }
}

def private draw_pr_review(repository : RepositorySnapshot const;
                           worktree : WorktreeState const) {
    var commit_count = 0
    for (commit in repository.review_commits) {
        if (commit.worktree_path == worktree.path) commit_count++
    }
    let base = ((!empty(worktree.review_base_ref))
        ? worktree.review_base_ref : "detecting base...")
    text(GIT_GROUP_TITLE[12],
        (text = "OUTGOING COMMITS  {commit_count}   -> {base}"))
    separator(GIT_GROUP_SEPARATOR[12])
    if (worktree.review_loading) text_disabled("Reading outgoing commits...")
    if (!empty(worktree.review_error)) {
        text_colored(GIT_ERROR,
            (color = float4(1.0f, 0.35f, 0.30f, 1.0f),
            text = worktree.review_error))
    }
    if (commit_count == 0) {
        if (!worktree.review_loading && empty(worktree.review_error)) {
            text_disabled("No outgoing commits")
        }
        return
    }
    var commit_row = 0
    let table_flags = (ImGuiTableFlags.Resizable | ImGuiTableFlags.RowBg |
        ImGuiTableFlags.BordersInnerV | ImGuiTableFlags.ScrollY |
        ImGuiTableFlags.SizingStretchProp)
    data_table(GIT_PR_COMMIT_TABLE, (text = "##pr-commits", columns = 3,
            flags = table_flags, outer_size = float2(0.0f, 0.0f),
            inner_width = 0.0f)) {
        table_setup_scroll_freeze(0, 1)
        table_setup_column("SHA", ImGuiTableColumnFlags.WidthFixed, 112.0f)
        table_setup_column("Title", ImGuiTableColumnFlags.WidthStretch, 0.72f)
        table_setup_column("Author", ImGuiTableColumnFlags.WidthStretch, 0.28f)
        table_headers_row()
        for (commit in repository.review_commits) {
            if (commit.worktree_path != worktree.path) continue
            table_next_row()
            table_set_column_index(0)
            with_style((ImGuiCol.Text, float4(0.36f, 0.76f, 0.96f, 1.0f))) {
                selectable_hover(GIT_PR_COMMIT_HOVER[commit_row],
                    (text = "{commit.short_hash}##pr:{commit.hash}",
                    flags = ImGuiSelectableFlags.SpanAllColumns |
                        ImGuiSelectableFlags.AllowOverlap))
            }
            draw_sha_copy_icon(commit.hash, "pr-commit:{commit.hash}")
            table_set_column_index(1)
            text_colored(GIT_COMMIT_TITLE_TEXT[commit_row],
                (color = float4(0.91f, 0.93f, 0.96f, 1.0f),
                text = commit.subject))
            table_set_column_index(2)
            text_colored(GIT_COMMIT_AUTHOR_TEXT[commit_row],
                (color = float4(0.64f, 0.70f, 0.80f, 1.0f),
                text = commit.author))
            commit_row++
        }
    }
}

def private ensure_history(repository : RepositorySnapshot const;
                           worktree : WorktreeState const; ref_name : string) {
    if (g_client == null || empty(ref_name)) return
    let identity = "{repository.record.id}\n{worktree.path}\n{ref_name}"
    if (g_client.last_history_identity != identity) {
        g_client->request_repository_history(
            repository.record.id, worktree.path, ref_name)
    }
}

def private request_commit_files(repository : RepositorySnapshot const;
                                 worktree : WorktreeState const;
                                 commit_hash : string) {
    if (g_client == null || empty(commit_hash)) return
    g_requested_history_commit = commit_hash
    g_client->request_repository_commit_files(
        repository.record.id, worktree.path, commit_hash)
}

def private draw_commit_files(repository : RepositorySnapshot const;
                              worktree : WorktreeState const) {
    if (!empty(g_requested_history_commit)
            && g_requested_history_commit == worktree.history_commit) {
        g_requested_history_commit = ""
    }
    let selected_commit = !empty(g_requested_history_commit) ? g_requested_history_commit : worktree.history_commit
    var count = 0
    for (file in repository.commit_files) {
        if (file.worktree_path == worktree.path
                && file.commit_hash == selected_commit) count++
    }
    text(GIT_GROUP_TITLE[21],
        (text = "FILES IN {selected_commit}  {count}"))
    separator(GIT_GROUP_SEPARATOR[21])
    if (!empty(g_requested_history_commit)) {
        text_disabled("Loading selected commit files...")
        return
    }
    var row = 0
    for (file in repository.commit_files) {
        if (file.worktree_path != worktree.path
                || file.commit_hash != selected_commit) continue
        GIT_COMMIT_FILE_ROW[row].value = (g_inspector.repository_id == repository.record.id
            && g_inspector.worktree_path == worktree.path
            && g_inspector.file_path == file.path
            && g_inspector.comparison == "commit"
            && g_inspector.git_revision == file.commit_hash)
        let stats = git_line_stats(file.added_lines, file.deleted_lines,
            file.stats_known, file.binary)
        var selected = false
        with_style((ImGuiCol.Text, float4(0.0f, 0.0f, 0.0f, 0.0f))) {
            selected = selectable(GIT_COMMIT_FILE_ROW[row],
                (text = "{file.path}  {stats}##commit:{file.commit_hash}:{file.path}"))
        }
        var ignored_action = false
        decorate_git_file_row("commit:{file.commit_hash}:{file.path}",
            file.path, file.added_lines, file.deleted_lines,
            file.stats_known, file.binary, false, "", "", ignored_action)
        if (selected && g_client != null) {
            g_client->inspect_file(repository.record.id, worktree.path,
                file.path, "commit", "compact", file.commit_hash)
            FILE_INSPECTOR_WIN.open = true
            FILE_INSPECTOR_WIN.pending_raise = true
        }
        row++
    }
    if (count == 0 && !worktree.history_loading
            && empty(worktree.history_error)) {
        text_disabled("Select a commit to inspect its files")
    }
}

def private draw_history_commits(repository : RepositorySnapshot const;
                                 worktree : WorktreeState const;
                                 heading : string) {
    var inscope commit_indices : array<int>
    commit_indices |> reserve(length(repository.history_commits))
    for (index in range(length(repository.history_commits))) {
        if (repository.history_commits[index].worktree_path == worktree.path) {
            commit_indices |> push(index)
        }
    }
    let count = length(commit_indices)
    text(GIT_GROUP_TITLE[20], (text = "{heading}  {count}"))
    separator(GIT_GROUP_SEPARATOR[20])
    if (worktree.history_loading) text_disabled("Reading Git history...")
    if (!empty(worktree.history_error)) {
        text_colored(GIT_ERROR,
            (color = float4(1.0f, 0.35f, 0.30f, 1.0f),
            text = worktree.history_error))
    }
    if (count == 0) return
    let table_flags = (ImGuiTableFlags.Resizable | ImGuiTableFlags.RowBg |
        ImGuiTableFlags.BordersInnerV | ImGuiTableFlags.ScrollY |
        ImGuiTableFlags.SizingStretchProp)
    data_table(GIT_HISTORY_COMMIT_TABLE, (text = "##history-commits", columns = 3,
            flags = table_flags, outer_size = float2(0.0f, 0.0f),
            inner_width = 0.0f)) {
        table_setup_scroll_freeze(0, 1)
        table_setup_column("SHA", ImGuiTableColumnFlags.WidthFixed, 112.0f)
        table_setup_column("Title", ImGuiTableColumnFlags.WidthStretch, 0.72f)
        table_setup_column("Author", ImGuiTableColumnFlags.WidthStretch, 0.28f)
        table_headers_row()
        list_clipper(count) $(start, end_) {
            for (row in range(start, end_)) {
                let commit = repository.history_commits[commit_indices[row]]
                let selected_commit = !empty(g_requested_history_commit) ? g_requested_history_commit : worktree.history_commit
                table_next_row()
                table_set_column_index(0)
                var selected = false
                with_style((ImGuiCol.Text, float4(0.36f, 0.76f, 0.96f, 1.0f))) {
                    selected = selectable_label(GIT_HISTORY_COMMIT_ROW[row],
                        (text = "{commit.short_hash}##history:{commit.hash}",
                        selected = commit.hash == selected_commit,
                        flags = ImGuiSelectableFlags.SpanAllColumns |
                            ImGuiSelectableFlags.AllowOverlap))
                }
                draw_sha_copy_icon(commit.hash, "history:{commit.hash}")
                table_set_column_index(1)
                text_colored(GIT_COMMIT_TITLE_TEXT[row],
                    (color = float4(0.91f, 0.93f, 0.96f, 1.0f),
                    text = commit.subject))
                table_set_column_index(2)
                text_colored(GIT_COMMIT_AUTHOR_TEXT[row],
                    (color = float4(0.64f, 0.70f, 0.80f, 1.0f),
                    text = commit.author))
                if (selected && g_client != null && commit.hash != selected_commit) {
                    request_commit_files(repository, worktree, commit.hash)
                }
            }
        }
    }
}

def private draw_history_review(repository : RepositorySnapshot const;
                                worktree : WorktreeState const) {
    ensure_history(repository, worktree, worktree.branch)
    draw_history_commits(repository, worktree,
        "COMMITS ON {worktree.history_ref}")
}

def private git_lane_index(lanes : array<string> const; hash : string) : int {
    for (index in range(length(lanes))) {
        if (lanes[index] == hash) return index
    }
    return -1
}

def private git_graph_color(hash : string; alpha : float = 1.0f) : uint {
    let slot = empty(hash) ? 0 : git_status_byte(hash, 0) % 6
    if (slot == 0) return GetColorU32(float4(0.33f, 0.72f, 0.95f, alpha))
    if (slot == 1) return GetColorU32(float4(0.94f, 0.53f, 0.37f, alpha))
    if (slot == 2) return GetColorU32(float4(0.53f, 0.82f, 0.43f, alpha))
    if (slot == 3) return GetColorU32(float4(0.76f, 0.55f, 0.94f, alpha))
    if (slot == 4) return GetColorU32(float4(0.96f, 0.76f, 0.31f, alpha))
    return GetColorU32(float4(0.34f, 0.83f, 0.75f, alpha))
}

def private git_prepare_lanes(commit : GitCommitState const;
                              observed : table<string; bool> const;
                              var lanes : array<string>) : int {
    var lane = git_lane_index(lanes, commit.hash)
    if (lane < 0) {
        lanes |> push(commit.hash)
        lane = length(lanes) - 1
    }
    var parent_number = 0
    for (parent in split(commit.parents, " ")) {
        if (empty(parent) || !key_exists(observed, parent)) continue
        if (parent_number > 0 && git_lane_index(lanes, parent) < 0) {
            lanes |> push(parent)
        }
        parent_number++
    }
    return lane
}

def private git_finish_lanes(commit : GitCommitState const; lane : int;
                             observed : table<string; bool> const;
                             var lanes : array<string>) {
    var first_parent = ""
    for (parent in split(commit.parents, " ")) {
        if (!empty(parent) && key_exists(observed, parent)) {
            first_parent = parent
            break
        }
    }
    if (empty(first_parent)) {
        lanes |> erase(lane)
    } else {
        lanes[lane] = first_parent
        var duplicate = length(lanes) - 1
        while (duplicate > lane) {
            if (lanes[duplicate] == first_parent) lanes |> erase(duplicate)
            duplicate--
        }
    }
}

def private git_advance_lanes(commit : GitCommitState const;
                              observed : table<string; bool> const;
                              var lanes : array<string>) {
    let lane = git_prepare_lanes(commit, observed, lanes)
    git_finish_lanes(commit, lane, observed, lanes)
}

def private draw_commit_graph_decoration(commit : GitCommitState const;
                                         observed : table<string; bool> const;
                                         var lanes : array<string>;
                                         row_id : string) {
    let lane = git_prepare_lanes(commit, observed, lanes)
    let row_min = GetItemRectMin()
    let row_max = GetItemRectMax()
    let row_right = git_visible_row_right(row_max.x)
    let center_y = (row_min.y + row_max.y) * 0.5f
    let lane_spacing = 14.0f
    let graph_x = row_min.x + 8.0f
    var inscope parents <- split(commit.parents, " ")
    with_window_drawlist() $(var dl) {
        for (index in range(length(lanes))) {
            let x = graph_x + float(index) * lane_spacing
            dl |> add_line(float2(x, row_min.y), float2(x, row_max.y),
                git_graph_color(lanes[index], 0.72f), 2.0f)
        }
        let node_x = graph_x + float(lane) * lane_spacing
        var parent_number = 0
        for (parent in parents) {
            if (empty(parent) || !key_exists(observed, parent)) continue
            if (parent_number > 0) {
                let parent_lane = git_lane_index(lanes, parent)
                let parent_x = graph_x + float(parent_lane) * lane_spacing
                dl |> add_line(float2(node_x, center_y),
                    float2(parent_x, row_max.y), git_graph_color(parent, 0.85f), 2.0f)
            }
            parent_number++
        }
        dl |> add_circle_filled(float2(node_x, center_y), 4.5f,
            git_graph_color(commit.hash), 12)
        dl |> add_circle(float2(node_x, center_y), 5.8f,
            GetColorU32(ImGuiCol.WindowBg), 12, 1.4f)
    }

    let hovered = IsItemHovered()
    let icon_size = max(12.0f, GetTextLineHeight() - 3.0f)
    let icon_width = icon_size + GetStyle().FramePadding.x * 2.0f
    if (icon_button_draw_at("##{row_id}:copy-sha", "duplicate",
            float2(row_right - icon_width, row_min.y), icon_size, 1.0f,
            hovered)) {
        queue_sha_copy(commit.hash)
    }
    if (hovered && IsItemHovered()) {
        set_tooltip(GIT_HOVER_TOOLTIP, "Copy full SHA")
    }

    git_finish_lanes(commit, lane, observed, lanes)
}

def private git_graph_refs(repository : RepositorySnapshot const;
                           worktree : WorktreeState const;
                           commit : GitCommitState const) : string {
    var inscope names : array<string>
    names |> reserve(3)
    var count = 0
    for (ref in repository.refs) {
        if (ref.worktree_path != worktree.path || ref.hash != commit.hash
                || ends_with(ref.name, "/HEAD") || count >= 3) continue
        names |> push("[{ref.name}]")
        count++
    }
    return join(names, " ")
}

def private draw_commit_graph(repository : RepositorySnapshot const;
                              worktree : WorktreeState const) {
    var inscope lanes : array<string>
    var inscope commit_indices : array<int>
    var inscope observed : table<string; bool>
    commit_indices |> reserve(length(repository.history_commits))
    for (index in range(length(repository.history_commits))) {
        if (repository.history_commits[index].worktree_path == worktree.path) {
            commit_indices |> push(index)
            observed[repository.history_commits[index].hash] = true
        }
    }
    var graph_cursor = 0
    list_clipper(length(commit_indices)) $(start, end_) {
        while (graph_cursor < start) {
            git_advance_lanes(repository.history_commits[
                commit_indices[graph_cursor]], observed, lanes)
            graph_cursor++
        }
        for (row in range(start, end_)) {
            let commit = repository.history_commits[commit_indices[row]]
            let selected_commit = !empty(g_requested_history_commit) ? g_requested_history_commit : worktree.history_commit
            let refs = git_graph_refs(repository, worktree, commit)
            let ref_text = empty(refs) ? "" : "{refs}  "
            let space_width = max(1.0f, CalcTextSize(" ").x)
            let graph_pad = repeat(" ", int(56.0f / space_width) + 1)
            var selected = false
            with_item_flag(ImGuiItemFlags.None | ImGuiItemFlagsPrivate.AllowOverlap, true) {
                selected = selectable_label(GIT_HISTORY_COMMIT_ROW[row],
                    (text = "{graph_pad}{commit.short_hash}  {ref_text}{commit.subject}  - {commit.author}##graph:{commit.hash}",
                    selected = commit.hash == selected_commit))
            }
            draw_commit_graph_decoration(commit, observed, lanes,
                "graph:{commit.hash}")
            if (selected && g_client != null
                    && commit.hash != selected_commit) {
                request_commit_files(repository, worktree, commit.hash)
            }
            graph_cursor++
        }
    }
    if (worktree.history_loading) text_disabled("Reading commit graph...")
    if (!empty(worktree.history_error)) {
        text_colored(GIT_ERROR,
            (color = float4(1.0f, 0.35f, 0.30f, 1.0f),
            text = worktree.history_error))
    }
}

def private draw_tree_review(repository : RepositorySnapshot const;
                             worktree : WorktreeState const) {
    if (g_client != null) {
        let identity = "{repository.record.id}\n{worktree.path}"
        if (g_client.last_refs_identity != identity) {
            g_client->request_repository_refs(repository.record.id, worktree.path)
        }
    }
    if (worktree.refs_loading) text_disabled("Reading local and remote branches...")
    if (!empty(worktree.refs_error)) {
        text_colored(GIT_ERROR,
            (color = float4(1.0f, 0.35f, 0.30f, 1.0f),
            text = worktree.refs_error))
    }
    ensure_history(repository, worktree, "__all__")
    draw_history_commits(repository, worktree, "COMMITS ACROSS WORKTREE REFS")
}

def private draw_git_changelist_window() {
    var found = false
    if (g_client != null) {
        for (repository in g_client.repositories) {
            if (repository.record.id != g_selected_repository_id) continue
            for (worktree in repository.worktrees) {
                if (worktree.path != g_selected_worktree_path) continue
                found = true
                let title = worktree.is_main ? "MAIN - {worktree.branch}" : worktree.branch
                text(GIT_TITLE, (text = title))
                same_line()
                if (small_button(REFRESH_GIT, (text = repository.refreshing ? "Refreshing..." : "Refresh",
                        enabled = !repository.refreshing))) {
                    g_client->refresh_repository(repository.record.id)
                    g_client->request_repository_review(
                        repository.record.id, worktree.path)
                    g_client->request_repository_history(
                        repository.record.id, worktree.path, worktree.branch)
                    g_client->request_repository_refs(
                        repository.record.id, worktree.path)
                }
                text_disabled(GIT_WORKTREE_PATH, (text = worktree.path))
                let status_summary = "staged {worktree.staged}   unstaged {worktree.unstaged}   untracked {worktree.untracked}   conflicted {worktree.conflicted}"
                text(GIT_STATUS_SUMMARY, (text = status_summary))
                if (worktree.ahead != 0 || worktree.behind != 0) {
                    text(GIT_AHEAD_BEHIND,
                        (text = "ahead {worktree.ahead}   behind {worktree.behind}"))
                }
                if (!empty(repository.error)) {
                    text_colored(GIT_ERROR, (color = float4(1.0f, 0.35f, 0.30f, 1.0f),
                        text = repository.error))
                }
                separator()
                if (GIT_HISTORY_TAB.selected || GIT_TREE_TAB.selected) {
                    draw_commit_files(repository, worktree)
                } else {
                    draw_shared_pr_files(repository, worktree)
                }
            }
        }
    }
    if (!found) text_disabled("Select a worktree to inspect its changelist")
}

def private draw_git_activity_window() {
    var found = false
    if (g_client != null) {
        for (repository in g_client.repositories) {
            if (repository.record.id != g_selected_repository_id) continue
            for (worktree in repository.worktrees) {
                if (worktree.path != g_selected_worktree_path) continue
                found = true
                tab_bar(GIT_PERSPECTIVE_TABS,
                        (text = "Git perspectives", flags = ImGuiTabBarFlags.None)) {
                    tab_item(GIT_PR_TAB, (text = "PR", closable = false,
                            flags = ImGuiTabItemFlags.None)) {
                        draw_pr_review(repository, worktree)
                    }
                    tab_item(GIT_HISTORY_TAB, (text = "History", closable = false,
                            flags = ImGuiTabItemFlags.None)) {
                        draw_history_review(repository, worktree)
                    }
                    tab_item(GIT_TREE_TAB, (text = "Tree", closable = false,
                            flags = ImGuiTabItemFlags.None)) {
                        draw_tree_review(repository, worktree)
                    }
                }
            }
        }
    }
    if (!found) text_disabled("Select a worktree to inspect Git activity")
}

def private source_view_style(diff_background : bool = false) : TextSourceViewStyle {
    return TextSourceViewStyle(
        fonts = g_fonts,
        zoom = float(g_zoom_percent) / 100.0f,
        wrap = false,
        line_mark_background_alpha = diff_background ? 0.11f : 0.0f,
        line_mark_width = 2.0f,
        write_system_clipboard = !harness_is_headless())
}

def private syntax_cell_count(state : TextSourceViewState const) : int {
    var count = 0
    for (cell in state.cells) {
        if (cell.style_id > 1) count++
    }
    return count
}

def private syntax_style_cell_count(state : TextSourceViewState const;
                                    style : TextSourceSyntaxStyle) : int {
    var count = 0
    let style_id = 1 + int(style)
    for (cell in state.cells) {
        if (cell.style_id == style_id) count++
    }
    return count
}

def private inspector_request_state_name(state : InspectorRequestState) : string {
    if (state == InspectorRequestState.queued) return "queued"
    if (state == InspectorRequestState.running) return "running"
    return "idle"
}

def private inspector_content_state_name(state : InspectorContentState) : string {
    if (state == InspectorContentState.ready) return "ready"
    if (state == InspectorContentState.failed) return "failed"
    return "absent"
}

def private draw_binary_preview() {
    text("Binary file ({g_inspector.preview_byte_count} bytes)")
    if (g_inspector.preview_texture == 0u) {
        text_disabled(INSPECTOR_BINARY_UNAVAILABLE,
            (text = empty(g_inspector.preview_error)
                ? "Preview unavailable"
                : "Preview unavailable: {g_inspector.preview_error}"))
        return
    }
    text_disabled(INSPECTOR_BINARY_DIMENSIONS,
        (text = "Decoded by the bundled stb_image: " +
            "{g_inspector.preview_width} x {g_inspector.preview_height}"))
    let available = GetContentRegionAvail()
    let scale = min(1.0f, min(
        available.x / max(1.0f, float(g_inspector.preview_width)),
        available.y / max(1.0f, float(g_inspector.preview_height))))
    unsafe {
        image(INSPECTOR_BINARY_PREVIEW,
            (user_texture_id = inspector_preview_texture_ref(
                g_inspector.preview_texture),
             size = float2(max(1.0f, float(g_inspector.preview_width) * scale),
                           max(1.0f, float(g_inspector.preview_height) * scale)),
             uv0 = float2(0.0f, 0.0f), uv1 = float2(1.0f, 1.0f),
             tint_col = float4(1.0f, 1.0f, 1.0f, 1.0f),
             border_col = float4(0.35f, 0.35f, 0.35f, 1.0f)))
    }
}

def private draw_side_by_side_diff() {
    if (g_inspector.parsed.binary) {
        draw_binary_preview()
        return
    }
    let hunk_count = length(g_inspector.parsed.hunks)
    g_inspector.diff_current_hunk = hunk_count > 0 ? clamp(g_inspector.diff_current_hunk, 0, hunk_count - 1) : 0
    SetNextItemWidth(150.0f)
    combo(DIFF_CONTEXT,
        (text = "Context", items <- INSPECTOR_CONTEXT_LABELS))
    let selected_context = clamp(DIFF_CONTEXT.value, 0,
        length(INSPECTOR_CONTEXT_VALUES) - 1)
    if (selected_context != g_inspector.diff_context_choice
            && g_client != null && g_client.connected) {
        g_client->inspect_file(g_inspector.repository_id,
            g_inspector.worktree_path, g_inspector.file_path,
            g_inspector.comparison, INSPECTOR_CONTEXT_VALUES[selected_context],
            g_inspector.git_revision)
    }
    same_line()
    SetNextItemWidth(150.0f)
    combo(DIFF_SYNTAX, (text = "Syntax", items <- INSPECTOR_SYNTAX_LABELS))
    let selected_syntax = clamp(DIFF_SYNTAX.value, 0,
        length(INSPECTOR_SYNTAX_VALUES) - 1)
    if (selected_syntax != g_inspector.syntax_choice) {
        g_inspector.syntax_choice = selected_syntax
        g_inspector.resolved_language = inspector_language_for_path(
            g_inspector.file_path,
            INSPECTOR_SYNTAX_VALUES[g_inspector.syntax_choice])
        if (!empty(g_inspector.source_patch)) {
            queue_inspector_prepare("Applying syntax...")
        }
    }
    same_line()
    if (small_button(DIFF_PREVIOUS_CHANGE, (text = "Previous change",
            enabled = g_inspector.diff_current_hunk > 0))) {
        g_inspector.diff_current_hunk = max(0,
            g_inspector.diff_current_hunk - 1)
        if (g_inspector.diff_current_hunk < hunk_count) {
            let row = g_inspector.parsed.hunks[g_inspector.diff_current_hunk].row_start
            g_inspector.diff_scroll_y = float(row) * max(
                GetTextLineHeight(), g_inspector.new_view.key.line_height)
            g_inspector.diff_scroll_jump_pending = true
        }
    }
    same_line()
    if (small_button(DIFF_NEXT_CHANGE, (text = "Next change",
            enabled = g_inspector.diff_current_hunk + 1 < hunk_count))) {
        g_inspector.diff_current_hunk = min(hunk_count - 1,
            g_inspector.diff_current_hunk + 1)
        if (g_inspector.diff_current_hunk >= 0
                && g_inspector.diff_current_hunk < hunk_count) {
            let row = g_inspector.parsed.hunks[g_inspector.diff_current_hunk].row_start
            g_inspector.diff_scroll_y = float(row) * max(
                GetTextLineHeight(), g_inspector.new_view.key.line_height)
            g_inspector.diff_scroll_jump_pending = true
        }
    }
    same_line()
    text_disabled(DIFF_CHANGE_POSITION, (text = hunk_count > 0
        ? "Change {g_inspector.diff_current_hunk + 1} of {hunk_count}"
        : "No text changes"))

    let available = GetContentRegionAvail()
    let splitter_width = 8.0f
    let usable_width = max(2.0f, available.x - splitter_width)
    var left_width = usable_width * clamp(
        g_inspector.diff_split_fraction, 0.12f, 0.88f)
    var right_width = usable_width - left_width
    let origin = GetCursorScreenPos()
    let split_min = min(120.0f, usable_width * 0.4f)
    let split_box = float4(origin.x + left_width, origin.y,
        origin.x + left_width + splitter_width, origin.y + available.y)
    let splitting = SplitterBehavior(split_box, GetID("git_diff_splitter"),
        ImGuiAxis.X, left_width, right_width, split_min, split_min)
    g_inspector.diff_split_fraction = left_width / max(1.0f, left_width + right_width)

    child(DIFF_LEFT_PANE, (text = "before", size = float2(left_width, available.y),
            child_flags = ImGuiChildFlags.Borders,
            window_flags = ImGuiWindowFlags.NoScrollbar)) {
        let window_pos = GetWindowPos()
        let window_size = GetWindowSize()
        g_inspector.diff_left_rect = float4(window_pos.x, window_pos.y,
            window_pos.x + window_size.x, window_pos.y + window_size.y)
        let hovered = IsWindowHovered(ImGuiHoveredFlags.AllowWhenBlockedByActiveItem)
        if (g_inspector.diff_scroll_jump_pending || !hovered) {
            SetScrollY(g_inspector.diff_scroll_y)
        } else {
            g_inspector.diff_scroll_y = GetScrollY()
        }
        text_disabled("BEFORE")
        separator()
        var style = source_view_style(true)
        let _ = text_source_view("git diff before", g_inspector.old_document,
            g_inspector.old_view, style)
    }
    same_line((spacing = 0.0f))
    dummy(DIFF_SPLITTER_SPACE, (size = float2(splitter_width, available.y)))
    let splitter_hot = IsItemHovered() || splitting
    if (splitter_hot) SetMouseCursor(ImGuiMouseCursor.ResizeEW)
    let splitter_color = GetColorU32(splitter_hot
        ? ImGuiCol.SeparatorActive : ImGuiCol.Separator)
    with_window_drawlist() $(var dl) {
        dl |> add_line(
            float2(origin.x + left_width + splitter_width * 0.5f, origin.y),
            float2(origin.x + left_width + splitter_width * 0.5f,
                origin.y + available.y), splitter_color, 1.0f)
    }
    same_line((spacing = 0.0f))
    child(DIFF_RIGHT_PANE, (text = "after", size = float2(right_width, available.y),
            child_flags = ImGuiChildFlags.Borders,
            window_flags = ImGuiWindowFlags.HorizontalScrollbar)) {
        let window_pos = GetWindowPos()
        let window_size = GetWindowSize()
        let scrollbar_width = GetStyle().ScrollbarSize
        g_inspector.diff_right_rect = float4(window_pos.x, window_pos.y,
            window_pos.x + window_size.x, window_pos.y + window_size.y)
        g_inspector.diff_scrollbar_rect = float4(
            window_pos.x + window_size.x - scrollbar_width, window_pos.y,
            window_pos.x + window_size.x, window_pos.y + window_size.y)
        let hovered = IsWindowHovered(ImGuiHoveredFlags.AllowWhenBlockedByActiveItem)
        if (g_inspector.diff_scroll_jump_pending || !hovered) {
            SetScrollY(g_inspector.diff_scroll_y)
        } else {
            g_inspector.diff_scroll_y = GetScrollY()
        }
        text_disabled("AFTER")
        separator()
        var style = source_view_style(true)
        let _ = text_source_view("git diff after", g_inspector.new_document,
            g_inspector.new_view, style)
    }
    g_inspector.diff_scroll_jump_pending = false
}

def private draw_file_view() {
    if (g_inspector.parsed.binary) {
        draw_binary_preview()
        return
    }
    child(FILE_VIEW_PANE, (text = "file view", size = float2(0.0f, 0.0f),
            child_flags = ImGuiChildFlags.None,
            window_flags = ImGuiWindowFlags.HorizontalScrollbar)) {
        if (g_inspector.resolved_language == "markdown") {
            var style = MarkdownViewStyle(
                fonts = g_fonts,
                typography = MarkdownTypography(
                    inline_code_scale = 0.84f, code_block_scale = 0.90f),
                zoom = float(g_zoom_percent) / 100.0f,
                write_system_clipboard = !harness_is_headless())
            let _ <- markdown_view_with_change_gutter("git markdown view",
                g_inspector.markdown_document, g_inspector.markdown_view,
                style, g_inspector.revision, g_inspector.changed_source_ranges,
                g_inspector.markdown_gutter_segments,
                max(1.0f, 2.0f * float(g_zoom_percent) / 100.0f))
        } else {
            var style = source_view_style(false)
            let _ = text_source_view("git file view", g_inspector.view_document,
                g_inspector.view, style)
        }
    }
}

[live_command(description = "Inspect watcher connectivity and rich-client repository/session counts.")]
def herder_client_state(_input : JsonValue?) : JsonValue? {
    if (g_client == null) return JV((ok = false, error = "client is unavailable"))
    return JV((ok = g_client.connected,
        connected = g_client.connected,
        error = g_client.error,
        session_count = length(g_client.sessions),
        repository_count = length(g_client.repositories),
        selected_repository_id = g_selected_repository_id,
        selected_worktree_path = g_selected_worktree_path))
}

[live_command(description = "Restore the rich client's default dock layout.")]
def herder_reset_layout(_input : JsonValue?) : JsonValue? {
    reset_layout()
    return JV((ok = true))
}

[live_command(description = "Select a worktree for Git review. Args: repository_id, worktree_path.")]
def herder_git_select_worktree(input : JsonValue?) : JsonValue? {
    if (g_client == null || !g_client.connected) {
        return JV((ok = false, error = "watcher is not connected"))
    }
    let args = from_JV(input, type<HerderGitWorktreeArgs>)
    for (repository in g_client.repositories) {
        if (repository.record.id != args.repository_id) continue
        for (worktree in repository.worktrees) {
            if (worktree.path != args.worktree_path) continue
            g_selected_repository_id = args.repository_id
            g_selected_worktree_path = args.worktree_path
            g_requested_history_commit = ""
            g_client->refresh_repository(args.repository_id)
            g_client->request_repository_review(
                args.repository_id, args.worktree_path)
            g_client->request_repository_history(
                args.repository_id, args.worktree_path, worktree.branch)
            g_client->request_repository_refs(
                args.repository_id, args.worktree_path)
            GIT_CHANGELIST_WIN.open = true
            GIT_CHANGELIST_WIN.pending_raise = true
            GIT_ACTIVITY_WIN.open = true
            return JV((ok = true, repository_id = args.repository_id,
                worktree_path = args.worktree_path))
        }
    }
    return JV((ok = false, error = "unknown repository or worktree"))
}

[live_command(description = "Inspect the selected worktree PR model, including outgoing commits, included files, and not-included files.")]
def herder_git_review_state(_input : JsonValue?) : JsonValue? {
    var commits : array<JsonValue?>
    var included : array<JsonValue?>
    var not_included : array<JsonValue?>
    if (g_client != null) {
        for (repository in g_client.repositories) {
            if (repository.record.id != g_selected_repository_id) continue
            for (worktree in repository.worktrees) {
                if (worktree.path != g_selected_worktree_path) continue
                for (commit in repository.review_commits) {
                    if (commit.worktree_path != worktree.path) continue
                    commits |> push(JV((hash = commit.hash,
                        short_hash = commit.short_hash,
                        authored_at = commit.authored_at,
                        author = commit.author, subject = commit.subject,
                        parents = commit.parents)))
                }
                for (file in repository.review_files) {
                    if (file.worktree_path != worktree.path) continue
                    let status_code = git_status_file_code(
                        repository, worktree.path, file.path)
                    included |> push(JV((code = file.code, path = file.path,
                        old_path = file.old_path,
                        added_lines = file.added_lines,
                        deleted_lines = file.deleted_lines,
                        stats_known = file.stats_known, binary = file.binary,
                        staged = (!empty(status_code) && status_code != "??"
                            && git_status_byte(status_code, 0) != ' '))))
                }
                for (file in repository.files) {
                    if (file.worktree_path != worktree.path
                            || (file.code != "??"
                                && git_status_byte(file.code, 1) == ' ')) continue
                    not_included |> push(JV((code = file.code,
                        path = file.path, old_path = file.old_path,
                        added_lines = file.added_lines,
                        deleted_lines = file.deleted_lines,
                        stats_known = file.stats_known, binary = file.binary)))
                }
                return JV((ok = empty(worktree.review_error),
                    repository_id = repository.record.id,
                    worktree_path = worktree.path,
                    branch = worktree.branch,
                    base_ref = worktree.review_base_ref,
                    merge_base = worktree.review_merge_base,
                    loading = worktree.review_loading,
                    error = worktree.review_error,
                    commits = commits, included = included,
                    not_included = not_included,
                    git_action_loading = g_client.git_action_loading,
                    git_action_file_path = g_client.git_action_file_path,
                    git_action_status = g_client.git_action_status))
            }
        }
    }
    return JV((ok = false, error = "selected worktree is unavailable"))
}

[live_command(description = "Inspect selected worktree History and Tree data, including refs, commits, and selected-commit files.")]
def herder_git_history_state(_input : JsonValue?) : JsonValue? {
    var refs : array<JsonValue?>
    var commits : array<JsonValue?>
    var files : array<JsonValue?>
    if (g_client != null) {
        for (repository in g_client.repositories) {
            if (repository.record.id != g_selected_repository_id) continue
            for (worktree in repository.worktrees) {
                if (worktree.path != g_selected_worktree_path) continue
                for (ref in repository.refs) {
                    if (ref.worktree_path != worktree.path) continue
                    refs |> push(JV((name = ref.name, full_name = ref.full_name,
                        hash = ref.hash, current = ref.current, remote = ref.remote)))
                }
                for (commit in repository.history_commits) {
                    if (commit.worktree_path != worktree.path) continue
                    commits |> push(JV((hash = commit.hash,
                        short_hash = commit.short_hash,
                        authored_at = commit.authored_at,
                        author = commit.author, subject = commit.subject,
                        parents = commit.parents)))
                }
                for (file in repository.commit_files) {
                    if (file.worktree_path != worktree.path
                            || file.commit_hash != worktree.history_commit) continue
                    files |> push(JV((commit_hash = file.commit_hash,
                        code = file.code, path = file.path,
                        old_path = file.old_path,
                        added_lines = file.added_lines,
                        deleted_lines = file.deleted_lines,
                        stats_known = file.stats_known, binary = file.binary)))
                }
                return JV((ok = empty(worktree.history_error)
                        && empty(worktree.refs_error),
                    repository_id = repository.record.id,
                    worktree_path = worktree.path,
                    branch = worktree.branch,
                    history_ref = worktree.history_ref,
                    history_commit = worktree.history_commit,
                    requested_history_commit = g_requested_history_commit,
                    history_loading = worktree.history_loading,
                    history_error = worktree.history_error,
                    refs_loading = worktree.refs_loading,
                    refs_error = worktree.refs_error,
                    refs = refs, commits = commits, files = files))
            }
        }
    }
    return JV((ok = false, error = "selected worktree is unavailable"))
}

[live_command(description = "Select a commit and load its changelist. Args: sha.")]
def herder_git_select_commit(input : JsonValue?) : JsonValue? {
    if (g_client == null || !g_client.connected) {
        return JV((ok = false, error = "watcher is not connected"))
    }
    let args = from_JV(input, type<HerderGitShaArgs>)
    for (repository in g_client.repositories) {
        if (repository.record.id != g_selected_repository_id) continue
        for (worktree in repository.worktrees) {
            if (worktree.path != g_selected_worktree_path) continue
            for (commit in repository.history_commits) {
                if (commit.worktree_path == worktree.path
                        && commit.hash == args.sha) {
                    request_commit_files(repository, worktree, commit.hash)
                    return JV((ok = true, sha = commit.hash))
                }
            }
        }
    }
    return JV((ok = false, error = "commit is not present in the observed history"))
}

[live_command(description = "Copy a Git SHA through the same verified clipboard path used by hover controls. Args: sha.")]
def herder_git_copy_sha(input : JsonValue?) : JsonValue? {
    let args = from_JV(input, type<HerderGitShaArgs>)
    if (empty(args.sha)) return JV((ok = false, error = "sha is empty"))
    queue_sha_copy(args.sha)
    return JV((ok = g_clipboard_copy_verified,
        sha = args.sha,
        status = int(g_clipboard_copy_status),
        pending = !empty(g_pending_clipboard_sha)))
}

[live_command(description = "Inspect the last Git SHA clipboard write without exposing unrelated clipboard contents.")]
def herder_git_clipboard_state(_input : JsonValue?) : JsonValue? {
    var copied_text = ""
    let read_status = clipboard_get_text(copied_text)
    let expected = !empty(g_pending_clipboard_sha) ? g_pending_clipboard_sha : g_last_copied_sha
    return JV((ok = g_clipboard_copy_verified && read_status == ClipboardStatus.ok
            && copied_text == expected,
        last_sha = g_last_copied_sha,
        pending_sha = g_pending_clipboard_sha,
        write_status = int(g_clipboard_copy_status),
        read_status = int(read_status),
        attempts = g_clipboard_copy_attempts,
        matches = read_status == ClipboardStatus.ok && copied_text == expected))
}

[live_command(description = "Switch Git Status perspective. Args: perspective = pr | history | tree.")]
def herder_git_perspective(input : JsonValue?) : JsonValue? {
    let args = from_JV(input, type<HerderGitPerspectiveArgs>)
    let perspective = to_lower(args.perspective)
    if (perspective == "pr") {
        GIT_PR_TAB.pending_select = true
    } elif (perspective == "history") {
        GIT_HISTORY_TAB.pending_select = true
    } elif (perspective == "tree") {
        GIT_TREE_TAB.pending_select = true
        if (g_client != null && g_client.connected
                && !empty(g_selected_repository_id)
                && !empty(g_selected_worktree_path)) {
            g_client->request_repository_history(g_selected_repository_id,
                g_selected_worktree_path, "__all__")
        }
    } else {
        return JV((ok = false, error = "perspective must be pr, history, or tree"))
    }
    GIT_CHANGELIST_WIN.open = true
    GIT_ACTIVITY_WIN.open = true
    GIT_ACTIVITY_WIN.pending_raise = true
    return JV((ok = true, perspective = perspective))
}

[live_command(description = "Stage or unstage one observed worktree file. Args: repository_id, worktree_path, file_path, action.")]
def herder_git_file_action(input : JsonValue?) : JsonValue? {
    if (g_client == null || !g_client.connected) {
        return JV((ok = false, error = "watcher is not connected"))
    }
    let args = from_JV(input, type<HerderGitFileActionArgs>)
    if (args.action != "stage" && args.action != "unstage") {
        return JV((ok = false, error = "action must be stage or unstage"))
    }
    if (g_client.git_action_loading) {
        return JV((ok = false, error = "another Git file action is running"))
    }
    g_client->git_file_action(args.repository_id, args.worktree_path,
        args.file_path, args.action)
    return JV((ok = true, request_id = g_client.git_action_request_id,
        action = args.action, file_path = args.file_path))
}

[live_command(description = "Inspect the selected Git file, comparison, render mode, and changed-line telemetry.")]
def herder_file_inspector_state(_input : JsonValue?) : JsonValue? {
    let view_selection = text_source_view_selection(g_inspector.view)
    return JV((
        ok = !empty(g_inspector.file_path) && !g_inspector.loading
            && empty(g_inspector.error),
        repository_id = g_inspector.repository_id,
        worktree_path = g_inspector.worktree_path,
        file_path = g_inspector.file_path,
        comparison = g_inspector.comparison,
        git_revision = g_inspector.git_revision,
        zoom_percent = g_zoom_percent,
        diff_context = g_inspector.diff_context,
        diff_context_choice = g_inspector.diff_context_choice,
        mode = g_inspector.mode == GitInspectorMode.diff ? "diff" : "view",
        loading = g_inspector.loading,
        prepare_request = inspector_request_state_name(g_inspector.prepare.request),
        prepare_outcome = inspector_content_state_name(
            g_inspector.prepare.latest_outcome),
        prepare_generation = g_inspector.prepare.generation,
        prepare_has_retained_content = g_inspector.prepare.has_retained_content,
        prepare_retained_generation = g_inspector.prepare.retained_generation,
        prepare_status = g_inspector.prepare.status,
        prepare_error = g_inspector.prepare.error,
        prepare_elapsed_usec = g_inspector.prepare_elapsed_usec,
        status = g_inspector.status,
        error = g_inspector.error,
        task_session_id = g_inspector.task_session_id,
        revision = g_inspector.revision,
        diff_row_count = length(g_inspector.parsed.rows),
        diff_hunk_count = length(g_inspector.parsed.hunks),
        diff_current_hunk = g_inspector.diff_current_hunk,
        diff_scroll_y = g_inspector.diff_scroll_y,
        diff_split_fraction = g_inspector.diff_split_fraction,
        diff_left_rect = g_inspector.diff_left_rect,
        diff_right_rect = g_inspector.diff_right_rect,
        diff_scrollbar_rect = g_inspector.diff_scrollbar_rect,
        syntax_choice = g_inspector.syntax_choice,
        resolved_language = g_inspector.resolved_language,
        binary = g_inspector.parsed.binary,
        preview_texture_ready = g_inspector.preview_texture != 0u,
        preview_width = g_inspector.preview_width,
        preview_height = g_inspector.preview_height,
        preview_byte_count = g_inspector.preview_byte_count,
        preview_error = g_inspector.preview_error,
        old_changed_line_count = length(g_inspector.parsed.old_changed_lines),
        new_changed_line_count = length(g_inspector.parsed.new_changed_lines),
        old_byte_count = length(g_inspector.parsed.old_text),
        new_byte_count = length(g_inspector.parsed.new_text),
        new_document_byte_count = length(g_inspector.new_document.source),
        new_document_utf8_valid = is_utf8_string_valid(
            g_inspector.new_document.source),
        view_byte_count = length(g_inspector.view_source),
        view_prepared = g_inspector.view_prepared,
        old_line_mark_count = length(g_inspector.old_document.line_marks),
        new_line_mark_count = length(g_inspector.new_document.line_marks),
        old_line_number_count = length(g_inspector.old_document.line_numbers),
        new_line_number_count = length(g_inspector.new_document.line_numbers),
        view_line_mark_count = length(g_inspector.view_document.line_marks),
        changed_source_range_count = length(g_inspector.changed_source_ranges),
        old_syntax_span_count = length(g_inspector.old_document.spans),
        new_syntax_span_count = length(g_inspector.new_document.spans),
        view_syntax_span_count = length(g_inspector.view_document.spans),
        old_syntax_cell_count = syntax_cell_count(g_inspector.old_view),
        new_syntax_cell_count = syntax_cell_count(g_inspector.new_view),
        view_syntax_cell_count = syntax_cell_count(g_inspector.view),
        view_virtual_mode = g_inspector.view.virtual_mode,
        view_visible_syntax_runs = g_inspector.view.virtual_visible_syntax_runs,
        view_selection_start = view_selection.start_byte,
        view_selection_end = view_selection.end_byte,
        view_selection_revision = g_inspector.view.selection_revision,
        view_selection_copy_revision = g_inspector.view.selection_copy_revision,
        old_keyword_cell_count = syntax_style_cell_count(
            g_inspector.old_view, TextSourceSyntaxStyle.keyword),
        new_keyword_cell_count = syntax_style_cell_count(
            g_inspector.new_view, TextSourceSyntaxStyle.keyword),
        old_type_cell_count = syntax_style_cell_count(
            g_inspector.old_view, TextSourceSyntaxStyle.type_),
        new_type_cell_count = syntax_style_cell_count(
            g_inspector.new_view, TextSourceSyntaxStyle.type_),
        old_function_cell_count = syntax_style_cell_count(
            g_inspector.old_view, TextSourceSyntaxStyle.function_),
        new_function_cell_count = syntax_style_cell_count(
            g_inspector.new_view, TextSourceSyntaxStyle.function_),
        old_render_revision = g_inspector.old_view.render_revision,
        new_render_revision = g_inspector.new_view.render_revision,
        view_render_revision = g_inspector.view.render_revision,
        markdown_render_revision = g_inspector.markdown_view.render_revision,
        markdown_current_layout_count = markdown_view_current_layout_count(
            g_inspector.markdown_view),
        markdown_changed_layout_count = markdown_view_changed_layout_count(
            g_inspector.markdown_document, g_inspector.markdown_view,
            g_inspector.changed_source_ranges),
        markdown_gutter_segments = g_inspector.markdown_gutter_segments))
}

[live_command(description = "Open a changed Git file immediately. Args: repository_id, worktree_path, file_path, comparison, optional revision for commit.")]
def herder_file_inspector_open(input : JsonValue?) : JsonValue? {
    if (g_client == null || !g_client.connected) {
        return JV((ok = false, error = "watcher is not connected"))
    }
    let args = from_JV(input, type<HerderInspectorOpenArgs>)
    if (empty(args.repository_id) || empty(args.worktree_path)
            || empty(args.file_path)) {
        return JV((ok = false,
            error = "repository_id, worktree_path, and file_path are required"))
    }
    let comparison = empty(args.comparison) ? "working" : args.comparison
    g_client->inspect_file(args.repository_id, args.worktree_path,
        args.file_path, comparison, args.diff_context, args.revision)
    FILE_INSPECTOR_WIN.open = true
    FILE_INSPECTOR_WIN.pending_raise = true
    return JV((ok = true, request_id = g_inspector.request_id,
        file_path = args.file_path, comparison = comparison,
        revision = args.revision))
}

[live_command(description = "Read exact inspector text. Args: side = before | after | view.")]
def herder_file_inspector_text(input : JsonValue?) : JsonValue? {
    let args = from_JV(input, type<HerderInspectorTextArgs>)
    let side = to_lower(args.side)
    if (side == "before" || side == "old") {
        return JV((ok = true, side = "before",
            text = g_inspector.parsed.old_text,
            line_mark_count = length(g_inspector.old_document.line_marks),
            syntax_span_count = length(g_inspector.old_document.spans)))
    }
    if (side == "after" || side == "new" || side == "view") {
        let is_view = side == "view"
        let source = is_view ? g_inspector.view_source : g_inspector.parsed.new_text
        return JV((ok = true, side = side == "view" ? "view" : "after",
            text = source,
            line_mark_count = is_view
                ? length(g_inspector.view_document.line_marks)
                : length(g_inspector.new_document.line_marks),
            syntax_span_count = is_view
                ? length(g_inspector.view_document.spans)
                : length(g_inspector.new_document.spans)))
    }
    return JV((ok = false, error = "side must be before, after, or view"))
}

[live_command(description = "Switch the Git file inspector between Diff and View without mouse input.")]
def herder_file_inspector_mode(input : JsonValue?) : JsonValue? {
    let args = from_JV(input, type<HerderInspectorModeArgs>)
    let mode = to_lower(args.mode)
    if (mode == "diff") {
        INSPECTOR_DIFF_TAB.pending_select = true
    } elif (mode == "view") {
        INSPECTOR_VIEW_TAB.pending_select = true
    } else {
        return JV((ok = false, error = "mode must be diff or view"))
    }
    return JV((ok = true, mode = mode))
}

def private draw_file_inspector_window() {
    let io & = unsafe(GetIO())
    if (IsWindowHovered(ImGuiHoveredFlags.RootAndChildWindows) && io.KeyCtrl) {
        if (io.MouseWheel > 0.0f) zoom_by(5)
        elif (io.MouseWheel < 0.0f) zoom_by(-5)
    }
    if (empty(g_inspector.file_path)) {
        text_disabled("Select a changed file")
        return
    }
    text(INSPECTOR_PATH, (text = g_inspector.file_path))
    same_line()
    text_disabled(INSPECTOR_SCOPE, (text = to_upper(g_inspector.comparison)))
    if (g_inspector.loading) {
        text_disabled(INSPECTOR_LOADING,
            (text = empty(g_inspector.status) ? "Loading diff..." : g_inspector.status))
        if (!g_inspector.prepare.has_retained_content) return
    }
    if (!empty(g_inspector.error)) {
        text_colored(INSPECTOR_ERROR,
            (color = float4(1.0f, 0.35f, 0.30f, 1.0f), text = g_inspector.error))
        if (!empty(g_inspector.task_session_id)) {
            text_disabled("Task: {g_inspector.task_session_id}")
        }
        if (!g_inspector.prepare.has_retained_content) return
    }
    if (g_inspector.prepare.request != InspectorRequestState.idle) {
        text_disabled(INSPECTOR_LOADING,
            (text = empty(g_inspector.prepare.status)
                ? "Preparing diff..." : g_inspector.prepare.status))
        if (!g_inspector.prepare.has_retained_content) return
    }
    tab_bar(INSPECTOR_MODE_TABS,
            (text = "InspectorModes", flags = ImGuiTabBarFlags.None)) {
        tab_item(INSPECTOR_DIFF_TAB, (text = "Diff", closable = false,
                flags = ImGuiTabItemFlags.None)) {
            g_inspector.mode = GitInspectorMode.diff
            draw_side_by_side_diff()
        }
        tab_item(INSPECTOR_VIEW_TAB, (text = "View", closable = false,
                flags = ImGuiTabItemFlags.None)) {
            g_inspector.mode = GitInspectorMode.view
            draw_file_view()
        }
    }
}

def private draw_settings_window() {
    text("Global user configuration")
    text_disabled("Repository registration and layout settings will be stored here.")
}

def private draw_terminal_window() {
    let io & = unsafe(GetIO())
    if (IsWindowHovered(ImGuiHoveredFlags.RootAndChildWindows) && io.KeyCtrl) {
        if (io.MouseWheel > 0.0f) zoom_by(5)
        elif (io.MouseWheel < 0.0f) zoom_by(-5)
    }
    text(SESSION_LABEL, (text = selected_session_label()))
    if (g_client != null && !empty(g_client.error)) {
        text_colored(CLIENT_ERROR,
            (color = float4(1.0f, 0.35f, 0.30f, 1.0f), text = g_client.error))
    }
    separator()
    let terminal_style = ImGuiTerminalStyle(
        external_input = true,
        view_height = max(GetTextLineHeight(),
            GetContentRegionAvail().y - GetFrameHeightWithSpacing()
                - GetStyle().ItemSpacing.y))
    with_font(g_terminal_font) {
        if (g_focus_terminal_pending) {
            set_keyboard_focus_here()
            g_focus_terminal_pending = false
        }
        let result <- imgui_terminal("watcher", g_terminal, g_terminal_view,
            g_terminal_cache, terminal_style)
        if (g_client != null) {
            if (!empty(result.input_text)) g_client->send_input(result.input_text)
            if (result.resized) g_client->request_resize(result.columns, result.rows)
        }
    }
    separator()
    draw_terminal_footer()
}

[export]
def init() {
    harness_init("dasHerd", 1280, 800)
    harness_maximize_window()
    DIFF_SYNTAX.value = 0
    DIFF_CONTEXT.value = 0
    var io & = unsafe(GetIO())
    io.ConfigFlags |= ImGuiConfigFlags.DockingEnable
    g_fonts = load_unicode_font_roles(14.0f, g_font_sources)
    g_terminal_font = g_fonts.monospace
    reset_terminal()
    create_job_que()
    g_jobque_owned = true
    inspector_prepare_worker_start(g_prepare_worker)
    g_selected_worktree_path = get_das_root()
    SETTINGS_WIN.open = false
    let port = clamp(to_int(option_value("--port=", "9191")), 1024, 65535)
    g_token = option_value("--token=")
    g_url = "ws://127.0.0.1:{port}/ws?token={g_token}"
    g_client = new WatcherRichClient()
    if (empty(g_token)) {
        g_client.error = "Pass the watcher token with --token=<token>"
        return
    }
    let result = g_client->init(g_url)
    if (result != 0) {
        g_client.error = "watcher connection failed ({result})"
    }
}

[export]
def update() {
    if (!harness_begin_frame()) return
    if (g_client != null) g_client->pump()
    service_sha_copy()
    drain_inspector_prepare_events()
    harness_new_frame()
    GetStyle().FontScaleMain = float(g_zoom_percent) / 100.0f
    draw_main_menu()
    dockspace(DOCK_ROOT, (flags = ImGuiDockNodeFlags.PassthruCentralNode)) {
        if (!DOCK_ROOT.has_initial_layout && DOCK_ROOT.dock_id != 0u) {
            setup_default_layout(DOCK_ROOT.dock_id)
            DOCK_ROOT.has_initial_layout = true
        }
        dock_window(REPOSITORIES_WIN, (text = "Repositories & Worktrees", closable = true,
                flags = ImGuiWindowFlags.None)) {
            draw_repositories_window()
        }
        dock_window(SESSIONS_WIN, (text = "Sessions & Activity", closable = true,
                flags = ImGuiWindowFlags.None)) {
            draw_session_list()
        }
        dock_window(GIT_ACTIVITY_WIN, (text = "Git Activity", closable = true,
                flags = ImGuiWindowFlags.None)) {
            draw_git_activity_window()
        }
        dock_window(GIT_CHANGELIST_WIN, (text = "Git Changelist", closable = true,
                flags = ImGuiWindowFlags.None)) {
            draw_git_changelist_window()
        }
        dock_window(FILE_INSPECTOR_WIN, (text = "File Inspector", closable = true,
                flags = ImGuiWindowFlags.None)) {
            draw_file_inspector_window()
        }
        dock_window(TERMINAL_WIN, (text = "Terminal", closable = true,
                flags = ImGuiWindowFlags.None)) {
            draw_terminal_window()
        }
        dock_window(SETTINGS_WIN, (text = "Settings", closable = true,
                flags = ImGuiWindowFlags.None)) {
            draw_settings_window()
        }
    }
    harness_end_frame()
}

[export]
def shutdown() {
    let prepare_stopped = inspector_prepare_worker_shutdown(g_prepare_worker)
    if (g_client != null) {
        g_client->close()
        g_client->cleanup()
        delete g_client.sessions
        delete g_client.repositories
        delete g_client.git_failure_revisions
        unsafe { delete g_client }
        g_client = null
    }
    release_inspector_content()
    text_source_view_dispose(g_inspector.old_view)
    text_source_view_dispose(g_inspector.new_view)
    text_source_view_dispose(g_inspector.view)
    terminal_viewport_snapshot_dispose(g_terminal_cache.snapshot)
    terminal_dispose(g_terminal)
    delete g_font_sources
    harness_shutdown()
    if (g_jobque_owned && prepare_stopped) {
        destroy_job_que()
        g_jobque_owned = false
    }
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
