options gen2
options rtti
options persistent_heap
options gc
options indenting = 4

module repository_core shared public

require ./watcher_core.das public
require daslib/fio public
require daslib/json_boost public
require daslib/strings_boost
require strings public

let private REPOSITORY_CONFIG_VERSION = 1
let private GIT_TIMEOUT_MS = 15000l
let private GIT_CAPTURE_COLUMNS = 240
let private GIT_CAPTURE_ROWS = 100

struct public RepositoryRecord {
    id : string
    target_id : string = "local"
    checkout_path : string
    display_name : string
    base_ref : string
}

struct public GitFileState {
    worktree_path : string
    code : string
    path : string
    old_path : string
}

struct public WorktreeState {
    path : string
    head : string
    branch : string
    is_main : bool
    detached : bool
    bare : bool
    locked : bool
    prunable : bool
    ahead : int
    behind : int
    staged : int
    unstaged : int
    untracked : int
    conflicted : int
    error : string
}

struct public RepositorySnapshot {
    record : RepositoryRecord = RepositoryRecord()
    worktrees : array<WorktreeState>
    files : array<GitFileState>
    refreshing : bool
    error : string
    revision : int64
    last_success_ms : int64
    task_session_id : string
}

struct public RepositoryAddRequest {
    path : string
    display_name : string
    target_id : string = "local"
}

struct public RepositoryRefreshRequest {
    repository_id : string
}

struct private RepositoryConfigFile {
    version : int = REPOSITORY_CONFIG_VERSION
    repositories : array<RepositoryRecord>
}

class private RepositoryRuntime {
    snapshot : RepositorySnapshot = RepositorySnapshot()
    phase : int
    worktree_index : int
    refresh_pending : bool = true
    rerun_pending : bool
    def operator delete {
        delete snapshot.worktrees
        delete snapshot.files
    }
}

var private g_repositories : array<RepositoryRuntime?>
var private g_config_path : string
var private g_revision : int64

def public repository_default_config_path : string {
    var base : string
    if (has_env_variable("APPDATA")) {
        base = get_env_variable("APPDATA")
    } elif (has_env_variable("XDG_CONFIG_HOME")) {
        base = get_env_variable("XDG_CONFIG_HOME")
    } elif (has_env_variable("HOME")) {
        base = path_join(get_env_variable("HOME"), ".config")
    } else {
        base = "."
    }
    return path_join(path_join(base, "daScript"), "dasHerd.json")
}

def private valid_path(path : string) : bool {
    return !empty(path) && find(path, "\r") < 0 && find(path, "\n") < 0
}

def private normalized_path(path : string) : string {
    return get_full_file_name(path)
}

def private repository_id(target_id, path : string) : string {
    let identity_path = get_platform_name() == "windows" ? to_lower(path) : path
    return "{target_id}:{identity_path}"
}

def public repository_valid_relative_file_path(path : string) : bool {
    if (!valid_path(path) || starts_with(path, "/") || starts_with(path, "\\")
            || is_absolute(path)) return false
    if (get_platform_name() == "windows" && length(path) >= 2
            && byte_at(path, 1) == ':') return false
    return (path != ".." && !ends_with(path, "/..") && !ends_with(path, "\\..")
        && find(path, "../") < 0 && find(path, "..\\") < 0)
}

def private changed(var runtime : RepositoryRuntime?) {
    runtime.snapshot.revision++
    g_revision++
}

def private save_config : string {
    var inscope config = RepositoryConfigFile()
    config.repositories |> reserve(length(g_repositories))
    for (runtime in g_repositories) {
        if (runtime != null) config.repositories |> push(runtime.snapshot.record)
    }
    var error : string
    if (!mkdir_rec(dir_name(g_config_path), error)) return "could not create config directory: {error}"
    if (!fwrite(g_config_path, sprint_json(config, true))) return "could not write {g_config_path}"
    return ""
}

def private add_runtime(record : RepositoryRecord) {
    g_repositories |> push(new RepositoryRuntime(
        snapshot = RepositorySnapshot(record = record)))
    g_revision++
}

def private load_config : string {
    if (!fexist(g_config_path)) return ""
    var inscope config = RepositoryConfigFile()
    if (!sscan_json(fread(g_config_path), config)) return "invalid repository config JSON"
    if (config.version != REPOSITORY_CONFIG_VERSION) {
        return "unsupported repository config version {config.version}"
    }
    for (record in config.repositories) {
        if (valid_path(record.checkout_path)) add_runtime(record)
    }
    return ""
}

def public repository_init(config_path, bootstrap_checkout : string) : string {
    g_config_path = config_path
    let load_error = load_config()
    if (!empty(load_error)) return load_error
    if (empty(g_repositories) && valid_path(bootstrap_checkout)) {
        let path = normalized_path(bootstrap_checkout)
        add_runtime(RepositoryRecord(id = repository_id("local", path), target_id = "local",
            checkout_path = path, display_name = base_name(path)))
        return save_config()
    }
    return ""
}

def public repository_shutdown() {
    for (runtime in g_repositories) {
        if (runtime != null) {
            unsafe { delete runtime }
        }
    }
    g_repositories |> clear()
    unsafe { delete g_repositories }
    g_config_path = ""
    g_revision = 0l
}

def public repository_revision : int64 {
    return g_revision
}

def public repository_list_json : string {
    var inscope parts : array<string>
    parts |> reserve(length(g_repositories))
    for (runtime in g_repositories) {
        if (runtime != null) parts |> push(sprint_json(runtime.snapshot, false))
    }
    return "[" + join(parts, ",") + "]"
}

def private find_repository(id : string) : int {
    for (index in range(length(g_repositories))) {
        if (g_repositories[index] != null && g_repositories[index].snapshot.record.id == id) return index
    }
    return -1
}

def public repository_add(request : RepositoryAddRequest) : string {
    if (request.target_id != "local" && !empty(request.target_id)) return "only the local target is implemented"
    if (!valid_path(request.path)) return "repository path is required and cannot contain CR/LF"
    let path = normalized_path(request.path)
    let id = repository_id("local", path)
    if (find_repository(id) >= 0) return "repository is already registered"
    add_runtime(RepositoryRecord(id = id, target_id = "local", checkout_path = path,
        display_name = !empty(request.display_name) ? request.display_name : base_name(path)))
    let error = save_config()
    if (!empty(error)) return error
    return ""
}

def public repository_refresh(id : string) : string {
    let index = find_repository(id)
    if (index < 0) return "unknown repository"
    var runtime = g_repositories[index]
    if (runtime.phase != 0) {
        runtime.rerun_pending = true
    } else {
        runtime.refresh_pending = true
    }
    return ""
}

def public repository_build_file_diff_argv(worktree_path : string;
                                           file : GitFileState const;
                                           comparison : string;
                                           var argv : array<string>&;
                                           var expected_exit_code : int64&;
                                           diff_context : string = "compact") : string {
    delete argv
    expected_exit_code = 0l
    let staged = file.code != "??" && byte_at(file.code, 0) != ' '
    let working = file.code == "??" || byte_at(file.code, 1) != ' '
    if (comparison == "staged" && !staged) return "file has no staged change"
    if (comparison == "working" && !working) return "file has no working-tree change"
    if (comparison != "staged" && comparison != "working") {
        return "comparison is not implemented"
    }
    var context_lines = 3
    if (diff_context == "expanded") {
        context_lines = 20
    } elif (diff_context == "full") {
        context_lines = 1000000
    } elif (diff_context != "compact") {
        return "diff context is not implemented"
    }
    var inscope common <- ["git", "-C", worktree_path, "--no-pager", "-c",
        "color.ui=false", "diff"]
    argv := common
    if (comparison == "staged") argv |> push("--cached")
    if (comparison == "working" && file.code == "??") {
        argv |> push("--no-index")
        expected_exit_code = 1l
    }
    argv |> push("--no-ext-diff")
    argv |> push("--unified={context_lines}")
    argv |> push("--")
    if (comparison == "working" && file.code == "??") {
        argv |> push(get_platform_name() == "windows" ? "NUL" : "/dev/null")
    }
    argv |> push(file.path)
    return ""
}

def public repository_file_diff_argv(repository_id, worktree_path, file_path,
                                     comparison : string;
                                     var argv : array<string>&;
                                     var expected_exit_code : int64&;
                                     diff_context : string = "compact") : string {
    delete argv
    expected_exit_code = 0l
    let repository_index = find_repository(repository_id)
    if (repository_index < 0 || g_repositories[repository_index] == null) {
        return "unknown repository"
    }
    let runtime = g_repositories[repository_index]
    var worktree_found = false
    for (worktree in runtime.snapshot.worktrees) {
        if (worktree.path == worktree_path) {
            worktree_found = true
            break
        }
    }
    if (!worktree_found) return "unknown worktree"

    var file_index = -1
    for (index in range(length(runtime.snapshot.files))) {
        let file = runtime.snapshot.files[index]
        if (file.worktree_path == worktree_path && file.path == file_path) {
            file_index = index
            break
        }
    }
    if (file_index < 0) return "file is not present in the observed Git status"
    let file = runtime.snapshot.files[file_index]
    return repository_build_file_diff_argv(
        worktree_path, file, comparison, argv, expected_exit_code,
        diff_context)
}

def public repository_working_file_text(repository_id, worktree_path, file_path : string;
                                        var text : string&) : string {
    text = ""
    let repository_index = find_repository(repository_id)
    if (repository_index < 0 || g_repositories[repository_index] == null) {
        return "unknown repository"
    }
    let runtime = g_repositories[repository_index]
    var file_found = false
    for (file in runtime.snapshot.files) {
        if (file.worktree_path == worktree_path && file.path == file_path
                && (file.code == "??" || byte_at(file.code, 1) != ' ')) {
            file_found = true
            break
        }
    }
    if (!file_found) return "file has no observed working-tree change"
    if (!repository_valid_relative_file_path(file_path)) {
        return "invalid worktree-relative file path"
    }
    let absolute_path = path_join(worktree_path, file_path)
    if (!fexist(absolute_path)) return "working file does not exist"
    text = fread(absolute_path)
    return ""
}

def public repository_working_file_bytes(repository_id, worktree_path, file_path : string;
                                         max_bytes : int;
                                         var bytes : array<uint8>;
                                         var byte_count : int&) : string {
    delete bytes
    byte_count = 0
    let repository_index = find_repository(repository_id)
    if (repository_index < 0 || g_repositories[repository_index] == null) {
        return "unknown repository"
    }
    let runtime = g_repositories[repository_index]
    var file_found = false
    for (file in runtime.snapshot.files) {
        if (file.worktree_path == worktree_path && file.path == file_path
                && (file.code == "??" || byte_at(file.code, 1) != ' ')) {
            file_found = true
            break
        }
    }
    if (!file_found) return "file has no observed working-tree change"
    if (!repository_valid_relative_file_path(file_path)) {
        return "invalid worktree-relative file path"
    }
    let absolute_path = path_join(worktree_path, file_path)
    if (!fexist(absolute_path)) return "working file does not exist"
    var loaded = false
    fopen(absolute_path, "rb") $(file) {
        if (file != null) {
            fmap(file) $(var data : array<uint8>#) {
                byte_count = length(data)
                if (byte_count <= max_bytes) {
                    bytes := data
                }
                loaded = true
            }
        }
    }
    return loaded ? "" : "could not read working file bytes"
}

def private launch_git(var runtime : RepositoryRuntime?; argv : array<string>;
                       kind, worktree_path : string) : bool {
    var inscope request = WatcherLaunchRequest(
        cwd = runtime.snapshot.record.checkout_path,
        display_name = "Git: " + join(argv, " "), kind = kind,
        target_id = runtime.snapshot.record.target_id,
        repository_id = runtime.snapshot.record.id,
        worktree_path = worktree_path,
        columns = GIT_CAPTURE_COLUMNS, rows = GIT_CAPTURE_ROWS, timeout_ms = GIT_TIMEOUT_MS)
    request.argv := argv
    let _pruned = watcher_prune_completed_sessions(kind,
        runtime.snapshot.record.id, worktree_path)
    let result = watcher_launch(request)
    runtime.snapshot.task_session_id = result.session_id
    if (!result.ok) {
        runtime.snapshot.error = result.error
        runtime.snapshot.refreshing = false
        runtime.phase = 0
        changed(runtime)
        return false
    }
    return true
}

def private launch_inventory(var runtime : RepositoryRuntime?) {
    runtime.refresh_pending = false
    runtime.snapshot.refreshing = true
    runtime.snapshot.error = ""
    runtime.phase = 1
    changed(runtime)
    let _ = launch_git(runtime, ["git", "-C", runtime.snapshot.record.checkout_path,
        "worktree", "list", "--porcelain"], "git-inventory",
        runtime.snapshot.record.checkout_path)
}

def private trim_cr(value : string) : string {
    if (!empty(value) && ends_with(value, "\r")) return slice(value, 0, length(value) - 1)
    return value
}

def private byte_at(value : string; index : int) : int {
    var result = 0
    peek_data(value) $(data) {
        result = int(data[index])
    }
    return result
}

def private append_worktree(var worktrees : array<WorktreeState>;
                            var current : WorktreeState) {
    worktrees |> push(WorktreeState(path = current.path, head = current.head,
        branch = current.branch, is_main = empty(worktrees), detached = current.detached,
        bare = current.bare, locked = current.locked, prunable = current.prunable))
}

def public repository_parse_inventory(text : string; var worktrees : array<WorktreeState>) {
    var current <- WorktreeState()
    for (raw_line in split(text, "\n")) {
        let line = trim_cr(raw_line)
        if (empty(line)) {
            if (!empty(current.path)) {
                append_worktree(worktrees, current)
                current <- WorktreeState()
            }
        } elif (starts_with(line, "worktree ")) {
            current.path = slice(line, 9)
        } elif (starts_with(line, "HEAD ")) {
            current.head = slice(line, 5)
        } elif (starts_with(line, "branch ")) {
            current.branch = slice(line, 7)
            if (starts_with(current.branch, "refs/heads/")) {
                current.branch = slice(current.branch, 11)
            }
        } elif (line == "detached") {
            current.detached = true
        } elif (line == "bare") {
            current.bare = true
        } elif (starts_with(line, "locked")) {
            current.locked = true
        } elif (starts_with(line, "prunable")) {
            current.prunable = true
        }
    }
    if (!empty(current.path)) {
        append_worktree(worktrees, current)
    }
}

def private conflict_code(code : string) : bool {
    return code == "DD" || code == "AU" || code == "UD" || code == "UA" || code == "DU" || code == "AA" || code == "UU"
}

def private parse_counter(text, marker : string) : int {
    let start = find(text, marker)
    if (start < 0) return 0
    let value_start = start + length(marker)
    var value_end = value_start
    while (value_end < length(text) && is_number(byte_at(text, value_end))) {
        value_end++
    }
    return to_int(slice(text, value_start, value_end))
}

def public repository_parse_status(text : string; var worktree : WorktreeState;
                                   var files : array<GitFileState>) {
    worktree.ahead = 0
    worktree.behind = 0
    worktree.staged = 0
    worktree.unstaged = 0
    worktree.untracked = 0
    worktree.conflicted = 0
    for (raw_line in split(text, "\n")) {
        let line = trim_cr(raw_line)
        if (starts_with(line, "## ")) {
            worktree.ahead = parse_counter(line, "ahead ")
            worktree.behind = parse_counter(line, "behind ")
            continue
        }
        if (length(line) < 3) continue
        let code = slice(line, 0, 2)
        var path = slice(line, 3)
        var old_path : string
        let rename_separator = find(path, " -> ")
        if (rename_separator >= 0) {
            old_path = slice(path, 0, rename_separator)
            path = slice(path, rename_separator + 4)
        }
        files |> push(GitFileState(worktree_path = worktree.path, code = code,
            path = path, old_path = old_path))
        if (code == "??") worktree.untracked++
        else {
            if (byte_at(line, 0) != ' ') worktree.staged++
            if (byte_at(line, 1) != ' ') worktree.unstaged++
            if (conflict_code(code)) worktree.conflicted++
        }
    }
}

def private launch_status(var runtime : RepositoryRuntime?) : bool {
    if (runtime.worktree_index >= length(runtime.snapshot.worktrees)) return false
    let path = runtime.snapshot.worktrees[runtime.worktree_index].path
    runtime.snapshot.record.checkout_path = runtime.snapshot.worktrees[0].path
    return launch_git(runtime, ["git", "-C", path, "-c", "core.quotePath=false",
        "status", "--porcelain=1", "--branch", "--untracked-files=all"],
        "git-status", path)
}

def private finish_refresh(var runtime : RepositoryRuntime?) {
    runtime.phase = 0
    runtime.snapshot.refreshing = false
    runtime.snapshot.last_success_ms = int64(get_clock()) * 1000l
    runtime.snapshot.task_session_id = ""
    changed(runtime)
    if (runtime.rerun_pending) {
        runtime.rerun_pending = false
        runtime.refresh_pending = true
    }
}

def private fail_refresh(var runtime : RepositoryRuntime?; message : string) {
    runtime.phase = 0
    runtime.snapshot.refreshing = false
    runtime.snapshot.error = message
    changed(runtime)
}

def private advance_repository(var runtime : RepositoryRuntime?) {
    if (runtime.phase == 0) {
        if (runtime.refresh_pending) launch_inventory(runtime)
        return
    }
    var info = WatcherSessionInfo()
    if (!watcher_get_session_info(runtime.snapshot.task_session_id, info) || !info.drained) return
    var scrollback_rows = 0
    let output = watcher_terminal_text(runtime.snapshot.task_session_id, scrollback_rows)
    if (scrollback_rows > 0) {
        let _compacted_overflow = watcher_compact_session(runtime.snapshot.task_session_id)
        fail_refresh(runtime, "Git structured output exceeded the {GIT_CAPTURE_ROWS}-row capture window; inspect session {runtime.snapshot.task_session_id}")
        return
    }
    let _compacted = watcher_compact_session(runtime.snapshot.task_session_id)
    if (info.exit_code != 0l) {
        fail_refresh(runtime, "git exited with {info.exit_code}: {output}")
        return
    }
    if (runtime.phase == 1) {
        var worktrees : array<WorktreeState>
        repository_parse_inventory(output, worktrees)
        if (empty(worktrees)) {
            fail_refresh(runtime, "git returned no worktrees")
            return
        }
        delete runtime.snapshot.worktrees
        runtime.snapshot.worktrees <- worktrees
        runtime.snapshot.files |> clear()
        let main_path = runtime.snapshot.worktrees[0].path
        if (runtime.snapshot.record.checkout_path != main_path) {
            let old_default_name = base_name(runtime.snapshot.record.checkout_path)
            runtime.snapshot.record.checkout_path = main_path
            runtime.snapshot.record.id = repository_id(runtime.snapshot.record.target_id, main_path)
            if (empty(runtime.snapshot.record.display_name) || runtime.snapshot.record.display_name == old_default_name) {
                runtime.snapshot.record.display_name = base_name(main_path)
            }
            let _save_error = save_config()
        }
        runtime.phase = 2
        runtime.worktree_index = 0
        let _status_started = launch_status(runtime)
        changed(runtime)
        return
    }
    repository_parse_status(output, runtime.snapshot.worktrees[runtime.worktree_index], runtime.snapshot.files)
    runtime.worktree_index++
    if (runtime.worktree_index < length(runtime.snapshot.worktrees)) {
        let _next_status_started = launch_status(runtime)
        changed(runtime)
    } else {
        finish_refresh(runtime)
    }
}

def public repository_tick() {
    for (runtime in g_repositories) {
        if (runtime != null) advance_repository(runtime)
    }
}
