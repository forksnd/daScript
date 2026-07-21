options gen2
options rtti

require dastest/testing_boost public
require ../watcher_core.das
require daslib/fio
require strings

[test]
def test_watcher_machine_output(t : T?) {
    t |> run("ConPTY wrapper controls are removed without changing task text") <| @(t : T?) {
        let esc = to_char(27)
        let bel = to_char(7)
        let wrapped = "{esc}[?9001h{esc}[2J{esc}[Hfirst\r\nsecond\tvalue{esc}]0;git.exe{bel}{esc}[?25h"
        t |> equal(watcher_strip_terminal_controls(wrapped), "first\r\nsecond\tvalue")
        let unicode_wrapped = "{esc}[31mΚαλημέρα, мир, こんにちは, 🐑{esc}[0m"
        t |> equal(watcher_strip_terminal_controls(unicode_wrapped),
            "Καλημέρα, мир, こんにちは, 🐑")
    }
}

def private wait_for(session_id, state : string; timeout_ms : int64) : bool {
    let started = ref_time_ticks()
    while (get_time_nsec(started) / 1000000l < timeout_ms) {
        watcher_tick()
        let snapshot = watcher_snapshot_json(session_id)
        if (find(snapshot, "\"state\":\"{state}\"") >= 0 &&
                find(snapshot, "\"drained\":true") >= 0) return true
        sleep(1u)
    }
    return false
}

[test]
def test_watcher_process_lifecycle(t : T?) {
    if (get_platform_name() != "windows") return
    t |> run("persistent core logs output, times out hangs, and arbitrates control") <| @(t : T?) {
        let temp = create_temp_directory_result("dasherd_watcher_test_")
        if (!(temp is value)) {
            t |> failure("could not create temporary test directory")
            return
        }
        let root = temp as value
        t |> equal(watcher_init(path_join(root, "sessions")), "")

        let oversized_display = watcher_launch(WatcherLaunchRequest(
            command_line = "cmd.exe /d /c exit 0", cwd = root,
            display_name = repeat("x", 32 * 1024 + 1)))
        t |> success(!oversized_display.ok
                && find(oversized_display.error, "display_name") >= 0,
            "oversized display metadata is rejected before session allocation")
        var oversized_argv <- ["cmd.exe", repeat("x", 32 * 1024)]
        let oversized_arguments = watcher_launch(WatcherLaunchRequest(
            argv <- oversized_argv, cwd = root))
        t |> success(!oversized_arguments.ok
                && find(oversized_arguments.error, "argv") >= 0,
            "oversized structured argv is rejected before display joining")

        let missing_command = path_join(root, "missing-watcher-command.exe")
        let failed = watcher_launch(WatcherLaunchRequest(
            command_line = "\"{missing_command}\"", cwd = root,
            columns = 1, rows = 999, timeout_ms = 5000l))
        t |> success(!failed.ok, "failed PTY launch is reported to the caller")
        t |> success(!empty(failed.session_id), "failed PTY launch retains its diagnostic session")
        let failed_snapshot = watcher_snapshot_json(failed.session_id)
        t |> success(find(failed_snapshot, "\"state\":\"failed\"") >= 0,
            "failed PTY launch is retained in watcher state")
        t |> success(find(failed_snapshot, "\"pid\":0") >= 0,
            "failed PTY launch retains a deterministic zero process id")
        let failed_metadata = fread(path_join(path_join(path_join(root, "sessions"),
            failed.session_id), "session.json"))
        t |> success(find(failed_metadata, "\"columns\":20") >= 0
                && find(failed_metadata, "\"rows\":200") >= 0,
            "session metadata retains the clamped launch dimensions")
        watcher_tick()
        t |> success(find(watcher_snapshot_json(failed.session_id), "\"state\":\"failed\"") >= 0,
            "process exit observation preserves the failed state")

        let quick = watcher_launch(WatcherLaunchRequest(
            command_line = "powershell.exe -NoProfile -Command \"Write-Output watcher-core-ok\"",
            cwd = root, columns = 80, rows = 20, timeout_ms = 5000l))
        t |> success(quick.ok, quick.error)
        t |> success(watcher_accepts_input(quick.session_id), "running session accepts input")
        t |> success(wait_for(quick.session_id, "exited", 5000l), "quick command exits and drains")
        t |> success(!watcher_accepts_input(quick.session_id), "exited session is replay-only")
        let quick_snapshot = watcher_snapshot_json(quick.session_id)
        t |> success(find(quick_snapshot, "watcher-core-ok") >= 0, "viewport retains command output")
        t |> success(find(quick_snapshot, "\"exit_code\":0") >= 0, "exit code is retained")
        let raw <- watcher_output_since(quick.session_id, 0l)
        t |> success(find(string(raw.bytes), "watcher-core-ok") >= 0, "raw PTY history can be replayed")
        let raw_tail <- watcher_output_since(quick.session_id, raw.next_offset)
        t |> equal(length(raw_tail.bytes), 0, "raw replay offset does not duplicate output")
        t |> success(fexist(path_join(path_join(path_join(root, "sessions"), quick.session_id), "output.raw")),
            "raw output log exists")

        let spaced_dir = path_join(root, "argv space")
        var mkdir_error : string
        t |> success(mkdir_rec(spaced_dir, mkdir_error), mkdir_error)
        let echo_script = path_join(spaced_dir, "echo args.ps1")
        t |> success(fwrite(echo_script, "[Console]::Write(($args -join '|'))"),
            "argv echo script is written")
        var structured_argv <- ["powershell.exe", "-NoLogo", "-NoProfile", "-File", echo_script,
            "space value", "quote\"value", "trailing\\"]
        let structured = watcher_launch(WatcherLaunchRequest(argv <- structured_argv,
            cwd = root, display_name = "structured argv test", kind = "test",
            target_id = "local", repository_id = "test-repository", worktree_path = root,
            columns = 80, rows = 20, timeout_ms = 5000l))
        t |> success(structured.ok, structured.error)
        t |> success(wait_for(structured.session_id, "exited", 5000l),
            "structured argv command exits and drains")
        let structured_snapshot = watcher_snapshot_json(structured.session_id)
        t |> success(find(structured_snapshot, "space value|quote\\\"value|trailing\\\\") >= 0,
            "structured argv preserves spaces, quotes, and trailing backslashes")
        t |> success(find(structured_snapshot, "\"worktree_path\":") >= 0,
            "session snapshot retains launch worktree association")
        var before_compact = WatcherSessionInfo()
        t |> success(watcher_get_session_info(structured.session_id, before_compact),
            "completed structured session remains addressable")
        t |> success(watcher_compact_session(structured.session_id, 60, 10),
            "completed session can release its oversized terminal state")
        var after_compact = WatcherSessionInfo()
        t |> success(watcher_get_session_info(structured.session_id, after_compact),
            "compacted session remains addressable")
        t |> equal(after_compact.pid, before_compact.pid,
            "compaction retains the original process id")
        t |> equal(after_compact.exit_code, before_compact.exit_code,
            "compaction retains the exit code")
        t |> success(find(watcher_snapshot_json(structured.session_id), "space value") >= 0,
            "compaction replays retained raw output into an inspectable terminal")
        let structured_log = path_join(path_join(path_join(root, "sessions"), structured.session_id), "output.raw")
        t |> equal(watcher_prune_completed_sessions("test", "test-repository", root), 1,
            "completed observation session can leave the in-memory working set")
        t |> equal(watcher_find_session(structured.session_id), -1,
            "pruned observation session is removed from the live list")
        t |> success(fexist(structured_log), "pruning preserves its durable raw log")

        for (iteration in range(6)) {
            let close_race = watcher_launch(WatcherLaunchRequest(
                command_line = "cmd.exe /d /c exit 0", cwd = root,
                columns = 80, rows = 20, timeout_ms = 5000l))
            t |> success(close_race.ok, close_race.error)
            t |> success(wait_for(close_race.session_id, "exited", 5000l),
                "pipe closure still converges to process exit ({iteration})")
        }

        let hung = watcher_launch(WatcherLaunchRequest(
            command_line = "powershell.exe -NoProfile -Command \"Start-Sleep -Seconds 30\"",
            cwd = root, columns = 80, rows = 20, timeout_ms = 250l))
        t |> success(hung.ok, hung.error)
        t |> success(wait_for(hung.session_id, "exited", 4000l), "hung command is externally terminated")
        t |> success(find(watcher_snapshot_json(hung.session_id), "\"reason\":\"wall_timeout\"") >= 0,
            "timeout reason is retained")

        let controlled = watcher_launch(WatcherLaunchRequest(
            command_line = "powershell.exe -NoProfile -Command \"Start-Sleep -Seconds 5\"",
            cwd = root, columns = 80, rows = 20, timeout_ms = 10000l))
        t |> success(controlled.ok, controlled.error)
        t |> success(watcher_claim_controller(controlled.session_id, 11), "first controller claims lease")
        t |> success(!watcher_claim_controller(controlled.session_id, 12), "second controller is rejected")
        watcher_release_controller(controlled.session_id, 11, "test_detach")
        t |> success(watcher_claim_controller(controlled.session_id, 12), "released lease can be reclaimed")
        t |> equal(watcher_terminate(controlled.session_id, "test_complete"), "")
        t |> success(wait_for(controlled.session_id, "exited", 4000l), "terminated command exits and drains")
        watcher_shutdown(true)
        rmdir_rec(root)
    }
}
