options gen2
options rtti

require dastest/testing_boost public
require ../watcher_core.das
require daslib/fio
require strings

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
    t |> run("persistent core logs output, times out hangs, and arbitrates control") <| @(t : T?) {
        let temp = create_temp_directory_result("dasherd_watcher_test_")
        if (!(temp is value)) {
            t |> failure("could not create temporary test directory")
            return
        }
        var root = temp as value
        t |> equal(watcher_init(path_join(root, "sessions")), "")

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
        var raw <- watcher_output_since(quick.session_id, 0l)
        t |> success(find(string(raw.bytes), "watcher-core-ok") >= 0, "raw PTY history can be replayed")
        var raw_tail <- watcher_output_since(quick.session_id, raw.next_offset)
        t |> equal(length(raw_tail.bytes), 0, "raw replay offset does not duplicate output")
        t |> success(fexist(path_join(path_join(path_join(root, "sessions"), quick.session_id), "output.raw")),
            "raw output log exists")

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
