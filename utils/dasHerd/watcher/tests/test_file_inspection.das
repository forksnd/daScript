options gen2

require dastest/testing_boost public
require ../file_inspection.das

[test]
def test_unified_patch_projection(t : T?) {
    t |> run("unified hunks become aligned Diff and resulting View text") <| @(t : T?) {
        let source = "diff --git a/sample.das b/sample.das\n--- a/sample.das\n+++ b/sample.das\n@@ -1,4 +1,5 @@\n options gen2\n-let old = 1\n+let next = 2\n+let added = 3\n keep()\n-last()\n"
        var inscope parsed <- git_parse_unified_patch(source)
        t |> equal(parsed.error, "")
        t |> equal(parsed.old_text, "options gen2\nlet old = 1\nkeep()\nlast()")
        t |> equal(parsed.new_text, "options gen2\nlet next = 2\nlet added = 3\nkeep()")
        t |> equal(length(parsed.rows), 5)
        t |> equal(parsed.rows[1].old_kind, GitDiffLineKind.removed)
        t |> equal(parsed.rows[1].new_kind, GitDiffLineKind.added)
        t |> equal(parsed.rows[2].old_kind, GitDiffLineKind.empty)
        t |> equal(parsed.rows[2].new_kind, GitDiffLineKind.added)
        t |> equal(parsed.rows[4].old_kind, GitDiffLineKind.removed)
        t |> equal(parsed.rows[4].new_kind, GitDiffLineKind.empty)
        t |> equal(length(parsed.new_changed_lines), 2)
    }
}

[test]
def test_inspector_load_state(t : T?) {
    t |> run("replacement retains ready content until the matching result arrives") <| @(t : T?) {
        var state = InspectorLoadState()
        let first = inspector_load_queue(state, "Queued first")
        t |> equal(first, 1)
        t |> equal(state.request, InspectorRequestState.queued)
        t |> equal(inspector_load_start(state, first), true)
        t |> equal(inspector_load_succeed(state, first), true)
        t |> equal(state.has_retained_content, true)
        t |> equal(state.retained_generation, first)

        let replacement = inspector_load_queue(state, "Queued replacement")
        t |> equal(state.request, InspectorRequestState.queued)
        t |> equal(state.has_retained_content, true)
        t |> equal(state.retained_generation, first)
        t |> equal(inspector_load_start(state, replacement), true)
        t |> equal(inspector_load_fail(state, replacement, "parse failed"), true)
        t |> equal(state.latest_outcome, InspectorContentState.failed)
        t |> equal(state.has_retained_content, true)
        t |> equal(state.retained_generation, first)
        t |> equal(state.error, "parse failed")
    }

    t |> run("superseded worker events cannot mutate the current generation") <| @(t : T?) {
        var state = InspectorLoadState()
        let stale = inspector_load_queue(state)
        let current = inspector_load_queue(state)
        t |> equal(inspector_load_start(state, stale), false)
        t |> equal(inspector_load_succeed(state, stale), false)
        t |> equal(inspector_load_fail(state, stale, "stale failure"), false)
        t |> equal(state.generation, current)
        t |> equal(state.request, InspectorRequestState.queued)
        t |> equal(state.error, "")
        t |> equal(inspector_load_start(state, current), true)
        t |> equal(inspector_load_succeed(state, current), true)
        t |> equal(state.retained_generation, current)
    }

    t |> run("reset invalidates in-flight events and removes retained content") <| @(t : T?) {
        var state = InspectorLoadState()
        let generation = inspector_load_queue(state)
        t |> equal(inspector_load_start(state, generation), true)
        inspector_load_reset(state)
        t |> equal(inspector_load_succeed(state, generation), false)
        t |> equal(state.request, InspectorRequestState.idle)
        t |> equal(state.latest_outcome, InspectorContentState.absent)
        t |> equal(state.has_retained_content, false)
    }

    t |> run("superseding a refresh invalidates work but retains installed content") <| @(t : T?) {
        var state = InspectorLoadState()
        let ready = inspector_load_queue(state)
        let started = inspector_load_start(state, ready)
        let succeeded = inspector_load_succeed(state, ready)
        t |> equal(started && succeeded, true)
        inspector_load_supersede(state, "Requesting diff...")
        t |> equal(state.generation, ready + 1)
        t |> equal(state.request, InspectorRequestState.idle)
        t |> equal(state.has_retained_content, true)
        t |> equal(state.retained_generation, ready)
        t |> equal(state.status, "Requesting diff...")
        t |> equal(inspector_load_fail(state, ready, "stale"), false)
    }
}
