options gen2
options indenting = 4
options no_unused_block_arguments = false
options no_unused_function_arguments = false

require dastest/testing_boost public
require imgui/imgui_playwright public
require daslib/json public
require daslib/json_boost public

//! Event-driven smoke coverage for the user-facing text viewer. Deadlines are
//! deadlock/crash guards only; every transition waits for semantic app state.

let FEATURE = "examples/text/main.das"
let DEADLOCK_GUARD_SEC = 30.0f

def private viewer_state(app : ImguiApp) : JsonValue? {
    return post_command(app, "text_viewer_state", null)
}

[test]
def test_text_viewer_open_modes_and_zoom(t : T?) {
    with_imgui_app(FEATURE) $(app) {
        var initial : JsonValue?
        let ready = wait_cond(app, DEADLOCK_GUARD_SEC) $() : bool {
            initial = viewer_state(app)
            return ((initial?["ok"] ?? false)
                && (initial?["render_revision"] ?? 0) > 0)
        }
        t |> success(ready, "welcome Markdown reaches a rendered revision")
        if (!ready) return

        let old_revision = initial?["document_revision"] ?? 0
        let fixture = "{get_das_root()}/examples/text/README.md"
        let opened = post_command(app, "text_viewer_open", JV((path = fixture)))
        t |> success(opened?["ok"] ?? false, "live open accepts a UTF-8 Markdown file")

        let file_observed = wait_cond(app, DEADLOCK_GUARD_SEC) $() : bool {
            var state = viewer_state(app)
            return ((state?["path"] ?? "") == fixture
                && (state?["document_revision"] ?? 0) > old_revision
                && (state?["mode"] ?? "") == "markdown"
                && (state?["render_revision"] ?? 0) > 0)
        }
        t |> success(file_observed, "loaded Markdown becomes the rendered document")
        if (!file_observed) return

        click(app, "DOCK_ROOT/DOCUMENT_WINDOW/MODE_PLAIN")
        let plain_observed = wait_cond(app, DEADLOCK_GUARD_SEC) $() : bool {
            return (viewer_state(app)?["mode"] ?? "") == "plain"
        }
        t |> success(plain_observed, "bottom mode control switches to plain text")

        let zoomed = post_command(app, "text_viewer_set_zoom", JV((percent = 137)))
        t |> equal(zoomed?["zoom_percent"] ?? 0, 140,
                   "zoom command snaps to a ten-percent step")
        let zoom_observed = wait_cond(app, DEADLOCK_GUARD_SEC) $() : bool {
            return (viewer_state(app)?["zoom_percent"] ?? 0) == 140
        }
        t |> success(zoom_observed, "snapped zoom reaches rendered app state")

        click(app, "DOCK_ROOT/DOCUMENT_WINDOW/MODE_MARKDOWN")
        let markdown_observed = wait_cond(app, DEADLOCK_GUARD_SEC) $() : bool {
            var state = viewer_state(app)
            return ((state?["mode"] ?? "") == "markdown"
                && (state?["render_revision"] ?? 0) > 0)
        }
        t |> success(markdown_observed, "bottom mode control returns to Markdown")
    }
}
