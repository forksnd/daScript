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

def private source_range(app : ImguiApp; text : string) : JsonValue? {
    return post_command(app, "text_viewer_source_range", JV((text = text)))
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

        click(app, "DOCK_ROOT/DOCUMENT_WINDOW/MODE_TOGGLE")
        let plain_observed = wait_cond(app, DEADLOCK_GUARD_SEC) $() : bool {
            var state = viewer_state(app)
            return ((state?["mode"] ?? "") == "plain"
                && (state?["source_render_revision"] ?? 0) > 0
                && (state?["source_marker_count"] ?? 0) > 0
                && (state?["source_cell_count"] ?? 0) > 0
                && (state?["source_layout_hits"] ?? 0) > 0)
        }
        t |> success(plain_observed,
                     "bottom mode control switches to styled selectable source")
        if (!plain_observed) return

        let marker = source_range(app, "**")
        t |> success(marker?["ok"] ?? false,
                     "Markdown delimiter range is inspectable without a screenshot")
        t |> success(marker?["all_markers"] ?? false,
                     "both strong delimiter bytes use marker styling")
        t |> equal(marker?["matched_cells"] ?? 0, 2,
                   "two marker bytes map to two source cells")
        t |> equal(marker?["marker_cells"] ?? 0, 2,
                   "matched cells retain marker style")
        t |> equal(marker?["first_cell_start"] ?? -1,
                   marker?["source_start"] ?? -2,
                   "first marker cell starts at requested byte")
        t |> equal(marker?["last_cell_end"] ?? -1,
                   marker?["source_end"] ?? -2,
                   "last marker cell ends at requested byte")
        t |> equal(marker?["columns"] ?? 0, 2,
                   "two ASCII marker bytes retain two monospace cells")

        let emoji = source_range(app, "👩‍💻")
        t |> success(emoji?["ok"] ?? false, "color emoji source range is inspectable")
        t |> equal(emoji?["columns"] ?? 0, 2,
                   "ZWJ color emoji reserves exactly two source cells")

        let marker_start = marker?["source_start"] ?? -1
        let marker_end = marker?["source_end"] ?? -1
        let marker_x0 = marker?["x0"] ?? 0.0f
        let marker_x1 = marker?["x1"] ?? 0.0f
        let marker_y = ((marker?["y0"] ?? 0.0f) + (marker?["y1"] ?? 0.0f)) * 0.5f
        var before_selection = viewer_state(app)
        let commit_before = before_selection?["source_selection_commit_revision"] ?? 0
        post_command(app, "imgui_mouse_pos", JV((x = marker_x0 + 1.0f, y = marker_y)))
        post_command(app, "imgui_mouse_click", JV((button = 0, action = "press")))
        let press_observed = wait_cond(app, DEADLOCK_GUARD_SEC) $() : bool {
            return viewer_state(app)?["source_selection_dragging"] ?? false
        }
        t |> success(press_observed, "source selection press is acknowledged")
        post_command(app, "imgui_mouse_pos", JV((x = marker_x1 - 1.0f, y = marker_y)))
        let marker_selected = wait_cond(app, DEADLOCK_GUARD_SEC) $() : bool {
            var state = viewer_state(app)
            return ((state?["source_selection_start"] ?? -1) == marker_start
                && (state?["source_selection_end"] ?? -1) == marker_end
                && (state?["source_selection_has_markers"] ?? false))
        }
        t |> success(marker_selected,
                     "source selection retains exact marker byte offsets and style")
        post_command(app, "imgui_mouse_click", JV((button = 0, action = "release")))
        let release_observed = wait_cond(app, DEADLOCK_GUARD_SEC) $() : bool {
            var state = viewer_state(app)
            return (!(state?["source_selection_dragging"] ?? true)
                && (state?["source_selection_commit_revision"] ?? 0) > commit_before)
        }
        t |> success(release_observed, "source selection release is acknowledged")

        let zoomed = post_command(app, "text_viewer_set_zoom", JV((percent = 137)))
        t |> equal(zoomed?["zoom_percent"] ?? 0, 135,
                   "zoom command snaps to a five-percent step")
        let zoom_observed = wait_cond(app, DEADLOCK_GUARD_SEC) $() : bool {
            return (viewer_state(app)?["zoom_percent"] ?? 0) == 135
        }
        t |> success(zoom_observed, "snapped zoom reaches rendered app state")

        click(app, "DOCK_ROOT/DOCUMENT_WINDOW/MODE_TOGGLE")
        let markdown_observed = wait_cond(app, DEADLOCK_GUARD_SEC) $() : bool {
            var state = viewer_state(app)
            return ((state?["mode"] ?? "") == "markdown"
                && (state?["render_revision"] ?? 0) > 0)
        }
        t |> success(markdown_observed, "bottom mode control returns to Markdown")
    }
}
