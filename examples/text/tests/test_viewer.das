options gen2
options indenting = 4
options no_unused_block_arguments = false
options no_unused_function_arguments = false

require dastest/testing_boost public
require imgui/imgui_playwright public
require daslib/fio public
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

        let opened_window = post_command(app, "text_viewer_open_window",
                                         JV((path = fixture)))
        t |> success(opened_window?["ok"] ?? false,
                     "another document opens inside the running viewer")
        t |> equal(opened_window?["document_count"] ?? 0, 2,
                   "in-process open adds a second dockable document")
        let second_document_observed = wait_cond(app, DEADLOCK_GUARD_SEC) $() : bool {
            return (viewer_state(app)?["document_count"] ?? 0) == 2
        }
        t |> success(second_document_observed,
                     "the viewer retains both documents without spawning another app")
    }
}

[test]
def test_text_viewer_checkbox_save_reload_roundtrip(t : T?) {
    var temp_error : string
    let temp_dir = create_temp_directory("das_text_viewer_roundtrip", temp_error)
    t |> success(empty(temp_error) && !empty(temp_dir),
                 "roundtrip test receives a dedicated temporary directory")
    if (!empty(temp_error) || empty(temp_dir)) return
    let fixture = path_join(temp_dir, "checkbox.md")
    let bom = "\xEF\xBB\xBF"
    let original = bom + "# Tasks\r\n\r\n- [ ] first\r\n- [X] second\n"
    let expected = bom + "# Tasks\r\n\r\n- [x] first\r\n- [X] second\n"
    t |> success(fwrite(fixture, original), "mixed-ending BOM fixture is written")

    with_imgui_app(FEATURE) $(app) {
        let opened = post_command(app, "text_viewer_open", JV((path = fixture)))
        t |> success(opened?["ok"] ?? false, "checkbox fixture opens")

        var initial : JsonValue?
        var checkbox : JsonValue?
        let ready = wait_cond(app, DEADLOCK_GUARD_SEC) $() : bool {
            initial = viewer_state(app)
            checkbox = post_command(app, "text_viewer_checkbox", JV((index = 0)))
            return ((initial?["ok"] ?? false)
                && (initial?["checkbox_count"] ?? 0) == 2
                && (initial?["has_utf8_bom"] ?? false)
                && !(initial?["dirty"] ?? true)
                && (checkbox?["ok"] ?? false))
        }
        t |> success(ready, "checkbox projection and screen geometry become observable")

        if (ready) {
            // Command responses are transient live-API values; snapshot the
            // geometry before issuing any later commands.
            let checkbox_width_before = (checkbox?["x1"] ?? 0.0f)
                - (checkbox?["x0"] ?? 0.0f)
            let checkbox_height_before = (checkbox?["y1"] ?? 0.0f)
                - (checkbox?["y0"] ?? 0.0f)
            let no_op_save = post_command(app, "text_viewer_save", JV((path = fixture)))
            t |> success(no_op_save?["ok"] ?? false, "no-op save succeeds")
            t |> equal(fread(fixture), original,
                       "no-op app save preserves BOM, spelling, and mixed endings byte-for-byte")

            let old_revision = initial?["document_revision"] ?? 0
            let old_activation = initial?["control_activation_revision"] ?? 0
            let x = ((checkbox?["x0"] ?? 0.0f) + (checkbox?["x1"] ?? 0.0f)) * 0.5f
            let y = ((checkbox?["y0"] ?? 0.0f) + (checkbox?["y1"] ?? 0.0f)) * 0.5f
            post_command(app, "imgui_mouse_pos", JV((x = x, y = y)))
            post_command(app, "imgui_mouse_click", JV((button = 0, action = "press")))
            post_command(app, "imgui_mouse_click", JV((button = 0, action = "release")))

            let toggled = wait_cond(app, DEADLOCK_GUARD_SEC) $() : bool {
                var state = viewer_state(app)
                var current = post_command(app, "text_viewer_checkbox", JV((index = 0)))
                return ((state?["document_revision"] ?? 0) == old_revision + 1
                    && (state?["control_activation_revision"] ?? 0) > old_activation
                    && (state?["dirty"] ?? false)
                    && (current?["ok"] ?? false)
                    && (current?["checked"] ?? false))
            }
            t |> success(toggled,
                         "click emits a control event, edits source, and reparses the projection")

            if (toggled) {
                let toggled_checkbox = post_command(
                    app, "text_viewer_checkbox", JV((index = 0)))
                let width_after = (toggled_checkbox?["x1"] ?? 0.0f)
                    - (toggled_checkbox?["x0"] ?? 0.0f)
                let height_after = (toggled_checkbox?["y1"] ?? 0.0f)
                    - (toggled_checkbox?["y0"] ?? 0.0f)
                t |> equal(width_after, checkbox_width_before,
                           "checked and unchecked controls reserve identical width")
                t |> equal(height_after, checkbox_height_before,
                           "checked and unchecked controls reserve identical height")
                let saved = post_command(app, "text_viewer_save", JV((path = fixture)))
                t |> success(saved?["ok"] ?? false, "edited source saves")
                t |> success(!(saved?["dirty"] ?? true), "successful save advances saved revision")
                t |> equal(fread(fixture), expected,
                           "checkbox save changes exactly its marker byte")

                let reloaded = post_command(app, "text_viewer_open", JV((path = fixture)))
                t |> success(reloaded?["ok"] ?? false, "saved document reloads")
                let reload_observed = wait_cond(app, DEADLOCK_GUARD_SEC) $() : bool {
                    var state = viewer_state(app)
                    var current = post_command(app, "text_viewer_checkbox", JV((index = 0)))
                    return ((state?["has_utf8_bom"] ?? false)
                        && !(state?["dirty"] ?? true)
                        && (current?["ok"] ?? false)
                        && (current?["checked"] ?? false))
                }
                t |> success(reload_observed,
                             "saved checkbox state survives a fresh load/projection roundtrip")
            }
        }
    }
    rmdir_rec_result(temp_dir)
}
