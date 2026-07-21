options gen2
options gc
options persistent_heap

require dastest/testing_boost public
require ../inspection_worker.das
require daslib/jobque_boost
require daslib/utf8_utils
require imgui/imgui_text_tree_sitter

[test]
def test_inspection_prepare_worker(t : T?) {
    t |> run("installed syntax providers highlight representative Diff sources") <| @(tt : T?) {
        var cpp <- tree_sitter_source_document(
            "#include <string>\n\nstatic std::string greeting() \{\n    const auto value = std::string\{\"working tree\"\};\n    return value;\n\}\n",
            "cpp", 1)
        tt |> equal(length(cpp.spans) > 1, true)
        text_source_document_dispose(cpp)
    }
    t |> run("worker prepares syntax off-thread and shuts down cleanly") <| @(t : T?) {
        t |> equal(is_utf8_string_valid("Καλημέρα, мир, こんにちは, 🐑"), true)
        create_job_que()
        var worker = InspectorPrepareWorker()
        t |> equal(inspector_prepare_worker_start(worker), true)
        t |> equal(inspector_prepare_worker_queue(worker, 7,
            "diff --git a/a.das b/a.das\n--- a/a.das\n+++ b/a.das\n@@ -1 +1 @@\n-let answer = 41\n+let answer = \"🐑\"\n",
            "options gen2\nlet answer = \"🐑\"\n", "das"), true)

        var ready = false
        var started = false
        var completed = false
        var span_count = 0
        let wait_started = ref_time_ticks()
        while (!completed && int64(get_time_usec(wait_started)) < 5000000l) {
            inspector_prepare_worker_try_event(worker) $(event : InspectorPrepareEvent#) {
                if (event.kind == InspectorPrepareEventKind.ready) {
                    ready = true
                } elif (event.kind == InspectorPrepareEventKind.started) {
                    started = true
                } elif (event.kind == InspectorPrepareEventKind.completed) {
                    completed = true
                    span_count = length(event.new_document.spans)
                    t |> equal(event.generation, 7)
                    t |> equal(length(event.parsed.rows), 1)
                    t |> equal(event.view_source,
                        "options gen2\nlet answer = \"🐑\"\n")
                }
            }
        }
        t |> equal(ready, true)
        t |> equal(started, true)
        t |> equal(completed, true)
        t |> equal(span_count > 1, true)
        t |> equal(inspector_prepare_worker_shutdown(worker), true)
        destroy_job_que()
    }
    t |> run("worker reports malformed patches and still shuts down") <| @(tt : T?) {
        create_job_que()
        var worker = InspectorPrepareWorker()
        tt |> equal(inspector_prepare_worker_start(worker), true)
        tt |> equal(inspector_prepare_worker_queue(worker, 11,
            "not a unified patch", "", "das"), true)
        var failed = false
        var failure = ""
        let wait_started = ref_time_ticks()
        while (!failed && int64(get_time_usec(wait_started)) < 5000000l) {
            inspector_prepare_worker_try_event(worker) $(event : InspectorPrepareEvent#) {
                if (event.kind == InspectorPrepareEventKind.failed) {
                    failed = true
                    failure = clone_string(event.error)
                    tt |> equal(event.generation, 11)
                }
            }
        }
        tt |> equal(failed, true)
        tt |> equal(empty(failure), false)
        tt |> equal(inspector_prepare_worker_shutdown(worker), true)
        destroy_job_que()
    }
}
