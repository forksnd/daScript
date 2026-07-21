options gen2
options gc
options persistent_heap
options indenting = 4

module inspection_worker shared public

require daslib/jobque_boost
require daslib/rtti
require imgui/imgui_text_tree_sitter
require ./file_inspection.das

enum public InspectorPrepareEventKind {
    ready
    started
    completed
    failed
    stopped
}

struct public InspectorPrepareJob {
    generation : int
    patch : string
    view_text : string
    language : string
    stop : bool
}

[safe_when_uninitialized]
struct public InspectorPrepareEvent {
    kind : InspectorPrepareEventKind
    generation : int
    parsed : GitParsedPatch = GitParsedPatch()
    old_document : TextSourceDocument = TextSourceDocument()
    new_document : TextSourceDocument = TextSourceDocument()
    view_source : string
    error : string
    elapsed_usec : int64
}

struct public InspectorPrepareWorker {
    jobs : Channel?
    events : Channel?
    started : bool
    stopped : bool
}

def private emit_prepare_event(events : Channel?;
                               event : InspectorPrepareEvent) {
    events |> push_clone(event)
}

def private catch_prepare_panic(var failure : string&;
                                blk : block<() : void>) : bool {
    var caught = false
    try {
        invoke(blk)
    } recover {
        caught = true
        failure = this_context().last_exception
    }
    return caught
}

def private prepare_worker_main(var jobs : Channel?&;
                                var events : Channel?&) {
    this_context().name := "dasHerd_inspector_prepare"
    emit_prepare_event(events, InspectorPrepareEvent(
        kind = InspectorPrepareEventKind.ready))
    var stopping = false
    while (!stopping) {
        jobs |> pop_with_timeout_clone(250) $(first : InspectorPrepareJob#) {
            var job : InspectorPrepareJob
            job := first

            // Only the newest waiting projection matters. This bounds rapid
            // file-selection churn without trying to interrupt native parsing
            // which is already in progress.
            var draining = true
            while (draining) {
                draining = jobs |> try_pop_clone() $(next : InspectorPrepareJob#) {
                    delete job
                    job := next
                }
            }
            if (job.stop) {
                stopping = true
                delete job
                return
            }

            emit_prepare_event(events, InspectorPrepareEvent(
                kind = InspectorPrepareEventKind.started,
                generation = job.generation))
            let started_at = ref_time_ticks()
            var failure = ""
            var parsed = GitParsedPatch()
            var old_document = TextSourceDocument()
            var new_document = TextSourceDocument()
            var view_source = ""
            let failed = catch_prepare_panic(failure) {
                var prepared_patch <- git_parse_unified_patch(job.patch)
                if (!empty(prepared_patch.error)) {
                    failure = prepared_patch.error
                } else {
                    var prepared_old <- tree_sitter_source_document(
                        prepared_patch.old_aligned_text, job.language,
                        job.generation)
                    var prepared_new <- tree_sitter_source_document(
                        prepared_patch.new_aligned_text, job.language,
                        job.generation)
                    for (index in range(length(prepared_patch.rows))) {
                        if (prepared_patch.rows[index].old_kind
                                == GitDiffLineKind.removed) {
                            prepared_old.line_marks |> push(TextSourceLineMark(
                                line_index = index,
                                kind = TextSourceLineMarkKind.removed))
                        }
                        if (prepared_patch.rows[index].new_kind
                                == GitDiffLineKind.added) {
                            prepared_new.line_marks |> push(TextSourceLineMark(
                                line_index = index,
                                kind = TextSourceLineMarkKind.added))
                        }
                    }
                    view_source = !empty(job.view_text) ? job.view_text : prepared_patch.new_text
                    parsed <- prepared_patch
                    old_document <- prepared_old
                    new_document <- prepared_new
                }
            }
            let elapsed_usec = int64(get_time_usec(started_at))
            if (failed || !empty(failure)) {
                emit_prepare_event(events, InspectorPrepareEvent(
                    kind = InspectorPrepareEventKind.failed,
                    generation = job.generation,
                    error = failure,
                    elapsed_usec = elapsed_usec))
            } else {
                var event = InspectorPrepareEvent(
                    kind = InspectorPrepareEventKind.completed,
                    generation = job.generation,
                    view_source = view_source,
                    elapsed_usec = elapsed_usec)
                event.parsed <- parsed
                event.old_document <- old_document
                event.new_document <- new_document
                emit_prepare_event(events, event)
                git_parsed_patch_dispose(event.parsed)
                text_source_document_dispose(event.old_document)
                text_source_document_dispose(event.new_document)
            }
            git_parsed_patch_dispose(parsed)
            text_source_document_dispose(old_document)
            text_source_document_dispose(new_document)
            delete job
        }
    }
    emit_prepare_event(events, InspectorPrepareEvent(
        kind = InspectorPrepareEventKind.stopped))
    jobs |> release
    events |> release
}

def public inspector_prepare_worker_start(var worker : InspectorPrepareWorker) : bool {
    if (worker.started) return false
    worker.jobs = unsafe(channel_create())
    worker.events = unsafe(channel_create())
    worker.started = true
    worker.stopped = false
    var jobs_capture = worker.jobs
    var events_capture = worker.events
    new_thread() <| @capture(= jobs_capture, = events_capture) {
        prepare_worker_main(jobs_capture, events_capture)
    }
    return true
}

def public inspector_prepare_worker_queue(var worker : InspectorPrepareWorker;
                                          generation : int;
                                          patch, view_text, language : string) : bool {
    if (!worker.started || worker.stopped || worker.jobs == null) return false
    worker.jobs |> push_clone(InspectorPrepareJob(
        generation = generation, patch = patch, view_text = view_text,
        language = language))
    return true
}

def public inspector_prepare_worker_try_event(
        var worker : InspectorPrepareWorker;
        blk : block<(event : InspectorPrepareEvent#) : void>) : bool {
    if (!worker.started || worker.events == null) return false
    return worker.events |> try_pop_clone() $(event : InspectorPrepareEvent#) {
        if (event.kind == InspectorPrepareEventKind.stopped) {
            worker.stopped = true
        }
        invoke(blk, event)
    }
}

def public inspector_prepare_worker_shutdown(var worker : InspectorPrepareWorker;
                                             timeout_ms : int = 10000) : bool {
    if (!worker.started) return true
    if (!worker.stopped && worker.jobs != null) {
        worker.jobs |> push_clone(InspectorPrepareJob(stop = true))
    }
    let started_at = ref_time_ticks()
    while (!worker.stopped
            && int64(get_time_usec(started_at)) < int64(timeout_ms) * 1000l) {
        if (worker.events != null) {
            worker.events |> pop_with_timeout_clone(50) $(event : InspectorPrepareEvent#) {
                if (event.kind == InspectorPrepareEventKind.stopped) {
                    worker.stopped = true
                }
            }
        }
    }
    if (!worker.stopped) return false

    var jobs = worker.jobs
    var events = worker.events
    worker.jobs = null
    worker.events = null
    worker.started = false
    unsafe {
        if (jobs != null) jobs |> channel_remove
        if (events != null) events |> channel_remove
    }
    return true
}
