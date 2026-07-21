options gen2
options gc
options persistent_heap
options indenting = 4

module inspection_worker shared public

require daslib/jobque_boost
require daslib/rtti
require daslib/base64
require daslib/safe_addr
require imgui/imgui_text_tree_sitter
require imgui/imgui_markdown_tree_sitter
require stbimage/stbimage_boost
require ./file_inspection.das

let private MAX_BINARY_PREVIEW_PIXELS = 64l * 1024l * 1024l

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
    view_base64 : string
    view_byte_count : int
    view_available : bool
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
    view_document : TextSourceDocument = TextSourceDocument()
    markdown_document : RichDocument = RichDocument()
    markdown_view : MarkdownViewState = MarkdownViewState()
    changed_source_ranges : array<TextSourceRange>
    view_source : string
    preview_pixels : array<uint8>
    preview_width : int
    preview_height : int
    preview_byte_count : int
    preview_error : string
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

def private prepare_changed_source_ranges(source : string;
                                          lines : array<int> const) : array<TextSourceRange> {
    var result : array<TextSourceRange>
    if (empty(lines)) return <- result
    var target = 0
    var line_index = 0
    var line_start = 0
    peek_data(source) $(bytes) {
        for (at in range(length(bytes))) {
            if (int(bytes[at]) != '\n') continue
            while (target < length(lines) && lines[target] < line_index) {
                target++
            }
            if (target < length(lines) && lines[target] == line_index) {
                result |> push(TextSourceRange(start_byte = line_start, end_byte = at + 1))
                while (target < length(lines) && lines[target] == line_index) {
                    target++
                }
            }
            line_index++
            line_start = at + 1
        }
    }
    while (target < length(lines) && lines[target] < line_index) {
        target++
    }
    if (target < length(lines) && lines[target] == line_index) {
        result |> push(TextSourceRange(start_byte = line_start, end_byte = length(source)))
    }
    return <- result
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
            var view_document = TextSourceDocument()
            var markdown_document = RichDocument()
            var markdown_view = MarkdownViewState()
            var changed_ranges : array<TextSourceRange>
            var view_source = ""
            var preview_pixels : array<uint8>
            var preview_width = 0
            var preview_height = 0
            var preview_error = ""
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
                    prepared_old.line_numbers |> reserve(length(prepared_patch.rows))
                    prepared_new.line_numbers |> reserve(length(prepared_patch.rows))
                    for (index in range(length(prepared_patch.rows))) {
                        prepared_old.line_numbers |> push(
                            prepared_patch.rows[index].old_line)
                        prepared_new.line_numbers |> push(
                            prepared_patch.rows[index].new_line)
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
                    view_source = (job.view_available || !empty(job.view_text)) ? job.view_text : prepared_patch.new_text
                    if (!prepared_patch.binary) {
                        var prepared_view <- tree_sitter_source_document(
                            view_source, job.language, job.generation)
                        prepared_view.line_marks |> reserve(
                            length(prepared_patch.new_changed_lines))
                        for (line in prepared_patch.new_changed_lines) {
                            prepared_view.line_marks |> push(TextSourceLineMark(
                                line_index = line, kind = TextSourceLineMarkKind.added))
                        }
                        view_document <- prepared_view
                        if (job.language == "markdown") {
                            var prepared_markdown <- md_parse(view_source)
                            if (!prepared_markdown.ok) {
                                failure = (empty(prepared_markdown.error)
                                    ? "Markdown preparation failed"
                                    : prepared_markdown.error)
                            } else {
                                var prepared_markdown_view = MarkdownViewState()
                                let _ = markdown_tree_sitter_prepare(prepared_markdown,
                                    prepared_markdown_view, job.generation)
                                markdown_document <- prepared_markdown
                                markdown_view <- prepared_markdown_view
                                changed_ranges <- prepare_changed_source_ranges(
                                    view_source, prepared_patch.new_changed_lines)
                            }
                        }
                    }
                    if (prepared_patch.binary) {
                        if (job.view_byte_count > MAX_FILE_PREVIEW_RAW_BYTES) {
                            preview_error = "Binary preview exceeds the 8 MiB raw-byte limit"
                        } elif (empty(job.view_base64)) {
                            preview_error = "Binary preview bytes are unavailable for this comparison"
                        } else {
                            var inscope encoded : array<uint8>
                            let decoded = base64_decode(job.view_base64, encoded)
                            if (decoded < 0 || decoded != job.view_byte_count) {
                                preview_error = "Binary transport failed exact-byte validation"
                            } else {
                                var width, height, channels : int
                                let image_info = stbi_info_from_memory(
                                    unsafe(addr<uint8 const?>(encoded[0])),
                                    length(encoded), safe_addr(width),
                                    safe_addr(height), safe_addr(channels))
                                if (image_info == 0) {
                                    preview_error = stbi_failure_reason()
                                } elif (width <= 0 || height <= 0 ||
                                        int64(width) * int64(height)
                                            > MAX_BINARY_PREVIEW_PIXELS) {
                                    preview_error = "Image dimensions exceed the 64 megapixel preview limit"
                                } else {
                                    var image : Image
                                    let (image_ok, image_error) = image->load_from_memory(encoded, 4)
                                    if (image_ok) {
                                        preview_width = image.width
                                        preview_height = image.height
                                        preview_pixels <- image.bytes
                                    } else {
                                        preview_error = image_error
                                    }
                                }
                            }
                        }
                    }
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
                    preview_width = preview_width,
                    preview_height = preview_height,
                    preview_byte_count = job.view_byte_count,
                    preview_error = preview_error,
                    elapsed_usec = elapsed_usec)
                event.parsed <- parsed
                event.old_document <- old_document
                event.new_document <- new_document
                event.view_document <- view_document
                event.markdown_document <- markdown_document
                event.markdown_view <- markdown_view
                event.changed_source_ranges <- changed_ranges
                event.preview_pixels <- preview_pixels
                emit_prepare_event(events, event)
                git_parsed_patch_dispose(event.parsed)
                text_source_document_dispose(event.old_document)
                text_source_document_dispose(event.new_document)
                text_source_document_dispose(event.view_document)
                rich_document_dispose(event.markdown_document)
                markdown_view_dispose(event.markdown_view)
                delete event.changed_source_ranges
                delete event.preview_pixels
            }
            git_parsed_patch_dispose(parsed)
            text_source_document_dispose(old_document)
            text_source_document_dispose(new_document)
            text_source_document_dispose(view_document)
            rich_document_dispose(markdown_document)
            markdown_view_dispose(markdown_view)
            delete changed_ranges
            delete preview_pixels
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
                                          patch, view_text, language : string;
                                          view_base64 : string = "";
                                          view_byte_count : int = 0;
                                          view_available : bool = false) : bool {
    if (!worker.started || worker.stopped || worker.jobs == null) return false
    worker.jobs |> push_clone(InspectorPrepareJob(
        generation = generation, patch = patch, view_text = view_text,
        language = language, view_base64 = view_base64,
        view_byte_count = view_byte_count,
        view_available = view_available))
    return true
}

def public inspector_prepare_worker_take_event(var worker : InspectorPrepareWorker;
                                               var event : InspectorPrepareEvent) : bool {
    if (!worker.started || worker.events == null) return false
    delete event
    let popped = _builtin_channel_try_pop(worker.events) $(data) {
        event := *(unsafe(reinterpret<InspectorPrepareEvent?#>(data)))
    }
    if (!popped) return false
    if (event.kind == InspectorPrepareEventKind.stopped) {
        worker.stopped = true
    }
    return true
}

def public inspector_prepare_worker_try_event(
                                              var worker : InspectorPrepareWorker;
                                              blk : block<(event : InspectorPrepareEvent#) : void>) : bool {
    var event : InspectorPrepareEvent
    if (!inspector_prepare_worker_take_event(worker, event)) return false
    invoke(blk, unsafe(reinterpret<InspectorPrepareEvent -& -const#>(event)))
    delete event
    return true
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
