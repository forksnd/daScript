options gen2
options rtti

require dastest/testing_boost public
require ../repository_core.das
require strings

[test]
def test_repository_porcelain_parsing(t : T?) {
    t |> run("Git 2.17 worktree and status porcelain become structured state") <| @(t : T?) {
        let inventory = "worktree D:/Work/daScript\r\nHEAD 0123456789012345678901234567890123456789\r\nbranch refs/heads/main-feature\r\n\r\nworktree D:/Work/daScript/.codex/worktrees/space tree\r\nHEAD abcdefabcdefabcdefabcdefabcdefabcdefabcd\r\ndetached\r\nlocked reason\r\nprunable reason\r\n\r\n"
        var worktrees : array<WorktreeState>
        repository_parse_inventory(inventory, worktrees)
        t |> equal(length(worktrees), 2)
        t |> success(worktrees[0].is_main, "first porcelain entry is the visually distinct main checkout")
        t |> equal(worktrees[0].branch, "main-feature")
        t |> equal(worktrees[1].path, "D:/Work/daScript/.codex/worktrees/space tree")
        t |> success(worktrees[1].detached && worktrees[1].locked && worktrees[1].prunable,
            "detached, locked, and prunable markers are retained")

        var files : array<GitFileState>
        repository_parse_status("## main-feature...origin/main [ahead 2, behind 3]\r\nM  staged file.das\r\n M unstaged file.das\r\n?? new file.txt\r\nUU conflict.txt\r\nR  old name -> new name\r\n M literal -> name.txt\r\n", worktrees[0], files)
        t |> equal(worktrees[0].ahead, 2)
        t |> equal(worktrees[0].behind, 3)
        t |> equal(worktrees[0].staged, 3)
        t |> equal(worktrees[0].unstaged, 3)
        t |> equal(worktrees[0].untracked, 1)
        t |> equal(worktrees[0].conflicted, 1)
        t |> equal(length(files), 6)
        t |> equal(files[4].old_path, "old name")
        t |> equal(files[4].path, "new name")
        t |> equal(files[5].old_path, "")
        t |> equal(files[5].path, "literal -> name.txt")
        t |> equal(files[0].worktree_path, "D:/Work/daScript")
    }
    t |> run("unquoted porcelain retains UTF-8 file paths") <| @(t : T?) {
        var worktree = WorktreeState(path = "D:/Work/fixture")
        var files : array<GitFileState>
        repository_parse_status("## fixture\r\n M unicode-Привет.txt\r\n",
            worktree, files)
        t |> equal(length(files), 1)
        t |> equal(files[0].path, "unicode-Привет.txt")
    }
}

[test]
def test_repository_untracked_diff_contract(t : T?) {
    t |> run("Untracked diffs accept git --no-index exit 1") <| @(t : T?) {
        let file = GitFileState(code = "??", path = "new file.das")
        var argv : array<string>
        var expected_exit_code = 0l
        let error = repository_build_file_diff_argv(
            "D:/Work/tree", file, "working", argv, expected_exit_code)
        t |> equal(error, "")
        t |> equal(expected_exit_code, 1l)
        t |> equal(argv[1], "--literal-pathspecs")
        t |> equal(argv[8], "--no-index")
        t |> equal(argv[10], "--unified=3")
        var forces_text = false
        for (argument in argv) {
            if (argument == "--text") {
                forces_text = true
            }
        }
        t |> equal(forces_text, false, "Git retains native binary detection")
        let null_device = get_platform_name() == "windows" ? "NUL" : "/dev/null"
        t |> equal(argv[length(argv) - 2], null_device)
        t |> equal(argv[length(argv) - 1], "new file.das")

        let staged_file = GitFileState(code = "M ", path = "staged.das")
        expected_exit_code = -1l
        let staged_error = repository_build_file_diff_argv(
            "D:/Work/tree", staged_file, "staged", argv, expected_exit_code)
        t |> equal(staged_error, "")
        t |> equal(expected_exit_code, 0l)
        t |> equal(argv[8], "--cached")
        t |> equal(argv[10], "--unified=3")
        t |> equal(argv[length(argv) - 1], "staged.das")

        let full_error = repository_build_file_diff_argv(
            "D:/Work/tree", staged_file, "staged", argv,
            expected_exit_code, "full")
        t |> equal(full_error, "")
        t |> equal(argv[10], "--unified=1000000")

        let expanded_error = repository_build_file_diff_argv(
            "D:/Work/tree", staged_file, "staged", argv,
            expected_exit_code, "expanded")
        t |> equal(expanded_error, "")
        t |> equal(argv[10], "--unified=20")

        let magic_file = GitFileState(code = " M", path = ":(glob)**.das")
        let magic_error = repository_build_file_diff_argv(
            "D:/Work/tree", magic_file, "working", argv, expected_exit_code)
        t |> equal(magic_error, "")
        t |> equal(argv[1], "--literal-pathspecs")
        t |> equal(argv[length(argv) - 1], ":(glob)**.das")

        let staged_view_error = repository_staged_file_view_argv(
            "D:/Work/tree", "staged.das", "D:/Temp/staged-view.raw", argv)
        t |> equal(staged_view_error, "")
        t |> equal(argv[length(argv) - 2],
            "--output=D:/Temp/staged-view.raw")
        t |> equal(argv[length(argv) - 1], ":0:staged.das")
    }
}

[test]
def test_repository_relative_file_path_contract(t : T?) {
    t |> run("file previews cannot escape their observed worktree") <| @(t : T?) {
        t |> success(repository_valid_relative_file_path("src/main.das"))
        t |> success(!repository_valid_relative_file_path("../outside.txt"))
        t |> success(!repository_valid_relative_file_path("src/../../outside.txt"))
        t |> success(!repository_valid_relative_file_path("/etc/passwd"))
        t |> success(!repository_valid_relative_file_path("\\\\server\\share\\file.txt"))
        if (get_platform_name() == "windows") {
            t |> success(!repository_valid_relative_file_path("C:\\Windows\\win.ini"))
            t |> success(!repository_valid_relative_file_path("C:drive-relative.txt"))
        }
    }
}

[test]
def test_repository_text_read_contract(t : T?) {
    t |> run("empty files remain distinct from read failures") <| @(t : T?) {
        let temp = create_temp_directory_result("dasherd_repository_text_")
        if (!(temp is value)) {
            t |> failure("could not create temporary test directory")
            return
        }
        let root = temp as value
        let empty_path = path_join(root, "empty.txt")
        t |> success(fwrite(empty_path, ""), "empty fixture is written")

        var text : string
        t |> equal(repository_read_text_file(empty_path, 1024, text), "")
        t |> equal(text, "", "an opened empty file is a successful preview")
        t |> success(!empty(repository_read_text_file(root, 1024, text)),
            "a path that cannot be read as a regular file reports an error")

        let bytes_path = path_join(root, "bytes.bin")
        t |> success(fwrite(bytes_path, "abc"), "byte fixture is written")
        var preview_path = ""
        t |> equal(repository_working_preview_path(
            root, "bytes.bin", preview_path), "")
        t |> equal(preview_path, bytes_path)
        t |> equal(repository_read_text_file(bytes_path, 2, text),
            "file exceeds the raw-byte View limit")
        t |> equal(text, "", "oversized mapped text is never materialized")
        t |> equal(repository_read_text_file(bytes_path, 3, text), "")
        t |> equal(text, "abc", "mapped text at the raw-byte limit is materialized")
        var bytes : array<uint8>
        var byte_count = 0
        t |> equal(repository_read_file_bytes(
            bytes_path, 3, bytes, byte_count), "")
        t |> equal(byte_count, 3)
        t |> equal(string(bytes), "abc")

        rmdir_rec(root)
    }
}
