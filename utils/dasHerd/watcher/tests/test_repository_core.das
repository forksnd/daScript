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
        repository_parse_status("## main-feature...origin/main [ahead 2, behind 3]\r\nM  staged file.das\r\n M unstaged file.das\r\n?? new file.txt\r\nUU conflict.txt\r\nR  old name -> new name\r\n", worktrees[0], files)
        t |> equal(worktrees[0].ahead, 2)
        t |> equal(worktrees[0].behind, 3)
        t |> equal(worktrees[0].staged, 3)
        t |> equal(worktrees[0].unstaged, 2)
        t |> equal(worktrees[0].untracked, 1)
        t |> equal(worktrees[0].conflicted, 1)
        t |> equal(length(files), 5)
        t |> equal(files[4].old_path, "old name")
        t |> equal(files[4].path, "new name")
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
        t |> equal(argv[7], "--no-index")
        t |> equal(argv[9], "--unified=3")
        var forces_text = false
        for (argument in argv) {
            if (argument == "--text") {
                forces_text = true
            }
        }
        t |> equal(forces_text, false, "Git retains native binary detection")
        t |> equal(argv[length(argv) - 2], "NUL")
        t |> equal(argv[length(argv) - 1], "new file.das")

        let staged_file = GitFileState(code = "M ", path = "staged.das")
        expected_exit_code = -1l
        let staged_error = repository_build_file_diff_argv(
            "D:/Work/tree", staged_file, "staged", argv, expected_exit_code)
        t |> equal(staged_error, "")
        t |> equal(expected_exit_code, 0l)
        t |> equal(argv[7], "--cached")
        t |> equal(argv[9], "--unified=3")
        t |> equal(argv[length(argv) - 1], "staged.das")

        let full_error = repository_build_file_diff_argv(
            "D:/Work/tree", staged_file, "staged", argv,
            expected_exit_code, "full")
        t |> equal(full_error, "")
        t |> equal(argv[9], "--unified=1000000")

        let expanded_error = repository_build_file_diff_argv(
            "D:/Work/tree", staged_file, "staged", argv,
            expected_exit_code, "expanded")
        t |> equal(expanded_error, "")
        t |> equal(argv[9], "--unified=20")
    }
}
