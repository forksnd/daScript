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
