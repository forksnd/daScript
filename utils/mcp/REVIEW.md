# mcp Code Review Checklist

**Read `REVIEW_COMMON.md` (repo root) first - its contract binds this checklist.** Architecture doc:
`README.md`. Planned work: `ROADMAP.md`.

**A diff that adds a top-level file under `utils/mcp/` that the shipped SDK runs or loads -
`main.das` reaches it, the `.mcp.json` entry `setup.das` writes or the `.cmd` launcher runs
it, or it has its own `main` that something in the shipped SDK runs - also adds it to an
`install(FILES ...)` block with `DESTINATION utils/mcp` in `CMakeLists.txt` (repo root), in
the same change.** `tools/` and `subtools/` are globbed; a top-level file left out of the
list is absent in the shipped SDK while the in-tree server keeps working.

**Weakening any case in `test_tools.das` that pins which comments the formatter keeps is a
defect.**
