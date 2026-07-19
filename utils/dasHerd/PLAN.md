# dasHerd

Status: working design

`dasHerd` is a local control plane for coding-agent workspaces. It launches and
reattaches to agent sessions, gives every session an owned worktree, presents
the complete state of that worktree, and unifies local Git and GitHub review
workflows.

The name is intentional: this is an agent herder, not a general-purpose IDE.
The initial implementation may force a narrow workflow when that makes state
unambiguous and reliable.

## Product contract

- One writable agent session owns one worktree.
- dasHerd creates the worktree, branch, session, and agent process together.
- The main checkout can be adopted as an explicit special workspace.
- Relative paths emitted by an agent resolve against its session worktree.
- Closing or restarting the dasHerd UI does not stop attached agents.
- Git, process, and GitHub state are observed independently; terminal text is
  never the sole source of truth.
- Source positions use workspace identity, worktree-relative path, revision,
  and UTF-8 byte or line range. Screen coordinates are never persisted.
- Direct editing starts with the PR body in Markdown. Code views are initially
  read-only, selectable, and addressable.
- Accessibility is a default, not a preference hidden in settings: terminal
  text starts at 150%, exposes a visible 50-300% zoom control plus Ctrl+wheel,
  and the desktop shell opens maximized.

## First vertical slice: terminal

The terminal is the first subsystem because it proves the hardest lifetime
contract and becomes the spine of every later workspace feature.

The existing dasImgui text stack already provides much of the presentation
foundation: grapheme grouping, 0/1/2-cell widths, selection, clipboard, font
roles, URI recognition, wrapping, source ranges, and deterministic interaction
tests. The substantial missing pieces are terminal emulation, pseudoterminal
ownership, durable session hosting, and terminal-specific input modes.

The split is strict:

1. `modules/dasTerminal` is renderless. It owns terminal protocol and state.
2. `modules/dasImgui/terminal` renders and interacts with terminal snapshots.
3. `utils/dasHerd` owns workspace policy, session discovery, and application UI.

This follows the shared text architecture recorded in
[`modules/dasImgui/plans/text-components-architecture.md`](../../modules/dasImgui/plans/text-components-architecture.md).

## Architecture

```text
dasHerd UI
   |
   | local IPC: attach, input lease, resize, snapshots, updates
   v
per-session host process
   |
   | ConPTY on Windows; PTY on POSIX
   v
agent process: Codex / Claude Code / Pi / OpenCode / shell

per-session host
   +-- dasTerminal emulator and scrollback
   +-- child/PTY lifetime and exit status
   +-- worktree/session metadata
   +-- versioned attach endpoint
   +-- bounded transcript and diagnostics
```

The UI never owns the agent process or pseudoterminal. Detaching closes only
the IPC connection. A session host stays alive until the agent exits or the
user explicitly terminates the session.

One host process per session provides fault isolation. dasHerd discovers hosts
through persisted session records and verifies them by connecting to their IPC
endpoints. A later supervisor may monitor and restart hosts, but no design may
claim that a live PTY can be recovered after its owning host has died.

Only one attached client holds the input lease. Additional clients may observe
the same session read-only. Lease transfer is explicit so two windows cannot
interleave keystrokes.

## `dasTerminal` responsibilities

The renderless module owns:

- a terminal screen grid, primary and alternate screens, cursor, saved cursor,
  tab stops, margins, modes, and bounded scrollback;
- incremental VT/ANSI parsing across arbitrarily split byte chunks;
- UTF-8 decoding, combining graphemes, wide cells, and continuation cells;
- SGR attributes and indexed/true-color values without UI colors or fonts;
- OSC metadata, including OSC 7 current-directory hints and OSC 8 hyperlinks;
- resize/reflow policy and terminal buffer revisions;
- semantic selection/copy queries over terminal cells;
- normalized input operations for keys, UTF-8 text, paste, focus, and mouse;
- a platform-neutral pseudoterminal/session interface;
- ConPTY and POSIX PTY implementations behind that interface.

It does not know about ImGui, pixels, textures, worktrees, GitHub, or agent
brands.

PTY transport and VT emulation remain separate interfaces so the engine can be
replaced without disturbing the renderer or dasHerd. T0 starts with a small
in-tree state machine behind the canonical snapshot contract. This is not a
commitment to implement every terminal escape ourselves: the conformance
corpus is the decision instrument. If the required compatibility surface grows
faster than the engine, evaluate `libghostty-vt`, `libtsm`, and `libvterm`
behind the same tests.

The initial dependency decision favors the in-tree slice because it gives us:

- one exact snapshot and future IPC schema rather than adapting a library's
  renderer-shaped callbacks;
- C++11, MSVC, and daScript bindings with no new build or license surface;
- explicit unknown-sequence diagnostics instead of silently accepting an
  unmeasured compatibility claim;
- a measurable point at which adopting a proven engine becomes cheaper.

The first research-derived test sources are:

- [Termless](https://termless.dev/) for structured cross-backend semantic test
  shape;
- [terminfo.dev](https://terminfo.dev/) for a machine-readable capability and
  probe catalog (author original vectors or preserve its CC BY attribution);
- [xterm.js escape fixtures](https://github.com/xtermjs/xterm.js/tree/master/test/fixtures/escape_sequence_files)
  for replay streams, with the limitation that its text goldens omit styles,
  cursor, and modes;
- [Charmbracelet `x/vttest`](https://pkg.go.dev/github.com/charmbracelet/x/vttest)
  for the closest precedent to the canonical rich snapshot schema;
- `esctest2`, classic `vttest`, Ghostty fuzz seeds, Microsoft Terminal adapter
  tests, and Neovim/libvterm tests as external oracles and robustness inputs,
  not copied test code.

## dasImgui terminal responsibilities

The dasImgui counterpart owns:

- measuring and drawing a terminal snapshot with the shared monospace/font
  stack;
- cursor, selection, hover, search highlights, and visible scrollback;
- keyboard, text, paste, focus, mouse-mode, wheel, and resize input mapping;
- color-theme mapping from terminal attributes to ImGui colors;
- OSC 8 link hit testing;
- fallback URI recognition through the existing `text_uris` service;
- context actions such as copy, open link, copy link, and create source anchor;
- stable screen geometry for automation and deterministic headless tests.

Terminal cells remain the authority for layout. Font fallback and color emoji
are drawing concerns inside allocated cells and may not move later columns.

## Links and source anchors

Links have three levels:

1. OSC 8 hyperlinks are authoritative when emitted by the child process.
2. URI-shaped text is recognized by the shared `text_uris` fallback.
3. dasHerd recognizes workspace file references such as `foo.das:42` or
   `/examples/foo/bar.das` and resolves them against the session worktree.

The terminal renderer reports a clicked payload and terminal location; dasHerd
decides whether to open a browser, a source view, a Markdown view, or another
workspace action.

Agent skills should eventually ask supported agents to emit canonical
worktree-relative file references or structured hyperlinks. The tool may also
inject a selected source anchor into the terminal input as a normal agent
prompt. Structured hints improve the experience but never replace observable
Git/worktree state.

## Session record

Each persisted session record contains at least:

- stable session ID and display name;
- agent profile and launch command;
- repository and worktree roots;
- branch and intended base branch;
- host PID, IPC endpoint, creation time, and last observed state;
- terminal dimensions and scrollback policy;
- optional PR repository/number;
- paths to bounded transcript and diagnostic logs.

Runtime state belongs in a user-data location, not in a deployment directory
or the repository. Diagnostic logs use a dedicated `logs/` directory.

## Demonstrable milestones

### T0: emulator contract and fixtures (in progress)

- Define renderless cell, line, cursor, style, snapshot, update, and revision
  contracts.
- Select or reject candidate VT engines with a small written rationale.
- Feed chunked deterministic fixtures for text, color, cursor movement,
  alternate screen, clear operations, OSC 7, OSC 8, Unicode, and resizing.

The canonical test surface is:

```text
create(columns, rows)
feed(bytes)
resize(columns, rows)
encode_key(event)
encode_paste(text)
send_mouse(event)
snapshot()
drain_replies()
take_events()
```

The first implementation already covers create/feed/resize/snapshot/replies/
events and tests one-shot, byte-at-a-time, and seeded random chunk equivalence.
The semantic table pins the selected Core cursor, erase, and SGR controls,
including wide-cell invariants when erasing or overwriting a continuation cell.
Normalized key and paste encoding is mode-sensitive and covered in both C++ and
daScript. Mouse encoding joins the contract with the renderer/ConPTY slice.

### T1: local interactive terminal (in progress)

- Launch PowerShell through ConPTY on Windows. (implemented and covered by a
  semantic-handshake process test; `pwsh` is preferred with Windows PowerShell
  fallback)
- Render it through the dasImgui terminal view. (implemented as
  `imgui/imgui_terminal` with a runnable headless/windowed PowerShell example)
- Support typing, paste, resize, selection, clipboard, colors, cursor, and
  scrollback. (first interactive slice implemented; deterministic interaction
  now covers measured resize, click-to-focus, synthesized typing, and semantic
  screen output; paste/scrollback/selection/clipboard coverage and
  terminal-specific selection polish remain)
- Run a full-screen agent TUI rather than validating only line-oriented shells.

### T2: attach and detach

- Move PTY and emulator ownership into a per-session host executable.
- Attach the dasHerd UI over local IPC.
- Close the UI while the agent continues working, reopen it, and recover the
  exact current screen plus scrollback.
- Enforce and visibly transfer the single input lease.

### T3: links and source handoff

- Activate OSC 8 and fallback URI links.
- Resolve worktree-relative file references.
- Open a full source or Markdown view at the referenced range.
- Select a source range and send its canonical anchor to the active agent.

### T4: first dasHerd workspace shell

- List sessions with agent, worktree, branch, dirty state, process state, and
  last activity.
- Create a worktree and launch an agent from a forced profile.
- Switch between terminal and file inspection without losing context.
- Show changed files, full files, working diff, and aggregate base diff.

### T5: local PR workflow

- Associate or create a PR for the workspace.
- Edit a locally autosaved Markdown PR-body draft with explicit publish.
- Detect remote-body conflicts before publishing.
- Show CI, review threads, and Copilot activity.
- Support reply, resolve, and deliberate batch resolve operations.

## Testing rules

- VT parsing tests split every interesting sequence at multiple byte
  boundaries; correctness cannot depend on read chunking.
- Emulator tests are renderless and use exact snapshots/revisions.
- PTY process tests use deterministic child probes and explicit handshakes.
- UI tests wait for terminal buffer, layout, hover, selection, and input
  revisions. Timeouts are deadlock guards, never test semantics.
- Attach/detach tests prove the child PID survives UI exit.
- Tests cover non-ASCII text, combining graphemes, CJK, emoji, CR/LF variants,
  bracketed paste, alternate screen, and resize during output.
- Logs and abandoned probe artifacts are cleaned when work completes.

## Explicit non-goals for the first slice

- SSH sessions; the protocol must permit a remote host in v2, but v1 is local.
- A general code editor or language server client.
- Per-agent authorship inside a shared worktree.
- Recovering a live terminal after the session-host process itself dies.
- Inferring authoritative agent state by scraping prose from terminal output.
- Implementing every obscure VT escape before Codex/Claude/shell workflows are
  demonstrably correct.

## Immediate next step

Continue T1 from the working local terminal:

1. extend deterministic headless interaction tests from the landed
   resize/focus/typing flow to paste, scrollback, selection, and clipboard;
2. run a full-screen agent TUI through the embedded-terminal example;
3. close the emulator gaps that TUI exposes, then pin them as semantic tests;
4. move PTY/emulator ownership into the first detachable session host.

## Decision log

- 2026-07-18: Name the tool `dasHerd`.
- 2026-07-18: Force one writable agent session per worktree.
- 2026-07-18: Begin with the terminal vertical slice.
- 2026-07-18: Keep `dasTerminal` renderless and put rendering in dasImgui.
- 2026-07-18: The session host, not the UI, owns the child process and PTY.
- 2026-07-18: Start direct editing with a local Markdown PR-body draft.
- 2026-07-18: Let semantic conformance tests, replay streams, live PTY tests,
  and chunk-invariance/fuzz tests shape the terminal implementation.
- 2026-07-18: Start T0 with an in-tree C++11 state machine behind a replaceable
  canonical snapshot API; reconsider a library when measured coverage demands
  it.
- 2026-07-18: Keep the first live daScript adapter thin: one terminal handle
  pumps a separately implemented PTY into the emulator and automatically routes
  terminal replies, while exposing process status and explicit termination.
- 2026-07-18: Land the first dasImgui terminal view with snapshot-owned cell
  geometry, semantic input encoding, scrollback, and clipboard selection before
  introducing the detachable host boundary.
- 2026-07-18: Gate terminal UI automation on observable geometry, focus,
  input/content revisions, and semantic screen text; never use frame counts or
  sleeps as interaction-success criteria.
- 2026-07-18: Default dasHerd terminal typography to 150% with visible zoom and
  Ctrl+wheel, and maximize the desktop terminal shell on launch.
