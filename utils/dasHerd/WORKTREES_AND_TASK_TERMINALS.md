# Worktrees and task terminals

Status: discussion draft

This is a focused design document for the first dasHerd workspace application.
It is intentionally separate from `PLAN.md`: the worktree, Git, task-terminal,
and automation contracts have enough nuance that they must be discussed one
section at a time.

No section below authorizes implementation merely by appearing here. Discuss
and settle each section before implementing the stage that depends on it.

## Established direction

- A detachable command host is the foundation beneath dasHerd. Callers do not
  own launched OS processes directly: JIT invocations, Git commands, setup
  scripts, shells, tests, and agents all execute as hosted command sessions.
- Every hosted command session has process state, a terminal, and a durable
  log whether or not its terminal is currently visible. The UI and automation
  attach to the session; detaching or closing the UI does not stop it.
- Git observation is asynchronous from the first implementation. No Git
  command runs on the UI/render thread.
- The application lists the main checkout and linked worktrees, and inspects
  the selected tree.
- The repository base and the main checkout are distinct. For example, a
  repository may use `origin/master` as its base while its main checkout is on
  a feature branch. The UI must show both values rather than label the main
  checkout "master".
- A worktree can have a PowerShell terminal rooted in that tree and later an
  agent session.
- Worktree creation supports a setup recipe. The built-in MCP bootstrap
  (`utils/mcp/setup.das`) is the default recipe; `None` and a user-specified
  recipe are alternatives.
- Removal is preceded by an explainable "is it safe to remove?" evaluation.
- Terminals are a general control and debugging surface. Git and setup work
  can have visible, colored, scrollable terminals whose output is also logged.
- Every terminal is addressable through live commands. A human, Codex, another
  agent, or later automation uses the same read/write terminal operations.
- Execution is not assumed to be local forever. A repository, worktree, Git
  process, setup task, shell, or agent may live behind SSH. The UI must always
  provide a route to the terminal that owns or can repair the operation.

## Section 0: detachable command host

Discussion status: foundational direction agreed; contract details open

The host is not an agent-specific service added late in the product. It is the
single execution building block used whenever dasHerd calls something. A JIT
compiler run, a Git query, a setup recipe, a test, an interactive PowerShell,
and a long-lived coding agent differ in profile and lifetime, not in whether
they receive a hosted session.

The caller supplies a target, working directory, structured command and
arguments, environment additions, terminal geometry, and retention policy.
The host owns:

- process creation and lifetime;
- the PTY and renderless terminal emulator;
- input serialization;
- terminal revisions, scrollback, title, CWD hints, and process status;
- an append-only session log and bounded retained terminal state;
- attach/detach and recovery after the UI disconnects;
- explicit termination and final exit status.

The same session remains inspectable while running and after exit. A terminal
window is one client of that state, not its owner. Live commands, the human UI,
and future agent routing attach through the same session contract.

Structured callers such as the Git observer still receive an explicit command
result and raw/captured output suitable for parsing. They do not reconstruct a
result by scraping rendered cells. Their hosted terminal and log remain
available for diagnostics even when the normal UI shows only structured state.

The host must survive dasHerd UI exit from its first useful version. The later
SSH design decides where the host process physically runs, but not whether the
session has this contract.

Questions to settle before implementation:

- Host topology: one OS host process per command session, a supervisor with
  child sessions, or a small supervisor plus isolated per-session hosts.
- The local attach protocol, discovery records, versioning, and authentication.
- Log format, rotation, crash consistency, terminal replay, and input-payload
  privacy.
- Whether very short structured commands allocate a PTY or use a capture mode
  that still exposes the same terminal/log/session API.
- Session retention after successful exit, failed exit, and explicit cleanup.
- What state can be reconstructed after the command host itself crashes; a PTY
  cannot be recovered if its owning process is gone.

### Chunk 0: measure terminal execution overhead

Discussion status: agreed

Before choosing a capture fast path or optimizing host topology, measure local
Git through the terminal stack that already exists. SSH Git is a separate Git
installation and execution target; it gets the same probe after SSH exists.

Compare the same forced-color Git argv through:

1. direct process capture;
2. direct process capture plus a raw durable log;
3. ConPTY, terminal emulation, and a raw durable log.

The probe records launch-to-first-output, launch-to-exit, final-output drain,
terminal poll/feed time, log-write time, byte count, and exit code. It separates
the cold first run from warmed median and p95 samples. The command set contains
a process-launch baseline, a representative colored repository status, and an
intentional failure to prove nonzero exit propagation.

One windowed run displays the real terminal with color and scrolling while
retaining its raw log and final exit code. Repeated timing runs are headless.
This first comparison measures in-process ConPTY/emulator/log overhead; the
same probe is rerun through detach/IPC after the minimal host exists.

No capture-mode or short-command optimization is selected until these numbers
exist.

The first measurement found raw logging indistinguishable from noise and
in-process terminal overhead of roughly 5-16 ms at process exit. dasHerd keeps
one terminal execution path; it does not add a direct-capture fast path. The
conservative post-exit drain must not block result delivery or scheduling.

### Chunk 1: measure an isolated session process

Discussion status: agreed; first measurement complete

The first process-isolation probe launches a fresh interpreted daScript worker
for every session. The worker owns ConPTY, terminal emulation, and logging and
acknowledges readiness over a pipe. This deliberately simple prototype measures
the complete extra process and runtime boundary before it becomes architecture.

The fresh worker added roughly 334-345 ms at median and peaked near 71 MiB
working set in one status sample. Almost all latency occurred before the worker
ready acknowledgement; terminal execution inside the worker continued to match
the in-process lane. Therefore an interpreted worker compiled afresh per Git
query is rejected. Per-session isolation itself remains open pending the same
probe with a small AOT/native host; a persistent per-worktree host is the next
fallback, not an unmeasured optimization.

## Section 1: repository, base, and worktree identity

Discussion status: open

The first model needs to distinguish:

- an execution target, initially the local machine and later an SSH target;
- a repository identity on that target;
- the repository's configured or detected base ref;
- the main checkout, which is a special worktree but not necessarily the base
  branch;
- linked worktrees discovered from Git;
- worktrees created and managed by dasHerd;
- a stable session identity independent of a branch name.

A path alone is not a global identity once SSH exists. Candidate identities
therefore include at least `target ID + repository Git-common-dir` for a
repository and `repository ID + canonical worktree path` for an initial
worktree identity. Session records get their own generated IDs.

Questions to settle:

- Is the base ref stored per repository, per worktree, or both with an
  override?
- Are externally created worktrees fully manageable or initially read-only?
- Where are default managed worktrees placed, and how editable is that path?
- Do we show a virtual base row, or show the base only in the repository header
  until it has a real worktree?

## Section 2: asynchronous Git observation

Discussion status: open

Background observation needs structured, non-interactive results. It submits
commands through the detachable host, uses Git's porcelain/plumbing output
without color, and publishes versioned snapshots to the UI. Its hosted
terminal and log remain addressable for diagnostics, but terminal text is
never the authority for repository state.

The initial observer covers:

- repository and Git-common-dir discovery;
- default-base detection, preferring `origin/HEAD` when available;
- main and linked worktree inventory;
- branch or detached HEAD and commit identity;
- staged, unstaged, untracked, renamed, and conflicted state;
- ahead/behind or containment relative to the selected base;
- locked and prunable worktree state;
- errors, last-success time, and whether a result is refreshing or stale.

The local Git is currently 2.17.1, so the implementation cannot require modern
options such as `git worktree list --porcelain -z` or
`git branch --show-current`. Managed paths can reject CR/LF, while spaces and
Unicode remain supported.

Worker rules:

- invoke Git through hosted sessions with argument arrays, never interpolated
  shell commands;
- send immutable/versioned results back to the UI thread;
- discard stale generations and coalesce redundant refresh requests;
- allow independent reads while serializing repository mutations;
- refresh after every mutation;
- do not fetch the network implicitly.

Questions to settle:

- Refresh triggers and cadence for selected and unselected worktrees.
- Whether a manual refresh means local observation only, with a separate
  explicit Fetch action.
- How much per-command timing and argv detail is retained for the Git debug
  window.

## Section 3: execution targets and command sessions

Discussion status: open

Git cannot be designed as "run a Windows process in this local directory."
Commands run against an execution target. The first target implementation is
local; an SSH target later provides the same operations:

- execute a structured non-interactive command and return a result;
- start or attach to a PTY terminal;
- identify a working directory on that target;
- observe process state and exit status;
- stream updates and retain diagnostics;
- terminate only through an explicit authorized action.

Target-specific path syntax stays behind this boundary. UI and persisted
records carry a target ID together with paths rather than treating local
absolute paths as universal.

Questions to settle:

- How much of the target protocol is defined before implementing local-only
  execution?
- Whether the future SSH host runs a small dasHerd helper or begins with direct
  SSH command and PTY channels.
- Which environment, credential, and agent-command settings belong to a target
  profile.

## Section 4: Git and setup task terminals

Discussion status: open

User-initiated Git mutations, setup recipes, and repair work are activities
with terminals. Their terminal provides:

- live colored output through a PTY;
- normal terminal scrolling, selection, search, links, and zoom;
- retained logs and structured process status;
- input when credentials, confirmation, or repair is required;
- an obvious "go to terminal" action from the worktree and activity UI;
- a recovery shell in the same target and working directory when a task fails.

Short background observations remain structured jobs rather than flooding a
terminal. The Git debug UI can show their invocation, timing, exit status, and
bounded output, and can offer a terminal rerun or recovery shell. After any
interactive/mutating task, the structured observer refreshes Git state; the
colored terminal output is not parsed as truth.

Logs live in the dasHerd user-data `logs/` area with rotation. A completed task
terminal remains inspectable after its process exits. Long setup tasks should
survive closing the UI, which may pull the detachable session host earlier in
the implementation order than agent launching.

Questions to settle:

- Which successful task terminals close automatically, remain as history, or
  leave an interactive shell open?
- Whether a failed command automatically drops to a shell or offers an
  explicit Open recovery shell action.
- Log retention, rotation, and whether input payloads are logged. Input can
  contain secrets, so logging output and logging typed input are separate
  decisions.
- Whether every mutation must be visibly terminal-backed or whether very short
  non-interactive mutations may use the job path while retaining a one-click
  diagnostic terminal.

## Section 5: terminal identity and live-command control

Discussion status: open

Every terminal has a stable terminal/session ID independent of whether its
window is currently visible. Live commands must be able to:

- list terminals with target, worktree, task/session, process, and activity
  state;
- open or focus a terminal in the UI;
- inspect its bounded visible viewport and requested scrollback;
- read semantic text or changes since a known terminal revision;
- write UTF-8 text atomically;
- send normalized keys and paste operations;
- observe cursor, title, current-directory hint, exit state, and revisions.

The same operations serve the user, Codex controlling dasHerd, another agent,
and future automation. The PTY receives the same bytes regardless of source;
there is no weaker "automation terminal." The control plane may still record
origin metadata outside the byte stream for audit, loop prevention, and
permissions.

Input operations are serialized as complete events so a paste or prompt cannot
interleave byte-by-byte with human typing. This must be reconciled with the
single-input-lease rule for detachable sessions: live commands from the
lease-owning client are straightforward, while cross-client and cross-agent
writes need an explicit transfer, queue, or policy.

Questions to settle:

- The exact live-command names and bounded read/inspection schemas.
- Text, key, and paste operations versus an expert raw-byte operation.
- Whether writes require the input lease, can be queued for its owner, or can
  use separately granted automation capability.
- What provenance is retained without making terminal behavior differ by
  writer.

## Section 6: basic application layout and navigation

Discussion status: open

The first application shape is:

- a repository header that shows target and base;
- a draggable left rail containing the main checkout and linked worktrees;
- a right workspace for inspection and the selected terminal;
- a Git/tasks activity window containing running and retained jobs;
- status and failure affordances that always navigate to the relevant
  terminal or recovery terminal.

Each worktree row eventually shows branch/HEAD, dirty/conflict state,
refresh/error state, setup state, and live terminal/agent activity. Switching
worktrees does not restart their terminals.

The Git/tasks window is a terminal when a task has a terminal, not a custom
plain-text log approximation. Structured Git observations may appear as debug
records beside the terminal activities.

Questions to settle:

- Default docking: fixed worktree rail plus terminal, or fully dockable
  inspector/activity windows from the first version.
- Whether inspection sits above the terminal, in a separate dock, or behind a
  tab.
- How completed task terminals are grouped and pruned from the activity UI.

## Section 7: worktree creation and setup recipes

Discussion status: open

The initial creation flow collects a name, new branch, base ref, target path,
setup recipe, and post-create action. The default setup recipe is the MCP
worktree bootstrap; `None` and a user-entered recipe remain available.

Recipes are structured command profiles with an executable, argument array,
working directory, environment additions, and placeholders such as repository
root, worktree root, target, and bootstrap daslang. Repository-provided recipes
must not execute merely because a repository was opened.

Creation progresses through visible, retryable states:

1. validate branch, base, target, and path;
2. create the worktree;
3. publish refreshed structured state;
4. run the selected setup recipe in a task terminal;
5. preserve and clearly mark the tree if setup fails;
6. open a shell or launch the selected agent after success.

Once Git has created a real worktree, setup failure does not silently delete
it. The user can repair it in the task/recovery terminal, retry setup, keep the
tree, or proceed to the separately guarded removal flow.

Questions to settle:

- New branches only in v1, or attaching an existing branch too.
- Default worktree path and branch-name derivation.
- Recipe configuration location and first-use trust UI.
- Whether a shell may open while setup is still running.

## Section 8: removal safety and recovery

Discussion status: open

"Is it safe to remove?" produces `Safe`, `Needs review`, or `Blocked` with
specific evidence. The mutation worker revalidates immediately before removal
to avoid acting on a stale UI snapshot.

Initial evidence includes:

- main checkout and protected-target rules;
- locked/prunable state;
- dasHerd-owned terminals, agents, setup jobs, and mutations;
- staged, unstaged, untracked, and conflicted changes;
- branch containment relative to the selected base;
- whether the branch is checked out elsewhere;
- whether repository, target, and path still match the inspected identity.

Every blocking or review condition offers a route to the relevant terminal so
the user or an agent can inspect, commit, move, clean, or otherwise repair the
tree and ask again.

Initial removal uses `git worktree remove` without force. Worktree removal and
branch deletion are separate operations; branch deletion uses the safe Git
operation rather than forced deletion. A force-removal design, if added, is a
later discussion.

Questions to settle:

- Exact distinction between `Needs review` and `Blocked`.
- Whether unpushed and not-contained-in-base are separate warnings.
- What evidence is sufficient for remote targets whose connection is down.
- Whether removal of worktrees not created by dasHerd is permitted in v1.

## Section 9: agents, cross-agent communication, and self-prompting

Discussion status: open, future-facing

Agent integration should be built on terminal/session capabilities rather than
brand-specific screen scraping. A hosted PowerShell rooted in the worktree can
launch Codex, Claude Code, Pi, OpenCode, or a custom command. Closing the UI
must not stop the hosted terminal or agent.

Cross-agent communication is a routed terminal write to a stable session ID.
Self-prompting is the same operation with source and destination equal. Both
use the ordinary terminal read/write contract; neither requires a second
hidden agent-control protocol.

Terminal equivalence does not mean uncontrolled recursion. The control plane
can retain source session, destination session, event ID, parent event, and
timestamp outside the PTY stream. Later policy can use that provenance for
authorization, rate limits, hop/loop protection, and an inspectable audit
trail without changing what bytes the terminal receives.

Questions to settle later:

- Per-session permission to accept user, live-command, cross-agent, and
  self-prompt writes.
- Loop and runaway protection that remains visible and overridable.
- Delivery acknowledgement: accepted into the terminal input queue versus
  semantically acted upon by the agent.
- How an agent discovers addressable sessions without exposing unrelated
  repositories or targets.

## Tentative discussion and implementation order

This order is a discussion aid, not an approved implementation schedule:

1. Settle Section 0 and the core of Section 5: host, log, terminal, attach, and
   live-command contracts.
2. Implement and stress the minimal local detachable host before building
   worktree or agent policy on it.
3. Settle Sections 1-3: identity, asynchronous observation, execution target.
4. Implement the tested hosted Git observer and basic worktree/inspection app.
5. Settle Sections 4 and 6: task-terminal behavior and UI navigation.
6. Add worktree PowerShell terminals, Git activity/debug terminals, and the
   live-command terminal surface.
7. Settle and implement Section 7 creation/setup.
8. Settle and implement Section 8 safe removal/recovery.
9. Settle Section 9 before one-click durable agent launching and any
   cross-agent or self-prompt automation.
