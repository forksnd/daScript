# Agent-assisted review workflows

Status: product direction; local Review Focus first; communication protocol
under discussion

This note records the two agreed product tracks that follow a usable local Git
inspector. Both tracks will be implemented. The local Review Focus vertical
slice comes first, followed by the GitHub PR surface.

1. Make local Git review useful for preparing and reviewing a PR through
   explicit human-agent attention handoff.
2. Add a GitHub PR surface for status, body, CI, comments, logs, and supervised
   review operations.

They are not separate interaction models. Both should use one inspectable
`Review Focus` contract.

## Shared contract: Review Focus

A Review Focus is a visible, clearable set of things that a human or agent asks
the other party to inspect. It must be structured state, not prose inferred
from a terminal and not an invisible filter.

A focus item identifies:

- repository, workspace/worktree, and revision or comparison;
- a target such as file, symbol/function, source range, ref/SHA, PR, check,
  review thread, or log range;
- a role such as primary, risk, question, failure, context, or plumbing;
- a short explanation of why it matters;
- origin (human, agent, or tool), session, and creation time;
- acknowledgement or resolution state; and
- enough snapshot identity to report when the target has become stale.

Source targets use the existing canonical identity: workspace, worktree-
relative path, revision/comparison, and UTF-8 byte or line range. Symbol names
help a human read a focus item but do not replace the source range and revision.
GitHub targets add stable repository, PR, check-run, thread/comment, and log
identities without weakening the local source anchor.

The UI may subdue unrelated material, but it always shows who supplied the
focus, why each item is present, and how to clear it or return to the ordinary
Auto/All view. Focus never changes Git state by itself.

## Human-agent communication protocol

The rendering is deliberately simple: focused files/ranges keep their ordinary
or brighter treatment, unrelated material is darkened, and a toggle restores
the unfiltered view. The protocol that delivers focus state is the harder
contract.

### Agent to dasHerd

An agent-facing `dasherder.md` skill explains the available operations and how
to discover the current session. The agent publishes structured focus sets,
notes, acknowledgements, and results to dasHerd. MCP is the preferred ergonomic
surface when installed; a small CLI is the portable surface; localhost HTTP is
the underlying fallback. All three map to the same versioned protocol and
identities.

The agent should not have to synthesize UI input or emit terminal prose for
dasHerd to scrape. A typical operation is semantically:

```text
publish_focus(session, focus_items, summary)
```

### dasHerd to agent: durable payload plus wake-up

Local HTTP/MCP state cannot wake a passive agent model. Conversely, pasting the
complete request into a TTY is fragile, noisy, difficult to acknowledge, and
unsafe for large source selections. The proposed transport therefore splits
notification from payload:

1. dasHerd stores the complete structured request in a durable per-session
   inbox and assigns a stable message ID.
2. After the human previews and sends it, dasHerd injects one short natural-
   language notification through the agent's normal TTY input path.
3. The notification tells the agent to use the dasHerd skill and fetch that
   exact message ID.
4. The agent fetches the payload through MCP/CLI/HTTP and acknowledges it.
5. dasHerd shows pending, notified, fetched, acknowledged, completed, failed,
   canceled, and stale state rather than assuming that pasted text was read.

For example, the visible TTY message can be:

> dasHerd review request `hf_42` is ready (3 source anchors). Use the dasHerd
> skill to fetch and acknowledge it.

The nudge is intentionally sufficient for the model to choose the skill but
does not duplicate the source payload in terminal scrollback or context.

MCP resource-change notifications or a future direct agent callback may remove
the TTY wake-up for agents that support true push. They are optimizations of
the wake transport, not replacements for the durable inbox and acknowledgement
model. Pure polling is not the default because an idle agent has no reason to
take another turn.

### Session discovery and capability scope

dasHerd supplies an agent session with one capability descriptor containing the
protocol version, localhost endpoint, session ID, and session-scoped
credential. The launch profile exposes only the descriptor location; the
`dasherder.md` skill describes how MCP, CLI, or HTTP consumes it. Credentials
must not appear in generated prompts, terminal history, review briefs, or the
repository.

The capability may read and acknowledge only its own inbox and publish focus
for its own session/worktree. It cannot impersonate a human-origin message or
mutate Git/GitHub state through the focus protocol.

### Delivery rules

- Human-to-agent delivery is explicit. The human previews the generated nudge
  and request before sending it.
- The TTY nudge uses the same serialized input ownership as ordinary terminal
  input; it must not interleave with a human's partial line or paste.
- Message IDs make retries idempotent. Re-sending a nudge does not duplicate
  the inbox payload or lose acknowledgement state.
- The inbox is authoritative. TTY text is only a wake signal and never proof
  that the agent received, understood, or completed the request.
- Agent profiles may replace TTY notification with a stronger native wake
  transport, but every profile exposes the same message lifecycle.
- A Markdown review brief remains the portable export. It can reference the
  same message/focus IDs, but it contains no credentials and does not become
  the authoritative mutable inbox record.

## Track A: local PR preparation and review

### Agent to human

The agent can publish a review focus equivalent to:

> Look at these two files and these three functions. The remaining changes are
> mechanical PR plumbing.

The Changelist emphasizes selected files and labels supporting/plumbing files.
Opening Diff or View emphasizes exact symbols and ranges while retaining the
complete file. Tree applies the same idea to topology: it shows the refs, SHAs,
and ancestry paths named by the agent, with unrelated branches subdued.

This supports:

- a PR-preparation summary of important implementation and risk areas;
- a reviewer guide through semantic changes hidden among generated or
  mechanical files;
- questions and caveats attached to exact source ranges; and
- an agent-directed Tree focus for primary, incoming, and base/context paths.

### Human to agent

A human can select files, functions, source ranges, commits, or Tree paths and
turn that selection into either:

- a generated chat prompt inserted through the normal, visible agent input
  path; or
- an exported Markdown review brief that the human can ask the agent to read.

The Markdown artifact is the portable human-readable projection of the same
structured focus. It lists canonical anchors, selected text only when useful,
and the human's notes/questions. A machine-readable representation may travel
beside or inside it, but the `.md` must remain sufficient to inspect, edit, and
discuss without dasHerd.

No prompt is sent silently. The human sees and may edit the generated request
before sending it.

## Track B: GitHub PR observation and supervised operation

When a session/worktree has an open PR, dasHerd indicates that association and
offers a PR Status surface containing:

- PR identity, state, base/head, mergeability, and current review summary;
- rendered PR body plus a locally autosaved editable Markdown draft and an
  explicit conflict-aware publish action;
- CI/check rows with live state, timestamps, and links;
- review comments and threads, including replies and resolution state; and
- failure logs that can be opened, searched, selected, and attached to a
  Review Focus.

The human can point the agent at a failing check, a red log range, or a review
thread using the same handoff as source code: generated visible prompt or
exported review brief. The agent can answer questions such as "why is this
broken?" against exact observed state rather than a pasted approximation.

Copilot and other automated-review interaction should use supervised
suggestions: dasHerd presents a proposed accept, reject, reply, resolve, rerun,
or merge operation as a ghost action with its rationale and exact target. The
human may approve, reject, or add a comment. Read/monitor work can be automatic;
remote mutations remain explicit and observable.

The existing babysit workflow should ultimately consume this state directly:
monitor checks and reviews, prepare proposed responses/actions, surface only
decisions that need a human, and merge when the user's declared policy permits.

## Proposed order

Implement Track A first, beginning with the shared Review Focus model and its
two-way handoff, then build Track B on that contract.

Reasons:

- The local Git inspector has just become usable, so this is the shortest path
  from a viewer to a genuinely useful review tool.
- File/symbol/range/ref anchoring is the common semantic foundation. CI lines,
  comments, and logs can later become additional focus target kinds instead of
  inventing a second selection and communication system.
- The complete human-agent loop can be tested deterministically against local
  repository state before adding authentication, API limits, changing remote
  state, and provider-specific behavior.
- It establishes the agent-facing publish/consume contract and the human-facing
  prompt/Markdown export contract that GitHub babysitting will need anyway.
- GitHub work can then concentrate on observation and supervised operations
  rather than simultaneously discovering the underlying interaction language.

The strongest argument for Track B first is immediate operational value for
already-open PRs: status, CI, comments, and babysitting are useful before rich
source focus exists. That remains a valid priority if remote operations are the
near-term bottleneck. The tradeoff is a higher chance of building GitHub-
specific selection/actions that must later be reconciled with source and Tree
focus.

## Suggested implementation sequence

1. Define and persist the headless Review Focus/item schema, revision/staleness
   rules, and live inspection surface.
2. Render agent-to-human file, source, and Tree focus with visible provenance,
   reasons, and clear/Auto/All controls.
3. Capture human selections and generate an editable prompt plus Markdown
   review brief.
4. Add agent tools/skill flow to publish and consume focus sets during local PR
   preparation and review.
5. Add read-only GitHub PR association, body, checks, comments, and logs.
6. Extend focus targets to GitHub checks, threads, and log ranges.
7. Add supervised ghost actions and integrate the babysit/merge policy.

Each stage must expose semantic state to live commands. Screenshots are useful
for presentation, never the correctness oracle.

## Discussion record

- 2026-07-21: Both tracks are committed product direction; only order is open.
- 2026-07-21: Local review should support agent-selected important files,
  functions/ranges, and Tree branches while treating the rest as plumbing.
- 2026-07-21: Human selections must travel back to an agent through a visible
  generated prompt or an editable/exportable Markdown review brief.
- 2026-07-21: GitHub review should cover PR body, CI, comments, logs, questions
  about exact failures, Copilot review, supervised actions, and eventual
  babysit-to-merge operation.
- 2026-07-21: Proposed preference is local Review Focus first, GitHub second,
  because Review Focus is the shared interaction foundation.
- 2026-07-21: Adopt local Review Focus first and begin protocol design before
  UI implementation.
- 2026-07-21: Proposed human-to-agent transport is a durable per-session inbox
  fetched through MCP/CLI/HTTP, paired with a short visible TTY wake-up and an
  explicit acknowledgement lifecycle.
