# dasHerd session mailbox

Use this protocol when a dasHerd terminal prints a wake-up such as:

`dasHerd review request hf_42 is ready. Use the dasHerd skill to fetch and acknowledge it.`

The watcher URL, token, and agent session id are routing context, not an
authorization model. Configure `DASHERD_URL`, `DASHERD_TOKEN`, and
`DASHERD_SESSION_ID`, or pass `--url`, `--token`, and `--session` to the CLI.

Run the CLI from the repository root:

```powershell
powershell.exe -NoProfile -File utils/dasHerd/dasherd.ps1 whoami
powershell.exe -NoProfile -File utils/dasHerd/dasherd.ps1 inbox list
powershell.exe -NoProfile -File utils/dasHerd/dasherd.ps1 inbox get hf_42
powershell.exe -NoProfile -File utils/dasHerd/dasherd.ps1 inbox ack hf_42
powershell.exe -NoProfile -File utils/dasHerd/dasherd.ps1 inbox complete hf_42
```

Read the exact UTF-8 byte ranges in each message's `focus.targets[].ranges`.
An empty range list with `whole_file: true` targets the whole file. Acknowledge
after fetching, and complete only after handling the request.

To point the human at important code, write a Focus Set JSON file outside any
deployment directory and send it atomically:

```powershell
powershell.exe -NoProfile -File utils/dasHerd/dasherd.ps1 outbox send `
  --subject "Ready for PR. Check it out; I've highlighted the important bits." `
  --focus-json path/to/focus-set.json
```

A Focus Set contains `repository_id`, `worktree_path`, `comparison`, optional
`revision` and `summary`, and one or more file targets. Each target has
`file_path`, `whole_file`, and zero or more `{start_byte,end_byte,caption}`
ranges. One message may target several files and several disjoint ranges per
file. The human receives persistent, non-modal Attention; publishing must not
assume that their current file, scroll position, mode, or keyboard focus changed.

Use `outbox reply <inbox-id> --subject ... --focus-json ...` to preserve the
conversation link. Inbox means human/system to agent; Outbox means agent to
human. Senders are provenance strings: `agent_session:<id>`, `human:local`, or
`system:dasHerd`.
