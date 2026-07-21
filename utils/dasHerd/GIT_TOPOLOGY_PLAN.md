# Git activity and topology plan

This plan records the agreed direction for the rich client's Git review surface. Each stage is reviewed before the next one starts.

## Stage 1: stable review layout

- Split the current combined Git window into independently dockable `Git Changelist` and `Git Activity` windows.
- Keep worktree identity, status, refresh state, errors, and the selected PR/commit file list in Changelist.
- Keep only the PR, History, and Tree perspectives in Activity.
- Use real resizable tables for commits: SHA, title, and author are separate columns and colors.
- Show Tree with the same readable table as History until topology has a trustworthy implementation.
- Reserve a permanent bottom status bar in each Git window; loading and error text replaces the stable summary there without moving content.

This prevents asynchronous file-list replacement from resizing or shifting the activity view.

## Stage 2: Git-owned topology stream

- Treat `git log --graph --topo-order --full-history` as the topology oracle.
- Preserve Git's physical output rows, including transition-only rows between commits.
- Parse the graph prefix separately from commit metadata into a small headless model.
- Add fixtures for forks, merges, octopus merges, crossings, truncation, and decorations before drawing anything.

The initial scope is the current worktree plus its base. An explicit scope selector can later include more refs; loading every local and remote ref is not the default.

## Stage 3: topology rendering

- Extend the Tree table to four columns: SHA, title, author, topology.
- Render the parsed graph prefix as vector lanes, nodes, and transitions in the fourth (rightmost) column.
- Assign stable colors by active lane identity, not commit-hash randomness.
- Keep commit metadata aligned as a table while the topology column explains ancestry.

## Stage 4: selection contract

- PR selects the complete outgoing range against the detected base.
- History and Tree commit rows select one commit and populate Changelist with that commit's files.
- A Tree ref badge selects that branch's outgoing range against its base.
- SHA cells retain hover-copy behavior and copy the full hash.

## Stage 5: robustness and scale

- Add pending and failure states per window without changing dock geometry.
- Stress-test large histories, many refs, rapid selection, and repeated async replacement.
- Measure parsing, rendering, and selection latency before optimizing.

## Research basis

- Git's graph output and topological ordering are the behavioral oracle: <https://git-scm.com/docs/git-log>
- Git's implementation documents the graph state machine and transition rows: <https://github.com/git/git/blob/master/graph.c>
- The target information layout follows the reviewed Git GUI examples: commit metadata remains tabular while topology is a dedicated graph column.
