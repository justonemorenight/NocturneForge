# Native conversation forks

## Implementation plan

1. Snapshot the transcript through the previous user turn, excluding the current
   tool invocation. Validate tool-call/result pairs and persist separate lineage.
2. Expose `fork_thread` to the native agent. Auto/Orchestrate use the existing
   tool permission policy; Direct and descendants require confirmation. Plan and
   subagents cannot start forks. Deny rules take precedence.
3. Save and open a native session using the existing sidebar/session lifecycle,
   retaining the model, profile and tool restrictions. Optionally create a linked
   worktree. Do not copy live runtime, grants, usage, drafts or worker ownership.
4. Verify checkpoint isolation, persistence compatibility, permission gates,
   inherited replay, cancellation, and UI compilation before packaging.

## Semantics

`fork_thread` is an independent conversation, not a worker task. Use `spawn_agent`
when the parent needs to collect results or manage cancellation. The tool returns
the new session ID after scheduling its session in the sidebar, not a claim that
the model ran successfully or finished. Model/authentication errors remain visible
in that conversation. Cancelling the parent after creation does not stop a fork.

The default is the same workspace and files. The optional new worktree starts at
HEAD; it does not carry uncommitted edits. In either case the fork reuses the
conversation's model and profile, not the currently selected sidebar provider.
Historical compaction before the checkpoint is retained; later summaries and the
current user request are not copied. The caller supplies a new focused prompt.
Inherited tool traffic is historical and must not reconnect to source workers.

Forks are labelled `Fork: <title>`. Source session and checkpoint are persisted
separately from subagent ownership, so they remain top-level sidebar entries.
Old database blobs load without lineage. This is not a token-saving guarantee:
the inherited transcript still contributes to each branch's input.

## Controls

Enable or disable `fork_thread` in the native agent profile. Configure its usual
allow/confirm/deny rules in **Tool Permissions → Fork Conversation**; rule patterns
match the task prompt. Auto/Orchestrate do not bypass those permissions. Direct
and forks of forks additionally require confirmation to prevent silent recursive
fan-out. No new hard token budget is introduced.

## Follow-up scope

Automatic result handoff/merge, a branch-tree UI, user-selectable historical
checkpoints, and ACP session cloning are separate work. No canonical-event-log
reconstruction or cross-provider replay is claimed by this implementation.

## Verification

- `cargo check -p agent -p agent_ui -p settings_ui --tests --locked`
- `cargo test -p agent --lib fork --locked`: snapshot isolation, legacy loading,
  lineage round-trip, state reset, incomplete tool rejection, manual compaction,
  execution policy and historical replay.
- `cargo fmt -p agent -p agent_ui -p settings_ui -- --check`
- `git diff --check`

Live-provider execution, sidebar interaction and linked-worktree smoke tests are
still required on a packaged build. The source check is not an app-install test.
