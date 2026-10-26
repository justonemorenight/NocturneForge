# ACP v2 migration

The fork follows Zed's ACP v2 implementation pinned at
[`604a0ff023ec719045bbdafa46d1e384e907ba7b`](https://github.com/zed-industries/zed/tree/604a0ff023ec719045bbdafa46d1e384e907ba7b),
with agent-client-protocol 3.5.0 and schema 1.12.0. This supersedes the v1-only
adaptations recorded in [the October 23 port](upstream-ports-2026-10-23.md).

## Transport and session contract

- Connection selection follows the upstream ACP beta setting. Explicit v1-only
  negotiation permits one fresh v1 process; other initialization errors retain
  their structured error and do not trigger a fallback.
- The v1 adapter retains completion-based prompts and Claude Multi's older
  notification extensions. Both adapters preserve worker sandbox policy and
  process ownership. A shared connection cannot be terminated as an owned worker.
- A v2 prompt response acknowledges a submission. The agent's state updates
  determine running, action required, and idle; acceptance is not completion.
- Receipt IDs reconcile optimistic messages and full replay. Disconnection with
  unknown acceptance preserves recoverable input without automatically resending it.
- Reconnect restores queued input paused. A newer live draft, including an
  intentionally cleared draft in another owner, takes precedence; a displaced
  recovered draft remains in the paused queue.
- Source content and protocol metadata survive rendering and persistence. Draft
  storage reads version 1 through an explicit converter and writes version 2.
  Unsupported storage versions are rejected.
- Session messages preserve their keyed identity, participants, and timeline
  position. Subagent updates keep the fork's richer child registry while exposing
  the official v2 association and agent-owned submission restrictions.

## Native execution and subagents

Native execution uses the same receipt and reported-state model as v2 adapters.
The orchestration worker waits for actual completion after acceptance and reports
connection loss instead of leaving its caller waiting indefinitely.
Cancellation transport errors propagate to the scheduler, leaving the worker
eligible for cleanup or retry instead of reporting a successful cancellation.

A subagent cleanup guard is fenced by the captured submission ID. Legacy turn
IDs are absent on receipt-driven native sessions, so using them could let an old
send future cancel or unregister the next resume.

Automatic continuations retain the previous execution driver until completion.
GPUI cancels a task when its owner is dropped; cloning a completion receiver alone
cannot keep that driver alive. Cancellation must reach the retained execution
while the continuation is waiting for it.

Removing a parked draft suppresses the `ConversationView` release-time prompt
flush before deleting metadata. Otherwise, the asynchronous flush can recreate
the draft-prompt KVP entry after the user deletes the draft.

The fork's native model/role selection, orchestration, Claude review hooks,
read-only policies, deferred Markdown rendering, and lazy legacy diff buffers
remain part of the implementation. V2 patch previews use upstream's structured
patch model.

## Verification

Targeted validation on the current source passed:

- `acp_thread`: 341 passed.
- `agent`: 903 passed, 12 ignored; `agent_orchestration`: 203 passed.
- `agent_ui`: 592 passed; `sidebar`: 147 passed.
- `agent_servers`: 153 passed, 1 ignored. Two timer-dependent tests remain
  filtered because their GPUI fake-clock setup does not advance while parking
  is allowed; those cases remain unverified.

`cargo fmt --all --check` and `git diff --check` passed.

On October 26, 2026, the Apple Silicon release was built and installed as
`/Applications/NocturneForge.app`, version `1.25.0`, build `20261026.140240`.
The installed executable matches the verified bundle, and strict signature,
identity, architecture, runtime-file, and document-icon checks passed. The app
restored the editor and terminal, and a native Agent draft opened successfully.
This smoke check did not submit a live model request. The prior app remains at
`/Applications/NocturneForge.backup-20261026-210548.app`.

Build artifacts, temporary build files, and verification logs use the external
SSD. Iterative verification uses `release-fast`, four Cargo jobs, and
`gpui_platform/runtime_shaders` on this Apple Silicon machine.
