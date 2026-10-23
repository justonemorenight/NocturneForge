# Upstream ports — 2026-10-23

Branch: `codex/upstream-complete-20261023`, based on `b73af25220a4`.
These are adapted source ports, including selected open proposals. They retain
NocturneForge's ACP v1 transport and existing Claude Multi routing.

| Upstream PR | Fork implementation |
| --- | --- |
| [#65956](https://github.com/zed-industries/zed/pull/65956) | Update Wasmtime, TLS trust dependencies and managed Node. |
| [#65824](https://github.com/zed-industries/zed/pull/65824) | Ignore events from replaced thread entities. |
| [#65995](https://github.com/zed-industries/zed/pull/65995) | Normalize filesystem watcher keys by path component. |
| [#65998](https://github.com/zed-industries/zed/pull/65998) | Keep recursive backend watches alive while child registrations need them; preserve backend calls outside the state lock. |
| [#65939](https://github.com/zed-industries/zed/pull/65939) | Register symlink watch ownership before directory enumeration so deletion can clean it up. |
| [#65856](https://github.com/zed-industries/zed/pull/65856) | Preserve FIFO ordering between LSP notifications and requests. |
| [#65769](https://github.com/zed-industries/zed/pull/65769) | Resolve on-type formatting position after autoindent and preserve cursor anchors. |
| [#65559](https://github.com/zed-industries/zed/pull/65559) | Discard unstaged changes against the index without discarding staged changes. |
| [#65280](https://github.com/zed-industries/zed/pull/65280) | Isolate npm working directory/prefix from project package-manager configuration. |
| [#65852](https://github.com/zed-industries/zed/pull/65852) | Cache direct, fallback and missing font resolutions in GPUI. |
| [#65926](https://github.com/zed-industries/zed/pull/65926) | Prefer one reasoning field when a gateway mirrors reasoning deltas. |
| [#65938](https://github.com/zed-industries/zed/pull/65938) | Support Responses reasoning_text events and tolerate unknown reasoning content parts. |
| [#65940](https://github.com/zed-industries/zed/pull/65940) | Preserve typed transport errors through provider stream mapping. |
| [#65189](https://github.com/zed-industries/zed/pull/65189) | Suppress duplicate worktree removal updates. |
| [#65850](https://github.com/zed-industries/zed/pull/65850) | Apply default session configuration only to new sessions. |
| [#65444](https://github.com/zed-industries/zed/pull/65444) | Existing lazy diff resolution already supplies the relevant behavior; retained. |
| [#65470](https://github.com/zed-industries/zed/pull/65470) | Defer Markdown parsing, gate list measurement on readiness, preserve deferred search and scroll intent. |
| [#65912](https://github.com/zed-industries/zed/pull/65912), [#65976](https://github.com/zed-industries/zed/pull/65976) | Adapt child routing, activity and cancellation handling to existing native/Claude Multi ownership. No ACP v2 schema migration. |
| [#65918](https://github.com/zed-industries/zed/pull/65918) | Parent thread owns/deduplicates child entities; late loads check the current owner. |
| [#65923](https://github.com/zed-industries/zed/pull/65923) | Centralize subagent card state, errors and preview bounds. |
| [#65841](https://github.com/zed-industries/zed/pull/65841) | Preserve unsent drafts and queued input across reconnect; restore the queue paused. ACP v2 receipt reconciliation is unavailable and is not emulated by replaying submissions. |
| [#66008](https://github.com/zed-industries/zed/pull/66008) | Verbose/compact/grouped tool display; permission/subagent boundaries remain visible, search uses individual cards. Summary timing stays in expanded terminal cards. |
| [#65988](https://github.com/zed-industries/zed/pull/65988) | Optional worktree grouping in the threads sidebar. |
| [#65999](https://github.com/zed-industries/zed/pull/65999) | File-scoped branch diff; intentionally not persisted across restart. |
| [#65793](https://github.com/zed-industries/zed/pull/65793) | Named terminal profiles with backward-compatible settings and persistence. Existing remote default-shell behavior is retained. |
| [#64938](https://github.com/zed-industries/zed/pull/64938) | Opt-in preservation of compaction tools/thinking/cache prefix for compatible same-model requests; Anthropic retains its existing strategy. No measured cache-hit improvement is claimed. |
| [#65238](https://github.com/zed-industries/zed/pull/65238) | Serialize OAuth provider initialization and refresh per shared provider. |
| [#65250](https://github.com/zed-industries/zed/pull/65250) | Normalize CRLF in bracketed terminal paste. |
| [#65483](https://github.com/zed-industries/zed/pull/65483) | Ignore linked-worktree fsmonitor noise. |

## Validation

Validated on Apple Silicon with build artifacts and temporary files on the external SSD.

- 320 core tests passed; one existing settings test is ignored.
- 36 GPUI list/scroll tests and one font-resolution test passed.
- Two agent compaction/cache-isolation tests passed.
- 27 targeted tests passed across subagent ownership, reconnect/paused queue,
  tool grouping/cards, staged Git discard, on-type formatting, symlink watch
  cleanup, fsmonitor filtering, shared OAuth refresh, terminal profiles and paste.
- Release source/test checks passed for Zed, remote_server and the visual fixtures.
- `./script/clippy` passed with warnings denied for fs, node_runtime, open_ai,
  google_ai, lsp, settings_content and gpui_platform.

The initial incompatible terminal/symlink fixtures were adapted to fork helpers
before their successful runs. No coverage percentage, cache-hit benchmark, visual
smoke test or packaged application installation is claimed.

