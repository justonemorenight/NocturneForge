# Lessons Learned

### Orchestration ID constructors are not interchangeable

- **Pattern:** Updating orchestration tests by copying constructors between ID types causes argument-count errors.
- **Wrong:** Use `RunId::new("run")` by analogy with `TaskId::new("task")`.
- **Right:** Read `crates/agent_orchestration/src/ids.rs`; use `RunId::from_string("run")` for fixed fixtures, `RunId::new()` for generated IDs, and `TaskId::new("task")` for named tasks.
- **Rule:** Rust E0061.
- **Seen:** 2x — this session, 2026-09-26.

### Buffer text snapshots already own their strings

- **Pattern:** Capturing ACP file-review snapshots with `Buffer::text().to_string()` duplicates the returned owned `String`.
- **Wrong:** Call `.to_string()` before constructing `Arc<str>` or comparing a buffer's text.
- **Right:** Use `Buffer::text()` directly; check a method's return type before adding ownership conversions.
- **Rule:** Clippy `redundant_clone`.
- **Seen:** 5x — this session, 2026-09-26.

### GPUI test spawn passes an owned AsyncApp

- **Pattern:** Native subagent tests call `SubagentHandle::send` from `TestAppContext::spawn`.
- **Wrong:** Pass `cx` directly as though it were the borrowed context from `Context<T>::spawn`.
- **Right:** Pass `&cx`; check the spawn closure's context type before calling APIs that take `&AsyncApp`.
- **Rule:** Rust E0308.
- **Seen:** 2x — 2026-09-27.

### Harness fixtures should move values on their last use

- **Pattern:** Constructing cache-affinity requests, compaction message batches, and native turn guards in agent tests.
- **Wrong:** Clone the cache key, final text chunk, or owned test `AsyncApp` when the fixture never uses it again.
- **Right:** Move the value into the final request/message/guard; clone only earlier shared uses.
- **Rule:** Clippy `redundant_clone`.
- **Seen:** 4x — 2026-09-27.

### Find colocated Zed tests and stubs before using paths

- **Pattern:** Zed keeps many tests inside source modules and ACP stubs in `crates/acp_thread/src/connection.rs`; guessed filename globs may match nothing.
- **Wrong:** Read guessed source/state filenames or pass unverified globs such as `crates/git_ui/src/*tests*` or `crates/acp_thread/src/stub*` to a zsh command.
- **Right:** Use `rg --files` to discover source and state paths before reading them, or search the containing directory with `rg -n`; quote any glob supplied through an `rg -g` option.
- **Rule:** zsh `NOMATCH` stops the command before `rg` runs.
- **Seen:** 11x — 2026-09-30 through 2026-10-03; includes missing stub, thread-history, adapter diff, and relocated Metal source paths.

### Move GPUI handles into their final callback

- **Pattern:** Zed menu callbacks capture a cloned handle that is never used again afterward.
- **Wrong:** Clone the local handle again inside `.on_click` before a `move` listener.
- **Right:** Clone the view field once, then move that local handle directly into its final listener.
- **Rule:** Clippy `redundant_clone`.
- **Seen:** 2x — ledger fixtures and quota menu callback, 2026-09-30.

### Runtime shaders apply to both app and remote release builds

- **Pattern:** This Mac has Command Line Tools without Xcode's `metal`; `remote_server` also directly depends on `gpui_platform`.
- **Wrong:** Enable runtime shaders for app/tests but omit them from the separate remote-server build.
- **Right:** Keep remote-server compilation separate and select `gpui_platform/runtime_shaders` in both invocations; add `-p gpui_platform` with its empty defaults when selecting workspace features.
- **Rule:** `xcrun` cannot find `metal`; GPUI `runtime_shaders`.
- **Seen:** 2x — GPUI test build and remote release build, 2026-09-30.

### GPUI test clock APIs depend on the context type

- **Pattern:** Timer tests mix `Context<T>`, `TestAppContext`, and `VisualTestContext` clock accessors.
- **Wrong:** Copy `cx.advance_clock(...)` or `cx.background_executor()` from a different context type.
- **Right:** In test contexts use `cx.executor().advance_clock(...)`; `VisualTestContext` exposes `cx.background_executor.now()` as a field, while entity contexts use `cx.background_executor().now()`.
- **Rule:** Rust E0599; use the GPUI executor clock for deterministic timeout tests.
- **Seen:** 2x — cache icon and ACP reaper tests, 2026-10-01.
