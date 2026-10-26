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
- **Right:** Use `rg --files` to discover source and state paths before reading them, or search the containing directory with `rg -n`; quote any glob supplied through an `rg -g` option. Check upstream diff paths against the fork too: newly introduced upstream files may not exist locally.
- **Rule:** zsh `NOMATCH` stops the command before `rg` runs.
- **Seen:** 16x — 2026-09-30 through 2026-10-18; includes missing stub, thread-history, adapter diff, relocated Metal, fake-provider, colocated item-test, and upstream-only source paths.

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
- **Seen:** 3x — GPUI test build and remote release build, 2026-09-30; remote release build, 2026-10-08.

### GPUI test clock APIs depend on the context type

- **Pattern:** Timer tests mix `Context<T>`, `TestAppContext`, and `VisualTestContext` clock accessors.
- **Wrong:** Copy `cx.advance_clock(...)` or `cx.background_executor()` from a different context type.
- **Right:** In test contexts use `cx.executor().advance_clock(...)`; `VisualTestContext` exposes `cx.background_executor.now()` as a field, while entity contexts use `cx.background_executor().now()`.
- **Rule:** Rust E0599; use the GPUI executor clock for deterministic timeout tests.
- **Seen:** 3x — cache icon and ACP reaper tests, 2026-10-01; upstream streaming-save tests, 2026-10-08.

### Cherry-pick conflicts can absorb unrelated fork code

- **Pattern:** Porting upstream ACP and search changes with a distant merge base can place whole neighboring functions in a conflict, including telemetry and existing tests.
- **Wrong:** Select an entire conflict side based only on the changed API name.
- **Right:** Review the full diff against HEAD after resolution; preserve fork functions, deduplicate tests, and port only the actual upstream edits when conflict boundaries expand.
- **Rule:** Three-way merge conflict boundaries are not the PR change scope.
- **Seen:** 5x — ACP telemetry and Git/search ports, 2026-10-03; unrelated Node discovery, terminal hyperlink and Git merge-base tests in expanded upstream conflicts, 2026-10-23.

### Upstream ports can depend on manifest changes outside the PR

- **Pattern:** Ported source imports depend on upstream's existing runtime dependencies or opt-in SDK schema features.
- **Wrong:** Assume a successful cherry-pick or no-dependencies metadata check proves the fork exposes those imports.
- **Right:** Compare imported crates and feature-gated modules against the fork's production dependency declarations; ACP's `unstable` feature does not include `unstable_protocol_v2`.
- **Rule:** Rust E0432; dev-dependencies do not cover production code, and umbrella features need not include experimental schemas.
- **Seen:** 2x — Project `encoding_rs` and ACP v2 schema imports, 2026-10-03.

### Upstream test capability builders differ from the fork

- **Pattern:** New upstream Agent UI tests use plain model fields and ACP v2 prompt capability objects.
- **Wrong:** Copy `model.supports_images = true` or the v2 image builder into this fork's test setup.
- **Right:** Use `FakeLanguageModel::set_supports_images` through `ConfiguredModel.model`, and the receiving editor's v1 `PromptCapabilities::image(true)` API.
- **Rule:** Rust E0433 and mismatched model/capability APIs; inspect the receiving type before porting test helpers.
- **Seen:** 2x — 2026-10-17, inline-prompt and message-editor paste regressions.

### Restored source timestamps can leave Cargo metadata stale

- **Pattern:** Restoring backed-up source with preserved mtimes after experimental builds leaves Cargo treating older intermediate metadata as current.
- **Wrong:** Use `copy2` to restore source and trust contradictory type errors from cached metadata.
- **Right:** Restore file contents with fresh modification times, then rerun the targeted check; keep build artifacts on the external SSD.
- **Seen:** Multiple provider/check failures during upstream ports, 2026-10-23.

### Batch test targets before expensive release linking

- **Pattern:** Separate Cargo test commands choose different feature graphs and repeatedly build shared dependencies; this repository's `release` profile also uses thin LTO with one codegen unit.
- **Wrong:** Discover missing upstream test helpers only after linking several large release test binaries, then change the feature graph for each package.
- **Right:** Check all selected test targets, including required integration-test features, before linking. Build their binaries with one feature graph and run focused filters against the discovered executables. Prefer the existing `release-fast` profile for iterative test builds; retain release source checks and external SSD target/temp paths.
- **Seen:** Repeated GPUI, agent and terminal validation builds during the 2026-10-23 upstream port.

### Scope Rust AST migrations by owner and exact item

- **Pattern:** Bulk ACP migrations encounter repeated method names and trait implementations with the same target type; macro arguments are opaque to the Rust parser.
- **Wrong:** Select an impl by target type alone, map methods globally by name, or extend an item replacement through adjacent declarations.
- **Right:** Match the full owner and trait (including absence of a trait), retain primitive byte spans, handle macro fixtures separately, and inspect the exact diff before applying it. Keep upstream tests only once and preserve fork-specific assertions.
- **Rule:** AST migration boundaries; preserve Rust attributes and UTF-8 byte offsets.
- **Seen:** 3x — ACP v2 migration, 2026-10-26.
