# Rust coding guidelines

* Prioritize code correctness and clarity. Speed and efficiency are secondary priorities unless otherwise specified.
* Do not write organizational or comments that summarize the code. Comments should only be written in order to explain "why" the code is written in some way in the case there is a reason that is tricky / non-obvious.
* Prefer implementing functionality in existing files unless it is a new logical component. Avoid creating many small files.
* Avoid using functions that panic like `unwrap()`, instead use mechanisms like `?` to propagate errors.
* Be careful with operations like indexing which may panic if the indexes are out of bounds.
* Never silently discard errors with `let _ =` on fallible operations. Always handle errors appropriately:
  - Propagate errors with `?` when the calling function should handle them
  - Use `.log_err()` or similar when you need to ignore errors but want visibility
  - Use explicit error handling with `match` or `if let Err(...)` when you need custom logic
  - Example: avoid `let _ = client.request(...).await?;` - use `client.request(...).await?;` instead
* When implementing async operations that may fail, ensure errors propagate to the UI layer so users get meaningful feedback.
* Never create files with `mod.rs` paths - prefer `src/some_module.rs` instead of `src/some_module/mod.rs`.
* When creating new crates, prefer specifying the library root path in `Cargo.toml` using `[lib] path = "...rs"` instead of the default `lib.rs`, to maintain consistent and descriptive naming (e.g., `gpui.rs` or `main.rs`).
* Avoid creative additions unless explicitly requested
* Use full words for variable names (no abbreviations like "q" for "queue")
* Use variable shadowing to scope clones in async contexts for clarity, minimizing the lifetime of borrowed references.
  Example:
  ```rust
  executor.spawn({
      let task_ran = task_ran.clone();
      async move {
          *task_ran.borrow_mut() = true;
      }
  });
  ```

# Timers in tests

* In GPUI tests, prefer GPUI executor timers over `smol::Timer::after(...)` when you need timeouts, delays, or to drive `run_until_parked()`:
  - Use `cx.background_executor().timer(duration).await` (or `cx.background_executor.timer(duration).await` in `TestAppContext`) so the work is scheduled on GPUI's dispatcher.
  - Avoid `smol::Timer::after(...)` for test timeouts when you rely on `run_until_parked()`, because it may not be tracked by GPUI's scheduler and can lead to "nothing left to run" when pumping.

# GPUI

GPUI is a UI framework which also provides primitives for state and concurrency management.

## Context

Context types allow interaction with global state, windows, entities, and system services. They are typically passed to functions as the argument named `cx`. When a function takes callbacks they come after the `cx` parameter.

* `App` is the root context type, providing access to global state and read and update of entities.
* `Context<T>` is provided when updating an `Entity<T>`. This context dereferences into `App`, so functions which take `&App` can also take `&Context<T>`.
* `AsyncApp` and `AsyncWindowContext` are provided by `cx.spawn` and `cx.spawn_in`. These can be held across await points.

## `Window`

`Window` provides access to the state of an application window. It is passed to functions as an argument named `window` and comes before `cx` when present. It is used for managing focus, dispatching actions, directly drawing, getting user input state, etc.

## Entities

An `Entity<T>` is a handle to state of type `T`. With `thing: Entity<T>`:

* `thing.entity_id()` returns `EntityId`
* `thing.downgrade()` returns `WeakEntity<T>`
* `thing.read(cx: &App)` returns `&T`.
* `thing.read_with(cx, |thing: &T, cx: &App| ...)` returns the closure's return value.
* `thing.update(cx, |thing: &mut T, cx: &mut Context<T>| ...)` allows the closure to mutate the state, and provides a `Context<T>` for interacting with the entity. It returns the closure's return value.
* `thing.update_in(cx, |thing: &mut T, window: &mut Window, cx: &mut Context<T>| ...)` takes a `AsyncWindowContext` or `VisualTestContext`. It's the same as `update` while also providing the `Window`.

Within the closures, the inner `cx` provided to the closure must be used instead of the outer `cx` to avoid issues with multiple borrows.

Trying to update an entity while it's already being updated must be avoided as this will cause a panic.

`WeakEntity<T>` is a weak handle. It has `read_with`, `update`, and `update_in` methods that work the same, but always return an `anyhow::Result` so that they can fail if the entity no longer exists. This can be useful to avoid memory leaks - if entities have mutually recursive handles to each other they will never be dropped.

## Concurrency

All use of entities and UI rendering occurs on a single foreground thread.

`cx.spawn(async move |cx| ...)` runs an async closure on the foreground thread. Within the closure, `cx` is `&mut AsyncApp`.

When the outer cx is a `Context<T>`, the use of `spawn` instead looks like `cx.spawn(async move |this, cx| ...)`, where `this: WeakEntity<T>` and `cx: &mut AsyncApp`.

To do work on other threads, `cx.background_spawn(async move { ... })` is used. Often this background task is awaited on by a foreground task which uses the results to update state.

Both `cx.spawn` and `cx.background_spawn` return a `Task<R>`, which is a future that can be awaited upon. If this task is dropped, then its work is cancelled. To prevent this one of the following must be done:

* Awaiting the task in some other async context.
* Detaching the task via `task.detach()` or `task.detach_and_log_err(cx)`, allowing it to run indefinitely.
* Storing the task in a field, if the work should be halted when the struct is dropped.

A task which doesn't do anything but provide a value can be created with `Task::ready(value)`.

## Elements

The `Render` trait is used to render some state into an element tree that is laid out using flexbox layout. An `Entity<T>` where `T` implements `Render` is sometimes called a "view".

Example:

```
struct TextWithBorder(SharedString);

impl Render for TextWithBorder {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        div().border_1().child(self.0.clone())
    }
}
```

Since `impl IntoElement for SharedString` exists, it can be used as an argument to `child`. `SharedString` is used to avoid copying strings, and is either an `&'static str` or `Arc<str>`.

UI components that are constructed just to be turned into elements can instead implement the `RenderOnce` trait, which is similar to `Render`, but its `render` method takes ownership of `self` and receives `&mut App` instead of `&mut Context<Self>`. Types that implement this trait can use `#[derive(IntoElement)]` to use them directly as children.

The style methods on elements are similar to those used by Tailwind CSS.

If some attributes or children of an element tree are conditional, `.when(condition, |this| ...)` can be used to run the closure only when `condition` is true. Similarly, `.when_some(option, |this, value| ...)` runs the closure when the `Option` has a value.

## Input events

Input event handlers can be registered on an element via methods like `.on_click(|event, window, cx: &mut App| ...)`.

Often event handlers will want to update the entity that's in the current `Context<T>`. The `cx.listener` method provides this - its use looks like `.on_click(cx.listener(|this: &mut T, event, window, cx: &mut Context<T>| ...)`.

## Actions

Actions are dispatched via user keyboard interaction or in code via `window.dispatch_action(SomeAction.boxed_clone(), cx)` or `focus_handle.dispatch_action(&SomeAction, window, cx)`.

Actions with no data defined with the `actions!(some_namespace, [SomeAction, AnotherAction])` macro call. Otherwise the `Action` derive macro is used. Doc comments on actions are displayed to the user.

Action handlers can be registered on an element via the event handler `.on_action(|action, window, cx| ...)`. Like other event handlers, this is often used with `cx.listener`.

## Notify

When a view's state has changed in a way that may affect its rendering, it should call `cx.notify()`. This will cause the view to be rerendered. It will also cause any observe callbacks registered for the entity with `cx.observe` to be called.

## Entity events

While updating an entity (`cx: Context<T>`), it can emit an event using `cx.emit(event)`. Entities register which events they can emit by declaring `impl EventEmitter<EventType> for EntityType {}`.

Other entities can then register a callback to handle these events by doing `cx.subscribe(other_entity, |this, other_entity, event, cx| ...)`. This will return a `Subscription` which deregisters the callback when dropped.  Typically `cx.subscribe` happens when creating a new entity and the subscriptions are stored in a `_subscriptions: Vec<Subscription>` field.

## Build guidelines

- Use `./script/clippy` instead of `cargo clippy`

## macOS release build and installation

For the local fork, run source checks before packaging. On this machine the supported release target is Apple Silicon (`aarch64-apple-darwin`). Keep the currently installed app recoverable until the new bundle passes verification and smoke testing.

### Build

`script/bundle-mac` already builds `zed`, `cli`, and `remote_server` before invoking `cargo bundle`. Set `CARGO_BUNDLE_SKIP_BUILD=1` to prevent cargo-bundle from compiling the same targets a second time. Use a color-capable `TERM`; `TERM=dumb` can make cargo-bundle panic while formatting an error.

```bash
TERM=xterm-256color \
CARGO_BUNDLE_SKIP_BUILD=1 \
./script/bundle-mac aarch64-apple-darwin
```

The release bundle is written under `target/aarch64-apple-darwin/release/bundle/osx/`. Do not run multiple bundle commands concurrently because Cargo's artifact lock will serialize them and can trigger redundant builds. Do not use the script's `-i` install path for the release flow; prepare and sign the bundle first, then install it explicitly.

### Prepare the local Beta bundle

When the fork is distributed as `Zed Preview Beta`, make the app identity consistent before signing:

- `Contents/MacOS/zed`, `cli`, and `git` must be present and executable.
- `Contents/embedded.provisionprofile` and `Contents/Resources/Document.icns` must be present.
- If cargo-bundle produced `Zed Dev.app`, update `Info.plist` before signing: set `CFBundleName`, `CFBundleDisplayName`, `CFBundleIdentifier` (`dev.zed.Zed-Preview-Beta`), `CFBundleIconFile`, and the URL type display name.
- Never edit `Info.plist`, copy runtime files, or rename the icon after signing; any such change invalidates the signature.

### Sign, install, and verify

Sign the completed bundle with the repository entitlements, then verify it before replacing the installed app:

```bash
/usr/bin/codesign --force --deep \
  --entitlements crates/zed/resources/zed.entitlements \
  --sign - "$APP_PATH"
/usr/bin/codesign --verify --deep --strict "$APP_PATH"
```

Move the existing `/Applications/Zed Preview Beta.app` to `/Applications/Zed Preview Beta.backup.app`, move the verified bundle into `/Applications/Zed Preview Beta.app`, and launch it. Do not delete the backup until the new app starts and the critical Agent, terminal, and editor flows have been smoke-tested.

```bash
/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' \
  '/Applications/Zed Preview Beta.app/Contents/Info.plist'
/usr/bin/codesign --verify --deep --strict \
  '/Applications/Zed Preview Beta.app'
open '/Applications/Zed Preview Beta.app'
```

If cargo-bundle reports `ColorOutOfRange`, rerun with `TERM=xterm-256color`. A successful build is not proof that the installed app is the new bundle: verify the version, commit, architecture, and running executable path after launch.

# Pull request hygiene

When an agent opens or updates a pull request, it must:

- Use a clear, correctly capitalized, imperative PR title (for example, `Fix crash in project panel`).
- Avoid conventional commit prefixes in PR titles (`fix:`, `feat:`, `docs:`, etc.).
- Avoid trailing punctuation in PR titles.
- Optionally prefix the title with a crate name when one crate is the clear scope (for example, `git_ui: Add history view`).
- Include a `Release Notes:` section as the final section in the PR body.
- Use one bullet under `Release Notes:`:
  - `- Added ...`, `- Fixed ...`, or `- Improved ...` for user-facing changes, or
  - `- N/A` for docs-only and other non-user-facing changes.
- Format release notes exactly with a blank line after the heading, for example:

```
Release Notes:

- N/A
```

# Crash Investigation

## Sentry Integration
- Crash investigation prompts: `.factory/prompts/crash/investigate.md`
- Crash fix prompts: `.factory/prompts/crash/fix.md`
- Fetch crash reports: `script/sentry-fetch <issue-id>`
- Generate investigation prompt from crash: `script/crash-to-prompt <issue-id>`

# Rules Hygiene

These `.rules` files are read by every agent session. Keep them high-signal.

## After any agentic session
If you discover a non-obvious pattern that would help future sessions, include a **"Suggested .rules additions"** heading in your PR description with the proposed text. Do **not** edit `.rules` inline during normal feature/fix work. Reviewers decide what gets merged.

## High bar for new rules
Editing or clarifying existing rules is always welcome. New rules must meet **all three** criteria:
1. **Non-obvious** — someone familiar with the codebase would still get it wrong without the rule.
2. **Repeatedly encountered** — it came up more than once (multiple hits in one session counts).
3. **Specific enough to act on** — a concrete instruction, not a vague principle.

Rules that apply to a single crate belong in that crate's own `.rules` file, not the repo root.

## What NOT to put in `.rules`
Avoid architectural descriptions of a crate (module layout, data flow, key types). These go stale fast and the agent can gather them by reading the code. Rules should be **traps to avoid**, not **maps to follow**.

## No drive-by additions
Rules emerge from validated patterns, not one-off observations. The workflow is:
1. Agent notes a pattern during a session.
2. Team validates the pattern in code review.
3. A dedicated commit adds the rule with context on *why* it exists.

<!-- REPOWISE_AGENTS:START — Do not edit below this line. Auto-generated by Repowise. -->
## Codebase Intelligence for zed-preview-custom (Repowise)

Indexed by [Repowise](https://repowise.dev). Last indexed: 2026-08-31 (commit 9f45b38). Confidence: 100%.
### How to work in this repo

- **Trust the index.** `verified: true` means the bytes were checked against the live tree, so never re-read those lines. Re-read only on `bounds: "approximate"`, `_meta.stale_warning`, `search_method: "bm25"` or `confidence: "low"`; `index_behind: true` alone is informational.
- **Pre-edit, not instead-of-edit.** These tools decide *which* files to read and edit. Reading a file before you edit it is correct and expected.
- **Noisy commands** (tests, builds, `git log`/`diff`, searches, listings): prefer `repowise distill <cmd>`, the same command with its exit code preserved and errors-first output. A `[repowise#<ref>: N lines omitted]` marker is recoverable via `repowise expand <ref>` (add `-q <regex>` to filter); never re-run the command to see omitted output.
- **Recording a decision** you had to reason out: `repowise decision add --title T --decision D` records it without prompting and prints the id (`--format json` to parse it back). It lands `proposed`, for a person to confirm.

### Tools

| Tool | When and why |
|------|--------------|
| `get_answer(question)` | First call for any how/where/why question. Cite `confidence: "high"` or `grounding: "extracted"` directly; `degraded` means judge by `retrieval_quality`. `symbol_bodies` has live bodies. |
| `get_context(targets=[...])` | Triage card for files/modules/symbols: docs, signatures, hotspot, fix history. No source bytes — `include=["skeleton"]` for the whole file verified, `["callers"|"decisions"]` for depth. Batch targets. |
| `get_symbol(id, depth?)` | **Follow-up, not an entry point** — one verified body for an id a prior response named (`path.py::Name`, `path.py:140-180`, `repowise#<hex>`). Never walk a file symbol by symbol; Read it. |
| `search_codebase(query)` | Hybrid search, auto-routed by query shape; force with `mode=symbol|path|concept|hybrid`. A hit whose `sources` are `[fts]` only has no semantic agreement, so verify it. |
| `get_why(query, targets?)` | Why the code is shaped this way: decision records, git archaeology, rationale comments. Call before a refactor or a pattern divergence. |
| `get_risk(targets, changed_files?, include?)` | File history and structural reach. PR mode leads with `directive`; its 0-10 structural heuristic is uncalibrated, not a probability. Read typed test recommendations and coverage state first. |
| `get_change_risk(revspec?, extensions?, exclude_patterns?)` | Deterministic live-diff review signal for a commit or range. Lead with benchmarked percentile/classification; the 0-10 diff-shape score is supporting, not a probability. `get_risk` scores paths. |
| `get_health(targets?, include?)` | Defect / maintainability / performance scores and findings. Self-check the files you touched before finishing. |
| `get_dead_code(tier?, min_confidence?, safe_only?)` | Confidence-tiered unreachable files / unused exports / zombie packages. For cleanup sweeps, not targeted fixes. |
| `get_overview()` | Architecture map. Call once, first, in an unfamiliar repo; skip it after that. |

### Architecture
Zed is a high-performance, multiplayer code editor and AI development environment that ingests user input, file system events, Language Server Protocol (LSP) diagnostics, and LLM streams through a custom GPU-accelerated rendering pipeline and CRDT-backed buffer synchronization engine to deliver real-time interactive code editing, semantic navigation, and collaborative workspaces. ---
---
---

### Key modules
- `crates/gpui/src` — State and execution lifetimes in GPUI resolve through a hierarchy of context handles anchored by crates/gpui/src/app.rs and…
- `crates/editor/src` — User interactions with an active buffer enter the subsystem via input handlers in crates/editor/src/input.rs and the declarative action…
- `crates/ui/src/components/notification` — User-facing components in this subsystem manage foreground alerts, status updates, and operation progress without coupling directly to…
- `crates/agent_ui/src` — The primary workspace viewport for conversational sessions is housed in crates/agent_ui/src/agent_panel.rs and bound into the editor…
- `crates/ui/src/components` — Specialized visual features and workflow-specific components are organized into dedicated sub-directories
- `crates/eval_cli` — The evaluation runtime divides responsibilities between native Rust binaries and high-level Python workflow scripts
- `crates/language_tools/src` — The primary entry point at crates/languages/src/lib.rs coordinates registration for out-of-the-box language bundles
- `crates/gpui/examples` — The styling examples illustrate GPUI’s utility-first approach to UI construction
- `crates/project/src` — At the center of workspace operations, crates/project/src/project.rs aggregates open working directories, tracking buffers, file tree…
- `crates/collab/src` — The subsystem's runtime starts at the binary entry point crates/collab/src/main.rs, which bootstraps configuration via…

### Entry points
- `script/update_top_ranking_issues/main.py`
- `crates/gpui/src/app.rs`
- `crates/edit_prediction_cli/src/main.rs`
- `crates/zed/src/main.rs`
- `crates/edit_prediction_metrics/src/main.rs`
- `crates/remote_server/src/server.rs`

### Files that need care (bug-fix history first, then churn — check `get_risk` before editing)
- `crates/editor/src/editor.rs` — 50 bug fixes, last fix 5 weeks ago (bug magnet); 22 commits/90d
- `crates/editor/src/editor_tests.rs` — 38 bug fixes, last fix 6 weeks ago (bug magnet); 25 commits/90d
- `crates/agent_ui/src/agent_panel.rs` — 33 bug fixes, last fix 7 weeks ago (bug magnet); 26 commits/90d
- `crates/agent_ui/src/conversation_view/thread_view.rs` — 24 bug fixes, last fix 6 weeks ago (bug magnet); 44 commits/90d
- `crates/sidebar/src/sidebar.rs` — 30 bug fixes, last fix 3 months ago (bug magnet); 2 commits/90d

### Code health
Three co-equal signals: defect risk 4.84/10 avg, hotspot health 2.69/10 (stable), worst `crates/agent/src/tests/mod.rs` at 1.0/10 · maintainability 5.08/10 · performance risk 478 open static I/O-in-loop / N+1 findings. Detail: `get_health()`.

Critical files:
- `crates/ui/src/components/label/label.rs` — ownership risk — impact −2.8
- `crates/ui/src/components/keybinding_hint.rs` — ownership risk — impact −2.8
- `crates/repl/src/notebook/cell.rs` — ownership risk — impact −2.8
- `crates/prompt_store/src/prompts.rs` — ownership risk — impact −2.8
- `crates/language/src/outline.rs` — ownership risk — impact −2.8

<!-- REPOWISE_AGENTS:END -->
