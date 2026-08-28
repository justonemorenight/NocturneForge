# NocturneForge

NocturneForge is an independent, AI-native code editor built on the open-source
[Zed](https://github.com/zed-industries/zed) foundation. It is developed by
`justonemorenight` and is not an official product of Zed Industries.

The project keeps Zed's full Git ancestry and tracks the upstream repository so
that fixes and improvements can be reviewed, fetched, and cherry-picked without
losing provenance.

## Why NocturneForge

NocturneForge focuses on making agent work observable, controllable, and
recoverable while preserving Zed's fast native editor, LSP, DAP, terminal,
sandbox, and diff-review workflows.

- Native orchestration runtime for task planning, scheduling, verification,
  retry, cancellation, and artifact tracking.
- `Direct`, `Plan`, `Orchestrate`, and `Auto` execution strategies with
  `Manual`, `Supervised`, and `Autonomous` policies.
- Agent Activity Center for native agents and ACP sessions.
- ACP interoperability with OMP (Oh My Pi), OpenCode, and other ACP clients.
- Structured plans, execution checklists, context visibility, and reviewable
  changes instead of opaque prompt-only orchestration.
- The editor remains native Zed: GPU rendering, LSP/DAP, terminal, sandbox,
  collaboration, and project context are retained.

## Project status

| Area | Status |
| --- | --- |
| Native agent strategy selector | Implemented |
| Native and ACP activity surfaces | Implemented / iterating |
| Structured plans and execution policies | Implemented / iterating |
| Orchestration runtime and verification loop | In progress |
| OMP/OpenCode interoperability | In progress |
| Cross-platform branded releases | Planned |

## Architecture

```text
Native Thread / ACP Session
            │
            ▼
   Orchestration Runtime
     ├─ Plan Graph
     ├─ Scheduler + Task Registry
     ├─ Verification / Retry Loop
     ├─ Cancellation Tree
     └─ Context + Artifact Store
            │
            ▼
     Activity Center / Review UI
```

The runtime is deliberately separate from the conversation view. A thread can
run directly, present a plan for approval, or coordinate independent tasks in
waves. Each task exposes lifecycle, model/effort, context, outputs, and
cancellation state to the UI.

## Build on macOS

The supported local release target is Apple Silicon:

```bash
TERM=xterm-256color \
ZED_BUNDLE_CHANNEL=preview \
CARGO_BUNDLE_SKIP_BUILD=1 \
./script/bundle-mac aarch64-apple-darwin
```

The build script supports an external `target` symlink, which is useful when
the local disk is small. Verify the mounted volume before building and never
run two bundle jobs concurrently. The signed app and DMG are written below
`target/aarch64-apple-darwin/release/`.

The executable and CLI intentionally remain named `zed` for compatibility with
existing settings, extensions, scripts, and shell aliases. The packaged app is
branded NocturneForge with its own bundle identifier.

## Agent configuration

Native Agent strategies are selected in the conversation header. ACP servers
can be configured with their normal command, for example:

```json
{
  "agent_servers": {
    "Oh My Pi": { "command": "omp", "args": ["acp"] },
    "OpenCode": { "command": "opencode", "args": ["acp"] }
  }
}
```

Use `Plan` when you want to inspect and approve a checklist, `Orchestrate` for
multi-task work with verification and retry, and `Direct` for a focused change.
`Auto` chooses a strategy from task scope and policy; the user can override it.

## Staying in sync with Zed

The repository remotes are intended to be:

```text
origin   → https://github.com/justonemorenight/NocturneForge.git
upstream → https://github.com/zed-industries/zed.git
backup   → https://github.com/justonemorenight/zed.git
```

Fetch upstream, inspect the diff, and preserve provenance when porting a
merged change:

```bash
git fetch upstream main
git log --oneline upstream/main..HEAD
git cherry-pick -x <upstream-commit>
```

NocturneForge keeps Zed's commit ancestry; upstream attribution is retained in
history and release notes.

## Development and testing

See the upstream development guides under docs/src/development for platform
prerequisites. Run focused crate checks while iterating, then use
`./script/clippy` and the relevant test suites before publishing a release.

Changes that affect the agent runtime should include coverage for cancellation,
retry, persistence, ACP compatibility, and native rendering. Release builds
must pass bundle, code-signing, arm64, icon, and smoke-test verification.

## Contributing and releases

Issues and pull requests should describe the user-facing behavior, affected
agent/runtime surfaces, tests run, and any upstream PR or commit being ported.
Release notes call out changes to Native Agent, ACP, orchestration, packaging,
and compatibility independently. Follow the existing contribution and license
requirements in this repository.

## License and attribution

NocturneForge remains licensed under GPL-3.0-or-later where applicable, with
Apache-2.0 components and third-party licenses preserved as marked. License
metadata and attribution generated by `cargo-about` are part of the release
process. NocturneForge is a downstream project and does not claim Zed
Industries' trademarks, branding, or official support.
