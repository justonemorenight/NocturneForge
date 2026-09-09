---
title: AI Agents in Zed
description: Compare Zed Agent, External Agents, and Terminal Threads.
---

# Agents

Zed supports three agent paths. Choose the path based on how you want agentic work to run.

| Agent path                                | Runs in                         | Uses                                                                  | Best when                                                                              |
| ----------------------------------------- | ------------------------------- | --------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| [Zed Agent](./zed-agent.md)               | Agent Panel and Threads Sidebar | Zed-configured LLM providers, native tools, skills, instructions, MCP | You want Zed's native agent integration                                                |
| [External Agents](./external-agents.md)   | Agent Panel and Threads Sidebar | ACP agent process and its own auth/config                             | You want Claude, Codex, OpenCode, Copilot, Cursor, Pi, or another ACP-integrated agent |
| [Terminal Threads](./terminal-threads.md) | Threads Sidebar and terminal    | Native CLI/TUI auth/config                                            | You want the tool's command-line experience organized in Zed                           |

An agent path is sometimes called a harness. It is the way agentic work is started, displayed, configured, and controlled in Zed.

## Agent Path vs. LLM Provider {#agent-path-vs-llm-provider}

| Question                                  | Start here                          |
| ----------------------------------------- | ----------------------------------- |
| Which agent or CLI should run the work?   | This page                           |
| Which model should power Zed AI features? | [LLM Providers](./llm-providers.md) |

The [Zed Agent](./zed-agent.md) uses models configured in Zed. [External Agents](./external-agents.md) and [Terminal Threads](./terminal-threads.md) may use their own model setup.

## Thread Types {#thread-types}

Threads are the units shown in the [Threads Sidebar](./parallel-agents.md#threads-sidebar). Thread types include:

- [Zed Agent](./zed-agent.md) threads
- [External Agent](./external-agents.md) threads
- [Terminal Threads](./terminal-threads.md)

Use [Parallel Agents](./parallel-agents.md) to run and manage multiple threads at once.

## Optional prompt-cache warming

NocturneForge can replay a successful native ChatGPT Subscription request to
attempt to retain its provider-side prompt cache. This consumes account usage;
retention and quota savings are not guaranteed. It is off by default.

1. Set `agent.cache_keepalive` to `true` in settings.
2. In a native ChatGPT Subscription thread, click **Cache: Off** to opt that
   thread in. The next successful real request arms warming.
3. Hover the cache control for status and app-session warming usage. Click
   **Cache: On** to stop. Thread opt-ins are not persisted across app restarts.

`agent.cache_keepalive_config` controls estimated TTL, lead time, idle window,
attempt limits, estimated input budget, deadline, output limit and capture size.
Defaults allow at most two attempts per idle period and four attempts per app
hour, with an estimated one-million-input-token hourly budget. Budgets are shared
across threads and account switches, and failed or cancelled attempts are not
refunded. Restarting the app resets the app-session ledger and all thread opt-ins.
Disabling the global setting also clears every thread opt-in.

Only one warming request runs at a time. New turns, cancellation, model changes
and thread opt-out invalidate the old capture. Account and settings changes are
also checked during warming. Captures are memory-only, bounded in size and
count, and released when their owner disappears or their idle window expires.
Errors, missing usage, abnormal completions and insufficient cache reuse stop
warming until another successful real request. Subagents are not warmed.

Warming responses never enter the transcript and never execute tools. Output
and timeout limits are client-side cancellation guards, **not server-enforced
billing caps**. Usage for interrupted streams may be incomplete; the status
explicitly counts these attempts rather than reporting them as free.

External ACP agents, including Claude Code, expose cache statistics when their
adapter reports them, but are not sent synthetic warming prompts. Their own
runtime controls their provider requests.
