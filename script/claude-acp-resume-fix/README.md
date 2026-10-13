# Claude ACP resumed-subagent routing fix

Confirmed working in NocturneForge on 2026-10-13 after restarting the app and
resuming a completed Claude Multi subagent. The affected adapter is
`@agentclientprotocol/claude-agent-acp` 0.88.0, using Claude Agent SDK 0.3.293.
The fix belongs in that external adapter, rather than in Zed's Rust UI.

## Root cause

On the first run, the SDK associates both lifecycle and output with the
original `Agent` tool-use ID. When `SendMessage` resumes the same agent,
`task_started` can carry the new `SendMessage` tool-use ID, while the child's
Read calls and answer still carry the original `parent_tool_use_id`.

The adapter creates `<agentId>:generation:2` using the new ID. Its old
parent-to-child entry still targets generation 1, which is Completed. As a
result, it drops the resumed tool updates and answer with
`ignoring late update for terminal subagent`. The new row can show zero tools
although the worker actually ran and completed.

The provider/model suffix is a symptom of the resume path using the SDK task
description instead of the original Agent input description. It does not
cause the routing failure.

## Fix and invariants

`native-subagents.patch` separates stream-parent aliases from lifecycle
ownership. Each generation inherits previous stream-parent IDs, routing fresh
output to the current session. The aliases are stored on the child so runtime
reconstruction restores them.

Known tool-call ownership is unchanged: late updates and captured background
routes stay with their original generation. Old lifecycle IDs also stay with
the original generation, so an old completion cannot finish a new generation
through a stream alias. Existing SDK run-ID guards remain in place.

Text/thought chunks without a tool or run identity remain ambiguous if the
SDK delivers them late under a reused parent ID. This patch cannot infer an
identity absent from the wire protocol.

## Apply after an adapter update

The patch targets the unmodified generated JavaScript in adapter 0.88.0 and
requires no trace instrumentation. Package updates may overwrite the local
fix. Do not blindly apply it to another version: inspect the upstream code
and run the regression tests first.

From this repository, on macOS:

```sh
adapter="$HOME/Library/Application Support/Zed/external_agents/registry/npx/claude-acp/node_modules/@agentclientprotocol/claude-agent-acp"
patch_file="$PWD/script/claude-acp-resume-fix/native-subagents.patch"
cp "$adapter/dist/native-subagents.js" "$adapter/dist/native-subagents.js.before-resume-fix"
patch --dry-run -d "$adapter" -p1 < "$patch_file"
patch -d "$adapter" -p1 < "$patch_file"
node --test script/claude-acp-resume-fix/resume-routing.test.mjs
```

If the dry run fails, stop and compare the installed code. The currently
patched adapter also contains optional trace instrumentation; the clean patch
is for reapplication to an unmodified 0.88.0 installation, not for stacking
onto the already fixed installation.

For a different installation, set `NATIVE_SUBAGENTS_MODULE` to the absolute
path of its `dist/native-subagents.js` when running the test.

Restart the app so the adapter loads the changed file. Let a subagent finish,
then use SendMessage to ask the same agent to Read a file and return three
remarks. Both the Read and answer should appear under the resumed session.

## Verification

The seven regression tests exercise the real installed runtime without API
requests. Before the fix, six failed and one passed. After the fix, all seven
passed; JavaScript syntax checks passed. The user then confirmed the live
resume scenario works in Zed.

Coverage includes the observed split-ID sequence, three generations,
duplicate resume signals, late old tools, background routes, old lifecycle
completion, runtime reconstruction, same-ID/result-only resume, and nested
child attribution. The full upstream test suite was not run.

The diagnostic flag `CLAUDE_ACP_SUBAGENT_TRACE=1` was enabled locally for
Claude Multi. Its logs use `[claude-acp-subagent-trace]` in Zed.log and record
IDs, source-of-name flags, routing and lifecycle events, without prompt or
tool-output content. Trace instrumentation is separate from this patch.

## Related upstream work

- [Claude ACP issue #1158](https://github.com/agentclientprotocol/claude-agent-acp/issues/1158)
  reports resumed generations being announced only when they finish.
- [Claude ACP PR #1173](https://github.com/agentclientprotocol/claude-agent-acp/pull/1173)
  fixes the announcement timing and is already present in 0.88.0. Its
  regression uses the same tool-use ID across generations, so it does not
  cover the split-ID sequence observed here.
- [Mjolnir issue #1120](https://github.com/BrokkAi/mjolnir/issues/1120)
  reports the related zero-working-subagent symptom in another ACP client.
