import { test } from 'node:test';
import assert from 'node:assert/strict';
import { pathToFileURL } from 'node:url';
import { homedir } from 'node:os';
import { join } from 'node:path';
const modulePath = process.env.NATIVE_SUBAGENTS_MODULE ?? join(homedir(), 'Library/Application Support/Zed/external_agents/registry/npx/claude-acp/node_modules/@agentclientprotocol/claude-agent-acp/dist/native-subagents.js');
const { NativeSubagentRuntime } = await import(pathToFileURL(modulePath));
function fixture(state = {}, events = []) {
  const runtime = new NativeSubagentRuntime(true, 'root', state, async event => events.push(event), { log() {} });
  const deliver = async notification => {
    const routed = await runtime.route(notification, deliver);
    if (routed) events.push(routed);
  };
  return { runtime, state, events, deliver };
}
function notification(parentToolUseId, sessionUpdate = 'agent_message_chunk', toolCallId) {
  return { sessionId: 'root', update: {
    sessionUpdate, ...(toolCallId ? { toolCallId } : {}),
    ...(sessionUpdate === 'agent_message_chunk' ? { content: { type: 'text', text: 'Finished read-only task' } } : {}),
    _meta: { claudeCode: { parentToolUseId } },
  }};
}
async function start(f, toolUseId = 'original-agent-call') {
  if (!f.state.nativeSubagentsByTaskId?.has('worker')) {
    await f.runtime.route({ sessionId: 'root', update: { sessionUpdate: 'tool_call', toolCallId: toolUseId, rawInput: { description: 'Read file', subagent_type: 'multi-openai' }, _meta: { claudeCode: { toolName: 'Agent' } } } }, f.deliver);
  }
  await f.runtime.taskStarted({ taskId: 'worker', toolUseId, subagentType: 'multi-openai', description: 'Read file · OpenAI · gpt-6.1-sol' }, f.deliver);
}
async function finish(f, toolUseId = 'original-agent-call') {
  await f.runtime.finishTask('worker', 'completed', f.deliver, toolUseId);
}
async function resume(f, toolUseId = 'send-message-2') {
  await start(f, toolUseId);
  await f.runtime.taskResumed('worker', f.deliver, 'Read again');
}

test('observed trace: resumed Read and answer reach generation 2 despite original parent ID', async () => {
  const f = fixture();
  await start(f);
  await finish(f);
  await resume(f);
  const resumed = 'worker:generation:2';
  const read = await f.runtime.route(notification('original-agent-call', 'tool_call', 'new-read'), f.deliver);
  assert.equal(read?.sessionId, resumed, 'Read must not be dropped by completed generation 1');
  const readResult = await f.runtime.route(notification('original-agent-call', 'tool_call_update', 'new-read'), f.deliver);
  assert.equal(readResult?.sessionId, resumed);
  assert.equal((await f.runtime.route(notification('original-agent-call'), f.deliver))?.sessionId, resumed);
  await finish(f, 'send-message-2');
  assert.equal(f.events.at(-1).update.subagentSessionId, resumed);
  assert.equal(f.events.at(-1).update.state, 'completed');
});

test('multiple resumes accept original, previous resume and current parent IDs without spawning duplicates', async () => {
  const f = fixture(); await start(f); await finish(f); await resume(f); await finish(f, 'send-message-2');
  await resume(f, 'send-message-3');
  for (const parent of ['original-agent-call', 'send-message-2', 'send-message-3']) {
    assert.equal((await f.runtime.route(notification(parent, 'tool_call', `read-${parent}`), f.deliver))?.sessionId, 'worker:generation:3');
  }
  assert.deepEqual(f.events.filter(e => e.update.sessionUpdate === 'subagent_spawned').map(e => e.update.subagentSessionId), ['worker', 'worker:generation:2', 'worker:generation:3']);
});

test('late tools keep old ownership even when their parent ID becomes a stream alias', async () => {
  const f = fixture(); await start(f);
  await f.runtime.route(notification('original-agent-call', 'tool_call', 'old-read'), f.deliver);
  const asyncRoute = f.runtime.routeOfToolCall('old-read');
  await finish(f); await resume(f);
  assert.equal(await f.runtime.route(notification('original-agent-call', 'tool_call_update', 'old-read'), f.deliver), null);
  assert.equal(await f.runtime.route({ sessionId: 'root', update: { sessionUpdate: 'tool_call_update', toolCallId: 'old-read' } }, f.deliver), null);
  assert.equal(asyncRoute({ sessionId: 'root', update: { sessionUpdate: 'async_task_progress', taskId: 'old-job' } }).sessionId, 'worker');
  assert.equal((await f.runtime.route(notification('original-agent-call', 'tool_call', 'fresh-read'), f.deliver))?.sessionId, 'worker:generation:2');
});

test('old lifecycle notification cannot finish generation 2 through its stream alias', async () => {
  const f = fixture(); await start(f); await finish(f); await resume(f);
  const before = f.events.length;
  await f.runtime.finishTask('worker', 'completed', f.deliver, 'original-agent-call');
  assert.equal(f.events.length, before);
  assert.equal((await f.runtime.route(notification('original-agent-call'), f.deliver))?.sessionId, 'worker:generation:2');
});

test('runtime reconstruction restores stream aliases of current generation', async () => {
  const f = fixture(); await start(f); await finish(f); await resume(f);
  const restored = fixture(f.state, f.events);
  assert.equal((await restored.runtime.route(notification('original-agent-call', 'tool_call', 'restored-read'), restored.deliver))?.sessionId, 'worker:generation:2');
});

test('resume without task_started and same-ID task_started retain existing behavior', async () => {
  for (const method of ['result-only', 'same-id']) {
    const f = fixture(); await start(f); await finish(f);
    if (method === 'same-id') await start(f);
    await f.runtime.taskResumed('worker', f.deliver, 'Read again');
    assert.equal((await f.runtime.route(notification('original-agent-call', 'tool_call', 'read'), f.deliver))?.sessionId, 'worker:generation:2');
  }
});

test('a nested agent spawned under the original stream ID belongs to the resumed parent', async () => {
  const f = fixture(); await start(f); await finish(f); await resume(f);
  await f.runtime.route({ sessionId: 'root', update: { sessionUpdate: 'tool_call', toolCallId: 'nested-launch', rawInput: { description: 'Nested read', subagent_type: 'Explore' }, _meta: { claudeCode: { parentToolUseId: 'original-agent-call', toolName: 'Agent' } } } }, f.deliver);
  await f.runtime.taskStarted({ taskId: 'nested-worker', toolUseId: 'nested-launch', subagentType: 'Explore' }, f.deliver);
  const spawned = f.events.find(e => e.update.subagentSessionId === 'nested-worker');
  assert.equal(spawned.sessionId, 'worker:generation:2');
});
