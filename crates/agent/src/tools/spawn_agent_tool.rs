use acp_thread::{SUBAGENT_SESSION_INFO_META_KEY, SubagentSessionInfo};
use agent_client_protocol::schema::v1 as acp;
use anyhow::Result;
use futures::future::join_all;
use gpui::{App, SharedString, Task};
use language_model::LanguageModelToolResultContent;
use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::HashSet;
use std::rc::Rc;
use std::sync::Arc;

use crate::{AgentTool, SubagentRole, ThreadEnvironment, ToolCallEventStream, ToolInput};

const MAX_PARALLEL_SUBAGENTS: usize = 4;
const MAX_SUBAGENT_RETRIES: u8 = 2;

/// Spawn a sub-agent for a well-scoped task.
///
/// ### Designing delegated subtasks
/// - An agent does not see your conversation history. Include all relevant context (file paths, requirements, constraints) in the message.
/// - Subtasks must be concrete, well-defined, and self-contained.
/// - Delegated subtasks must materially advance the main task.
/// - Do not duplicate work between your work and delegated subtasks.
/// - Do not use this tool for tasks you could accomplish directly with one or two tool calls. For example, don't ask the agent to read a single file and return the contents, you can do this yourself.
/// - When you delegate work, focus on coordinating and synthesizing results instead of duplicating the same work yourself.
/// - Avoid issuing multiple delegate calls for the same unresolved subproblem unless the new delegated task is genuinely different and necessary.
/// - Narrow the delegated ask to the concrete output you need next.
/// - For code-edit subtasks, decompose work so each delegated task has a disjoint write set.
/// - When sending a follow-up using an existing agent session_id, the agent already has the context from the previous turn. Send only a short, direct message. Do NOT repeat the original task or context.
///
/// ### Parallel delegation patterns
/// - Run multiple independent information-seeking subtasks in parallel when you have distinct questions that can be answered independently.
/// - Split implementation into disjoint codebase slices and spawn multiple agents for them in parallel when the write scopes do not overlap.
/// - When a plan has multiple independent steps, prefer delegating those steps in parallel rather than serializing them unnecessarily.
/// - Reuse the returned session_id when you want to follow up on the same delegated subproblem instead of creating a duplicate session.
///
/// ### Restricting subagent tools
/// - By default a subagent inherits all of your tools. Pass `tools` to restrict it to an allowlist — for example read-only tools like ["read_file", "grep", "find_path"] for a search task, or an empty list for a pure reasoning task over content in the message.
/// - A scoped allowlist keeps focused subtasks from performing side effects you did not intend.
///
/// ### Output
/// - You will receive only the agent's final message as output.
/// - Successful calls return a session_id that you can use for follow-up messages.
/// - Error results may also include a session_id if a session was already created.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct SpawnAgentToolInput {
    /// Short label displayed in the UI while the agent runs (e.g., "Researching alternatives")
    pub label: String,
    /// The prompt for the agent. For new sessions, include full context needed for the task. For follow-ups (with session_id), you can rely on the agent already having the previous message.
    pub message: String,
    /// Agent type for a new native ChatGPT Subscription subagent. Use explorer for
    /// file/symbol lookup, flow-reader for flow/log analysis, and coding-worker
    /// for bounded implementation. Required for new ChatGPT Subscription
    /// sessions and ignored when continuing an existing session.
    pub agent_type: Option<SubagentRole>,
    /// Session ID of an existing agent session to continue instead of creating a new one. Omit to create a new agent.
    #[serde(default, deserialize_with = "deserialize_session_id")]
    pub session_id: Option<acp::SessionId>,
    /// Optional allowlist of tool names the subagent may use (e.g. ["read_file", "grep"]). When present, the subagent is restricted to only those tools; when omitted, it inherits your full tool set. An empty list gives the subagent no tools at all. Names must be a subset of your available tools. Ignored when resuming an existing session — the session keeps the filter it was created with.
    #[serde(default)]
    pub tools: Option<Vec<String>>,
    /// Optional independent tasks to start concurrently. When present, the
    /// single-task fields above are ignored.
    #[serde(default)]
    pub tasks: Option<Vec<SpawnAgentTask>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct SpawnAgentTask {
    /// Stable identifier used by other tasks in `depends_on`. When omitted,
    /// the scheduler assigns `task-N` in input order.
    #[serde(default)]
    pub id: Option<String>,
    pub label: String,
    pub message: String,
    pub agent_type: Option<SubagentRole>,
    #[serde(default)]
    pub tools: Option<Vec<String>>,
    /// Task IDs that must complete successfully before this task starts.
    #[serde(default)]
    pub depends_on: Vec<String>,
    /// Number of times to retry this task after a failed request.
    #[serde(default)]
    pub max_retries: u8,
    /// Optional hard budget for the task's cumulative token usage.
    #[serde(default)]
    pub token_budget: Option<u64>,
}

fn deserialize_session_id<'de, D>(deserializer: D) -> Result<Option<acp::SessionId>, D::Error>
where
    D: Deserializer<'de>,
{
    let Some(value) = Option::<serde_json::Value>::deserialize(deserializer)? else {
        return Ok(None);
    };

    if value
        .as_str()
        .is_some_and(|session_id| session_id.trim().is_empty())
    {
        return Ok(None);
    }

    serde_json::from_value(value)
        .map(Some)
        .map_err(serde::de::Error::custom)
}

fn validate_task_graph(tasks: &[(String, SpawnAgentTask)]) -> Result<()> {
    let task_ids = tasks
        .iter()
        .map(|(task_id, _)| task_id.clone())
        .collect::<HashSet<_>>();
    if task_ids.len() != tasks.len() {
        anyhow::bail!("task IDs must be unique");
    }
    if tasks
        .iter()
        .any(|(_, task)| task.max_retries > MAX_SUBAGENT_RETRIES)
    {
        anyhow::bail!("max_retries cannot exceed {MAX_SUBAGENT_RETRIES}");
    }
    if tasks
        .iter()
        .any(|(_, task)| task.token_budget.is_some_and(|budget| budget == 0))
    {
        anyhow::bail!("token_budget must be greater than zero");
    }
    if let Some(dependency) = tasks
        .iter()
        .flat_map(|(_, task)| task.depends_on.iter())
        .find(|dependency| !task_ids.contains(*dependency))
    {
        anyhow::bail!("unknown task dependency `{dependency}`");
    }

    let mut completed = HashSet::new();
    while completed.len() < tasks.len() {
        let ready = tasks
            .iter()
            .filter(|(task_id, task)| {
                !completed.contains(task_id)
                    && task
                        .depends_on
                        .iter()
                        .all(|dependency| completed.contains(dependency))
            })
            .map(|(task_id, _)| task_id.clone())
            .collect::<Vec<_>>();
        if ready.is_empty() {
            anyhow::bail!("task dependency graph contains a cycle");
        }
        completed.extend(ready);
    }
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
#[serde(rename_all = "snake_case")]
pub enum SpawnAgentToolOutput {
    Success {
        session_id: acp::SessionId,
        output: String,
        session_info: SubagentSessionInfo,
    },
    Error {
        #[serde(skip_serializing_if = "Option::is_none")]
        #[serde(default)]
        session_id: Option<acp::SessionId>,
        error: String,
        session_info: Option<SubagentSessionInfo>,
    },
    BatchSuccess {
        results: Vec<SpawnAgentBatchResult>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpawnAgentBatchResult {
    pub task_id: String,
    pub session_id: acp::SessionId,
    pub label: String,
    pub output: String,
}

impl From<SpawnAgentToolOutput> for LanguageModelToolResultContent {
    fn from(output: SpawnAgentToolOutput) -> Self {
        match output {
            SpawnAgentToolOutput::Success {
                session_id,
                output,
                session_info: _, // Don't show this to the model
            } => serde_json::to_string(
                &serde_json::json!({ "session_id": session_id, "output": output }),
            )
            .unwrap_or_else(|e| format!("Failed to serialize spawn_agent output: {e}"))
            .into(),
            SpawnAgentToolOutput::Error {
                session_id,
                error,
                session_info: _, // Don't show this to the model
            } => serde_json::to_string(
                &serde_json::json!({ "session_id": session_id, "error": error }),
            )
            .unwrap_or_else(|e| format!("Failed to serialize spawn_agent output: {e}"))
            .into(),
            SpawnAgentToolOutput::BatchSuccess { results } => serde_json::to_string(&results)
                .unwrap_or_else(|e| format!("Failed to serialize spawn_agent output: {e}"))
                .into(),
        }
    }
}

/// Tool that spawns an agent thread to work on a task.
pub struct SpawnAgentTool {
    environment: Rc<dyn ThreadEnvironment>,
}

impl SpawnAgentTool {
    pub fn new(environment: Rc<dyn ThreadEnvironment>) -> Self {
        Self { environment }
    }
}

impl AgentTool for SpawnAgentTool {
    type Input = SpawnAgentToolInput;
    type Output = SpawnAgentToolOutput;

    const NAME: &'static str = "spawn_agent";

    fn kind() -> acp::ToolKind {
        acp::ToolKind::Other
    }

    fn initial_title(
        &self,
        input: Result<Self::Input, serde_json::Value>,
        _cx: &mut App,
    ) -> SharedString {
        match input {
            Ok(i) => i.label.into(),
            Err(value) => value
                .get("label")
                .and_then(|v| v.as_str())
                .map(|s| SharedString::from(s.to_owned()))
                .unwrap_or_else(|| "Spawning agent".into()),
        }
    }

    fn run(
        self: Arc<Self>,
        input: ToolInput<Self::Input>,
        event_stream: ToolCallEventStream,
        cx: &mut App,
    ) -> Task<Result<Self::Output, Self::Output>> {
        cx.spawn(async move |cx| {
            let input = input
                .recv()
                .await
                .map_err(|e| SpawnAgentToolOutput::Error {
                    session_id: None,
                    error: e.to_string(),
                    session_info: None,
                })?;

            if let Some(tasks) = input.tasks {
                if tasks.is_empty() {
                    return Err(SpawnAgentToolOutput::Error {
                        session_id: None,
                        error: "tasks must contain at least one subagent task".to_string(),
                        session_info: None,
                    });
                }
                let mut pending = tasks
                    .into_iter()
                    .enumerate()
                    .map(|(index, task)| {
                        let task_id = task
                            .id
                            .clone()
                            .unwrap_or_else(|| format!("task-{}", index + 1));
                        (task_id, task)
                    })
                    .collect::<Vec<_>>();
                validate_task_graph(&pending).map_err(|error| SpawnAgentToolOutput::Error {
                    session_id: None,
                    error: error.to_string(),
                    session_info: None,
                })?;

                let mut completed = HashSet::new();
                let mut batch = Vec::with_capacity(pending.len());
                while !pending.is_empty() {
                    if event_stream.was_cancelled_by_user() {
                        return Err(SpawnAgentToolOutput::Error {
                            session_id: None,
                            error: "parallel orchestration cancelled".to_string(),
                            session_info: None,
                        });
                    }
                    let (ready, blocked): (Vec<_>, Vec<_>) =
                        pending.into_iter().partition(|(_, task)| {
                            task.depends_on
                                .iter()
                                .all(|dependency| completed.contains(dependency))
                        });
                    if ready.is_empty() {
                        return Err(SpawnAgentToolOutput::Error {
                            session_id: None,
                            error: "task dependency graph contains a cycle".to_string(),
                            session_info: None,
                        });
                    }
                    let (ready, deferred) = if ready.len() > MAX_PARALLEL_SUBAGENTS {
                        let mut ready = ready;
                        let deferred = ready.split_off(MAX_PARALLEL_SUBAGENTS);
                        (ready, deferred)
                    } else {
                        (ready, Vec::new())
                    };
                    pending = blocked.into_iter().chain(deferred).collect();
                    let spawned = cx
                        .update(|cx| {
                            ready
                                .into_iter()
                                .map(|(task_id, task)| {
                                    let tool_filter = task.tools.map(|tools| {
                                        tools.into_iter().map(SharedString::from).collect()
                                    });
                                    let subagent = self.environment.create_subagent(
                                        task.label.clone(),
                                        task.agent_type,
                                        tool_filter,
                                        cx,
                                    )?;
                                    event_stream.subagent_spawned(subagent.id());
                                    anyhow::Ok((
                                        task_id,
                                        task.label,
                                        task.message,
                                        task.max_retries,
                                        task.token_budget,
                                        subagent,
                                    ))
                                })
                                .collect::<Result<Vec<_>>>()
                        })
                        .map_err(|error| SpawnAgentToolOutput::Error {
                            session_id: None,
                            error: error.to_string(),
                            session_info: None,
                        })?;
                    let results = join_all(spawned.into_iter().map(
                        |(task_id, label, message, max_retries, token_budget, subagent)| {
                            let app = cx.clone();
                            async move {
                                let session_id = subagent.id();
                                let initial_entries = app
                                    .update(|cx| subagent.num_entries(cx));
                                let mut attempts = 0;
                                let output = loop {
                                    match subagent.send(message.clone(), &app).await {
                                        Ok(output) => {
                                            if let Some(budget) = token_budget {
                                                let exceeded = app
                                                    .update(|cx| subagent.used_tokens(cx))
                                                    .is_some_and(|used| used > budget);
                                                if exceeded {
                                                    break Err(anyhow::anyhow!(
                                                        "token budget of {budget} was exceeded"
                                                    ));
                                                }
                                            }
                                            break Ok(output);
                                        }
                                        Err(error) => {
                                            let retry_safe = app
                                                .update(|cx| subagent.num_entries(cx) == initial_entries);
                                            if retry_safe && attempts < max_retries {
                                                attempts += 1;
                                            } else {
                                                break Err(error);
                                            }
                                        }
                                    }
                                };
                                (task_id, label, session_id, output)
                            }
                        },
                    ))
                    .await;
                    if event_stream.was_cancelled_by_user() {
                        return Err(SpawnAgentToolOutput::Error {
                            session_id: None,
                            error: "parallel orchestration cancelled".to_string(),
                            session_info: None,
                        });
                    }
                    for (task_id, label, session_id, output) in results {
                        match output {
                            Ok(output) => {
                                completed.insert(task_id.clone());
                                batch.push(SpawnAgentBatchResult {
                                    task_id,
                                    session_id,
                                    label,
                                    output,
                                });
                            }
                            Err(error) => {
                                let completed_summary = batch
                                    .iter()
                                    .map(|result| result.task_id.as_str())
                                    .collect::<Vec<_>>()
                                    .join(", ");
                                let skipped_count = pending.len();
                                return Err(SpawnAgentToolOutput::Error {
                                    session_id: Some(session_id),
                                    error: format!(
                                        "{label}: {error}; completed: [{}]; skipped: {skipped_count}",
                                        completed_summary
                                    ),
                                    session_info: None,
                                });
                            }
                        }
                    }
                }
                let raw_output = match serde_json::to_value(&batch) {
                    Ok(raw_output) => raw_output,
                    Err(error) => {
                        return Err(SpawnAgentToolOutput::Error {
                            session_id: None,
                            error: format!("Failed to serialize batch output: {error}"),
                            session_info: None,
                        });
                    }
                };
                event_stream.update_fields(
                    acp::ToolCallUpdateFields::new()
                        .title("Parallel agents completed")
                        .raw_output(raw_output),
                );
                return Ok(SpawnAgentToolOutput::BatchSuccess { results: batch });
            }

            let (subagent, mut session_info) = cx.update(|cx| {
                let subagent = if let Some(session_id) = input.session_id {
                    // A resumed session keeps the tool filter it was created
                    // with; `tools` is intentionally ignored here.
                    self.environment.resume_subagent(session_id, cx)
                } else {
                    let tool_filter = input
                        .tools
                        .map(|tools| tools.into_iter().map(SharedString::from).collect());
                    self.environment
                        .create_subagent(input.label, input.agent_type, tool_filter, cx)
                };
                let subagent = subagent.map_err(|err| SpawnAgentToolOutput::Error {
                    session_id: None,
                    error: err.to_string(),
                    session_info: None,
                })?;
                let session_info = SubagentSessionInfo {
                    session_id: subagent.id(),
                    message_start_index: subagent.num_entries(cx),
                    message_end_index: None,
                };

                event_stream.subagent_spawned(subagent.id());
                event_stream.update_fields_with_meta(
                    acp::ToolCallUpdateFields::new(),
                    Some(acp::Meta::from_iter([(
                        SUBAGENT_SESSION_INFO_META_KEY.into(),
                        serde_json::json!(&session_info),
                    )])),
                );

                Ok((subagent, session_info))
            })?;

            let send_result = subagent.send(input.message, cx).await;

            let status = if send_result.is_ok() {
                "completed"
            } else {
                "error"
            };
            telemetry::event!(
                "Subagent Completed",
                subagent_session = session_info.session_id.to_string(),
                status,
            );

            session_info.message_end_index =
                cx.update(|cx| Some(subagent.num_entries(cx).saturating_sub(1)));

            let meta = Some(acp::Meta::from_iter([(
                SUBAGENT_SESSION_INFO_META_KEY.into(),
                serde_json::json!(&session_info),
            )]));

            let (output, result) = match send_result {
                Ok(output) => (
                    output.clone(),
                    Ok(SpawnAgentToolOutput::Success {
                        session_id: session_info.session_id.clone(),
                        session_info,
                        output,
                    }),
                ),
                Err(e) => {
                    let error = e.to_string();
                    (
                        error.clone(),
                        Err(SpawnAgentToolOutput::Error {
                            session_id: Some(session_info.session_id.clone()),
                            error,
                            session_info: Some(session_info),
                        }),
                    )
                }
            };
            event_stream.update_fields_with_meta(
                acp::ToolCallUpdateFields::new().content(vec![output.into()]),
                meta,
            );
            result
        })
    }

    fn replay(
        &self,
        _input: Self::Input,
        output: Self::Output,
        event_stream: ToolCallEventStream,
        _cx: &mut App,
    ) -> Result<()> {
        let (content, session_info) = match output {
            SpawnAgentToolOutput::Success {
                output,
                session_info,
                ..
            } => (output.into(), Some(session_info)),
            SpawnAgentToolOutput::Error {
                error,
                session_info,
                ..
            } => (error.into(), session_info),
            SpawnAgentToolOutput::BatchSuccess { results } => (
                serde_json::to_string(&results)
                    .unwrap_or_else(|error| format!("Failed to serialize batch output: {error}"))
                    .into(),
                None,
            ),
        };

        let meta = session_info.map(|session_info| {
            acp::Meta::from_iter([(
                SUBAGENT_SESSION_INFO_META_KEY.into(),
                serde_json::json!(&session_info),
            )])
        });
        event_stream.update_fields_with_meta(
            acp::ToolCallUpdateFields::new().content(vec![content]),
            meta,
        );

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn deserializes_blank_session_id_as_absent() {
        for session_id in [json!(null), json!(""), json!("   ")] {
            let input: SpawnAgentToolInput = serde_json::from_value(json!({
                "label": "label",
                "message": "message",
                "session_id": session_id,
            }))
            .unwrap();

            assert!(input.session_id.is_none());
        }

        let input: SpawnAgentToolInput = serde_json::from_value(json!({
            "label": "label",
            "message": "message",
        }))
        .unwrap();
        assert!(input.session_id.is_none());

        let input: SpawnAgentToolInput = serde_json::from_value(json!({
            "label": "label",
            "message": "message",
            "session_id": "existing-session",
        }))
        .unwrap();
        assert_eq!(input.session_id.unwrap().to_string(), "existing-session");
    }

    #[test]
    fn deserializes_tools_allowlist() {
        // Absent or explicit null means no restriction.
        let input: SpawnAgentToolInput = serde_json::from_value(json!({
            "label": "label",
            "message": "message",
        }))
        .unwrap();
        assert!(input.tools.is_none());

        let input: SpawnAgentToolInput = serde_json::from_value(json!({
            "label": "label",
            "message": "message",
            "tools": null,
        }))
        .unwrap();
        assert!(input.tools.is_none());

        // An empty list means the subagent gets no tools at all.
        let input: SpawnAgentToolInput = serde_json::from_value(json!({
            "label": "label",
            "message": "message",
            "tools": [],
        }))
        .unwrap();
        assert_eq!(input.tools, Some(Vec::new()));

        let input: SpawnAgentToolInput = serde_json::from_value(json!({
            "label": "label",
            "message": "message",
            "tools": ["read_file", "grep"],
        }))
        .unwrap();
        assert_eq!(
            input.tools,
            Some(vec!["read_file".to_string(), "grep".to_string()])
        );
    }

    #[test]
    fn deserializes_parallel_tasks() {
        let input: SpawnAgentToolInput = serde_json::from_value(json!({
            "label": "ignored",
            "message": "ignored",
            "tasks": [
                {"label": "one", "message": "first", "agent_type": "explorer"},
                {"label": "two", "message": "second", "tools": ["read_file"]}
            ]
        }))
        .unwrap();

        let tasks = input.tasks.expect("parallel tasks should deserialize");
        assert_eq!(tasks.len(), 2);
        assert_eq!(tasks[0].label, "one");
        assert_eq!(tasks[1].tools, Some(vec!["read_file".to_string()]));
    }

    #[test]
    fn rejects_cyclic_task_graphs_before_dispatch() {
        let tasks = vec![
            (
                "one".to_string(),
                SpawnAgentTask {
                    id: Some("one".to_string()),
                    label: "one".to_string(),
                    message: "one".to_string(),
                    agent_type: None,
                    tools: None,
                    depends_on: vec!["two".to_string()],
                    max_retries: 0,
                    token_budget: None,
                },
            ),
            (
                "two".to_string(),
                SpawnAgentTask {
                    id: Some("two".to_string()),
                    label: "two".to_string(),
                    message: "two".to_string(),
                    agent_type: None,
                    tools: None,
                    depends_on: vec!["one".to_string()],
                    max_retries: 0,
                    token_budget: None,
                },
            ),
        ];
        assert!(validate_task_graph(&tasks).is_err());
    }

    #[test]
    fn rejects_unbounded_task_retries() {
        let tasks = vec![(
            "one".to_string(),
            SpawnAgentTask {
                id: Some("one".to_string()),
                label: "one".to_string(),
                message: "one".to_string(),
                agent_type: None,
                tools: None,
                depends_on: Vec::new(),
                max_retries: MAX_SUBAGENT_RETRIES + 1,
                token_budget: None,
            },
        )];
        assert!(validate_task_graph(&tasks).is_err());
    }

    #[test]
    fn rejects_zero_token_budget() {
        let tasks = vec![(
            "one".to_string(),
            SpawnAgentTask {
                id: Some("one".to_string()),
                label: "one".to_string(),
                message: "one".to_string(),
                agent_type: None,
                tools: None,
                depends_on: Vec::new(),
                max_retries: 0,
                token_budget: Some(0),
            },
        )];
        assert!(validate_task_graph(&tasks).is_err());
    }
}
