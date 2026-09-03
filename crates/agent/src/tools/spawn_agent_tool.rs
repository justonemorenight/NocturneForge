use acp_thread::{SUBAGENT_SESSION_INFO_META_KEY, SubagentSessionInfo};
use agent_client_protocol::schema::v1 as acp;
use anyhow::Result;
use futures::future::LocalBoxFuture;
use gpui::{App, SharedString, Task};
use language_model::LanguageModelToolResultContent;
use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::HashSet;
use std::rc::Rc;
use std::sync::Arc;

use crate::{AgentTool, SubagentRole, Thread, ThreadEnvironment, ToolCallEventStream, ToolInput};
use agent_orchestration::{VERIFICATION_END, VERIFICATION_START};

fn cumulative_token_delta(
    before: Option<language_model::TokenUsage>,
    after: Option<language_model::TokenUsage>,
) -> u64 {
    let before = before.unwrap_or_default();
    let Some(after) = after else {
        return 0;
    };
    after.total_tokens().saturating_sub(before.total_tokens())
}

fn task_execution_prompt(task: &agent_orchestration::OrchestrationTask) -> String {
    if task.acceptance_criteria.is_empty()
        && task.expected_output.is_none()
        && !task.evidence_required
    {
        return task.description.clone();
    }

    let criteria = task
        .acceptance_criteria
        .iter()
        .map(|criterion| format!("- {criterion}"))
        .collect::<Vec<_>>()
        .join("\n");
    let expected_output = task.expected_output.as_deref().unwrap_or("Not specified");
    format!(
        "{}\n\nAcceptance criteria:\n{}\n\nExpected output: {}\n\nAt the end of your response, include a JSON verification claim between `{VERIFICATION_START}` and `{VERIFICATION_END}`. Use this exact shape:\n{{\"criteria\":[{{\"criterion\":\"copy each criterion exactly\",\"passed\":true,\"evidence\":\"specific evidence\"}}],\"expected_output_satisfied\":true,\"citations\":[\"path/to/file.rs:123\"]}}\nDo not claim a criterion passed without concrete evidence.",
        task.description, criteria, expected_output
    )
}

fn verify_task_output(
    task: &agent_orchestration::OrchestrationTask,
    output: &str,
) -> agent_orchestration::VerificationResult {
    agent_orchestration::VerificationRunner.verify(task, output)
}

struct SubagentRuntimeExecutor {
    environment: Rc<dyn ThreadEnvironment>,
    app: gpui::AsyncApp,
    event_stream: ToolCallEventStream,
    roles_map: Arc<
        parking_lot::RwLock<
            collections::HashMap<agent_orchestration::TaskId, Option<SubagentRole>>,
        >,
    >,
}

impl SubagentRuntimeExecutor {
    pub(crate) fn new(
        environment: Rc<dyn ThreadEnvironment>,
        app: gpui::AsyncApp,
        event_stream: ToolCallEventStream,
        roles_map: Arc<
            parking_lot::RwLock<
                collections::HashMap<agent_orchestration::TaskId, Option<SubagentRole>>,
            >,
        >,
    ) -> Self {
        Self {
            environment,
            app,
            event_stream,
            roles_map,
        }
    }
}

impl agent_orchestration::TaskExecutor for SubagentRuntimeExecutor {
    fn execute(
        &self,
        context: agent_orchestration::TaskExecutionContext,
    ) -> LocalBoxFuture<'static, Result<agent_orchestration::TaskExecutionOutput>> {
        let env = self.environment.clone();
        let app = self.app.clone();
        let event_stream = self.event_stream.clone();
        let reporter = context.reporter.clone();
        let task = context.task;
        let role = self.roles_map.read().get(&task.id).cloned().flatten();
        let existing_session = context.existing_session_id;

        Box::pin(async move {
            if context.cancellation_token.is_cancelled() || event_stream.was_cancelled_by_user() {
                anyhow::bail!("task execution cancelled");
            }

            let subagent = app.update(|cx| {
                let subagent = if let Some(session_id) = existing_session {
                    env.resume_subagent(session_id, cx)
                } else {
                    let tool_filter = task
                        .tools
                        .clone()
                        .map(|tools| tools.into_iter().map(SharedString::from).collect());
                    env.create_subagent(task.label.clone(), role, tool_filter, cx)
                }?;
                event_stream.subagent_spawned(subagent.id());
                anyhow::Ok(subagent)
            })?;

            let session_id = subagent.id();
            if let Err(exceeded) = reporter.report_tool_call_started("subagent") {
                anyhow::bail!("{exceeded}");
            }
            let usage_before = app.update(|cx| subagent.cumulative_token_usage(cx));
            let send_result = subagent.send(task_execution_prompt(&task), &app).await;
            reporter.report_tool_call_finished();

            let usage_after = app.update(|cx| subagent.cumulative_token_usage(cx));
            let tokens_used = cumulative_token_delta(usage_before, usage_after);
            let output = match send_result {
                Ok(output) => output,
                Err(error) => {
                    reporter.report_tokens(tokens_used)?;
                    return Err(error);
                }
            };

            let artifact = agent_orchestration::Artifact::new(
                task.id.clone(),
                format!("Output for {}", task.label),
                agent_orchestration::ArtifactKind::Text,
                output.clone(),
            );

            Ok(agent_orchestration::TaskExecutionOutput {
                session_id: Some(session_id),
                output,
                tokens_used: Some(tokens_used),
                artifacts: vec![artifact],
            })
        })
    }

    fn verify(
        &self,
        task: &agent_orchestration::OrchestrationTask,
        output: &agent_orchestration::TaskExecutionOutput,
    ) -> LocalBoxFuture<'static, Result<agent_orchestration::VerificationResult>> {
        let task = task.clone();
        let output = output.clone();
        Box::pin(async move { Ok(verify_task_output(&task, &output.output)) })
    }

    fn repair(
        &self,
        task: &agent_orchestration::OrchestrationTask,
        feedback: &str,
        context: agent_orchestration::TaskExecutionContext,
    ) -> LocalBoxFuture<'static, Result<agent_orchestration::TaskExecutionOutput>> {
        let env = self.environment.clone();
        let app = self.app.clone();
        let reporter = context.reporter.clone();
        let task = task.clone();
        let feedback = feedback.to_string();
        let session_id = context.existing_session_id;

        Box::pin(async move {
            let Some(session_id) = session_id else {
                anyhow::bail!("cannot repair without an existing session");
            };

            let subagent = app.update({
                let session_id = session_id.clone();
                move |cx| env.resume_subagent(session_id, cx)
            })?;
            let repair_message = format!(
                "{}\n\nVerification feedback for task `{}`:\n\n{}\n\nPlease address the feedback and return a corrected result with a new structured verification claim.",
                task_execution_prompt(&task),
                task.label,
                feedback
            );
            if let Err(exceeded) = reporter.report_tool_call_started("subagent") {
                anyhow::bail!("{exceeded}");
            }
            let usage_before = app.update(|cx| subagent.cumulative_token_usage(cx));
            let send_result = subagent.send(repair_message, &app).await;
            reporter.report_tool_call_finished();
            let usage_after = app.update(|cx| subagent.cumulative_token_usage(cx));
            let tokens_used = cumulative_token_delta(usage_before, usage_after);
            let output = match send_result {
                Ok(output) => output,
                Err(error) => {
                    reporter.report_tokens(tokens_used)?;
                    return Err(error);
                }
            };
            let artifact = agent_orchestration::Artifact::new(
                task.id.clone(),
                format!("Repaired output for {}", task.label),
                agent_orchestration::ArtifactKind::Text,
                output.clone(),
            );

            Ok(agent_orchestration::TaskExecutionOutput {
                session_id: Some(session_id),
                output,
                tokens_used: Some(tokens_used),
                artifacts: vec![artifact],
            })
        })
    }
}

const MAX_PARALLEL_SUBAGENTS: usize = 4;
const MAX_SUBAGENT_RETRIES: u8 = 2;

fn batch_launch_policy(
    configured_strategy: agent_settings::AgentExecutionStrategy,
    resolved_turn_policy: Option<&agent_orchestration::ResolvedTurnPolicy>,
    fallback_prompt: &str,
    task_count: usize,
    autonomy: agent_settings::AgentAutonomy,
) -> (
    agent_settings::AgentExecutionPolicy,
    agent_orchestration::RuntimeLaunchDisposition,
) {
    let strategy = if configured_strategy == agent_settings::AgentExecutionStrategy::Auto {
        resolved_turn_policy
            .map(|policy| policy.strategy)
            .unwrap_or_else(|| {
                agent_orchestration::OrchestrationPlanner::resolve_auto(
                    fallback_prompt,
                    agent_orchestration::AutoPolicyContext {
                        work_item_count: task_count,
                        available_tool_count: None,
                        can_orchestrate: true,
                    },
                )
                .strategy
            })
    } else {
        configured_strategy
    };

    let disposition = match strategy {
        agent_settings::AgentExecutionStrategy::Direct => {
            agent_orchestration::RuntimeLaunchDisposition::Approved
        }
        agent_settings::AgentExecutionStrategy::Plan => {
            agent_orchestration::RuntimeLaunchDisposition::AwaitApproval
        }
        agent_settings::AgentExecutionStrategy::Orchestrate
            if configured_strategy == agent_settings::AgentExecutionStrategy::Auto =>
        {
            agent_orchestration::RuntimeLaunchDisposition::AwaitApproval
        }
        agent_settings::AgentExecutionStrategy::Orchestrate => match autonomy {
            agent_settings::AgentAutonomy::Autonomous => {
                agent_orchestration::RuntimeLaunchDisposition::Approved
            }
            agent_settings::AgentAutonomy::Manual | agent_settings::AgentAutonomy::Supervised => {
                agent_orchestration::RuntimeLaunchDisposition::AwaitApproval
            }
        },
        agent_settings::AgentExecutionStrategy::Auto => {
            agent_orchestration::RuntimeLaunchDisposition::Approved
        }
    };

    (
        agent_settings::AgentExecutionPolicy { strategy, autonomy },
        disposition,
    )
}

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
/// ### Budgeting batch tasks
/// - Omit `token_budget` unless the task needs a strict total-usage ceiling.
/// - `token_budget` includes all provider-reported input, output, and cache tokens across every attempt; it is commonly much larger than the desired response length.
/// - `tool_call_budget` counts executor-visible delegation calls. It does not currently count tools invoked inside a native subagent session.
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
    /// Optional hard budget for cumulative provider-reported tokens across all
    /// task attempts, including input, output, and cache tokens.
    #[serde(default)]
    pub token_budget: Option<u64>,
    /// Criteria required for this task's output to be considered verified.
    #[serde(default)]
    pub acceptance_criteria: Option<Vec<String>>,
    /// Expected format or structure of the task output.
    #[serde(default)]
    pub expected_output: Option<String>,
    /// Whether this task requires citation evidence in its output.
    #[serde(default)]
    pub evidence_required: Option<bool>,
    /// Execution timeout in seconds for this task.
    #[serde(default)]
    pub time_budget_secs: Option<u64>,
    /// Optional hard budget for executor-visible tool calls. Native subagent
    /// execution currently counts each delegated subagent turn as one call.
    #[serde(default)]
    pub tool_call_budget: Option<u64>,
    /// Stated high-level objective for this task.
    #[serde(default)]
    pub objective: Option<String>,
    /// Bounded write scope / affected file patterns.
    #[serde(default)]
    pub scope: Option<String>,
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
    PlanProposed {
        run_id: String,
        plan: Box<agent_orchestration::OrchestrationPlan>,
        strategy: agent_settings::AgentExecutionStrategy,
        reason: String,
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
            SpawnAgentToolOutput::PlanProposed {
                run_id,
                plan,
                strategy,
                reason,
            } => serde_json::to_string(&serde_json::json!({
                "run_id": run_id,
                "status": "awaiting_approval",
                "plan": plan,
                "strategy": strategy,
                "reason": reason,
            }))
            .unwrap_or_else(|e| format!("Failed to serialize spawn_agent output: {e}"))
            .into(),
        }
    }
}

/// Tool that spawns an agent thread to work on a task.
pub struct SpawnAgentTool {
    environment: Rc<dyn ThreadEnvironment>,
    thread: gpui::WeakEntity<Thread>,
}

impl SpawnAgentTool {
    pub fn new(environment: Rc<dyn ThreadEnvironment>, thread: gpui::WeakEntity<Thread>) -> Self {
        Self {
            environment,
            thread,
        }
    }

    /// Resumes a previously persisted orchestration run that was interrupted.
    /// The scheduler starts immediately with an `Interrupted` state; unlike a
    /// fresh proposal, no approval is required.
    pub(crate) async fn resume_orchestration_run(
        &self,
        persisted: agent_orchestration::PersistedRun,
        event_stream: ToolCallEventStream,
        cx: &mut gpui::AsyncApp,
    ) -> Result<agent_orchestration::RunHandle> {
        let mut runtime_config = agent_orchestration::RuntimeConfig::default();
        runtime_config.scheduler.max_parallel_tasks = MAX_PARALLEL_SUBAGENTS;
        runtime_config.foreground_executor = Some(cx.foreground_executor().clone());
        runtime_config.scheduler.background_executor = Some(cx.background_executor().clone());

        let executor = Rc::new(SubagentRuntimeExecutor::new(
            self.environment.clone(),
            cx.clone(),
            event_stream,
            Arc::new(parking_lot::RwLock::new(collections::HashMap::default())),
        ));

        let (run_handle, completion_rx) =
            agent_orchestration::OrchestrationRuntime::resume(persisted, executor, runtime_config)?;

        cx.update(|cx| {
            if let Some(parent) = self.thread.upgrade() {
                parent.update(cx, |parent, cx| {
                    parent.set_orchestration_run(run_handle.clone(), cx);
                });
            }
        });

        let runtime_events = run_handle.subscribe_live();
        cx.foreground_executor()
            .spawn({
                let app = cx.clone();
                let thread = self.thread.clone();
                async move {
                    while let Ok(event) = runtime_events.receiver.recv().await {
                        app.update(|cx| {
                            if let Some(parent) = thread.upgrade() {
                                parent.update(cx, |_, cx| cx.notify());
                            }
                        });
                        if matches!(
                            event.event,
                            agent_orchestration::RuntimeEvent::RunStateChanged { state, .. }
                                if state.is_terminal()
                        ) {
                            break;
                        }
                    }
                }
            })
            .detach();

        let completion_handle = run_handle.clone();
        cx.foreground_executor()
            .spawn({
                let app = cx.clone();
                let thread = self.thread.clone();
                async move {
                    let _ = completion_rx.await;
                    let snapshot = completion_handle.snapshot();
                    app.update(|cx| {
                        if let Some(parent) = thread.upgrade() {
                            parent.update(cx, |parent, cx| {
                                parent.set_persisted_orchestration_run(Some(snapshot), cx);
                            });
                        }
                    });
                }
            })
            .detach();

        Ok(run_handle)
    }
}

async fn run_batch_tasks(
    environment: Rc<dyn ThreadEnvironment>,
    thread: gpui::WeakEntity<Thread>,
    tasks: Vec<SpawnAgentTask>,
    event_stream: ToolCallEventStream,
    cx: &mut gpui::AsyncApp,
) -> Result<SpawnAgentToolOutput, SpawnAgentToolOutput> {
    if tasks.is_empty() {
        return Err(SpawnAgentToolOutput::Error {
            session_id: None,
            error: "tasks must contain at least one subagent task".to_string(),
            session_info: None,
        });
    }
    let pending = tasks
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

    let (configured_strategy, resolved_turn_policy, autonomy) = cx
        .update(|cx| {
            thread.upgrade().map(|t| {
                let t = t.read(cx);
                (
                    t.execution_strategy(),
                    t.resolved_turn_policy().cloned(),
                    t.autonomy(),
                )
            })
        })
        .unwrap_or((
            agent_settings::AgentExecutionStrategy::Direct,
            None,
            agent_settings::AgentAutonomy::default(),
        ));

    let prompt = pending
        .iter()
        .map(|(task_id, task)| {
            let mut message = task.message.clone();
            if !task.depends_on.is_empty() {
                message.push_str(&format!("\n[depends on: {}]", task.depends_on.join(", ")));
            }
            format!("[{task_id}] {message}")
        })
        .collect::<Vec<_>>()
        .join("\n");

    let (policy, disposition) = batch_launch_policy(
        configured_strategy,
        resolved_turn_policy.as_ref(),
        &prompt,
        pending.len(),
        autonomy,
    );

    let mut roles_map = collections::HashMap::default();
    let orchestration_tasks = pending
        .iter()
        .map(|(task_id, task)| {
            let id = agent_orchestration::TaskId::new(task_id);
            roles_map.insert(id.clone(), task.agent_type);
            let mut orchestration_task = agent_orchestration::OrchestrationTask::new(
                id,
                task.label.clone(),
                task.message.clone(),
            );
            orchestration_task.tools = task.tools.clone();
            orchestration_task.depends_on = task
                .depends_on
                .iter()
                .map(agent_orchestration::TaskId::new)
                .collect();
            orchestration_task.max_retries = Some(task.max_retries);
            orchestration_task.token_budget = task.token_budget;
            if let Some(criteria) = &task.acceptance_criteria {
                orchestration_task.acceptance_criteria = criteria.clone();
            }
            orchestration_task.expected_output = task.expected_output.clone();
            orchestration_task.evidence_required = task.evidence_required.unwrap_or(false);
            orchestration_task.time_budget_secs = task.time_budget_secs;
            orchestration_task.tool_call_budget = task.tool_call_budget;
            orchestration_task.objective = task.objective.clone();
            orchestration_task.scope = task.scope.clone();
            orchestration_task
        })
        .collect();

    let plan =
        agent_orchestration::OrchestrationPlan::new("Parallel delegation", orchestration_tasks);

    let mut runtime_config = agent_orchestration::RuntimeConfig::default();
    runtime_config.scheduler.max_parallel_tasks = MAX_PARALLEL_SUBAGENTS;
    runtime_config.foreground_executor = Some(cx.foreground_executor().clone());
    runtime_config.scheduler.background_executor = Some(cx.background_executor().clone());

    let executor = Rc::new(SubagentRuntimeExecutor::new(
        environment.clone(),
        cx.clone(),
        event_stream.clone(),
        Arc::new(parking_lot::RwLock::new(roles_map)),
    ));

    let (run_handle, completion_rx) =
        agent_orchestration::OrchestrationRuntime::start_with_disposition(
            plan,
            policy,
            disposition,
            executor,
            runtime_config,
        )
        .map_err(|error| SpawnAgentToolOutput::Error {
            session_id: None,
            error: error.to_string(),
            session_info: None,
        })?;

    cx.update(|cx| {
        if let Some(parent) = thread.upgrade() {
            parent.update(cx, |parent, cx| {
                parent.set_orchestration_run(run_handle.clone(), cx);
            });
        }
    });

    let runtime_events = run_handle.subscribe_live();
    cx.foreground_executor()
        .spawn({
            let app = cx.clone();
            let thread = thread.clone();
            async move {
                while let Ok(event) = runtime_events.receiver.recv().await {
                    app.update(|cx| {
                        if let Some(parent) = thread.upgrade() {
                            parent.update(cx, |_, cx| cx.notify());
                        }
                    });
                    if matches!(
                        event.event,
                        agent_orchestration::RuntimeEvent::RunStateChanged { state, .. }
                            if state.is_terminal()
                    ) {
                        break;
                    }
                }
            }
        })
        .detach();

    if disposition == agent_orchestration::RuntimeLaunchDisposition::AwaitApproval {
        // Keep the tool future alive while the detached runtime waits for the
        // approval action so the parent turn receives the real batch result.
        let proposal = run_handle.snapshot();
        cx.update(|cx| {
            if let Some(parent) = thread.upgrade() {
                parent.update(cx, |parent, cx| {
                    parent.set_persisted_orchestration_run(Some(proposal), cx);
                });
            }
        });
    }

    let completion_result = completion_rx.await;

    let snapshot = run_handle.snapshot();
    cx.update(|cx| {
        if let Some(parent) = thread.upgrade() {
            parent.update(cx, |parent, cx| {
                parent.set_persisted_orchestration_run(Some(snapshot), cx);
            });
        }
    });

    if event_stream.was_cancelled_by_user()
        || matches!(
            completion_result,
            Ok(Ok(agent_orchestration::RunState::Cancelled))
        )
    {
        run_handle.cancel(agent_orchestration::CancellationReason::UserRequested);
        return Err(SpawnAgentToolOutput::Error {
            session_id: None,
            error: "parallel orchestration cancelled".to_string(),
            session_info: None,
        });
    }
    if let Err(error) = completion_result {
        return Err(SpawnAgentToolOutput::Error {
            session_id: None,
            error: error.to_string(),
            session_info: None,
        });
    }
    if let Ok(Err(error)) = completion_result {
        return Err(SpawnAgentToolOutput::Error {
            session_id: None,
            error: error.to_string(),
            session_info: None,
        });
    }

    let mut batch = Vec::new();
    let mut last_error = None;
    let mut last_session_id = None;
    for (task_id, task) in pending {
        let id = agent_orchestration::TaskId::new(&task_id);
        let Some(status) = run_handle.task_status(&id) else {
            continue;
        };
        if status.state == agent_orchestration::TaskState::Completed {
            if let Some(session_id) = status.active_session_id {
                batch.push(SpawnAgentBatchResult {
                    task_id,
                    session_id,
                    label: task.label,
                    output: status.latest_output.unwrap_or_default(),
                });
            }
        } else if status.state == agent_orchestration::TaskState::Failed {
            last_error = status
                .latest_error
                .or(Some(format!("Task `{task_id}` failed")));
            last_session_id = status.active_session_id;
        }
    }
    if let Some(error) = last_error {
        let completed_summary = batch
            .iter()
            .map(|result| result.task_id.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        return Err(SpawnAgentToolOutput::Error {
            session_id: last_session_id,
            error: format!("{error}; completed: [{completed_summary}]"),
            session_info: None,
        });
    }
    let raw_output = serde_json::to_value(&batch).map_err(|error| SpawnAgentToolOutput::Error {
        session_id: None,
        error: format!("Failed to serialize batch output: {error}"),
        session_info: None,
    })?;
    event_stream.update_fields(
        acp::ToolCallUpdateFields::new()
            .title("Parallel agents completed")
            .raw_output(raw_output),
    );
    Ok(SpawnAgentToolOutput::BatchSuccess { results: batch })
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
                return run_batch_tasks(
                    self.environment.clone(),
                    self.thread.clone(),
                    tasks,
                    event_stream,
                    cx,
                )
                .await;
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
            SpawnAgentToolOutput::PlanProposed {
                run_id,
                plan,
                strategy,
                reason,
            } => (
                serde_json::to_string(&serde_json::json!({
                    "run_id": run_id,
                    "status": "awaiting_approval",
                    "plan": plan,
                    "strategy": strategy,
                    "reason": reason,
                }))
                .unwrap_or_else(|error| format!("Failed to serialize plan proposal: {error}"))
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
    fn cumulative_token_delta_uses_all_provider_reported_tokens() {
        let before = language_model::TokenUsage {
            input_tokens: 40_000,
            output_tokens: 1_000,
            cache_creation_input_tokens: 2_000,
            cache_read_input_tokens: 80_000,
        };
        let after = language_model::TokenUsage {
            input_tokens: 90_000,
            output_tokens: 4_500,
            cache_creation_input_tokens: 5_000,
            cache_read_input_tokens: 200_000,
        };

        assert_eq!(cumulative_token_delta(Some(before), Some(after)), 176_500);
        assert_eq!(cumulative_token_delta(None, Some(after)), 299_500);
        assert_eq!(cumulative_token_delta(Some(after), None), 0);
    }

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
    fn batch_uses_the_pre_turn_auto_decision() {
        let decision = agent_orchestration::AutoPolicyDecision {
            strategy: agent_settings::AgentExecutionStrategy::Direct,
            confidence: 0.9,
            reason: "pre-turn route".to_string(),
            heuristics: collections::HashMap::default(),
        };
        let resolved_turn_policy = agent_orchestration::ResolvedTurnPolicy::automatic(decision);
        let (policy, disposition) = batch_launch_policy(
            agent_settings::AgentExecutionStrategy::Auto,
            Some(&resolved_turn_policy),
            "Use subagents in parallel for many tasks",
            8,
            agent_settings::AgentAutonomy::Autonomous,
        );

        assert_eq!(
            policy.strategy,
            agent_settings::AgentExecutionStrategy::Direct
        );
        assert_eq!(
            disposition,
            agent_orchestration::RuntimeLaunchDisposition::Approved
        );
    }

    #[test]
    fn auto_orchestration_route_keeps_the_approval_checkpoint() {
        let decision = agent_orchestration::AutoPolicyDecision {
            strategy: agent_settings::AgentExecutionStrategy::Orchestrate,
            confidence: 0.9,
            reason: "pre-turn route".to_string(),
            heuristics: collections::HashMap::default(),
        };
        let resolved_turn_policy = agent_orchestration::ResolvedTurnPolicy::automatic(decision);
        let (policy, disposition) = batch_launch_policy(
            agent_settings::AgentExecutionStrategy::Auto,
            Some(&resolved_turn_policy),
            "A batch",
            4,
            agent_settings::AgentAutonomy::Autonomous,
        );

        assert_eq!(
            policy.strategy,
            agent_settings::AgentExecutionStrategy::Orchestrate
        );
        assert_eq!(
            disposition,
            agent_orchestration::RuntimeLaunchDisposition::AwaitApproval
        );
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
                    acceptance_criteria: None,
                    expected_output: None,
                    evidence_required: None,
                    time_budget_secs: None,
                    tool_call_budget: None,
                    objective: None,
                    scope: None,
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
                    acceptance_criteria: None,
                    expected_output: None,
                    evidence_required: None,
                    time_budget_secs: None,
                    tool_call_budget: None,
                    objective: None,
                    scope: None,
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
                acceptance_criteria: None,
                expected_output: None,
                evidence_required: None,
                time_budget_secs: None,
                tool_call_budget: None,
                objective: None,
                scope: None,
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
                acceptance_criteria: None,
                expected_output: None,
                evidence_required: None,
                time_budget_secs: None,
                tool_call_budget: None,
                objective: None,
                scope: None,
            },
        )];
        assert!(validate_task_graph(&tasks).is_err());
    }
    #[test]
    fn test_batch_orchestration_plan_graph_waves() {
        let t1 = SpawnAgentTask {
            id: Some("task-1".to_string()),
            label: "Search codebase".to_string(),
            message: "Find usages".to_string(),
            agent_type: Some(SubagentRole::Explorer),
            tools: None,
            depends_on: Vec::new(),
            max_retries: 1,
            token_budget: Some(1000),
            acceptance_criteria: None,
            expected_output: None,
            evidence_required: None,
            time_budget_secs: None,
            tool_call_budget: None,
            objective: None,
            scope: None,
        };
        let t2 = SpawnAgentTask {
            id: Some("task-2".to_string()),
            label: "Implement changes".to_string(),
            message: "Apply refactor".to_string(),
            agent_type: Some(SubagentRole::CodingWorker),
            tools: None,
            depends_on: vec!["task-1".to_string()],
            max_retries: 1,
            token_budget: Some(2000),
            acceptance_criteria: None,
            expected_output: None,
            evidence_required: None,
            time_budget_secs: None,
            tool_call_budget: None,
            objective: None,
            scope: None,
        };
        let pending = vec![("task-1".to_string(), t1), ("task-2".to_string(), t2)];
        assert!(validate_task_graph(&pending).is_ok());

        let orch_tasks = pending
            .into_iter()
            .map(|(task_id, task)| {
                let mut orch =
                    agent_orchestration::OrchestrationTask::new(task_id, task.label, task.message);
                orch.depends_on = task
                    .depends_on
                    .into_iter()
                    .map(agent_orchestration::TaskId::new)
                    .collect();
                orch
            })
            .collect();

        let plan = agent_orchestration::OrchestrationPlan::new("Test Plan", orch_tasks);
        let graph = agent_orchestration::PlanGraph::new(plan).expect("valid graph");
        assert_eq!(graph.waves().len(), 2);
        assert_eq!(
            graph.waves()[0],
            vec![agent_orchestration::TaskId::new("task-1")]
        );
        assert_eq!(
            graph.waves()[1],
            vec![agent_orchestration::TaskId::new("task-2")]
        );
    }

    #[test]
    fn structured_verification_requires_every_criterion_and_evidence() {
        let mut task = agent_orchestration::OrchestrationTask::new("task", "Task", "Do work");
        task.acceptance_criteria = vec!["tests pass".to_string(), "diff reviewed".to_string()];
        task.expected_output = Some("Summary with evidence".to_string());
        task.evidence_required = true;

        let incomplete = format!(
            "Done\n{VERIFICATION_START}{{\"criteria\":[{{\"criterion\":\"tests pass\",\"passed\":true,\"evidence\":\"cargo test passed\"}}],\"expected_output_satisfied\":true,\"citations\":[\"src/main.rs:12\"]}}{VERIFICATION_END}"
        );
        assert!(!verify_task_output(&task, &incomplete).passed);

        let complete = format!(
            "Done\n{VERIFICATION_START}{{\"criteria\":[{{\"criterion\":\"tests pass\",\"passed\":true,\"evidence\":\"cargo test passed\"}},{{\"criterion\":\"diff reviewed\",\"passed\":true,\"evidence\":\"reviewed src/main.rs\"}}],\"expected_output_satisfied\":true,\"citations\":[\"src/main.rs:12\"]}}{VERIFICATION_END}"
        );
        let result = verify_task_output(&task, &complete);
        assert!(result.passed);
        assert_eq!(
            result.verdict,
            agent_orchestration::VerificationVerdict::Claimed
        );
    }

    #[test]
    fn structured_verification_rejects_invalid_citations() {
        let mut task = agent_orchestration::OrchestrationTask::new("task", "Task", "Do work");
        task.evidence_required = true;
        let output = format!(
            "Done\n{VERIFICATION_START}{{\"criteria\":[],\"expected_output_satisfied\":true,\"citations\":[\"src/main.rs\"]}}{VERIFICATION_END}"
        );
        let result = verify_task_output(&task, &output);
        assert!(!result.passed);
        assert_eq!(result.citations_valid, Some(false));
    }
}
