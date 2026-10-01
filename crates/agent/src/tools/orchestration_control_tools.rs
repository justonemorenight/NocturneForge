use agent_client_protocol::schema::v1 as acp;
use agent_orchestration::{
    AgentMessageKind, AgentPath, GoalSnapshot, RunHandle, RunState, TaskState,
};
use anyhow::Result;
use gpui::{App, SharedString, Task, WeakEntity};
use language_model::LanguageModelToolResultContent;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;

use crate::{AgentTool, Thread, ToolCallEventStream, ToolInput};

#[derive(Clone)]
pub struct AgentControlToolConfig {
    pub default_wait: Duration,
    pub maximum_wait: Duration,
    pub maximum_wait_targets: usize,
}

impl Default for AgentControlToolConfig {
    fn default() -> Self {
        Self {
            default_wait: Duration::from_secs(30),
            maximum_wait: Duration::from_secs(120),
            maximum_wait_targets: 8,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
/// List the canonical agent tree and current orchestration status.
pub struct ListOrchestrationAgentsInput {
    /// Restrict the listing to one run. Omit to list every tracked run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OrchestrationAgentSummary {
    pub path: String,
    pub parent: Option<String>,
    pub task_id: Option<String>,
    pub role: Option<String>,
    pub target: Option<String>,
    pub state: Option<TaskState>,
    pub phase: Option<String>,
    pub current_attempt: Option<u32>,
    pub total_attempts: Option<u32>,
    pub tokens_used: Option<u64>,
    pub model: Option<String>,
    pub current_tool: Option<String>,
    pub session_id: Option<String>,
    pub latest_output: Option<String>,
    pub latest_error: Option<String>,
    pub verification_passed: Option<bool>,
    pub wait_reason: Option<String>,
    pub context_revision: Option<u64>,
    pub context_paths: usize,
    pub context_symbols: usize,
    pub context_artifacts: usize,
    pub queued_messages: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OrchestrationRunSummary {
    pub run_id: String,
    pub run_state: RunState,
    pub goal: Option<GoalSnapshot>,
    pub agents: Vec<OrchestrationAgentSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OrchestrationControlOutput {
    Success {
        run_id: String,
        run_state: RunState,
        goal: GoalSnapshot,
        agents: Vec<OrchestrationAgentSummary>,
    },
    Runs {
        runs: Vec<OrchestrationRunSummary>,
    },
    MessageQueued {
        run_id: String,
        sequence: u64,
        recipient: String,
        delivery: String,
    },
    GoalUpdated {
        run_id: Option<String>,
        goal: GoalSnapshot,
    },
    Error {
        error: String,
    },
}

impl From<OrchestrationControlOutput> for LanguageModelToolResultContent {
    fn from(output: OrchestrationControlOutput) -> Self {
        serde_json::to_string(&output)
            .unwrap_or_else(|error| format!("Failed to serialize agent control output: {error}"))
            .into()
    }
}

pub struct ListOrchestrationAgentsTool {
    thread: WeakEntity<Thread>,
}

impl ListOrchestrationAgentsTool {
    pub fn new(thread: WeakEntity<Thread>) -> Self {
        Self { thread }
    }
}

impl AgentTool for ListOrchestrationAgentsTool {
    type Input = ListOrchestrationAgentsInput;
    type Output = OrchestrationControlOutput;

    const NAME: &'static str = "list_orchestration_agents";

    fn kind() -> acp::ToolKind {
        acp::ToolKind::Other
    }

    fn initial_title(
        &self,
        _input: Result<Self::Input, serde_json::Value>,
        _cx: &mut App,
    ) -> SharedString {
        "List orchestration agents".into()
    }

    fn run(
        self: Arc<Self>,
        input: ToolInput<Self::Input>,
        _event_stream: ToolCallEventStream,
        cx: &mut App,
    ) -> Task<Result<Self::Output, Self::Output>> {
        let input = input.recv();
        // Tool dispatch holds the parent Thread's update lease until run returns.
        // Defer reading it to the foreground executor, as the other control tools do.
        cx.spawn(async move |cx| {
            let input = input
                .await
                .map_err(|error| OrchestrationControlOutput::Error {
                    error: error.to_string(),
                })?;
            let runs = self
                .thread
                .read_with(cx, |thread, _cx| {
                    let live_runs = thread
                        .orchestration_runs()
                        .iter()
                        .filter(|run| {
                            input
                                .run_id
                                .as_deref()
                                .is_none_or(|run_id| run.run_id().as_str() == run_id)
                        })
                        .cloned()
                        .collect::<Vec<_>>();
                    let persisted_runs = thread
                        .persisted_orchestration_runs()
                        .iter()
                        .filter(|run| {
                            input
                                .run_id
                                .as_deref()
                                .is_none_or(|run_id| run.run_id.as_str() == run_id)
                                && thread.orchestration_run_by_id(&run.run_id).is_none()
                        })
                        .cloned()
                        .collect::<Vec<_>>();
                    (live_runs, persisted_runs)
                })
                .map_err(|error| OrchestrationControlOutput::Error {
                    error: error.to_string(),
                })?;
            let (live_runs, persisted_runs) = runs;
            if live_runs.is_empty() && persisted_runs.is_empty() {
                return Err(OrchestrationControlOutput::Error {
                    error: input
                        .run_id
                        .map(|run_id| format!("orchestration run '{run_id}' was not found"))
                        .unwrap_or_else(|| "no orchestration runs are available".to_string()),
                });
            }
            let mut summaries = live_runs
                .iter()
                .map(run_summary)
                .collect::<Result<Vec<_>>>()
                .map_err(|error| OrchestrationControlOutput::Error {
                    error: error.to_string(),
                })?;
            summaries.extend(persisted_runs.iter().map(persisted_run_summary));
            if summaries.len() == 1
                && let Some(run) = live_runs.first()
            {
                return Ok(success_output(run, summaries.remove(0).agents));
            }
            Ok(OrchestrationControlOutput::Runs { runs: summaries })
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
/// Send context or a follow-up to an agent in the active orchestration run.
pub struct SendMessageToAgentInput {
    /// Required when multiple orchestration runs are active.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
    /// Canonical path or unambiguous agent name.
    pub recipient: String,
    pub message: String,
    /// When true, interrupt the active worker at a safe turn boundary and run this follow-up.
    #[serde(default)]
    pub interrupt: bool,
}

pub struct SendMessageToAgentTool {
    thread: WeakEntity<Thread>,
}

impl SendMessageToAgentTool {
    pub fn new(thread: WeakEntity<Thread>) -> Self {
        Self { thread }
    }
}

impl AgentTool for SendMessageToAgentTool {
    type Input = SendMessageToAgentInput;
    type Output = OrchestrationControlOutput;

    const NAME: &'static str = "send_message_to_agent";

    fn kind() -> acp::ToolKind {
        acp::ToolKind::Other
    }

    fn initial_title(
        &self,
        input: Result<Self::Input, serde_json::Value>,
        _cx: &mut App,
    ) -> SharedString {
        input
            .ok()
            .map(|input| format!("Message {}", input.recipient).into())
            .unwrap_or_else(|| "Message agent".into())
    }

    fn run(
        self: Arc<Self>,
        input: ToolInput<Self::Input>,
        _event_stream: ToolCallEventStream,
        cx: &mut App,
    ) -> Task<Result<Self::Output, Self::Output>> {
        let input = input.recv();
        let thread = self.thread.clone();
        cx.spawn(async move |cx| {
            let input = input
                .await
                .map_err(|error| OrchestrationControlOutput::Error {
                    error: error.to_string(),
                })?;
            let run = thread
                .read_with(cx, |thread, _cx| {
                    select_live_run(thread, input.run_id.as_deref())
                })
                .map_err(|error| OrchestrationControlOutput::Error {
                    error: error.to_string(),
                })?
                .map_err(|error| OrchestrationControlOutput::Error {
                    error: error.to_string(),
                })?;
            let recipient = run
                .agent_control_plane()
                .resolve(&AgentPath::root(), &input.recipient)
                .map_err(|error| OrchestrationControlOutput::Error {
                    error: error.to_string(),
                })?;
            if run
                .agent_control_plane()
                .identity(&recipient)
                .is_none_or(|identity| identity.task_id.is_none())
            {
                return Err(OrchestrationControlOutput::Error {
                    error: "recipient must identify a worker agent, not the orchestration root"
                        .to_string(),
                });
            }
            let kind = if input.interrupt {
                AgentMessageKind::FollowUp
            } else {
                AgentMessageKind::Message
            };
            let message = run
                .send_agent_message(AgentPath::root(), recipient.clone(), kind, input.message)
                .await
                .map_err(|error| OrchestrationControlOutput::Error {
                    error: error.to_string(),
                })?;
            Ok(OrchestrationControlOutput::MessageQueued {
                run_id: run.run_id().to_string(),
                sequence: message.sequence,
                recipient: recipient.to_string(),
                delivery: if input.interrupt {
                    "steered".to_string()
                } else {
                    "queued".to_string()
                },
            })
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
/// Update the parent orchestration goal by observing a blocker, clearing it, or completing the goal.
pub struct UpdateOrchestrationGoalInput {
    pub action: UpdateOrchestrationGoalAction,
    /// Required when updating one goal while multiple runs are active.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum UpdateOrchestrationGoalAction {
    ObserveBlocker,
    ClearBlocker,
    Complete,
}

pub struct UpdateOrchestrationGoalTool {
    thread: WeakEntity<Thread>,
}

impl UpdateOrchestrationGoalTool {
    pub fn new(thread: WeakEntity<Thread>) -> Self {
        Self { thread }
    }
}

impl AgentTool for UpdateOrchestrationGoalTool {
    type Input = UpdateOrchestrationGoalInput;
    type Output = OrchestrationControlOutput;

    const NAME: &'static str = "update_orchestration_goal";

    fn kind() -> acp::ToolKind {
        acp::ToolKind::Other
    }

    fn initial_title(
        &self,
        input: Result<Self::Input, serde_json::Value>,
        _cx: &mut App,
    ) -> SharedString {
        match input {
            Ok(UpdateOrchestrationGoalInput {
                action: UpdateOrchestrationGoalAction::ObserveBlocker,
                ..
            }) => "Record orchestration blocker".into(),
            Ok(UpdateOrchestrationGoalInput {
                action: UpdateOrchestrationGoalAction::ClearBlocker,
                ..
            }) => "Clear orchestration blocker".into(),
            Ok(UpdateOrchestrationGoalInput {
                action: UpdateOrchestrationGoalAction::Complete,
                ..
            }) => "Complete orchestration goal".into(),
            Err(_) => "Update orchestration goal".into(),
        }
    }

    fn run(
        self: Arc<Self>,
        input: ToolInput<Self::Input>,
        _event_stream: ToolCallEventStream,
        cx: &mut App,
    ) -> Task<Result<Self::Output, Self::Output>> {
        let input = input.recv();
        let thread = self.thread.clone();
        cx.spawn(async move |cx| {
            let input = input
                .await
                .map_err(|error| OrchestrationControlOutput::Error {
                    error: error.to_string(),
                })?;
            let target = thread
                .read_with(cx, |thread, _cx| {
                    select_goal_target(thread, input.run_id.as_deref())
                })
                .map_err(|error| OrchestrationControlOutput::Error {
                    error: error.to_string(),
                })?
                .map_err(|error| OrchestrationControlOutput::Error {
                    error: error.to_string(),
                })?;
            let run_id = match &target {
                OrchestrationGoalTarget::Live(run) => Some(run.run_id().to_string()),
                OrchestrationGoalTarget::Persisted(run) => Some(run.run_id.to_string()),
                OrchestrationGoalTarget::Parent(_) => None,
            };
            if input.action == UpdateOrchestrationGoalAction::Complete {
                match &target {
                    OrchestrationGoalTarget::Live(run) => {
                        ensure_goal_run_complete(run.state(), &run.task_statuses()).map_err(
                            |error| OrchestrationControlOutput::Error {
                                error: error.to_string(),
                            },
                        )?;
                    }
                    OrchestrationGoalTarget::Persisted(run) => {
                        ensure_goal_run_complete(run.state, &run.task_statuses).map_err(
                            |error| OrchestrationControlOutput::Error {
                                error: error.to_string(),
                            },
                        )?;
                    }
                    OrchestrationGoalTarget::Parent(_) => {}
                }
                let has_running_children = thread
                    .read_with(cx, |thread, _cx| thread.has_running_subagents())
                    .map_err(|error| OrchestrationControlOutput::Error {
                        error: error.to_string(),
                    })?;
                if has_running_children {
                    return Err(OrchestrationControlOutput::Error {
                        error:
                            "cannot complete parent goal while child subagents are still running"
                                .to_string(),
                    });
                }
            }
            let (goal_controller, persisted_run_id) = match target {
                OrchestrationGoalTarget::Live(run) => (run.goal_controller(), None),
                OrchestrationGoalTarget::Persisted(run) => {
                    let goal = run.goal.ok_or_else(|| OrchestrationControlOutput::Error {
                        error: format!("orchestration run '{}' has no persisted goal", run.run_id),
                    })?;
                    let controller = agent_orchestration::GoalController::restore(
                        goal,
                        agent_orchestration::GoalControllerConfig::default(),
                    )
                    .map_err(|error| OrchestrationControlOutput::Error {
                        error: error.to_string(),
                    })?;
                    (controller, Some(run.run_id))
                }
                OrchestrationGoalTarget::Parent(goal) => (goal, None),
            };
            let goal = apply_goal_action(&goal_controller, input.action, input.code, input.detail)
                .map_err(|error| OrchestrationControlOutput::Error {
                    error: error.to_string(),
                })?;
            if let Some(persisted_run_id) = persisted_run_id {
                let updated = thread
                    .update(cx, |thread, cx| {
                        thread.update_persisted_orchestration_goal(
                            &persisted_run_id,
                            goal.clone(),
                            cx,
                        )
                    })
                    .map_err(|error| OrchestrationControlOutput::Error {
                        error: error.to_string(),
                    })?;
                if !updated {
                    return Err(OrchestrationControlOutput::Error {
                        error: format!(
                            "orchestration run '{persisted_run_id}' is no longer available"
                        ),
                    });
                }
            }
            Ok(OrchestrationControlOutput::GoalUpdated { run_id, goal })
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
/// Wait until one of the selected orchestration agents changes state or needs attention.
pub struct WaitForAgentsInput {
    /// Required when multiple orchestration runs are active.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
    /// Canonical paths or unambiguous agent names. Empty waits for every child agent.
    #[serde(default)]
    pub agents: Vec<String>,
    /// Maximum wait in milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,
}

pub struct WaitForAgentsTool {
    thread: WeakEntity<Thread>,
    config: AgentControlToolConfig,
}

impl WaitForAgentsTool {
    pub fn new(thread: WeakEntity<Thread>) -> Self {
        Self {
            thread,
            config: AgentControlToolConfig::default(),
        }
    }
}

impl AgentTool for WaitForAgentsTool {
    type Input = WaitForAgentsInput;
    type Output = OrchestrationControlOutput;

    const NAME: &'static str = "wait_for_agents";

    fn kind() -> acp::ToolKind {
        acp::ToolKind::Other
    }

    fn initial_title(
        &self,
        _input: Result<Self::Input, serde_json::Value>,
        _cx: &mut App,
    ) -> SharedString {
        "Wait for agents".into()
    }

    fn run(
        self: Arc<Self>,
        input: ToolInput<Self::Input>,
        _event_stream: ToolCallEventStream,
        cx: &mut App,
    ) -> Task<Result<Self::Output, Self::Output>> {
        let input = input.recv();
        let thread = self.thread.clone();
        let config = self.config.clone();
        cx.spawn(async move |cx| {
            let input = input
                .await
                .map_err(|error| OrchestrationControlOutput::Error {
                    error: error.to_string(),
                })?;
            if input.agents.len() > config.maximum_wait_targets {
                return Err(OrchestrationControlOutput::Error {
                    error: format!(
                        "cannot wait for more than {} agents",
                        config.maximum_wait_targets
                    ),
                });
            }
            let run = thread
                .read_with(cx, |thread, _cx| {
                    select_live_run(thread, input.run_id.as_deref())
                })
                .map_err(|error| OrchestrationControlOutput::Error {
                    error: error.to_string(),
                })?
                .map_err(|error| OrchestrationControlOutput::Error {
                    error: error.to_string(),
                })?;
            if run.state() == agent_orchestration::RunState::Proposed {
                return Err(OrchestrationControlOutput::Error {
                    error: "orchestration run is awaiting approval; do not wait before approving the native plan".to_string(),
                });
            }
            if run.state().is_terminal() {
                return Err(OrchestrationControlOutput::Error {
                    error: format!(
                        "orchestration run is already terminal ({:?}); inspect its results instead of waiting",
                        run.state()
                    ),
                });
            }
            let paths = resolve_paths(&run, &input.agents).map_err(|error| {
                OrchestrationControlOutput::Error {
                    error: error.to_string(),
                }
            })?;
            if paths.is_empty() {
                return Err(OrchestrationControlOutput::Error {
                    error: "no worker agents are registered in the active run".to_string(),
                });
            }
            let subscription = run.subscribe_live();
            let before = selected_summaries(&run, &paths).map_err(|error| {
                OrchestrationControlOutput::Error {
                    error: error.to_string(),
                }
            })?;
            if before.iter().any(agent_needs_attention) {
                return Ok(success_output(&run, before));
            }

            let default_wait_ms =
                u64::try_from(config.default_wait.as_millis()).unwrap_or(u64::MAX);
            let requested = Duration::from_millis(input.timeout_ms.unwrap_or(default_wait_ms));
            let timeout = requested.min(config.maximum_wait);
            let timer = cx.background_executor().timer(timeout);
            futures::pin_mut!(timer);
            loop {
                let event = subscription.receiver.recv();
                futures::pin_mut!(event);
                match futures::future::select(event, timer.as_mut()).await {
                    futures::future::Either::Left((Ok(_), _)) => {
                        let current = selected_summaries(&run, &paths).map_err(|error| {
                            OrchestrationControlOutput::Error {
                                error: error.to_string(),
                            }
                        })?;
                        if current != before || current.iter().any(agent_needs_attention) {
                            return Ok(success_output(&run, current));
                        }
                    }
                    futures::future::Either::Left((Err(error), _)) => {
                        return Err(OrchestrationControlOutput::Error {
                            error: format!("agent event stream closed: {error}"),
                        });
                    }
                    futures::future::Either::Right(((), _)) => {
                        let agents = selected_summaries(&run, &paths).map_err(|error| {
                            OrchestrationControlOutput::Error {
                                error: error.to_string(),
                            }
                        })?;
                        return Ok(success_output(&run, agents));
                    }
                }
            }
        })
    }
}

fn success_output(
    run: &RunHandle,
    agents: Vec<OrchestrationAgentSummary>,
) -> OrchestrationControlOutput {
    OrchestrationControlOutput::Success {
        run_id: run.run_id().to_string(),
        run_state: run.state(),
        goal: run.goal_snapshot(),
        agents,
    }
}

fn run_summary(run: &RunHandle) -> Result<OrchestrationRunSummary> {
    Ok(OrchestrationRunSummary {
        run_id: run.run_id().to_string(),
        run_state: run.state(),
        goal: Some(run.goal_snapshot()),
        agents: summaries(run)?,
    })
}

fn persisted_run_summary(run: &agent_orchestration::PersistedRun) -> OrchestrationRunSummary {
    let agents = run
        .plan
        .tasks
        .iter()
        .map(|task| {
            let identity = run.agent_control_plane.as_ref().and_then(|control_plane| {
                control_plane
                    .identities
                    .iter()
                    .find(|identity| identity.task_id.as_ref() == Some(&task.id))
            });
            let status = run
                .task_statuses
                .iter()
                .find(|status| &status.task_id == &task.id);
            let queued_messages = run
                .agent_control_plane
                .as_ref()
                .and_then(|control_plane| {
                    control_plane.mailboxes.iter().find(|mailbox| {
                        Some(&mailbox.recipient) == identity.map(|identity| &identity.path)
                    })
                })
                .map_or(0, |mailbox| mailbox.messages.len());
            OrchestrationAgentSummary {
                path: identity
                    .map_or_else(|| task.id.to_string(), |identity| identity.path.to_string()),
                parent: identity
                    .and_then(|identity| identity.parent.as_ref().map(ToString::to_string)),
                task_id: Some(task.id.to_string()),
                role: identity
                    .and_then(|identity| identity.role.clone())
                    .or_else(|| task.role.clone())
                    .or_else(|| task.native_role.clone()),
                target: identity
                    .and_then(|identity| identity.target.as_ref().map(ToString::to_string))
                    .or_else(|| Some(task.target.to_string())),
                state: status.map(|status| status.state),
                phase: status.and_then(|status| status.phase.clone()),
                current_attempt: status.map(|status| status.current_attempt),
                total_attempts: status.map(|status| status.total_attempts),
                tokens_used: status.map(|status| status.tokens_used),
                model: status.and_then(|status| status.model.clone()),
                current_tool: status.and_then(|status| status.current_tool.clone()),
                session_id: status
                    .and_then(|status| status.active_session_id.as_ref())
                    .map(ToString::to_string),
                latest_output: status.and_then(|status| status.latest_output.clone()),
                latest_error: status.and_then(|status| status.latest_error.clone()),
                verification_passed: status
                    .and_then(|status| status.latest_verification.as_ref())
                    .map(|verification| verification.passed),
                wait_reason: status
                    .and_then(|status| status.wait_reason.as_ref())
                    .map(|reason| reason.description()),
                context_revision: None,
                context_paths: 0,
                context_symbols: 0,
                context_artifacts: 0,
                queued_messages,
            }
        })
        .collect();
    OrchestrationRunSummary {
        run_id: run.run_id.to_string(),
        run_state: run.state,
        goal: run.goal.clone(),
        agents,
    }
}

fn select_optional_live_run(
    thread: &Thread,
    requested_run_id: Option<&str>,
) -> Result<Option<RunHandle>> {
    if let Some(requested_run_id) = requested_run_id {
        let run_id = agent_orchestration::RunId::from_string(requested_run_id);
        return thread
            .orchestration_run_by_id(&run_id)
            .cloned()
            .map(Some)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "orchestration run '{requested_run_id}' is not active in this thread; resume it first if it was interrupted"
                )
            });
    }

    let active_runs = thread
        .orchestration_runs()
        .iter()
        .filter(|run| !run.state().is_terminal())
        .collect::<Vec<_>>();
    match active_runs.as_slice() {
        [run] => Ok(Some((*run).clone())),
        [] => match thread.orchestration_runs() {
            [run] => Ok(Some(run.clone())),
            [] => Ok(None),
            _ => {
                anyhow::bail!("run_id is required when this thread has multiple orchestration runs")
            }
        },
        _ => anyhow::bail!("run_id is required when multiple orchestration runs are active"),
    }
}

enum OrchestrationGoalTarget {
    Live(RunHandle),
    Persisted(agent_orchestration::PersistedRun),
    Parent(agent_orchestration::GoalController),
}

fn select_goal_target(
    thread: &Thread,
    requested_run_id: Option<&str>,
) -> Result<OrchestrationGoalTarget> {
    if let Some(requested_run_id) = requested_run_id {
        let run_id = agent_orchestration::RunId::from_string(requested_run_id);
        if let Some(run) = thread.orchestration_run_by_id(&run_id) {
            return Ok(OrchestrationGoalTarget::Live(run.clone()));
        }
        if let Some(run) = thread
            .persisted_orchestration_runs()
            .iter()
            .find(|run| run.run_id == run_id)
        {
            return Ok(OrchestrationGoalTarget::Persisted(run.clone()));
        }
        anyhow::bail!("orchestration run '{requested_run_id}' was not found in this thread");
    }

    let active_runs = thread
        .orchestration_runs()
        .iter()
        .filter(|run| !run.state().is_terminal())
        .collect::<Vec<_>>();
    match active_runs.as_slice() {
        [run] => return Ok(OrchestrationGoalTarget::Live((*run).clone())),
        [_, _, ..] => {
            anyhow::bail!("run_id is required when multiple orchestration runs are active")
        }
        [] => {}
    }

    let persisted_runs = thread
        .persisted_orchestration_runs()
        .iter()
        .filter(|run| thread.orchestration_run_by_id(&run.run_id).is_none())
        .collect::<Vec<_>>();
    match persisted_runs.as_slice() {
        [run] => return Ok(OrchestrationGoalTarget::Persisted((*run).clone())),
        [_, _, ..] => {
            anyhow::bail!("run_id is required when multiple orchestration runs are tracked")
        }
        [] => {}
    }

    match thread.orchestration_runs() {
        [run] => Ok(OrchestrationGoalTarget::Live(run.clone())),
        [_, _, ..] => {
            anyhow::bail!("run_id is required when multiple orchestration runs are tracked")
        }
        [] => thread
            .parent_orchestration_goal()
            .map(OrchestrationGoalTarget::Parent)
            .ok_or_else(|| anyhow::anyhow!("no orchestration goal is active")),
    }
}

fn ensure_goal_run_complete(
    run_state: RunState,
    task_statuses: &[agent_orchestration::TaskStatus],
) -> Result<()> {
    anyhow::ensure!(
        run_state.is_terminal(),
        "the orchestration run must reach a terminal state before its goal can be completed"
    );
    let has_active_tasks = task_statuses.iter().any(|status| {
        status.state.is_active() || matches!(status.state, TaskState::Queued | TaskState::Starting)
    });
    anyhow::ensure!(
        !has_active_tasks,
        "cannot complete orchestration goal while tasks are queued, starting, or running"
    );
    Ok(())
}

fn apply_goal_action(
    goal: &agent_orchestration::GoalController,
    action: UpdateOrchestrationGoalAction,
    code: Option<String>,
    detail: Option<String>,
) -> Result<GoalSnapshot> {
    match action {
        UpdateOrchestrationGoalAction::ObserveBlocker => goal.record_blocker(
            code.ok_or_else(|| anyhow::anyhow!("code is required when observing a blocker"))?,
            detail.ok_or_else(|| anyhow::anyhow!("detail is required when observing a blocker"))?,
        ),
        UpdateOrchestrationGoalAction::ClearBlocker => Ok(goal.clear_blocker()),
        UpdateOrchestrationGoalAction::Complete => goal.mark_achieved(),
    }
}

fn select_live_run(thread: &Thread, requested_run_id: Option<&str>) -> Result<RunHandle> {
    select_optional_live_run(thread, requested_run_id)?
        .ok_or_else(|| anyhow::anyhow!("no orchestration run is active"))
}

fn summaries(run: &RunHandle) -> Result<Vec<OrchestrationAgentSummary>> {
    let paths = run
        .agent_control_plane()
        .list(None)
        .into_iter()
        .map(|identity| identity.path)
        .collect::<Vec<_>>();
    selected_summaries(run, &paths)
}

fn resolve_paths(run: &RunHandle, references: &[String]) -> Result<Vec<AgentPath>> {
    if references.is_empty() {
        return Ok(run
            .agent_control_plane()
            .list(None)
            .into_iter()
            .filter_map(|identity| identity.task_id.map(|_| identity.path))
            .collect());
    }
    references
        .iter()
        .map(|reference| {
            let path = run
                .agent_control_plane()
                .resolve(&AgentPath::root(), reference)?;
            let identity = run
                .agent_control_plane()
                .identity(&path)
                .ok_or_else(|| anyhow::anyhow!("agent '{path}' is no longer registered"))?;
            anyhow::ensure!(
                identity.task_id.is_some(),
                "agent '{path}' is the orchestration root, not a worker"
            );
            Ok(path)
        })
        .collect()
}

fn selected_summaries(
    run: &RunHandle,
    paths: &[AgentPath],
) -> Result<Vec<OrchestrationAgentSummary>> {
    paths
        .iter()
        .map(|path| {
            let identity = run
                .agent_control_plane()
                .identity(path)
                .ok_or_else(|| anyhow::anyhow!("agent '{path}' is no longer registered"))?;
            let status = identity
                .task_id
                .as_ref()
                .and_then(|task_id| run.task_status(task_id));
            let checkpoint = run.latest_context_checkpoint(path);
            Ok(OrchestrationAgentSummary {
                path: path.to_string(),
                parent: identity.parent.map(|parent| parent.to_string()),
                task_id: identity.task_id.map(|task_id| task_id.to_string()),
                role: identity.role,
                target: identity.target.map(|target| target.to_string()),
                state: status.as_ref().map(|status| status.state),
                phase: status.as_ref().and_then(|status| status.phase.clone()),
                current_attempt: status.as_ref().map(|status| status.current_attempt),
                total_attempts: status.as_ref().map(|status| status.total_attempts),
                tokens_used: status.as_ref().map(|status| status.tokens_used),
                model: status.as_ref().and_then(|status| status.model.clone()),
                current_tool: status
                    .as_ref()
                    .and_then(|status| status.current_tool.clone()),
                session_id: status
                    .as_ref()
                    .and_then(|status| status.active_session_id.as_ref())
                    .map(ToString::to_string),
                latest_output: status
                    .as_ref()
                    .and_then(|status| status.latest_output.clone()),
                latest_error: status
                    .as_ref()
                    .and_then(|status| status.latest_error.clone()),
                verification_passed: status
                    .as_ref()
                    .and_then(|status| status.latest_verification.as_ref())
                    .map(|verification| verification.passed),
                wait_reason: status
                    .as_ref()
                    .and_then(|status| status.wait_reason.as_ref())
                    .map(|reason| reason.description()),
                context_revision: checkpoint.as_ref().map(|checkpoint| checkpoint.revision),
                context_paths: checkpoint
                    .as_ref()
                    .map(|checkpoint| checkpoint.state.paths.len())
                    .unwrap_or_default(),
                context_symbols: checkpoint
                    .as_ref()
                    .map(|checkpoint| checkpoint.state.symbols.len())
                    .unwrap_or_default(),
                context_artifacts: checkpoint
                    .as_ref()
                    .map(|checkpoint| checkpoint.state.artifact_references.len())
                    .unwrap_or_default(),
                queued_messages: run.agent_control_plane().mailbox_depth(path)?,
            })
        })
        .collect()
}

fn agent_needs_attention(agent: &OrchestrationAgentSummary) -> bool {
    agent.state.is_some_and(|state| {
        state.is_terminal()
            || matches!(
                state,
                TaskState::Blocked | TaskState::Parked | TaskState::AwaitingApply
            )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn restored_goal_accepts_control_actions() {
        let controller = agent_orchestration::GoalController::new(
            agent_orchestration::RunId::new(),
            "verify restored orchestration",
            Vec::new(),
            agent_orchestration::GoalControllerConfig::default(),
        )
        .expect("goal controller should be valid");
        let restored = agent_orchestration::GoalController::restore(
            controller.snapshot(),
            agent_orchestration::GoalControllerConfig::default(),
        )
        .expect("goal should restore");

        let blocked = apply_goal_action(
            &restored,
            UpdateOrchestrationGoalAction::ObserveBlocker,
            Some("missing-evidence".to_string()),
            Some("the worker output needs independent verification".to_string()),
        )
        .expect("blocker should be recorded");
        assert!(blocked.blocker.is_some());

        let cleared = apply_goal_action(
            &restored,
            UpdateOrchestrationGoalAction::ClearBlocker,
            None,
            None,
        )
        .expect("blocker should clear");
        assert!(cleared.blocker.is_none());

        let completed = apply_goal_action(
            &restored,
            UpdateOrchestrationGoalAction::Complete,
            None,
            None,
        )
        .expect("goal should complete");
        assert_eq!(completed.status, agent_orchestration::GoalStatus::Achieved);
    }

    #[test]
    fn goal_completion_requires_a_terminal_run_and_no_active_tasks() {
        let mut active_task =
            agent_orchestration::TaskStatus::new(agent_orchestration::TaskId::new("active-task"));
        active_task.state = agent_orchestration::TaskState::Running;
        assert!(ensure_goal_run_complete(RunState::Running, &[]).is_err());
        assert!(ensure_goal_run_complete(RunState::Completed, &[active_task]).is_err());
        assert!(ensure_goal_run_complete(RunState::Completed, &[]).is_ok());
    }
}
