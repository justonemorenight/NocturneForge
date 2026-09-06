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
pub struct ListOrchestrationAgentsInput {}

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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OrchestrationControlOutput {
    Success {
        run_id: String,
        run_state: RunState,
        goal: GoalSnapshot,
        agents: Vec<OrchestrationAgentSummary>,
    },
    MessageQueued {
        sequence: u64,
        recipient: String,
        delivery: String,
    },
    GoalUpdated {
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
        _input: ToolInput<Self::Input>,
        _event_stream: ToolCallEventStream,
        cx: &mut App,
    ) -> Task<Result<Self::Output, Self::Output>> {
        let result = self
            .thread
            .read_with(cx, |thread, _cx| thread.orchestration_run().cloned())
            .map_err(|error| error.to_string())
            .and_then(|run| run.ok_or_else(|| "no orchestration run is active".to_string()))
            .and_then(|run| {
                summaries(&run)
                    .map(|agents| (run, agents))
                    .map_err(|error| error.to_string())
            });
        Task::ready(match result {
            Ok((run, agents)) => Ok(success_output(&run, agents)),
            Err(error) => Err(OrchestrationControlOutput::Error { error }),
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
/// Send context or a follow-up to an agent in the active orchestration run.
pub struct SendMessageToAgentInput {
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
                .read_with(cx, |thread, _cx| thread.orchestration_run().cloned())
                .map_err(|error| OrchestrationControlOutput::Error {
                    error: error.to_string(),
                })?
                .ok_or_else(|| OrchestrationControlOutput::Error {
                    error: "no orchestration run is active".to_string(),
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
/// Record a repeated blocker observation or clear a resolved blocker for the active goal.
pub struct UpdateOrchestrationGoalInput {
    pub action: UpdateOrchestrationGoalAction,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum UpdateOrchestrationGoalAction {
    ObserveBlocker,
    ClearBlocker,
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
            let run = thread
                .read_with(cx, |thread, _cx| thread.orchestration_run().cloned())
                .map_err(|error| OrchestrationControlOutput::Error {
                    error: error.to_string(),
                })?
                .ok_or_else(|| OrchestrationControlOutput::Error {
                    error: "no orchestration run is active".to_string(),
                })?;
            let goal = match input.action {
                UpdateOrchestrationGoalAction::ObserveBlocker => {
                    let code = input
                        .code
                        .ok_or_else(|| OrchestrationControlOutput::Error {
                            error: "code is required when observing a blocker".to_string(),
                        })?;
                    let detail = input
                        .detail
                        .ok_or_else(|| OrchestrationControlOutput::Error {
                            error: "detail is required when observing a blocker".to_string(),
                        })?;
                    run.record_goal_blocker(code, detail).map_err(|error| {
                        OrchestrationControlOutput::Error {
                            error: error.to_string(),
                        }
                    })?
                }
                UpdateOrchestrationGoalAction::ClearBlocker => run.clear_goal_blocker(),
            };
            Ok(OrchestrationControlOutput::GoalUpdated { goal })
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
/// Wait until one of the selected orchestration agents changes state or needs attention.
pub struct WaitForAgentsInput {
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
                .read_with(cx, |thread, _cx| thread.orchestration_run().cloned())
                .map_err(|error| OrchestrationControlOutput::Error {
                    error: error.to_string(),
                })?
                .ok_or_else(|| OrchestrationControlOutput::Error {
                    error: "no orchestration run is active".to_string(),
                })?;
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
