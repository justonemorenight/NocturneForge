use crate::events::RuntimeEvent;
use crate::ids::TaskId;
use crate::plan_graph::OrchestrationTask;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Serializable representation of an ACP task conversion for tools.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct AcpTaskBridge {
    pub task_id: String,
    pub label: String,
    pub prompt: String,
    #[serde(default)]
    pub tools: Option<Vec<String>>,
    #[serde(default)]
    pub depends_on: Vec<String>,
}

impl From<OrchestrationTask> for AcpTaskBridge {
    fn from(task: OrchestrationTask) -> Self {
        Self {
            task_id: task.id.to_string(),
            label: task.label,
            prompt: task.description,
            tools: task.tools,
            depends_on: task
                .depends_on
                .into_iter()
                .map(|id| id.to_string())
                .collect(),
        }
    }
}

impl From<AcpTaskBridge> for OrchestrationTask {
    fn from(bridge: AcpTaskBridge) -> Self {
        Self {
            id: TaskId::new(bridge.task_id),
            label: bridge.label,
            description: bridge.prompt,
            role: None,
            model_override: None,
            thinking_effort: None,
            tools: bridge.tools,
            depends_on: bridge.depends_on.into_iter().map(TaskId::new).collect(),
            acceptance_criteria: Vec::new(),
            max_retries: None,
            token_budget: None,
            time_budget_secs: None,
            repair_on_failure: true,
            context_paths: Vec::new(),
            expected_output: None,
            evidence_required: false,
            tool_call_budget: None,
            objective: None,
            scope: None,
            native_role: None,
            target: crate::worker::WorkerTarget::Native,
            mode: None,
            workspace_policy: crate::worker::WorkspacePolicy::default(),
            verification_command: None,
        }
    }
}

/// Adapter converting RuntimeEvents into human-readable ACP tool call updates.
pub struct AcpEventAdapter;

impl AcpEventAdapter {
    pub fn format_event_for_acp(event: &RuntimeEvent) -> Option<String> {
        match event {
            RuntimeEvent::TaskDispatched {
                task_id, attempt, ..
            } => Some(format!("Starting task `{task_id}` (attempt {attempt})...")),
            RuntimeEvent::TaskVerifying { task_id, .. } => {
                Some(format!("Verifying output for task `{task_id}`..."))
            }
            RuntimeEvent::TaskRepairing {
                task_id, reason, ..
            } => Some(format!("Repairing task `{task_id}`: {reason}")),
            RuntimeEvent::TaskRetrying {
                task_id,
                next_attempt,
                reason,
                ..
            } => Some(format!(
                "Retrying task `{task_id}` (attempt {next_attempt}): {reason}"
            )),
            RuntimeEvent::TaskCompleted {
                task_id,
                tokens_used,
                ..
            } => Some(format!(
                "Task `{task_id}` completed ({tokens_used} tokens used)"
            )),
            RuntimeEvent::TaskFailed { task_id, error, .. } => {
                Some(format!("Task `{task_id}` failed: {error}"))
            }
            RuntimeEvent::TaskCancelled {
                task_id, reason, ..
            } => Some(format!("Task `{task_id}` cancelled: {reason}")),
            RuntimeEvent::RunCompleted {
                total_tokens_used,
                duration_ms,
                ..
            } => Some(format!(
                "Orchestration run completed in {}ms (total tokens: {total_tokens_used})",
                duration_ms
            )),
            _ => None,
        }
    }
}
