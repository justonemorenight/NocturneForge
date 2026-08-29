use crate::budget::TaskBudgetState;
use crate::ids::TaskId;
use crate::verification::VerificationResult;
use agent_client_protocol::schema::v1 as acp;
use chrono::{DateTime, Utc};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// High-level lifecycle state of an orchestration run.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "snake_case")]
pub enum RunState {
    /// Plan is being drafted or composed.
    #[default]
    Draft,
    /// Plan has been proposed and is waiting for user approval.
    Proposed,
    /// Plan is approved by the user and queued for execution.
    Approved,
    /// Tasks are actively executing through the scheduler.
    Running,
    /// Execution is paused by user request.
    Paused,
    /// Final verification phase across completed tasks.
    Verifying,
    /// Repair phase addressing verification failures.
    Repairing,
    /// Run finished successfully with all tasks completed and verified.
    Completed,
    /// Run halted due to an unrecoverable failure or policy limit.
    Failed,
    /// Run was cancelled by user or system shutdown.
    Cancelled,
    /// Run was interrupted (e.g. process crash/restart) and needs resume.
    Interrupted,
}

impl RunState {
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }

    pub fn is_active(&self) -> bool {
        matches!(self, Self::Running | Self::Verifying | Self::Repairing)
    }

    pub fn can_transition_to(&self, next: Self) -> bool {
        match (self, next) {
            (Self::Draft, Self::Proposed) => true,
            (Self::Draft, Self::Approved) => true,
            (Self::Draft, Self::Cancelled) => true,
            (Self::Proposed, Self::Approved) => true,
            (Self::Proposed, Self::Cancelled) => true,
            (Self::Proposed, Self::Draft) => true,
            (Self::Approved, Self::Running) => true,
            (Self::Approved, Self::Cancelled) => true,
            (Self::Running, Self::Paused) => true,
            (Self::Running, Self::Verifying) => true,
            (Self::Running, Self::Repairing) => true,
            (Self::Running, Self::Completed) => true,
            (Self::Running, Self::Failed) => true,
            (Self::Running, Self::Cancelled) => true,
            (Self::Running, Self::Interrupted) => true,
            (Self::Paused, Self::Running) => true,
            (Self::Paused, Self::Cancelled) => true,
            (Self::Verifying, Self::Repairing) => true,
            (Self::Verifying, Self::Completed) => true,
            (Self::Verifying, Self::Failed) => true,
            (Self::Verifying, Self::Cancelled) => true,
            (Self::Repairing, Self::Running) => true,
            (Self::Repairing, Self::Verifying) => true,
            (Self::Repairing, Self::Completed) => true,
            (Self::Repairing, Self::Failed) => true,
            (Self::Repairing, Self::Cancelled) => true,
            (Self::Interrupted, Self::Approved) => true,
            (Self::Interrupted, Self::Running) => true,
            (Self::Interrupted, Self::Cancelled) => true,
            _ => false,
        }
    }
}

/// Detailed lifecycle state of an individual task in a plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "snake_case")]
pub enum TaskState {
    /// In queue, waiting for scheduling.
    #[default]
    Pending,
    /// Blocked waiting for predecessor tasks in the dependency graph.
    WaitingDependency,
    /// Blocked due to user decision or external dependency.
    Blocked,
    /// Actively running.
    Running,
    /// Verifying output against acceptance criteria.
    Verifying,
    /// Running repair action after verification failure.
    Repairing,
    /// Waiting for backoff timer before retrying.
    Retrying,
    /// Parked/idle waiting for external event or confirmation.
    Parked,
    /// Completed and verified successfully.
    Completed,
    /// Failed after exhausting allowed retries.
    Failed,
    /// Cancelled before completion.
    Cancelled,
    /// Interrupted by process restart.
    Interrupted,
}

impl TaskState {
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }

    pub fn is_active(&self) -> bool {
        matches!(
            self,
            Self::Running | Self::Verifying | Self::Repairing | Self::Retrying
        )
    }

    pub fn is_waiting(&self) -> bool {
        matches!(
            self,
            Self::Pending | Self::WaitingDependency | Self::Blocked | Self::Parked
        )
    }
}

/// Record of a single execution attempt for a task.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskAttempt {
    pub attempt_index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<acp::SessionId>,
    pub started_at: DateTime<Utc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finished_at: Option<DateTime<Utc>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens_used: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verification: Option<VerificationResult>,
}

impl TaskAttempt {
    pub fn new(attempt_index: u32, session_id: Option<acp::SessionId>) -> Self {
        Self {
            attempt_index,
            session_id,
            started_at: Utc::now(),
            finished_at: None,
            output: None,
            error: None,
            tokens_used: None,
            verification: None,
        }
    }
}

/// Live status summary of a task.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskStatus {
    pub task_id: TaskId,
    pub state: TaskState,
    pub current_attempt: u32,
    pub total_attempts: u32,
    #[serde(default)]
    pub tokens_used: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub active_session_id: Option<acp::SessionId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub latest_output: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub latest_error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub latest_verification: Option<VerificationResult>,
    pub updated_at: DateTime<Utc>,
    /// Model assigned to this task, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Phase of the current attempt (e.g. "running", "verifying").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase: Option<String>,
    /// Tool currently being invoked, when the executor reports one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub current_tool: Option<String>,
    /// Rolling budget state: usage counters and stop reason, if stopped.
    #[serde(default)]
    pub budget_state: TaskBudgetState,
}

impl TaskStatus {
    pub fn new(task_id: TaskId) -> Self {
        Self {
            task_id,
            state: TaskState::Pending,
            current_attempt: 0,
            total_attempts: 0,
            tokens_used: 0,
            active_session_id: None,
            latest_output: None,
            latest_error: None,
            latest_verification: None,
            updated_at: Utc::now(),
            model: None,
            phase: None,
            current_tool: None,
            budget_state: TaskBudgetState::default(),
        }
    }
}
