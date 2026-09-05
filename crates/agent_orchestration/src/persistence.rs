use crate::artifacts::Artifact;
use crate::events::SequencedRuntimeEvent;
use crate::ids::{PlanId, RunId, TaskId};
use crate::plan_graph::{OrchestrationPlan, OrchestrationTask};
use crate::state::{RunState, TaskAttempt, TaskState, TaskStatus};
use agent_settings::AgentExecutionPolicy;
use anyhow::{Context as _, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

pub const PERSISTENCE_SCHEMA_VERSION: u32 = 3;

/// A completely serializable, database-safe snapshot of an orchestration run.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PersistedRun {
    pub schema_version: u32,
    pub run_id: RunId,
    pub plan: OrchestrationPlan,
    pub state: RunState,
    #[serde(default)]
    pub policy: AgentExecutionPolicy,
    pub task_statuses: Vec<TaskStatus>,
    pub task_attempts: Vec<(TaskId, Vec<TaskAttempt>)>,
    #[serde(default)]
    pub artifacts: Vec<Artifact>,
    #[serde(default)]
    pub event_log: Vec<SequencedRuntimeEvent>,
    /// Sequence number of the last event persisted. Subscribers replay events
    /// with `seq > last_event_seq` on resume to bridge the persistence gap.
    #[serde(default)]
    pub last_event_seq: u64,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl PersistedRun {
    pub fn new(
        run_id: RunId,
        plan: OrchestrationPlan,
        state: RunState,
        policy: AgentExecutionPolicy,
        task_statuses: Vec<TaskStatus>,
        task_attempts: Vec<(TaskId, Vec<TaskAttempt>)>,
        artifacts: Vec<Artifact>,
        event_log: Vec<SequencedRuntimeEvent>,
    ) -> Self {
        let last_event_seq = event_log
            .iter()
            .map(|event| event.seq)
            .max()
            .unwrap_or_default();
        let now = Utc::now();
        Self {
            schema_version: PERSISTENCE_SCHEMA_VERSION,
            run_id,
            plan,
            state,
            policy,
            task_statuses,
            task_attempts,
            artifacts,
            event_log,
            last_event_seq,
            created_at: now,
            updated_at: now,
        }
    }

    /// Converts this snapshot to JSON.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(self).context("failed to serialize PersistedRun")
    }

    /// Deserializes a snapshot from JSON with schema compatibility checks.
    pub fn from_json(json_str: &str) -> Result<Self> {
        let mut run: Self =
            serde_json::from_str(json_str).context("failed to deserialize PersistedRun")?;
        run.upgrade_schema();
        Ok(run)
    }

    /// Converts this snapshot to a `serde_json::Value`.
    pub fn to_value(&self) -> Result<serde_json::Value> {
        serde_json::to_value(self).context("failed to convert PersistedRun to value")
    }

    /// Restores a snapshot from a `serde_json::Value`.
    pub fn from_value(value: serde_json::Value) -> Result<Self> {
        let mut run: Self =
            serde_json::from_value(value).context("failed to convert value to PersistedRun")?;
        run.upgrade_schema();
        Ok(run)
    }

    /// Fills defaults introduced in newer schema versions. The `#[serde(default)]`
    /// attributes already deserialize missing fields, so this only back-fills
    /// derived values for older snapshots.
    fn upgrade_schema(&mut self) {
        if self.schema_version < PERSISTENCE_SCHEMA_VERSION {
            self.schema_version = PERSISTENCE_SCHEMA_VERSION;
        }
        if self.last_event_seq == 0 {
            self.last_event_seq = self
                .event_log
                .iter()
                .map(|event| event.seq)
                .max()
                .unwrap_or_default();
        }
    }

    /// Creates a synthetic PersistedRun from legacy plan steps for backward compatibility.
    pub fn from_legacy_steps(title: String, steps: Vec<(String, bool)>) -> Self {
        let run_id = RunId::new();
        let tasks = steps
            .into_iter()
            .enumerate()
            .map(|(index, (step, completed))| {
                let task_id = TaskId::new(format!("step-{}", index + 1));
                let mut task = OrchestrationTask::new(task_id, format!("Step {}", index + 1), step);
                if index > 0 {
                    task.depends_on = vec![TaskId::new(format!("step-{}", index))];
                }
                (task, completed)
            })
            .collect::<Vec<_>>();

        let mut statuses = Vec::new();
        let plan_tasks = tasks
            .iter()
            .map(|(t, completed)| {
                let mut status = TaskStatus::new(t.id.clone());
                if *completed {
                    status.state = TaskState::Completed;
                }
                statuses.push(status);
                t.clone()
            })
            .collect();

        let plan = OrchestrationPlan {
            id: PlanId::new(),
            title,
            explanation: None,
            tasks: plan_tasks,
        };

        Self::new(
            run_id,
            plan,
            RunState::Completed,
            AgentExecutionPolicy::default(),
            statuses,
            Vec::new(),
            Vec::new(),
            Vec::new(),
        )
    }

    /// Prepares a non-terminal snapshot for resume by reconciling unfinished tasks.
    pub fn for_resume(self) -> Self {
        self.reconcile_on_restart()
    }

    /// Reconciles task and run states upon restart or reload:
    /// - Active tasks (Running, Verifying, Repairing, Retrying) lack live execution handles.
    /// - If worker capabilities allow session resume, tasks are parked awaiting reconnect.
    /// - Otherwise, tasks are marked Interrupted with an explicit message requiring a restart attempt.
    /// - Non-terminal runs are set to Interrupted or AwaitingApply.
    pub fn reconcile_on_restart(mut self) -> Self {
        if !self.state.is_terminal() && !matches!(self.state, RunState::Proposed | RunState::Paused)
        {
            let has_runnable_work = self.task_statuses.iter().any(|status| {
                !status.state.is_terminal()
                    && !matches!(status.state, TaskState::AwaitingApply | TaskState::Blocked)
            });
            let has_awaiting_apply = self
                .task_statuses
                .iter()
                .any(|status| status.state == TaskState::AwaitingApply);
            self.state = if !has_runnable_work && has_awaiting_apply {
                RunState::AwaitingApply
            } else {
                RunState::Interrupted
            };
        }
        for status in &mut self.task_statuses {
            if status.state.is_active() {
                let can_resume = status
                    .worker_metadata
                    .as_ref()
                    .map(|m| m.capabilities.can_resume || m.capabilities.can_load_session)
                    .unwrap_or(false);
                if can_resume && status.active_session_id.is_some() {
                    status.state = TaskState::Parked;
                    status.wait_reason = Some(
                        crate::worker::StructuredWaitReason::AwaitingWorkerReconnect {
                            worker: status.target.clone(),
                            attempt: status.current_attempt,
                        },
                    );
                } else {
                    status.state = if status.target.is_acp() {
                        TaskState::Parked
                    } else {
                        TaskState::Interrupted
                    };
                    status.latest_error = Some(
                        "task was running when process terminated; worker does not support resume, restart attempt required"
                            .to_string(),
                    );
                    status.wait_reason = status.target.is_acp().then(|| crate::worker::StructuredWaitReason::AwaitingUserInput {
                        question: "Worker cannot resume after restart. Restart this attempt or cancel the task.".to_string(),
                    });
                }
                status.updated_at = Utc::now();
            }
        }
        self.updated_at = Utc::now();
        self
    }
}
