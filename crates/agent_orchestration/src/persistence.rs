use crate::artifacts::Artifact;
use crate::events::SequencedRuntimeEvent;
use crate::ids::{PlanId, RunId, TaskId};
use crate::plan_graph::{OrchestrationPlan, OrchestrationTask};
use crate::state::{RunState, TaskAttempt, TaskState, TaskStatus};
use agent_settings::AgentExecutionPolicy;
use anyhow::{Context as _, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

pub const PERSISTENCE_SCHEMA_VERSION: u32 = 2;

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

    /// Prepares a non-terminal snapshot for resume by marking unfinished tasks as interrupted.
    pub fn for_resume(mut self) -> Self {
        if !self.state.is_terminal() {
            self.state = RunState::Interrupted;
        }
        for status in &mut self.task_statuses {
            if !status.state.is_terminal() {
                status.state = TaskState::Interrupted;
            }
        }
        self.updated_at = Utc::now();
        self
    }
}
