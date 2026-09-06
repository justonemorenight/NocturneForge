use crate::ids::{RunId, TaskId};
use crate::state::{RunState, TaskState, TaskStatus};
use anyhow::{Result, bail};
use chrono::{DateTime, Utc};
use collections::HashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GoalControllerConfig {
    pub repeated_blocker_threshold: u32,
    pub max_blocker_bytes: usize,
}

impl Default for GoalControllerConfig {
    fn default() -> Self {
        Self {
            repeated_blocker_threshold: 3,
            max_blocker_bytes: 16 * 1024,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GoalStatus {
    Active,
    Achieved,
    Blocked,
    Failed,
    Cancelled,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GoalBlocker {
    pub code: String,
    pub detail: String,
    pub consecutive_observations: u32,
    pub last_observed_at: DateTime<Utc>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GoalSnapshot {
    pub run_id: RunId,
    pub objective: String,
    pub status: GoalStatus,
    pub task_ids: Vec<TaskId>,
    pub total_tasks: usize,
    pub completed_tasks: usize,
    pub verified_tasks: usize,
    pub failed_tasks: Vec<TaskId>,
    pub cancelled_tasks: Vec<TaskId>,
    pub total_tokens_used: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blocker: Option<GoalBlocker>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Clone)]
pub struct GoalController {
    config: GoalControllerConfig,
    snapshot: Arc<RwLock<GoalSnapshot>>,
}

impl GoalController {
    pub fn new(
        run_id: RunId,
        objective: impl Into<String>,
        task_ids: Vec<TaskId>,
        config: GoalControllerConfig,
    ) -> Result<Self> {
        validate_config(&config)?;
        let objective = objective.into();
        if objective.trim().is_empty() {
            bail!("goal objective cannot be empty");
        }
        let mut task_ids = task_ids;
        task_ids.sort();
        task_ids.dedup();
        let total_tasks = task_ids.len();
        let now = Utc::now();
        Ok(Self {
            config,
            snapshot: Arc::new(RwLock::new(GoalSnapshot {
                run_id,
                objective,
                status: GoalStatus::Active,
                task_ids,
                total_tasks,
                completed_tasks: 0,
                verified_tasks: 0,
                failed_tasks: Vec::new(),
                cancelled_tasks: Vec::new(),
                total_tokens_used: 0,
                blocker: None,
                created_at: now,
                updated_at: now,
            })),
        })
    }

    pub fn restore(snapshot: GoalSnapshot, config: GoalControllerConfig) -> Result<Self> {
        validate_config(&config)?;
        let unique_task_count = snapshot
            .task_ids
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len();
        if snapshot.objective.trim().is_empty()
            || unique_task_count != snapshot.task_ids.len()
            || snapshot.total_tasks != snapshot.task_ids.len()
            || snapshot.completed_tasks > snapshot.total_tasks
            || snapshot.verified_tasks > snapshot.completed_tasks
        {
            bail!("persisted goal snapshot is inconsistent");
        }
        Ok(Self {
            config,
            snapshot: Arc::new(RwLock::new(snapshot)),
        })
    }

    pub fn snapshot(&self) -> GoalSnapshot {
        self.snapshot.read().clone()
    }

    pub fn observe(&self, run_state: RunState, statuses: &[TaskStatus]) -> GoalSnapshot {
        let mut snapshot = self.snapshot.write();
        let by_task = statuses
            .iter()
            .map(|status| (status.task_id.clone(), status))
            .collect::<HashMap<_, _>>();
        snapshot.completed_tasks = snapshot
            .task_ids
            .iter()
            .filter_map(|task_id| by_task.get(task_id))
            .filter(|status| status.state == TaskState::Completed)
            .count();
        snapshot.verified_tasks = snapshot
            .task_ids
            .iter()
            .filter_map(|task_id| by_task.get(task_id))
            .filter(|status| {
                status.state == TaskState::Completed
                    && status
                        .latest_verification
                        .as_ref()
                        .is_none_or(|verification| verification.passed)
            })
            .count();
        snapshot.failed_tasks = task_ids_in_state(&snapshot.task_ids, &by_task, TaskState::Failed);
        snapshot.cancelled_tasks =
            task_ids_in_state(&snapshot.task_ids, &by_task, TaskState::Cancelled);
        snapshot.total_tokens_used = snapshot
            .task_ids
            .iter()
            .filter_map(|task_id| by_task.get(task_id))
            .map(|status| status.tokens_used)
            .sum();
        snapshot.status = match run_state {
            RunState::Completed
                if snapshot.completed_tasks == snapshot.total_tasks
                    && snapshot.verified_tasks == snapshot.total_tasks =>
            {
                GoalStatus::Achieved
            }
            RunState::Completed => GoalStatus::Failed,
            RunState::Failed => GoalStatus::Failed,
            RunState::Cancelled => GoalStatus::Cancelled,
            _ if snapshot.status == GoalStatus::Blocked => GoalStatus::Blocked,
            _ => GoalStatus::Active,
        };
        if snapshot.status == GoalStatus::Achieved {
            snapshot.blocker = None;
        }
        snapshot.updated_at = Utc::now();
        snapshot.clone()
    }

    pub fn record_blocker(
        &self,
        code: impl Into<String>,
        detail: impl Into<String>,
    ) -> Result<GoalSnapshot> {
        let code = code.into();
        let detail = detail.into();
        if code.trim().is_empty() || detail.trim().is_empty() {
            bail!("goal blocker code and detail cannot be empty");
        }
        if code.len().saturating_add(detail.len()) > self.config.max_blocker_bytes {
            bail!("goal blocker exceeds its configured byte limit");
        }
        let mut snapshot = self.snapshot.write();
        if snapshot.status != GoalStatus::Active && snapshot.status != GoalStatus::Blocked {
            bail!("cannot record a blocker for a terminal goal");
        }
        let observations = snapshot
            .blocker
            .as_ref()
            .filter(|blocker| blocker.code == code)
            .map(|blocker| blocker.consecutive_observations.saturating_add(1))
            .unwrap_or(1);
        let now = Utc::now();
        snapshot.blocker = Some(GoalBlocker {
            code,
            detail,
            consecutive_observations: observations,
            last_observed_at: now,
        });
        snapshot.status = if observations >= self.config.repeated_blocker_threshold {
            GoalStatus::Blocked
        } else {
            GoalStatus::Active
        };
        snapshot.updated_at = now;
        Ok(snapshot.clone())
    }

    pub fn clear_blocker(&self) -> GoalSnapshot {
        let mut snapshot = self.snapshot.write();
        snapshot.blocker = None;
        if snapshot.status == GoalStatus::Blocked {
            snapshot.status = GoalStatus::Active;
        }
        snapshot.updated_at = Utc::now();
        snapshot.clone()
    }
}

fn task_ids_in_state(
    expected_task_ids: &[TaskId],
    statuses: &HashMap<TaskId, &TaskStatus>,
    state: TaskState,
) -> Vec<TaskId> {
    let mut task_ids = expected_task_ids
        .iter()
        .filter(|task_id| {
            statuses
                .get(*task_id)
                .is_some_and(|status| status.state == state)
        })
        .cloned()
        .collect::<Vec<_>>();
    task_ids.sort();
    task_ids
}

fn validate_config(config: &GoalControllerConfig) -> Result<()> {
    if config.repeated_blocker_threshold == 0 || config.max_blocker_bytes == 0 {
        bail!("goal controller limits must be greater than zero");
    }
    Ok(())
}

#[cfg(test)]
#[path = "goal_controller_tests.rs"]
mod tests;
