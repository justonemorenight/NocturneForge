use crate::budget::{BudgetExceeded, BudgetUsage};
use crate::ids::TaskId;
use crate::state::{TaskAttempt, TaskState, TaskStatus};
use crate::verification::VerificationResult;
use agent_client_protocol::schema::v1 as acp;
use chrono::Utc;
use collections::{HashMap, HashSet};
use parking_lot::RwLock;
use std::sync::Arc;

/// Central thread-safe registry tracking all task states, attempts, and metrics in an orchestration run.
#[derive(Clone, Default)]
pub struct TaskRegistry {
    statuses: Arc<RwLock<HashMap<TaskId, TaskStatus>>>,
    attempts: Arc<RwLock<HashMap<TaskId, Vec<TaskAttempt>>>>,
    tasks_by_session: Arc<RwLock<HashMap<acp::SessionId, TaskId>>>,
}

impl TaskRegistry {
    fn update_lifecycle(status: &mut TaskStatus, state: TaskState) {
        status.state = state;
        status.phase = match state {
            TaskState::Running => Some("running".to_string()),
            TaskState::Verifying => Some("verifying".to_string()),
            TaskState::Repairing => Some("repairing".to_string()),
            TaskState::Retrying => Some("retrying".to_string()),
            _ => None,
        };
        if state != TaskState::Running {
            status.current_tool = None;
        }
    }

    pub fn new() -> Self {
        Self {
            statuses: Arc::new(RwLock::new(HashMap::default())),
            attempts: Arc::new(RwLock::new(HashMap::default())),
            tasks_by_session: Arc::new(RwLock::new(HashMap::default())),
        }
    }

    /// Registers a new task with initial Pending status.
    pub fn register_task(&self, task_id: TaskId) {
        let mut statuses = self.statuses.write();
        statuses
            .entry(task_id.clone())
            .or_insert_with(|| TaskStatus::new(task_id));
    }

    /// Restores a previously persisted attempt without changing the task's
    /// lifecycle state. This is deliberately separate from execution APIs.
    pub fn restore_attempt(&self, task_id: TaskId, attempt: TaskAttempt) {
        self.attempts
            .write()
            .entry(task_id)
            .or_default()
            .push(attempt);
    }

    /// Restores a complete persisted status and rebuilds its session lookup.
    pub fn restore_status(&self, status: TaskStatus) {
        let mut statuses = self.statuses.write();
        let mut tasks_by_session = self.tasks_by_session.write();
        if let Some(previous_session_id) = statuses
            .get(&status.task_id)
            .and_then(|previous| previous.active_session_id.as_ref())
        {
            tasks_by_session.remove(previous_session_id);
        }
        if let Some(session_id) = status.active_session_id.as_ref() {
            tasks_by_session.insert(session_id.clone(), status.task_id.clone());
        }
        statuses.insert(status.task_id.clone(), status);
    }

    /// Updates the state of a registered task.
    pub fn set_state(&self, task_id: &TaskId, state: TaskState) {
        let mut statuses = self.statuses.write();
        if let Some(status) = statuses.get_mut(task_id) {
            Self::update_lifecycle(status, state);
            status.updated_at = Utc::now();
        }
    }

    /// Begins a new execution attempt for a task.
    pub fn start_attempt(&self, task_id: &TaskId, session_id: Option<acp::SessionId>) -> u32 {
        let mut statuses = self.statuses.write();
        let mut attempts = self.attempts.write();

        let attempt_index = {
            let list = attempts.entry(task_id.clone()).or_default();
            let next_index = list.len() as u32 + 1;
            list.push(TaskAttempt::new(next_index, session_id.clone()));
            next_index
        };

        if let Some(status) = statuses.get_mut(task_id) {
            Self::update_lifecycle(status, TaskState::Running);
            status.current_attempt = attempt_index;
            status.total_attempts = attempt_index;
            self.replace_active_session(status, session_id);
            status.updated_at = Utc::now();
        }

        attempt_index
    }

    /// Records a session allocated by an executor after an attempt has started.
    pub fn set_active_session_id(&self, task_id: &TaskId, session_id: acp::SessionId) {
        let mut statuses = self.statuses.write();
        let mut attempts = self.attempts.write();

        let Some(status) = statuses.get_mut(task_id) else {
            return;
        };
        self.replace_active_session(status, Some(session_id.clone()));
        if let Some(attempt) = attempts
            .get_mut(task_id)
            .and_then(|attempts| attempts.last_mut())
        {
            attempt.session_id = Some(session_id);
        }
        status.updated_at = Utc::now();
    }

    fn replace_active_session(&self, status: &mut TaskStatus, session_id: Option<acp::SessionId>) {
        let mut tasks_by_session = self.tasks_by_session.write();
        if let Some(previous_session_id) = status.active_session_id.take() {
            tasks_by_session.remove(&previous_session_id);
        }
        if let Some(session_id) = session_id {
            tasks_by_session.insert(session_id.clone(), status.task_id.clone());
            status.active_session_id = Some(session_id);
        }
    }

    /// Completes an execution attempt with success or verification outcome.
    pub fn complete_attempt(
        &self,
        task_id: &TaskId,
        attempt: u32,
        output: Option<String>,
        tokens_used: Option<u64>,
        verification: Option<VerificationResult>,
    ) {
        let mut statuses = self.statuses.write();
        let mut attempts = self.attempts.write();

        if let Some(list) = attempts.get_mut(task_id) {
            if let Some(record) = list.iter_mut().find(|a| a.attempt_index == attempt) {
                record.finished_at = Some(Utc::now());
                record.output = output.clone();
                record.tokens_used = tokens_used;
                record.verification = verification.clone();
            }
        }

        if let Some(status) = statuses.get_mut(task_id) {
            if let Some(tokens) = tokens_used {
                status.tokens_used += tokens;
            }
            status.latest_output = output;
            status.latest_verification = verification.clone();
            status.updated_at = Utc::now();

            let state = if let Some(ver) = verification {
                if ver.passed {
                    TaskState::Completed
                } else if ver.repairable {
                    TaskState::Repairing
                } else if ver.retryable {
                    TaskState::Retrying
                } else {
                    TaskState::Failed
                }
            } else {
                TaskState::Completed
            };
            Self::update_lifecycle(status, state);
        }
    }

    /// Records failure on an execution attempt.
    pub fn fail_attempt(&self, task_id: &TaskId, attempt: u32, error: String, retryable: bool) {
        self.fail_attempt_with_usage(
            task_id,
            attempt,
            error,
            retryable,
            BudgetUsage::default(),
            None,
        );
    }

    /// Records failure together with attempt-local usage and an optional typed budget stop.
    pub fn fail_attempt_with_usage(
        &self,
        task_id: &TaskId,
        attempt: u32,
        error: String,
        retryable: bool,
        usage: BudgetUsage,
        stopped_reason: Option<BudgetExceeded>,
    ) {
        let mut statuses = self.statuses.write();
        let mut attempts = self.attempts.write();

        if let Some(list) = attempts.get_mut(task_id) {
            if let Some(record) = list.iter_mut().find(|a| a.attempt_index == attempt) {
                record.finished_at = Some(Utc::now());
                record.error = Some(error.clone());
                record.tokens_used = Some(usage.tokens_used);
            }
        }

        if let Some(status) = statuses.get_mut(task_id) {
            status.tokens_used = status.tokens_used.saturating_add(usage.tokens_used);
            status.budget_state.tokens_used = status
                .budget_state
                .tokens_used
                .saturating_add(usage.tokens_used);
            status.budget_state.tool_calls_used = status
                .budget_state
                .tool_calls_used
                .saturating_add(usage.tool_calls);
            if stopped_reason.is_some() {
                status.budget_state.stopped_reason = stopped_reason;
            }
            status.latest_error = Some(error);
            status.updated_at = Utc::now();
            let state = if retryable {
                TaskState::Retrying
            } else {
                TaskState::Failed
            };
            Self::update_lifecycle(status, state);
        }
    }

    /// Records a tool call on a task, enforcing the tool-call budget if set.
    pub fn add_tool_call(
        &self,
        task_id: &TaskId,
        budget: Option<u64>,
    ) -> Result<(), BudgetExceeded> {
        let mut statuses = self.statuses.write();
        let Some(status) = statuses.get_mut(task_id) else {
            return Ok(());
        };
        status.budget_state.tool_calls_used = status.budget_state.tool_calls_used.saturating_add(1);
        if let Some(budget) = budget
            && status.budget_state.tool_calls_used > budget
        {
            let exceeded = BudgetExceeded::ToolCallBudgetExceeded {
                budget,
                used: status.budget_state.tool_calls_used,
            };
            status.budget_state.stopped_reason = Some(exceeded.clone());
            status.updated_at = Utc::now();
            return Err(exceeded);
        }
        status.updated_at = Utc::now();
        Ok(())
    }

    /// Merges budget usage reported by the executor into the task status.
    pub fn update_budget_usage(&self, task_id: &TaskId, tokens_used: u64, tool_calls: u64) {
        let mut statuses = self.statuses.write();
        if let Some(status) = statuses.get_mut(task_id) {
            status.budget_state.tokens_used =
                status.budget_state.tokens_used.saturating_add(tokens_used);
            status.budget_state.tool_calls_used = status
                .budget_state
                .tool_calls_used
                .saturating_add(tool_calls);
            status.updated_at = Utc::now();
        }
    }

    /// Records a budget stop reason on a task (never completes the task).
    pub fn stop_for_budget(&self, task_id: &TaskId, reason: BudgetExceeded) {
        let mut statuses = self.statuses.write();
        if let Some(status) = statuses.get_mut(task_id) {
            status.budget_state.stopped_reason = Some(reason);
            Self::update_lifecycle(status, TaskState::Failed);
            status.updated_at = Utc::now();
        }
    }

    /// Sets the model assigned to a task, preserving an earlier assignment.
    pub fn set_model(&self, task_id: &TaskId, model: impl Into<String>) {
        let mut statuses = self.statuses.write();
        if let Some(status) = statuses.get_mut(task_id) {
            if status.model.is_none() {
                status.model = Some(model.into());
                status.updated_at = Utc::now();
            }
        }
    }

    /// Updates the current attempt phase (e.g. "running", "verifying").
    pub fn set_phase(&self, task_id: &TaskId, phase: impl Into<String>) {
        let mut statuses = self.statuses.write();
        if let Some(status) = statuses.get_mut(task_id) {
            status.phase = Some(phase.into());
            status.updated_at = Utc::now();
        }
    }

    /// Sets or clears the tool currently being invoked by the task.
    pub fn set_current_tool(&self, task_id: &TaskId, tool: Option<impl Into<String>>) {
        let mut statuses = self.statuses.write();
        if let Some(status) = statuses.get_mut(task_id) {
            status.current_tool = tool.map(Into::into);
            status.updated_at = Utc::now();
        }
    }

    /// Retrieves status for a single task.
    pub fn status(&self, task_id: &TaskId) -> Option<TaskStatus> {
        let statuses = self.statuses.read();
        statuses.get(task_id).cloned()
    }

    /// Retrieves a task status by its active agent session without cloning the full registry.
    pub fn status_by_session_id(&self, session_id: &acp::SessionId) -> Option<TaskStatus> {
        let tasks_by_session = self.tasks_by_session.read();
        let task_id = tasks_by_session.get(session_id)?.clone();
        drop(tasks_by_session);
        self.statuses.read().get(&task_id).cloned()
    }

    /// Retrieves all task statuses.
    pub fn all_statuses(&self) -> Vec<TaskStatus> {
        let statuses = self.statuses.read();
        let mut result = statuses.values().cloned().collect::<Vec<_>>();
        result.sort_by(|left, right| left.task_id.cmp(&right.task_id));
        result
    }

    /// Retrieves all attempts for a given task.
    pub fn attempts_for(&self, task_id: &TaskId) -> Vec<TaskAttempt> {
        let attempts = self.attempts.read();
        attempts.get(task_id).cloned().unwrap_or_default()
    }

    /// Set of completed task IDs.
    pub fn completed_tasks(&self) -> HashSet<TaskId> {
        let statuses = self.statuses.read();
        statuses
            .iter()
            .filter(|(_, s)| s.state == TaskState::Completed)
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Set of failed task IDs.
    pub fn failed_tasks(&self) -> HashSet<TaskId> {
        let statuses = self.statuses.read();
        statuses
            .iter()
            .filter(|(_, s)| s.state == TaskState::Failed)
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Set of cancelled task IDs.
    pub fn cancelled_tasks(&self) -> HashSet<TaskId> {
        let statuses = self.statuses.read();
        statuses
            .iter()
            .filter(|(_, s)| s.state == TaskState::Cancelled)
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Set of currently running / verifying / repairing tasks.
    pub fn in_progress_tasks(&self) -> HashSet<TaskId> {
        let statuses = self.statuses.read();
        statuses
            .iter()
            .filter(|(_, s)| s.state.is_active())
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Sum of all tokens consumed across all tasks.
    pub fn total_tokens_used(&self) -> u64 {
        let statuses = self.statuses.read();
        statuses.values().map(|s| s.tokens_used).sum()
    }
}
