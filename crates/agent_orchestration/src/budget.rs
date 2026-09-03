use crate::events::{RuntimeEvent, RuntimeEventStream};
use crate::ids::{RunId, TaskId};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Hard limits for a single task execution. Every field is optional: an
/// absent limit is not enforced. The scheduler copies these from the task
/// definition into the execution context so the executor and scheduler share
/// one view of the task's budgets.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionBudget {
    /// Cumulative provider-reported token ceiling across all attempts, including
    /// input, output, cache-read, and cache-creation tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_budget: Option<u64>,
    /// Ceiling on tool calls reported by the task executor.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_budget: Option<u64>,
    /// Wall-clock ceiling in seconds for a single attempt.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_budget_secs: Option<u64>,
}

impl ExecutionBudget {
    pub fn unlimited() -> Self {
        Self::default()
    }
}

/// Typed reason a task was stopped by budget enforcement. Distinct from
/// `ErrorClass` because budget failures must never be silently retried or
/// converted into a successful completion.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BudgetExceeded {
    TokenBudgetExceeded { budget: u64, used: u64 },
    ToolCallBudgetExceeded { budget: u64, used: u64 },
}

impl BudgetExceeded {
    pub fn error_class(&self) -> crate::verification::ErrorClass {
        crate::verification::ErrorClass::TokenBudgetExceeded
    }

    pub fn message(&self) -> String {
        match self {
            Self::TokenBudgetExceeded { budget, used } => {
                format!("token budget of {budget} was exceeded (used {used})")
            }
            Self::ToolCallBudgetExceeded { budget, used } => {
                format!("tool call budget of {budget} was exceeded (used {used})")
            }
        }
    }
}

impl std::fmt::Display for BudgetExceeded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message())
    }
}

impl std::error::Error for BudgetExceeded {}

/// Live usage counters for one task, shared between the executor (which
/// reports usage as it happens) and the scheduler (which enforces limits).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BudgetUsage {
    pub tokens_used: u64,
    pub tool_calls: u64,
}

/// Rolling state of a task's budget: current usage plus the reason it was
/// stopped, if any. Persisted in task snapshots so recovery can distinguish
/// budget stops from other failures.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskBudgetState {
    pub tokens_used: u64,
    pub tool_calls_used: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stopped_reason: Option<BudgetExceeded>,
}

impl TaskBudgetState {
    pub fn is_stopped(&self) -> bool {
        self.stopped_reason.is_some()
    }
}

/// Telemetry reporter handed to an executor for the duration of one task
/// attempt. Every call updates shared usage state and emits a bounded event
/// so Activity Center can render live progress without polling.
#[derive(Clone)]
pub struct TaskExecutionReporter {
    run_id: RunId,
    task_id: TaskId,
    model_id: Option<Arc<str>>,
    budget: ExecutionBudget,
    usage: Arc<RwLock<BudgetUsage>>,
    phase: Arc<RwLock<Option<Arc<str>>>>,
    current_tool: Arc<RwLock<Option<Arc<str>>>>,
    message: Arc<RwLock<Option<Arc<str>>>>,
    progress_percent: Arc<RwLock<Option<f32>>>,
    event_stream: Option<RuntimeEventStream>,
}

const MAX_TELEMETRY_MESSAGE_BYTES: usize = 2_048;
const MAX_TOOL_NAME_BYTES: usize = 512;

impl TaskExecutionReporter {
    pub fn new(
        run_id: RunId,
        task_id: TaskId,
        model_id: Option<Arc<str>>,
        budget: ExecutionBudget,
        event_stream: Option<RuntimeEventStream>,
    ) -> Self {
        Self {
            run_id,
            task_id,
            model_id,
            budget,
            usage: Arc::new(RwLock::new(BudgetUsage::default())),
            phase: Arc::new(RwLock::new(None)),
            current_tool: Arc::new(RwLock::new(None)),
            message: Arc::new(RwLock::new(None)),
            progress_percent: Arc::new(RwLock::new(None)),
            event_stream,
        }
    }

    pub fn run_id(&self) -> &RunId {
        &self.run_id
    }

    pub fn task_id(&self) -> &TaskId {
        &self.task_id
    }

    pub fn model_id(&self) -> Option<&str> {
        self.model_id.as_deref()
    }

    pub fn usage(&self) -> BudgetUsage {
        *self.usage.read()
    }

    pub fn phase(&self) -> Option<String> {
        self.phase.read().as_ref().map(|phase| phase.to_string())
    }

    pub fn current_tool(&self) -> Option<String> {
        self.current_tool
            .read()
            .as_ref()
            .map(|tool| tool.to_string())
    }

    pub fn progress_message(&self) -> Option<String> {
        self.message
            .read()
            .as_ref()
            .map(|message| message.to_string())
    }

    pub fn progress_percent(&self) -> Option<f32> {
        *self.progress_percent.read()
    }

    /// Reports a tool call about to start. Returns `Ok(())` when the call may
    /// proceed, or `Err` when the tool-call budget is already exhausted.
    pub fn report_tool_call_started(&self, tool: impl Into<String>) -> Result<(), BudgetExceeded> {
        let tool = crate::artifacts::truncate_text(tool.into(), MAX_TOOL_NAME_BYTES);
        {
            let mut usage = self.usage.write();
            usage.tool_calls = usage.tool_calls.saturating_add(1);
            let result = self.check_usage(&usage);
            drop(usage);
            if let Err(exceeded) = result {
                self.emit_budget_update();
                return Err(exceeded);
            }
        }
        self.current_tool.write().replace(tool.clone().into());
        self.emit(RuntimeEvent::TaskToolCallStarted {
            run_id: self.run_id.clone(),
            task_id: self.task_id.clone(),
            tool,
        });
        Ok(())
    }

    /// Reports that the current tool call finished.
    pub fn report_tool_call_finished(&self) {
        self.current_tool.write().take();
        self.emit(RuntimeEvent::TaskToolCallFinished {
            run_id: self.run_id.clone(),
            task_id: self.task_id.clone(),
        });
    }

    /// Reports additional token usage discovered mid-execution (e.g. after a
    /// subagent turn). Returns an error when the token budget is exceeded.
    pub fn report_tokens(&self, tokens: u64) -> Result<(), BudgetExceeded> {
        {
            let mut usage = self.usage.write();
            usage.tokens_used = usage.tokens_used.saturating_add(tokens);
            let result = self.check_usage(&usage);
            drop(usage);
            if let Err(exceeded) = result {
                self.emit_budget_update();
                return Err(exceeded);
            }
        }
        self.emit_budget_update();
        Ok(())
    }

    /// Reports a cumulative token counter (as exposed by a child session)
    /// without charging the same tokens again on subsequent turns.
    pub fn report_total_tokens(&self, total_tokens: u64) -> Result<(), BudgetExceeded> {
        {
            let mut usage = self.usage.write();
            usage.tokens_used = usage.tokens_used.max(total_tokens);
            let result = self.check_usage(&usage);
            drop(usage);
            if let Err(exceeded) = result {
                self.emit_budget_update();
                return Err(exceeded);
            }
        }
        self.emit_budget_update();
        Ok(())
    }

    /// Returns an error when the current usage exceeds the enforced budget.
    /// Executors can call this between tool calls to stop cooperatively.
    pub fn check_budget(&self) -> Result<(), BudgetExceeded> {
        let usage = self.usage.read();
        self.check_usage(&usage)
    }

    fn check_usage(&self, usage: &BudgetUsage) -> Result<(), BudgetExceeded> {
        if let Some(budget) = self.budget.token_budget
            && usage.tokens_used > budget
        {
            return Err(BudgetExceeded::TokenBudgetExceeded {
                budget,
                used: usage.tokens_used,
            });
        }
        if let Some(budget) = self.budget.tool_call_budget
            && usage.tool_calls > budget
        {
            return Err(BudgetExceeded::ToolCallBudgetExceeded {
                budget,
                used: usage.tool_calls,
            });
        }
        Ok(())
    }

    /// The budget this reporter enforces, for display and persistence.
    pub fn budget(&self) -> &ExecutionBudget {
        &self.budget
    }

    pub fn set_phase(&self, phase: &str) {
        let bounded = crate::artifacts::truncate_text(phase.to_string(), 256);
        self.phase.write().replace(bounded.clone().into());
        self.emit(RuntimeEvent::TaskPhaseChanged {
            run_id: self.run_id.clone(),
            task_id: self.task_id.clone(),
            phase: bounded,
        });
    }

    /// Reports a short progress message with an optional completion percent.
    pub fn report_progress(&self, message: impl Into<String>, percent: Option<f32>) {
        let bounded = crate::artifacts::truncate_text(message.into(), MAX_TELEMETRY_MESSAGE_BYTES);
        self.message.write().replace(bounded.clone().into());
        if let Some(percent) = percent {
            self.progress_percent
                .write()
                .replace(percent.clamp(0.0, 100.0));
        }
        let usage = self.usage.read();
        self.emit(RuntimeEvent::TaskProgress {
            run_id: self.run_id.clone(),
            task_id: self.task_id.clone(),
            message: bounded,
            tokens_used: Some(usage.tokens_used),
            percent: percent.unwrap_or_else(|| self.progress_percent.read().unwrap_or(0.0)),
        });
    }

    /// Records the model assigned to this task, if not already set.
    pub fn report_model_assigned(&self, model_id: impl Into<Arc<str>>) {
        let model_id = model_id.into();
        self.emit(RuntimeEvent::TaskModelAssigned {
            run_id: self.run_id.clone(),
            task_id: self.task_id.clone(),
            model_id: model_id.to_string(),
        });
    }

    fn emit_budget_update(&self) {
        let usage = self.usage.read();
        self.emit(RuntimeEvent::TaskBudgetUpdated {
            run_id: self.run_id.clone(),
            task_id: self.task_id.clone(),
            tokens_used: usage.tokens_used,
            tool_calls_used: usage.tool_calls,
        });
    }

    fn emit(&self, event: RuntimeEvent) {
        if let Some(stream) = &self.event_stream {
            stream.emit(event);
        }
    }
}
