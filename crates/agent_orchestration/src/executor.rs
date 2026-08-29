use crate::artifacts::Artifact;
use crate::budget::{ExecutionBudget, TaskExecutionReporter};
use crate::cancellation::CancellationToken;
use crate::ids::{CorrelationId, RunId, TaskId};
use crate::plan_graph::OrchestrationTask;
use crate::verification::VerificationResult;
use agent_client_protocol::schema::v1 as acp;
use anyhow::Result;
use futures::future::LocalBoxFuture;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Execution context passed to an executor when dispatching a task.
#[derive(Clone)]
pub struct TaskExecutionContext {
    pub task: OrchestrationTask,
    pub attempt: u32,
    pub cancellation_token: CancellationToken,
    pub correlation_id: CorrelationId,
    pub existing_session_id: Option<acp::SessionId>,
    /// Run-level identifier for correlation and telemetry.
    pub run_id: RunId,
    /// Budget limits copied from the task definition.
    pub budget: ExecutionBudget,
    /// Telemetry reporter for tool calls, tokens, phases, and progress.
    pub reporter: TaskExecutionReporter,
}

/// Output produced upon completing a task execution attempt.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskExecutionOutput {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<acp::SessionId>,
    pub output: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens_used: Option<u64>,
    #[serde(default)]
    pub artifacts: Vec<Artifact>,
}

impl TaskExecutionOutput {
    pub fn new(output: impl Into<String>) -> Self {
        Self {
            session_id: None,
            output: output.into(),
            tokens_used: None,
            artifacts: Vec::new(),
        }
    }

    pub fn with_session_id(mut self, session_id: acp::SessionId) -> Self {
        self.session_id = Some(session_id);
        self
    }

    pub fn with_tokens_used(mut self, tokens: u64) -> Self {
        self.tokens_used = Some(tokens);
        self
    }

    pub fn with_artifacts(mut self, artifacts: Vec<Artifact>) -> Self {
        self.artifacts = artifacts;
        self
    }
}

/// Abstract contract for executing, verifying, and repairing tasks.
pub trait TaskExecutor: 'static {
    /// Executes a task given its context and cancellation token.
    fn execute(
        &self,
        context: TaskExecutionContext,
    ) -> LocalBoxFuture<'static, Result<TaskExecutionOutput>>;

    /// Verifies task output against its acceptance criteria.
    fn verify(
        &self,
        task: &OrchestrationTask,
        _output: &TaskExecutionOutput,
    ) -> LocalBoxFuture<'static, Result<VerificationResult>> {
        let has_criteria = !task.acceptance_criteria.is_empty() || task.evidence_required;
        Box::pin(async move {
            if has_criteria {
                Ok(VerificationResult::fail(
                    "executor does not implement acceptance-criteria verification",
                    crate::verification::ErrorClass::FatalError,
                ))
            } else {
                Ok(VerificationResult::pass())
            }
        })
    }

    /// Attempts to repair a failed task output using verification feedback.
    fn repair(
        &self,
        _task: &OrchestrationTask,
        _feedback: &str,
        context: TaskExecutionContext,
    ) -> LocalBoxFuture<'static, Result<TaskExecutionOutput>> {
        self.execute(context)
    }
}

/// A mock executor useful for tests and simulation.
pub struct MockTaskExecutor {
    handler:
        Arc<dyn Fn(TaskExecutionContext) -> Result<TaskExecutionOutput> + Send + Sync + 'static>,
}

impl MockTaskExecutor {
    pub fn new(
        handler: impl Fn(TaskExecutionContext) -> Result<TaskExecutionOutput> + Send + Sync + 'static,
    ) -> Self {
        Self {
            handler: Arc::new(handler),
        }
    }
}

impl TaskExecutor for MockTaskExecutor {
    fn execute(
        &self,
        context: TaskExecutionContext,
    ) -> LocalBoxFuture<'static, Result<TaskExecutionOutput>> {
        let handler = self.handler.clone();
        Box::pin(async move { handler(context) })
    }
}

/// Helper for tests that need a TaskId without the full context.
pub fn task_id_of(context: &TaskExecutionContext) -> &TaskId {
    &context.task.id
}
