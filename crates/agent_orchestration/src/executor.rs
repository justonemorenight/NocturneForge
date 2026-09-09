use crate::artifacts::Artifact;
use crate::budget::TaskExecutionReporter;
use crate::cancellation::CancellationToken;
use crate::context_checkpoint::ContextCheckpoint;
use crate::control_plane::{AgentControlPlane, AgentIdentity};
use crate::ids::{CorrelationId, RunId, TaskId};
use crate::plan_graph::OrchestrationTask;
use crate::verification::VerificationResult;
use crate::worker::WorkerMetadata;
use agent_client_protocol::schema::v1 as acp;
use anyhow::Result;
use futures::future::LocalBoxFuture;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Execution context passed to an executor when dispatching a task.
#[derive(Clone)]
pub struct TaskExecutionContext {
    pub task: OrchestrationTask,
    /// Stable identity used for mailbox routing and child registration.
    pub agent_identity: AgentIdentity,
    /// Run-scoped registry and mailbox service shared by native and ACP workers.
    pub agent_control_plane: AgentControlPlane,
    /// Latest materialized editor/runtime context for this agent.
    pub context_checkpoint: Option<ContextCheckpoint>,
    pub attempt: u32,
    pub cancellation_token: CancellationToken,
    pub correlation_id: CorrelationId,
    pub existing_session_id: Option<acp::SessionId>,
    /// Metadata retained from the preceding attempt, used to reconnect a
    /// worker to the same managed workspace without trusting model output.
    pub previous_worker_metadata: Option<WorkerMetadata>,
    /// Bounded, verified outputs from completed dependency tasks. Executors
    /// pass these to workers so they can reuse prior work without recrawling.
    pub dependency_inputs: Vec<DependencyInput>,
    pub worker_config: crate::worker::AcpWorkerRuntimeConfig,
    pub background_executor: Option<gpui::BackgroundExecutor>,
    /// Run-level identifier for correlation and telemetry.
    pub run_id: RunId,
    /// Telemetry reporter for tool calls, tokens, phases, and progress.
    pub reporter: TaskExecutionReporter,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DependencyInput {
    pub task_id: TaskId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub artifacts: Vec<Artifact>,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub worker_metadata: Option<WorkerMetadata>,
}

impl TaskExecutionOutput {
    pub fn new(output: impl Into<String>) -> Self {
        Self {
            session_id: None,
            output: output.into(),
            tokens_used: None,
            artifacts: Vec::new(),
            worker_metadata: None,
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

    pub fn with_worker_metadata(mut self, metadata: WorkerMetadata) -> Self {
        self.worker_metadata = Some(metadata);
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

    /// Requests cancellation of the active worker turn and resolves once the
    /// worker has acknowledged the request or no live session is available.
    fn cancel(
        &self,
        _task: &OrchestrationTask,
        _session_id: Option<acp::SessionId>,
    ) -> LocalBoxFuture<'static, Result<()>> {
        Box::pin(async { Ok(()) })
    }

    /// Delivers a message to a worker turn that is already running. When
    /// `interrupt` is false, the worker should finish its current turn before
    /// starting the message. Executors must fail explicitly when their
    /// transport cannot deliver to an active turn.
    fn deliver_message(
        &self,
        _task: &OrchestrationTask,
        _session_id: acp::SessionId,
        _message: String,
        _interrupt: bool,
    ) -> LocalBoxFuture<'static, Result<()>> {
        Box::pin(async { anyhow::bail!("active worker messaging is not supported") })
    }

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
