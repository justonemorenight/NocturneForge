use crate::cancellation::CancellationReason;
use crate::executor::{TaskExecutionContext, TaskExecutionOutput, TaskExecutor};
use crate::ids::TaskId;
use crate::plan_graph::OrchestrationTask;
use crate::verification::VerificationResult;
use agent_client_protocol::schema::v1 as acp;
use anyhow::{Result, bail};
use chrono::{DateTime, Utc};
use collections::HashMap;
use futures::future::LocalBoxFuture;
use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;
use std::rc::Rc;
use std::time::Duration;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AcpWorkerRuntimeConfig {
    pub max_concurrency: usize,
    pub connection_timeout: Duration,
    pub cancel_grace_period: Duration,
    pub reconnect_attempts: u32,
    pub reconnect_base_delay: Duration,
    pub reconnect_max_delay: Duration,
    pub terminal_session_retention: Duration,
}

impl Default for AcpWorkerRuntimeConfig {
    fn default() -> Self {
        Self {
            max_concurrency: 2,
            connection_timeout: Duration::from_secs(20),
            cancel_grace_period: Duration::from_secs(5),
            reconnect_attempts: 3,
            reconnect_base_delay: Duration::from_millis(500),
            reconnect_max_delay: Duration::from_secs(8),
            terminal_session_retention: Duration::from_secs(15 * 60),
        }
    }
}

impl AcpWorkerRuntimeConfig {
    pub fn from_settings(settings: &agent_settings::AgentSettings) -> Self {
        const MAX_WORKERS: usize = 64;
        const MAX_TIMEOUT_SECONDS: u64 = 300;
        const MAX_RETENTION_SECONDS: u64 = 86_400;
        let mut config = Self::default();
        if let Some(limit) = settings.orchestration.acp_max_concurrency {
            config.max_concurrency = limit.clamp(1, MAX_WORKERS);
        }
        if let Some(seconds) = settings.orchestration.connection_timeout_seconds {
            config.connection_timeout = Duration::from_secs(seconds.clamp(1, MAX_TIMEOUT_SECONDS));
        }
        if let Some(seconds) = settings.orchestration.terminal_session_retention_seconds {
            config.terminal_session_retention =
                Duration::from_secs(seconds.clamp(1, MAX_RETENTION_SECONDS));
        }
        config
    }

    pub fn reconnect_delay(&self, attempt: u32) -> Duration {
        self.reconnect_base_delay
            .saturating_mul(4_u32.saturating_pow(attempt.saturating_sub(1).min(16)))
            .min(self.reconnect_max_delay)
    }
}

async fn connect_with_timeout<T>(
    future: impl std::future::Future<Output = Result<T>>,
    context: &TaskExecutionContext,
) -> Result<T> {
    let executor = context.background_executor.as_ref().ok_or_else(|| {
        anyhow::anyhow!("ACP worker connection timeout requires a background executor")
    })?;
    let timer = executor.timer(context.worker_config.connection_timeout);
    futures::pin_mut!(future, timer);
    match futures::future::select(future, timer).await {
        futures::future::Either::Left((result, _)) => result,
        futures::future::Either::Right(_) => bail!("ACP worker connection timed out"),
    }
}

async fn execute_external_worker(
    mut worker: Box<dyn WorkerHandle>,
    context: TaskExecutionContext,
) -> Result<TaskExecutionOutput> {
    let metadata = worker.metadata().clone();
    context
        .reporter
        .report_worker_started(worker.session_id().cloned(), metadata.clone());
    let result = worker.execute(context.clone()).await;
    if result.is_ok() {
        let close = worker.close();
        if let Some(executor) = &context.background_executor {
            let timeout = executor.timer(context.worker_config.cancel_grace_period);
            futures::pin_mut!(close, timeout);
            match futures::future::select(close, timeout).await {
                futures::future::Either::Left((Err(error), _)) => {
                    log::warn!("ACP worker session close failed: {error}")
                }
                futures::future::Either::Right(_) => {
                    log::warn!("ACP worker session close timed out; stopping its process")
                }
                _ => {}
            }
        }
    }
    drop(worker);
    let mut output = result?;
    output.worker_metadata.get_or_insert(metadata);
    if context.task.workspace_policy.isolation == WorkspaceIsolation::DedicatedWorktree {
        let path = output
            .worker_metadata
            .as_ref()
            .and_then(|metadata| metadata.worktree_path.as_ref())
            .ok_or_else(|| {
                anyhow::anyhow!("isolated worker did not provide its managed workspace")
            })?;
        let worktree = crate::IsolatedWorktree::reopen_managed(
            path.into(),
            &context.run_id,
            &context.task.id,
            context.attempt,
        )
        .await?;
        worktree
            .validate_scope(context.task.workspace_policy.allowed_subpaths.as_deref())
            .await?;
        let patch = worktree.collect_diff().await?;
        output.artifacts.push(crate::Artifact::new(
            context.task.id.clone(),
            "Worker patch",
            crate::ArtifactKind::Patch,
            patch,
        ));
    }
    Ok(output)
}

#[derive(Clone)]
struct AcpConcurrencyLimiter {
    available: async_channel::Receiver<()>,
    returned: async_channel::Sender<()>,
}

impl AcpConcurrencyLimiter {
    fn new(max_concurrency: usize) -> Self {
        let max_concurrency = max_concurrency.max(1);
        let (returned, available) = async_channel::bounded(max_concurrency);
        for _ in 0..max_concurrency {
            if returned.try_send(()).is_err() {
                log::error!("failed to initialize ACP worker concurrency limiter");
                break;
            }
        }
        Self {
            available,
            returned,
        }
    }

    async fn acquire(&self) -> Result<AcpConcurrencyPermit> {
        self.available
            .recv()
            .await
            .map_err(|error| anyhow::anyhow!("ACP worker concurrency limiter closed: {error}"))?;
        Ok(AcpConcurrencyPermit {
            returned: self.returned.clone(),
        })
    }
}

struct AcpConcurrencyPermit {
    returned: async_channel::Sender<()>,
}

impl Drop for AcpConcurrencyPermit {
    fn drop(&mut self) {
        if let Err(error) = self.returned.try_send(()) {
            log::error!("failed to return ACP worker concurrency permit: {error}");
        }
    }
}

/// Target worker environment for task execution.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, JsonSchema)]
pub enum WorkerTarget {
    #[default]
    Native,
    Acp {
        agent_id: String,
    },
}

impl WorkerTarget {
    pub fn native() -> Self {
        Self::Native
    }

    pub fn acp(agent_id: impl Into<String>) -> Self {
        Self::Acp {
            agent_id: agent_id.into(),
        }
    }

    pub fn from_identifier(id: &str) -> Self {
        let trimmed = id.trim();
        if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("native") {
            Self::Native
        } else {
            Self::Acp {
                agent_id: trimmed.to_string(),
            }
        }
    }

    pub fn is_native(&self) -> bool {
        matches!(self, Self::Native)
    }

    pub fn is_acp(&self) -> bool {
        matches!(self, Self::Acp { .. })
    }

    pub fn agent_id(&self) -> Option<&str> {
        match self {
            Self::Native => None,
            Self::Acp { agent_id } => Some(agent_id.as_str()),
        }
    }

    pub fn display_name(&self) -> &str {
        match self {
            Self::Native => "Native",
            Self::Acp { agent_id } => agent_id.as_str(),
        }
    }
}

impl fmt::Display for WorkerTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Native => write!(f, "native"),
            Self::Acp { agent_id } => write!(f, "acp:{}", agent_id),
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum WorkerTargetRepr {
    Native,
    Acp { agent_id: String },
}

impl Serialize for WorkerTarget {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Native => WorkerTargetRepr::Native.serialize(serializer),
            Self::Acp { agent_id } => WorkerTargetRepr::Acp {
                agent_id: agent_id.clone(),
            }
            .serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for WorkerTarget {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum TargetHelper {
            String(String),
            Structured(WorkerTargetRepr),
        }

        match TargetHelper::deserialize(deserializer)? {
            TargetHelper::String(s) => Ok(WorkerTarget::from_identifier(&s)),
            TargetHelper::Structured(WorkerTargetRepr::Native) => Ok(WorkerTarget::Native),
            TargetHelper::Structured(WorkerTargetRepr::Acp { agent_id }) => {
                Ok(WorkerTarget::Acp { agent_id })
            }
        }
    }
}

/// Workspace isolation strategy for a worker task.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "snake_case")]
pub enum WorkspaceIsolation {
    #[default]
    SharedParent,
    DedicatedWorktree,
}

/// Policy governing the workspace and filesystem access for a worker.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct WorkspacePolicy {
    #[serde(default)]
    pub isolation: WorkspaceIsolation,
    #[serde(default)]
    pub read_only: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allowed_subpaths: Option<Vec<String>>,
}

impl WorkspacePolicy {
    pub fn shared_parent() -> Self {
        Self {
            isolation: WorkspaceIsolation::SharedParent,
            read_only: false,
            allowed_subpaths: None,
        }
    }

    pub fn isolated_worktree() -> Self {
        Self {
            isolation: WorkspaceIsolation::DedicatedWorktree,
            read_only: false,
            allowed_subpaths: None,
        }
    }

    pub fn read_only() -> Self {
        Self {
            isolation: WorkspaceIsolation::SharedParent,
            read_only: true,
            allowed_subpaths: None,
        }
    }
}

impl Default for WorkspacePolicy {
    fn default() -> Self {
        Self::shared_parent()
    }
}

/// Snapshot of capabilities supported by a specific worker target.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct CapabilitySnapshot {
    #[serde(default)]
    pub can_resume: bool,
    #[serde(default)]
    pub can_load_session: bool,
    #[serde(default)]
    pub can_cancel: bool,
    #[serde(default)]
    pub can_stream_tokens: bool,
    #[serde(default)]
    pub can_report_usage: bool,
    #[serde(default)]
    pub can_enforce_read_only: bool,
    #[serde(default)]
    pub can_select_model: bool,
    #[serde(default)]
    pub can_select_mode: bool,
    #[serde(default)]
    pub supports_worktree_isolation: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub supported_models: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub supported_modes: Vec<String>,
}

impl Default for CapabilitySnapshot {
    fn default() -> Self {
        Self::native_default()
    }
}

impl CapabilitySnapshot {
    pub fn native_default() -> Self {
        Self {
            can_resume: true,
            can_load_session: false,
            can_cancel: true,
            can_stream_tokens: true,
            can_report_usage: true,
            can_enforce_read_only: true,
            can_select_model: true,
            can_select_mode: true,
            supports_worktree_isolation: true,
            supported_models: Vec::new(),
            supported_modes: vec![
                "explorer".to_string(),
                "flow-reader".to_string(),
                "coding-worker".to_string(),
            ],
        }
    }

    pub fn omp_default() -> Self {
        Self {
            can_resume: false,
            can_load_session: false,
            can_cancel: true,
            can_stream_tokens: true,
            can_report_usage: true,
            can_enforce_read_only: false,
            can_select_model: true,
            can_select_mode: true,
            supports_worktree_isolation: true,
            supported_models: Vec::new(),
            supported_modes: Vec::new(),
        }
    }
}

/// Serializable metadata describing a worker running or that ran a task attempt.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerMetadata {
    pub target: WorkerTarget,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_display_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_provider: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_provider_display_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_effort: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fallback_from_model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fallback_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    #[serde(default)]
    pub workspace_policy: WorkspacePolicy,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub worktree_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub baseline_commit: Option<String>,
    #[serde(default)]
    pub capabilities: CapabilitySnapshot,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_activity_at: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nested_agent_count: Option<u64>,
}

impl WorkerMetadata {
    pub fn new(target: WorkerTarget) -> Self {
        let capabilities = match &target {
            WorkerTarget::Native => CapabilitySnapshot::native_default(),
            WorkerTarget::Acp { .. } => CapabilitySnapshot {
                can_resume: false,
                can_load_session: false,
                can_cancel: false,
                can_stream_tokens: false,
                can_report_usage: false,
                can_enforce_read_only: false,
                can_select_model: false,
                can_select_mode: false,
                supports_worktree_isolation: false,
                supported_models: Vec::new(),
                supported_modes: Vec::new(),
            },
        };
        Self {
            agent_id: target.agent_id().map(ToString::to_string),
            display_name: Some(target.display_name().to_string()),
            target,
            model: None,
            model_display_name: None,
            model_provider: None,
            model_provider_display_name: None,
            thinking_effort: None,
            fallback_from_model: None,
            fallback_reason: None,
            mode: None,
            version: None,
            workspace_policy: WorkspacePolicy::default(),
            worktree_path: None,
            baseline_commit: None,
            capabilities,
            last_activity_at: Some(Utc::now()),
            nested_agent_count: None,
        }
    }
}

/// Structured explanation of why a task or run is waiting.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "reason", rename_all = "snake_case")]
pub enum StructuredWaitReason {
    AwaitingApproval,
    AwaitingDependency {
        task_ids: Vec<TaskId>,
    },
    AwaitingApply {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        patch_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        worktree_path: Option<String>,
    },
    AwaitingRetryBackoff {
        next_attempt: u32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        delay_ms: Option<u64>,
    },
    AwaitingUserInput {
        question: String,
    },
    AwaitingWorkerReconnect {
        worker: WorkerTarget,
        attempt: u32,
    },
    ApplyConflict {
        error: String,
        worktree_path: String,
    },
    WorktreeVerificationFailed {
        error: String,
        worktree_path: String,
    },
    PostApplyVerificationFailed {
        error: String,
        worktree_path: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        rollback_error: Option<String>,
    },
    AwaitingProcessExit {
        process_id: u32,
    },
    Custom {
        description: String,
    },
}

impl StructuredWaitReason {
    pub fn description(&self) -> String {
        match self {
            Self::AwaitingApproval => "waiting for plan approval".to_string(),
            Self::AwaitingDependency { task_ids } => {
                let ids = task_ids
                    .iter()
                    .map(|id| id.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("waiting for dependency tasks: [{}]", ids)
            }
            Self::AwaitingApply { worktree_path, .. } => {
                if let Some(path) = worktree_path {
                    format!("awaiting review and apply from worktree '{}'", path)
                } else {
                    "awaiting review and apply".to_string()
                }
            }
            Self::AwaitingRetryBackoff {
                next_attempt,
                delay_ms,
            } => {
                if let Some(delay) = delay_ms {
                    format!(
                        "waiting for retry backoff before attempt {} ({}ms)",
                        next_attempt, delay
                    )
                } else {
                    format!("waiting for retry backoff before attempt {}", next_attempt)
                }
            }
            Self::AwaitingUserInput { question } => {
                format!("waiting for user input: {}", question)
            }
            Self::AwaitingWorkerReconnect { worker, attempt } => {
                format!(
                    "waiting for worker '{}' to reconnect (attempt {})",
                    worker, attempt
                )
            }
            Self::ApplyConflict {
                error,
                worktree_path,
            } => format!(
                "worker patch conflicts with the parent checkout: {error}; worktree retained at '{worktree_path}'"
            ),
            Self::WorktreeVerificationFailed {
                error,
                worktree_path,
            } => format!(
                "worktree verification failed before apply: {error}; worktree retained at '{worktree_path}'"
            ),
            Self::PostApplyVerificationFailed {
                error,
                worktree_path,
                rollback_error,
            } => {
                if let Some(rollback_error) = rollback_error {
                    format!(
                        "parent post-apply verification failed: {error}; rollback failed: {rollback_error}; inspect the parent checkout before continuing; worktree retained at '{worktree_path}'"
                    )
                } else {
                    format!(
                        "parent post-apply verification failed: {error}; parent checkout was rolled back and worktree retained at '{worktree_path}'"
                    )
                }
            }
            Self::AwaitingProcessExit { process_id } => {
                format!("waiting for process {} to exit", process_id)
            }
            Self::Custom { description } => description.clone(),
        }
    }
}

/// Interface to a live task worker execution instance.
pub trait WorkerHandle: 'static {
    fn target(&self) -> &WorkerTarget;
    fn session_id(&self) -> Option<&acp::SessionId>;
    fn metadata(&self) -> &WorkerMetadata;
    fn execute(
        &mut self,
        context: TaskExecutionContext,
    ) -> LocalBoxFuture<'static, Result<TaskExecutionOutput>>;
    fn cancel(&mut self, reason: CancellationReason) -> LocalBoxFuture<'static, Result<()>>;
    fn close(&mut self) -> LocalBoxFuture<'static, Result<()>>;
    fn resume(
        &mut self,
        session_id: &acp::SessionId,
        context: TaskExecutionContext,
    ) -> LocalBoxFuture<'static, Result<TaskExecutionOutput>>;
}

/// Host providing worker execution capabilities for a particular target.
pub trait WorkerHost: 'static {
    fn target(&self) -> WorkerTarget;
    fn capabilities(&self) -> CapabilitySnapshot;
    fn create_worker(
        &self,
        task: &OrchestrationTask,
        context: &TaskExecutionContext,
    ) -> LocalBoxFuture<'static, Result<Box<dyn WorkerHandle>>>;
    fn resume_worker(
        &self,
        session_id: &acp::SessionId,
        task: &OrchestrationTask,
        context: &TaskExecutionContext,
    ) -> LocalBoxFuture<'static, Result<Box<dyn WorkerHandle>>>;
}

#[derive(Clone, Default)]
pub struct WorkerHostRegistry {
    hosts: HashMap<WorkerTarget, Rc<dyn WorkerHost>>,
}

impl WorkerHostRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_host(mut self, target: WorkerTarget, host: Rc<dyn WorkerHost>) -> Self {
        self.register(target, host);
        self
    }

    pub fn register(&mut self, target: WorkerTarget, host: Rc<dyn WorkerHost>) {
        self.hosts.insert(target, host);
    }

    pub fn get(&self, target: &WorkerTarget) -> Option<Rc<dyn WorkerHost>> {
        self.hosts.get(target).cloned()
    }
}

struct WorkerTaskValidator<'a> {
    host_registry: &'a WorkerHostRegistry,
    acp_delegation_enabled: bool,
}

impl<'a> WorkerTaskValidator<'a> {
    fn new(host_registry: &'a WorkerHostRegistry, acp_delegation_enabled: bool) -> Self {
        Self {
            host_registry,
            acp_delegation_enabled,
        }
    }

    fn validate(&self, task: &OrchestrationTask) -> Result<()> {
        match &task.target {
            WorkerTarget::Native => validate_native_task_parameters(task),
            WorkerTarget::Acp { agent_id } => self.validate_acp(task, agent_id),
        }
    }

    fn validate_acp(&self, task: &OrchestrationTask, agent_id: &str) -> Result<()> {
        if !self.acp_delegation_enabled {
            bail!(
                "ACP delegation is currently disabled (feature flag is off); cannot delegate to agent '{}'",
                agent_id
            );
        }

        let host = self.host_registry.get(&task.target).ok_or_else(|| {
            anyhow::anyhow!("worker agent '{}' is not configured or available", agent_id)
        })?;
        let capabilities = host.capabilities();
        validate_acp_model(task.model_override.as_deref(), agent_id, &capabilities)?;
        validate_acp_mode(task.mode.as_deref(), agent_id, &capabilities)?;
        validate_acp_workspace(task, agent_id, &capabilities)
    }
}

/// Worker broker routing tasks to Native or ACP worker hosts based on target and capabilities.
struct WorkerBrokerConfiguration {
    acp_delegation_enabled: bool,
    acp_runtime: AcpWorkerRuntimeConfig,
}

pub struct WorkerBroker {
    host_registry: WorkerHostRegistry,
    configuration: WorkerBrokerConfiguration,
    native_executor: Option<Rc<dyn TaskExecutor>>,
    acp_concurrency_limiter: AcpConcurrencyLimiter,
}

impl WorkerBroker {
    pub fn new(feature_flag_acp_delegation: bool) -> Self {
        let acp_runtime_config = AcpWorkerRuntimeConfig::default();
        Self {
            host_registry: WorkerHostRegistry::new(),
            configuration: WorkerBrokerConfiguration {
                acp_delegation_enabled: feature_flag_acp_delegation,
                acp_runtime: acp_runtime_config.clone(),
            },
            native_executor: None,
            acp_concurrency_limiter: AcpConcurrencyLimiter::new(acp_runtime_config.max_concurrency),
        }
    }

    pub fn with_acp_runtime_config(mut self, config: AcpWorkerRuntimeConfig) -> Self {
        self.acp_concurrency_limiter = AcpConcurrencyLimiter::new(config.max_concurrency);
        self.configuration.acp_runtime = config;
        self
    }

    pub fn acp_runtime_config(&self) -> &AcpWorkerRuntimeConfig {
        &self.configuration.acp_runtime
    }

    pub fn with_host_registry(mut self, host_registry: WorkerHostRegistry) -> Self {
        self.host_registry = host_registry;
        self
    }

    pub fn with_native_executor(mut self, executor: Rc<dyn TaskExecutor>) -> Self {
        self.native_executor = Some(executor);
        self
    }

    pub fn set_feature_flag_acp_delegation(&mut self, enabled: bool) {
        self.configuration.acp_delegation_enabled = enabled;
    }

    pub fn validate_task_parameters(&self, task: &OrchestrationTask) -> Result<()> {
        WorkerTaskValidator::new(
            &self.host_registry,
            self.configuration.acp_delegation_enabled,
        )
        .validate(task)
    }
}

fn validate_native_task_parameters(task: &OrchestrationTask) -> Result<()> {
    if task.workspace_policy.isolation == WorkspaceIsolation::DedicatedWorktree {
        bail!("Native worker is not connected to a managed isolated worktree");
    }
    if task.workspace_policy.read_only
        && !matches!(
            task.native_role.as_deref(),
            Some("explorer" | "flow-reader")
        )
    {
        bail!("Native read-only workspace requires native_role `explorer` or `flow-reader`");
    }
    if task.mode.is_some() {
        bail!("mode override is not supported for native agent; use native_role for a Native role");
    }
    validate_native_model(task.model_override.as_deref(), "model")?;
    validate_native_model(task.fallback_model_override.as_deref(), "fallback model")
}

fn validate_native_model(model: Option<&str>, field_name: &str) -> Result<()> {
    if let Some(model) = model
        && !is_valid_native_model_identifier(model)
    {
        bail!("invalid native {field_name} identifier '{model}'; expected provider/model");
    }
    Ok(())
}

fn validate_acp_model(
    model: Option<&str>,
    agent_id: &str,
    capabilities: &CapabilitySnapshot,
) -> Result<()> {
    let Some(model) = model else {
        return Ok(());
    };
    if !capabilities.can_select_model {
        bail!("worker '{agent_id}' does not support selecting a model");
    }
    if !capabilities.supported_models.is_empty()
        && !capabilities
            .supported_models
            .iter()
            .any(|supported| supported == model)
    {
        bail!(
            "model '{model}' is not supported by worker '{agent_id}'; supported models: {:?}",
            capabilities.supported_models
        );
    }
    Ok(())
}

fn validate_acp_mode(
    mode: Option<&str>,
    agent_id: &str,
    capabilities: &CapabilitySnapshot,
) -> Result<()> {
    let Some(mode) = mode else {
        return Ok(());
    };
    if !capabilities.can_select_mode {
        bail!("worker '{agent_id}' does not support selecting a session mode");
    }
    if !capabilities.supported_modes.is_empty()
        && !capabilities
            .supported_modes
            .iter()
            .any(|supported| supported == mode)
    {
        bail!(
            "mode '{mode}' is not supported by worker '{agent_id}'; supported modes: {:?}",
            capabilities.supported_modes
        );
    }
    Ok(())
}

fn validate_acp_workspace(
    task: &OrchestrationTask,
    agent_id: &str,
    capabilities: &CapabilitySnapshot,
) -> Result<()> {
    if task.workspace_policy.read_only && !capabilities.can_enforce_read_only {
        bail!(
            "worker '{agent_id}' cannot enforce read-only execution; read-only tasks cannot run without proper enforcement"
        );
    }
    if task.workspace_policy.read_only
        && task.workspace_policy.isolation != WorkspaceIsolation::SharedParent
    {
        bail!("worker '{agent_id}' read-only tasks must use the shared parent workspace");
    }
    if !task.workspace_policy.read_only
        && task.workspace_policy.isolation != WorkspaceIsolation::DedicatedWorktree
    {
        bail!("worker '{agent_id}' write tasks require a dedicated managed worktree");
    }
    if task.workspace_policy.isolation == WorkspaceIsolation::DedicatedWorktree
        && !capabilities.supports_worktree_isolation
    {
        bail!("worker '{agent_id}' is not connected to a managed isolated worktree");
    }
    Ok(())
}

fn is_valid_native_model_identifier(model: &str) -> bool {
    model
        .split_once('/')
        .is_some_and(|(provider, model)| !provider.is_empty() && !model.is_empty())
}

impl TaskExecutor for WorkerBroker {
    fn execute(
        &self,
        context: TaskExecutionContext,
    ) -> LocalBoxFuture<'static, Result<TaskExecutionOutput>> {
        let task = context.task.clone();
        if let Err(err) = self.validate_task_parameters(&task) {
            return Box::pin(async move { Err(err) });
        }

        match &task.target {
            WorkerTarget::Native => {
                if let Some(executor) = self.native_executor.clone() {
                    return executor.execute(context);
                }
                if let Some(host) = self.host_registry.get(&task.target) {
                    return Box::pin(async move {
                        let mut worker = host.create_worker(&task, &context).await?;
                        let metadata = worker.metadata().clone();
                        context
                            .reporter
                            .report_worker_started(worker.session_id().cloned(), metadata.clone());
                        worker.execute(context).await.map(|mut output| {
                            output.worker_metadata.get_or_insert(metadata);
                            output
                        })
                    });
                }
                Box::pin(async move {
                    bail!("no native task executor or host registered in worker broker");
                })
            }
            WorkerTarget::Acp { .. } => {
                let host = match self.host_registry.get(&task.target) {
                    Some(h) => h,
                    None => {
                        let target = task.target.clone();
                        return Box::pin(async move {
                            bail!("no worker host registered for target '{:?}'", target);
                        });
                    }
                };
                let limiter = self.acp_concurrency_limiter.clone();

                Box::pin(async move {
                    let _concurrency_permit = limiter.acquire().await?;
                    let worker = if let Some(session_id) = &context.existing_session_id {
                        let caps = host.capabilities();
                        if !caps.can_resume && !caps.can_load_session {
                            bail!(
                                "worker target '{:?}' does not support resuming sessions",
                                task.target
                            );
                        }
                        connect_with_timeout(
                            host.resume_worker(session_id, &task, &context),
                            &context,
                        )
                        .await?
                    } else {
                        connect_with_timeout(host.create_worker(&task, &context), &context).await?
                    };

                    execute_external_worker(worker, context).await
                })
            }
        }
    }

    fn cancel(
        &self,
        task: &OrchestrationTask,
        session_id: Option<acp::SessionId>,
    ) -> LocalBoxFuture<'static, Result<()>> {
        if task.target.is_native()
            && let Some(executor) = self.native_executor.clone()
        {
            return executor.cancel(task, session_id);
        }
        Box::pin(async { Ok(()) })
    }

    fn deliver_message(
        &self,
        task: &OrchestrationTask,
        session_id: acp::SessionId,
        message: String,
        interrupt: bool,
    ) -> LocalBoxFuture<'static, Result<()>> {
        if task.target.is_native()
            && let Some(executor) = self.native_executor.clone()
        {
            return executor.deliver_message(task, session_id, message, interrupt);
        }
        let target = task.target.clone();
        Box::pin(
            async move { bail!("active messaging is not supported for worker target '{target}'") },
        )
    }

    fn verify(
        &self,
        task: &OrchestrationTask,
        output: &TaskExecutionOutput,
    ) -> LocalBoxFuture<'static, Result<VerificationResult>> {
        if task.target.is_native()
            && let Some(executor) = self.native_executor.clone()
        {
            return executor.verify(task, output);
        }
        let task = task.clone();
        let output = output.clone();
        Box::pin(async move {
            let runner = crate::verification::VerificationRunner;
            Ok(runner.verify(&task, &output.output))
        })
    }

    fn repair(
        &self,
        task: &OrchestrationTask,
        feedback: &str,
        context: TaskExecutionContext,
    ) -> LocalBoxFuture<'static, Result<TaskExecutionOutput>> {
        if task.target.is_native() {
            if let Some(executor) = self.native_executor.clone() {
                return executor.repair(task, feedback, context);
            }
            let task_id = task.id.clone();
            return Box::pin(async move {
                bail!(
                    "no native executor is registered to repair task '{}'",
                    task_id
                )
            });
        }

        if let Err(error) = self.validate_task_parameters(task) {
            return Box::pin(async move { Err(error) });
        }
        let Some(host) = self.host_registry.get(&task.target) else {
            let target = task.target.clone();
            return Box::pin(async move {
                bail!("no worker host registered for repair target '{}'", target)
            });
        };
        let task = task.clone();
        let feedback = feedback.to_string();
        let limiter = self.acp_concurrency_limiter.clone();
        Box::pin(async move {
            let _concurrency_permit = limiter.acquire().await?;
            let session_id = context.existing_session_id.clone().ok_or_else(|| {
                anyhow::anyhow!("cannot repair ACP task '{}' without a session", task.id)
            })?;
            let mut repair_task = task;
            repair_task.description = format!(
                "{}\n\nVerification feedback:\n{}\n\nAddress the feedback and return a corrected result with a new structured verification claim.",
                repair_task.description, feedback
            );
            let mut repair_context = context;
            repair_context.task = repair_task.clone();
            let worker = connect_with_timeout(
                host.resume_worker(&session_id, &repair_task, &repair_context),
                &repair_context,
            )
            .await?;
            execute_external_worker(worker, repair_context).await
        })
    }
}
