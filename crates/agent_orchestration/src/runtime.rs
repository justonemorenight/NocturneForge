use crate::artifacts::ArtifactStore;
use crate::cancellation::{CancellationReason, CancellationTree};
use crate::control_plane::{AgentControlPlane, AgentControlPlaneConfig};
use crate::events::{EventSubscription, RuntimeEvent, RuntimeEventStream};
use crate::executor::TaskExecutor;
use crate::ids::{RunId, TaskId};
use crate::persistence::PersistedRun;
use crate::plan_graph::{OrchestrationPlan, PlanGraph};
use crate::scheduler::{RuntimeControl, Scheduler, SchedulerConfig};
use crate::state::{RunState, TaskState, TaskStatus};
use crate::task_registry::TaskRegistry;
use crate::worktree_isolation::IsolatedWorktree;
use agent_settings::AgentExecutionPolicy;
use anyhow::{Context as _, Result};
use chrono::Utc;
use parking_lot::Mutex;
use std::collections::HashSet;
use std::rc::Rc;
use std::sync::Arc;

struct PatchOperationGuard {
    task_id: TaskId,
    active: Arc<Mutex<HashSet<TaskId>>>,
}

impl PatchOperationGuard {
    fn acquire(task_id: &TaskId, active: &Arc<Mutex<HashSet<TaskId>>>) -> Result<Self> {
        if !active.lock().insert(task_id.clone()) {
            anyhow::bail!(
                "a patch operation is already running for task '{}'",
                task_id
            );
        }
        Ok(Self {
            task_id: task_id.clone(),
            active: active.clone(),
        })
    }
}

impl Drop for PatchOperationGuard {
    fn drop(&mut self) {
        self.active.lock().remove(&self.task_id);
    }
}

/// Global runtime configuration and kill-switch controls.
#[derive(Clone)]
pub struct RuntimeConfig {
    /// Kill switch to disable orchestration runtime and fall back to direct execution.
    pub enabled: bool,
    /// Concurrency and scheduler settings.
    pub scheduler: SchedulerConfig,
    /// Agent identity, relationship, and mailbox limits for this run.
    pub control_plane: AgentControlPlaneConfig,
    /// Optional GPUI foreground executor.
    pub foreground_executor: Option<gpui::ForegroundExecutor>,
    /// Optional GPUI background executor.
    pub background_executor: Option<gpui::BackgroundExecutor>,
    /// Feature flag enabling delegation to ACP agents (defaults to false).
    pub enable_acp_delegation: bool,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            scheduler: SchedulerConfig::default(),
            control_plane: AgentControlPlaneConfig::default(),
            foreground_executor: None,
            background_executor: None,
            enable_acp_delegation: false,
        }
    }
}

/// Initial launch disposition for an orchestration run, decoupling policy from launch approval state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum RuntimeLaunchDisposition {
    #[default]
    Approved,
    AwaitApproval,
    /// Resuming an interrupted run from a persisted snapshot. The scheduler
    /// starts immediately (the run is already approved) and skips approval.
    Resumed,
}

/// Handle to an active or completed orchestration run.
#[derive(Clone)]
pub struct RunHandle {
    run_id: RunId,
    plan_graph: PlanGraph,
    task_registry: TaskRegistry,
    agent_control_plane: AgentControlPlane,
    artifact_store: ArtifactStore,
    cancellation_tree: Arc<CancellationTree>,
    event_stream: RuntimeEventStream,
    control: RuntimeControl,
    policy: AgentExecutionPolicy,
    patch_operations: Arc<Mutex<HashSet<TaskId>>>,
}

impl RunHandle {
    pub fn run_id(&self) -> &RunId {
        &self.run_id
    }

    pub fn plan(&self) -> &OrchestrationPlan {
        self.plan_graph.plan()
    }

    pub fn state(&self) -> RunState {
        self.control.state()
    }

    pub fn policy(&self) -> AgentExecutionPolicy {
        self.policy
    }

    pub fn task_statuses(&self) -> Vec<TaskStatus> {
        self.task_registry.all_statuses()
    }

    pub fn agent_control_plane(&self) -> &AgentControlPlane {
        &self.agent_control_plane
    }

    pub fn task_status(&self, task_id: &TaskId) -> Option<TaskStatus> {
        self.task_registry.status(task_id)
    }

    pub fn task_status_by_session_id(
        &self,
        session_id: &agent_client_protocol::schema::v1::SessionId,
    ) -> Option<TaskStatus> {
        self.task_registry.status_by_session_id(session_id)
    }

    pub fn subscribe(&self) -> EventSubscription {
        self.event_stream.subscribe_with_replay()
    }

    pub fn subscribe_live(&self) -> EventSubscription {
        self.event_stream.subscribe()
    }

    pub fn approve(&self) -> Result<()> {
        if self.control.transition(RunState::Approved)? {
            self.event_stream.emit(RuntimeEvent::PlanApproved {
                run_id: self.run_id.clone(),
            });
            self.event_stream.emit(RuntimeEvent::RunStateChanged {
                run_id: self.run_id.clone(),
                state: RunState::Approved,
            });
        }
        Ok(())
    }

    pub fn cancel(&self, reason: CancellationReason) {
        match self.control.transition(RunState::Cancelled) {
            Ok(true) => {}
            Ok(false) => return,
            Err(error) => {
                log::warn!("failed to cancel orchestration run: {error}");
                return;
            }
        }
        self.cancellation_tree.cancel_run(reason.clone());
        for status in self.task_registry.all_statuses() {
            if !status.state.is_terminal()
                && !self.patch_operations.lock().contains(&status.task_id)
            {
                self.task_registry
                    .set_state(&status.task_id, TaskState::Cancelled);
                self.event_stream.emit(RuntimeEvent::TaskCancelled {
                    run_id: self.run_id.clone(),
                    task_id: status.task_id,
                    reason: reason.description().to_string(),
                });
            }
        }
        self.event_stream.emit(RuntimeEvent::RunCancelled {
            run_id: self.run_id.clone(),
            reason: reason.description().to_string(),
        });
        self.event_stream.emit(RuntimeEvent::RunStateChanged {
            run_id: self.run_id.clone(),
            state: RunState::Cancelled,
        });
    }

    pub fn cancel_task(&self, task_id: &TaskId, reason: CancellationReason) {
        if self
            .task_status(task_id)
            .is_none_or(|status| status.state.is_terminal())
        {
            return;
        }
        self.cancellation_tree.cancel_task(task_id, reason.clone());
        if self
            .task_status(task_id)
            .is_some_and(|status| !status.state.is_active())
            && !self.patch_operations.lock().contains(task_id)
        {
            self.task_registry.set_state(task_id, TaskState::Cancelled);
            for blocked in self.plan_graph.blocked_by_failure(task_id) {
                self.task_registry.set_state(&blocked, TaskState::Blocked);
            }
            if self.control.state() == RunState::AwaitingApply {
                if let Err(error) = self.control.transition(RunState::Running) {
                    log::warn!("failed to wake scheduler after task cancellation: {error}");
                }
            }
        }
        self.event_stream.emit(RuntimeEvent::TaskCancelled {
            run_id: self.run_id.clone(),
            task_id: task_id.clone(),
            reason: reason.description().to_string(),
        });
    }

    pub fn pause(&self) {
        self.control.set_user_paused(true);
        match self.control.transition(RunState::Paused) {
            Ok(true) => {
                self.event_stream.emit(RuntimeEvent::RunPaused {
                    run_id: self.run_id.clone(),
                });
                self.event_stream.emit(RuntimeEvent::RunStateChanged {
                    run_id: self.run_id.clone(),
                    state: RunState::Paused,
                });
            }
            Ok(false) => {}
            Err(error) => log::warn!("failed to pause orchestration run: {error}"),
        }
    }

    pub fn resume(&self) {
        self.control.set_user_paused(false);
        match self.control.transition(RunState::Running) {
            Ok(true) => {
                self.event_stream.emit(RuntimeEvent::RunResumed {
                    run_id: self.run_id.clone(),
                });
                self.event_stream.emit(RuntimeEvent::RunStateChanged {
                    run_id: self.run_id.clone(),
                    state: RunState::Running,
                });
            }
            Ok(false) => {}
            Err(error) => log::warn!("failed to resume orchestration run: {error}"),
        }
    }

    pub fn restart_task(&self, task_id: &TaskId) -> Result<bool> {
        anyhow::ensure!(
            !self.state().is_terminal(),
            "cannot restart a task in a terminal run"
        );
        if !self.task_registry.restart_parked_task(task_id) {
            return Ok(false);
        }
        self.event_stream.emit(RuntimeEvent::TaskWaitReasonChanged {
            run_id: self.run_id.clone(),
            task_id: task_id.clone(),
            wait_reason: None,
        });
        if self.control.state() == RunState::Paused {
            self.resume();
        }
        Ok(true)
    }

    pub async fn retry_cleanup(&self, task_id: &TaskId) -> Result<()> {
        let status = self
            .task_registry
            .status(task_id)
            .context("worker task not found")?;
        anyhow::ensure!(
            status.state.is_terminal(),
            "worker must be terminal before cleanup"
        );
        let _operation = PatchOperationGuard::acquire(task_id, &self.patch_operations)?;
        let metadata = status
            .worker_metadata
            .context("worker has no workspace metadata")?;
        if let Some(path) = &metadata.worktree_path {
            if std::path::Path::new(path).exists() {
                let worktree = IsolatedWorktree::reopen_managed(
                    path.into(),
                    &self.run_id,
                    task_id,
                    status.current_attempt,
                )
                .await?;
                worktree.cleanup(true).await?;
            }
        }
        self.clear_worktree_metadata(task_id, metadata);
        Ok(())
    }

    fn clear_worktree_metadata(
        &self,
        task_id: &TaskId,
        mut metadata: crate::worker::WorkerMetadata,
    ) {
        metadata.worktree_path = None;
        self.task_registry
            .set_worker_metadata(task_id, metadata.clone());
        self.event_stream.emit(RuntimeEvent::WorkerMetadataUpdated {
            run_id: self.run_id.clone(),
            task_id: task_id.clone(),
            metadata,
        });
    }

    /// Returns the unified diff for a task awaiting apply, if an isolated worktree was used.
    pub async fn review_diff(&self, task_id: &TaskId) -> Result<Option<String>> {
        let status = self
            .task_registry
            .status(task_id)
            .ok_or_else(|| anyhow::anyhow!("task '{}' not found", task_id))?;
        let _operation = PatchOperationGuard::acquire(task_id, &self.patch_operations)?;

        let worktree_path = status
            .worker_metadata
            .as_ref()
            .and_then(|metadata| metadata.worktree_path.as_ref())
            .ok_or_else(|| anyhow::anyhow!("task '{}' has no managed worktree", task_id))?;
        let worktree = IsolatedWorktree::reopen_managed(
            std::path::PathBuf::from(worktree_path),
            &self.run_id,
            task_id,
            status.current_attempt,
        )
        .await?;
        Ok(Some(worktree.collect_diff().await?))
    }

    /// Applies changes from an `AwaitingApply` task into the workspace,
    /// marking the task `Completed` and resuming the run if it was waiting.
    pub async fn apply_task(&self, task_id: &TaskId) -> Result<bool> {
        let Some(status) = self.task_registry.status(task_id) else {
            anyhow::bail!("task '{}' not found", task_id);
        };
        let retrying_conflict = status.state == TaskState::Parked
            && matches!(
                &status.wait_reason,
                Some(crate::worker::StructuredWaitReason::ApplyConflict { .. })
                    | Some(crate::worker::StructuredWaitReason::WorktreeVerificationFailed { .. })
                    | Some(
                        crate::worker::StructuredWaitReason::PostApplyVerificationFailed {
                            rollback_error: None,
                            ..
                        }
                    )
            );
        if status.state != TaskState::AwaitingApply && !retrying_conflict {
            return Ok(false);
        }
        let _operation = PatchOperationGuard::acquire(task_id, &self.patch_operations)?;
        let worktree_path = status
            .worker_metadata
            .as_ref()
            .and_then(|metadata| metadata.worktree_path.as_ref())
            .ok_or_else(|| {
                anyhow::anyhow!("task '{}' has no managed worktree to apply", task_id)
            })?;
        let worktree = IsolatedWorktree::reopen_managed(
            std::path::PathBuf::from(worktree_path),
            &self.run_id,
            task_id,
            status.current_attempt,
        )
        .await?;
        let task = self.plan_graph.task(task_id);
        let allowed_subpaths = task.and_then(|t| t.workspace_policy.allowed_subpaths.as_deref());
        // Every isolated write gets a lightweight, deterministic baseline
        // verifier even when a trusted plan builder did not provide a custom
        // command. Custom commands still take precedence.
        let verification_cmd = task
            .and_then(|task| task.verification_command.as_deref())
            .or_else(|| {
                task.filter(|task| {
                    task.workspace_policy.isolation
                        == crate::worker::WorkspaceIsolation::DedicatedWorktree
                })
                .map(|_| "git diff --check")
            });

        // 1. Tier 2: Worktree verification (scope validation + optional verification command)
        if let Err(error) = worktree
            .verify_worktree(verification_cmd, allowed_subpaths)
            .await
        {
            let error = format!("{error:#}");
            self.task_registry.park_worktree_verification_failed(
                task_id,
                error.clone(),
                worktree_path.clone(),
            );
            let wait_reason = crate::worker::StructuredWaitReason::WorktreeVerificationFailed {
                error: error.clone(),
                worktree_path: worktree_path.clone(),
            };
            self.event_stream.emit(RuntimeEvent::TaskWaitReasonChanged {
                run_id: self.run_id.clone(),
                task_id: task_id.clone(),
                wait_reason: Some(wait_reason),
            });
            if self.control.state() == RunState::AwaitingApply
                && self.control.transition(RunState::Paused)?
            {
                self.event_stream.emit(RuntimeEvent::RunStateChanged {
                    run_id: self.run_id.clone(),
                    state: RunState::Paused,
                });
            }
            anyhow::bail!(
                "worktree verification failed; patch was not applied and worktree was retained: {error}"
            );
        }

        // 2. Prepare pre-apply snapshot for safe rollback
        let pre_apply = worktree
            .prepare_parent_apply(&worktree.repo_path, allowed_subpaths)
            .await?;

        // 3. Apply changes into parent repository
        if let Err(error) = worktree
            .apply_to_parent(&worktree.repo_path, allowed_subpaths)
            .await
        {
            let error = format!("{error:#}");
            self.task_registry
                .park_apply_conflict(task_id, error.clone(), worktree_path.clone());
            let wait_reason = crate::worker::StructuredWaitReason::ApplyConflict {
                error: error.clone(),
                worktree_path: worktree_path.clone(),
            };
            self.event_stream.emit(RuntimeEvent::TaskWaitReasonChanged {
                run_id: self.run_id.clone(),
                task_id: task_id.clone(),
                wait_reason: Some(wait_reason),
            });
            if self.control.state() == RunState::AwaitingApply
                && self.control.transition(RunState::Paused)?
            {
                self.event_stream.emit(RuntimeEvent::RunStateChanged {
                    run_id: self.run_id.clone(),
                    state: RunState::Paused,
                });
            }
            anyhow::bail!("worker patch was not applied and its worktree was retained: {error}");
        }

        // 4. Tier 3: Parent post-apply verification
        if let Err(error) =
            IsolatedWorktree::verify_parent(&worktree.repo_path, verification_cmd).await
        {
            let error = format!("{error:#}");
            let rollback_error = if let Err(rollback_error) = pre_apply.rollback().await {
                log::error!(
                    "failed to rollback parent checkout after verification failure for task '{}': {rollback_error}",
                    task_id
                );
                Some(format!("{rollback_error:#}"))
            } else {
                None
            };
            self.task_registry.park_verification_failed(
                task_id,
                error.clone(),
                worktree_path.clone(),
                rollback_error.clone(),
            );
            let wait_reason = crate::worker::StructuredWaitReason::PostApplyVerificationFailed {
                error: error.clone(),
                worktree_path: worktree_path.clone(),
                rollback_error: rollback_error.clone(),
            };
            self.event_stream.emit(RuntimeEvent::TaskWaitReasonChanged {
                run_id: self.run_id.clone(),
                task_id: task_id.clone(),
                wait_reason: Some(wait_reason),
            });
            if self.control.state() == RunState::AwaitingApply
                && self.control.transition(RunState::Paused)?
            {
                self.event_stream.emit(RuntimeEvent::RunStateChanged {
                    run_id: self.run_id.clone(),
                    state: RunState::Paused,
                });
            }
            if let Some(rollback_error) = rollback_error {
                anyhow::bail!(
                    "parent post-apply verification failed and rollback did not complete; parent checkout requires inspection ({rollback_error}): {error}"
                );
            }
            anyhow::bail!(
                "parent post-apply verification failed; parent checkout was rolled back and worktree retained: {error}"
            );
        }
        if !self.task_registry.mark_applied(task_id) {
            anyhow::bail!(
                "task '{}' changed state while its patch was being applied",
                task_id
            );
        }
        let cleanup_error = worktree.cleanup(true).await.err();
        if cleanup_error.is_none()
            && let Some(metadata) = status.worker_metadata.clone()
        {
            self.clear_worktree_metadata(task_id, metadata);
        }
        self.event_stream.emit(RuntimeEvent::TaskCompleted {
            run_id: self.run_id.clone(),
            task_id: task_id.clone(),
            output: self
                .task_registry
                .status(task_id)
                .and_then(|s| s.latest_output),
            tokens_used: self
                .task_registry
                .status(task_id)
                .map(|s| s.tokens_used)
                .unwrap_or(0),
            duration_ms: 0,
        });
        if !self.control.is_user_paused()
            && matches!(
                self.control.state(),
                RunState::AwaitingApply | RunState::Paused
            )
        {
            if self.control.transition(RunState::Running)? {
                self.event_stream.emit(RuntimeEvent::RunStateChanged {
                    run_id: self.run_id.clone(),
                    state: RunState::Running,
                });
            }
        }
        if let Some(error) = cleanup_error {
            anyhow::bail!(
                "task '{}' patch was applied, but its managed worktree could not be removed: {error}",
                task_id
            );
        }
        Ok(true)
    }

    /// Rejects changes from an `AwaitingApply` task, marking the task `Cancelled`,
    /// marking its dependent tasks `Blocked`, and resuming the run if it was waiting.
    pub async fn reject_task(&self, task_id: &TaskId, reason: Option<String>) -> Result<bool> {
        let Some(status) = self.task_registry.status(task_id) else {
            anyhow::bail!("task '{}' not found", task_id);
        };
        let parked_conflict = status.state == TaskState::Parked
            && matches!(
                &status.wait_reason,
                Some(crate::worker::StructuredWaitReason::ApplyConflict { .. })
                    | Some(crate::worker::StructuredWaitReason::WorktreeVerificationFailed { .. })
                    | Some(crate::worker::StructuredWaitReason::PostApplyVerificationFailed { .. })
            );
        if status.state != TaskState::AwaitingApply && !parked_conflict {
            return Ok(false);
        }
        let _operation = PatchOperationGuard::acquire(task_id, &self.patch_operations)?;
        let worktree_path = status
            .worker_metadata
            .as_ref()
            .and_then(|metadata| metadata.worktree_path.as_ref())
            .ok_or_else(|| {
                anyhow::anyhow!("task '{}' has no managed worktree to reject", task_id)
            })?;
        let worktree = IsolatedWorktree::reopen_managed(
            std::path::PathBuf::from(worktree_path),
            &self.run_id,
            task_id,
            status.current_attempt,
        )
        .await?;
        worktree.cleanup(true).await?;
        if let Some(metadata) = status.worker_metadata.clone() {
            self.clear_worktree_metadata(task_id, metadata);
        }
        if !self.task_registry.mark_rejected(task_id, reason.clone()) {
            anyhow::bail!(
                "task '{}' changed state while its worktree was being removed",
                task_id
            );
        }
        let blocked = self.plan_graph.blocked_by_failure(task_id);
        for blocked_id in blocked {
            self.task_registry
                .set_state(&blocked_id, TaskState::Blocked);
        }
        self.event_stream.emit(RuntimeEvent::TaskCancelled {
            run_id: self.run_id.clone(),
            task_id: task_id.clone(),
            reason: reason.unwrap_or_else(|| "task output rejected by user".to_string()),
        });
        if !self.control.is_user_paused()
            && matches!(
                self.control.state(),
                RunState::AwaitingApply | RunState::Paused
            )
        {
            if self.control.transition(RunState::Running)? {
                self.event_stream.emit(RuntimeEvent::RunStateChanged {
                    run_id: self.run_id.clone(),
                    state: RunState::Running,
                });
            }
        }
        Ok(true)
    }

    pub fn snapshot(&self) -> PersistedRun {
        let statuses = self.task_registry.all_statuses();
        let mut attempts = Vec::new();
        for status in &statuses {
            attempts.push((
                status.task_id.clone(),
                self.task_registry.attempts_for(&status.task_id),
            ));
        }

        PersistedRun::new(
            self.run_id.clone(),
            self.plan_graph.plan().clone(),
            self.state(),
            self.policy,
            statuses,
            attempts,
            self.artifact_store.all(),
            self.event_stream.history(),
        )
    }

    /// Returns the sequence number of the last event emitted, used as the
    /// resume cursor for replaying events after a persistence gap.
    pub fn last_event_seq(&self) -> u64 {
        self.event_stream.latest_seq().saturating_sub(1)
    }
}

/// The main entry point to initiate and manage orchestration runs.
pub struct OrchestrationRuntime;

impl OrchestrationRuntime {
    /// Creates and starts an orchestration run with the given plan, policy, and task executor.
    pub fn start(
        plan: OrchestrationPlan,
        policy: AgentExecutionPolicy,
        executor: Rc<dyn TaskExecutor>,
        config: RuntimeConfig,
    ) -> Result<(
        RunHandle,
        futures::channel::oneshot::Receiver<Result<RunState>>,
    )> {
        let disposition = match policy.autonomy {
            agent_settings::AgentAutonomy::Autonomous => RuntimeLaunchDisposition::Approved,
            agent_settings::AgentAutonomy::Manual | agent_settings::AgentAutonomy::Supervised => {
                RuntimeLaunchDisposition::AwaitApproval
            }
        };
        Self::start_with_disposition(plan, policy, disposition, executor, config)
    }

    /// Creates and starts an orchestration run with an explicit launch disposition.
    pub fn start_with_disposition(
        plan: OrchestrationPlan,
        policy: AgentExecutionPolicy,
        disposition: RuntimeLaunchDisposition,
        executor: Rc<dyn TaskExecutor>,
        config: RuntimeConfig,
    ) -> Result<(
        RunHandle,
        futures::channel::oneshot::Receiver<Result<RunState>>,
    )> {
        if !config.enabled {
            anyhow::bail!("orchestration runtime is disabled");
        }
        let plan_graph = PlanGraph::new(plan.clone()).context("invalid plan graph")?;
        let run_id = RunId::new();
        let task_registry = TaskRegistry::new();
        let artifact_store = ArtifactStore::new();
        let cancellation_tree = Arc::new(CancellationTree::new());
        let event_stream = RuntimeEventStream::new();
        let agent_control_plane = AgentControlPlane::from_plan(&plan, config.control_plane.clone())?;
        let initial_state = match disposition {
            RuntimeLaunchDisposition::Approved => RunState::Approved,
            RuntimeLaunchDisposition::AwaitApproval => RunState::Proposed,
            RuntimeLaunchDisposition::Resumed => RunState::Interrupted,
        };
        let control = RuntimeControl::new(initial_state);

        event_stream.emit(RuntimeEvent::RunCreated {
            run_id: run_id.clone(),
            plan_id: plan.id.clone(),
            policy,
            created_at: Utc::now(),
        });
        if initial_state == RunState::Proposed {
            event_stream.emit(RuntimeEvent::PlanProposed {
                run_id: run_id.clone(),
                plan,
            });
        }
        let agent_control_plane =
            agent_control_plane.with_runtime_events(run_id.clone(), event_stream.clone());

        let scheduler = Scheduler::new_with_control(
            run_id.clone(),
            plan_graph.clone(),
            task_registry.clone(),
            artifact_store.clone(),
            cancellation_tree.clone(),
            agent_control_plane.clone(),
            event_stream.clone(),
            executor,
            config.scheduler,
            control.clone(),
        );

        let handle = RunHandle {
            run_id,
            plan_graph,
            task_registry,
            agent_control_plane,
            artifact_store,
            cancellation_tree,
            event_stream,
            control: control.clone(),
            policy,
            patch_operations: Arc::new(Mutex::new(HashSet::new())),
        };

        let (tx, rx) = futures::channel::oneshot::channel();
        let control_clone = control;
        let exec = config
            .foreground_executor
            .context("foreground_executor is required in RuntimeConfig")?;

        exec.spawn(async move {
            let res = scheduler.run().await;
            if let Ok(final_state) = res {
                if !control_clone.state().is_terminal()
                    && let Err(error) = control_clone.transition(final_state)
                {
                    log::error!("failed to finalize orchestration run: {error}");
                }
                if tx.send(Ok(control_clone.state())).is_err() {
                    log::debug!("orchestration completion receiver dropped");
                }
            } else if let Err(err) = res {
                if let Err(error) = control_clone.transition(RunState::Failed) {
                    log::error!("failed to mark orchestration run failed: {error}");
                }
                if tx.send(Err(err)).is_err() {
                    log::debug!("orchestration completion receiver dropped");
                }
            }
        })
        .detach();
        Ok((handle, rx))
    }

    /// Resumes a previously saved run from persistence.
    pub fn resume(
        persisted: PersistedRun,
        executor: Rc<dyn TaskExecutor>,
        config: RuntimeConfig,
    ) -> Result<(
        RunHandle,
        futures::channel::oneshot::Receiver<Result<RunState>>,
    )> {
        if !config.enabled {
            anyhow::bail!("orchestration runtime is disabled");
        }
        let persisted = persisted.for_resume();
        let plan_graph = PlanGraph::new(persisted.plan.clone()).context("invalid plan graph")?;
        let run_id = persisted.run_id.clone();
        let task_registry = TaskRegistry::new();
        let artifact_store = ArtifactStore::new();
        let cancellation_tree = Arc::new(CancellationTree::new());
        let persisted_policy = persisted.policy;
        let event_stream = RuntimeEventStream::from_history(persisted.event_log.clone());
        let agent_control_plane =
            AgentControlPlane::from_plan(&persisted.plan, config.control_plane.clone())?
                .with_runtime_events(run_id.clone(), event_stream.clone());
        let control = RuntimeControl::new(persisted.state);

        // Restore task statuses and attempts
        for mut status in persisted.task_statuses {
            if status.state == TaskState::Parked
                && matches!(
                    status.wait_reason,
                    Some(crate::worker::StructuredWaitReason::AwaitingWorkerReconnect { .. })
                )
            {
                status.state = TaskState::Interrupted;
                status.wait_reason = None;
            }
            task_registry.restore_status(status);
        }
        for (task_id, attempts) in persisted.task_attempts {
            for attempt in attempts {
                task_registry.restore_attempt(task_id.clone(), attempt);
            }
        }
        for artifact in persisted.artifacts {
            artifact_store.record(artifact);
        }

        let scheduler = Scheduler::new_with_control(
            run_id.clone(),
            plan_graph.clone(),
            task_registry.clone(),
            artifact_store.clone(),
            cancellation_tree.clone(),
            agent_control_plane.clone(),
            event_stream.clone(),
            executor,
            config.scheduler,
            control.clone(),
        );

        let handle = RunHandle {
            run_id,
            plan_graph,
            task_registry,
            agent_control_plane,
            artifact_store,
            cancellation_tree,
            event_stream,
            control: control.clone(),
            policy: persisted_policy,
            patch_operations: Arc::new(Mutex::new(HashSet::new())),
        };
        let (tx, rx) = futures::channel::oneshot::channel();
        let control_clone = control;
        let exec = config
            .foreground_executor
            .context("foreground_executor is required in RuntimeConfig")?;

        exec.spawn(async move {
            let res = scheduler.run().await;
            if let Ok(final_state) = res {
                if !control_clone.state().is_terminal()
                    && let Err(error) = control_clone.transition(final_state)
                {
                    log::error!("failed to finalize resumed orchestration run: {error}");
                }
                if tx.send(Ok(control_clone.state())).is_err() {
                    log::debug!("orchestration completion receiver dropped");
                }
            } else if let Err(err) = res {
                if let Err(error) = control_clone.transition(RunState::Failed) {
                    log::error!("failed to mark resumed orchestration run failed: {error}");
                }
                if tx.send(Err(err)).is_err() {
                    log::debug!("orchestration completion receiver dropped");
                }
            }
        })
        .detach();

        Ok((handle, rx))
    }
}
