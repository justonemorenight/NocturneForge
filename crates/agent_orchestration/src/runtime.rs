use crate::artifacts::ArtifactStore;
use crate::cancellation::{CancellationReason, CancellationTree};
use crate::events::{EventSubscription, RuntimeEvent, RuntimeEventStream};
use crate::executor::TaskExecutor;
use crate::ids::{RunId, TaskId};
use crate::persistence::PersistedRun;
use crate::plan_graph::{OrchestrationPlan, PlanGraph};
use crate::scheduler::{RuntimeControl, Scheduler, SchedulerConfig};
use crate::state::{RunState, TaskStatus};
use crate::task_registry::TaskRegistry;
use agent_settings::AgentExecutionPolicy;
use anyhow::{Context as _, Result};
use chrono::Utc;
use std::rc::Rc;
use std::sync::Arc;

/// Global runtime configuration and kill-switch controls.
#[derive(Clone)]
pub struct RuntimeConfig {
    /// Kill switch to disable orchestration runtime and fall back to direct execution.
    pub enabled: bool,
    /// Concurrency and scheduler settings.
    pub scheduler: SchedulerConfig,
    /// Optional GPUI foreground executor.
    pub foreground_executor: Option<gpui::ForegroundExecutor>,
    /// Optional GPUI background executor.
    pub background_executor: Option<gpui::BackgroundExecutor>,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            scheduler: SchedulerConfig::default(),
            foreground_executor: None,
            background_executor: None,
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
    artifact_store: ArtifactStore,
    cancellation_tree: Arc<CancellationTree>,
    event_stream: RuntimeEventStream,
    control: RuntimeControl,
    policy: AgentExecutionPolicy,
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
        if !matches!(self.control.transition(RunState::Cancelled), Ok(true)) {
            return;
        }
        self.cancellation_tree.cancel_run(reason.clone());
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
            .is_some_and(|status| status.state.is_terminal())
        {
            return;
        }
        self.cancellation_tree.cancel_task(task_id, reason.clone());
        self.event_stream.emit(RuntimeEvent::TaskCancelled {
            run_id: self.run_id.clone(),
            task_id: task_id.clone(),
            reason: reason.description().to_string(),
        });
    }

    pub fn pause(&self) {
        if self.control.transition(RunState::Paused).unwrap_or(false) {
            self.event_stream.emit(RuntimeEvent::RunPaused {
                run_id: self.run_id.clone(),
            });
            self.event_stream.emit(RuntimeEvent::RunStateChanged {
                run_id: self.run_id.clone(),
                state: RunState::Paused,
            });
        }
    }

    pub fn resume(&self) {
        if self.control.transition(RunState::Running).unwrap_or(false) {
            self.event_stream.emit(RuntimeEvent::RunResumed {
                run_id: self.run_id.clone(),
            });
            self.event_stream.emit(RuntimeEvent::RunStateChanged {
                run_id: self.run_id.clone(),
                state: RunState::Running,
            });
        }
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

        let scheduler = Scheduler::new_with_control(
            run_id.clone(),
            plan_graph.clone(),
            task_registry.clone(),
            artifact_store.clone(),
            cancellation_tree.clone(),
            event_stream.clone(),
            executor,
            config.scheduler,
            control.clone(),
        );

        let handle = RunHandle {
            run_id,
            plan_graph,
            task_registry,
            artifact_store,
            cancellation_tree,
            event_stream,
            control: control.clone(),
            policy,
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
        let control = RuntimeControl::new(persisted.state);

        // Restore task statuses and attempts
        for status in persisted.task_statuses {
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
            event_stream.clone(),
            executor,
            config.scheduler,
            control.clone(),
        );

        let handle = RunHandle {
            run_id,
            plan_graph,
            task_registry,
            artifact_store,
            cancellation_tree,
            event_stream,
            control: control.clone(),
            policy: persisted_policy,
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
