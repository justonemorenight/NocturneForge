pub mod acp_adapter;
pub mod artifacts;
pub mod auto_policy;
pub mod budget;
pub mod cancellation;
pub mod events;
pub mod executor;
pub mod ids;
pub mod persistence;
pub mod plan_graph;
pub mod planner;
pub mod runtime;
pub mod scheduler;
pub mod state;
pub mod task_registry;
pub mod verification;
pub mod worker;
pub mod worktree_isolation;

pub use acp_adapter::{AcpEventAdapter, AcpTaskBridge};
pub use artifacts::{
    Artifact, ArtifactKind, ArtifactStore, ContextSnapshot, MAX_DEPENDENCY_CONTEXT_BYTES,
    MAX_INLINE_OUTPUT_BYTES, truncate_text,
};
pub use auto_policy::{
    AgentToolProfile, AutoPolicyConfig, AutoPolicyContext, AutoPolicyDecision, AutoPolicyEngine,
    ResolvedTurnPolicy, TurnPolicySource,
};
pub use budget::{
    BudgetExceeded, BudgetUsage, ExecutionBudget, TaskBudgetState, TaskExecutionReporter,
};
pub use cancellation::{CancellationReason, CancellationToken, CancellationTree};
pub use events::{RuntimeEvent, RuntimeEventStream};
pub use executor::{
    DependencyInput, MockTaskExecutor, TaskExecutionContext, TaskExecutionOutput, TaskExecutor,
};
pub use ids::{CorrelationId, PlanId, RunId, TaskId};
pub use persistence::{PERSISTENCE_SCHEMA_VERSION, PersistedRun};
pub use plan_graph::{GraphValidationError, OrchestrationPlan, OrchestrationTask, PlanGraph};
pub use planner::{OrchestrationPlanner, PlanProposal};
pub use runtime::{OrchestrationRuntime, RunHandle, RuntimeConfig, RuntimeLaunchDisposition};
pub use scheduler::{RuntimeControl, Scheduler, SchedulerConfig};
pub use state::{RunState, TaskAttempt, TaskState, TaskStatus};
pub use task_registry::TaskRegistry;
pub use verification::{
    ErrorClass, RetryReason, VERIFICATION_END, VERIFICATION_START, VerificationPolicy,
    VerificationResult, VerificationRunner, VerificationVerdict,
};
pub use worker::{
    AcpWorkerRuntimeConfig, CapabilitySnapshot, StructuredWaitReason, WorkerBroker, WorkerHandle,
    WorkerHost, WorkerMetadata, WorkerTarget, WorkspaceIsolation, WorkspacePolicy,
};
pub use worktree_isolation::{IsolatedWorktree, WorktreeManager, WorktreeOwnershipMarker};

#[cfg(test)]
mod tests {
    use super::*;
    use agent_client_protocol::schema::v1 as acp;
    use agent_settings::{AgentAutonomy, AgentExecutionPolicy, AgentExecutionStrategy};
    use std::rc::Rc;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    struct PendingExecutor;

    impl TaskExecutor for PendingExecutor {
        fn execute(
            &self,
            _context: TaskExecutionContext,
        ) -> futures::future::LocalBoxFuture<'static, anyhow::Result<TaskExecutionOutput>> {
            Box::pin(futures::future::pending())
        }
    }

    struct DependencyProgressExecutor {
        slow_release: async_channel::Receiver<()>,
        dependent_started: async_channel::Sender<()>,
    }

    impl TaskExecutor for DependencyProgressExecutor {
        fn execute(
            &self,
            context: TaskExecutionContext,
        ) -> futures::future::LocalBoxFuture<'static, anyhow::Result<TaskExecutionOutput>> {
            let slow_release = self.slow_release.clone();
            let dependent_started = self.dependent_started.clone();
            Box::pin(async move {
                match context.task.id.as_str() {
                    "slow" => slow_release
                        .recv()
                        .await
                        .map_err(|error| anyhow::anyhow!(error))?,
                    "dependent" => dependent_started
                        .send(())
                        .await
                        .map_err(|error| anyhow::anyhow!(error))?,
                    _ => {}
                }
                Ok(TaskExecutionOutput {
                    session_id: Some(acp::SessionId::new(format!("session-{}", context.task.id))),
                    output: format!("completed {}", context.task.id),
                    tokens_used: None,
                    artifacts: Vec::new(),
                    worker_metadata: None,
                })
            })
        }
    }

    #[test]
    fn test_plan_graph_validation_rejects_empty_tasks() {
        let plan = OrchestrationPlan::new("Empty", vec![]);
        let graph = PlanGraph::new(plan);
        assert!(matches!(graph, Err(GraphValidationError::EmptyTasks)));
    }

    #[test]
    fn test_plan_graph_validation_rejects_duplicate_ids() {
        let t1 = OrchestrationTask::new("task-1", "Task 1", "desc");
        let t2 = OrchestrationTask::new("task-1", "Task 2", "desc");
        let plan = OrchestrationPlan::new("Dupes", vec![t1, t2]);
        let graph = PlanGraph::new(plan);
        assert!(matches!(
            graph,
            Err(GraphValidationError::DuplicateTaskId(id)) if id.as_str() == "task-1"
        ));
    }

    #[test]
    fn test_plan_graph_validation_rejects_missing_dependency() {
        let mut t1 = OrchestrationTask::new("task-1", "Task 1", "desc");
        t1.depends_on = vec![TaskId::new("non-existent")];
        let plan = OrchestrationPlan::new("MissingDep", vec![t1]);
        let graph = PlanGraph::new(plan);
        assert!(matches!(
            graph,
            Err(GraphValidationError::MissingDependency { .. })
        ));
    }

    #[test]
    fn test_plan_graph_validation_rejects_cycles() {
        let mut t1 = OrchestrationTask::new("task-1", "Task 1", "desc");
        let mut t2 = OrchestrationTask::new("task-2", "Task 2", "desc");
        t1.depends_on = vec![TaskId::new("task-2")];
        t2.depends_on = vec![TaskId::new("task-1")];
        let plan = OrchestrationPlan::new("Cycle", vec![t1, t2]);
        let graph = PlanGraph::new(plan);
        assert!(matches!(
            graph,
            Err(GraphValidationError::CyclicDependency(_))
        ));
    }

    #[test]
    fn test_plan_graph_wave_ordering() {
        let t1 = OrchestrationTask::new("task-1", "Task 1", "desc");
        let t2 = OrchestrationTask::new("task-2", "Task 2", "desc");
        let mut t3 = OrchestrationTask::new("task-3", "Task 3", "desc");
        t3.depends_on = vec![TaskId::new("task-1"), TaskId::new("task-2")];
        let mut t4 = OrchestrationTask::new("task-4", "Task 4", "desc");
        t4.depends_on = vec![TaskId::new("task-3")];

        let plan = OrchestrationPlan::new("Waves", vec![t1, t2, t3, t4]);
        let graph = PlanGraph::new(plan).expect("valid graph");
        let waves = graph.waves();

        assert_eq!(waves.len(), 3);
        assert_eq!(waves[0].len(), 2);
        assert_eq!(waves[1], vec![TaskId::new("task-3")]);
        assert_eq!(waves[2], vec![TaskId::new("task-4")]);
    }

    #[test]
    fn test_error_classification_and_retry_decision() {
        let policy = VerificationPolicy::default();

        let network_err = "failed to connect: connection timed out";
        let class = VerificationPolicy::classify_error(network_err);
        assert_eq!(class, ErrorClass::TransientNetwork);
        assert!(policy.should_retry(1, &class, Some(2)));
        assert!(policy.should_retry(2, &class, Some(2)));
        assert!(!policy.should_retry(3, &class, Some(2)));

        let budget_err = "token budget of 1000 was exceeded";
        let class = VerificationPolicy::classify_error(budget_err);
        assert_eq!(class, ErrorClass::TokenBudgetExceeded);
        assert!(!policy.should_retry(0, &class, Some(2)));
    }

    #[test]
    fn test_exponential_backoff_calculation() {
        let policy = VerificationPolicy {
            max_retries: 3,
            backoff_initial_ms: 100,
            backoff_factor: 2.0,
            max_backoff_ms: 1000,
            verify_outputs: true,
            allow_repair_tasks: true,
        };

        assert_eq!(policy.backoff_duration(0), Duration::from_millis(0));
        assert_eq!(policy.backoff_duration(1), Duration::from_millis(100));
        assert_eq!(policy.backoff_duration(2), Duration::from_millis(200));
        assert_eq!(policy.backoff_duration(3), Duration::from_millis(400));
    }

    #[test]
    fn test_cancellation_tree_propagation() {
        let tree = CancellationTree::new();
        let t1_token = tree.task_token(&TaskId::new("task-1"));
        let t2_token = tree.task_token(&TaskId::new("task-2"));

        assert!(!t1_token.is_cancelled());
        assert!(!t2_token.is_cancelled());

        // Cancel specific task
        tree.cancel_task(&TaskId::new("task-1"), CancellationReason::UserRequested);
        assert!(t1_token.is_cancelled());
        assert!(!t2_token.is_cancelled());
        assert!(!tree.root_token().is_cancelled());

        // Cancel root run
        tree.cancel_run(CancellationReason::Shutdown);
        assert!(tree.root_token().is_cancelled());
        assert!(t2_token.is_cancelled());
    }

    #[gpui::test]
    async fn test_cancellation_wakes_all_waiters(_cx: &mut gpui::TestAppContext) {
        let token = CancellationToken::new();
        let first_token = token.clone();
        let second_token = token.clone();
        let mut first_waiter = Box::pin(first_token.cancelled());
        let mut second_waiter = Box::pin(second_token.cancelled());

        assert!(futures::poll!(first_waiter.as_mut()).is_pending());
        assert!(futures::poll!(second_waiter.as_mut()).is_pending());
        token.cancel(CancellationReason::UserRequested);

        first_waiter.await;
        second_waiter.await;
    }

    #[test]
    fn test_task_registration_is_idempotent_for_restore() {
        let registry = TaskRegistry::new();
        let task_id = TaskId::new("task-1");
        registry.register_task(task_id.clone());
        registry.set_state(&task_id, TaskState::Completed);
        registry.register_task(task_id.clone());
        assert_eq!(
            registry.status(&task_id).map(|status| status.state),
            Some(TaskState::Completed)
        );
    }

    #[test]
    fn test_event_history_is_bounded() {
        let stream = RuntimeEventStream::new();
        let run_id = RunId::new();
        for _ in 0..10_050 {
            stream.emit(RuntimeEvent::RunStarted {
                run_id: run_id.clone(),
            });
        }
        assert_eq!(stream.history().len(), 10_000);
    }

    #[test]
    fn test_event_history_is_bounded_by_serialized_bytes() {
        let stream = RuntimeEventStream::new();
        let run_id = RunId::new();
        for _ in 0..20 {
            stream.emit(RuntimeEvent::TaskProgress {
                run_id: run_id.clone(),
                task_id: TaskId::new("large-task"),
                message: "x".repeat(1024 * 1024),
                percent: 0.0,
                tokens_used: None,
            });
        }
        assert!(stream.history().len() < 10);
    }

    #[test]
    fn test_event_replay_includes_history_larger_than_live_buffer() {
        let stream = RuntimeEventStream::new();
        let run_id = RunId::new();
        for _ in 0..300 {
            stream.emit(RuntimeEvent::RunStarted {
                run_id: run_id.clone(),
            });
        }

        let subscription = stream.subscribe_with_replay();
        let mut replayed = 0;
        while subscription.receiver.try_recv().is_ok() {
            replayed += 1;
        }
        assert_eq!(replayed, 300);
    }

    #[test]
    fn test_registry_records_session_allocated_during_attempt() {
        let registry = TaskRegistry::new();
        let task_id = TaskId::new("task-1");
        let session_id = acp::SessionId::new("session-1");
        registry.register_task(task_id.clone());
        registry.start_attempt(&task_id, None);
        registry.set_active_session_id(&task_id, session_id.clone());

        assert_eq!(
            registry
                .status(&task_id)
                .and_then(|status| status.active_session_id),
            Some(session_id.clone())
        );
        assert_eq!(
            registry
                .status_by_session_id(&session_id)
                .map(|status| status.task_id),
            Some(task_id.clone())
        );
        assert_eq!(
            registry
                .attempts_for(&task_id)
                .last()
                .and_then(|attempt| attempt.session_id.clone()),
            Some(session_id)
        );
    }

    #[test]
    fn test_registry_restores_complete_status_and_session_lookup() {
        let registry = TaskRegistry::new();
        let task_id = TaskId::new("task-1");
        let session_id = acp::SessionId::new("session-1");
        let mut status = TaskStatus::new(task_id.clone());
        status.state = TaskState::Interrupted;
        status.current_attempt = 2;
        status.total_attempts = 2;
        status.tokens_used = 1_024;
        status.active_session_id = Some(session_id.clone());
        status.latest_output = Some("partial result".to_string());
        registry.restore_status(status.clone());

        assert_eq!(registry.status(&task_id), Some(status.clone()));
        assert_eq!(registry.status_by_session_id(&session_id), Some(status));
    }

    #[test]
    fn restart_only_accepts_parked_user_decisions_and_detaches_old_session() {
        let registry = TaskRegistry::new();
        let task_id = TaskId::new("restart-worker");
        let session_id = acp::SessionId::new("old-session");
        let mut status = TaskStatus::new(task_id.clone());
        status.state = TaskState::Parked;
        status.active_session_id = Some(session_id.clone());
        status.wait_reason = Some(StructuredWaitReason::AwaitingUserInput {
            question: "Restart worker?".into(),
        });
        registry.restore_status(status);
        assert!(registry.restart_parked_task(&task_id));
        let restarted = registry.status(&task_id).expect("registered task");
        assert_eq!(restarted.state, TaskState::Pending);
        assert!(restarted.active_session_id.is_none());
        assert!(restarted.wait_reason.is_none());
        assert!(registry.status_by_session_id(&session_id).is_none());
        assert!(!registry.restart_parked_task(&task_id));
        registry.set_state(&task_id, TaskState::Blocked);
        assert!(!registry.restart_parked_task(&task_id));
    }

    #[test]
    fn cancelled_apply_conflict_cannot_be_restarted_or_completed() {
        let registry = TaskRegistry::new();
        let task_id = TaskId::new("apply-worker");
        registry.register_task(task_id.clone());
        registry.set_state(&task_id, TaskState::AwaitingApply);
        assert!(registry.park_apply_conflict(&task_id, "conflict".into(), "/worktree".into()));
        assert!(!registry.restart_parked_task(&task_id));
        registry.set_state(&task_id, TaskState::Cancelled);
        assert!(!registry.mark_applied(&task_id));
        assert!(!registry.mark_rejected(&task_id, None));
        assert!(!registry.park_apply_conflict(
            &task_id,
            "late conflict".into(),
            "/worktree".into()
        ));
        assert_eq!(
            registry.status(&task_id).expect("registered task").state,
            TaskState::Cancelled
        );
    }

    #[test]
    fn test_large_artifacts_are_bounded_and_marked() {
        let store = ArtifactStore::new();
        let artifact = Artifact::new(
            TaskId::new("task-1"),
            "large output",
            ArtifactKind::Text,
            "x".repeat(5 * 1024 * 1024),
        );
        let artifact_id = artifact.id.clone();
        store.record(artifact);
        let stored = store.get(&artifact_id).expect("artifact retained");
        assert_eq!(stored.data.len(), MAX_INLINE_OUTPUT_BYTES);
        assert_eq!(
            stored
                .metadata
                .as_ref()
                .and_then(|metadata| metadata.get("truncated"))
                .and_then(serde_json::Value::as_bool),
            Some(true)
        );
    }

    #[test]
    fn test_persisted_run_serialization_and_resume_preparation() {
        let t1 = OrchestrationTask::new("task-1", "Task 1", "desc");
        let t2 = OrchestrationTask::new("task-2", "Task 2", "desc");
        let plan = OrchestrationPlan::new("Persist", vec![t1, t2]);
        let mut status = TaskStatus::new(TaskId::new("task-1"));
        status.state = TaskState::Running;
        let pending_status = TaskStatus::new(TaskId::new("task-2"));

        let persisted = PersistedRun::new(
            RunId::new(),
            plan,
            RunState::Running,
            agent_settings::AgentExecutionPolicy::default(),
            vec![status, pending_status],
            vec![],
            vec![],
            vec![],
        );

        let json = persisted.to_json().expect("serialize");
        let loaded = PersistedRun::from_json(&json).expect("deserialize");
        assert_eq!(loaded.schema_version, PERSISTENCE_SCHEMA_VERSION);

        let resumed = loaded.for_resume();
        assert_eq!(resumed.state, RunState::Interrupted);
        assert_eq!(resumed.task_statuses[0].state, TaskState::Interrupted);
        assert_eq!(resumed.task_statuses[1].state, TaskState::Pending);
    }

    #[test]
    fn test_auto_policy_engine_heuristics() {
        let context = AutoPolicyContext {
            work_item_count: 1,
            available_tool_count: Some(5),
            can_orchestrate: true,
        };
        let decision_direct = AutoPolicyEngine::evaluate("fix typo in readme", context);
        assert_eq!(decision_direct.strategy, AgentExecutionStrategy::Direct);

        let decision_plan = AutoPolicyEngine::evaluate(
            "create an architectural RFC and design plan for the new database layer",
            context,
        );
        assert_eq!(decision_plan.strategy, AgentExecutionStrategy::Plan);

        let decision_orch = AutoPolicyEngine::evaluate(
            "refactor across all files in parallel: step 1 create models, step 2 update migrations",
            AutoPolicyContext {
                work_item_count: 5,
                available_tool_count: Some(10),
                can_orchestrate: true,
            },
        );
        assert_eq!(decision_orch.strategy, AgentExecutionStrategy::Orchestrate);
    }

    #[gpui::test]
    async fn test_runtime_execution_end_to_end(cx: &mut gpui::TestAppContext) {
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();

        let executor = Rc::new(MockTaskExecutor::new(move |context| {
            counter_clone.fetch_add(1, Ordering::SeqCst);
            Ok(TaskExecutionOutput::new(format!(
                "Completed {}",
                context.task.id
            )))
        }));

        let t1 = OrchestrationTask::new("task-1", "Task 1", "desc 1");
        let mut t2 = OrchestrationTask::new("task-2", "Task 2", "desc 2");
        t2.depends_on = vec![TaskId::new("task-1")];

        let plan = OrchestrationPlan::new("E2E", vec![t1, t2]);
        let policy = AgentExecutionPolicy {
            strategy: AgentExecutionStrategy::Orchestrate,
            autonomy: AgentAutonomy::Supervised,
        };

        let mut config = RuntimeConfig::default();
        config.foreground_executor = Some(cx.foreground_executor().clone());
        config.background_executor = Some(cx.background_executor.clone());
        config.scheduler.background_executor = Some(cx.background_executor.clone());
        let (handle, completion_rx) =
            OrchestrationRuntime::start(plan, policy, executor, config).expect("start run");
        handle.approve().expect("approve proposed plan");
        let state = completion_rx.await.expect("receive").expect("ok state");
        assert_eq!(state, RunState::Completed);
        assert_eq!(counter.load(Ordering::SeqCst), 2);

        let statuses = handle.task_statuses();
        assert_eq!(statuses.len(), 2);
        assert!(statuses.iter().all(|s| s.state == TaskState::Completed));
    }

    #[gpui::test]
    async fn test_scheduler_dispatches_newly_ready_task_without_waiting_for_sibling(
        cx: &mut gpui::TestAppContext,
    ) {
        let (slow_release_sender, slow_release_receiver) = async_channel::bounded(1);
        let (dependent_started_sender, dependent_started_receiver) = async_channel::bounded(1);
        let executor = Rc::new(DependencyProgressExecutor {
            slow_release: slow_release_receiver,
            dependent_started: dependent_started_sender,
        });

        let fast = OrchestrationTask::new("fast", "Fast", "finish immediately");
        let slow = OrchestrationTask::new("slow", "Slow", "wait for release");
        let mut dependent = OrchestrationTask::new("dependent", "Dependent", "run after fast");
        dependent.depends_on = vec![fast.id.clone()];
        let plan = OrchestrationPlan::new("Progressive scheduling", vec![fast, slow, dependent]);
        let policy = AgentExecutionPolicy {
            strategy: AgentExecutionStrategy::Orchestrate,
            autonomy: AgentAutonomy::Autonomous,
        };
        let mut config = RuntimeConfig::default();
        config.foreground_executor = Some(cx.foreground_executor().clone());
        config.scheduler.background_executor = Some(cx.background_executor.clone());
        config.scheduler.max_parallel_tasks = 2;
        let (handle, completion) =
            OrchestrationRuntime::start(plan, policy, executor, config).expect("start run");

        let dependent_started = dependent_started_receiver.recv();
        let timeout = cx.background_executor.timer(Duration::from_secs(1));
        futures::pin_mut!(dependent_started, timeout);
        match futures::future::select(dependent_started, timeout).await {
            futures::future::Either::Left((result, _)) => {
                result.expect("dependent task started");
            }
            futures::future::Either::Right(_) => {
                panic!("dependent task remained blocked by an unrelated slow sibling");
            }
        }

        slow_release_sender
            .send(())
            .await
            .expect("release slow task");
        assert_eq!(
            completion.await.expect("receive").expect("complete"),
            RunState::Completed
        );
        assert_eq!(
            handle
                .task_status_by_session_id(&acp::SessionId::new("session-dependent"))
                .map(|status| status.task_id),
            Some(TaskId::new("dependent"))
        );
    }

    #[gpui::test]
    async fn test_runtime_waits_for_manual_approval(cx: &mut gpui::TestAppContext) {
        let counter = Arc::new(AtomicUsize::new(0));
        let executor = Rc::new(MockTaskExecutor::new({
            let counter = counter.clone();
            move |_| {
                counter.fetch_add(1, Ordering::SeqCst);
                Ok(TaskExecutionOutput::new("done"))
            }
        }));
        let plan = OrchestrationPlan::new(
            "Approval",
            vec![OrchestrationTask::new("task-1", "Task", "desc")],
        );
        let policy = AgentExecutionPolicy {
            strategy: AgentExecutionStrategy::Orchestrate,
            autonomy: AgentAutonomy::Manual,
        };
        let mut config = RuntimeConfig::default();
        config.foreground_executor = Some(cx.foreground_executor().clone());
        config.scheduler.background_executor = Some(cx.background_executor.clone());

        let (handle, completion) =
            OrchestrationRuntime::start(plan, policy, executor, config).expect("start run");
        assert_eq!(handle.state(), RunState::Proposed);
        assert_eq!(counter.load(Ordering::SeqCst), 0);
        handle.approve().expect("approve run");
        assert_eq!(
            completion.await.expect("receive").expect("complete"),
            RunState::Completed
        );
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[gpui::test]
    async fn test_cancel_interrupts_non_cooperative_executor(cx: &mut gpui::TestAppContext) {
        let plan = OrchestrationPlan::new(
            "Cancellation",
            vec![OrchestrationTask::new("task-1", "Task", "desc")],
        );
        let policy = AgentExecutionPolicy {
            strategy: AgentExecutionStrategy::Orchestrate,
            autonomy: AgentAutonomy::Autonomous,
        };
        let mut config = RuntimeConfig::default();
        config.foreground_executor = Some(cx.foreground_executor().clone());
        config.scheduler.background_executor = Some(cx.background_executor.clone());
        let (handle, completion) =
            OrchestrationRuntime::start(plan, policy, Rc::new(PendingExecutor), config)
                .expect("start run");
        cx.run_until_parked();
        handle.cancel(CancellationReason::UserRequested);
        assert_eq!(
            handle
                .task_status(&TaskId::new("task-1"))
                .map(|status| status.state),
            Some(TaskState::Cancelled)
        );
        assert_eq!(
            completion.await.expect("receive").expect("complete"),
            RunState::Cancelled
        );
    }

    #[gpui::test]
    async fn test_task_timeout_fails_non_cooperative_executor(cx: &mut gpui::TestAppContext) {
        let plan = OrchestrationPlan::new(
            "Timeout",
            vec![OrchestrationTask::new("task-1", "Task", "desc")],
        );
        let policy = AgentExecutionPolicy {
            strategy: AgentExecutionStrategy::Orchestrate,
            autonomy: AgentAutonomy::Autonomous,
        };
        let mut config = RuntimeConfig::default();
        config.foreground_executor = Some(cx.foreground_executor().clone());
        config.scheduler.background_executor = Some(cx.background_executor.clone());
        config.scheduler.task_timeout_secs = Some(0);
        let (handle, completion) =
            OrchestrationRuntime::start(plan, policy, Rc::new(PendingExecutor), config)
                .expect("start run");
        assert_eq!(
            completion.await.expect("receive").expect("complete"),
            RunState::Failed
        );
        assert_eq!(
            handle
                .task_status(&TaskId::new("task-1"))
                .map(|status| status.state),
            Some(TaskState::Failed)
        );
    }

    #[gpui::test]
    async fn test_timeout_and_cancel_interrupt_verification_and_repair(
        cx: &mut gpui::TestAppContext,
    ) {
        struct PendingStageExecutor {
            repair: bool,
        }
        impl TaskExecutor for PendingStageExecutor {
            fn execute(
                &self,
                _context: TaskExecutionContext,
            ) -> futures::future::LocalBoxFuture<'static, anyhow::Result<TaskExecutionOutput>>
            {
                Box::pin(async { Ok(TaskExecutionOutput::new("initial output")) })
            }
            fn verify(
                &self,
                _task: &OrchestrationTask,
                _output: &TaskExecutionOutput,
            ) -> futures::future::LocalBoxFuture<'static, anyhow::Result<VerificationResult>>
            {
                let repair = self.repair;
                Box::pin(async move {
                    if repair {
                        Ok(VerificationResult::fail(
                            "needs repair",
                            ErrorClass::VerificationAssertionFailure,
                        ))
                    } else {
                        futures::future::pending().await
                    }
                })
            }
            fn repair(
                &self,
                _task: &OrchestrationTask,
                _feedback: &str,
                _context: TaskExecutionContext,
            ) -> futures::future::LocalBoxFuture<'static, anyhow::Result<TaskExecutionOutput>>
            {
                Box::pin(futures::future::pending())
            }
        }
        for (repair, cancel) in [(false, false), (true, false), (false, true), (true, true)] {
            let mut task = OrchestrationTask::new("task", "Task", "verify output");
            task.acceptance_criteria = vec!["verified".to_string()];
            task.repair_on_failure = true;
            let plan = OrchestrationPlan::new("Stage timeout", vec![task]);
            let policy = AgentExecutionPolicy {
                strategy: AgentExecutionStrategy::Orchestrate,
                autonomy: AgentAutonomy::Autonomous,
            };
            let mut config = RuntimeConfig::default();
            config.foreground_executor = Some(cx.foreground_executor().clone());
            config.scheduler.background_executor = Some(cx.background_executor.clone());
            config.scheduler.task_timeout_secs = Some(1);
            config.scheduler.verification_policy.allow_repair_tasks = true;
            let (handle, completion) = OrchestrationRuntime::start(
                plan,
                policy,
                Rc::new(PendingStageExecutor { repair }),
                config,
            )
            .expect("start run");
            cx.run_until_parked();
            assert_eq!(
                handle
                    .task_status(&TaskId::new("task"))
                    .map(|status| status.state),
                Some(if repair {
                    TaskState::Repairing
                } else {
                    TaskState::Verifying
                })
            );
            if cancel {
                handle.cancel_task(&TaskId::new("task"), CancellationReason::UserRequested);
            } else {
                cx.background_executor
                    .advance_clock(std::time::Duration::from_secs(2));
            }
            let outcome = completion.await.expect("completion").expect("run result");
            assert_ne!(outcome, RunState::Completed);
            assert_eq!(
                handle
                    .task_status(&TaskId::new("task"))
                    .map(|status| status.state),
                Some(if cancel {
                    TaskState::Cancelled
                } else {
                    TaskState::Failed
                })
            );
        }
    }

    #[gpui::test]
    async fn test_default_verifier_rejects_unchecked_criteria(cx: &mut gpui::TestAppContext) {
        let executor = MockTaskExecutor::new(|_| Ok(TaskExecutionOutput::new("done")));
        let mut task = OrchestrationTask::new("task-1", "Task", "desc");
        task.acceptance_criteria = vec!["must be verified".to_string()];
        let result = executor
            .verify(&task, &TaskExecutionOutput::new("done"))
            .await
            .expect("verification result");
        assert!(!result.passed);
        assert_eq!(result.error_class, Some(ErrorClass::FatalError));
        cx.run_until_parked();
    }

    #[gpui::test]
    async fn test_unverified_acceptance_criteria_fail_the_run(cx: &mut gpui::TestAppContext) {
        let executor = Rc::new(MockTaskExecutor::new(|_| {
            Ok(TaskExecutionOutput::new("unchecked output"))
        }));
        let mut task = OrchestrationTask::new("task-1", "Task", "desc");
        task.acceptance_criteria = vec!["must be verified".to_string()];
        task.max_retries = Some(0);
        let plan = OrchestrationPlan::new("Verification", vec![task]);
        let policy = AgentExecutionPolicy {
            strategy: AgentExecutionStrategy::Orchestrate,
            autonomy: AgentAutonomy::Autonomous,
        };
        let mut config = RuntimeConfig::default();
        config.foreground_executor = Some(cx.foreground_executor().clone());
        config.scheduler.background_executor = Some(cx.background_executor.clone());
        let (handle, completion) =
            OrchestrationRuntime::start(plan, policy, executor, config).expect("start run");
        assert_eq!(
            completion.await.expect("receive").expect("complete"),
            RunState::Failed
        );
        assert_eq!(
            handle
                .task_status(&TaskId::new("task-1"))
                .map(|status| status.state),
            Some(TaskState::Failed)
        );
    }

    #[gpui::test]
    async fn test_paused_scheduler_does_not_dispatch(cx: &mut gpui::TestAppContext) {
        let counter = Arc::new(AtomicUsize::new(0));
        let executor = Rc::new(MockTaskExecutor::new({
            let counter = counter.clone();
            move |_| {
                counter.fetch_add(1, Ordering::SeqCst);
                Ok(TaskExecutionOutput::new("done"))
            }
        }));
        let plan = OrchestrationPlan::new(
            "Pause",
            vec![OrchestrationTask::new("task-1", "Task", "desc")],
        );
        let plan_graph = PlanGraph::new(plan).expect("valid plan");
        let control = RuntimeControl::new(RunState::Paused);
        let mut scheduler_config = SchedulerConfig::default();
        scheduler_config.background_executor = Some(cx.background_executor.clone());
        let scheduler = Scheduler::new_with_control(
            RunId::new(),
            plan_graph,
            TaskRegistry::new(),
            ArtifactStore::new(),
            Arc::new(CancellationTree::new()),
            RuntimeEventStream::new(),
            executor,
            scheduler_config,
            control.clone(),
        );
        let completion = cx
            .foreground_executor()
            .spawn(async move { scheduler.run().await });
        cx.run_until_parked();
        assert_eq!(counter.load(Ordering::SeqCst), 0);
        control
            .transition(RunState::Running)
            .expect("resume scheduler");
        assert_eq!(
            completion.await.expect("scheduler completes"),
            RunState::Completed
        );
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[gpui::test]
    async fn dependency_outputs_are_bounded_and_forwarded_to_downstream_tasks(
        cx: &mut gpui::TestAppContext,
    ) {
        let captured = Arc::new(parking_lot::Mutex::new(None));
        let executor = Rc::new(MockTaskExecutor::new({
            let captured = captured.clone();
            move |context| {
                if context.task.id.as_str() == "upstream" {
                    return Ok(
                        TaskExecutionOutput::new("o".repeat(64 * 1024)).with_artifacts(
                            (0..20)
                                .map(|index| {
                                    Artifact::new(
                                        context.task.id.clone(),
                                        format!("artifact-{index}"),
                                        ArtifactKind::Text,
                                        "a".repeat(32 * 1024),
                                    )
                                })
                                .collect(),
                        ),
                    );
                }
                captured.lock().replace(context.dependency_inputs);
                Ok(TaskExecutionOutput::new("done"))
            }
        }));
        let upstream = OrchestrationTask::new("upstream", "Upstream", "produce context");
        let mut downstream = OrchestrationTask::new("downstream", "Downstream", "consume context");
        downstream.depends_on = vec![upstream.id.clone()];
        let plan = OrchestrationPlan::new("Context bundle", vec![upstream, downstream]);
        let mut config = RuntimeConfig::default();
        config.foreground_executor = Some(cx.foreground_executor().clone());
        config.scheduler.background_executor = Some(cx.background_executor.clone());
        let policy = AgentExecutionPolicy {
            strategy: AgentExecutionStrategy::Orchestrate,
            autonomy: AgentAutonomy::Autonomous,
        };
        let (_handle, completion) =
            OrchestrationRuntime::start(plan, policy, executor, config).expect("start runtime");

        assert_eq!(
            completion.await.expect("receive").expect("complete"),
            RunState::Completed
        );
        let inputs = captured.lock().take().expect("dependency inputs captured");
        assert_eq!(inputs.len(), 1);
        assert_eq!(inputs[0].task_id.as_str(), "upstream");
        assert!(
            inputs[0]
                .output
                .as_ref()
                .is_some_and(|output| output.len() <= artifacts::MAX_DEPENDENCY_OUTPUT_BYTES)
        );
        assert!(inputs[0].artifacts.len() <= artifacts::MAX_DEPENDENCY_ARTIFACTS);
        let total_bytes = inputs
            .iter()
            .map(|input| {
                input.output.as_ref().map_or(0, String::len)
                    + input
                        .artifacts
                        .iter()
                        .map(|artifact| artifact.data.len())
                        .sum::<usize>()
            })
            .sum::<usize>();
        assert!(total_bytes <= artifacts::MAX_DEPENDENCY_CONTEXT_BYTES);
    }

    #[gpui::test]
    async fn test_launch_disposition_controls_initial_state(cx: &mut gpui::TestAppContext) {
        let plan = OrchestrationPlan::new(
            "Disposition Test",
            vec![OrchestrationTask::new("t1", "Task 1", "desc")],
        );
        let policy = AgentExecutionPolicy {
            strategy: AgentExecutionStrategy::Orchestrate,
            autonomy: AgentAutonomy::Manual,
        };
        let mut config = RuntimeConfig::default();
        config.foreground_executor = Some(cx.foreground_executor().clone());
        config.scheduler.background_executor = Some(cx.background_executor.clone());

        // Direct / Approved disposition starts in Approved and executes immediately
        let executor = Rc::new(MockTaskExecutor::new(|_| {
            Ok(TaskExecutionOutput::new("done"))
        }));
        let (handle, completion) = OrchestrationRuntime::start_with_disposition(
            plan.clone(),
            policy,
            RuntimeLaunchDisposition::Approved,
            executor,
            config.clone(),
        )
        .expect("start approved run");
        assert_eq!(handle.state(), RunState::Approved);
        assert_eq!(
            completion.await.expect("receive").expect("complete"),
            RunState::Completed
        );

        // AwaitApproval disposition starts in Proposed and waits for approval
        let executor2 = Rc::new(MockTaskExecutor::new(|_| {
            Ok(TaskExecutionOutput::new("done"))
        }));
        let (handle2, completion2) = OrchestrationRuntime::start_with_disposition(
            plan,
            policy,
            RuntimeLaunchDisposition::AwaitApproval,
            executor2,
            config,
        )
        .expect("start await approval run");
        assert_eq!(handle2.state(), RunState::Proposed);
        handle2.approve().expect("approve");
        assert_eq!(
            completion2.await.expect("receive").expect("complete"),
            RunState::Completed
        );
    }

    #[test]
    fn test_auto_policy_single_step_guardrail_prefers_direct() {
        let decision = AutoPolicyEngine::evaluate(
            "fix typo in readme",
            AutoPolicyContext {
                work_item_count: 1,
                available_tool_count: Some(5),
                can_orchestrate: true,
            },
        );
        assert_eq!(decision.strategy, AgentExecutionStrategy::Direct);
        assert!(decision.confidence >= 0.8);
    }

    #[gpui::test]
    async fn test_evidence_required_enforces_verification(cx: &mut gpui::TestAppContext) {
        let mut task = OrchestrationTask::new("task-evidence", "Evidence Task", "Find proof");
        task.evidence_required = true;
        task.max_retries = Some(0);
        let plan = OrchestrationPlan::new("Evidence Verification", vec![task]);
        let policy = AgentExecutionPolicy {
            strategy: AgentExecutionStrategy::Orchestrate,
            autonomy: AgentAutonomy::Autonomous,
        };
        let mut config = RuntimeConfig::default();
        config.foreground_executor = Some(cx.foreground_executor().clone());
        config.scheduler.background_executor = Some(cx.background_executor.clone());

        // Output with no citations fails verification
        let executor_failing = Rc::new(MockTaskExecutor::new(|_| {
            Ok(TaskExecutionOutput::new(
                "I found something but cited no files.",
            ))
        }));
        let (_handle, completion) =
            OrchestrationRuntime::start(plan.clone(), policy, executor_failing, config.clone())
                .expect("start run");
        assert_eq!(
            completion.await.expect("receive").expect("complete"),
            RunState::Failed
        );

        // Output with citations passes verification
        struct CitationVerifier;
        impl TaskExecutor for CitationVerifier {
            fn execute(
                &self,
                _context: TaskExecutionContext,
            ) -> futures::future::LocalBoxFuture<'static, anyhow::Result<TaskExecutionOutput>>
            {
                Box::pin(async move {
                    Ok(TaskExecutionOutput::new(
                        "Verified finding in crates/agent/src/thread.rs:42",
                    ))
                })
            }
            fn verify(
                &self,
                task: &OrchestrationTask,
                output: &TaskExecutionOutput,
            ) -> futures::future::LocalBoxFuture<'static, anyhow::Result<VerificationResult>>
            {
                let has_citation = task.evidence_required && output.output.contains(".rs:");
                Box::pin(async move {
                    if has_citation {
                        Ok(VerificationResult::pass())
                    } else {
                        Ok(VerificationResult::fail(
                            "missing citation",
                            ErrorClass::FatalError,
                        ))
                    }
                })
            }
        }

        let executor_passing = Rc::new(CitationVerifier);
        let (_handle, completion) =
            OrchestrationRuntime::start(plan, policy, executor_passing, config).expect("start run");
        assert_eq!(
            completion.await.expect("receive").expect("complete"),
            RunState::Completed
        );
    }

    #[gpui::test]
    async fn test_token_budget_exceeded_fails_task_without_retry(cx: &mut gpui::TestAppContext) {
        let mut task = OrchestrationTask::new("task-token", "Token Task", "desc");
        task.max_retries = Some(3);
        task.token_budget = Some(100);
        let plan = OrchestrationPlan::new("Token Budget", vec![task]);
        let policy = AgentExecutionPolicy {
            strategy: AgentExecutionStrategy::Orchestrate,
            autonomy: AgentAutonomy::Autonomous,
        };
        let mut config = RuntimeConfig::default();
        config.foreground_executor = Some(cx.foreground_executor().clone());
        config.scheduler.background_executor = Some(cx.background_executor.clone());

        let executor = Rc::new(MockTaskExecutor::new(|_| {
            Ok(TaskExecutionOutput::new("done").with_tokens_used(200))
        }));
        let (handle, completion) =
            OrchestrationRuntime::start(plan, policy, executor, config).expect("start run");
        assert_eq!(
            completion.await.expect("receive").expect("complete"),
            RunState::Failed
        );
        let status = handle
            .task_status_by_session_id(&acp::SessionId::new("done"))
            .or_else(|| {
                handle
                    .task_status(&TaskId::new("task-token"))
                    .or_else(|| handle.task_statuses().into_iter().next())
            });
        let status = status.expect("task status present");
        assert_eq!(status.state, TaskState::Failed);
        assert_eq!(status.tokens_used, 200);
        assert_eq!(status.budget_state.tokens_used, 200);
        assert_eq!(status.phase, None);
        assert_eq!(status.current_tool, None);
        assert_eq!(
            status.budget_state.stopped_reason,
            Some(BudgetExceeded::TokenBudgetExceeded {
                budget: 100,
                used: 200,
            })
        );
        let snapshot = handle.snapshot();
        assert_eq!(snapshot.task_attempts[0].1[0].tokens_used, Some(200));
        assert_eq!(
            snapshot.task_attempts[0].1[0].output.as_deref(),
            Some("done")
        );
    }

    #[gpui::test]
    async fn test_midflight_token_budget_failure_persists_usage(cx: &mut gpui::TestAppContext) {
        let mut task = OrchestrationTask::new("task-token", "Token Task", "desc");
        task.max_retries = Some(3);
        task.token_budget = Some(100);
        let plan = OrchestrationPlan::new("Token Budget", vec![task]);
        let policy = AgentExecutionPolicy {
            strategy: AgentExecutionStrategy::Orchestrate,
            autonomy: AgentAutonomy::Autonomous,
        };
        let mut config = RuntimeConfig::default();
        config.foreground_executor = Some(cx.foreground_executor().clone());
        config.scheduler.background_executor = Some(cx.background_executor.clone());

        struct ReportingExecutor;
        impl TaskExecutor for ReportingExecutor {
            fn execute(
                &self,
                context: TaskExecutionContext,
            ) -> futures::future::LocalBoxFuture<'static, anyhow::Result<TaskExecutionOutput>>
            {
                Box::pin(async move {
                    context.reporter.report_tokens(200)?;
                    Ok(TaskExecutionOutput::new("unreachable"))
                })
            }
        }

        let (handle, completion) =
            OrchestrationRuntime::start(plan, policy, Rc::new(ReportingExecutor), config)
                .expect("start run");
        assert_eq!(
            completion.await.expect("receive").expect("complete"),
            RunState::Failed
        );
        let status = handle
            .task_status(&TaskId::new("task-token"))
            .expect("task status present");
        assert_eq!(status.state, TaskState::Failed);
        assert_eq!(status.tokens_used, 200);
        assert_eq!(status.budget_state.tokens_used, 200);
        assert_eq!(status.phase, None);
        assert_eq!(
            status.budget_state.stopped_reason,
            Some(BudgetExceeded::TokenBudgetExceeded {
                budget: 100,
                used: 200,
            })
        );
        let snapshot = handle.snapshot();
        assert_eq!(snapshot.task_attempts[0].1[0].tokens_used, Some(200));
        assert!(snapshot.task_attempts[0].1[0].output.is_none());
    }

    #[gpui::test]
    async fn test_tool_call_budget_exceeded_fails_task(cx: &mut gpui::TestAppContext) {
        let mut task = OrchestrationTask::new("task-tool", "Tool Task", "desc");
        task.max_retries = Some(3);
        task.tool_call_budget = Some(2);
        let plan = OrchestrationPlan::new("Tool Call Budget", vec![task]);
        let policy = AgentExecutionPolicy {
            strategy: AgentExecutionStrategy::Orchestrate,
            autonomy: AgentAutonomy::Autonomous,
        };
        let mut config = RuntimeConfig::default();
        config.foreground_executor = Some(cx.foreground_executor().clone());
        config.scheduler.background_executor = Some(cx.background_executor.clone());

        struct ToolCallExecutor;
        impl TaskExecutor for ToolCallExecutor {
            fn execute(
                &self,
                context: TaskExecutionContext,
            ) -> futures::future::LocalBoxFuture<'static, anyhow::Result<TaskExecutionOutput>>
            {
                let reporter = context.reporter;
                Box::pin(async move {
                    // Two tool calls exceed the budget of 2.
                    reporter.report_tool_call_started("read_file")?;
                    reporter.report_tool_call_finished();
                    reporter.report_tool_call_started("read_file")?;
                    reporter.report_tool_call_finished();
                    reporter.report_tool_call_started("bash")?;
                    reporter.report_tool_call_finished();
                    Ok(TaskExecutionOutput::new("done"))
                })
            }
        }
        let (handle, completion) =
            OrchestrationRuntime::start(plan, policy, Rc::new(ToolCallExecutor), config)
                .expect("start run");
        assert_eq!(
            completion.await.expect("receive").expect("complete"),
            RunState::Failed
        );
        let status = handle
            .task_status(&TaskId::new("task-tool"))
            .or_else(|| handle.task_statuses().into_iter().next());
        let status = status.expect("task status present");
        assert_eq!(status.state, TaskState::Failed);
        assert_eq!(status.budget_state.tool_calls_used, 3);
        assert_eq!(status.phase, None);
        assert_eq!(status.current_tool, None);
        assert_eq!(
            status.budget_state.stopped_reason,
            Some(BudgetExceeded::ToolCallBudgetExceeded { budget: 2, used: 3 })
        );
    }

    #[gpui::test]
    async fn test_within_budget_completes_task(cx: &mut gpui::TestAppContext) {
        let mut task = OrchestrationTask::new("task-ok", "Ok Task", "desc");
        task.token_budget = Some(1_000);
        task.tool_call_budget = Some(5);
        let plan = OrchestrationPlan::new("Within Budget", vec![task]);
        let policy = AgentExecutionPolicy {
            strategy: AgentExecutionStrategy::Orchestrate,
            autonomy: AgentAutonomy::Autonomous,
        };
        let mut config = RuntimeConfig::default();
        config.foreground_executor = Some(cx.foreground_executor().clone());
        config.scheduler.background_executor = Some(cx.background_executor.clone());

        let executor = Rc::new(MockTaskExecutor::new(|_| {
            Ok(TaskExecutionOutput::new("done").with_tokens_used(50))
        }));
        let (handle, completion) =
            OrchestrationRuntime::start(plan, policy, executor, config).expect("start run");
        assert_eq!(
            completion.await.expect("receive").expect("complete"),
            RunState::Completed
        );
        let status = handle
            .task_status(&TaskId::new("task-ok"))
            .or_else(|| handle.task_statuses().into_iter().next());
        let status = status.expect("task status present");
        assert_eq!(status.state, TaskState::Completed);
        assert!(status.budget_state.stopped_reason.is_none());
    }

    #[gpui::test]
    async fn test_repair_usage_is_included_in_attempt_total(cx: &mut gpui::TestAppContext) {
        let mut task = OrchestrationTask::new("task-repair", "Repair Task", "desc");
        task.acceptance_criteria = vec!["result is valid".to_string()];
        task.token_budget = Some(1_000);
        let plan = OrchestrationPlan::new("Repair Budget", vec![task]);
        let policy = AgentExecutionPolicy {
            strategy: AgentExecutionStrategy::Orchestrate,
            autonomy: AgentAutonomy::Autonomous,
        };
        let mut config = RuntimeConfig::default();
        config.foreground_executor = Some(cx.foreground_executor().clone());
        config.scheduler.background_executor = Some(cx.background_executor.clone());

        struct RepairExecutor {
            verification_count: AtomicUsize,
        }

        impl TaskExecutor for RepairExecutor {
            fn execute(
                &self,
                _context: TaskExecutionContext,
            ) -> futures::future::LocalBoxFuture<'static, anyhow::Result<TaskExecutionOutput>>
            {
                Box::pin(
                    async move { Ok(TaskExecutionOutput::new("initial").with_tokens_used(100)) },
                )
            }

            fn verify(
                &self,
                _task: &OrchestrationTask,
                _output: &TaskExecutionOutput,
            ) -> futures::future::LocalBoxFuture<'static, anyhow::Result<VerificationResult>>
            {
                let verification_count = self.verification_count.fetch_add(1, Ordering::SeqCst);
                Box::pin(async move {
                    if verification_count == 0 {
                        Ok(VerificationResult::fail(
                            "repair required",
                            ErrorClass::VerificationAssertionFailure,
                        ))
                    } else {
                        Ok(VerificationResult::pass())
                    }
                })
            }

            fn repair(
                &self,
                _task: &OrchestrationTask,
                _feedback: &str,
                _context: TaskExecutionContext,
            ) -> futures::future::LocalBoxFuture<'static, anyhow::Result<TaskExecutionOutput>>
            {
                Box::pin(
                    async move { Ok(TaskExecutionOutput::new("repaired").with_tokens_used(150)) },
                )
            }
        }

        let executor = Rc::new(RepairExecutor {
            verification_count: AtomicUsize::new(0),
        });
        let (handle, completion) =
            OrchestrationRuntime::start(plan, policy, executor, config).expect("start run");
        assert_eq!(
            completion.await.expect("receive").expect("complete"),
            RunState::Completed
        );
        let status = handle
            .task_status(&TaskId::new("task-repair"))
            .expect("task status present");
        assert_eq!(status.tokens_used, 250);
        assert_eq!(status.budget_state.tokens_used, 250);
        assert_eq!(status.phase, None);
        let snapshot = handle.snapshot();
        assert_eq!(snapshot.task_attempts[0].1[0].tokens_used, Some(250));
        assert_eq!(
            snapshot.task_attempts[0].1[0].output.as_deref(),
            Some("repaired")
        );
    }

    #[test]
    fn test_worker_target_serialization_and_deserialization() {
        let native = WorkerTarget::Native;
        let json = serde_json::to_string(&native).expect("serialize native");
        assert!(json.contains("native"));
        let parsed: WorkerTarget = serde_json::from_str(&json).expect("deserialize native");
        assert_eq!(parsed, WorkerTarget::Native);

        let from_str: WorkerTarget =
            serde_json::from_str("\"native\"").expect("deserialize string native");
        assert_eq!(from_str, WorkerTarget::Native);

        let from_omp: WorkerTarget =
            serde_json::from_str("\"omp\"").expect("deserialize string omp");
        assert_eq!(
            from_omp,
            WorkerTarget::Acp {
                agent_id: "omp".to_string()
            }
        );

        let acp = WorkerTarget::Acp {
            agent_id: "opencode".to_string(),
        };
        let json_acp = serde_json::to_string(&acp).expect("serialize acp");
        let parsed_acp: WorkerTarget = serde_json::from_str(&json_acp).expect("deserialize acp");
        assert_eq!(parsed_acp, acp);
    }

    #[test]
    fn test_legacy_persisted_run_defaults_to_native_target() {
        let legacy_json = r#"{
            "schema_version": 2,
            "run_id": "run-legacy-1",
            "plan": {
                "id": "plan-legacy-1",
                "title": "Legacy Plan",
                "tasks": [
                    {
                        "id": "task-1",
                        "label": "Legacy Task",
                        "description": "A task from an older snapshot",
                        "depends_on": [],
                        "acceptance_criteria": [],
                        "context_paths": [],
                        "repair_on_failure": true,
                        "evidence_required": false
                    }
                ]
            },
            "policy": { "strategy": "direct", "autonomy": "manual" },
            "state": "completed",
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:01:00Z",
            "task_statuses": [
                {
                    "task_id": "task-1",
                    "state": "completed",
                    "current_attempt": 1,
                    "total_attempts": 1,
                    "tokens_used": 100,
                    "updated_at": "2026-01-01T00:01:00Z",
                    "budget_state": { "tokens_used": 100, "tool_calls_used": 1 }
                }
            ],
            "task_attempts": [],
            "event_log": [],
            "last_event_seq": 0
        }"#;

        let run = PersistedRun::from_json(legacy_json).expect("deserialize legacy snapshot");
        assert_eq!(run.plan.tasks[0].target, WorkerTarget::Native);
        assert_eq!(run.task_statuses[0].target, WorkerTarget::Native);
        assert_eq!(
            run.plan.tasks[0].workspace_policy.isolation,
            WorkspaceIsolation::SharedParent
        );
    }

    #[gpui::test]
    async fn test_unmanaged_write_output_fails_closed_and_blocks_downstream(
        cx: &mut gpui::TestAppContext,
    ) {
        let mut task_write = OrchestrationTask::new("task-write", "Write task", "Do work");
        task_write.workspace_policy.isolation = WorkspaceIsolation::DedicatedWorktree;

        let task_downstream =
            OrchestrationTask::new("task-downstream", "Downstream task", "Review work")
                .with_depends_on(vec![TaskId::new("task-write")]);

        let plan = OrchestrationPlan::new("Isolated pipeline", vec![task_write, task_downstream]);

        let (write_finished_tx, write_finished_rx) = async_channel::bounded(1);
        let (downstream_started_tx, downstream_started_rx) = async_channel::bounded(1);

        let executor = Rc::new(MockTaskExecutor::new({
            let write_finished_tx = write_finished_tx.clone();
            let downstream_started_tx = downstream_started_tx.clone();
            move |context| {
                if context.task.id.as_str() == "task-write" {
                    let _ = write_finished_tx.try_send(());
                    Ok(TaskExecutionOutput::new("write done"))
                } else {
                    let _ = downstream_started_tx.try_send(());
                    Ok(TaskExecutionOutput::new("downstream done"))
                }
            }
        }));

        let mut config = RuntimeConfig::default();
        config.foreground_executor = Some(cx.foreground_executor().clone());
        config.scheduler.background_executor = Some(cx.background_executor.clone());

        let policy = AgentExecutionPolicy {
            strategy: AgentExecutionStrategy::Orchestrate,
            autonomy: AgentAutonomy::Autonomous,
        };

        let (handle, completion) =
            OrchestrationRuntime::start(plan, policy, executor, config).expect("start runtime");

        let write_finished = write_finished_rx.recv();
        let timeout1 = cx.background_executor.timer(Duration::from_secs(2));
        futures::pin_mut!(write_finished, timeout1);
        match futures::future::select(write_finished, timeout1).await {
            futures::future::Either::Left((res, _)) => {
                res.expect("write finished");
            }
            futures::future::Either::Right(_) => panic!("task-write execution timed out"),
        }

        // A write executor that does not return a managed worktree must fail
        // closed instead of exposing a fake Apply action.
        assert!(downstream_started_rx.try_recv().is_err());
        let write_status = handle
            .task_status(&TaskId::new("task-write"))
            .expect("status");
        assert_eq!(write_status.state, TaskState::Failed);
        assert!(
            write_status
                .latest_error
                .as_deref()
                .is_some_and(|error| error.contains("managed worktree descriptor"))
        );
        let downstream_status = handle
            .task_status(&TaskId::new("task-downstream"))
            .expect("status");
        assert_eq!(downstream_status.state, TaskState::Blocked);

        let final_state = completion.await.expect("receive").expect("complete");
        assert_eq!(final_state, RunState::Failed);
    }

    #[test]
    fn test_worker_broker_parameter_validation() {
        let broker_disabled = WorkerBroker::new(false);
        let mut acp_task = OrchestrationTask::new("acp-1", "ACP task", "run");
        acp_task.target = WorkerTarget::Acp {
            agent_id: "omp".to_string(),
        };

        // Rejects when feature flag is off
        let err = broker_disabled.validate_task_parameters(&acp_task);
        assert!(err.is_err());
        assert!(
            err.unwrap_err()
                .to_string()
                .contains("ACP delegation is currently disabled")
        );

        // When enabled, rejects unknown / unconfigured worker
        let broker_enabled = WorkerBroker::new(true);
        let err2 = broker_enabled.validate_task_parameters(&acp_task);
        assert!(err2.is_err());
        assert!(
            err2.unwrap_err()
                .to_string()
                .contains("is not configured or available")
        );

        // Rejects invalid mode on native task
        let mut native_task = OrchestrationTask::new("native-1", "Native task", "run");
        native_task.mode = Some("completely_unknown_mode".to_string());
        let err_native = broker_enabled.validate_task_parameters(&native_task);
        assert!(err_native.is_err());
        assert!(
            err_native
                .unwrap_err()
                .to_string()
                .contains("is not supported for native agent")
        );
        native_task.mode = Some("ask".to_string());
        assert!(
            broker_enabled
                .validate_task_parameters(&native_task)
                .is_err()
        );
        native_task.mode = None;
        native_task.model_override = Some("requested-model".to_string());
        assert!(
            broker_enabled
                .validate_task_parameters(&native_task)
                .is_err()
        );
    }

    #[test]
    fn cancelled_task_rejects_late_completion_and_failure() {
        let registry = TaskRegistry::new();
        let task_id = TaskId::new("cancelled-worker");
        registry.register_task(task_id.clone());
        let attempt = registry.start_attempt(&task_id, None);
        registry.set_state(&task_id, TaskState::Cancelled);
        registry.complete_attempt(
            &task_id,
            attempt,
            Some("late output".into()),
            Some(100),
            None,
        );
        registry.fail_attempt(&task_id, attempt, "late error".into(), false);
        let status = registry.status(&task_id).expect("registered task");
        assert_eq!(status.state, TaskState::Cancelled);
        assert!(status.latest_output.is_none());
        assert_eq!(status.tokens_used, 0);
    }

    #[test]
    fn test_late_attempt_completion_discarded() {
        let registry = TaskRegistry::new();
        let task_id = TaskId::new("task-corr-1");
        registry.register_task(task_id.clone());

        // Start attempt 1
        registry.start_attempt(&task_id, None);
        assert_eq!(registry.status(&task_id).unwrap().current_attempt, 1);

        // Start attempt 2 (e.g. after retry)
        registry.start_attempt(&task_id, None);
        assert_eq!(registry.status(&task_id).unwrap().current_attempt, 2);
        assert_eq!(registry.status(&task_id).unwrap().state, TaskState::Running);

        // Late completion from attempt 1 arrives!
        registry.complete_attempt(
            &task_id,
            1,
            Some("late output from attempt 1".to_string()),
            Some(50),
            Some(VerificationResult::pass()),
        );

        // Verify active status was NOT corrupted: state remains Running on attempt 2!
        let status = registry.status(&task_id).unwrap();
        assert_eq!(status.current_attempt, 2);
        assert_eq!(status.state, TaskState::Running);
        assert_ne!(
            status.latest_output.as_deref(),
            Some("late output from attempt 1")
        );

        // Now attempt 2 completes
        registry.complete_attempt(
            &task_id,
            2,
            Some("valid output from attempt 2".to_string()),
            Some(100),
            Some(VerificationResult::pass()),
        );

        let status2 = registry.status(&task_id).unwrap();
        assert_eq!(status2.state, TaskState::Completed);
        assert_eq!(
            status2.latest_output.as_deref(),
            Some("valid output from attempt 2")
        );
    }

    #[test]
    fn test_reconcile_on_restart_clears_active_states() {
        let run_id = RunId::new();
        let task1 = OrchestrationTask::new("t1", "T1", "desc");
        let task2 = OrchestrationTask::new("t2", "T2", "desc");
        let plan = OrchestrationPlan::new("Plan", vec![task1, task2]);

        let mut s1 = TaskStatus::new(TaskId::new("t1"));
        s1.state = TaskState::Running;
        s1.active_session_id = Some(acp::SessionId::new("sess-1"));
        let mut meta = WorkerMetadata::new(WorkerTarget::acp("omp"));
        meta.capabilities.can_resume = true;
        s1.worker_metadata = Some(meta);

        let mut s2 = TaskStatus::new(TaskId::new("t2"));
        s2.state = TaskState::Running;
        // Worker does not support resume
        let mut meta2 = WorkerMetadata::new(WorkerTarget::Native);
        meta2.capabilities.can_resume = false;
        s2.worker_metadata = Some(meta2);

        let run = PersistedRun::new(
            run_id,
            plan,
            RunState::Running,
            AgentExecutionPolicy::default(),
            vec![s1, s2],
            Vec::new(),
            Vec::new(),
            Vec::new(),
        );

        let reconciled = run.reconcile_on_restart();
        assert_eq!(reconciled.state, RunState::Interrupted);

        // t1 had resume capability -> Parked awaiting reconnect
        assert_eq!(reconciled.task_statuses[0].state, TaskState::Parked);
        assert!(matches!(
            reconciled.task_statuses[0].wait_reason,
            Some(StructuredWaitReason::AwaitingWorkerReconnect { .. })
        ));

        // t2 could not resume -> Interrupted requiring restart
        assert_eq!(reconciled.task_statuses[1].state, TaskState::Interrupted);
        assert!(
            reconciled.task_statuses[1]
                .latest_error
                .as_ref()
                .unwrap()
                .contains("worker does not support resume")
        );
    }

    #[test]
    fn test_reconcile_preserves_approval_and_pause_boundaries() {
        for state in [RunState::Proposed, RunState::Paused] {
            let run = PersistedRun::new(
                RunId::new(),
                OrchestrationPlan::new("Plan", vec![OrchestrationTask::new("t1", "T1", "desc")]),
                state,
                AgentExecutionPolicy::default(),
                vec![TaskStatus::new(TaskId::new("t1"))],
                Vec::new(),
                Vec::new(),
                Vec::new(),
            );

            assert_eq!(run.reconcile_on_restart().state, state);
        }
    }

    #[test]
    fn test_reconcile_resumes_independent_work_before_waiting_for_apply() {
        let mut awaiting = TaskStatus::new(TaskId::new("awaiting"));
        awaiting.state = TaskState::AwaitingApply;
        let pending = TaskStatus::new(TaskId::new("pending"));
        let run = PersistedRun::new(
            RunId::new(),
            OrchestrationPlan::new(
                "Plan",
                vec![
                    OrchestrationTask::new("awaiting", "Awaiting", "desc"),
                    OrchestrationTask::new("pending", "Pending", "desc"),
                ],
            ),
            RunState::AwaitingApply,
            AgentExecutionPolicy::default(),
            vec![awaiting, pending],
            Vec::new(),
            Vec::new(),
            Vec::new(),
        );

        assert_eq!(run.reconcile_on_restart().state, RunState::Interrupted);
    }

    #[gpui::test]
    async fn test_unmanaged_write_cannot_be_rejected_as_a_real_patch(
        cx: &mut gpui::TestAppContext,
    ) {
        let mut task_write = OrchestrationTask::new("task-write", "Write task", "Do work");
        task_write.workspace_policy.isolation = WorkspaceIsolation::DedicatedWorktree;

        let task_downstream =
            OrchestrationTask::new("task-downstream", "Downstream task", "Follow up")
                .with_depends_on(vec![TaskId::new("task-write")]);

        let plan = OrchestrationPlan::new("Rejection pipeline", vec![task_write, task_downstream]);

        let (write_finished_tx, write_finished_rx) = async_channel::bounded(1);
        let downstream_started = Arc::new(std::sync::atomic::AtomicBool::new(false));

        let executor = Rc::new(MockTaskExecutor::new({
            let write_finished_tx = write_finished_tx.clone();
            let downstream_started = downstream_started.clone();
            move |context| {
                if context.task.id.as_str() == "task-write" {
                    let _ = write_finished_tx.try_send(());
                    Ok(TaskExecutionOutput::new("write proposal"))
                } else {
                    downstream_started.store(true, Ordering::SeqCst);
                    Ok(TaskExecutionOutput::new("downstream executed"))
                }
            }
        }));

        let mut config = RuntimeConfig::default();
        config.foreground_executor = Some(cx.foreground_executor().clone());
        config.scheduler.background_executor = Some(cx.background_executor.clone());

        let policy = AgentExecutionPolicy {
            strategy: AgentExecutionStrategy::Orchestrate,
            autonomy: AgentAutonomy::Autonomous,
        };

        let (handle, completion) =
            OrchestrationRuntime::start(plan, policy, executor, config).expect("start runtime");

        let write_finished = write_finished_rx.recv();
        let timeout = cx.background_executor.timer(Duration::from_secs(2));
        futures::pin_mut!(write_finished, timeout);
        match futures::future::select(write_finished, timeout).await {
            futures::future::Either::Left((res, _)) => {
                res.expect("write finished");
            }
            futures::future::Either::Right(_) => panic!("write timed out"),
        }

        // Missing managed-worktree metadata is terminal and cannot be rejected
        // as though a real isolated patch existed.
        let status = handle
            .task_status(&TaskId::new("task-write"))
            .expect("status");
        assert_eq!(status.state, TaskState::Failed);
        assert!(
            !handle
                .reject_task(
                    &TaskId::new("task-write"),
                    Some("user disliked change".to_string()),
                )
                .await
                .expect("reject")
        );

        // Verify downstream NEVER executes
        assert!(!downstream_started.load(Ordering::SeqCst));
        let final_state = completion.await.expect("receive").expect("completion");
        assert_eq!(final_state, RunState::Failed);

        let write_status = handle
            .task_status(&TaskId::new("task-write"))
            .expect("status");
        assert_eq!(write_status.state, TaskState::Failed);

        let downstream_status = handle
            .task_status(&TaskId::new("task-downstream"))
            .expect("status");
        assert_eq!(downstream_status.state, TaskState::Blocked);
    }

    #[gpui::test]
    async fn test_unmanaged_write_does_not_expose_fake_diff_or_apply(
        cx: &mut gpui::TestAppContext,
    ) {
        let mut task_write = OrchestrationTask::new("task-write", "Write task", "Create patch");
        task_write.workspace_policy.isolation = WorkspaceIsolation::DedicatedWorktree;

        let task_parallel =
            OrchestrationTask::new("task-parallel", "Read task", "Analyze codebase");

        let task_downstream =
            OrchestrationTask::new("task-downstream", "Downstream task", "Apply feedback")
                .with_depends_on(vec![TaskId::new("task-write")]);

        let plan = OrchestrationPlan::new(
            "Full Acceptance Pipeline",
            vec![task_write, task_parallel, task_downstream],
        );

        let (write_tx, write_rx) = async_channel::bounded(1);
        let (parallel_tx, parallel_rx) = async_channel::bounded(1);
        let (downstream_tx, downstream_rx) = async_channel::bounded(1);

        let executor = Rc::new(MockTaskExecutor::new({
            let write_tx = write_tx.clone();
            let parallel_tx = parallel_tx.clone();
            let downstream_tx = downstream_tx.clone();
            move |context| match context.task.id.as_str() {
                "task-write" => {
                    let _ = write_tx.try_send(());
                    Ok(TaskExecutionOutput::new("write proposal ready"))
                }
                "task-parallel" => {
                    let _ = parallel_tx.try_send(());
                    Ok(TaskExecutionOutput::new("analysis ready"))
                }
                "task-downstream" => {
                    let _ = downstream_tx.try_send(());
                    Ok(TaskExecutionOutput::new("downstream completed"))
                }
                _ => Ok(TaskExecutionOutput::new("done")),
            }
        }));

        let mut config = RuntimeConfig::default();
        config.foreground_executor = Some(cx.foreground_executor().clone());
        config.scheduler.background_executor = Some(cx.background_executor.clone());
        config.scheduler.max_parallel_tasks = 4;

        let policy = AgentExecutionPolicy {
            strategy: AgentExecutionStrategy::Orchestrate,
            autonomy: AgentAutonomy::Autonomous,
        };

        let (handle, completion) =
            OrchestrationRuntime::start(plan, policy, executor, config).expect("start runtime");

        // 1. Parallel and write tasks execute
        let write_fut = write_rx.recv();
        let timeout1 = cx.background_executor.timer(Duration::from_secs(2));
        futures::pin_mut!(write_fut, timeout1);
        match futures::future::select(write_fut, timeout1).await {
            futures::future::Either::Left((res, _)) => res.expect("write finished"),
            futures::future::Either::Right(_) => panic!("write timed out"),
        }

        let parallel_fut = parallel_rx.recv();
        let timeout2 = cx.background_executor.timer(Duration::from_secs(2));
        futures::pin_mut!(parallel_fut, timeout2);
        match futures::future::select(parallel_fut, timeout2).await {
            futures::future::Either::Left((res, _)) => res.expect("parallel finished"),
            futures::future::Either::Right(_) => panic!("parallel timed out"),
        }

        // 2. task-parallel completed; the isolated write fails closed because
        // this mock did not create a managed worktree.
        let parallel_status = handle
            .task_status(&TaskId::new("task-parallel"))
            .expect("status");
        assert_eq!(parallel_status.state, TaskState::Completed);

        let write_status = handle
            .task_status(&TaskId::new("task-write"))
            .expect("status");
        assert_eq!(write_status.state, TaskState::Failed);

        // 3. task-downstream is blocked and never executes.
        let downstream_status = handle
            .task_status(&TaskId::new("task-downstream"))
            .expect("status");
        assert_eq!(downstream_status.state, TaskState::Blocked);
        assert!(downstream_rx.try_recv().is_err());

        assert!(
            handle
                .review_diff(&TaskId::new("task-write"))
                .await
                .expect_err("no fake diff should be available")
                .to_string()
                .contains("no managed worktree")
        );

        let final_state = completion.await.expect("receive").expect("completion");
        assert_eq!(final_state, RunState::Failed);
        assert!(
            handle
                .task_statuses()
                .iter()
                .all(|status| !status.state.is_active())
        );
    }

    async fn git(directory: &std::path::Path, arguments: &[&str]) -> anyhow::Result<()> {
        let output = smol::process::Command::new("git")
            .arg("-C")
            .arg(directory)
            .args(arguments)
            .env("GIT_CONFIG_NOSYSTEM", "1")
            .env("GIT_CONFIG_GLOBAL", "/dev/null")
            .output()
            .await?;
        anyhow::ensure!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        Ok(())
    }

    #[gpui::test]
    async fn test_apply_task_with_post_apply_verification_failure_rolls_back_and_parks(
        cx: &mut gpui::TestAppContext,
    ) {
        cx.executor().allow_parking();
        let root =
            std::env::temp_dir().join(format!("test-post-apply-rollback-{}", uuid::Uuid::new_v4()));
        let parent = root.join("parent");
        std::fs::create_dir_all(&parent).unwrap();
        git(&parent, &["init", "--quiet"]).await.unwrap();
        std::fs::write(parent.join("hello.txt"), "original parent content\n").unwrap();
        git(&parent, &["add", "."]).await.unwrap();
        git(
            &parent,
            &[
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.invalid",
                "commit",
                "-m",
                "Initial",
            ],
        )
        .await
        .unwrap();
        std::fs::write(parent.join("hello.txt"), "staged parent content\n").unwrap();
        git(&parent, &["add", "hello.txt"]).await.unwrap();
        std::fs::write(parent.join("hello.txt"), "working parent content\n").unwrap();
        let index_before_apply = std::fs::read(parent.join(".git/index")).unwrap();

        let mut task = OrchestrationTask::new("task-1", "Write task", "edit file");
        task.workspace_policy = WorkspacePolicy::isolated_worktree();
        task.verification_command = Some("test ! -f parent_canary.txt".to_string());
        let plan = OrchestrationPlan::new("Verification Failure", vec![task]);

        let (applied_tx, applied_rx) = async_channel::bounded(1);
        let parent_clone = parent.clone();
        let root_clone = root.clone();

        let executor = Rc::new(MockTaskExecutor::new(move |context| {
            let parent_path = parent_clone.clone();
            let root_path = root_clone.clone();
            let applied_tx = applied_tx.clone();
            smol::block_on(async move {
                let manager = crate::worktree_isolation::WorktreeManager::new(
                    &parent_path,
                    root_path.join("workers"),
                );
                let worker = manager
                    .create_isolated_worktree(&context.run_id, &context.task.id, context.attempt)
                    .await?;

                std::fs::write(
                    worker.worktree_path.join("hello.txt"),
                    "modified by worker\n",
                )?;

                // Introduce parent_canary.txt only in parent checkout so Tier 2 passes in worktree, but Tier 3 fails in parent
                std::fs::write(parent_path.join("parent_canary.txt"), "canary\n")?;

                let mut metadata =
                    crate::worker::WorkerMetadata::new(crate::worker::WorkerTarget::Native);
                metadata.worktree_path = Some(worker.worktree_path.to_string_lossy().to_string());
                metadata.baseline_commit = Some(worker.baseline_commit.clone());
                context
                    .reporter
                    .report_worker_started(None, metadata.clone());

                let _ = applied_tx.try_send(worker.worktree_path);
                Ok(TaskExecutionOutput::new("worker finished").with_worker_metadata(metadata))
            })
        }));

        let mut config = RuntimeConfig::default();
        config.foreground_executor = Some(cx.foreground_executor().clone());
        config.scheduler.background_executor = Some(cx.background_executor.clone());

        let policy = AgentExecutionPolicy {
            strategy: AgentExecutionStrategy::Orchestrate,
            autonomy: AgentAutonomy::Autonomous,
        };

        let (handle, _completion) =
            OrchestrationRuntime::start(plan, policy, executor, config).expect("start runtime");

        let wt_path = applied_rx.recv().await.expect("worker finished");

        let mut awaiting = false;
        for _ in 0..20 {
            if let Some(status) = handle.task_status(&TaskId::new("task-1")) {
                if status.state == TaskState::AwaitingApply {
                    awaiting = true;
                    break;
                }
            }
            cx.background_executor
                .timer(Duration::from_millis(50))
                .await;
        }
        assert!(awaiting, "task should enter AwaitingApply");

        // Calling apply_task should fail post-apply verification
        let apply_result = handle.apply_task(&TaskId::new("task-1")).await;
        assert!(apply_result.is_err());
        let err = apply_result.unwrap_err().to_string();
        assert!(err.contains("parent post-apply verification failed"));

        // Parent repo must be rolled back to original content
        assert_eq!(
            std::fs::read_to_string(parent.join("hello.txt")).unwrap(),
            "working parent content\n"
        );
        assert_eq!(
            std::fs::read(parent.join(".git/index")).unwrap(),
            index_before_apply,
            "rollback must preserve the parent's staged index state"
        );

        // Task must be parked with PostApplyVerificationFailed
        let status = handle.task_status(&TaskId::new("task-1")).unwrap();
        assert_eq!(status.state, TaskState::Parked);
        assert!(matches!(
            status.wait_reason,
            Some(crate::worker::StructuredWaitReason::PostApplyVerificationFailed { .. })
        ));

        // Worktree must be retained on disk
        assert!(
            wt_path.exists(),
            "worktree must be retained after verification failure"
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    #[gpui::test]
    async fn test_apply_task_with_tier2_worktree_verification_failure_parks_before_apply(
        cx: &mut gpui::TestAppContext,
    ) {
        cx.executor().allow_parking();
        let root =
            std::env::temp_dir().join(format!("test-tier2-verification-{}", uuid::Uuid::new_v4()));
        let parent = root.join("parent");
        std::fs::create_dir_all(&parent).unwrap();
        git(&parent, &["init", "--quiet"]).await.unwrap();
        std::fs::write(parent.join("hello.txt"), "original\n").unwrap();
        git(&parent, &["add", "."]).await.unwrap();
        git(
            &parent,
            &[
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.invalid",
                "commit",
                "-m",
                "Initial",
            ],
        )
        .await
        .unwrap();

        let mut task = OrchestrationTask::new("task-1", "Write task", "edit file");
        task.workspace_policy = WorkspacePolicy::isolated_worktree();
        task.verification_command = Some("false".to_string());
        let plan = OrchestrationPlan::new("Tier 2 Failure", vec![task]);

        let (applied_tx, applied_rx) = async_channel::bounded(1);
        let parent_clone = parent.clone();
        let root_clone = root.clone();

        let executor = Rc::new(MockTaskExecutor::new(move |context| {
            let parent_path = parent_clone.clone();
            let root_path = root_clone.clone();
            let applied_tx = applied_tx.clone();
            smol::block_on(async move {
                let manager = crate::worktree_isolation::WorktreeManager::new(
                    &parent_path,
                    root_path.join("workers"),
                );
                let worker = manager
                    .create_isolated_worktree(&context.run_id, &context.task.id, context.attempt)
                    .await?;

                std::fs::write(worker.worktree_path.join("hello.txt"), "modified\n")?;

                let mut metadata =
                    crate::worker::WorkerMetadata::new(crate::worker::WorkerTarget::Native);
                metadata.worktree_path = Some(worker.worktree_path.to_string_lossy().to_string());
                metadata.baseline_commit = Some(worker.baseline_commit.clone());
                context
                    .reporter
                    .report_worker_started(None, metadata.clone());

                let _ = applied_tx.try_send(worker.worktree_path);
                Ok(TaskExecutionOutput::new("worker finished").with_worker_metadata(metadata))
            })
        }));

        let mut config = RuntimeConfig::default();
        config.foreground_executor = Some(cx.foreground_executor().clone());
        config.scheduler.background_executor = Some(cx.background_executor.clone());

        let policy = AgentExecutionPolicy {
            strategy: AgentExecutionStrategy::Orchestrate,
            autonomy: AgentAutonomy::Autonomous,
        };

        let (handle, _completion) =
            OrchestrationRuntime::start(plan, policy, executor, config).expect("start runtime");

        let wt_path = applied_rx.recv().await.expect("worker finished");

        let mut awaiting = false;
        for _ in 0..20 {
            if let Some(status) = handle.task_status(&TaskId::new("task-1")) {
                if status.state == TaskState::AwaitingApply {
                    awaiting = true;
                    break;
                }
            }
            cx.background_executor
                .timer(Duration::from_millis(50))
                .await;
        }
        assert!(awaiting, "task should enter AwaitingApply");

        // Calling apply_task should fail Tier 2 worktree verification
        let apply_result = handle.apply_task(&TaskId::new("task-1")).await;
        assert!(apply_result.is_err());
        let err = apply_result.unwrap_err().to_string();
        assert!(err.contains("worktree verification failed"));

        // Parent repo must be untouched
        assert_eq!(
            std::fs::read_to_string(parent.join("hello.txt")).unwrap(),
            "original\n"
        );

        // Tier 2 verification has its own wait reason because no parent patch was applied.
        let status = handle.task_status(&TaskId::new("task-1")).unwrap();
        assert_eq!(status.state, TaskState::Parked);
        assert!(matches!(
            status.wait_reason,
            Some(crate::worker::StructuredWaitReason::WorktreeVerificationFailed { .. })
        ));

        // Worktree must be retained on disk
        assert!(wt_path.exists());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[gpui::test]
    async fn test_apply_task_respects_manual_pause(cx: &mut gpui::TestAppContext) {
        cx.executor().allow_parking();
        let root =
            std::env::temp_dir().join(format!("test-manual-pause-apply-{}", uuid::Uuid::new_v4()));
        let parent = root.join("parent");
        std::fs::create_dir_all(&parent).unwrap();
        git(&parent, &["init", "--quiet"]).await.unwrap();
        std::fs::write(parent.join("hello.txt"), "original\n").unwrap();
        git(&parent, &["add", "."]).await.unwrap();
        git(
            &parent,
            &[
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.invalid",
                "commit",
                "-m",
                "Initial",
            ],
        )
        .await
        .unwrap();

        let mut task = OrchestrationTask::new("task-1", "Write task", "edit file");
        task.workspace_policy = WorkspacePolicy::isolated_worktree();
        let plan = OrchestrationPlan::new("Manual Pause Test", vec![task]);

        let (applied_tx, applied_rx) = async_channel::bounded(1);
        let parent_clone = parent.clone();
        let root_clone = root.clone();

        let executor = Rc::new(MockTaskExecutor::new(move |context| {
            let parent_path = parent_clone.clone();
            let root_path = root_clone.clone();
            let applied_tx = applied_tx.clone();
            smol::block_on(async move {
                let manager = crate::worktree_isolation::WorktreeManager::new(
                    &parent_path,
                    root_path.join("workers"),
                );
                let worker = manager
                    .create_isolated_worktree(&context.run_id, &context.task.id, context.attempt)
                    .await?;

                std::fs::write(worker.worktree_path.join("hello.txt"), "modified\n")?;

                let mut metadata =
                    crate::worker::WorkerMetadata::new(crate::worker::WorkerTarget::Native);
                metadata.worktree_path = Some(worker.worktree_path.to_string_lossy().to_string());
                metadata.baseline_commit = Some(worker.baseline_commit.clone());
                context
                    .reporter
                    .report_worker_started(None, metadata.clone());

                let _ = applied_tx.try_send(worker.worktree_path);
                Ok(TaskExecutionOutput::new("worker finished").with_worker_metadata(metadata))
            })
        }));

        let mut config = RuntimeConfig::default();
        config.foreground_executor = Some(cx.foreground_executor().clone());
        config.scheduler.background_executor = Some(cx.background_executor.clone());

        let policy = AgentExecutionPolicy {
            strategy: AgentExecutionStrategy::Orchestrate,
            autonomy: AgentAutonomy::Autonomous,
        };

        let (handle, completion) =
            OrchestrationRuntime::start(plan, policy, executor, config).expect("start runtime");

        let _wt_path = applied_rx.recv().await.expect("worker finished");

        let mut awaiting = false;
        for _ in 0..20 {
            if let Some(status) = handle.task_status(&TaskId::new("task-1")) {
                if status.state == TaskState::AwaitingApply {
                    awaiting = true;
                    break;
                }
            }
            cx.background_executor
                .timer(Duration::from_millis(50))
                .await;
        }
        assert!(awaiting, "task should enter AwaitingApply");

        // User manually pauses the run
        handle.pause();
        assert_eq!(handle.state(), RunState::Paused);

        // Apply task while manually paused
        let applied = handle
            .apply_task(&TaskId::new("task-1"))
            .await
            .expect("apply");
        assert!(applied);

        // Run state MUST remain Paused, not auto-resumed to Running
        assert_eq!(handle.state(), RunState::Paused);

        // Task itself is completed
        assert_eq!(
            handle.task_status(&TaskId::new("task-1")).unwrap().state,
            TaskState::Completed
        );

        // Parent repo has modified content
        assert_eq!(
            std::fs::read_to_string(parent.join("hello.txt")).unwrap(),
            "modified\n"
        );

        // Resuming unpauses to completion
        handle.resume();
        let final_state = completion.await.expect("receive").expect("completion");
        assert_eq!(final_state, RunState::Completed);

        let _ = std::fs::remove_dir_all(&root);
    }
}
