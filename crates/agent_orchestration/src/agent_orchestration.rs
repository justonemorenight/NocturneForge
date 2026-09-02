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

pub use acp_adapter::{AcpEventAdapter, AcpTaskBridge};
pub use artifacts::{
    Artifact, ArtifactKind, ArtifactStore, ContextSnapshot, MAX_INLINE_OUTPUT_BYTES, truncate_text,
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
pub use executor::{MockTaskExecutor, TaskExecutionContext, TaskExecutionOutput, TaskExecutor};
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
        assert_eq!(resumed.task_statuses[1].state, TaskState::Interrupted);
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
        assert!(
            status.budget_state.stopped_reason.is_some(),
            "budget stop reason recorded"
        );
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
        assert!(
            status.budget_state.stopped_reason.is_some(),
            "budget stop reason recorded"
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
}
