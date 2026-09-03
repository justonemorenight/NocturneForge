use crate::artifacts::{ArtifactStore, MAX_INLINE_OUTPUT_BYTES, truncate_text};
use crate::budget::{BudgetExceeded, ExecutionBudget, TaskExecutionReporter};
use crate::cancellation::CancellationTree;
use crate::events::{RuntimeEvent, RuntimeEventStream};
use crate::executor::{TaskExecutionContext, TaskExecutor};
use crate::ids::{CorrelationId, RunId, TaskId};
use crate::plan_graph::PlanGraph;
use crate::state::{RunState, TaskState};
use crate::task_registry::TaskRegistry;
use crate::verification::{VerificationPolicy, VerificationResult};
use anyhow::Result;
use collections::HashSet;
use futures::stream::{FuturesUnordered, StreamExt as _};
use parking_lot::RwLock;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Instant;

#[derive(Clone)]
pub struct RuntimeControl {
    state: Arc<RwLock<RunState>>,
    notifications: async_channel::Sender<()>,
    notification_receiver: async_channel::Receiver<()>,
}

impl RuntimeControl {
    pub fn new(initial_state: RunState) -> Self {
        let (notifications, notification_receiver) = async_channel::bounded(1);
        Self {
            state: Arc::new(RwLock::new(initial_state)),
            notifications,
            notification_receiver,
        }
    }

    pub fn state(&self) -> RunState {
        *self.state.read()
    }

    pub fn transition(&self, next: RunState) -> Result<bool> {
        let mut state = self.state.write();
        if *state == next {
            return Ok(false);
        }
        if state.is_terminal() || !state.can_transition_to(next) {
            anyhow::bail!("invalid run transition from `{:?}` to `{:?}`", *state, next);
        }
        *state = next;
        drop(state);
        let _notification_pending = self.notifications.try_send(());
        Ok(true)
    }

    pub async fn wait_until_runnable(&self) -> bool {
        loop {
            match self.state() {
                RunState::Approved | RunState::Running | RunState::Interrupted => return true,
                state if state.is_terminal() => return false,
                _ => {
                    if self.notification_receiver.recv().await.is_err() {
                        return false;
                    }
                }
            }
        }
    }
}

#[derive(Clone)]
pub struct SchedulerConfig {
    /// Maximum number of subagents or tasks running concurrently.
    pub max_parallel_tasks: usize,
    /// Default timeout for individual task executions in seconds.
    pub task_timeout_secs: Option<u64>,
    /// Verification and retry policy settings.
    pub verification_policy: VerificationPolicy,
    /// Optional GPUI background executor for timers and task scheduling.
    pub background_executor: Option<gpui::BackgroundExecutor>,
}

fn default_max_concurrency() -> usize {
    4
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_parallel_tasks: default_max_concurrency(),
            task_timeout_secs: None,
            verification_policy: VerificationPolicy::default(),
            background_executor: None,
        }
    }
}

/// Dependency-aware, concurrency-bounded execution scheduler.
pub struct Scheduler {
    run_id: RunId,
    plan_graph: PlanGraph,
    task_registry: TaskRegistry,
    artifact_store: ArtifactStore,
    cancellation_tree: Arc<CancellationTree>,
    event_stream: RuntimeEventStream,
    executor: Rc<dyn TaskExecutor>,
    config: SchedulerConfig,
    control: RuntimeControl,
}

impl Scheduler {
    pub fn new(
        run_id: RunId,
        plan_graph: PlanGraph,
        task_registry: TaskRegistry,
        artifact_store: ArtifactStore,
        cancellation_tree: Arc<CancellationTree>,
        event_stream: RuntimeEventStream,
        executor: Rc<dyn TaskExecutor>,
        config: SchedulerConfig,
    ) -> Self {
        Self::new_with_control(
            run_id,
            plan_graph,
            task_registry,
            artifact_store,
            cancellation_tree,
            event_stream,
            executor,
            config,
            RuntimeControl::new(RunState::Running),
        )
    }

    pub fn new_with_control(
        run_id: RunId,
        plan_graph: PlanGraph,
        task_registry: TaskRegistry,
        artifact_store: ArtifactStore,
        cancellation_tree: Arc<CancellationTree>,
        event_stream: RuntimeEventStream,
        executor: Rc<dyn TaskExecutor>,
        config: SchedulerConfig,
        control: RuntimeControl,
    ) -> Self {
        for task in &plan_graph.plan().tasks {
            task_registry.register_task(task.id.clone());
        }

        Self {
            run_id,
            plan_graph,
            task_registry,
            artifact_store,
            cancellation_tree,
            event_stream,
            executor,
            config,
            control,
        }
    }

    /// Executes the entire plan until completion, unrecoverable failure, or cancellation.
    pub async fn run(&self) -> Result<RunState> {
        let start_time = Instant::now();
        let mut in_flight = FuturesUnordered::new();
        let mut in_flight_task_ids = HashSet::default();
        if !self.control.wait_until_runnable().await {
            return Ok(self.control.state());
        }
        if self.control.state() != RunState::Running {
            self.control.transition(RunState::Running)?;
        }
        self.event_stream.emit(RuntimeEvent::RunStarted {
            run_id: self.run_id.clone(),
        });
        self.event_stream.emit(RuntimeEvent::RunStateChanged {
            run_id: self.run_id.clone(),
            state: RunState::Running,
        });

        loop {
            if !self.control.wait_until_runnable().await {
                return Ok(self.control.state());
            }
            if self.cancellation_tree.root_token().is_cancelled() {
                if self.control.state() != RunState::Cancelled
                    && self.control.transition(RunState::Cancelled).is_err()
                {
                    return Ok(self.control.state());
                }
                let reason = self
                    .cancellation_tree
                    .root_token()
                    .reason()
                    .map(|r| r.description().to_string())
                    .unwrap_or_else(|| "cancelled by user".to_string());
                self.event_stream.emit(RuntimeEvent::RunCancelled {
                    run_id: self.run_id.clone(),
                    reason,
                });
                self.event_stream.emit(RuntimeEvent::RunStateChanged {
                    run_id: self.run_id.clone(),
                    state: RunState::Cancelled,
                });
                return Ok(RunState::Cancelled);
            }

            let completed = self.task_registry.completed_tasks();
            let failed = self.task_registry.failed_tasks();
            let cancelled = self.task_registry.cancelled_tasks();
            let in_progress = self.task_registry.in_progress_tasks();
            let mut scheduled_or_in_progress = in_progress.clone();
            scheduled_or_in_progress.extend(in_flight_task_ids.iter().cloned());

            let mut failed_or_cancelled = failed.clone();
            failed_or_cancelled.extend(cancelled.clone());

            // Check if all tasks in the plan are completed
            if self.plan_graph.is_complete(&completed) && in_flight.is_empty() {
                if !matches!(self.control.transition(RunState::Completed), Ok(true)) {
                    return Ok(self.control.state());
                }
                let duration_ms = start_time.elapsed().as_millis() as u64;
                let total_tokens = self.task_registry.total_tokens_used();
                self.event_stream.emit(RuntimeEvent::RunCompleted {
                    run_id: self.run_id.clone(),
                    total_tokens_used: total_tokens,
                    duration_ms,
                });
                self.event_stream.emit(RuntimeEvent::RunStateChanged {
                    run_id: self.run_id.clone(),
                    state: RunState::Completed,
                });
                return Ok(RunState::Completed);
            }

            let ready = self.plan_graph.ready_tasks(
                &completed,
                &scheduled_or_in_progress,
                &failed_or_cancelled,
            );

            if ready.is_empty() && in_flight.is_empty() {
                if !matches!(self.control.transition(RunState::Failed), Ok(true)) {
                    return Ok(self.control.state());
                }
                // If there are blocked or failed tasks preventing further progress
                let error_msg = format!(
                    "Execution halted: {} tasks completed, {} failed, {} cancelled",
                    completed.len(),
                    failed.len(),
                    cancelled.len()
                );
                self.event_stream.emit(RuntimeEvent::RunFailed {
                    run_id: self.run_id.clone(),
                    error: error_msg,
                });
                self.event_stream.emit(RuntimeEvent::RunStateChanged {
                    run_id: self.run_id.clone(),
                    state: RunState::Failed,
                });
                return Ok(RunState::Failed);
            }

            let batch_size = self
                .config
                .max_parallel_tasks
                .saturating_sub(in_flight.len());
            let to_dispatch: Vec<TaskId> = ready.into_iter().take(batch_size).collect();

            for task_id in to_dispatch {
                in_flight_task_ids.insert(task_id.clone());
                in_flight.push(async move {
                    self.execute_task_workflow(task_id.clone()).await;
                    task_id
                });
            }

            if let Some(task_id) = in_flight.next().await {
                in_flight_task_ids.remove(&task_id);
            } else {
                smol::future::yield_now().await;
            }
        }
    }

    fn mark_blocked_dependents(&self, task_id: &TaskId) {
        let blocked = self.plan_graph.blocked_by_failure(task_id);
        for blocked_task in blocked {
            self.task_registry
                .set_state(&blocked_task, TaskState::Blocked);
        }
    }

    async fn execute_task_workflow(&self, task_id: TaskId) {
        let Some(task) = self.plan_graph.task(&task_id).cloned() else {
            return;
        };

        let task_token = self.cancellation_tree.task_token(&task_id);
        let mut attempt =
            u32::try_from(self.task_registry.attempts_for(&task_id).len()).unwrap_or(u32::MAX);
        let mut session_id = self
            .task_registry
            .status(&task_id)
            .and_then(|status| status.active_session_id);

        loop {
            attempt = attempt.saturating_add(1);
            let task_start = Instant::now();
            let correlation_id = CorrelationId::new();

            self.task_registry
                .start_attempt(&task_id, session_id.clone());
            self.event_stream.emit(RuntimeEvent::TaskDispatched {
                run_id: self.run_id.clone(),
                task_id: task_id.clone(),
                session_id: session_id.clone(),
                attempt,
            });

            let budget = ExecutionBudget {
                token_budget: task.token_budget,
                tool_call_budget: task.tool_call_budget,
                time_budget_secs: task.time_budget_secs,
            };
            let reporter = TaskExecutionReporter::new(
                self.run_id.clone(),
                task_id.clone(),
                task.model_override.clone().map(Into::into),
                budget.clone(),
                Some(self.event_stream.clone()),
            );
            if let Some(model) = task.model_override.clone() {
                self.task_registry.set_model(&task_id, &model);
                reporter.report_model_assigned(model);
            }
            self.task_registry.set_phase(&task_id, "running");

            let context = TaskExecutionContext {
                task: task.clone(),
                attempt,
                cancellation_token: task_token.clone(),
                correlation_id,
                existing_session_id: session_id.clone(),
                run_id: self.run_id.clone(),
                budget,
                reporter: reporter.clone(),
            };

            let exec_result = self
                .execute_attempt(
                    context.clone(),
                    &task_token,
                    task.time_budget_secs.or(self.config.task_timeout_secs),
                )
                .await;

            if task_token.is_cancelled() {
                if matches!(
                    task_token.reason(),
                    Some(crate::cancellation::CancellationReason::Timeout)
                ) {
                    let reporter_usage = reporter.usage();
                    let error = exec_result
                        .err()
                        .map(|error| error.to_string())
                        .unwrap_or_else(|| "task timed out".to_string());
                    self.task_registry.fail_attempt_with_usage(
                        &task_id,
                        attempt,
                        error.clone(),
                        false,
                        reporter_usage,
                        None,
                    );
                    self.event_stream.emit(RuntimeEvent::TaskFailed {
                        run_id: self.run_id.clone(),
                        task_id,
                        error,
                        retryable: false,
                    });
                    return;
                }
                self.task_registry.set_state(&task_id, TaskState::Cancelled);
                self.event_stream.emit(RuntimeEvent::TaskCancelled {
                    run_id: self.run_id.clone(),
                    task_id,
                    reason: "cancelled during execution".to_string(),
                });
                return;
            }

            match exec_result {
                Ok(output) => {
                    session_id = output.session_id.clone();
                    if let Some(session_id) = session_id.clone() {
                        self.task_registry
                            .set_active_session_id(&task_id, session_id);
                    }

                    for artifact in &output.artifacts {
                        let artifact = self.artifact_store.record(artifact.clone());
                        self.event_stream.emit(RuntimeEvent::ArtifactRecorded {
                            run_id: self.run_id.clone(),
                            task_id: task_id.clone(),
                            artifact,
                        });
                    }

                    self.event_stream.emit(RuntimeEvent::TaskOutput {
                        run_id: self.run_id.clone(),
                        task_id: task_id.clone(),
                        session_id: session_id.clone(),
                        output: truncate_text(output.output.clone(), MAX_INLINE_OUTPUT_BYTES),
                    });

                    let mut tokens = output.tokens_used.unwrap_or(0);
                    // Budget enforcement: a task that exceeds its budget must
                    // fail with a typed, non-retryable reason - never complete.
                    let reporter_usage = reporter.usage();
                    tokens = tokens.max(reporter_usage.tokens_used);
                    self.task_registry.update_budget_usage(
                        &task_id,
                        tokens,
                        reporter_usage.tool_calls,
                    );
                    let cumulative_usage = self
                        .task_registry
                        .status(&task_id)
                        .map(|status| status.budget_state)
                        .unwrap_or_default();
                    self.event_stream.emit(RuntimeEvent::TaskBudgetUpdated {
                        run_id: self.run_id.clone(),
                        task_id: task_id.clone(),
                        tokens_used: cumulative_usage.tokens_used,
                        tool_calls_used: cumulative_usage.tool_calls_used,
                    });
                    if let Some(budget) = task.token_budget
                        && cumulative_usage.tokens_used > budget
                    {
                        let exceeded = BudgetExceeded::TokenBudgetExceeded {
                            budget,
                            used: cumulative_usage.tokens_used,
                        };
                        let error = format!(
                            "token budget of {budget} was exceeded (used {})",
                            cumulative_usage.tokens_used
                        );
                        self.task_registry.complete_attempt(
                            &task_id,
                            attempt,
                            Some(truncate_text(
                                output.output.clone(),
                                MAX_INLINE_OUTPUT_BYTES,
                            )),
                            Some(tokens),
                            Some(VerificationResult::fail(
                                error.clone(),
                                exceeded.error_class(),
                            )),
                        );
                        self.task_registry.stop_for_budget(&task_id, exceeded);
                        self.event_stream.emit(RuntimeEvent::TaskFailed {
                            run_id: self.run_id.clone(),
                            task_id: task_id.clone(),
                            error,
                            retryable: false,
                        });
                        self.mark_blocked_dependents(&task_id);
                        return;
                    }
                    if let Some(budget) = task.tool_call_budget
                        && cumulative_usage.tool_calls_used > budget
                    {
                        let exceeded = BudgetExceeded::ToolCallBudgetExceeded {
                            budget,
                            used: cumulative_usage.tool_calls_used,
                        };
                        let error = format!(
                            "tool call budget of {budget} was exceeded (used {})",
                            cumulative_usage.tool_calls_used
                        );
                        self.task_registry.complete_attempt(
                            &task_id,
                            attempt,
                            Some(truncate_text(
                                output.output.clone(),
                                MAX_INLINE_OUTPUT_BYTES,
                            )),
                            Some(tokens),
                            Some(VerificationResult::fail(
                                error.clone(),
                                exceeded.error_class(),
                            )),
                        );
                        self.task_registry.stop_for_budget(&task_id, exceeded);
                        self.event_stream.emit(RuntimeEvent::TaskFailed {
                            run_id: self.run_id.clone(),
                            task_id: task_id.clone(),
                            error,
                            retryable: false,
                        });
                        self.mark_blocked_dependents(&task_id);
                        return;
                    }

                    // Verification phase
                    if self.config.verification_policy.verify_outputs
                        && (!task.acceptance_criteria.is_empty()
                            || task.expected_output.is_some()
                            || task.evidence_required)
                    {
                        self.task_registry.set_state(&task_id, TaskState::Verifying);
                        self.event_stream.emit(RuntimeEvent::TaskVerifying {
                            run_id: self.run_id.clone(),
                            task_id: task_id.clone(),
                            attempt,
                        });

                        let mut verified_output = output;
                        let mut attempt_tokens = tokens;
                        let mut verification = self
                            .executor
                            .verify(&task, &verified_output)
                            .await
                            .unwrap_or_else(|error| {
                                VerificationResult::fail(
                                    format!("verification failed to run: {error}"),
                                    crate::verification::ErrorClass::FatalError,
                                )
                            });
                        self.event_stream
                            .emit(RuntimeEvent::TaskVerificationResult {
                                run_id: self.run_id.clone(),
                                task_id: task_id.clone(),
                                result: verification.clone(),
                            });

                        if !verification.passed
                            && verification.repairable
                            && task.repair_on_failure
                            && self.config.verification_policy.allow_repair_tasks
                        {
                            self.task_registry.set_state(&task_id, TaskState::Repairing);
                            let feedback = verification
                                .feedback
                                .clone()
                                .unwrap_or_else(|| "Verification failed".into());
                            self.event_stream.emit(RuntimeEvent::TaskRepairing {
                                run_id: self.run_id.clone(),
                                task_id: task_id.clone(),
                                reason: feedback.clone(),
                            });
                            if let Ok(repaired_output) =
                                self.executor.repair(&task, &feedback, context).await
                            {
                                let repaired_usage = reporter.usage();
                                let additional_tokens =
                                    repaired_output.tokens_used.unwrap_or(0).max(
                                        repaired_usage
                                            .tokens_used
                                            .saturating_sub(reporter_usage.tokens_used),
                                    );
                                let additional_tool_calls = repaired_usage
                                    .tool_calls
                                    .saturating_sub(reporter_usage.tool_calls);
                                self.task_registry.update_budget_usage(
                                    &task_id,
                                    additional_tokens,
                                    additional_tool_calls,
                                );
                                attempt_tokens = attempt_tokens.saturating_add(additional_tokens);
                                if let Some(status) = self.task_registry.status(&task_id) {
                                    self.event_stream.emit(RuntimeEvent::TaskBudgetUpdated {
                                        run_id: self.run_id.clone(),
                                        task_id: task_id.clone(),
                                        tokens_used: status.budget_state.tokens_used,
                                        tool_calls_used: status.budget_state.tool_calls_used,
                                    });
                                }
                                verification = self
                                    .executor
                                    .verify(&task, &repaired_output)
                                    .await
                                    .unwrap_or_else(|error| {
                                        VerificationResult::fail(
                                            format!("repaired output verification failed: {error}"),
                                            crate::verification::ErrorClass::FatalError,
                                        )
                                    });
                                verified_output = repaired_output;
                                self.event_stream
                                    .emit(RuntimeEvent::TaskVerificationResult {
                                        run_id: self.run_id.clone(),
                                        task_id: task_id.clone(),
                                        result: verification.clone(),
                                    });
                            }
                        }

                        if let Some(status) = self.task_registry.status(&task_id) {
                            let usage = status.budget_state;
                            let stopped_reason = if let Some(budget) = task.token_budget
                                && usage.tokens_used > budget
                            {
                                Some(BudgetExceeded::TokenBudgetExceeded {
                                    budget,
                                    used: usage.tokens_used,
                                })
                            } else if let Some(budget) = task.tool_call_budget
                                && usage.tool_calls_used > budget
                            {
                                Some(BudgetExceeded::ToolCallBudgetExceeded {
                                    budget,
                                    used: usage.tool_calls_used,
                                })
                            } else {
                                None
                            };
                            if let Some(stopped_reason) = stopped_reason {
                                let error = stopped_reason.message();
                                self.task_registry.stop_for_budget(&task_id, stopped_reason);
                                verification = VerificationResult::fail(
                                    error,
                                    crate::verification::ErrorClass::TokenBudgetExceeded,
                                );
                            }
                        }

                        if verification.passed {
                            let duration_ms = task_start.elapsed().as_millis() as u64;
                            let verified_tokens = attempt_tokens;
                            let verified_output =
                                truncate_text(verified_output.output, MAX_INLINE_OUTPUT_BYTES);
                            self.task_registry.complete_attempt(
                                &task_id,
                                attempt,
                                Some(verified_output.clone()),
                                Some(verified_tokens),
                                Some(verification),
                            );
                            self.event_stream.emit(RuntimeEvent::TaskCompleted {
                                run_id: self.run_id.clone(),
                                task_id,
                                output: Some(verified_output),
                                tokens_used: verified_tokens,
                                duration_ms,
                            });
                            return;
                        }

                        let error_class = verification.error_class.unwrap_or(
                            crate::verification::ErrorClass::VerificationAssertionFailure,
                        );
                        let should_retry = verification.retryable
                            && self.config.verification_policy.should_retry(
                                attempt,
                                &error_class,
                                task.max_retries,
                            );
                        let error = verification
                            .feedback
                            .clone()
                            .unwrap_or_else(|| "verification failed".to_string());
                        let failed_output =
                            truncate_text(verified_output.output, MAX_INLINE_OUTPUT_BYTES);
                        self.task_registry.complete_attempt(
                            &task_id,
                            attempt,
                            Some(failed_output),
                            Some(attempt_tokens),
                            Some(verification),
                        );
                        if !should_retry {
                            self.task_registry.set_state(&task_id, TaskState::Failed);
                        }
                        self.event_stream.emit(RuntimeEvent::TaskFailed {
                            run_id: self.run_id.clone(),
                            task_id: task_id.clone(),
                            error: error.clone(),
                            retryable: should_retry,
                        });
                        if should_retry {
                            let delay = self.config.verification_policy.backoff_duration(attempt);
                            self.event_stream.emit(RuntimeEvent::TaskRetrying {
                                run_id: self.run_id.clone(),
                                task_id: task_id.clone(),
                                next_attempt: attempt + 1,
                                reason: error,
                                delay_ms: delay.as_millis() as u64,
                            });
                            if let Some(background_executor) =
                                self.config.background_executor.as_ref()
                                && !delay.is_zero()
                            {
                                background_executor.timer(delay).await;
                            }
                            continue;
                        }
                        for blocked in self.plan_graph.blocked_by_failure(&task_id) {
                            self.task_registry.set_state(&blocked, TaskState::Blocked);
                        }
                        return;
                    } else {
                        let duration_ms = task_start.elapsed().as_millis() as u64;
                        let output = truncate_text(output.output, MAX_INLINE_OUTPUT_BYTES);
                        self.task_registry.complete_attempt(
                            &task_id,
                            attempt,
                            Some(output.clone()),
                            Some(tokens),
                            Some(VerificationResult::pass()),
                        );
                        self.event_stream.emit(RuntimeEvent::TaskCompleted {
                            run_id: self.run_id.clone(),
                            task_id,
                            output: Some(output),
                            tokens_used: tokens,
                            duration_ms,
                        });
                        return;
                    }
                }
                Err(error) => {
                    let error_str = error.to_string();
                    let error_class = VerificationPolicy::classify_error(&error_str);
                    let reporter_usage = reporter.usage();
                    let previous_usage = self
                        .task_registry
                        .status(&task_id)
                        .map(|status| status.budget_state)
                        .unwrap_or_default();
                    // Budget-exceeded errors (mid-flight reporter failures) must
                    // record a typed stop reason and never be retried.
                    let stopped_reason = if matches!(
                        error_class,
                        crate::verification::ErrorClass::TokenBudgetExceeded
                    ) {
                        if let Some(budget) = task.token_budget
                            && previous_usage
                                .tokens_used
                                .saturating_add(reporter_usage.tokens_used)
                                > budget
                        {
                            Some(BudgetExceeded::TokenBudgetExceeded {
                                budget,
                                used: previous_usage
                                    .tokens_used
                                    .saturating_add(reporter_usage.tokens_used),
                            })
                        } else if let Some(budget) = task.tool_call_budget
                            && previous_usage
                                .tool_calls_used
                                .saturating_add(reporter_usage.tool_calls)
                                > budget
                        {
                            Some(BudgetExceeded::ToolCallBudgetExceeded {
                                budget,
                                used: previous_usage
                                    .tool_calls_used
                                    .saturating_add(reporter_usage.tool_calls),
                            })
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                    let should_retry = self.config.verification_policy.should_retry(
                        attempt,
                        &error_class,
                        task.max_retries,
                    );

                    self.task_registry.fail_attempt_with_usage(
                        &task_id,
                        attempt,
                        error_str.clone(),
                        should_retry,
                        reporter_usage,
                        stopped_reason,
                    );
                    self.event_stream.emit(RuntimeEvent::TaskFailed {
                        run_id: self.run_id.clone(),
                        task_id: task_id.clone(),
                        error: error_str.clone(),
                        retryable: should_retry,
                    });

                    if should_retry {
                        let delay = self.config.verification_policy.backoff_duration(attempt);
                        self.event_stream.emit(RuntimeEvent::TaskRetrying {
                            run_id: self.run_id.clone(),
                            task_id: task_id.clone(),
                            next_attempt: attempt + 1,
                            reason: error_str,
                            delay_ms: delay.as_millis() as u64,
                        });
                        if delay.as_millis() > 0 {
                            if let Some(exec) = &self.config.background_executor {
                                exec.timer(delay).await;
                            }
                        }
                        continue;
                    } else {
                        // Mark transitively blocked dependents
                        let blocked_tasks = self.plan_graph.blocked_by_failure(&task_id);
                        for blocked in blocked_tasks {
                            self.task_registry.set_state(&blocked, TaskState::Blocked);
                        }
                        return;
                    }
                }
            }
        }
    }

    async fn execute_attempt(
        &self,
        context: TaskExecutionContext,
        task_token: &crate::cancellation::CancellationToken,
        timeout_secs: Option<u64>,
    ) -> Result<crate::executor::TaskExecutionOutput> {
        let execution = self.executor.execute(context);
        let cancellation = task_token.cancelled();
        futures::pin_mut!(execution, cancellation);
        let controlled_execution = async {
            match futures::future::select(execution, cancellation).await {
                futures::future::Either::Left((result, _)) => result,
                futures::future::Either::Right(((), _)) => {
                    Err(anyhow::anyhow!("task execution cancelled"))
                }
            }
        };

        let Some(timeout_secs) = timeout_secs else {
            return controlled_execution.await;
        };
        let background_executor = self
            .config
            .background_executor
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("task timeout requires a background executor"))?;
        let timeout = background_executor.timer(std::time::Duration::from_secs(timeout_secs));
        futures::pin_mut!(controlled_execution, timeout);
        match futures::future::select(controlled_execution, timeout).await {
            futures::future::Either::Left((result, _)) => result,
            futures::future::Either::Right(((), _)) => {
                task_token.cancel(crate::cancellation::CancellationReason::Timeout);
                Err(anyhow::anyhow!(
                    "task timed out after {timeout_secs} seconds"
                ))
            }
        }
    }
}
