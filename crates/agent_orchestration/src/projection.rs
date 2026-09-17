use crate::control_plane::{AgentControlPlaneSnapshot, AgentPath};
use crate::events::{RuntimeEvent, SequencedRuntimeEvent};
use crate::ids::{RunId, TaskId};
use crate::plan_graph::{OrchestrationPlan, OrchestrationTask};
use crate::state::{RunState, TaskState, TaskStatus};
use collections::HashMap;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActivityAgentProjection {
    pub task_id: TaskId,
    pub path: AgentPath,
    pub queued_messages: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RunActivityProjection {
    pub run_id: RunId,
    pub state: RunState,
    pub plan: OrchestrationPlan,
    pub task_statuses: Vec<TaskStatus>,
    pub agents: Vec<ActivityAgentProjection>,
    pub last_event_seq: u64,
    pub resync_required: bool,
}

impl RunActivityProjection {
    pub fn from_snapshot(
        run_id: RunId,
        state: RunState,
        plan: OrchestrationPlan,
        task_statuses: Vec<TaskStatus>,
        control_plane: Option<&AgentControlPlaneSnapshot>,
        last_event_seq: u64,
    ) -> Self {
        let mut statuses = task_statuses
            .into_iter()
            .map(|status| (status.task_id.clone(), status))
            .collect::<HashMap<_, _>>();
        let task_statuses = plan
            .tasks
            .iter()
            .map(|task| {
                statuses.remove(&task.id).unwrap_or_else(|| {
                    let mut status = TaskStatus::new(task.id.clone());
                    status.target = task.target.clone();
                    status
                })
            })
            .collect();
        let agents = control_plane.map_or_else(Vec::new, |control_plane| {
            control_plane
                .identities
                .iter()
                .filter_map(|identity| {
                    let task_id = identity.task_id.clone()?;
                    let queued_messages = control_plane
                        .mailboxes
                        .iter()
                        .find(|mailbox| mailbox.recipient == identity.path)
                        .map_or(0, |mailbox| mailbox.messages.len());
                    Some(ActivityAgentProjection {
                        task_id,
                        path: identity.path.clone(),
                        queued_messages,
                    })
                })
                .collect()
        });
        Self {
            run_id,
            state,
            plan,
            task_statuses,
            agents,
            last_event_seq,
            resync_required: false,
        }
    }

    pub fn apply(&mut self, envelope: &SequencedRuntimeEvent) {
        if envelope.event.run_id() != &self.run_id || envelope.seq <= self.last_event_seq {
            return;
        }
        if self.last_event_seq != 0 && envelope.seq != self.last_event_seq.saturating_add(1) {
            self.resync_required = true;
            return;
        }
        self.last_event_seq = envelope.seq;
        let updated_at = envelope.occurred_at;

        match &envelope.event {
            RuntimeEvent::PlanProposed { plan, .. } => {
                self.plan = plan.clone();
                self.ensure_plan_tasks();
                self.state = RunState::Proposed;
            }
            RuntimeEvent::PlanApproved { .. } => self.state = RunState::Approved,
            RuntimeEvent::RunStarted { .. } | RuntimeEvent::RunResumed { .. } => {
                self.state = RunState::Running;
            }
            RuntimeEvent::RunPaused { .. } => self.state = RunState::Paused,
            RuntimeEvent::RunCompleted { .. } => self.state = RunState::Completed,
            RuntimeEvent::RunFailed { .. } => self.state = RunState::Failed,
            RuntimeEvent::RunCancelled { .. } => self.state = RunState::Cancelled,
            RuntimeEvent::RunStateChanged { state, .. } => self.state = *state,
            RuntimeEvent::AgentRegistered { identity, .. } => {
                if let Some(task_id) = &identity.task_id {
                    let queued_messages = self
                        .agents
                        .iter()
                        .find(|agent| agent.task_id == *task_id)
                        .map_or(0, |agent| agent.queued_messages);
                    self.agents.retain(|agent| agent.task_id != *task_id);
                    self.agents.push(ActivityAgentProjection {
                        task_id: task_id.clone(),
                        path: identity.path.clone(),
                        queued_messages,
                    });
                }
            }
            RuntimeEvent::AgentMessageQueued { message, .. } => {
                self.update_mailbox(&message.recipient, |count| count.saturating_add(1));
            }
            RuntimeEvent::AgentMessageDelivered { message, .. }
            | RuntimeEvent::AgentMessageDeliveryFailed { message, .. } => {
                self.update_mailbox(&message.recipient, |count| count.saturating_sub(1));
            }
            RuntimeEvent::AgentMailboxDrained {
                recipient, count, ..
            } => self.update_mailbox(recipient, |depth| depth.saturating_sub(*count)),
            RuntimeEvent::AgentUnregistered { path, .. } => {
                self.agents.retain(|agent| agent.path != *path);
            }
            RuntimeEvent::TaskScheduled { task_id, .. } => {
                self.update_status(task_id, updated_at, |status| {
                    status.state = TaskState::Pending;
                });
            }
            RuntimeEvent::TaskStateChanged {
                task_id,
                state,
                attempt,
                ..
            } => self.update_status(task_id, updated_at, |status| {
                status.state = *state;
                status.current_attempt = *attempt;
                status.total_attempts = status.total_attempts.max(*attempt);
                status.phase = phase_for_state(*state).map(str::to_string);
                if *state != TaskState::Running {
                    status.current_tool = None;
                }
            }),
            RuntimeEvent::TaskDispatched {
                task_id,
                session_id,
                attempt,
                ..
            } => self.update_status(task_id, updated_at, |status| {
                status.state = TaskState::Running;
                status.phase = Some("running".to_string());
                status.active_session_id = session_id.clone();
                status.current_attempt = *attempt;
                status.total_attempts = status.total_attempts.max(*attempt);
                status.wait_reason = None;
                status.latest_error = None;
            }),
            RuntimeEvent::TaskPhaseChanged { task_id, phase, .. } => {
                self.update_status(task_id, updated_at, |status| {
                    status.phase = Some(phase.clone());
                    status.state = state_for_phase(phase).unwrap_or(status.state);
                });
            }
            RuntimeEvent::TaskProgress {
                task_id,
                tokens_used,
                ..
            } => self.update_status(task_id, updated_at, |status| {
                if let Some(tokens_used) = tokens_used {
                    status.tokens_used = status.tokens_used.max(*tokens_used);
                    status.budget_state.tokens_used =
                        status.budget_state.tokens_used.max(*tokens_used);
                }
            }),
            RuntimeEvent::TaskToolCallStarted { task_id, tool, .. } => {
                self.update_status(task_id, updated_at, |status| {
                    status.current_tool = Some(tool.clone());
                });
            }
            RuntimeEvent::TaskToolCallFinished { task_id, .. } => {
                self.update_status(task_id, updated_at, |status| status.current_tool = None);
            }
            RuntimeEvent::TaskModelAssigned {
                task_id, model_id, ..
            } => self.update_status(task_id, updated_at, |status| {
                status.model = Some(model_id.clone());
            }),
            RuntimeEvent::TaskBudgetUpdated {
                task_id,
                tokens_used,
                tool_calls_used,
                ..
            } => self.update_status(task_id, updated_at, |status| {
                status.tokens_used = status.tokens_used.max(*tokens_used);
                status.budget_state.tokens_used = *tokens_used;
                status.budget_state.tool_calls_used = *tool_calls_used;
            }),
            RuntimeEvent::TaskOutput {
                task_id,
                session_id,
                output,
                ..
            } => self.update_status(task_id, updated_at, |status| {
                status.active_session_id = session_id.clone();
                status.latest_output = Some(output.clone());
            }),
            RuntimeEvent::TaskVerifying {
                task_id, attempt, ..
            } => self.update_status(task_id, updated_at, |status| {
                status.state = TaskState::Verifying;
                status.phase = Some("verifying".to_string());
                status.current_attempt = *attempt;
                status.current_tool = None;
            }),
            RuntimeEvent::TaskVerificationResult {
                task_id, result, ..
            } => self.update_status(task_id, updated_at, |status| {
                status.latest_verification = Some(result.clone());
            }),
            RuntimeEvent::TaskRepairing {
                task_id, reason, ..
            } => self.update_status(task_id, updated_at, |status| {
                status.state = TaskState::Repairing;
                status.phase = Some("repairing".to_string());
                status.latest_error = Some(reason.clone());
                status.current_tool = None;
            }),
            RuntimeEvent::TaskRetrying {
                task_id, reason, ..
            } => self.update_status(task_id, updated_at, |status| {
                status.state = TaskState::Retrying;
                status.phase = Some("retrying".to_string());
                status.latest_error = Some(reason.clone());
                status.current_tool = None;
            }),
            RuntimeEvent::TaskCompleted {
                task_id,
                output,
                tokens_used,
                ..
            } => self.update_status(task_id, updated_at, |status| {
                status.state = TaskState::Completed;
                status.phase = None;
                status.current_tool = None;
                status.latest_output = output.clone().or_else(|| status.latest_output.clone());
                status.tokens_used = status.tokens_used.max(*tokens_used);
                status.wait_reason = None;
            }),
            RuntimeEvent::TaskFailed {
                task_id,
                error,
                retryable,
                ..
            } => {
                self.update_status(task_id, updated_at, |status| {
                    status.state = if *retryable {
                        TaskState::Retrying
                    } else {
                        TaskState::Failed
                    };
                    status.phase = retryable.then(|| "retrying".to_string());
                    status.current_tool = None;
                    status.latest_error = Some(error.clone());
                });
                if !retryable {
                    self.block_dependents(task_id, updated_at);
                }
            }
            RuntimeEvent::TaskAttemptFailed {
                task_id,
                attempt,
                error,
                ..
            } => self.update_status(task_id, updated_at, |status| {
                status.state = TaskState::Retrying;
                status.phase = Some("retrying".to_string());
                status.current_attempt = *attempt;
                status.total_attempts = status.total_attempts.max(*attempt);
                status.current_tool = None;
                status.latest_error = Some(error.clone());
            }),
            RuntimeEvent::TaskCancelled {
                task_id, reason, ..
            } => {
                self.update_status(task_id, updated_at, |status| {
                    status.state = TaskState::Cancelled;
                    status.phase = None;
                    status.current_tool = None;
                    status.latest_error = Some(reason.clone());
                });
                self.block_dependents(task_id, updated_at);
            }
            RuntimeEvent::TaskAwaitingApply { task_id, .. } => {
                self.update_status(task_id, updated_at, |status| {
                    status.state = TaskState::AwaitingApply;
                    status.phase = Some("awaiting_apply".to_string());
                    status.current_tool = None;
                });
            }
            RuntimeEvent::TaskWaitReasonChanged {
                task_id,
                wait_reason,
                ..
            } => self.update_status(task_id, updated_at, |status| {
                status.wait_reason = wait_reason.clone();
                if let Some(wait_reason) = wait_reason {
                    status.state = state_for_wait_reason(wait_reason);
                    status.phase = None;
                    status.current_tool = None;
                } else if status.state == TaskState::Parked {
                    status.state = TaskState::Pending;
                }
            }),
            RuntimeEvent::WorkerMetadataUpdated {
                task_id, metadata, ..
            } => self.update_status(task_id, updated_at, |status| {
                status.worker_metadata = Some(metadata.clone());
            }),
            RuntimeEvent::RunCreated { .. }
            | RuntimeEvent::TaskCancellationRequested { .. }
            | RuntimeEvent::ContextCheckpointRecorded { .. }
            | RuntimeEvent::AgentResidencyChanged { .. }
            | RuntimeEvent::GoalUpdated { .. }
            | RuntimeEvent::TaskContextUpdated { .. }
            | RuntimeEvent::ArtifactRecorded { .. } => {}
        }
    }

    pub fn task(&self, task_id: &TaskId) -> Option<&OrchestrationTask> {
        self.plan.tasks.iter().find(|task| task.id == *task_id)
    }

    pub fn agent_for_task(&self, task_id: &TaskId) -> Option<&ActivityAgentProjection> {
        self.agents.iter().find(|agent| agent.task_id == *task_id)
    }

    fn ensure_plan_tasks(&mut self) {
        for task in &self.plan.tasks {
            if self
                .task_statuses
                .iter()
                .all(|status| status.task_id != task.id)
            {
                let mut status = TaskStatus::new(task.id.clone());
                status.target = task.target.clone();
                self.task_statuses.push(status);
            }
        }
    }

    fn update_status(
        &mut self,
        task_id: &TaskId,
        updated_at: chrono::DateTime<chrono::Utc>,
        update: impl FnOnce(&mut TaskStatus),
    ) {
        let Some(status) = self
            .task_statuses
            .iter_mut()
            .find(|status| status.task_id == *task_id)
        else {
            self.resync_required = true;
            return;
        };
        update(status);
        status.updated_at = updated_at;
    }

    fn update_mailbox(&mut self, path: &AgentPath, update: impl FnOnce(usize) -> usize) {
        let Some(agent) = self.agents.iter_mut().find(|agent| agent.path == *path) else {
            self.resync_required = true;
            return;
        };
        agent.queued_messages = update(agent.queued_messages);
    }

    fn block_dependents(
        &mut self,
        failed_task_id: &TaskId,
        updated_at: chrono::DateTime<chrono::Utc>,
    ) {
        let mut blocked = vec![failed_task_id.clone()];
        let mut index = 0;
        while let Some(blocked_task_id) = blocked.get(index).cloned() {
            index += 1;
            for task in &self.plan.tasks {
                if task.depends_on.contains(&blocked_task_id) && !blocked.contains(&task.id) {
                    blocked.push(task.id.clone());
                }
            }
        }
        for task_id in blocked.into_iter().skip(1) {
            self.update_status(&task_id, updated_at, |status| {
                if !status.state.is_terminal() {
                    status.state = TaskState::Blocked;
                    status.phase = None;
                    status.current_tool = None;
                }
            });
        }
    }
}

fn state_for_phase(phase: &str) -> Option<TaskState> {
    match phase {
        "running" => Some(TaskState::Running),
        "verifying" => Some(TaskState::Verifying),
        "repairing" => Some(TaskState::Repairing),
        "retrying" => Some(TaskState::Retrying),
        "awaiting_apply" => Some(TaskState::AwaitingApply),
        _ => None,
    }
}

fn phase_for_state(state: TaskState) -> Option<&'static str> {
    match state {
        TaskState::Running => Some("running"),
        TaskState::Verifying => Some("verifying"),
        TaskState::Repairing => Some("repairing"),
        TaskState::Retrying => Some("retrying"),
        TaskState::AwaitingApply => Some("awaiting_apply"),
        _ => None,
    }
}

fn state_for_wait_reason(reason: &crate::worker::StructuredWaitReason) -> TaskState {
    use crate::worker::StructuredWaitReason;

    match reason {
        StructuredWaitReason::AwaitingDependency { .. } => TaskState::WaitingDependency,
        StructuredWaitReason::AwaitingRetryBackoff { .. } => TaskState::Retrying,
        StructuredWaitReason::AwaitingApply { .. } => TaskState::AwaitingApply,
        StructuredWaitReason::AwaitingApproval
        | StructuredWaitReason::AwaitingUserInput { .. }
        | StructuredWaitReason::AwaitingWorkerReconnect { .. }
        | StructuredWaitReason::ApplyConflict { .. }
        | StructuredWaitReason::WorktreeVerificationFailed { .. }
        | StructuredWaitReason::PostApplyVerificationFailed { .. }
        | StructuredWaitReason::AwaitingProcessExit { .. }
        | StructuredWaitReason::Custom { .. } => TaskState::Parked,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control_plane::AgentIdentity;
    use crate::ids::{EventId, PlanId};
    use crate::worker::WorkerTarget;
    use chrono::Utc;

    fn envelope(run_id: &RunId, seq: u64, event: RuntimeEvent) -> SequencedRuntimeEvent {
        SequencedRuntimeEvent {
            seq,
            event_id: EventId::from_sequence(run_id, seq),
            occurred_at: Utc::now(),
            correlation_id: None,
            caused_by: None,
            event,
        }
    }

    fn plan() -> OrchestrationPlan {
        OrchestrationPlan {
            id: PlanId::from_string("plan-test"),
            title: "Test plan".to_string(),
            explanation: None,
            tasks: vec![OrchestrationTask::new("task-1", "Task one", "Do work")],
        }
    }

    fn dependent_plan() -> OrchestrationPlan {
        OrchestrationPlan {
            id: PlanId::from_string("plan-dependent"),
            title: "Dependent plan".to_string(),
            explanation: None,
            tasks: vec![
                OrchestrationTask::new("task-1", "Task one", "Do work"),
                OrchestrationTask::new("task-2", "Task two", "Use result")
                    .with_depends_on(vec![TaskId::new("task-1")]),
            ],
        }
    }

    #[test]
    fn reduces_task_and_mailbox_lifecycle() -> anyhow::Result<()> {
        let run_id = RunId::from_string("run-test");
        let task_id = TaskId::new("task-1");
        let mut projection = RunActivityProjection::from_snapshot(
            run_id.clone(),
            RunState::Approved,
            plan(),
            Vec::new(),
            None,
            0,
        );
        let path = AgentPath::parse("/root/task-1")?;
        projection.apply(&envelope(
            &run_id,
            1,
            RuntimeEvent::AgentRegistered {
                run_id: run_id.clone(),
                identity: AgentIdentity {
                    path: path.clone(),
                    parent: Some(AgentPath::root()),
                    task_id: Some(task_id.clone()),
                    role: None,
                    target: Some(WorkerTarget::Native),
                },
            },
        ));
        projection.apply(&envelope(
            &run_id,
            2,
            RuntimeEvent::TaskDispatched {
                run_id: run_id.clone(),
                task_id: task_id.clone(),
                session_id: None,
                attempt: 1,
            },
        ));
        projection.apply(&envelope(
            &run_id,
            3,
            RuntimeEvent::TaskCompleted {
                run_id: run_id.clone(),
                task_id: task_id.clone(),
                output: Some("done".to_string()),
                tokens_used: 42,
                duration_ms: 10,
            },
        ));

        let status = &projection.task_statuses[0];
        assert_eq!(status.state, TaskState::Completed);
        assert_eq!(status.latest_output.as_deref(), Some("done"));
        assert_eq!(status.tokens_used, 42);
        assert_eq!(
            projection.agent_for_task(&task_id).map(|agent| &agent.path),
            Some(&path)
        );
        assert!(!projection.resync_required);
        Ok(())
    }

    #[test]
    fn rejects_event_gaps_without_mutating_projected_state() {
        let run_id = RunId::from_string("run-test");
        let mut projection = RunActivityProjection::from_snapshot(
            run_id.clone(),
            RunState::Approved,
            plan(),
            Vec::new(),
            None,
            3,
        );
        projection.apply(&envelope(
            &run_id,
            5,
            RuntimeEvent::RunCompleted {
                run_id: run_id.clone(),
                total_tokens_used: 0,
                duration_ms: 0,
            },
        ));

        assert_eq!(projection.state, RunState::Approved);
        assert_eq!(projection.last_event_seq, 3);
        assert!(projection.resync_required);
    }

    #[test]
    fn retryable_failures_do_not_block_dependents() {
        let run_id = RunId::from_string("run-test");
        let task_id = TaskId::new("task-1");
        let mut projection = RunActivityProjection::from_snapshot(
            run_id.clone(),
            RunState::Running,
            dependent_plan(),
            Vec::new(),
            None,
            0,
        );
        projection.apply(&envelope(
            &run_id,
            1,
            RuntimeEvent::TaskFailed {
                run_id: run_id.clone(),
                task_id: task_id.clone(),
                error: "temporary failure".to_string(),
                retryable: true,
            },
        ));
        assert_eq!(projection.task_statuses[0].state, TaskState::Retrying);
        assert_eq!(projection.task_statuses[1].state, TaskState::Pending);
        projection.apply(&envelope(
            &run_id,
            2,
            RuntimeEvent::TaskRetrying {
                run_id: run_id.clone(),
                task_id,
                next_attempt: 2,
                reason: "retrying".to_string(),
                delay_ms: 0,
            },
        ));

        assert_eq!(projection.task_statuses[0].state, TaskState::Retrying);
        assert_eq!(projection.task_statuses[1].state, TaskState::Pending);
        assert_eq!(projection.task_statuses[0].current_attempt, 0);
        assert_eq!(projection.task_statuses[0].total_attempts, 0);
    }

    #[test]
    fn clearing_a_parked_wait_reason_returns_the_task_to_pending() {
        let run_id = RunId::from_string("run-test");
        let task_id = TaskId::new("task-1");
        let mut status = TaskStatus::new(task_id.clone());
        status.state = TaskState::Parked;
        status.wait_reason = Some(crate::worker::StructuredWaitReason::AwaitingUserInput {
            question: "Restart?".to_string(),
        });
        let mut projection = RunActivityProjection::from_snapshot(
            run_id.clone(),
            RunState::Paused,
            plan(),
            vec![status],
            None,
            0,
        );
        projection.apply(&envelope(
            &run_id,
            1,
            RuntimeEvent::TaskWaitReasonChanged {
                run_id: run_id.clone(),
                task_id,
                wait_reason: None,
            },
        ));

        assert_eq!(projection.task_statuses[0].state, TaskState::Pending);
        assert!(projection.task_statuses[0].wait_reason.is_none());
    }

    #[test]
    fn canonical_state_transition_sets_exact_state_and_attempt() {
        let run_id = RunId::from_string("run-test");
        let task_id = TaskId::new("task-1");
        let mut projection = RunActivityProjection::from_snapshot(
            run_id.clone(),
            RunState::Running,
            plan(),
            Vec::new(),
            None,
            0,
        );

        projection.apply(&envelope(
            &run_id,
            1,
            RuntimeEvent::TaskStateChanged {
                run_id: run_id.clone(),
                task_id,
                previous_state: TaskState::Pending,
                state: TaskState::Verifying,
                reason: "verification started".to_string(),
                attempt: 3,
            },
        ));

        let status = &projection.task_statuses[0];
        assert_eq!(status.state, TaskState::Verifying);
        assert_eq!(status.phase.as_deref(), Some("verifying"));
        assert_eq!(status.current_attempt, 3);
        assert_eq!(status.total_attempts, 3);
    }

    #[test]
    fn cancellation_request_does_not_claim_the_task_is_cancelled() {
        let run_id = RunId::from_string("run-test");
        let task_id = TaskId::new("task-1");
        let mut status = TaskStatus::new(task_id.clone());
        status.state = TaskState::Running;
        let mut projection = RunActivityProjection::from_snapshot(
            run_id.clone(),
            RunState::Running,
            plan(),
            vec![status],
            None,
            0,
        );

        projection.apply(&envelope(
            &run_id,
            1,
            RuntimeEvent::TaskCancellationRequested {
                run_id: run_id.clone(),
                task_id,
                reason: "user requested cancellation".to_string(),
            },
        ));

        assert_eq!(projection.task_statuses[0].state, TaskState::Running);
    }
}
