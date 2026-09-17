use crate::budget::{BudgetExceeded, BudgetUsage};
use crate::events::{RuntimeEvent, RuntimeEventContext, RuntimeEventStream};
use crate::ids::{RunId, TaskId};
use crate::state::{TaskState, TaskStatus};
use crate::task_registry::TaskRegistry;
use crate::verification::VerificationResult;
use agent_client_protocol::schema::v1 as acp;
use parking_lot::Mutex;
use std::sync::Arc;

#[derive(Clone)]
pub struct TaskMutationGateway {
    run_id: RunId,
    registry: TaskRegistry,
    event_stream: RuntimeEventStream,
    lifecycle_lock: Arc<Mutex<()>>,
}

impl TaskMutationGateway {
    pub fn new(run_id: RunId, registry: TaskRegistry, event_stream: RuntimeEventStream) -> Self {
        Self {
            run_id,
            registry,
            event_stream,
            lifecycle_lock: Arc::new(Mutex::new(())),
        }
    }

    pub fn transition(&self, task_id: &TaskId, state: TaskState, reason: impl Into<String>) {
        self.commit(task_id, reason, None, || {
            self.registry.set_state(task_id, state)
        });
    }

    pub fn transition_in_context(
        &self,
        task_id: &TaskId,
        state: TaskState,
        reason: impl Into<String>,
        context: &RuntimeEventContext,
    ) {
        self.commit(task_id, reason, Some(context), || {
            self.registry.set_state(task_id, state)
        });
    }

    pub fn start_attempt(&self, task_id: &TaskId, session_id: Option<acp::SessionId>) -> u32 {
        let mut attempt = 0;
        self.commit(task_id, "attempt dispatched", None, || {
            attempt = self.registry.start_attempt(task_id, session_id);
        });
        attempt
    }

    pub fn start_attempt_in_context(
        &self,
        task_id: &TaskId,
        session_id: Option<acp::SessionId>,
        context: &RuntimeEventContext,
    ) -> u32 {
        let mut attempt = 0;
        self.commit(task_id, "attempt dispatched", Some(context), || {
            attempt = self.registry.start_attempt(task_id, session_id);
        });
        attempt
    }

    pub fn fail_attempt_with_usage(
        &self,
        task_id: &TaskId,
        attempt: u32,
        error: String,
        retryable: bool,
        usage: BudgetUsage,
        stopped_reason: Option<BudgetExceeded>,
    ) {
        let reason = if retryable {
            "attempt failed and will retry"
        } else {
            "attempt failed"
        };
        self.commit(task_id, reason, None, || {
            self.registry.fail_attempt_with_usage(
                task_id,
                attempt,
                error,
                retryable,
                usage,
                stopped_reason,
            );
        });
    }

    pub fn fail_attempt_with_usage_in_context(
        &self,
        task_id: &TaskId,
        attempt: u32,
        error: String,
        retryable: bool,
        usage: BudgetUsage,
        stopped_reason: Option<BudgetExceeded>,
        context: &RuntimeEventContext,
    ) {
        let reason = if retryable {
            "attempt failed and will retry"
        } else {
            "attempt failed"
        };
        self.commit(task_id, reason, Some(context), || {
            self.registry.fail_attempt_with_usage(
                task_id,
                attempt,
                error,
                retryable,
                usage,
                stopped_reason,
            );
        });
    }

    pub fn complete_attempt(
        &self,
        task_id: &TaskId,
        attempt: u32,
        output: Option<String>,
        tokens_used: Option<u64>,
        verification: Option<VerificationResult>,
    ) {
        self.commit(task_id, "attempt completed", None, || {
            self.registry
                .complete_attempt(task_id, attempt, output, tokens_used, verification);
        });
    }

    pub fn complete_attempt_in_context(
        &self,
        task_id: &TaskId,
        attempt: u32,
        output: Option<String>,
        tokens_used: Option<u64>,
        verification: Option<VerificationResult>,
        context: &RuntimeEventContext,
    ) {
        self.commit(task_id, "attempt completed", Some(context), || {
            self.registry
                .complete_attempt(task_id, attempt, output, tokens_used, verification);
        });
    }

    pub fn complete_attempt_with_awaiting_apply(
        &self,
        task_id: &TaskId,
        attempt: u32,
        output: Option<String>,
        tokens_used: Option<u64>,
        verification: Option<VerificationResult>,
        awaiting_apply: bool,
    ) {
        self.commit(task_id, "attempt completed", None, || {
            self.registry.complete_attempt_with_awaiting_apply(
                task_id,
                attempt,
                output,
                tokens_used,
                verification,
                awaiting_apply,
            );
        });
    }

    pub fn complete_attempt_with_awaiting_apply_in_context(
        &self,
        task_id: &TaskId,
        attempt: u32,
        output: Option<String>,
        tokens_used: Option<u64>,
        verification: Option<VerificationResult>,
        awaiting_apply: bool,
        context: &RuntimeEventContext,
    ) {
        self.commit(task_id, "attempt completed", Some(context), || {
            self.registry.complete_attempt_with_awaiting_apply(
                task_id,
                attempt,
                output,
                tokens_used,
                verification,
                awaiting_apply,
            );
        });
    }

    pub fn restart_parked_task(&self, task_id: &TaskId) -> bool {
        let mut restarted = false;
        self.commit(task_id, "task restarted", None, || {
            restarted = self.registry.restart_parked_task(task_id);
        });
        restarted
    }

    pub fn mark_applied(&self, task_id: &TaskId) -> bool {
        let mut applied = false;
        self.commit(task_id, "changes applied", None, || {
            applied = self.registry.mark_applied(task_id);
        });
        applied
    }

    pub fn mark_rejected(&self, task_id: &TaskId, reason: Option<String>) -> bool {
        let mut rejected = false;
        self.commit(task_id, "changes rejected", None, || {
            rejected = self.registry.mark_rejected(task_id, reason);
        });
        rejected
    }

    pub fn park_apply_conflict(
        &self,
        task_id: &TaskId,
        error: String,
        worktree_path: String,
    ) -> bool {
        let mut parked = false;
        self.commit(task_id, "apply conflict", None, || {
            parked = self
                .registry
                .park_apply_conflict(task_id, error, worktree_path);
        });
        parked
    }

    pub fn park_verification_failed(
        &self,
        task_id: &TaskId,
        error: String,
        worktree_path: String,
        rollback_error: Option<String>,
    ) -> bool {
        let mut parked = false;
        self.commit(task_id, "post-apply verification failed", None, || {
            parked = self.registry.park_verification_failed(
                task_id,
                error,
                worktree_path,
                rollback_error,
            );
        });
        parked
    }

    pub fn park_worktree_verification_failed(
        &self,
        task_id: &TaskId,
        error: String,
        worktree_path: String,
    ) -> bool {
        let mut parked = false;
        self.commit(task_id, "worktree verification failed", None, || {
            parked = self
                .registry
                .park_worktree_verification_failed(task_id, error, worktree_path);
        });
        parked
    }

    fn commit(
        &self,
        task_id: &TaskId,
        reason: impl Into<String>,
        context: Option<&RuntimeEventContext>,
        mutation: impl FnOnce(),
    ) {
        let _guard = self.lifecycle_lock.lock();
        let before = self.registry.status(task_id);
        mutation();
        let after = self.registry.status(task_id);
        self.emit_transition(
            task_id,
            before.as_ref(),
            after.as_ref(),
            reason.into(),
            context,
        );
    }

    fn emit_transition(
        &self,
        task_id: &TaskId,
        before: Option<&TaskStatus>,
        after: Option<&TaskStatus>,
        reason: String,
        context: Option<&RuntimeEventContext>,
    ) {
        let (Some(before), Some(after)) = (before, after) else {
            return;
        };
        if before.state == after.state {
            return;
        }
        let event = RuntimeEvent::TaskStateChanged {
            run_id: self.run_id.clone(),
            task_id: task_id.clone(),
            previous_state: before.state,
            state: after.state,
            reason,
            attempt: after.current_attempt,
        };
        if let Some(context) = context {
            self.event_stream.emit_in_context(event, context);
        } else {
            self.event_stream.emit(event);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transition_commits_registry_before_emitting_canonical_event() {
        let run_id = RunId::new();
        let task_id = TaskId::new("task");
        let registry = TaskRegistry::new();
        registry.register_task(task_id.clone());
        let event_stream = RuntimeEventStream::new();
        let subscription = event_stream.subscribe();
        let gateway = TaskMutationGateway::new(run_id, registry.clone(), event_stream);

        gateway.transition(&task_id, TaskState::Running, "dispatched");

        assert_eq!(
            registry.status(&task_id).map(|status| status.state),
            Some(TaskState::Running)
        );
        let envelope = subscription.receiver.try_recv().expect("state event");
        assert!(matches!(
            envelope.event,
            RuntimeEvent::TaskStateChanged {
                previous_state: TaskState::Pending,
                state: TaskState::Running,
                reason,
                ..
            } if reason == "dispatched"
        ));
    }

    #[test]
    fn no_op_transition_does_not_emit_an_event() {
        let run_id = RunId::new();
        let task_id = TaskId::new("task");
        let registry = TaskRegistry::new();
        registry.register_task(task_id.clone());
        let event_stream = RuntimeEventStream::new();
        let subscription = event_stream.subscribe();
        let gateway = TaskMutationGateway::new(run_id, registry, event_stream);

        gateway.transition(&task_id, TaskState::Pending, "already pending");

        assert!(subscription.receiver.try_recv().is_err());
    }
}
