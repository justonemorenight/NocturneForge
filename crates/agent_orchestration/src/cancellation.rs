use crate::ids::TaskId;
use collections::HashMap;
use parking_lot::RwLock;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Reason for triggering cancellation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CancellationReason {
    UserRequested,
    ParentCancelled,
    DependencyFailed(TaskId),
    Timeout,
    Shutdown,
    Interrupted,
    Custom(String),
}

impl CancellationReason {
    pub fn description(&self) -> &str {
        match self {
            Self::UserRequested => "cancelled by user",
            Self::ParentCancelled => "parent run was cancelled",
            Self::DependencyFailed(_) => "required dependency failed",
            Self::Timeout => "operation timed out",
            Self::Shutdown => "system is shutting down",
            Self::Interrupted => "operation was interrupted",
            Self::Custom(msg) => msg,
        }
    }
}

/// A lightweight cancellation token that can be checked and shared across async boundaries.
#[derive(Clone, Debug)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
    reason: Arc<RwLock<Option<CancellationReason>>>,
    notifications: async_channel::Sender<()>,
    notification_receiver: async_channel::Receiver<()>,
}

impl Default for CancellationToken {
    fn default() -> Self {
        Self::new()
    }
}

impl CancellationToken {
    pub fn new() -> Self {
        let (notifications, notification_receiver) = async_channel::bounded(1);
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
            reason: Arc::new(RwLock::new(None)),
            notifications,
            notification_receiver,
        }
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    pub fn cancel(&self, reason: CancellationReason) {
        let mut guard = self.reason.write();
        if guard.is_none() {
            *guard = Some(reason);
            self.cancelled.store(true, Ordering::SeqCst);
            self.notifications.close();
        }
    }

    pub fn reason(&self) -> Option<CancellationReason> {
        self.reason.read().clone()
    }

    pub async fn cancelled(&self) {
        if self.is_cancelled() {
            return;
        }
        let _channel_closed = self.notification_receiver.recv().await;
    }
}

/// Hierarchical cancellation tree coordinating parent run and child task lifecycles.
pub struct CancellationTree {
    root_token: CancellationToken,
    task_tokens: RwLock<HashMap<TaskId, CancellationToken>>,
}

impl Default for CancellationTree {
    fn default() -> Self {
        Self::new()
    }
}

impl CancellationTree {
    pub fn new() -> Self {
        Self {
            root_token: CancellationToken::new(),
            task_tokens: RwLock::new(HashMap::default()),
        }
    }

    /// Root token for the entire run.
    pub fn root_token(&self) -> CancellationToken {
        self.root_token.clone()
    }

    /// Retrieves or creates a cancellation token for a specific task.
    pub fn task_token(&self, task_id: &TaskId) -> CancellationToken {
        let mut tokens = self.task_tokens.write();
        tokens
            .entry(task_id.clone())
            .or_insert_with(CancellationToken::new)
            .clone()
    }

    /// Cancels the entire run, propagating cancellation to all child tasks.
    pub fn cancel_run(&self, reason: CancellationReason) {
        self.root_token.cancel(reason);
        let tokens = self.task_tokens.read();
        for token in tokens.values() {
            token.cancel(CancellationReason::ParentCancelled);
        }
    }

    /// Cancels a specific task without cancelling the parent run or sibling tasks.
    pub fn cancel_task(&self, task_id: &TaskId, reason: CancellationReason) {
        self.task_token(task_id).cancel(reason);
    }

    /// Checks if the run or a specific task is cancelled.
    pub fn is_cancelled(&self, task_id: Option<&TaskId>) -> bool {
        if self.root_token.is_cancelled() {
            return true;
        }
        if let Some(task_id) = task_id {
            let tokens = self.task_tokens.read();
            tokens
                .get(task_id)
                .map(|token| token.is_cancelled())
                .unwrap_or(false)
        } else {
            false
        }
    }
}
