use crate::events::{RuntimeEvent, RuntimeEventStream};
use crate::execution_limiter::{AgentExecutionLimiter, AgentExecutionLimiterConfig};
use crate::ids::{RunId, TaskId};
use crate::plan_graph::OrchestrationPlan;
use crate::worker::WorkerTarget;
use anyhow::{Result, bail};
use chrono::{DateTime, Utc};
use collections::HashMap;
use parking_lot::{Mutex, RwLock};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AgentControlPlaneConfig {
    pub max_registered_agents: usize,
    pub max_messages_per_agent: usize,
    pub max_message_bytes: usize,
    pub max_mailbox_bytes_per_agent: usize,
    pub execution_limiter: AgentExecutionLimiterConfig,
}

impl Default for AgentControlPlaneConfig {
    fn default() -> Self {
        Self {
            max_registered_agents: 128,
            max_messages_per_agent: 128,
            max_message_bytes: 64 * 1024,
            max_mailbox_bytes_per_agent: 512 * 1024,
            execution_limiter: AgentExecutionLimiterConfig::default(),
        }
    }
}

#[derive(
    Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
pub struct AgentPath(String);

impl AgentPath {
    pub const ROOT: &'static str = "/root";

    pub fn root() -> Self {
        Self(Self::ROOT.to_string())
    }

    pub fn parse(path: impl Into<String>) -> Result<Self> {
        let path = path.into();
        if path == Self::ROOT {
            return Ok(Self(path));
        }
        let Some(suffix) = path.strip_prefix("/root/") else {
            bail!("agent path must start with '/root'");
        };
        if suffix.is_empty() || suffix.split('/').any(|segment| !is_valid_segment(segment)) {
            bail!("agent path contains an invalid segment");
        }
        Ok(Self(path))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn name(&self) -> &str {
        self.0.rsplit('/').next().unwrap_or("root")
    }

    pub fn parent(&self) -> Option<Self> {
        if self.0 == Self::ROOT {
            return None;
        }
        self.0
            .rsplit_once('/')
            .map(|(parent, _)| Self(parent.to_string()))
    }

    fn child(&self, segment: &str) -> Result<Self> {
        if !is_valid_segment(segment) {
            bail!("agent name '{segment}' is not a valid path segment");
        }
        Self::parse(format!("{}/{segment}", self.0))
    }
}

impl std::fmt::Display for AgentPath {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentIdentity {
    pub path: AgentPath,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<AgentPath>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_id: Option<TaskId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<WorkerTarget>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum AgentMessageKind {
    Message,
    FollowUp,
    Completion,
}

impl AgentMessageKind {
    pub fn triggers_turn(self) -> bool {
        matches!(self, Self::FollowUp)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentMessage {
    pub sequence: u64,
    pub author: AgentPath,
    pub recipient: AgentPath,
    pub kind: AgentMessageKind,
    pub body: String,
    pub created_at: DateTime<Utc>,
}

#[derive(Clone)]
pub struct AgentControlPlane {
    inner: Arc<AgentControlPlaneInner>,
}

struct AgentControlPlaneInner {
    config: AgentControlPlaneConfig,
    state: RwLock<AgentControlPlaneState>,
    next_message_sequence: AtomicU64,
    runtime_events: RwLock<Option<ControlPlaneRuntimeEvents>>,
    execution_limiter: AgentExecutionLimiter,
}

#[derive(Clone)]
struct ControlPlaneRuntimeEvents {
    run_id: RunId,
    event_stream: RuntimeEventStream,
}

struct AgentControlPlaneState {
    identities: HashMap<AgentPath, AgentIdentity>,
    task_paths: HashMap<TaskId, AgentPath>,
    mailboxes: HashMap<AgentPath, AgentMailbox>,
}

#[derive(Clone)]
struct AgentMailbox {
    queue: Arc<Mutex<MailboxQueue>>,
    activity_sender: async_channel::Sender<()>,
    activity_receiver: async_channel::Receiver<()>,
}

#[derive(Default)]
struct MailboxQueue {
    messages: VecDeque<AgentMessage>,
    bytes: usize,
}

impl AgentMailbox {
    fn new() -> Self {
        // Activity is coalesced: a single wake is enough to make the owner drain
        // every message currently queued.
        let (activity_sender, activity_receiver) = async_channel::bounded(1);
        Self {
            queue: Arc::new(Mutex::new(MailboxQueue::default())),
            activity_sender,
            activity_receiver,
        }
    }

    fn drain(&self) -> Vec<AgentMessage> {
        while self.activity_receiver.try_recv().is_ok() {}
        let mut queue = self.queue.lock();
        queue.bytes = 0;
        queue.messages.drain(..).collect()
    }
}

impl AgentControlPlane {
    pub fn new(config: AgentControlPlaneConfig) -> Result<Self> {
        let execution_limiter = AgentExecutionLimiter::new(config.execution_limiter.clone())?;
        let root_path = AgentPath::root();
        let root = AgentIdentity {
            path: root_path.clone(),
            parent: None,
            task_id: None,
            role: Some("root".to_string()),
            target: None,
        };
        Ok(Self {
            inner: Arc::new(AgentControlPlaneInner {
                config,
                state: RwLock::new(AgentControlPlaneState {
                    identities: [(root_path.clone(), root)].into_iter().collect(),
                    task_paths: HashMap::default(),
                    mailboxes: [(root_path, AgentMailbox::new())].into_iter().collect(),
                }),
                next_message_sequence: AtomicU64::new(1),
                runtime_events: RwLock::new(None),
                execution_limiter,
            }),
        })
    }

    pub fn from_plan(plan: &OrchestrationPlan, config: AgentControlPlaneConfig) -> Result<Self> {
        let control_plane = Self::new(config)?;
        let root = AgentPath::root();
        for task in &plan.tasks {
            control_plane.register_child(
                &root,
                task.id.clone(),
                task.id.as_str(),
                task.role.clone().or_else(|| task.native_role.clone()),
                task.target.clone(),
            )?;
        }
        Ok(control_plane)
    }

    pub fn with_runtime_events(self, run_id: RunId, event_stream: RuntimeEventStream) -> Self {
        *self.inner.runtime_events.write() = Some(ControlPlaneRuntimeEvents {
            run_id: run_id.clone(),
            event_stream: event_stream.clone(),
        });
        for identity in self.list(None) {
            event_stream.emit(RuntimeEvent::AgentRegistered {
                run_id: run_id.clone(),
                identity,
            });
        }
        self
    }

    pub fn register_child(
        &self,
        parent: &AgentPath,
        task_id: TaskId,
        preferred_name: &str,
        role: Option<String>,
        target: WorkerTarget,
    ) -> Result<AgentPath> {
        let mut state = self.inner.state.write();
        if let Some(existing) = state.task_paths.get(&task_id) {
            let identity = state.identities.get(existing).ok_or_else(|| {
                anyhow::anyhow!("agent registry is inconsistent for task '{task_id}'")
            })?;
            if identity.parent.as_ref() != Some(parent)
                || identity.role != role
                || identity.target.as_ref() != Some(&target)
            {
                bail!("task '{task_id}' is already registered with a different agent identity");
            }
            return Ok(existing.clone());
        }
        if !state.identities.contains_key(parent) {
            bail!("parent agent '{parent}' is not registered");
        }
        if state.identities.len() >= self.inner.config.max_registered_agents {
            bail!(
                "agent registry reached its limit of {} entries",
                self.inner.config.max_registered_agents
            );
        }

        let base = canonical_segment(preferred_name);
        let mut suffix = 1_u32;
        let path = loop {
            let segment = if suffix == 1 {
                base.clone()
            } else {
                format!("{base}-{suffix}")
            };
            let candidate = parent.child(&segment)?;
            if !state.identities.contains_key(&candidate) {
                break candidate;
            }
            suffix = suffix.saturating_add(1);
        };
        let identity = AgentIdentity {
            path: path.clone(),
            parent: Some(parent.clone()),
            task_id: Some(task_id.clone()),
            role,
            target: Some(target),
        };
        state.task_paths.insert(task_id, path.clone());
        state.mailboxes.insert(path.clone(), AgentMailbox::new());
        state.identities.insert(path.clone(), identity.clone());
        drop(state);
        self.emit(|run_id| RuntimeEvent::AgentRegistered { run_id, identity });
        Ok(path)
    }

    pub fn identity(&self, path: &AgentPath) -> Option<AgentIdentity> {
        self.inner.state.read().identities.get(path).cloned()
    }

    pub fn execution_limiter(&self) -> &AgentExecutionLimiter {
        &self.inner.execution_limiter
    }

    pub fn identity_for_task(&self, task_id: &TaskId) -> Option<AgentIdentity> {
        let state = self.inner.state.read();
        state
            .task_paths
            .get(task_id)
            .and_then(|path| state.identities.get(path))
            .cloned()
    }

    pub fn list(&self, prefix: Option<&AgentPath>) -> Vec<AgentIdentity> {
        let state = self.inner.state.read();
        let mut identities = state
            .identities
            .values()
            .filter(|identity| {
                prefix.is_none_or(|prefix| {
                    identity.path == *prefix
                        || identity
                            .path
                            .as_str()
                            .strip_prefix(prefix.as_str())
                            .is_some_and(|suffix| suffix.starts_with('/'))
                })
            })
            .cloned()
            .collect::<Vec<_>>();
        identities.sort_by(|left, right| left.path.cmp(&right.path));
        identities
    }

    pub fn resolve(&self, caller: &AgentPath, reference: &str) -> Result<AgentPath> {
        let state = self.inner.state.read();
        if !state.identities.contains_key(caller) {
            bail!("calling agent '{caller}' is not registered");
        }
        let reference = reference.trim();
        if reference == "root" || reference == AgentPath::ROOT {
            return Ok(AgentPath::root());
        }
        if reference.starts_with('/') {
            let path = AgentPath::parse(reference)?;
            if state.identities.contains_key(&path) {
                return Ok(path);
            }
            bail!("agent '{path}' is not registered");
        }
        if reference.contains('/') || !is_valid_segment(reference) {
            bail!("agent reference '{reference}' is invalid");
        }

        let direct_child = caller.child(reference)?;
        if state.identities.contains_key(&direct_child) {
            return Ok(direct_child);
        }
        if let Some(parent) = caller.parent() {
            let sibling = parent.child(reference)?;
            if state.identities.contains_key(&sibling) {
                return Ok(sibling);
            }
        }
        let matches = state
            .identities
            .keys()
            .filter(|path| path.name() == reference)
            .cloned()
            .collect::<Vec<_>>();
        match matches.as_slice() {
            [path] => Ok(path.clone()),
            [] => bail!("agent reference '{reference}' was not found"),
            _ => bail!("agent reference '{reference}' is ambiguous; use its canonical path"),
        }
    }

    pub fn send(
        &self,
        author: &AgentPath,
        recipient: &AgentPath,
        kind: AgentMessageKind,
        body: impl Into<String>,
    ) -> Result<AgentMessage> {
        let body = body.into();
        if body.trim().is_empty() {
            bail!("agent messages cannot be empty");
        }
        if body.len() > self.inner.config.max_message_bytes {
            bail!(
                "agent message exceeds the {} byte limit",
                self.inner.config.max_message_bytes
            );
        }
        if author == recipient {
            bail!("an agent cannot send a mailbox message to itself");
        }
        let mailbox = {
            let state = self.inner.state.read();
            if !state.identities.contains_key(author) {
                bail!("author agent '{author}' is not registered in this run");
            }
            state.mailboxes.get(recipient).cloned().ok_or_else(|| {
                anyhow::anyhow!("recipient agent '{recipient}' is not registered in this run")
            })?
        };

        let message = AgentMessage {
            sequence: self
                .inner
                .next_message_sequence
                .fetch_add(1, Ordering::Relaxed),
            author: author.clone(),
            recipient: recipient.clone(),
            kind,
            body,
            created_at: Utc::now(),
        };
        {
            let mut queue = mailbox.queue.lock();
            if queue.messages.len() >= self.inner.config.max_messages_per_agent {
                bail!(
                    "mailbox for '{recipient}' reached its {} message limit",
                    self.inner.config.max_messages_per_agent
                );
            }
            if queue.bytes.saturating_add(message.body.len())
                > self.inner.config.max_mailbox_bytes_per_agent
            {
                bail!(
                    "mailbox for '{recipient}' reached its {} byte limit",
                    self.inner.config.max_mailbox_bytes_per_agent
                );
            }
            queue.bytes = queue.bytes.saturating_add(message.body.len());
            queue.messages.push_back(message.clone());
        }
        match mailbox.activity_sender.try_send(()) {
            Ok(()) | Err(async_channel::TrySendError::Full(())) => {}
            Err(async_channel::TrySendError::Closed(())) => {
                bail!("mailbox for '{recipient}' is closed")
            }
        }
        self.emit(|run_id| RuntimeEvent::AgentMessageQueued {
            run_id,
            message: message.clone(),
        });
        Ok(message)
    }

    pub fn drain(&self, recipient: &AgentPath) -> Result<Vec<AgentMessage>> {
        let mailbox = self.mailbox(recipient)?;
        let messages = mailbox.drain();
        self.emit_mailbox_drained(recipient, &messages);
        Ok(messages)
    }

    pub async fn wait_for_messages(&self, recipient: &AgentPath) -> Result<Vec<AgentMessage>> {
        let mailbox = self.mailbox(recipient)?;
        let queued = mailbox.drain();
        if !queued.is_empty() {
            self.emit_mailbox_drained(recipient, &queued);
            return Ok(queued);
        }
        mailbox
            .activity_receiver
            .recv()
            .await
            .map_err(|_| anyhow::anyhow!("mailbox for '{recipient}' is closed"))?;
        let mut queue = mailbox.queue.lock();
        queue.bytes = 0;
        let messages = queue.messages.drain(..).collect::<Vec<_>>();
        drop(queue);
        self.emit_mailbox_drained(recipient, &messages);
        Ok(messages)
    }

    pub fn unregister_subtree(&self, path: &AgentPath) -> Result<usize> {
        if path.as_str() == AgentPath::ROOT {
            bail!("the root agent cannot be unregistered");
        }
        let mut state = self.inner.state.write();
        if !state.identities.contains_key(path) {
            return Ok(0);
        }
        let prefix = format!("{}/", path.as_str());
        let removed = state
            .identities
            .keys()
            .filter(|candidate| *candidate == path || candidate.as_str().starts_with(&prefix))
            .cloned()
            .collect::<Vec<_>>();
        for removed_path in &removed {
            if let Some(identity) = state.identities.remove(removed_path)
                && let Some(task_id) = identity.task_id
            {
                state.task_paths.remove(&task_id);
            }
            state.mailboxes.remove(removed_path);
        }
        drop(state);
        for removed_path in &removed {
            self.emit(|run_id| RuntimeEvent::AgentUnregistered {
                run_id,
                path: removed_path.clone(),
            });
        }
        Ok(removed.len())
    }

    fn mailbox(&self, recipient: &AgentPath) -> Result<AgentMailbox> {
        self.inner
            .state
            .read()
            .mailboxes
            .get(recipient)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("agent '{recipient}' is not registered in this run"))
    }

    fn emit_mailbox_drained(&self, recipient: &AgentPath, messages: &[AgentMessage]) {
        let Some(last_message) = messages.last() else {
            return;
        };
        self.emit(|run_id| RuntimeEvent::AgentMailboxDrained {
            run_id,
            recipient: recipient.clone(),
            through_sequence: last_message.sequence,
            count: messages.len(),
        });
    }

    fn emit(&self, create_event: impl FnOnce(RunId) -> RuntimeEvent) {
        if let Some(runtime_events) = self.inner.runtime_events.read().as_ref() {
            runtime_events
                .event_stream
                .emit(create_event(runtime_events.run_id.clone()));
        }
    }
}

fn canonical_segment(value: &str) -> String {
    let mut result = String::new();
    let mut last_was_separator = false;
    for character in value.chars() {
        let normalized = character.to_ascii_lowercase();
        if normalized.is_ascii_alphanumeric() || matches!(normalized, '_' | '-') {
            result.push(normalized);
            last_was_separator = false;
        } else if !last_was_separator && !result.is_empty() {
            result.push('-');
            last_was_separator = true;
        }
        if result.len() >= 64 {
            break;
        }
    }
    while result.ends_with('-') {
        result.pop();
    }
    if result.is_empty() {
        "agent".to_string()
    } else {
        result
    }
}

fn is_valid_segment(segment: &str) -> bool {
    !segment.is_empty()
        && segment.len() <= 80
        && segment != "."
        && segment != ".."
        && segment
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
}

#[cfg(test)]
#[path = "control_plane_tests.rs"]
mod tests;
