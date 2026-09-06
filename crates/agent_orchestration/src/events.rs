use crate::artifacts::Artifact;
use crate::context_checkpoint::ContextCheckpoint;
use crate::control_plane::{AgentIdentity, AgentMessage, AgentPath};
use crate::ids::{PlanId, RunId, TaskId};
use crate::plan_graph::OrchestrationPlan;
use crate::state::RunState;
use crate::verification::VerificationResult;
use crate::worker::{StructuredWaitReason, WorkerMetadata};
use agent_client_protocol::schema::v1 as acp;
use agent_settings::AgentExecutionPolicy;
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// A fully serializable event capturing a state transition or update in the
/// orchestration engine.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RuntimeEvent {
    RunCreated {
        run_id: RunId,
        plan_id: PlanId,
        policy: AgentExecutionPolicy,
        created_at: DateTime<Utc>,
    },
    PlanProposed {
        run_id: RunId,
        plan: OrchestrationPlan,
    },
    PlanApproved {
        run_id: RunId,
    },
    RunStarted {
        run_id: RunId,
    },
    RunPaused {
        run_id: RunId,
    },
    RunResumed {
        run_id: RunId,
    },
    AgentRegistered {
        run_id: RunId,
        identity: AgentIdentity,
    },
    AgentMessageQueued {
        run_id: RunId,
        message: AgentMessage,
    },
    AgentMailboxDrained {
        run_id: RunId,
        recipient: AgentPath,
        through_sequence: u64,
        count: usize,
    },
    AgentUnregistered {
        run_id: RunId,
        path: AgentPath,
    },
    ContextCheckpointRecorded {
        run_id: RunId,
        checkpoint: ContextCheckpoint,
    },
    TaskScheduled {
        run_id: RunId,
        task_id: TaskId,
        wave_index: usize,
    },
    TaskDispatched {
        run_id: RunId,
        task_id: TaskId,
        session_id: Option<acp::SessionId>,
        attempt: u32,
    },
    TaskPhaseChanged {
        run_id: RunId,
        task_id: TaskId,
        phase: String,
    },
    TaskProgress {
        run_id: RunId,
        task_id: TaskId,
        message: String,
        tokens_used: Option<u64>,
        #[serde(default)]
        percent: f32,
    },
    TaskToolCallStarted {
        run_id: RunId,
        task_id: TaskId,
        tool: String,
    },
    TaskToolCallFinished {
        run_id: RunId,
        task_id: TaskId,
    },
    TaskContextUpdated {
        run_id: RunId,
        task_id: TaskId,
        scope: Option<String>,
        context_paths: Vec<String>,
    },
    TaskModelAssigned {
        run_id: RunId,
        task_id: TaskId,
        model_id: String,
    },
    TaskBudgetUpdated {
        run_id: RunId,
        task_id: TaskId,
        tokens_used: u64,
        tool_calls_used: u64,
    },
    TaskOutput {
        run_id: RunId,
        task_id: TaskId,
        session_id: Option<acp::SessionId>,
        output: String,
    },
    TaskVerifying {
        run_id: RunId,
        task_id: TaskId,
        attempt: u32,
    },
    TaskVerificationResult {
        run_id: RunId,
        task_id: TaskId,
        result: VerificationResult,
    },
    TaskRepairing {
        run_id: RunId,
        task_id: TaskId,
        reason: String,
    },
    TaskRetrying {
        run_id: RunId,
        task_id: TaskId,
        next_attempt: u32,
        reason: String,
        delay_ms: u64,
    },
    TaskCompleted {
        run_id: RunId,
        task_id: TaskId,
        output: Option<String>,
        tokens_used: u64,
        duration_ms: u64,
    },
    TaskFailed {
        run_id: RunId,
        task_id: TaskId,
        error: String,
        retryable: bool,
    },
    TaskCancelled {
        run_id: RunId,
        task_id: TaskId,
        reason: String,
    },
    TaskAwaitingApply {
        run_id: RunId,
        task_id: TaskId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        worktree_path: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        patch_id: Option<String>,
    },
    TaskWaitReasonChanged {
        run_id: RunId,
        task_id: TaskId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        wait_reason: Option<StructuredWaitReason>,
    },
    WorkerMetadataUpdated {
        run_id: RunId,
        task_id: TaskId,
        metadata: WorkerMetadata,
    },
    ArtifactRecorded {
        run_id: RunId,
        task_id: TaskId,
        artifact: Artifact,
    },
    RunCompleted {
        run_id: RunId,
        total_tokens_used: u64,
        duration_ms: u64,
    },
    RunFailed {
        run_id: RunId,
        error: String,
    },
    RunCancelled {
        run_id: RunId,
        reason: String,
    },
    RunStateChanged {
        run_id: RunId,
        state: RunState,
    },
}

impl RuntimeEvent {
    pub fn run_id(&self) -> &RunId {
        match self {
            Self::RunCreated { run_id, .. }
            | Self::PlanProposed { run_id, .. }
            | Self::PlanApproved { run_id, .. }
            | Self::RunStarted { run_id, .. }
            | Self::RunPaused { run_id, .. }
            | Self::RunResumed { run_id, .. }
            | Self::AgentRegistered { run_id, .. }
            | Self::AgentMessageQueued { run_id, .. }
            | Self::AgentMailboxDrained { run_id, .. }
            | Self::AgentUnregistered { run_id, .. }
            | Self::ContextCheckpointRecorded { run_id, .. }
            | Self::TaskScheduled { run_id, .. }
            | Self::TaskDispatched { run_id, .. }
            | Self::TaskPhaseChanged { run_id, .. }
            | Self::TaskProgress { run_id, .. }
            | Self::TaskToolCallStarted { run_id, .. }
            | Self::TaskToolCallFinished { run_id, .. }
            | Self::TaskContextUpdated { run_id, .. }
            | Self::TaskModelAssigned { run_id, .. }
            | Self::TaskBudgetUpdated { run_id, .. }
            | Self::TaskOutput { run_id, .. }
            | Self::TaskVerifying { run_id, .. }
            | Self::TaskVerificationResult { run_id, .. }
            | Self::TaskRepairing { run_id, .. }
            | Self::TaskRetrying { run_id, .. }
            | Self::TaskCompleted { run_id, .. }
            | Self::TaskFailed { run_id, .. }
            | Self::TaskCancelled { run_id, .. }
            | Self::TaskAwaitingApply { run_id, .. }
            | Self::TaskWaitReasonChanged { run_id, .. }
            | Self::WorkerMetadataUpdated { run_id, .. }
            | Self::ArtifactRecorded { run_id, .. }
            | Self::RunCompleted { run_id, .. }
            | Self::RunFailed { run_id, .. }
            | Self::RunCancelled { run_id, .. }
            | Self::RunStateChanged { run_id, .. } => run_id,
        }
    }
}

/// A runtime event tagged with its monotonic sequence number. Subscribers use
/// the sequence to detect gaps (slow subscribers) and replay from a known
/// position.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SequencedRuntimeEvent {
    pub seq: u64,
    pub event: RuntimeEvent,
}

/// A subscription to the event stream together with a per-subscriber flag
/// that is set when the subscriber's channel overflowed and events were
/// dropped. When `resync_required` becomes true, the subscriber should replay
/// from its last known sequence (`RuntimeEventStream::history_from`) or read
/// the latest task snapshots from the registry.
#[derive(Clone)]
pub struct EventSubscription {
    pub receiver: async_channel::Receiver<SequencedRuntimeEvent>,
    pub resync_required: Arc<AtomicBool>,
}

/// Pub/sub event stream manager for broadcasting and recording runtime events.
#[derive(Clone)]
pub struct RuntimeEventStream {
    event_log: Arc<RwLock<EventLog>>,
    subscribers: Arc<RwLock<Vec<Subscriber>>>,
    next_seq: Arc<AtomicU64>,
}

struct Subscriber {
    sender: async_channel::Sender<SequencedRuntimeEvent>,
    resync_required: Arc<AtomicBool>,
}

const MAX_EVENT_HISTORY: usize = 10_000;
const MAX_EVENT_HISTORY_BYTES: usize = 8 * 1024 * 1024;
const SUBSCRIBER_CAPACITY: usize = 256;

#[derive(Default)]
struct EventLog {
    events: VecDeque<SequencedRuntimeEvent>,
    serialized_bytes: usize,
}

impl EventLog {
    fn push(&mut self, event: SequencedRuntimeEvent) {
        self.serialized_bytes = self
            .serialized_bytes
            .saturating_add(serialized_event_size(&event));
        self.events.push_back(event);
        while self.events.len() > MAX_EVENT_HISTORY
            || self.serialized_bytes > MAX_EVENT_HISTORY_BYTES
        {
            let Some(event) = self.events.pop_front() else {
                self.serialized_bytes = 0;
                break;
            };
            self.serialized_bytes = self
                .serialized_bytes
                .saturating_sub(serialized_event_size(&event));
        }
    }
}

fn serialized_event_size(event: &SequencedRuntimeEvent) -> usize {
    serde_json::to_vec(event).map_or(0, |event| event.len())
}

impl Default for RuntimeEventStream {
    fn default() -> Self {
        Self::new()
    }
}

impl RuntimeEventStream {
    pub fn new() -> Self {
        Self {
            event_log: Arc::new(RwLock::new(EventLog::default())),
            subscribers: Arc::new(RwLock::new(Vec::new())),
            next_seq: Arc::new(AtomicU64::new(1)),
        }
    }

    pub fn from_history(history: Vec<SequencedRuntimeEvent>) -> Self {
        let mut event_log = EventLog::default();
        let mut max_seq = 0;
        for event in history {
            max_seq = max_seq.max(event.seq);
            event_log.push(event);
        }
        Self {
            event_log: Arc::new(RwLock::new(event_log)),
            subscribers: Arc::new(RwLock::new(Vec::new())),
            next_seq: Arc::new(AtomicU64::new(max_seq + 1)),
        }
    }

    /// Emits an event with the next sequence number to all active subscribers
    /// and stores it in the bounded event log.
    pub fn emit(&self, event: RuntimeEvent) {
        let seq = self.next_seq.fetch_add(1, Ordering::Relaxed);
        let sequenced = SequencedRuntimeEvent { seq, event };

        {
            let mut log = self.event_log.write();
            log.push(sequenced.clone());
        }

        let mut subscribers = self.subscribers.write();
        subscribers.retain(
            |subscriber| match subscriber.sender.try_send(sequenced.clone()) {
                Ok(()) => true,
                // The subscriber is slow: drop the event for it, keep the channel
                // open, and flag that it must re-sync before trusting live events.
                Err(async_channel::TrySendError::Full(_)) => {
                    subscriber.resync_required.store(true, Ordering::SeqCst);
                    true
                }
                Err(async_channel::TrySendError::Closed(_)) => false,
            },
        );
    }

    /// Creates a subscription channel that receives all future events.
    pub fn subscribe(&self) -> EventSubscription {
        let (sender, receiver) = async_channel::bounded(SUBSCRIBER_CAPACITY);
        let resync_required = Arc::new(AtomicBool::new(false));
        let mut subscribers = self.subscribers.write();
        subscribers.push(Subscriber {
            sender,
            resync_required: resync_required.clone(),
        });
        EventSubscription {
            receiver,
            resync_required,
        }
    }

    /// Creates a subscription that replays all past events and continues with
    /// live events.
    pub fn subscribe_with_replay(&self) -> EventSubscription {
        let log = self.event_log.read();
        let capacity = log.events.len().saturating_add(SUBSCRIBER_CAPACITY).max(1);
        let (sender, receiver) = async_channel::bounded(capacity);
        for event in &log.events {
            if sender.try_send(event.clone()).is_err() {
                log::error!("failed to enqueue orchestration event for replay");
                break;
            }
        }
        let resync_required = Arc::new(AtomicBool::new(false));
        let mut subscribers = self.subscribers.write();
        subscribers.push(Subscriber {
            sender,
            resync_required: resync_required.clone(),
        });
        drop(subscribers);
        drop(log);
        EventSubscription {
            receiver,
            resync_required,
        }
    }

    /// Creates a subscription that replays only events with `seq > from_seq`
    /// and continues with live events. Used by subscribers recovering from a
    /// detected gap.
    pub fn subscribe_from(&self, from_seq: u64) -> EventSubscription {
        let log = self.event_log.read();
        let replay = log
            .events
            .iter()
            .filter(|event| event.seq > from_seq)
            .cloned()
            .collect::<Vec<_>>();
        let capacity = replay.len().saturating_add(SUBSCRIBER_CAPACITY).max(1);
        let (sender, receiver) = async_channel::bounded(capacity);
        for event in &replay {
            if sender.try_send(event.clone()).is_err() {
                log::error!("failed to enqueue orchestration event for replay");
                break;
            }
        }
        let resync_required = Arc::new(AtomicBool::new(false));
        let mut subscribers = self.subscribers.write();
        subscribers.push(Subscriber {
            sender,
            resync_required: resync_required.clone(),
        });
        drop(subscribers);
        drop(log);
        EventSubscription {
            receiver,
            resync_required,
        }
    }

    /// Returns events with `seq > from_seq` from the bounded history.
    pub fn history_from(&self, from_seq: u64) -> Vec<SequencedRuntimeEvent> {
        let log = self.event_log.read();
        log.events
            .iter()
            .filter(|event| event.seq > from_seq)
            .cloned()
            .collect()
    }

    /// Returns the full recorded event history with sequence numbers.
    pub fn history(&self) -> Vec<SequencedRuntimeEvent> {
        let log = self.event_log.read();
        log.events.iter().cloned().collect()
    }

    /// Returns the sequence number of the next event that will be emitted.
    pub fn latest_seq(&self) -> u64 {
        self.next_seq.load(Ordering::Relaxed)
    }
}
