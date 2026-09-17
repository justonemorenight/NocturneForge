use crate::artifacts::Artifact;
use crate::context_checkpoint::ContextCheckpoint;
use crate::control_plane::{AgentIdentity, AgentMessage, AgentPath};
use crate::goal_controller::GoalSnapshot;
use crate::ids::{CorrelationId, EventId, PlanId, RunId, TaskId};
use crate::plan_graph::OrchestrationPlan;
use crate::projection::RunActivityProjection;
use crate::residency::AgentResidencyRecord;
use crate::state::RunState;
use crate::verification::VerificationResult;
use crate::worker::{StructuredWaitReason, WorkerMetadata};
use agent_client_protocol::schema::v1 as acp;
use agent_settings::AgentExecutionPolicy;
use chrono::{DateTime, Utc};
use parking_lot::{Mutex, RwLock};
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
    AgentMessageDelivered {
        run_id: RunId,
        message: AgentMessage,
    },
    AgentMessageDeliveryFailed {
        run_id: RunId,
        message: AgentMessage,
        error: String,
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
    AgentResidencyChanged {
        run_id: RunId,
        record: AgentResidencyRecord,
    },
    GoalUpdated {
        run_id: RunId,
        goal: GoalSnapshot,
    },
    TaskScheduled {
        run_id: RunId,
        task_id: TaskId,
        wave_index: usize,
    },
    TaskStateChanged {
        run_id: RunId,
        task_id: TaskId,
        previous_state: crate::state::TaskState,
        state: crate::state::TaskState,
        reason: String,
        attempt: u32,
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
    TaskAttemptFailed {
        run_id: RunId,
        task_id: TaskId,
        attempt: u32,
        error: String,
    },
    TaskCancelled {
        run_id: RunId,
        task_id: TaskId,
        reason: String,
    },
    TaskCancellationRequested {
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
            | Self::AgentMessageDelivered { run_id, .. }
            | Self::AgentMessageDeliveryFailed { run_id, .. }
            | Self::AgentMailboxDrained { run_id, .. }
            | Self::AgentUnregistered { run_id, .. }
            | Self::ContextCheckpointRecorded { run_id, .. }
            | Self::AgentResidencyChanged { run_id, .. }
            | Self::GoalUpdated { run_id, .. }
            | Self::TaskScheduled { run_id, .. }
            | Self::TaskStateChanged { run_id, .. }
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
            | Self::TaskAttemptFailed { run_id, .. }
            | Self::TaskCancelled { run_id, .. }
            | Self::TaskCancellationRequested { run_id, .. }
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
    #[serde(default)]
    pub event_id: EventId,
    #[serde(default = "default_event_timestamp")]
    pub occurred_at: DateTime<Utc>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub correlation_id: Option<CorrelationId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub caused_by: Option<EventId>,
    pub event: RuntimeEvent,
}

fn default_event_timestamp() -> DateTime<Utc> {
    std::time::SystemTime::UNIX_EPOCH.into()
}

/// A bounded page of runtime events for restart and UI replay.
///
/// Replay consumers should persist the last returned sequence and request the
/// next page instead of loading the complete in-memory history. `gap` is set
/// when the requested cursor predates the bounded history; consumers must then
/// rebuild from a run snapshot before applying the returned events.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EventReplayPage {
    pub events: Vec<SequencedRuntimeEvent>,
    pub next_seq: Option<u64>,
    pub oldest_seq: Option<u64>,
    pub latest_seq: Option<u64>,
    pub gap: bool,
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

#[derive(Clone)]
pub struct RuntimeEventContext {
    correlation_id: CorrelationId,
    latest_event_id: Arc<Mutex<Option<EventId>>>,
}

impl RuntimeEventContext {
    pub fn new(correlation_id: CorrelationId) -> Self {
        Self {
            correlation_id,
            latest_event_id: Arc::new(Mutex::new(None)),
        }
    }

    pub fn correlation_id(&self) -> &CorrelationId {
        &self.correlation_id
    }

    pub fn latest_event_id(&self) -> Option<EventId> {
        self.latest_event_id.lock().clone()
    }
}

/// Pub/sub event stream manager for broadcasting and recording runtime events.
#[derive(Clone)]
pub struct RuntimeEventStream {
    event_log: Arc<RwLock<EventLog>>,
    subscribers: Arc<RwLock<Vec<Subscriber>>>,
    next_seq: Arc<AtomicU64>,
    activity_projection: Arc<RwLock<Option<RunActivityProjection>>>,
}

struct Subscriber {
    sender: async_channel::Sender<SequencedRuntimeEvent>,
    resync_required: Arc<AtomicBool>,
}

const MAX_EVENT_HISTORY: usize = 10_000;
const MAX_EVENT_HISTORY_BYTES: usize = 8 * 1024 * 1024;
const SUBSCRIBER_CAPACITY: usize = 256;
const MAX_SUBSCRIPTION_REPLAY_EVENTS: usize = SUBSCRIBER_CAPACITY;
const MAX_SUBSCRIPTION_REPLAY_BYTES: usize = 512 * 1024;

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
            activity_projection: Arc::new(RwLock::new(None)),
        }
    }

    pub fn from_history(history: Vec<SequencedRuntimeEvent>) -> Self {
        let mut event_log = EventLog::default();
        let mut max_seq = 0;
        for mut event in history {
            if event.event_id.is_empty() {
                event.event_id = EventId::from_sequence(event.event.run_id(), event.seq);
            }
            max_seq = max_seq.max(event.seq);
            event_log.push(event);
        }
        Self {
            event_log: Arc::new(RwLock::new(event_log)),
            subscribers: Arc::new(RwLock::new(Vec::new())),
            next_seq: Arc::new(AtomicU64::new(max_seq.saturating_add(1))),
            activity_projection: Arc::new(RwLock::new(None)),
        }
    }

    pub fn install_activity_projection(&self, projection: RunActivityProjection) {
        *self.activity_projection.write() = Some(projection);
    }

    pub fn activity_projection(&self) -> Option<RunActivityProjection> {
        self.activity_projection.read().clone()
    }

    /// Emits an event with the next sequence number to all active subscribers
    /// and stores it in the bounded event log.
    pub fn emit(&self, event: RuntimeEvent) -> EventId {
        self.emit_with_context(event, None, None)
    }

    pub fn emit_in_context(&self, event: RuntimeEvent, context: &RuntimeEventContext) -> EventId {
        let mut latest_event_id = context.latest_event_id.lock();
        let event_id = self.emit_with_context(
            event,
            Some(context.correlation_id.clone()),
            latest_event_id.clone(),
        );
        *latest_event_id = Some(event_id.clone());
        event_id
    }

    pub fn emit_with_context(
        &self,
        event: RuntimeEvent,
        correlation_id: Option<CorrelationId>,
        caused_by: Option<EventId>,
    ) -> EventId {
        let mut log = self.event_log.write();
        let seq = self.next_seq.fetch_add(1, Ordering::Relaxed);
        let sequenced = SequencedRuntimeEvent {
            event_id: EventId::from_sequence(event.run_id(), seq),
            seq,
            occurred_at: Utc::now(),
            correlation_id,
            caused_by,
            event,
        };
        if let Some(projection) = self.activity_projection.write().as_mut() {
            projection.apply(&sequenced);
        }
        log.push(sequenced.clone());

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
        sequenced.event_id
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
    /// live events. Kept for compatibility with consumers that explicitly need
    /// the complete retained history; runtime UI callers should prefer
    /// `subscribe_from`.
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
    /// detected gap. Replay is capped independently from the retained event
    /// history. If the cursor is stale or the tail exceeds that cap, no partial
    /// history is enqueued and `resync_required` is set so the consumer can
    /// rebuild from a snapshot while already subscribed to future events.
    pub fn subscribe_from(&self, from_seq: u64) -> EventSubscription {
        let log = self.event_log.read();
        let cursor_gap = log
            .events
            .front()
            .is_some_and(|event| from_seq.saturating_add(1) < event.seq);
        let mut replay = Vec::new();
        let mut replay_bytes = 0usize;
        let mut resync = cursor_gap;
        if !cursor_gap {
            for event in log.events.iter().filter(|event| event.seq > from_seq) {
                let event_bytes = serialized_event_size(event);
                if replay.len() >= MAX_SUBSCRIPTION_REPLAY_EVENTS
                    || replay_bytes.saturating_add(event_bytes) > MAX_SUBSCRIPTION_REPLAY_BYTES
                {
                    replay.clear();
                    resync = true;
                    break;
                }
                replay_bytes = replay_bytes.saturating_add(event_bytes);
                replay.push(event.clone());
            }
        }
        let capacity = replay.len().saturating_add(SUBSCRIBER_CAPACITY).max(1);
        let (sender, receiver) = async_channel::bounded(capacity);
        for event in &replay {
            if sender.try_send(event.clone()).is_err() {
                log::error!("failed to enqueue orchestration event for replay");
                break;
            }
        }
        let resync_required = Arc::new(AtomicBool::new(resync));
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

    /// Returns a bounded page of events after `from_seq`.
    ///
    /// The page is limited by both event count and serialized size. At least
    /// one event is returned when available, even if that event alone exceeds
    /// `max_bytes`, so callers can always advance their cursor.
    pub fn history_page(&self, from_seq: u64, limit: usize, max_bytes: usize) -> EventReplayPage {
        let log = self.event_log.read();
        let oldest_seq = log.events.front().map(|event| event.seq);
        let latest_seq = log.events.back().map(|event| event.seq);
        let gap = oldest_seq.is_some_and(|oldest| from_seq.saturating_add(1) < oldest);

        if limit == 0 || max_bytes == 0 {
            return EventReplayPage {
                events: Vec::new(),
                next_seq: None,
                oldest_seq,
                latest_seq,
                gap,
            };
        }

        let mut events = Vec::with_capacity(limit.min(log.events.len()));
        let mut serialized_bytes: usize = 0;
        let mut has_more = false;
        for event in log.events.iter().filter(|event| event.seq > from_seq) {
            if events.len() >= limit {
                has_more = true;
                break;
            }
            let event_bytes = serialized_event_size(event);
            if !events.is_empty() && serialized_bytes.saturating_add(event_bytes) > max_bytes {
                has_more = true;
                break;
            }
            serialized_bytes = serialized_bytes.saturating_add(event_bytes);
            events.push(event.clone());
        }

        let next_seq = if has_more {
            events.last().map(|event| event.seq)
        } else {
            None
        };
        EventReplayPage {
            events,
            next_seq,
            oldest_seq,
            latest_seq,
            gap,
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::RunId;
    use crate::state::RunState;

    fn state_event(run_id: &RunId) -> RuntimeEvent {
        RuntimeEvent::RunStateChanged {
            run_id: run_id.clone(),
            state: RunState::Running,
        }
    }

    #[test]
    fn history_page_respects_count_and_cursor() {
        let stream = RuntimeEventStream::new();
        let run_id = RunId::new();
        for _ in 0..4 {
            stream.emit(state_event(&run_id));
        }

        let first = stream.history_page(0, 2, usize::MAX);
        assert_eq!(first.events.len(), 2);
        assert_eq!(first.events[0].seq, 1);
        assert_eq!(first.events[1].seq, 2);
        assert_eq!(first.next_seq, Some(2));
        assert_eq!(first.oldest_seq, Some(1));
        assert_eq!(first.latest_seq, Some(4));
        assert!(!first.gap);

        let second = stream.history_page(first.next_seq.unwrap_or(0), 8, usize::MAX);
        assert_eq!(second.events.len(), 2);
        assert_eq!(second.events[0].seq, 3);
        assert_eq!(second.events[1].seq, 4);
        assert_eq!(second.next_seq, None);
    }

    #[test]
    fn history_page_reports_cursor_gap_after_eviction() {
        let stream = RuntimeEventStream::new();
        let run_id = RunId::new();
        for _ in 0..10_001 {
            stream.emit(state_event(&run_id));
        }

        let page = stream.history_page(0, 1, usize::MAX);
        assert_eq!(page.oldest_seq, Some(2));
        assert!(page.gap);
        assert_eq!(page.events.first().map(|event| event.seq), Some(2));
    }

    #[test]
    fn subscribe_from_replays_a_small_tail() {
        let stream = RuntimeEventStream::new();
        let run_id = RunId::new();
        for _ in 0..4 {
            stream.emit(state_event(&run_id));
        }

        let subscription = stream.subscribe_from(1);
        assert!(!subscription.resync_required.load(Ordering::SeqCst));
        let replayed = (0..3)
            .map(|_| subscription.receiver.try_recv().expect("replayed event"))
            .map(|event| event.seq)
            .collect::<Vec<_>>();
        assert_eq!(replayed, vec![2, 3, 4]);
        assert!(subscription.receiver.try_recv().is_err());
    }

    #[test]
    fn subscribe_from_requests_resync_instead_of_enqueuing_a_large_tail() {
        let stream = RuntimeEventStream::new();
        let run_id = RunId::new();
        for _ in 0..=MAX_SUBSCRIPTION_REPLAY_EVENTS {
            stream.emit(state_event(&run_id));
        }

        let subscription = stream.subscribe_from(0);
        assert!(subscription.resync_required.load(Ordering::SeqCst));
        assert!(subscription.receiver.try_recv().is_err());
    }

    #[test]
    fn subscribe_from_requests_resync_for_an_evicted_cursor() {
        let stream = RuntimeEventStream::new();
        let run_id = RunId::new();
        for _ in 0..=MAX_EVENT_HISTORY {
            stream.emit(state_event(&run_id));
        }

        let subscription = stream.subscribe_from(0);
        assert!(subscription.resync_required.load(Ordering::SeqCst));
        assert!(subscription.receiver.try_recv().is_err());
    }

    #[test]
    fn concurrent_emit_preserves_sequence_order() {
        let stream = RuntimeEventStream::new();
        let run_id = RunId::new();
        let mut emitters = Vec::new();
        for _ in 0..4 {
            let stream = stream.clone();
            let run_id = run_id.clone();
            emitters.push(std::thread::spawn(move || {
                for _ in 0..500 {
                    stream.emit(state_event(&run_id));
                }
            }));
        }
        for emitter in emitters {
            emitter.join().expect("event emitter");
        }

        let history = stream.history();
        assert_eq!(history.len(), 2_000);
        assert!(
            history
                .windows(2)
                .all(|events| events[1].seq == events[0].seq + 1)
        );
    }

    #[test]
    fn legacy_envelopes_receive_compatible_defaults() -> anyhow::Result<()> {
        let run_id = RunId::from_string("run-legacy");
        let value = serde_json::json!({
            "seq": 7,
            "event": {
                "type": "run_state_changed",
                "run_id": run_id,
                "state": "running"
            }
        });

        let envelope: SequencedRuntimeEvent = serde_json::from_value(value)?;
        assert!(envelope.event_id.is_empty());
        assert_eq!(envelope.occurred_at, default_event_timestamp());

        let stream = RuntimeEventStream::from_history(vec![envelope]);
        let restored = stream.history();
        assert_eq!(restored[0].event_id.to_string(), "run-legacy:7");
        Ok(())
    }

    #[test]
    fn event_context_builds_a_single_causal_chain() {
        let stream = RuntimeEventStream::new();
        let run_id = RunId::from_string("run-causal");
        let context = RuntimeEventContext::new(CorrelationId::from_string("attempt-1"));

        let first_id = stream.emit_in_context(state_event(&run_id), &context);
        let second_id = stream.emit_in_context(state_event(&run_id), &context);
        let history = stream.history();

        assert_eq!(
            history[0].correlation_id.as_ref(),
            Some(context.correlation_id())
        );
        assert!(history[0].caused_by.is_none());
        assert_eq!(
            history[1].correlation_id.as_ref(),
            Some(context.correlation_id())
        );
        assert_eq!(history[1].caused_by.as_ref(), Some(&first_id));
        assert_eq!(context.latest_event_id(), Some(second_id));
    }
}
