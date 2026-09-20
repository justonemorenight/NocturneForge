use std::{
    io::{self, Write},
    sync::{Arc, Weak},
    time::{Duration, Instant},
};

use agent_settings::AgentSettings;
use futures::{StreamExt as _, future};
use gpui::{App, AsyncApp, Global, Task, WeakEntity};
use language_model::{
    LanguageModel, LanguageModelCompletionEvent, LanguageModelRequest, LanguageModelRequestMessage,
    LanguageModelToolChoice, MessageContent, Role, StopReason, TokenUsage,
};
use parking_lot::Mutex;
use settings::{Settings as _, SettingsStore};
use uuid::Uuid;

// Never append warming output to the conversation or route it through tool execution.
const PING_PROMPT: &str =
    "[cache keep-alive] Automated cache-warming ping. Do not call tools. Reply with exactly: ok";
const TICK: Duration = Duration::from_secs(15);
const VALIDITY_CHECK_INTERVAL: Duration = Duration::from_secs(1);
const BUDGET_WINDOW: Duration = Duration::from_secs(3600);

#[derive(Clone, Debug, PartialEq, Eq)]
struct Config {
    ttl: Duration,
    lead: Duration,
    idle_window: Duration,
    max_pings: u32,
    max_requests_per_hour: u32,
    max_input_tokens_per_hour: u64,
    timeout: Duration,
    max_output_bytes: usize,
    max_capture_bytes: usize,
    max_captures: usize,
    min_cache_hit_percent: u32,
}

impl Config {
    fn from_settings(settings: &settings::CacheKeepaliveSettings) -> Self {
        let ttl = settings
            .estimated_ttl_seconds
            .unwrap_or(1800)
            .clamp(60, 7200);
        Self {
            ttl: Duration::from_secs(ttl),
            lead: Duration::from_secs(settings.lead_seconds.unwrap_or(120).clamp(1, ttl - 1)),
            idle_window: Duration::from_secs(
                settings.idle_window_seconds.unwrap_or(5400).clamp(60, 7200),
            ),
            max_pings: settings.max_pings.unwrap_or(2).min(8),
            max_requests_per_hour: settings.max_requests_per_hour.unwrap_or(4).min(16),
            max_input_tokens_per_hour: settings
                .max_input_tokens_per_hour
                .unwrap_or(1_000_000)
                .min(8_000_000),
            timeout: Duration::from_secs(settings.timeout_seconds.unwrap_or(30).clamp(1, 120)),
            max_output_bytes: settings.max_output_bytes.unwrap_or(1024).clamp(1, 16384),
            max_capture_bytes: settings
                .max_capture_bytes
                .unwrap_or(1024 * 1024)
                .clamp(1024, 4 * 1024 * 1024),
            max_captures: settings.max_captures.unwrap_or(8).clamp(1, 16),
            min_cache_hit_percent: settings.min_cache_hit_percent.unwrap_or(90).clamp(50, 100),
        }
    }

    fn current(cx: &App) -> Self {
        Self::from_settings(&AgentSettings::get_global(cx).cache_keepalive_config)
    }
}

impl Default for Config {
    fn default() -> Self {
        Self::from_settings(&Default::default())
    }
}

struct CaptureSizeLimit {
    remaining: usize,
}

impl Write for CaptureSizeLimit {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.remaining = self
            .remaining
            .checked_sub(bytes.len())
            .ok_or_else(|| io::Error::other("Cache capture exceeds its byte budget"))?;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn validate_capture_size(request: &LanguageModelRequest, limit: usize) -> serde_json::Result<()> {
    serde_json::to_writer(CaptureSizeLimit { remaining: limit }, request)
}

fn sufficient_cache_hit(usage: &TokenUsage, minimum: u32) -> bool {
    // Native input_tokens excludes both cache reads and cache writes.
    let cached = u128::from(usage.cache_read_input_tokens);
    let total =
        cached + u128::from(usage.input_tokens) + u128::from(usage.cache_creation_input_tokens);
    cached > 0 && cached * 100 >= total * u128::from(minimum)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Capturing,
    Waiting,
    Warming,
    Paused,
}

impl Phase {
    fn label(self) -> &'static str {
        match self {
            Self::Capturing => "Waiting for real request",
            Self::Waiting => "Waiting (retention estimated)",
            Self::Warming => "Warming",
            Self::Paused => "Paused",
        }
    }
}

struct CaptureState {
    generation: Uuid,
    phase: Phase,
    real_turn_at: Instant,
    last_touch: Instant,
    attempts: u32,
    input_tokens: u64,
    reason: Option<String>,
}

impl CaptureState {
    fn new(now: Instant) -> Self {
        Self {
            generation: Uuid::new_v4(),
            phase: Phase::Capturing,
            real_turn_at: now,
            last_touch: now,
            attempts: 0,
            input_tokens: 0,
            reason: None,
        }
    }

    fn confirm(&mut self, generation: Uuid, now: Instant) {
        if self.generation == generation && self.phase == Phase::Capturing {
            self.phase = Phase::Waiting;
            self.last_touch = now;
        }
    }

    fn due(&mut self, now: Instant, config: &Config) -> bool {
        if self.phase != Phase::Waiting {
            return false;
        }
        let idle = now.saturating_duration_since(self.real_turn_at);
        let age = now.saturating_duration_since(self.last_touch);
        if idle >= config.idle_window || age >= config.ttl || self.attempts >= config.max_pings {
            self.pause("Idle window, estimated retention, or request limit reached");
            return false;
        }
        age >= config.ttl.saturating_sub(config.lead)
    }

    fn pause(&mut self, reason: &str) {
        self.phase = Phase::Paused;
        self.reason = Some(reason.chars().take(256).collect());
    }

    fn finish(&mut self, generation: Uuid, outcome: &PingOutcome, config: &Config, now: Instant) {
        if self.generation != generation {
            return;
        }
        if outcome.cancelled {
            self.phase = Phase::Waiting;
            self.reason = None;
            self.last_touch = now;
            return;
        }
        if let Some(error) = &outcome.error {
            self.pause(error);
        } else if !outcome.saw_usage
            || !sufficient_cache_hit(&outcome.usage, config.min_cache_hit_percent)
        {
            self.pause("Insufficient cache reuse");
        } else {
            self.phase = Phase::Waiting;
            self.last_touch = now;
        }
    }
}

#[derive(Default)]
struct Ledger {
    attempts: std::collections::VecDeque<(Instant, u64)>,
    total_attempts: u64,
    incomplete_attempts: u64,
    usage: TokenUsage,
}

impl Ledger {
    fn reserve(&mut self, now: Instant, limit: u32, input_tokens: u64, token_limit: u64) -> bool {
        while self
            .attempts
            .front()
            .is_some_and(|(at, _)| now.saturating_duration_since(*at) >= BUDGET_WINDOW)
        {
            self.attempts.pop_front();
        }
        let reserved = self
            .attempts
            .iter()
            .fold(0u64, |sum, (_, tokens)| sum.saturating_add(*tokens));
        if self.attempts.len() >= limit as usize
            || input_tokens == 0
            || input_tokens > token_limit.saturating_sub(reserved)
        {
            return false;
        }
        self.attempts.push_back((now, input_tokens));
        self.total_attempts = self.total_attempts.saturating_add(1);
        true
    }

    fn record(&mut self, outcome: &PingOutcome) {
        if let Some((_, reserved)) = self.attempts.back_mut() {
            let actual = outcome
                .usage
                .input_tokens
                .saturating_add(outcome.usage.cache_read_input_tokens)
                .saturating_add(outcome.usage.cache_creation_input_tokens);
            *reserved = (*reserved).max(actual);
        }
        self.usage.input_tokens = self
            .usage
            .input_tokens
            .saturating_add(outcome.usage.input_tokens);
        self.usage.output_tokens = self
            .usage
            .output_tokens
            .saturating_add(outcome.usage.output_tokens);
        self.usage.cache_read_input_tokens = self
            .usage
            .cache_read_input_tokens
            .saturating_add(outcome.usage.cache_read_input_tokens);
        self.usage.cache_creation_input_tokens = self
            .usage
            .cache_creation_input_tokens
            .saturating_add(outcome.usage.cache_creation_input_tokens);
        if !outcome.saw_usage || outcome.error.is_some() {
            self.incomplete_attempts = self.incomplete_attempts.saturating_add(1);
        }
    }
}

struct Capture {
    owner: WeakEntity<crate::Thread>,
    scope: String,
    model: Arc<dyn LanguageModel>,
    request: LanguageModelRequest,
    state: CaptureState,
    config: Config,
    abort: Option<future::AbortHandle>,
}

impl Drop for Capture {
    fn drop(&mut self) {
        if let Some(abort) = self.abort.take() {
            abort.abort();
        }
    }
}

impl Capture {
    fn valid(&self, cx: &App) -> bool {
        self.model.cache_warming_scope(cx).as_deref() == Some(self.scope.as_str())
            && self
                .owner
                .read_with(cx, |thread, _| {
                    !thread.is_subagent()
                        && thread.model().is_some_and(|model| {
                            model.id() == self.model.id()
                                && model.provider_id() == self.model.provider_id()
                        })
                })
                .unwrap_or(false)
    }

    fn idle(
        &self,
        cx: &App,
        affinity: &str,
        enabled_threads: &[(String, WeakEntity<crate::Thread>)],
    ) -> bool {
        if !self
            .owner
            .read_with(cx, |thread, _| thread.is_turn_complete())
            .unwrap_or(false)
        {
            return false;
        }
        !enabled_threads.iter().any(|(_, owner)| {
            owner
                .read_with(cx, |thread, _| {
                    thread.prompt_cache_affinity() == affinity && !thread.is_turn_complete()
                })
                .unwrap_or(false)
        })
    }
}

#[derive(Default)]
struct Registry {
    captures: Vec<(String, Capture)>,
    enabled_threads: Vec<(String, WeakEntity<crate::Thread>)>,
    ledger: Ledger,
}

struct PendingPing {
    affinity: String,
    generation: Uuid,
    model: Arc<dyn LanguageModel>,
    request: LanguageModelRequest,
    config: Config,
    cancellation: Option<future::AbortRegistration>,
}

impl Registry {
    fn disable_all(&mut self) {
        self.captures.clear();
        self.enabled_threads.clear();
    }

    fn insert(&mut self, affinity: String, capture: Capture) {
        self.captures.retain(|(id, _)| id != &affinity);
        while self.captures.len() >= capture.config.max_captures {
            self.captures.remove(0);
        }
        self.captures.push((affinity, capture));
    }

    fn interrupt_ping(&mut self, affinity: &str) {
        if let Some((_, capture)) = self.captures.iter_mut().find(|(id, _)| id == affinity) {
            if let Some(abort) = capture.abort.take() {
                abort.abort();
            }
        }
    }

    fn next_ping(&mut self, cx: &App, config: &Config, now: Instant) -> Option<PendingPing> {
        self.enabled_threads
            .retain(|(_, owner)| owner.upgrade().is_some());
        self.captures.retain(|(_, capture)| {
            capture.config == *config
                && capture.valid(cx)
                && now.saturating_duration_since(capture.state.real_turn_at) < config.idle_window
        });
        // Service earliest cache deadlines first, independently of thread creation order.
        self.captures
            .sort_by_key(|(_, capture)| capture.state.last_touch);
        for (id, capture) in &mut self.captures {
            if !capture.idle(cx, id, &self.enabled_threads) || !capture.state.due(now, config) {
                continue;
            }
            if !self.ledger.reserve(
                now,
                config.max_requests_per_hour,
                capture.state.input_tokens,
                config.max_input_tokens_per_hour,
            ) {
                capture
                    .state
                    .pause("App-wide hourly request or input budget reached");
                continue;
            }
            capture.state.phase = Phase::Warming;
            capture.state.attempts += 1;
            let (abort, cancellation) = future::AbortHandle::new_pair();
            capture.abort = Some(abort);
            return Some(PendingPing {
                affinity: id.clone(),
                generation: capture.state.generation,
                model: capture.model.clone(),
                request: capture.request.clone(),
                config: config.clone(),
                cancellation: Some(cancellation),
            });
        }
        None
    }

    fn finish(&mut self, pending: &PendingPing, outcome: &PingOutcome, now: Instant) {
        // A real turn may have replaced the capture while the ping was running.
        // Usage belongs to the app ledger even when the old generation is gone.
        self.ledger.record(outcome);
        if let Some((_, capture)) = self
            .captures
            .iter_mut()
            .find(|(id, _)| id == &pending.affinity)
        {
            if capture.state.generation == pending.generation {
                capture.abort = None;
            }
            capture
                .state
                .finish(pending.generation, outcome, &pending.config, now);
        }
    }
}

struct GlobalKeepAlive {
    registry: Arc<Mutex<Registry>>,
    _ticker: Task<()>,
}

impl Global for GlobalKeepAlive {}

pub struct RequestCapture {
    registry: Weak<Mutex<Registry>>,
    affinity: String,
    generation: Uuid,
    confirmed: bool,
}

impl RequestCapture {
    pub fn confirm(mut self, usage: TokenUsage) {
        if let Some(registry) = self.registry.upgrade() {
            if let Some((_, capture)) = registry
                .lock()
                .captures
                .iter_mut()
                .find(|(id, _)| id == &self.affinity)
            {
                if capture.state.generation == self.generation {
                    let input = usage
                        .input_tokens
                        .saturating_add(usage.cache_read_input_tokens)
                        .saturating_add(usage.cache_creation_input_tokens);
                    if input > 0 {
                        capture.state.input_tokens = input.saturating_add(PING_PROMPT.len() as u64);
                        capture.state.confirm(self.generation, Instant::now());
                    } else {
                        capture
                            .state
                            .pause("Real request did not report input usage");
                    }
                }
            }
        }
        self.confirmed = true;
    }
}

impl Drop for RequestCapture {
    fn drop(&mut self) {
        if self.confirmed {
            return;
        }
        if let Some(registry) = self.registry.upgrade() {
            registry.lock().captures.retain(|(id, capture)| {
                id != &self.affinity || capture.state.generation != self.generation
            });
        }
    }
}

pub fn invalidate(target: &str, cx: &App) {
    if let Some(global) = cx.try_global::<GlobalKeepAlive>() {
        global
            .registry
            .lock()
            .captures
            .retain(|(affinity, capture)| {
                if affinity == target || capture.request.thread_id.as_deref() == Some(target) {
                    return false;
                }
                let matches_owner = capture
                    .owner
                    .read_with(cx, |thread, _| {
                        thread.id().0.as_ref() == target || thread.prompt_cache_affinity() == target
                    })
                    .unwrap_or(false);
                !matches_owner
            });
    }
}

pub fn enabled_for_thread(thread_id: &str, cx: &App) -> bool {
    AgentSettings::get_global(cx).cache_keepalive
        && cx.try_global::<GlobalKeepAlive>().is_some_and(|global| {
            global
                .registry
                .lock()
                .enabled_threads
                .iter()
                .any(|(id, _)| id == thread_id)
        })
}

fn thread_affinity(id: &str, owner: &WeakEntity<crate::Thread>, cx: &App) -> String {
    owner
        .read_with(cx, |thread, _| thread.prompt_cache_affinity())
        .unwrap_or_else(|_| id.to_string())
}

pub fn toggle_thread(thread_id: String, owner: WeakEntity<crate::Thread>, cx: &App) {
    let Some(global) = cx.try_global::<GlobalKeepAlive>() else {
        return;
    };
    let mut registry = global.registry.lock();
    let affinity = thread_affinity(&thread_id, &owner, cx);
    registry
        .enabled_threads
        .retain(|(_, owner)| owner.upgrade().is_some());

    let was_enabled = registry
        .enabled_threads
        .iter()
        .any(|(id, _)| id == &thread_id);

    if was_enabled {
        registry.enabled_threads.retain(|(id, _)| id != &thread_id);
        // Drop captures owned by the disabled thread so remaining forks or threads can capture anew.
        registry.captures.retain(|(_id, capture)| {
            let owned_by_disabled = capture.request.thread_id.as_deref() == Some(&thread_id)
                || capture
                    .owner
                    .read_with(cx, |thread, _| thread.id().0.as_ref() == thread_id)
                    .unwrap_or(false);
            !owned_by_disabled
        });
    } else {
        let affinity_already_enabled = registry
            .enabled_threads
            .iter()
            .any(|(id, owner)| thread_affinity(id, owner, cx) == affinity);

        if !affinity_already_enabled {
            let mut unique_affinities: Vec<String> = Vec::new();
            for (id, owner) in &registry.enabled_threads {
                let aff = thread_affinity(id, owner, cx);
                if !unique_affinities.contains(&aff) {
                    unique_affinities.push(aff);
                }
            }
            if unique_affinities.len() >= Config::current(cx).max_captures {
                if let Some(oldest_affinity) = unique_affinities.first().cloned() {
                    registry
                        .enabled_threads
                        .retain(|(id, owner)| thread_affinity(id, owner, cx) != oldest_affinity);
                    registry.captures.retain(|(id, capture)| {
                        if id == &oldest_affinity {
                            return false;
                        }
                        let capture_affinity = capture
                            .owner
                            .read_with(cx, |thread, _| thread.prompt_cache_affinity())
                            .unwrap_or_else(|_| {
                                capture.request.thread_id.clone().unwrap_or_default()
                            });
                        capture_affinity != oldest_affinity
                    });
                }
            }
        }
        registry.enabled_threads.push((thread_id.clone(), owner));
    }
    let still_enabled_for_affinity = registry
        .enabled_threads
        .iter()
        .any(|(id, owner)| thread_affinity(id, owner, cx) == affinity);
    if !still_enabled_for_affinity {
        registry.captures.retain(|(id, capture)| {
            id != &affinity
                && id != &thread_id
                && capture.request.thread_id.as_deref() != Some(&thread_id)
        });
    }
}

pub fn status(thread_id: &str, cx: &App) -> String {
    if !AgentSettings::get_global(cx).cache_keepalive {
        return "Cache warming: Off".into();
    }
    let Some(global) = cx.try_global::<GlobalKeepAlive>() else {
        return "Cache warming: Unavailable".into();
    };
    let registry = global.registry.lock();
    if !registry
        .enabled_threads
        .iter()
        .any(|(id, _)| id == thread_id)
    {
        return "Cache warming: Off for this thread".into();
    }
    let owner_affinity = registry
        .enabled_threads
        .iter()
        .find(|(id, _)| id == thread_id)
        .and_then(|(_, owner)| {
            owner
                .read_with(cx, |thread, _| thread.prompt_cache_affinity())
                .ok()
        });
    let state = registry
        .captures
        .iter()
        .find(|(id, capture)| {
            id == thread_id
                || Some(id.as_str()) == owner_affinity.as_deref()
                || capture.request.thread_id.as_deref() == Some(thread_id)
        })
        .map(|(_, capture)| {
            capture
                .state
                .reason
                .as_deref()
                .unwrap_or(capture.state.phase.label())
        })
        .unwrap_or("Waiting for next successful request");
    format!(
        "Cache warming: {state}\nApp session: {} attempts; input {} / cached {} / cache write {} / output {} tokens; {} incomplete usage reports",
        registry.ledger.total_attempts,
        registry.ledger.usage.input_tokens,
        registry.ledger.usage.cache_read_input_tokens,
        registry.ledger.usage.cache_creation_input_tokens,
        registry.ledger.usage.output_tokens,
        registry.ledger.incomplete_attempts,
    )
}

pub fn record_turn_activity(
    cx: &App,
    owner: WeakEntity<crate::Thread>,
    model: Arc<dyn LanguageModel>,
    request: &LanguageModelRequest,
) -> Option<RequestCapture> {
    let global = cx.try_global::<GlobalKeepAlive>()?;
    if !AgentSettings::get_global(cx).cache_keepalive {
        global.registry.lock().disable_all();
        return None;
    }
    let thread_id = request.thread_id.clone()?;
    let affinity = request
        .prompt_cache_key
        .clone()
        .unwrap_or_else(|| thread_id.clone());

    if model.provider_id().0.as_ref() != "openai-subscribed"
        || owner
            .read_with(cx, |thread, _| thread.is_subagent())
            .unwrap_or(true)
    {
        return None;
    }
    if !enabled_for_thread(&thread_id, cx) {
        return None;
    }

    // Family activity interrupts any active ping for this affinity without discarding confirmed captures.
    global.registry.lock().interrupt_ping(&affinity);
    let scope = model.cache_warming_scope(cx)?;
    let config = Config::current(cx);
    if config.max_pings == 0 || config.max_requests_per_hour == 0 {
        return None;
    }
    if let Err(error) = validate_capture_size(request, config.max_capture_bytes) {
        log::debug!("Skipping cache capture: {error}");
        return None;
    }
    let state = CaptureState::new(Instant::now());
    let guard = RequestCapture {
        registry: Arc::downgrade(&global.registry),
        affinity: affinity.clone(),
        generation: state.generation,
        confirmed: false,
    };
    global.registry.lock().insert(
        affinity,
        Capture {
            owner,
            scope,
            model,
            request: request.clone(),
            state,
            config,
            abort: None,
        },
    );
    Some(guard)
}

pub fn init(cx: &mut App) {
    if cx.try_global::<GlobalKeepAlive>().is_some() {
        return;
    }
    let ticker = cx.spawn(async move |cx| ticker(cx).await);
    cx.set_global(GlobalKeepAlive {
        registry: Arc::new(Mutex::new(Registry::default())),
        _ticker: ticker,
    });
    cx.observe_global::<SettingsStore>(|cx| {
        if !AgentSettings::get_global(cx).cache_keepalive
            && let Some(global) = cx.try_global::<GlobalKeepAlive>()
        {
            global.registry.lock().disable_all();
        }
    })
    .detach();
}

fn pending_valid(pending: &PendingPing, cx: &App) -> bool {
    if !AgentSettings::get_global(cx).cache_keepalive || Config::current(cx) != pending.config {
        return false;
    }
    cx.try_global::<GlobalKeepAlive>().is_some_and(|global| {
        let registry = global.registry.lock();
        registry.captures.iter().any(|(id, capture)| {
            id == &pending.affinity
                && capture.state.generation == pending.generation
                && capture.state.phase == Phase::Warming
                && capture.valid(cx)
                && capture.idle(cx, &pending.affinity, &registry.enabled_threads)
        })
    })
}

async fn ticker(cx: &mut AsyncApp) {
    loop {
        cx.background_executor().timer(TICK).await;
        let pending = cx.update(|cx| {
            let global = cx.try_global::<GlobalKeepAlive>()?;
            if !AgentSettings::get_global(cx).cache_keepalive {
                global.registry.lock().disable_all();
                return None;
            }
            global
                .registry
                .lock()
                .next_ping(cx, &Config::current(cx), Instant::now())
        });
        let Some(mut pending) = pending else {
            continue;
        };
        let mut outcome = PingOutcome::default();
        let Some(cancellation) = pending.cancellation.take() else {
            continue;
        };
        let (error, cancelled) = {
            let monitor_context = cx.clone();
            let validity = async {
                loop {
                    if !monitor_context.update(|cx| pending_valid(&pending, cx)) {
                        return;
                    }
                    monitor_context
                        .background_executor()
                        .timer(VALIDITY_CHECK_INTERVAL)
                        .await;
                }
            };
            let ping = future::Abortable::new(
                run_ping_stream(
                    pending.model.clone(),
                    pending.request.clone(),
                    &pending.config,
                    &mut outcome,
                    cx,
                ),
                cancellation,
            );
            futures::pin_mut!(validity, ping);
            match future::select(validity, ping).await {
                future::Either::Left(_) => (Some("Session or settings changed".to_owned()), true),
                future::Either::Right((Ok(result), _)) => (result.err(), false),
                future::Either::Right((Err(_), _)) => {
                    (Some("Cache warming cancelled".into()), true)
                }
            }
        };
        outcome.error = error;
        outcome.cancelled = cancelled;
        cx.update(|cx| {
            if let Some(global) = cx.try_global::<GlobalKeepAlive>() {
                global
                    .registry
                    .lock()
                    .finish(&pending, &outcome, Instant::now());
            }
        });
    }
}

#[derive(Default, Debug)]
struct PingOutcome {
    usage: TokenUsage,
    saw_usage: bool,
    error: Option<String>,
    cancelled: bool,
}

async fn run_ping_stream(
    model: Arc<dyn LanguageModel>,
    mut request: LanguageModelRequest,
    config: &Config,
    outcome: &mut PingOutcome,
    cx: &mut AsyncApp,
) -> Result<(), String> {
    let timeout = cx.background_executor().timer(config.timeout);
    let operation = async {
        request.tool_choice = Some(LanguageModelToolChoice::None);
        request.compact_at_tokens = None;
        request.messages.push(LanguageModelRequestMessage {
            role: Role::User,
            content: vec![MessageContent::Text(PING_PROMPT.to_string())],
            cache: false,
            reasoning_details: None,
        });
        let mut events = model
            .stream_completion(request, cx)
            .await
            .map_err(|error| error.to_string())?;
        let mut output_bytes = 0usize;
        let mut completed = false;
        while let Some(event) = events.next().await {
            match event.map_err(|error| error.to_string())? {
                LanguageModelCompletionEvent::UsageUpdate(usage) => {
                    outcome.usage = usage;
                    outcome.saw_usage = true;
                }
                LanguageModelCompletionEvent::Text(text)
                | LanguageModelCompletionEvent::Thinking { text, .. }
                | LanguageModelCompletionEvent::RedactedThinking { data: text } => {
                    output_bytes = output_bytes.saturating_add(text.len());
                    if output_bytes > config.max_output_bytes {
                        return Err("Warming output limit exceeded".into());
                    }
                }
                LanguageModelCompletionEvent::ReasoningDetails(details) => {
                    let mut limit = CaptureSizeLimit {
                        remaining: config.max_output_bytes.saturating_sub(output_bytes),
                    };
                    serde_json::to_writer(&mut limit, &details)
                        .map_err(|_| "Warming reasoning output limit exceeded".to_string())?;
                    output_bytes = config.max_output_bytes - limit.remaining;
                }
                LanguageModelCompletionEvent::ToolUse(_)
                | LanguageModelCompletionEvent::ToolUseJsonParseError { .. }
                | LanguageModelCompletionEvent::Compaction(_) => {
                    return Err("Unexpected tool or compaction during warming".into());
                }
                LanguageModelCompletionEvent::Stop(StopReason::EndTurn) => completed = true,
                LanguageModelCompletionEvent::Stop(_) => {
                    return Err("Warming ended without a normal completion".into());
                }
                _ => {}
            }
        }
        if !completed || !outcome.saw_usage {
            return Err("Incomplete warming response or missing usage".into());
        }
        Ok(())
    };
    futures::pin_mut!(operation, timeout);
    match future::select(operation, timeout).await {
        future::Either::Left((result, _)) => result,
        future::Either::Right(_) => Err("Warming request timed out".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_client_protocol::schema::v1 as acp;
    use gpui::AppContext as _;

    fn warm_outcome() -> PingOutcome {
        PingOutcome {
            usage: TokenUsage {
                input_tokens: 10,
                cache_read_input_tokens: 90,
                ..Default::default()
            },
            saw_usage: true,
            error: None,
            cancelled: false,
        }
    }

    #[test]
    fn config_clamps_untrusted_limits() {
        let config = Config::from_settings(&settings::CacheKeepaliveSettings {
            estimated_ttl_seconds: Some(0),
            lead_seconds: Some(u64::MAX),
            max_requests_per_hour: Some(u32::MAX),
            max_capture_bytes: Some(usize::MAX),
            max_captures: Some(usize::MAX),
            min_cache_hit_percent: Some(0),
            ..Default::default()
        });
        assert_eq!(config.ttl, Duration::from_secs(60));
        assert!(config.lead < config.ttl);
        assert_eq!(config.max_requests_per_hour, 16);
        assert_eq!(config.max_capture_bytes, 4 * 1024 * 1024);
        assert_eq!(config.max_captures, 16);
        assert_eq!(config.min_cache_hit_percent, 50);
    }

    #[test]
    fn capture_must_be_confirmed_and_cannot_warm_after_expiry() {
        let now = Instant::now();
        let config = Config::default();
        let mut state = CaptureState::new(now);
        assert!(!state.due(now + config.ttl - config.lead, &config));
        state.confirm(state.generation, now);
        assert!(state.due(now + config.ttl - config.lead, &config));
        assert!(!state.due(now + config.ttl, &config));
        assert_eq!(state.phase, Phase::Paused);
    }

    #[test]
    fn stale_confirmation_and_completion_cannot_mutate_new_generation() {
        let now = Instant::now();
        let mut state = CaptureState::new(now);
        let stale = Uuid::new_v4();
        state.confirm(stale, now);
        assert_eq!(state.phase, Phase::Capturing);
        state.phase = Phase::Warming;
        state.finish(stale, &warm_outcome(), &Config::default(), now);
        assert_eq!(state.phase, Phase::Warming);
    }

    #[test]
    fn warm_touch_does_not_extend_idle_window() {
        let now = Instant::now();
        let config = Config::default();
        let mut state = CaptureState::new(now);
        state.finish(
            state.generation,
            &warm_outcome(),
            &config,
            now + config.idle_window,
        );
        assert!(!state.due(now + config.idle_window, &config));
    }

    #[test]
    fn failure_missing_usage_and_partial_hit_pause_without_retry() {
        for outcome in [
            PingOutcome::default(),
            PingOutcome {
                error: Some("cancelled".into()),
                ..warm_outcome()
            },
            PingOutcome {
                usage: TokenUsage {
                    input_tokens: 99,
                    cache_read_input_tokens: 1,
                    ..Default::default()
                },
                ..warm_outcome()
            },
        ] {
            let now = Instant::now();
            let mut state = CaptureState::new(now);
            state.finish(state.generation, &outcome, &Config::default(), now);
            assert_eq!(state.phase, Phase::Paused);
        }
    }

    #[test]
    fn app_budget_is_sliding_and_does_not_refund_failures() {
        let now = Instant::now();
        let mut ledger = Ledger::default();
        assert!(ledger.reserve(now, 1, 100, 100));
        ledger.record(&PingOutcome::default());
        assert!(!ledger.reserve(now + BUDGET_WINDOW - Duration::from_secs(1), 1, 100, 100));
        assert!(ledger.reserve(now + BUDGET_WINDOW, 1, 100, 100));
        assert_eq!(ledger.incomplete_attempts, 1);
        assert_eq!(ledger.total_attempts, 2);
        assert!(!ledger.reserve(now + BUDGET_WINDOW * 2, 0, 100, 100));
        assert!(!ledger.reserve(now + BUDGET_WINDOW * 2, 1, 101, 100));
    }

    #[test]
    fn accounting_preserves_partial_usage_on_cancellation_and_saturates() {
        let mut ledger = Ledger::default();
        ledger.record(&PingOutcome {
            error: Some("cancelled".into()),
            ..warm_outcome()
        });
        assert_eq!(ledger.usage.cache_read_input_tokens, 90);
        assert_eq!(ledger.incomplete_attempts, 1);
        ledger.usage.input_tokens = u64::MAX;
        ledger.record(&warm_outcome());
        assert_eq!(ledger.usage.input_tokens, u64::MAX);
    }

    #[test]
    fn size_limit_and_ratio_are_overflow_safe() {
        assert!(validate_capture_size(&LanguageModelRequest::default(), 1024).is_ok());
        let request = LanguageModelRequest {
            thread_id: Some("x".repeat(2048)),
            ..Default::default()
        };
        assert!(validate_capture_size(&request, 1024).is_err());
        let mut writer = CaptureSizeLimit { remaining: 3 };
        assert!(writer.write_all(b"abc").is_ok());
        assert!(writer.write_all(b"d").is_err());
        assert!(!sufficient_cache_hit(
            &TokenUsage {
                cache_read_input_tokens: u64::MAX,
                input_tokens: u64::MAX,
                ..Default::default()
            },
            90
        ));
    }

    fn spawn_ping(
        cx: &mut gpui::TestAppContext,
        config: Config,
    ) -> (
        Arc<language_model::fake_provider::FakeLanguageModel>,
        Task<PingOutcome>,
    ) {
        let model = Arc::new(language_model::fake_provider::FakeLanguageModel::default());
        let task = cx.update(|cx| {
            let model = model.clone();
            cx.spawn(async move |cx| {
                let mut outcome = PingOutcome::default();
                outcome.error = run_ping_stream(
                    model,
                    LanguageModelRequest::default(),
                    &config,
                    &mut outcome,
                    cx,
                )
                .await
                .err();
                outcome
            })
        });
        cx.run_until_parked();
        (model, task)
    }

    #[gpui::test(iterations = 10)]
    async fn stream_success_has_usage_and_never_runs_tools(cx: &mut gpui::TestAppContext) {
        let (model, task) = spawn_ping(cx, Config::default());
        let request = model.pending_completions().pop().unwrap();
        assert_eq!(request.tool_choice, Some(LanguageModelToolChoice::None));
        assert_eq!(request.messages.len(), 1);
        model.send_last_completion_stream_event(LanguageModelCompletionEvent::UsageUpdate(
            warm_outcome().usage,
        ));
        model.send_last_completion_stream_event(LanguageModelCompletionEvent::Stop(
            StopReason::EndTurn,
        ));
        model.end_last_completion_stream();
        let outcome = task.await;
        assert!(outcome.error.is_none());
        assert_eq!(outcome.usage.cache_read_input_tokens, 90);
    }

    #[gpui::test(iterations = 10)]
    async fn output_limit_keeps_usage_received_before_abort(cx: &mut gpui::TestAppContext) {
        let (model, task) = spawn_ping(
            cx,
            Config {
                max_output_bytes: 1,
                ..Default::default()
            },
        );
        model.send_last_completion_stream_event(LanguageModelCompletionEvent::UsageUpdate(
            warm_outcome().usage,
        ));
        model.send_last_completion_stream_text_chunk("too long");
        let outcome = task.await;
        assert!(outcome.error.as_deref().unwrap().contains("output limit"));
        assert_eq!(outcome.usage.cache_read_input_tokens, 90);
        model.end_last_completion_stream();
    }

    #[gpui::test(iterations = 10)]
    async fn stream_timeout_is_driven_by_gpui_clock(cx: &mut gpui::TestAppContext) {
        let config = Config {
            timeout: Duration::from_secs(1),
            ..Default::default()
        };
        let (model, task) = spawn_ping(cx, config);
        cx.executor().advance_clock(Duration::from_secs(2));
        let outcome = task.await;
        assert!(outcome.error.as_deref().unwrap().contains("timed out"));
        model.end_last_completion_stream();
    }

    #[gpui::test]
    async fn eof_without_usage_is_not_success(cx: &mut gpui::TestAppContext) {
        let (model, task) = spawn_ping(cx, Config::default());
        model.end_last_completion_stream();
        assert!(task.await.error.is_some());
    }

    #[gpui::test(iterations = 10)]
    async fn cancellation_aborts_stream_and_preserves_partial_usage(cx: &mut gpui::TestAppContext) {
        let model = Arc::new(language_model::fake_provider::FakeLanguageModel::default());
        let (abort, cancellation) = future::AbortHandle::new_pair();
        let task = cx.update(|cx| {
            let model = model.clone();
            cx.spawn(async move |cx| {
                let mut outcome = PingOutcome::default();
                let result = future::Abortable::new(
                    run_ping_stream(
                        model,
                        LanguageModelRequest::default(),
                        &Config::default(),
                        &mut outcome,
                        cx,
                    ),
                    cancellation,
                )
                .await;
                assert!(result.is_err());
                outcome
            })
        });
        cx.run_until_parked();
        model.send_last_completion_stream_event(LanguageModelCompletionEvent::UsageUpdate(
            warm_outcome().usage,
        ));
        cx.run_until_parked();
        abort.abort();
        let outcome = task.await;
        assert_eq!(outcome.usage.cache_read_input_tokens, 90);
        model.end_last_completion_stream();
    }

    #[gpui::test]
    async fn request_guard_cleans_capture_and_does_not_remove_new_generation(
        cx: &mut gpui::TestAppContext,
    ) {
        cx.update(settings::init);
        let fs = fs::FakeFs::new(cx.background_executor.clone());
        fs.insert_tree(std::path::Path::new("/cache-test"), serde_json::json!({}))
            .await;
        let project = project::Project::test(fs, [std::path::Path::new("/cache-test")], cx).await;
        let project_context = cx.new(|_| prompt_store::ProjectContext::default());
        let server_store = project.read_with(cx, |project, _| project.context_server_store());
        let server_registry =
            cx.new(|cx| crate::tools::ContextServerRegistry::new(server_store, cx));
        let model: Arc<dyn LanguageModel> =
            Arc::new(language_model::fake_provider::FakeLanguageModel::default());
        let thread = cx.new(|cx| {
            crate::Thread::new(
                project,
                project_context,
                server_registry,
                crate::Templates::new(),
                Some(model.clone()),
                cx,
            )
        });
        let registry = Arc::new(Mutex::new(Registry::default()));
        let (abort, cancellation) = future::AbortHandle::new_pair();
        let capture = Capture {
            owner: thread.downgrade(),
            scope: "test".into(),
            model,
            request: LanguageModelRequest::default(),
            state: CaptureState::new(Instant::now()),
            config: Config::default(),
            abort: Some(abort),
        };
        let generation = capture.state.generation;
        registry.lock().insert("test".into(), capture);
        let guard = RequestCapture {
            registry: Arc::downgrade(&registry),
            affinity: "test".into(),
            generation,
            confirmed: false,
        };
        let newer_generation = Uuid::new_v4();
        registry
            .lock()
            .captures
            .first_mut()
            .unwrap()
            .1
            .state
            .generation = newer_generation;
        drop(guard);
        assert_eq!(registry.lock().captures.len(), 1);
        drop(RequestCapture {
            registry: Arc::downgrade(&registry),
            affinity: "test".into(),
            generation: newer_generation,
            confirmed: false,
        });
        assert!(registry.lock().captures.is_empty());
        assert!(
            future::Abortable::new(future::pending::<()>(), cancellation)
                .await
                .is_err()
        );
        drop(thread);
        cx.run_until_parked();
    }

    #[gpui::test]
    async fn unexpected_tool_stop_pauses_warming(cx: &mut gpui::TestAppContext) {
        let (model, task) = spawn_ping(cx, Config::default());
        model.send_last_completion_stream_event(LanguageModelCompletionEvent::Stop(
            StopReason::ToolUse,
        ));
        assert!(task.await.error.is_some());
        model.end_last_completion_stream();
    }
    #[gpui::test]
    async fn test_cache_keepalive_deduplicates_by_prompt_cache_affinity(
        cx: &mut gpui::TestAppContext,
    ) {
        cx.update(settings::init);
        let fs = fs::FakeFs::new(cx.background_executor.clone());
        fs.insert_tree(std::path::Path::new("/cache-test"), serde_json::json!({}))
            .await;
        let project = project::Project::test(fs, [std::path::Path::new("/cache-test")], cx).await;
        let project_context = cx.new(|_| prompt_store::ProjectContext::default());
        let server_store = project.read_with(cx, |project, _| project.context_server_store());
        let server_registry =
            cx.new(|cx| crate::tools::ContextServerRegistry::new(server_store, cx));
        let model: Arc<dyn LanguageModel> =
            Arc::new(language_model::fake_provider::FakeLanguageModel::default());
        let root_thread = cx.new(|cx| {
            crate::Thread::new(
                project.clone(),
                project_context.clone(),
                server_registry.clone(),
                crate::Templates::new(),
                Some(model.clone()),
                cx,
            )
        });
        let fork_thread = cx.new(|cx| {
            crate::Thread::new(
                project,
                project_context,
                server_registry,
                crate::Templates::new(),
                Some(model.clone()),
                cx,
            )
        });

        let registry = Arc::new(Mutex::new(Registry::default()));
        let affinity = "shared-root-affinity".to_string();

        let (abort1, _) = future::AbortHandle::new_pair();
        let capture1 = Capture {
            owner: root_thread.downgrade(),
            scope: "test".into(),
            model: model.clone(),
            request: LanguageModelRequest {
                thread_id: Some("root-id".into()),
                prompt_cache_key: Some(affinity.clone()),
                ..Default::default()
            },
            state: CaptureState::new(Instant::now()),
            config: Config::default(),
            abort: Some(abort1),
        };
        registry.lock().insert(affinity.clone(), capture1);
        assert_eq!(registry.lock().captures.len(), 1);

        let (abort2, _) = future::AbortHandle::new_pair();
        let capture2 = Capture {
            owner: fork_thread.downgrade(),
            scope: "test".into(),
            model: model.clone(),
            request: LanguageModelRequest {
                thread_id: Some("fork-id".into()),
                prompt_cache_key: Some(affinity.clone()),
                ..Default::default()
            },
            state: CaptureState::new(Instant::now()),
            config: Config::default(),
            abort: Some(abort2),
        };
        registry.lock().insert(affinity, capture2);
        // Shared affinity replaces previous capture so root and fork do not create competing captures.
        assert_eq!(registry.lock().captures.len(), 1);
        assert_eq!(
            registry.lock().captures[0].1.request.thread_id.as_deref(),
            Some("fork-id")
        );
    }

    #[gpui::test]
    async fn test_subagent_activity_does_not_discard_root_cache_capture(
        cx: &mut gpui::TestAppContext,
    ) {
        cx.update(|cx| {
            let mut settings_store = settings::SettingsStore::test(cx);
            settings_store.update_user_settings(cx, |settings| {
                settings.agent.get_or_insert_default().cache_keepalive = Some(true);
            });
            cx.set_global(settings_store);
            language_model::init(cx);
        });
        let fs = fs::FakeFs::new(cx.background_executor.clone());
        fs.insert_tree(
            std::path::Path::new("/subagent-cache-test"),
            serde_json::json!({}),
        )
        .await;
        let project =
            project::Project::test(fs, [std::path::Path::new("/subagent-cache-test")], cx).await;
        let project_context = cx.new(|_| prompt_store::ProjectContext::default());
        let server_store = project.read_with(cx, |project, _| project.context_server_store());
        let server_registry =
            cx.new(|cx| crate::tools::ContextServerRegistry::new(server_store, cx));
        let model: Arc<dyn LanguageModel> =
            Arc::new(language_model::fake_provider::FakeLanguageModel::default());
        let root_thread = cx.new(|cx| {
            crate::Thread::new(
                project.clone(),
                project_context.clone(),
                server_registry.clone(),
                crate::Templates::new(),
                Some(model.clone()),
                cx,
            )
        });
        let subagent_thread = cx.update(|cx| {
            cx.new(|cx| {
                crate::Thread::new_subagent(
                    &root_thread,
                    Some(crate::SubagentRole::CodingWorker),
                    cx,
                )
            })
        });

        let root_id = root_thread.read_with(cx, |thread, _| thread.id().to_string());
        cx.update(|cx| {
            init(cx);
            toggle_thread(root_id.clone(), root_thread.downgrade(), cx);
        });

        let registry = cx.read(|cx| cx.global::<GlobalKeepAlive>().registry.clone());
        let affinity = root_thread.read_with(cx, |thread, _| thread.prompt_cache_affinity());

        let (abort, _) = future::AbortHandle::new_pair();
        let capture = Capture {
            owner: root_thread.downgrade(),
            scope: "test".into(),
            model: model.clone(),
            request: LanguageModelRequest {
                thread_id: Some(root_thread.read_with(cx, |thread, _| thread.id().to_string())),
                prompt_cache_key: Some(affinity.clone()),
                ..Default::default()
            },
            state: CaptureState::new(Instant::now()),
            config: Config::default(),
            abort: Some(abort),
        };
        registry.lock().insert(affinity.clone(), capture);
        assert_eq!(registry.lock().captures.len(), 1);

        // Now subagent runs turn activity with the root's affinity
        let subagent_request = LanguageModelRequest {
            thread_id: Some("subagent-id".into()),
            prompt_cache_key: Some(affinity.clone()),
            ..Default::default()
        };
        let guard = cx.update(|cx| {
            record_turn_activity(cx, subagent_thread.downgrade(), model, &subagent_request)
        });
        // Subagent request is excluded from capturing, but must NOT wipe out root's capture!
        assert!(guard.is_none());
        assert_eq!(registry.lock().captures.len(), 1);
        assert_eq!(
            registry.lock().captures[0].1.request.thread_id.as_deref(),
            Some(root_id.as_str())
        );
    }

    #[gpui::test]
    async fn test_disabling_owner_drops_capture_while_fork_retains_affinity(
        cx: &mut gpui::TestAppContext,
    ) {
        cx.update(|cx| {
            let mut settings_store = settings::SettingsStore::test(cx);
            settings_store.update_user_settings(cx, |settings| {
                settings.agent.get_or_insert_default().cache_keepalive = Some(true);
            });
            cx.set_global(settings_store);
            language_model::init(cx);
        });
        let fs = fs::FakeFs::new(cx.background_executor.clone());
        fs.insert_tree(
            std::path::Path::new("/fork-cache-test"),
            serde_json::json!({}),
        )
        .await;
        let project =
            project::Project::test(fs, [std::path::Path::new("/fork-cache-test")], cx).await;
        let project_context = cx.new(|_| prompt_store::ProjectContext::default());
        let server_store = project.read_with(cx, |project, _| project.context_server_store());
        let server_registry =
            cx.new(|cx| crate::tools::ContextServerRegistry::new(server_store, cx));
        let model: Arc<dyn LanguageModel> =
            Arc::new(language_model::fake_provider::FakeLanguageModel::default());
        let root_thread = cx.new(|cx| {
            crate::Thread::new(
                project.clone(),
                project_context.clone(),
                server_registry.clone(),
                crate::Templates::new(),
                Some(model.clone()),
                cx,
            )
        });
        let fork_thread = cx.new(|cx| {
            crate::Thread::new(
                project,
                project_context,
                server_registry,
                crate::Templates::new(),
                Some(model.clone()),
                cx,
            )
        });

        let root_id = root_thread.read_with(cx, |t, _| t.id().to_string());
        let fork_id = fork_thread.read_with(cx, |t, _| t.id().to_string());

        // Make fork_thread inherit root_thread's prompt cache affinity
        cx.update(|cx| {
            fork_thread.update(cx, |fork, _| {
                fork.fork_origin = Some(crate::ForkOrigin {
                    session_id: acp::SessionId::new("source"),
                    root_session_id: Some(acp::SessionId::new(root_id.clone())),
                    checkpoint_user_message_id: acp_thread::ClientUserMessageId::new(),
                    inherited_message_count: 0,
                    inherited_user_message_ids: collections::HashSet::default(),
                });
            });
            init(cx);
            toggle_thread(root_id.clone(), root_thread.downgrade(), cx);
            toggle_thread(fork_id.clone(), fork_thread.downgrade(), cx);
        });

        let registry = cx.read(|cx| cx.global::<GlobalKeepAlive>().registry.clone());
        let affinity = root_thread.read_with(cx, |t, _| t.prompt_cache_affinity());

        // Root captures
        let capture = Capture {
            owner: root_thread.downgrade(),
            scope: "test".into(),
            model: model.clone(),
            request: LanguageModelRequest {
                thread_id: Some(root_id.clone()),
                prompt_cache_key: Some(affinity.clone()),
                ..Default::default()
            },
            state: CaptureState::new(Instant::now()),
            config: Config::default(),
            abort: None,
        };
        registry.lock().insert(affinity.clone(), capture);
        assert_eq!(registry.lock().captures.len(), 1);

        // Now toggle off root_thread
        cx.update(|cx| {
            toggle_thread(root_id.clone(), root_thread.downgrade(), cx);
        });

        // The capture owned by root must be dropped, even though fork is still enabled for that affinity
        assert!(registry.lock().captures.is_empty());
        // Fork is still enabled
        assert!(cx.read(|cx| enabled_for_thread(&fork_id, cx)));
    }

    #[gpui::test]
    async fn test_max_captures_counts_affinity_families_not_threads(cx: &mut gpui::TestAppContext) {
        cx.update(|cx| {
            let mut settings_store = settings::SettingsStore::test(cx);
            settings_store.update_user_settings(cx, |settings| {
                settings.agent.get_or_insert_default().cache_keepalive = Some(true);
            });
            cx.set_global(settings_store);
            language_model::init(cx);
        });
        let fs = fs::FakeFs::new(cx.background_executor.clone());
        fs.insert_tree(
            std::path::Path::new("/family-cache-test"),
            serde_json::json!({}),
        )
        .await;
        let project =
            project::Project::test(fs, [std::path::Path::new("/family-cache-test")], cx).await;
        let project_context = cx.new(|_| prompt_store::ProjectContext::default());
        let server_store = project.read_with(cx, |project, _| project.context_server_store());
        let server_registry =
            cx.new(|cx| crate::tools::ContextServerRegistry::new(server_store, cx));
        let model: Arc<dyn LanguageModel> =
            Arc::new(language_model::fake_provider::FakeLanguageModel::default());

        let mut threads = Vec::new();
        for _ in 0..5 {
            let t = cx.new(|cx| {
                crate::Thread::new(
                    project.clone(),
                    project_context.clone(),
                    server_registry.clone(),
                    crate::Templates::new(),
                    Some(model.clone()),
                    cx,
                )
            });
            threads.push(t);
        }

        // Thread 0, 1, 2 share affinity
        let root_id = threads[0].read_with(cx, |t, _| t.id().to_string());
        for i in 1..=2 {
            let root_id = root_id.clone();
            cx.update(|cx| {
                threads[i].update(cx, |fork, _| {
                    fork.fork_origin = Some(crate::ForkOrigin {
                        session_id: acp::SessionId::new("source"),
                        root_session_id: Some(acp::SessionId::new(root_id)),
                        checkpoint_user_message_id: acp_thread::ClientUserMessageId::new(),
                        inherited_message_count: 0,
                        inherited_user_message_ids: collections::HashSet::default(),
                    });
                });
            });
        }

        cx.update(init);
        for i in 0..=2 {
            let id = threads[i].read_with(cx, |t, _| t.id().to_string());
            cx.update(|cx| toggle_thread(id, threads[i].downgrade(), cx));
        }

        let registry = cx.read(|cx| cx.global::<GlobalKeepAlive>().registry.clone());
        assert_eq!(registry.lock().enabled_threads.len(), 3);
    }
}
