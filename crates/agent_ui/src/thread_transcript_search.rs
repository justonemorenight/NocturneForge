use std::time::{Duration, Instant};

use acp_thread::{
    AcpThread, AcpThreadEvent, AgentThreadEntry, AssistantMessageChunk,
    ContentBlock as RenderedContentBlock, ThreadStatus,
};
use agent::ThreadStore;
use agent_client_protocol::schema::v1 as acp;
use anyhow::Context as _;
use chrono::{DateTime, Utc};
use collections::HashMap;
use db::sqlez::{bindable::Column, connection::Connection, statement::Statement};
use gpui::{
    App, AppContext as _, Context, Entity, EntityId, Global, Subscription, Task, TaskExt,
    WeakEntity, Window,
};
use language_model::Role;
use unicode_normalization::UnicodeNormalization as _;

use crate::{
    conversation_view::{ConversationView, ThreadView},
    thread_metadata_store::{ThreadId, ThreadMetadataDb, ThreadMetadataStore},
};

const EXTRACTOR_VERSION: i64 = 1;
const DOCUMENT_CHUNK_BYTES: usize = 16 * 1024;
const DOCUMENT_CHUNK_OVERLAP_BYTES: usize = 256;
const BACKFILL_BATCH_SIZE: usize = 12;
const BACKFILL_BATCH_PAUSE: Duration = Duration::from_millis(12);
const BACKFILL_DEBOUNCE: Duration = Duration::from_millis(250);
const NAVIGATION_TTL: Duration = Duration::from_secs(60);
const SEARCH_RESULT_LIMIT: i64 = 100;

struct GlobalThreadTranscriptSearchStore(Entity<ThreadTranscriptSearchStore>);

impl Global for GlobalThreadTranscriptSearchStore {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum ThreadSearchRole {
    User,
    Assistant,
}

impl ThreadSearchRole {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::User => "You",
            Self::Assistant => "Assistant",
        }
    }

    fn as_db_str(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }

    fn from_db_str(value: &str) -> Option<Self> {
        match value {
            "user" => Some(Self::User),
            "assistant" => Some(Self::Assistant),
            _ => None,
        }
    }
}

impl ThreadSearchRole {
    fn from_native_role(role: Role) -> Option<Self> {
        match role {
            Role::User => Some(Self::User),
            Role::Assistant => Some(Self::Assistant),
            Role::System => None,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ThreadSearchDocument {
    pub(crate) role: ThreadSearchRole,
    pub(crate) body: String,
}

#[derive(Clone, Debug)]
pub(crate) struct ThreadSearchSnippet {
    pub(crate) role: ThreadSearchRole,
    pub(crate) text: String,
    pub(crate) highlight_positions: Vec<usize>,
    pub(crate) fingerprint: String,
}

#[derive(Clone, Debug)]
pub(crate) struct ThreadSearchHit {
    pub(crate) thread_id: ThreadId,
    pub(crate) snippet: ThreadSearchSnippet,
}

#[derive(Clone, Debug)]
pub(crate) struct ThreadSearchNavigation {
    pub(crate) query: String,
    pub(crate) role: ThreadSearchRole,
    pub(crate) fingerprint: String,
}

struct PendingNavigation {
    navigation: ThreadSearchNavigation,
    expires_at: Instant,
}

struct LiveThreadState {
    view: WeakEntity<ThreadView>,
    segment_id: String,
    _subscription: Subscription,
}

#[derive(Clone)]
struct BackfillCandidate {
    thread_id: ThreadId,
    session_id: acp::SessionId,
    updated_at: DateTime<Utc>,
}

#[derive(Clone)]
struct IndexState {
    source_updated_at: DateTime<Utc>,
    extractor_version: i64,
    history_complete: bool,
}

struct SearchRow {
    thread_id: ThreadId,
    role: String,
    body: String,
}

impl Column for SearchRow {
    fn column(statement: &mut Statement, start_index: i32) -> anyhow::Result<(Self, i32)> {
        let (thread_id, next) = Column::column(statement, start_index)?;
        let (role, next) = Column::column(statement, next)?;
        let (body, next) = Column::column(statement, next)?;
        Ok((
            Self {
                thread_id,
                role,
                body,
            },
            next,
        ))
    }
}

pub(crate) fn init(cx: &mut App) {
    let store = cx.new(ThreadTranscriptSearchStore::new);
    cx.set_global(GlobalThreadTranscriptSearchStore(store.clone()));
    store.update(cx, |store, cx| {
        store.initialize_fts(cx);
        store.schedule_backfill(cx);
    });
}

pub(crate) struct ThreadTranscriptSearchStore {
    db: ThreadMetadataDb,
    fts_available: bool,
    is_indexing: bool,
    pending_navigation: HashMap<ThreadId, PendingNavigation>,
    live_threads: HashMap<EntityId, LiveThreadState>,
    backfill_generation: usize,
    _fts_initialization_task: Option<Task<()>>,
    _backfill_task: Option<Task<()>>,
    _metadata_subscription: Subscription,
}

impl ThreadTranscriptSearchStore {
    pub(crate) fn global(cx: &App) -> Entity<Self> {
        cx.global::<GlobalThreadTranscriptSearchStore>().0.clone()
    }

    fn try_global(cx: &App) -> Option<Entity<Self>> {
        cx.try_global::<GlobalThreadTranscriptSearchStore>()
            .map(|global| global.0.clone())
    }

    pub(crate) fn is_indexing(&self) -> bool {
        self.is_indexing
    }

    fn new(cx: &mut Context<Self>) -> Self {
        let weak_store = cx.weak_entity();
        cx.observe_new::<ThreadView>(move |view, window, cx| {
            if view.parent_session_id.is_some() {
                return;
            }

            let view_entity = cx.entity();
            let view_id = view_entity.entity_id();
            let thread = view.thread.clone();
            cx.on_release({
                let weak_store = weak_store.clone();
                move |_view, cx| {
                    weak_store
                        .update(cx, |store, _cx| {
                            store.live_threads.remove(&view_id);
                        })
                        .ok();
                }
            })
            .detach();

            weak_store
                .update(cx, |store, cx| {
                    store.track_live_thread(view_entity.clone(), thread, cx);
                })
                .ok();

            let deferred_store = weak_store.clone();
            cx.defer(move |cx| {
                deferred_store
                    .update(cx, |store, cx| store.index_live_thread(view_id, cx))
                    .ok();
            });

            if let Some(window) = window {
                let weak_view = view_entity.downgrade();
                window.defer(cx, move |window, cx| {
                    let Some(view) = weak_view.upgrade() else {
                        return;
                    };
                    Self::activate_pending_for_thread_view(&view, window, cx);
                });
            }
        })
        .detach();

        let metadata_store = ThreadMetadataStore::global(cx);
        let metadata_subscription = cx.observe(&metadata_store, |this, _, cx| {
            this.schedule_backfill(cx);
        });

        Self {
            db: ThreadMetadataDb::global(cx),
            fts_available: false,
            is_indexing: false,
            pending_navigation: HashMap::default(),
            live_threads: HashMap::default(),
            backfill_generation: 0,
            _fts_initialization_task: None,
            _backfill_task: None,
            _metadata_subscription: metadata_subscription,
        }
    }

    fn initialize_fts(&mut self, cx: &mut Context<Self>) {
        let db = self.db.clone();
        let task = cx.background_spawn(async move { db.ensure_transcript_fts().await });
        self._fts_initialization_task = Some(cx.spawn(async move |this, cx| {
            let available = match task.await {
                Ok(()) => true,
                Err(error) => {
                    log::warn!("conversation transcript search is unavailable: {error:#}");
                    false
                }
            };
            this.update(cx, |this, cx| {
                this.fts_available = available;
                cx.notify();
            })
            .ok();
        }));
    }

    fn track_live_thread(
        &mut self,
        view: Entity<ThreadView>,
        thread: Entity<AcpThread>,
        cx: &mut Context<Self>,
    ) {
        let view_id = view.entity_id();
        if self.live_threads.contains_key(&view_id) {
            return;
        }

        let segment_id = format!("live:{}", uuid::Uuid::new_v4().hyphenated());
        let weak_view = view.downgrade();
        let subscription = cx.subscribe(
            &thread,
            move |this: &mut Self, thread, event: &AcpThreadEvent, cx| {
                let should_index =
                    matches!(event, AcpThreadEvent::Stopped(_) | AcpThreadEvent::Error)
                        || matches!(event, AcpThreadEvent::EntriesRemoved(_))
                            && thread.read(cx).status() == ThreadStatus::Idle;
                if should_index {
                    this.index_live_thread(view_id, cx);
                }
            },
        );
        self.live_threads.insert(
            view_id,
            LiveThreadState {
                view: weak_view,
                segment_id,
                _subscription: subscription,
            },
        );
    }

    fn index_live_thread(&mut self, view_id: EntityId, cx: &mut Context<Self>) {
        let Some(state) = self.live_threads.get_mut(&view_id) else {
            return;
        };
        let Some(view) = state.view.upgrade() else {
            return;
        };

        let (thread_id, resumed_without_history, documents) = view.read_with(cx, |view, cx| {
            (
                view.root_thread_id,
                view.resumed_without_history,
                extract_acp_transcript(&view.thread.read(cx), cx),
            )
        });
        if documents.is_empty() {
            return;
        }

        let db = self.db.clone();
        let source_updated_at = Utc::now();
        let segment_id = state.segment_id.clone();
        cx.spawn(async move |this, cx| {
            let result = if resumed_without_history {
                db.replace_transcript_segment(thread_id, segment_id, source_updated_at, documents)
                    .await
            } else {
                db.replace_full_transcript(thread_id, source_updated_at, documents)
                    .await
            };
            result.context("index open conversation transcript")?;
            this.update(cx, |_this, cx| cx.notify())?;
            anyhow::Ok(())
        })
        .detach_and_log_err(cx);
    }

    fn schedule_backfill(&mut self, cx: &mut Context<Self>) {
        self.backfill_generation = self.backfill_generation.wrapping_add(1);
        let generation = self.backfill_generation;
        let metadata_reload = ThreadMetadataStore::global(cx).read(cx).reload_task();
        let thread_reload = ThreadStore::global(cx).read(cx).reload_task();
        let db = self.db.clone();

        self._backfill_task = Some(cx.spawn(async move |this, cx| {
            cx.background_executor().timer(BACKFILL_DEBOUNCE).await;
            metadata_reload.await;
            thread_reload.await;

            let candidates = cx.update(|cx| collect_backfill_candidates(cx));
            let states_task = cx.background_spawn({
                let db = db.clone();
                async move { db.list_transcript_index_states() }
            });
            let states = match states_task.await {
                Ok(states) => states,
                Err(error) => {
                    log::warn!("failed to read conversation transcript index state: {error:#}");
                    this.update(cx, |this, cx| {
                        if this.backfill_generation == generation && this.is_indexing {
                            this.is_indexing = false;
                            cx.notify();
                        }
                    })
                    .ok();
                    return;
                }
            };
            let candidates = candidates
                .into_iter()
                .filter(|candidate| {
                    states.get(&candidate.thread_id).is_none_or(|state| {
                        state.extractor_version != EXTRACTOR_VERSION
                            || !state.history_complete
                            || state.source_updated_at < candidate.updated_at
                    })
                })
                .collect::<Vec<_>>();

            if candidates.is_empty() {
                this.update(cx, |this, cx| {
                    if this.backfill_generation == generation && this.is_indexing {
                        this.is_indexing = false;
                        cx.notify();
                    }
                })
                .ok();
                return;
            }

            let should_continue = this
                .update(cx, |this, cx| {
                    if this.backfill_generation != generation {
                        return false;
                    }
                    this.is_indexing = true;
                    cx.notify();
                    true
                })
                .unwrap_or(false);
            if !should_continue {
                return;
            }

            for batch in candidates.chunks(BACKFILL_BATCH_SIZE) {
                for candidate in batch {
                    let load_task = cx.update(|cx| {
                        ThreadStore::global(cx).update(cx, |store, cx| {
                            store.load_thread(candidate.session_id.clone(), cx)
                        })
                    });
                    let thread = match load_task.await {
                        Ok(Some(thread)) => thread,
                        Ok(None) => continue,
                        Err(error) => {
                            log::warn!(
                                "failed to load conversation {} for search indexing: {error:#}",
                                candidate.session_id.0
                            );
                            continue;
                        }
                    };
                    let documents = cx
                        .background_spawn(async move {
                            thread
                                .searchable_transcript()
                                .into_iter()
                                .filter_map(|(role, body)| {
                                    Some(ThreadSearchDocument {
                                        role: ThreadSearchRole::from_native_role(role)?,
                                        body,
                                    })
                                })
                                .collect()
                        })
                        .await;
                    if let Err(error) = db
                        .replace_full_transcript(
                            candidate.thread_id,
                            candidate.updated_at,
                            documents,
                        )
                        .await
                    {
                        log::warn!(
                            "failed to backfill conversation {} search index: {error:#}",
                            candidate.session_id.0
                        );
                    }
                }

                let should_continue = this
                    .update(cx, |this, cx| {
                        if this.backfill_generation != generation {
                            return false;
                        }
                        // Archive views observe this and refresh after each
                        // bounded batch without invalidating their selection.
                        cx.notify();
                        true
                    })
                    .unwrap_or(false);
                if !should_continue {
                    return;
                }
                cx.background_executor().timer(BACKFILL_BATCH_PAUSE).await;
            }

            this.update(cx, |this, cx| {
                if this.backfill_generation == generation {
                    this.is_indexing = false;
                    cx.notify();
                }
            })
            .ok();
        }));
    }

    pub(crate) fn search(
        &self,
        query: String,
        archived_only: bool,
        cx: &App,
    ) -> Task<anyhow::Result<Vec<ThreadSearchHit>>> {
        if !self.fts_available || query.chars().count() < 3 {
            return Task::ready(Ok(Vec::new()));
        }

        let db = self.db.clone();
        cx.background_spawn(async move { db.search_transcripts(query, archived_only) })
    }

    pub(crate) fn queue_navigation(
        thread_id: ThreadId,
        navigation: ThreadSearchNavigation,
        cx: &mut App,
    ) {
        let Some(store) = Self::try_global(cx) else {
            return;
        };
        store.update(cx, |this, _cx| {
            let now = Instant::now();
            this.pending_navigation
                .retain(|_, pending| pending.expires_at > now);
            this.pending_navigation.insert(
                thread_id,
                PendingNavigation {
                    navigation,
                    expires_at: now + NAVIGATION_TTL,
                },
            );
        });
    }

    fn take_navigation(thread_id: ThreadId, cx: &mut App) -> Option<ThreadSearchNavigation> {
        Self::try_global(cx)?.update(cx, |this, _cx| {
            this.pending_navigation
                .remove(&thread_id)
                .filter(|pending| pending.expires_at > Instant::now())
                .map(|pending| pending.navigation)
        })
    }

    pub(crate) fn activate_pending_for_conversation(
        conversation: &Entity<ConversationView>,
        window: &mut Window,
        cx: &mut App,
    ) {
        let Some(thread_view) = conversation.read(cx).root_thread_view() else {
            return;
        };
        Self::activate_pending_for_thread_view(&thread_view, window, cx);
    }

    fn activate_pending_for_thread_view(
        thread_view: &Entity<ThreadView>,
        window: &mut Window,
        cx: &mut App,
    ) {
        let thread_id = thread_view.read(cx).root_thread_id;
        let Some(navigation) = Self::take_navigation(thread_id, cx) else {
            return;
        };
        thread_view.update(cx, |view, cx| {
            view.open_thread_search(navigation, window, cx);
        });
    }
}

fn collect_backfill_candidates(cx: &App) -> Vec<BackfillCandidate> {
    let thread_store = ThreadStore::global(cx);
    let native_threads = thread_store
        .read(cx)
        .entries()
        .map(|thread| (thread.id.clone(), thread.updated_at))
        .collect::<HashMap<_, _>>();

    ThreadMetadataStore::global(cx)
        .read(cx)
        .entries()
        .filter(|metadata| metadata.agent_id.as_ref() == agent::ZED_AGENT_ID.as_ref())
        .filter_map(|metadata| {
            let session_id = metadata.session_id.clone()?;
            let updated_at = *native_threads.get(&session_id)?;
            Some(BackfillCandidate {
                thread_id: metadata.thread_id,
                session_id,
                updated_at,
            })
        })
        .collect()
}

fn extract_acp_transcript(thread: &AcpThread, cx: &App) -> Vec<ThreadSearchDocument> {
    let mut transcript = Vec::new();
    for entry in thread.entries() {
        match entry {
            AgentThreadEntry::UserMessage(message) => {
                for chunk in &message.chunks {
                    if let acp::ContentBlock::Text(text) = chunk
                        && !text.text.is_empty()
                    {
                        transcript.push(ThreadSearchDocument {
                            role: ThreadSearchRole::User,
                            body: text.text.clone(),
                        });
                    }
                }
            }
            AgentThreadEntry::AssistantMessage(message) if !message.is_subagent_output => {
                for chunk in &message.chunks {
                    if let AssistantMessageChunk::Message { block, .. } = chunk
                        && matches!(block, RenderedContentBlock::Markdown { .. })
                        && let Some(body) = block.text_content(cx).filter(|body| !body.is_empty())
                    {
                        transcript.push(ThreadSearchDocument {
                            role: ThreadSearchRole::Assistant,
                            body: body.to_string(),
                        });
                    }
                }
            }
            AgentThreadEntry::AssistantMessage(_)
            | AgentThreadEntry::ToolCall(_)
            | AgentThreadEntry::Elicitation(_)
            | AgentThreadEntry::CompletedPlan(_)
            | AgentThreadEntry::ContextCompaction(_) => {}
        }
    }
    transcript
}

pub(crate) fn normalize_search_text(text: &str) -> String {
    text.nfc()
        .flat_map(char::to_lowercase)
        .filter(|character| *character != '\0')
        .collect()
}

pub(crate) fn search_context_fingerprint(text: &str, range: std::ops::Range<usize>) -> String {
    let before_start = previous_char_boundary(text, range.start.saturating_sub(512));
    let after_end = next_char_boundary(text, range.end.saturating_add(512).min(text.len()));
    let before = collapse_whitespace(&normalize_search_text(&text[before_start..range.start]));
    let matched = collapse_whitespace(&normalize_search_text(&text[range.clone()]));
    let after = collapse_whitespace(&normalize_search_text(&text[range.end..after_end]));
    format!(
        "{}\u{1f}{}\u{1f}{}",
        take_last_chars(&before, 48),
        matched,
        take_first_chars(&after, 48)
    )
}

fn collapse_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn take_first_chars(text: &str, count: usize) -> String {
    text.chars().take(count).collect()
}

fn take_last_chars(text: &str, count: usize) -> String {
    let mut characters = text.chars().rev().take(count).collect::<Vec<_>>();
    characters.reverse();
    characters.into_iter().collect()
}

fn build_search_snippet(body: &str, query: &str, role: ThreadSearchRole) -> ThreadSearchSnippet {
    let body = body.nfc().collect::<String>();
    let collapsed = collapse_whitespace(&body);
    let query = collapse_whitespace(&query.nfc().collect::<String>());
    let matched_range = literal_case_insensitive_range(&collapsed, &query);

    let (window_start, window_end) = matched_range
        .as_ref()
        .map(|range| {
            (
                char_window_start(&collapsed, range.start, 72),
                char_window_end(&collapsed, range.end, 112),
            )
        })
        .unwrap_or((0, collapsed.len().min(220)));
    let window_start = previous_char_boundary(&collapsed, window_start);
    let window_end = next_char_boundary(&collapsed, window_end.min(collapsed.len()));
    let prefix = (window_start > 0).then_some("…");
    let suffix = (window_end < collapsed.len()).then_some("…");

    let mut text = String::new();
    if let Some(prefix) = prefix {
        text.push_str(prefix);
    }
    let content_offset = text.len();
    text.push_str(&collapsed[window_start..window_end]);
    if let Some(suffix) = suffix {
        text.push_str(suffix);
    }

    let highlight_positions = matched_range
        .as_ref()
        .filter(|range| range.start >= window_start && range.end <= window_end)
        .map(|range| {
            let start = content_offset + range.start - window_start;
            let end = content_offset + range.end - window_start;
            text[start..end]
                .char_indices()
                .map(|(offset, _)| start + offset)
                .collect()
        })
        .unwrap_or_default();
    let fingerprint = matched_range
        .map(|range| search_context_fingerprint(&collapsed, range))
        .unwrap_or_else(|| collapse_whitespace(&normalize_search_text(&collapsed)));

    ThreadSearchSnippet {
        role,
        text,
        highlight_positions,
        fingerprint,
    }
}

fn literal_case_insensitive_range(text: &str, query: &str) -> Option<std::ops::Range<usize>> {
    if query.is_empty() {
        return None;
    }

    let normalized_query = normalize_search_text(query);
    let mut normalized = String::new();
    let mut mapping = Vec::new();
    for (source_start, character) in text.char_indices() {
        let source_end = source_start + character.len_utf8();
        for lowered in character.to_lowercase() {
            let normalized_start = normalized.len();
            normalized.push(lowered);
            mapping.push((normalized_start, normalized.len(), source_start, source_end));
        }
    }
    let normalized_start = normalized.find(&normalized_query)?;
    let normalized_end = normalized_start + normalized_query.len();
    let source_start = mapping
        .iter()
        .find(|(start, end, _, _)| *start <= normalized_start && normalized_start < *end)?
        .2;
    let source_end = mapping
        .iter()
        .rev()
        .find(|(start, _, _, _)| *start < normalized_end)?
        .3;
    Some(source_start..source_end)
}

fn char_window_start(text: &str, byte_index: usize, character_count: usize) -> usize {
    text[..previous_char_boundary(text, byte_index)]
        .char_indices()
        .rev()
        .nth(character_count)
        .map(|(index, _)| index)
        .unwrap_or(0)
}

fn char_window_end(text: &str, byte_index: usize, character_count: usize) -> usize {
    let byte_index = next_char_boundary(text, byte_index.min(text.len()));
    text[byte_index..]
        .char_indices()
        .nth(character_count)
        .map(|(index, _)| byte_index + index)
        .unwrap_or(text.len())
}

fn previous_char_boundary(text: &str, mut index: usize) -> usize {
    index = index.min(text.len());
    while !text.is_char_boundary(index) {
        index = index.saturating_sub(1);
    }
    index
}

fn next_char_boundary(text: &str, mut index: usize) -> usize {
    index = index.min(text.len());
    while index < text.len() && !text.is_char_boundary(index) {
        index += 1;
    }
    index
}

fn document_chunks(body: &str) -> Vec<String> {
    let body = body.nfc().collect::<String>();
    if body.len() <= DOCUMENT_CHUNK_BYTES {
        return vec![body];
    }

    let mut chunks = Vec::new();
    let mut start = 0;
    while start < body.len() {
        let mut end = (start + DOCUMENT_CHUNK_BYTES).min(body.len());
        end = previous_char_boundary(&body, end);
        if end <= start {
            end = next_char_boundary(&body, (start + 1).min(body.len()));
        }
        chunks.push(body[start..end].to_string());
        if end == body.len() {
            break;
        }
        let next = previous_char_boundary(&body, end.saturating_sub(DOCUMENT_CHUNK_OVERLAP_BYTES));
        start = if next > start { next } else { end };
    }
    chunks
}

impl ThreadMetadataDb {
    async fn ensure_transcript_fts(&self) -> anyhow::Result<()> {
        self.write(|connection| {
            connection.with_savepoint("initialize_thread_transcript_fts", || {
                let mut existed = connection
                    .select_row::<i64>(
                        "SELECT COUNT(*) FROM sqlite_master \
                         WHERE type = 'table' AND name = 'thread_transcript_fts'",
                    )?()?
                    .unwrap_or_default()
                    > 0;
                let incompatible_detail_mode = connection
                    .select_row::<i64>(
                        "SELECT COUNT(*) FROM sqlite_master \
                         WHERE type = 'table' AND name = 'thread_transcript_fts' \
                         AND sql NOT LIKE '%detail=''none''%'",
                    )?()?
                    .unwrap_or_default()
                    > 0;
                if incompatible_detail_mode {
                    connection.exec(
                        "DROP TRIGGER IF EXISTS thread_transcript_documents_ai; \
                         DROP TRIGGER IF EXISTS thread_transcript_documents_ad; \
                         DROP TRIGGER IF EXISTS thread_transcript_documents_au; \
                         DROP TABLE thread_transcript_fts",
                    )?()?;
                    existed = false;
                }

                connection.exec(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS thread_transcript_fts USING fts5(\
                        normalized_body, \
                        content='thread_transcript_documents', \
                        content_rowid='id', \
                        tokenize='trigram', \
                        detail='none'\
                    )",
                )?()?;
                connection.exec(
                    "CREATE TRIGGER IF NOT EXISTS thread_transcript_documents_ai \
                     AFTER INSERT ON thread_transcript_documents BEGIN \
                       INSERT INTO thread_transcript_fts(rowid, normalized_body) \
                       VALUES (new.id, new.normalized_body); \
                     END",
                )?()?;
                connection.exec(
                    "CREATE TRIGGER IF NOT EXISTS thread_transcript_documents_ad \
                     AFTER DELETE ON thread_transcript_documents BEGIN \
                       INSERT INTO thread_transcript_fts(thread_transcript_fts, rowid, normalized_body) \
                       VALUES ('delete', old.id, old.normalized_body); \
                     END",
                )?()?;
                connection.exec(
                    "CREATE TRIGGER IF NOT EXISTS thread_transcript_documents_au \
                     AFTER UPDATE ON thread_transcript_documents BEGIN \
                       INSERT INTO thread_transcript_fts(thread_transcript_fts, rowid, normalized_body) \
                       VALUES ('delete', old.id, old.normalized_body); \
                       INSERT INTO thread_transcript_fts(rowid, normalized_body) \
                       VALUES (new.id, new.normalized_body); \
                     END",
                )?()?;

                if !existed {
                    connection.exec(
                        "INSERT INTO thread_transcript_fts(thread_transcript_fts) VALUES ('rebuild')",
                    )?()?;
                }
                Ok(())
            })
        })
        .await
    }

    fn list_transcript_index_states(&self) -> anyhow::Result<HashMap<ThreadId, IndexState>> {
        let rows = self.select::<(ThreadId, String, i64, bool)>(
            "SELECT thread_id, source_updated_at, extractor_version, history_complete \
             FROM thread_transcript_index_state",
        )?()?;
        rows.into_iter()
            .map(
                |(thread_id, updated_at, extractor_version, history_complete)| {
                    let source_updated_at =
                        DateTime::parse_from_rfc3339(&updated_at)?.with_timezone(&Utc);
                    Ok((
                        thread_id,
                        IndexState {
                            source_updated_at,
                            extractor_version,
                            history_complete,
                        },
                    ))
                },
            )
            .collect()
    }

    async fn replace_full_transcript(
        &self,
        thread_id: ThreadId,
        source_updated_at: DateTime<Utc>,
        documents: Vec<ThreadSearchDocument>,
    ) -> anyhow::Result<()> {
        self.replace_transcript(
            thread_id,
            "full".to_string(),
            source_updated_at,
            documents,
            true,
        )
        .await
    }

    async fn replace_transcript_segment(
        &self,
        thread_id: ThreadId,
        segment_id: String,
        source_updated_at: DateTime<Utc>,
        documents: Vec<ThreadSearchDocument>,
    ) -> anyhow::Result<()> {
        self.replace_transcript(thread_id, segment_id, source_updated_at, documents, false)
            .await
    }

    async fn replace_transcript(
        &self,
        thread_id: ThreadId,
        segment_id: String,
        source_updated_at: DateTime<Utc>,
        documents: Vec<ThreadSearchDocument>,
        full_snapshot: bool,
    ) -> anyhow::Result<()> {
        self.write(move |connection| {
            connection.with_savepoint("replace_thread_transcript", || {
                if transcript_write_is_stale(
                    connection,
                    thread_id,
                    source_updated_at,
                    full_snapshot,
                )? {
                    return Ok(());
                }

                let extractor_version_mismatch = connection.select_row_bound::<ThreadId, i64>(
                    "SELECT extractor_version FROM thread_transcript_index_state \
                         WHERE thread_id = ?1",
                )?(thread_id)?
                .is_some_and(|version| version != EXTRACTOR_VERSION);
                if extractor_version_mismatch && !full_snapshot {
                    let mut delete_documents = Statement::prepare(
                        connection,
                        "DELETE FROM thread_transcript_documents WHERE thread_id = ?",
                    )?;
                    delete_documents.bind(&thread_id, 1)?;
                    delete_documents.exec()?;

                    let mut delete_state = Statement::prepare(
                        connection,
                        "DELETE FROM thread_transcript_index_state WHERE thread_id = ?",
                    )?;
                    delete_state.bind(&thread_id, 1)?;
                    delete_state.exec()?;
                }

                let base_ordinal = if full_snapshot {
                    let mut delete = Statement::prepare(
                        connection,
                        "DELETE FROM thread_transcript_documents WHERE thread_id = ?",
                    )?;
                    delete.bind(&thread_id, 1)?;
                    delete.exec()?;
                    0
                } else {
                    let existing_base = connection.select_row_bound::<(ThreadId, String), i64>(
                        "SELECT MIN(document_ordinal) \
                             FROM thread_transcript_documents \
                             WHERE thread_id = ?1 AND segment_id = ?2",
                    )?((thread_id, segment_id.clone()))?;
                    let base = match existing_base {
                        Some(base) => base,
                        None => connection.select_row_bound::<ThreadId, i64>(
                            "SELECT COALESCE(MAX(document_ordinal) + 1, 0) \
                                 FROM thread_transcript_documents WHERE thread_id = ?1",
                        )?(thread_id)?
                        .unwrap_or_default(),
                    };
                    let mut delete = Statement::prepare(
                        connection,
                        "DELETE FROM thread_transcript_documents \
                         WHERE thread_id = ?1 AND segment_id = ?2",
                    )?;
                    let next = delete.bind(&thread_id, 1)?;
                    delete.bind(&segment_id, next)?;
                    delete.exec()?;
                    base
                };

                let mut insert = connection
                    .exec_bound::<(ThreadId, String, i64, i64, String, String, String)>(
                        "INSERT INTO thread_transcript_documents(\
                        thread_id, segment_id, document_ordinal, chunk_ordinal, \
                        role, body, normalized_body\
                     ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                    )?;
                for (document_offset, document) in documents.into_iter().enumerate() {
                    for (chunk_ordinal, body) in
                        document_chunks(&document.body).into_iter().enumerate()
                    {
                        let normalized_body = normalize_search_text(&body);
                        insert((
                            thread_id,
                            segment_id.clone(),
                            base_ordinal + document_offset as i64,
                            chunk_ordinal as i64,
                            document.role.as_db_str().to_string(),
                            body,
                            normalized_body,
                        ))?;
                    }
                }

                let existing_complete = connection.select_row_bound::<ThreadId, bool>(
                    "SELECT history_complete FROM thread_transcript_index_state \
                         WHERE thread_id = ?1",
                )?(thread_id)?
                .unwrap_or(false);
                let history_complete = full_snapshot || existing_complete;
                connection.exec_bound::<(ThreadId, String, i64, bool)>(
                    "INSERT INTO thread_transcript_index_state(\
                        thread_id, source_updated_at, extractor_version, history_complete\
                     ) VALUES (?1, ?2, ?3, ?4) \
                     ON CONFLICT(thread_id) DO UPDATE SET \
                        source_updated_at = excluded.source_updated_at, \
                        extractor_version = excluded.extractor_version, \
                        history_complete = excluded.history_complete",
                )?((
                    thread_id,
                    source_updated_at.to_rfc3339(),
                    EXTRACTOR_VERSION,
                    history_complete,
                ))?;
                Ok(())
            })
        })
        .await
    }

    fn search_transcripts(
        &self,
        query: String,
        archived_only: bool,
    ) -> anyhow::Result<Vec<ThreadSearchHit>> {
        let normalized_query = normalize_search_text(&query);
        if normalized_query.chars().count() < 3 {
            return Ok(Vec::new());
        }
        let fts_query = trigram_prefilter_query(&normalized_query);
        let rows = self.select_bound::<(String, String, bool, i64), SearchRow>(
            "WITH ranked AS (\
                SELECT d.thread_id, d.role, d.body, s.updated_at, \
                       ROW_NUMBER() OVER (\
                           PARTITION BY d.thread_id \
                           ORDER BY d.document_ordinal DESC, d.chunk_ordinal DESC, d.id DESC\
                       ) AS match_rank \
                FROM thread_transcript_fts f \
                JOIN thread_transcript_documents d ON d.id = f.rowid \
                JOIN sidebar_threads s ON s.thread_id = d.thread_id \
                WHERE thread_transcript_fts MATCH ?1 \
                  AND instr(d.normalized_body, ?2) > 0 \
                  AND (?3 = 0 OR s.archived = 1)\
             ) \
             SELECT thread_id, role, body \
             FROM ranked \
             WHERE match_rank = 1 \
             ORDER BY updated_at DESC \
             LIMIT ?4",
        )?((
            fts_query,
            normalized_query,
            archived_only,
            SEARCH_RESULT_LIMIT,
        ))?;

        Ok(rows
            .into_iter()
            .filter_map(|row| {
                let role = ThreadSearchRole::from_db_str(&row.role)?;
                Some(ThreadSearchHit {
                    thread_id: row.thread_id,
                    snippet: build_search_snippet(&row.body, &query, role),
                })
            })
            .collect())
    }
}

fn trigram_prefilter_query(query: &str) -> String {
    const MAX_PREFILTER_TRIGRAMS: usize = 12;

    let characters = query.chars().collect::<Vec<_>>();
    let trigram_count = characters.len().saturating_sub(2);
    let indices = if trigram_count <= MAX_PREFILTER_TRIGRAMS {
        (0..trigram_count).collect::<Vec<_>>()
    } else {
        (0..MAX_PREFILTER_TRIGRAMS)
            .map(|index| index * (trigram_count - 1) / (MAX_PREFILTER_TRIGRAMS - 1))
            .collect()
    };

    indices
        .into_iter()
        .map(|index| {
            let trigram = characters[index..index + 3].iter().collect::<String>();
            format!("\"{}\"", trigram.replace('"', "\"\""))
        })
        .collect::<Vec<_>>()
        .join(" AND ")
}

fn transcript_write_is_stale(
    connection: &Connection,
    thread_id: ThreadId,
    source_updated_at: DateTime<Utc>,
    full_snapshot: bool,
) -> anyhow::Result<bool> {
    let existing = connection.select_row_bound::<ThreadId, (String, i64, bool)>(
        "SELECT source_updated_at, extractor_version, history_complete \
             FROM thread_transcript_index_state WHERE thread_id = ?1",
    )?(thread_id)?;
    let Some((existing_updated_at, existing_version, history_complete)) = existing else {
        return Ok(false);
    };
    if existing_version != EXTRACTOR_VERSION {
        return Ok(false);
    }
    if full_snapshot && !history_complete {
        return Ok(false);
    }
    let existing_updated_at =
        DateTime::parse_from_rfc3339(&existing_updated_at)?.with_timezone(&Utc);
    Ok(existing_updated_at > source_updated_at)
}
