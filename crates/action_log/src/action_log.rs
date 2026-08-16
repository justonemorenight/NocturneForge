use agent_settings::AgentSettings;
use anyhow::{Context as _, Result};
use buffer_diff::{BufferDiff, BufferDiffSnapshot};
use clock;
use collections::{BTreeMap, HashMap};
use fs::MTime;
use futures::{FutureExt, channel::oneshot};
use gpui::{
    App, AppContext, AsyncApp, Context, Entity, EntityId, SharedString, Subscription, Task,
    WeakEntity,
};
use language::{Anchor, Buffer, BufferEvent, Point, ToOffset, ToPoint};
use project::{Project, ProjectItem, lsp_store::OpenLspBufferHandle};
use settings::{Settings as _, SettingsStore};
use std::{
    cell::Cell,
    cmp,
    collections::VecDeque,
    ops::Range,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex, OnceLock, Weak as SyncWeak,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};
use text::{Edit, Patch, Rope};
use util::{RangeExt, ResultExt as _};

/// Stores undo information for a single buffer's rejected edits
#[derive(Clone)]
pub struct PerBufferUndo {
    pub buffer: WeakEntity<Buffer>,
    pub edits_to_restore: Vec<(Range<Anchor>, String)>,
    pub status: UndoBufferStatus,
    pub transaction_id: Option<clock::Lamport>,
}

/// Tracks the buffer status for undo purposes
#[derive(Clone, Debug)]
pub enum UndoBufferStatus {
    Modified,
    /// Buffer was created by the agent.
    /// - `had_existing_content: true` - Agent overwrote an existing file. On reject, the
    ///   original content was restored. Undo is supported: we restore the agent's content.
    /// - `had_existing_content: false` - Agent created a new file that didn't exist before.
    ///   On reject, the file was deleted. Undo is NOT currently supported (would require
    ///   recreating the file). Future TODO.
    Created {
        had_existing_content: bool,
    },
}

/// Stores undo information for the most recent reject operation
#[derive(Clone)]
pub struct LastRejectUndo {
    /// Per-buffer undo information
    pub buffers: Vec<PerBufferUndo>,
}

/// Tracks actions performed by tools in a thread
pub struct ActionLog {
    /// Buffers that we want to notify the model about when they change.
    tracked_buffers: BTreeMap<Entity<Buffer>, TrackedBuffer>,
    /// The project this action log is associated with
    project: Entity<Project>,
    /// An action log to forward all public methods to
    /// Useful in cases like subagents, where we want to track individual diffs for this subagent,
    /// but also want to associate the reads/writes with a parent review experience
    linked_action_log: Option<Entity<ActionLog>>,
    /// Stores undo information for the most recent reject operation
    last_reject_undo: Option<LastRejectUndo>,
    review_decisions: VecDeque<ReviewDecision>,
    /// Tracks the last time files were read by the agent, to detect external modifications
    file_read_times: HashMap<PathBuf, MTime>,
    last_reported_large_diff_state: Cell<Option<bool>>,
    lsp_lease_counters: Arc<AgentLspLeaseCounters>,
    lsp_lease_limiter: Arc<AgentLspLeaseLimiter>,
    settings_subscription: Option<Subscription>,
}

impl ActionLog {
    /// Creates a new, empty action log associated with the given project.
    pub fn new(project: Entity<Project>) -> Self {
        let lsp_lease_limiter = project_lsp_lease_limiter(project.entity_id());
        Self {
            tracked_buffers: BTreeMap::default(),
            project,
            linked_action_log: None,
            last_reject_undo: None,
            review_decisions: VecDeque::new(),
            file_read_times: HashMap::default(),
            last_reported_large_diff_state: Cell::new(None),
            lsp_lease_counters: Arc::default(),
            lsp_lease_limiter,
            settings_subscription: None,
        }
    }

    pub fn with_linked_action_log(mut self, linked_action_log: Entity<ActionLog>) -> Self {
        self.linked_action_log = Some(linked_action_log);
        self
    }

    pub fn project(&self) -> &Entity<Project> {
        &self.project
    }

    pub fn file_read_time(&self, path: &Path) -> Option<MTime> {
        self.file_read_times.get(path).copied()
    }

    fn update_file_read_time(&mut self, buffer: &Entity<Buffer>, cx: &App) {
        let buffer = buffer.read(cx);
        if let Some(file) = buffer.file() {
            if let Some(local_file) = file.as_local() {
                if let Some(mtime) = file.disk_state().mtime() {
                    let abs_path = local_file.abs_path(cx);
                    self.file_read_times.insert(abs_path, mtime);
                }
            }
        }
    }

    fn remove_file_read_time(&mut self, buffer: &Entity<Buffer>, cx: &App) {
        let buffer = buffer.read(cx);
        if let Some(file) = buffer.file() {
            if let Some(local_file) = file.as_local() {
                let abs_path = local_file.abs_path(cx);
                self.file_read_times.remove(&abs_path);
            }
        }
    }

    fn track_buffer_internal(
        &mut self,
        buffer: Entity<Buffer>,
        is_created: bool,
        cx: &mut Context<Self>,
    ) -> &mut TrackedBuffer {
        if self.settings_subscription.is_none() {
            self.settings_subscription =
                Some(cx.observe_global::<SettingsStore>(|this, cx| this.sync_lsp_lease_mode(cx)));
        }
        let status = if is_created {
            if let Some(tracked) = self.tracked_buffers.remove(&buffer) {
                match tracked.status {
                    TrackedBufferStatus::Created {
                        existing_file_content,
                    } => TrackedBufferStatus::Created {
                        existing_file_content,
                    },
                    TrackedBufferStatus::Modified | TrackedBufferStatus::Deleted => {
                        TrackedBufferStatus::Created {
                            existing_file_content: Some(tracked.diff_base),
                        }
                    }
                }
            } else if buffer
                .read(cx)
                .file()
                .is_some_and(|file| file.disk_state().exists())
            {
                TrackedBufferStatus::Created {
                    existing_file_content: Some(buffer.read(cx).as_rope().clone()),
                }
            } else {
                TrackedBufferStatus::Created {
                    existing_file_content: None,
                }
            }
        } else {
            TrackedBufferStatus::Modified
        };

        let experimental_lsp_leases = AgentSettings::get_global(cx).experimental_lsp_leases;
        let needs_legacy_lease = !experimental_lsp_leases
            && self
                .tracked_buffers
                .get(&buffer)
                .is_some_and(|tracked_buffer| tracked_buffer.lsp_lease.is_none());
        if needs_legacy_lease {
            let handle = self.project.update(cx, |project, cx| {
                project.register_buffer_with_language_servers(&buffer, cx)
            });
            if let Some(tracked_buffer) = self.tracked_buffers.get_mut(&buffer) {
                tracked_buffer.lsp_lease = Some(AgentLspLease::new(
                    handle,
                    AgentLspLeaseOwnership::Legacy,
                    self.lsp_lease_counters.clone(),
                    None,
                ));
            }
        } else if experimental_lsp_leases
            && let Some(tracked_buffer) = self.tracked_buffers.get_mut(&buffer)
            && tracked_buffer.active_edit_sessions == 0
            && tracked_buffer
                .lsp_lease
                .as_ref()
                .is_some_and(|lease| lease.ownership == AgentLspLeaseOwnership::Legacy)
        {
            tracked_buffer.lsp_lease.take();
        }

        let tracked_buffer = self
            .tracked_buffers
            .entry(buffer.clone())
            .or_insert_with(|| {
                let lsp_lease = if experimental_lsp_leases {
                    None
                } else {
                    let open_lsp_handle = self.project.update(cx, |project, cx| {
                        project.register_buffer_with_language_servers(&buffer, cx)
                    });
                    Some(AgentLspLease::new(
                        open_lsp_handle,
                        AgentLspLeaseOwnership::Legacy,
                        self.lsp_lease_counters.clone(),
                        None,
                    ))
                };

                let text_snapshot = buffer.read(cx).text_snapshot();
                let language = buffer.read(cx).language().cloned();
                let language_registry = buffer.read(cx).language_registry();
                let diff =
                    cx.new(|cx| BufferDiff::new(&text_snapshot, language, language_registry, cx));
                let (diff_update_tx, diff_update_rx) = watch::channel(());
                let diff_base;
                let unreviewed_edits;
                if is_created {
                    diff_base = Rope::default();
                    unreviewed_edits = Patch::new(vec![Edit {
                        old: 0..1,
                        new: 0..text_snapshot.max_point().row + 1,
                    }])
                } else {
                    diff_base = buffer.read(cx).as_rope().clone();
                    unreviewed_edits = Patch::default();
                }
                TrackedBuffer {
                    buffer: buffer.clone(),
                    diff_base,
                    unreviewed_edits,
                    snapshot: text_snapshot,
                    status,
                    version: buffer.read(cx).version(),
                    diff,
                    diff_update: diff_update_tx,
                    pending_diff_update: None,
                    in_flight_diff_update: None,
                    diff_generation: 0,
                    review_state_changed_before_recompute: false,
                    diff_complexity: DiffComplexity::default(),
                    lsp_lease,
                    lsp_lease_generation: 0,
                    active_edit_sessions: 0,
                    lsp_release_task: None,
                    _maintain_diff: cx.spawn({
                        let buffer = buffer.clone();
                        async move |this, cx| {
                            Self::maintain_diff(this, buffer, diff_update_rx, cx)
                                .await
                                .ok();
                        }
                    }),
                    _subscription: cx.subscribe(&buffer, Self::handle_buffer_event),
                }
            });
        tracked_buffer.version = buffer.read(cx).version();
        tracked_buffer
    }

    fn sync_lsp_lease_mode(&mut self, cx: &mut Context<Self>) {
        if AgentSettings::get_global(cx).experimental_lsp_leases {
            for tracked in self.tracked_buffers.values_mut() {
                if tracked.active_edit_sessions == 0
                    && tracked
                        .lsp_lease
                        .as_ref()
                        .is_some_and(|lease| lease.ownership == AgentLspLeaseOwnership::Legacy)
                {
                    tracked.lsp_lease.take();
                }
            }
        } else {
            let buffers = self
                .tracked_buffers
                .iter()
                .filter_map(|(buffer, tracked)| {
                    (!tracked
                        .lsp_lease
                        .as_ref()
                        .is_some_and(|lease| lease.ownership == AgentLspLeaseOwnership::Legacy))
                    .then(|| buffer.clone())
                })
                .collect::<Vec<_>>();
            for buffer in buffers {
                self.ensure_legacy_lsp_lease(&buffer, cx);
            }
        }
    }

    fn handle_buffer_event(
        &mut self,
        buffer: Entity<Buffer>,
        event: &BufferEvent,
        cx: &mut Context<Self>,
    ) {
        match event {
            BufferEvent::Edited { .. } => {
                let Some(tracked_buffer) = self.tracked_buffers.get_mut(&buffer) else {
                    return;
                };
                let buffer_version = buffer.read(cx).version();
                if !buffer_version.changed_since(&tracked_buffer.version) {
                    return;
                }
                self.handle_buffer_edited(buffer, cx);
            }
            BufferEvent::FileHandleChanged => {
                self.handle_buffer_file_changed(buffer, cx);
            }
            BufferEvent::TransactionUndone { transaction_id } => {
                self.restore_review_decision(&buffer, *transaction_id, true, cx);
            }
            BufferEvent::TransactionRedone { transaction_id } => {
                self.restore_review_decision(&buffer, *transaction_id, false, cx);
            }
            _ => {}
        };
    }

    fn record_review_decision(
        &mut self,
        buffer: &Entity<Buffer>,
        transaction_id: clock::Lamport,
        before: ReviewBufferState,
        after: ReviewBufferState,
    ) {
        const MAX_REVIEW_DECISIONS: usize = 256;

        self.review_decisions.push_back(ReviewDecision {
            buffer: buffer.downgrade(),
            transaction_id,
            before,
            after,
        });
        while self.review_decisions.len() > MAX_REVIEW_DECISIONS {
            self.review_decisions.pop_front();
        }
    }

    fn restore_review_decision(
        &mut self,
        buffer: &Entity<Buffer>,
        transaction_id: clock::Lamport,
        undo: bool,
        cx: &mut Context<Self>,
    ) {
        let Some(decision) = self.review_decisions.iter().rev().find(|decision| {
            decision.transaction_id == transaction_id
                && decision
                    .buffer
                    .upgrade()
                    .is_some_and(|candidate| candidate == *buffer)
        }) else {
            return;
        };
        let state = if undo {
            decision.before.clone()
        } else {
            decision.after.clone()
        };
        let Some(tracked_buffer) = self.tracked_buffers.get_mut(buffer) else {
            return;
        };

        state.restore(tracked_buffer, buffer.read(cx).text_snapshot());
        tracked_buffer.schedule_diff_update(ChangeAuthor::Agent, cx);
        cx.notify();
    }

    fn handle_buffer_edited(&mut self, buffer: Entity<Buffer>, cx: &mut Context<Self>) {
        let Some(tracked_buffer) = self.tracked_buffers.get_mut(&buffer) else {
            return;
        };
        tracked_buffer.schedule_diff_update(ChangeAuthor::User, cx);
    }

    fn handle_buffer_file_changed(&mut self, buffer: Entity<Buffer>, cx: &mut Context<Self>) {
        let Some(tracked_buffer) = self.tracked_buffers.get_mut(&buffer) else {
            return;
        };

        match tracked_buffer.status {
            TrackedBufferStatus::Created { .. } | TrackedBufferStatus::Modified => {
                if buffer
                    .read(cx)
                    .file()
                    .is_some_and(|file| file.disk_state().is_deleted())
                {
                    // If the buffer had been edited by a tool, but it got
                    // deleted externally, we want to stop tracking it.
                    self.tracked_buffers.remove(&buffer);
                }
                cx.notify();
            }
            TrackedBufferStatus::Deleted => {
                if buffer
                    .read(cx)
                    .file()
                    .is_some_and(|file| !file.disk_state().is_deleted())
                {
                    // If the buffer had been deleted by a tool, but it got
                    // resurrected externally, we want to clear the edits we
                    // were tracking and reset the buffer's state.
                    self.tracked_buffers.remove(&buffer);
                    self.track_buffer_internal(buffer, false, cx);
                }
                cx.notify();
            }
        }
    }

    async fn maintain_diff(
        this: WeakEntity<Self>,
        buffer: Entity<Buffer>,
        mut buffer_updates: watch::Receiver<()>,
        cx: &mut AsyncApp,
    ) -> Result<()> {
        let git_diff = this
            .update(cx, |this, cx| {
                this.project.update(cx, |project, cx| {
                    project.open_uncommitted_diff(buffer.clone(), cx)
                })
            })?
            .await
            .ok();
        let (mut git_diff_updates_tx, mut git_diff_updates_rx) = watch::channel(());
        let _diff_subscription = if let Some(git_diff) = git_diff.as_ref() {
            cx.update(|cx| {
                Some(cx.subscribe(git_diff, move |_, event, _cx| {
                    if matches!(event, buffer_diff::BufferDiffEvent::BaseTextChanged) {
                        git_diff_updates_tx.send(()).ok();
                    }
                }))
            })
        } else {
            None
        };

        let mut retry_git_diff = false;
        loop {
            if retry_git_diff {
                let has_pending_buffer_update = this.read_with(cx, |this, _cx| {
                    this.tracked_buffers
                        .get(&buffer)
                        .is_some_and(|tracked_buffer| tracked_buffer.pending_diff_update.is_some())
                })?;
                if !has_pending_buffer_update {
                    if let Some(git_diff) = git_diff.as_ref() {
                        retry_git_diff =
                            !Self::keep_committed_edits(&this, &buffer, git_diff, cx).await?;
                    } else {
                        retry_git_diff = false;
                    }
                    continue;
                }
            }

            futures::select_biased! {
                buffer_update = buffer_updates.changed().fuse() => {
                    if buffer_update.is_err() {
                        break;
                    }
                    let buffer_update = this.update(cx, |this, _cx| {
                        let tracked_buffer = this.tracked_buffers.get_mut(&buffer)?;
                        let pending = tracked_buffer.pending_diff_update.take()?;
                        tracked_buffer.in_flight_diff_update = Some((pending.generation, pending.author));
                        Some(pending)
                    })?;
                    if let Some(pending) = buffer_update {
                        let result = Self::track_edits(
                            &this,
                            &buffer,
                            pending.generation,
                            pending.author,
                            pending.snapshot,
                            cx,
                        )
                        .await;
                        this.update(cx, |this, _cx| {
                            if let Some(tracked_buffer) = this.tracked_buffers.get_mut(&buffer)
                                && tracked_buffer
                                    .in_flight_diff_update
                                    .is_some_and(|(generation, _)| generation == pending.generation)
                            {
                                tracked_buffer.in_flight_diff_update = None;
                            }
                        })?;
                        result?;
                    }
                }
                _ = git_diff_updates_rx.changed().fuse() => {
                    if let Some(git_diff) = git_diff.as_ref() {
                        if !Self::keep_committed_edits(&this, &buffer, git_diff, cx).await? {
                            // A buffer update superseded this reconciliation while it was
                            // running. Retry it after that buffer snapshot is canonical.
                            retry_git_diff = true;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    async fn track_edits(
        this: &WeakEntity<ActionLog>,
        buffer: &Entity<Buffer>,
        generation: u64,
        author: ChangeAuthor,
        buffer_snapshot: text::BufferSnapshot,
        cx: &mut AsyncApp,
    ) -> Result<()> {
        let rebase = this.update(cx, |this, cx| {
            let tracked_buffer = this
                .tracked_buffers
                .get_mut(buffer)
                .context("buffer not tracked")?;

            let rebase = cx.background_spawn({
                let mut base_text = tracked_buffer.diff_base.clone();
                let old_snapshot = tracked_buffer.snapshot.clone();
                let new_snapshot = buffer_snapshot.clone();
                let unreviewed_edits = tracked_buffer.unreviewed_edits.clone();
                let edits = diff_snapshots(&old_snapshot, &new_snapshot);
                async move {
                    if let ChangeAuthor::User = author {
                        apply_non_conflicting_edits(
                            &unreviewed_edits,
                            edits,
                            &mut base_text,
                            new_snapshot.as_rope(),
                        );
                    }

                    (Arc::from(base_text.to_string().as_str()), base_text)
                }
            });

            anyhow::Ok(rebase)
        })??;
        let (new_base_text, new_diff_base) = rebase.await;

        if !Self::is_diff_generation_current(this, buffer, generation, cx)? {
            return Ok(());
        }

        Self::update_diff(
            this,
            buffer,
            generation,
            buffer_snapshot,
            new_base_text,
            new_diff_base,
            cx,
        )
        .await
        .map(|_| ())
    }

    async fn keep_committed_edits(
        this: &WeakEntity<ActionLog>,
        buffer: &Entity<Buffer>,
        git_diff: &Entity<BufferDiff>,
        cx: &mut AsyncApp,
    ) -> Result<bool> {
        let (generation, buffer_snapshot) = this.read_with(cx, |this, _cx| {
            let tracked_buffer = this
                .tracked_buffers
                .get(buffer)
                .context("buffer not tracked")?;
            anyhow::Ok((
                tracked_buffer.diff_generation,
                tracked_buffer.snapshot.clone(),
            ))
        })??;
        let (new_base_text, new_diff_base) = this
            .read_with(cx, |this, cx| {
                let tracked_buffer = this
                    .tracked_buffers
                    .get(buffer)
                    .context("buffer not tracked")?;
                let old_unreviewed_edits = tracked_buffer.unreviewed_edits.clone();
                let agent_diff_base = tracked_buffer.diff_base.clone();
                let git_diff_base = git_diff.read(cx).base_text(cx).as_rope().clone();
                let buffer_text = tracked_buffer.snapshot.as_rope().clone();
                anyhow::Ok(cx.background_spawn(async move {
                    if buffer_text.len() == git_diff_base.len()
                        && buffer_text.chars_at(0).eq(git_diff_base.chars_at(0))
                    {
                        return (Arc::<str>::from(git_diff_base.to_string()), git_diff_base);
                    }
                    let mut old_unreviewed_edits = old_unreviewed_edits.into_iter().peekable();
                    let committed_edits = language::line_diff(
                        &agent_diff_base.to_string(),
                        &git_diff_base.to_string(),
                    )
                    .into_iter()
                    .map(|(old, new)| Edit { old, new });

                    let mut new_agent_diff_base = agent_diff_base.clone();
                    let mut row_delta = 0i32;
                    for committed in committed_edits {
                        while let Some(unreviewed) = old_unreviewed_edits.peek() {
                            // If the committed edit matches the unreviewed
                            // edit, assume the user wants to keep it.
                            if committed.old == unreviewed.old {
                                let unreviewed_new =
                                    buffer_text.slice_rows(unreviewed.new.clone()).to_string();
                                let committed_new =
                                    git_diff_base.slice_rows(committed.new.clone()).to_string();
                                if unreviewed_new == committed_new {
                                    let old_byte_start =
                                        new_agent_diff_base.point_to_offset(Point::new(
                                            (unreviewed.old.start as i32 + row_delta) as u32,
                                            0,
                                        ));
                                    let old_byte_end =
                                        new_agent_diff_base.point_to_offset(cmp::min(
                                            Point::new(
                                                (unreviewed.old.end as i32 + row_delta) as u32,
                                                0,
                                            ),
                                            new_agent_diff_base.max_point(),
                                        ));
                                    new_agent_diff_base
                                        .replace(old_byte_start..old_byte_end, &unreviewed_new);
                                    row_delta +=
                                        unreviewed.new_len() as i32 - unreviewed.old_len() as i32;
                                }
                            } else if unreviewed.old.start >= committed.old.end {
                                break;
                            }

                            old_unreviewed_edits.next().unwrap();
                        }
                    }

                    (
                        Arc::from(new_agent_diff_base.to_string().as_str()),
                        new_agent_diff_base,
                    )
                }))
            })??
            .await;

        Self::update_diff(
            this,
            buffer,
            generation,
            buffer_snapshot,
            new_base_text,
            new_diff_base,
            cx,
        )
        .await
    }

    async fn update_diff(
        this: &WeakEntity<ActionLog>,
        buffer: &Entity<Buffer>,
        generation: u64,
        buffer_snapshot: text::BufferSnapshot,
        new_base_text: Arc<str>,
        new_diff_base: Rope,
        cx: &mut AsyncApp,
    ) -> Result<bool> {
        if !Self::is_diff_generation_current(this, buffer, generation, cx)? {
            return Ok(false);
        }
        let diff = this.read_with(cx, |this, _cx| {
            let tracked_buffer = this
                .tracked_buffers
                .get(buffer)
                .context("buffer not tracked")?;
            anyhow::Ok(tracked_buffer.diff.clone())
        })??;
        diff.update(cx, |diff, cx| {
            diff.set_base_text(Some(new_base_text), buffer_snapshot.clone(), cx)
        })
        .await;
        let diff_snapshot = diff.update(cx, |diff, cx| diff.snapshot(cx));

        let (unreviewed_edits, diff_complexity) = cx
            .background_spawn({
                let buffer_snapshot = buffer_snapshot.clone();
                let new_diff_base = new_diff_base.clone();
                async move {
                    let mut unreviewed_edits = Patch::default();
                    for hunk in diff_snapshot.hunks_intersecting_range(
                        Anchor::min_for_buffer(buffer_snapshot.remote_id())
                            ..Anchor::max_for_buffer(buffer_snapshot.remote_id()),
                        &buffer_snapshot,
                    ) {
                        let old_range = new_diff_base
                            .offset_to_point(hunk.diff_base_byte_range.start)
                            ..new_diff_base.offset_to_point(hunk.diff_base_byte_range.end);
                        let new_range = hunk.range.start..hunk.range.end;
                        unreviewed_edits.push(point_to_row_edit(
                            Edit {
                                old: old_range,
                                new: new_range,
                            },
                            &new_diff_base,
                            buffer_snapshot.as_rope(),
                        ));
                    }
                    let diff_complexity =
                        DiffComplexity::from_diff(&diff_snapshot, &buffer_snapshot, &new_diff_base);
                    (unreviewed_edits, diff_complexity)
                }
            })
            .await;
        this.update(cx, |this, cx| {
            let tracked_buffer = this
                .tracked_buffers
                .get_mut(buffer)
                .context("buffer not tracked")?;
            if tracked_buffer.diff_generation != generation {
                return Ok(false);
            }
            let should_notify = tracked_buffer.review_state_changed_before_recompute
                || tracked_buffer.unreviewed_edits != unreviewed_edits
                || tracked_buffer.diff_complexity != diff_complexity;
            tracked_buffer.diff_base = new_diff_base;
            tracked_buffer.snapshot = buffer_snapshot;
            tracked_buffer.unreviewed_edits = unreviewed_edits;
            tracked_buffer.diff_complexity = diff_complexity;
            tracked_buffer.review_state_changed_before_recompute = false;
            if should_notify {
                cx.notify();
            }
            anyhow::Ok(true)
        })?
    }

    fn is_diff_generation_current(
        this: &WeakEntity<ActionLog>,
        buffer: &Entity<Buffer>,
        generation: u64,
        cx: &AsyncApp,
    ) -> Result<bool> {
        this.read_with(cx, |this, _cx| {
            this.tracked_buffers
                .get(buffer)
                .map(|tracked_buffer| tracked_buffer.diff_generation == generation)
                .context("buffer not tracked")
        })?
    }

    /// Track a buffer as read by agent, so we can notify the model about user edits.
    pub fn buffer_read(&mut self, buffer: Entity<Buffer>, cx: &mut Context<Self>) {
        self.buffer_read_impl(buffer, true, cx);
    }

    fn buffer_read_impl(
        &mut self,
        buffer: Entity<Buffer>,
        record_file_read_time: bool,
        cx: &mut Context<Self>,
    ) {
        if let Some(linked_action_log) = &self.linked_action_log {
            // We don't want to share read times since the other agent hasn't read it necessarily
            linked_action_log.update(cx, |log, cx| {
                log.buffer_read_impl(buffer.clone(), false, cx);
            });
        }
        if record_file_read_time {
            self.update_file_read_time(&buffer, cx);
        }
        self.track_buffer_internal(buffer, false, cx);
    }

    /// Mark a buffer as created by agent, so we can refresh it in the context
    pub fn buffer_created(&mut self, buffer: Entity<Buffer>, cx: &mut Context<Self>) {
        self.buffer_created_impl(buffer, true, cx);
    }

    fn buffer_created_impl(
        &mut self,
        buffer: Entity<Buffer>,
        record_file_read_time: bool,
        cx: &mut Context<Self>,
    ) {
        if let Some(linked_action_log) = &self.linked_action_log {
            // We don't want to share read times since the other agent hasn't read it necessarily
            linked_action_log.update(cx, |log, cx| {
                log.buffer_created_impl(buffer.clone(), false, cx);
            });
        }
        if record_file_read_time {
            self.update_file_read_time(&buffer, cx);
        }
        self.track_buffer_internal(buffer, true, cx);
    }

    /// Mark a buffer as edited by agent, so we can refresh it in the context
    pub fn buffer_edited(&mut self, buffer: Entity<Buffer>, cx: &mut Context<Self>) {
        self.buffer_edited_impl(buffer, true, cx);
    }

    pub fn acquire_edit_lsp_lease(
        &mut self,
        buffer: Entity<Buffer>,
        cx: &mut Context<Self>,
    ) -> Task<Result<()>> {
        if let Some(linked_action_log) = &self.linked_action_log {
            return linked_action_log
                .update(cx, |log, cx| log.acquire_edit_lsp_lease(buffer.clone(), cx));
        }

        let experimental = AgentSettings::get_global(cx).experimental_lsp_leases;
        self.track_buffer_internal(buffer.clone(), false, cx);
        if !experimental {
            self.ensure_legacy_lsp_lease(&buffer, cx);
            if let Some(tracked) = self.tracked_buffers.get_mut(&buffer) {
                tracked.active_edit_sessions = tracked.active_edit_sessions.saturating_add(1);
            }
            return Task::ready(Ok(()));
        }

        let Some(tracked_buffer) = self.tracked_buffers.get_mut(&buffer) else {
            return Task::ready(Ok(()));
        };
        tracked_buffer.lsp_release_task.take();
        tracked_buffer.lsp_lease_generation = tracked_buffer.lsp_lease_generation.saturating_add(1);
        if tracked_buffer
            .lsp_lease
            .as_ref()
            .is_some_and(|lease| lease.ownership == AgentLspLeaseOwnership::Edit)
        {
            tracked_buffer.active_edit_sessions =
                tracked_buffer.active_edit_sessions.saturating_add(1);
            return Task::ready(Ok(()));
        }

        let limiter = self.lsp_lease_limiter.clone();
        let counters = self.lsp_lease_counters.clone();
        let project = self.project.clone();
        cx.spawn(async move |this, cx| {
            let permit = limiter
                .acquire(AgentLspLeaseOwnership::Edit, &counters)
                .await;
            let should_register = this.update(cx, |_this, cx| {
                AgentSettings::get_global(cx).experimental_lsp_leases
            })?;
            if !should_register {
                this.update(cx, |this, cx| {
                    if !AgentSettings::get_global(cx).experimental_lsp_leases {
                        this.ensure_legacy_lsp_lease(&buffer, cx);
                    }
                })?;
                return Ok(());
            }

            let handle = project.update(cx, |project, cx| {
                project.register_buffer_with_language_servers(&buffer, cx)
            });
            let mut lease = Some(AgentLspLease::new(
                handle,
                AgentLspLeaseOwnership::Edit,
                counters,
                Some(permit),
            ));
            this.update(cx, |this, cx| {
                let experimental = AgentSettings::get_global(cx).experimental_lsp_leases;
                let Some(tracked) = this.tracked_buffers.get_mut(&buffer) else {
                    return;
                };
                if experimental {
                    tracked.active_edit_sessions = tracked.active_edit_sessions.saturating_add(1);
                    if !tracked
                        .lsp_lease
                        .as_ref()
                        .is_some_and(|lease| lease.ownership == AgentLspLeaseOwnership::Edit)
                    {
                        tracked.lsp_lease = lease.take();
                    }
                    return;
                }
                if !experimental {
                    this.ensure_legacy_lsp_lease(&buffer, cx);
                }
            })?;
            Ok(())
        })
    }

    pub fn acquire_diagnostic_lsp_lease(
        &mut self,
        buffer: Entity<Buffer>,
        cx: &mut Context<Self>,
    ) -> Task<Result<Option<AgentLspLease>>> {
        if let Some(linked_action_log) = &self.linked_action_log {
            return linked_action_log
                .update(cx, |log, cx| log.acquire_diagnostic_lsp_lease(buffer, cx));
        }
        if !AgentSettings::get_global(cx).experimental_lsp_leases {
            return Task::ready(Ok(None));
        }

        let limiter = self.lsp_lease_limiter.clone();
        let counters = self.lsp_lease_counters.clone();
        let project = self.project.clone();
        cx.spawn(async move |_this, cx| {
            let permit = limiter
                .acquire(AgentLspLeaseOwnership::Diagnostic, &counters)
                .await;
            if !cx.update(|cx| AgentSettings::get_global(cx).experimental_lsp_leases) {
                return Ok(None);
            }
            let handle = project.update(cx, |project, cx| {
                project.register_buffer_with_language_servers(&buffer, cx)
            });
            Ok(Some(AgentLspLease::new(
                handle,
                AgentLspLeaseOwnership::Diagnostic,
                counters,
                Some(permit),
            )))
        })
    }

    fn ensure_legacy_lsp_lease(&mut self, buffer: &Entity<Buffer>, cx: &mut Context<Self>) {
        let needs_legacy = self.tracked_buffers.get_mut(buffer).is_some_and(|tracked| {
            tracked.lsp_release_task.take();
            !tracked
                .lsp_lease
                .as_ref()
                .is_some_and(|lease| lease.ownership == AgentLspLeaseOwnership::Legacy)
        });
        if !needs_legacy {
            return;
        }
        let handle = self.project.update(cx, |project, cx| {
            project.register_buffer_with_language_servers(buffer, cx)
        });
        if let Some(tracked) = self.tracked_buffers.get_mut(buffer) {
            tracked.lsp_lease = Some(AgentLspLease::new(
                handle,
                AgentLspLeaseOwnership::Legacy,
                self.lsp_lease_counters.clone(),
                None,
            ));
        }
    }

    fn buffer_edited_impl(
        &mut self,
        buffer: Entity<Buffer>,
        record_file_read_time: bool,
        cx: &mut Context<Self>,
    ) {
        if let Some(linked_action_log) = &self.linked_action_log {
            // We don't want to share read times since the other agent hasn't read it necessarily
            linked_action_log.update(cx, |log, cx| {
                log.buffer_edited_impl(buffer.clone(), false, cx);
            });
        }
        if record_file_read_time {
            self.update_file_read_time(&buffer, cx);
        }
        let new_version = buffer.read(cx).version();
        let tracked_buffer = self.track_buffer_internal(buffer.clone(), false, cx);
        if let TrackedBufferStatus::Deleted = tracked_buffer.status {
            tracked_buffer.status = TrackedBufferStatus::Modified;
        }

        tracked_buffer.version = new_version;
        tracked_buffer.schedule_diff_update(ChangeAuthor::Agent, cx);
    }

    pub fn finish_edit_lsp_lease(&mut self, buffer: &Entity<Buffer>, cx: &mut Context<Self>) {
        if let Some(linked_action_log) = &self.linked_action_log {
            linked_action_log.update(cx, |log, cx| log.finish_edit_lsp_lease(buffer, cx));
            return;
        }
        let experimental = AgentSettings::get_global(cx).experimental_lsp_leases;
        let Some(tracked) = self.tracked_buffers.get_mut(buffer) else {
            return;
        };
        if tracked.active_edit_sessions > 0 {
            tracked.active_edit_sessions -= 1;
        }
        if tracked.active_edit_sessions > 0 {
            return;
        }
        if !experimental {
            self.ensure_legacy_lsp_lease(buffer, cx);
            return;
        }
        if tracked
            .lsp_lease
            .as_ref()
            .is_some_and(|lease| lease.ownership == AgentLspLeaseOwnership::Legacy)
        {
            tracked.lsp_lease.take();
            return;
        }
        if !tracked
            .lsp_lease
            .as_ref()
            .is_some_and(|lease| lease.ownership == AgentLspLeaseOwnership::Edit)
        {
            return;
        }

        tracked.lsp_lease_generation = tracked.lsp_lease_generation.saturating_add(1);
        let generation = tracked.lsp_lease_generation;
        let buffer = buffer.clone();
        let timer = cx
            .background_executor()
            .timer(AGENT_EDIT_LSP_LEASE_IDLE_TTL);
        tracked.lsp_release_task = Some(cx.spawn(async move |this, cx| {
            timer.await;
            this.update(cx, |this, cx| {
                if !AgentSettings::get_global(cx).experimental_lsp_leases {
                    this.ensure_legacy_lsp_lease(&buffer, cx);
                    return;
                }
                let Some(tracked) = this.tracked_buffers.get_mut(&buffer) else {
                    return;
                };
                if tracked.lsp_lease_generation == generation
                    && tracked
                        .lsp_lease
                        .as_ref()
                        .is_some_and(|lease| lease.ownership == AgentLspLeaseOwnership::Edit)
                {
                    tracked.lsp_lease.take();
                    tracked.lsp_release_task.take();
                }
            })
            .ok();
        }));
    }

    pub fn will_delete_buffer(&mut self, buffer: Entity<Buffer>, cx: &mut Context<Self>) {
        // Ok to propagate file read time removal to linked action log
        self.remove_file_read_time(&buffer, cx);
        let has_linked_action_log = self.linked_action_log.is_some();
        let tracked_buffer = self.track_buffer_internal(buffer.clone(), false, cx);
        match tracked_buffer.status {
            TrackedBufferStatus::Created { .. } => {
                self.tracked_buffers.remove(&buffer);
                cx.notify();
            }
            TrackedBufferStatus::Modified => {
                tracked_buffer.status = TrackedBufferStatus::Deleted;
                if !has_linked_action_log {
                    buffer.update(cx, |buffer, cx| buffer.set_text("", cx));
                    tracked_buffer.schedule_diff_update(ChangeAuthor::Agent, cx);
                }
            }

            TrackedBufferStatus::Deleted => {}
        }

        if let Some(linked_action_log) = &mut self.linked_action_log {
            linked_action_log.update(cx, |log, cx| log.will_delete_buffer(buffer.clone(), cx));
        }

        if has_linked_action_log && let Some(tracked_buffer) = self.tracked_buffers.get_mut(&buffer)
        {
            tracked_buffer.schedule_diff_update(ChangeAuthor::Agent, cx);
        }

        cx.notify();
    }

    pub fn keep_edits_in_range(
        &mut self,
        buffer: Entity<Buffer>,
        buffer_range: Range<impl language::ToPoint>,
        telemetry: Option<ActionLogTelemetry>,
        cx: &mut Context<Self>,
    ) {
        let Some(tracked_buffer) = self.tracked_buffers.get_mut(&buffer) else {
            return;
        };

        let mut metrics = ActionLogMetrics::for_buffer(buffer.read(cx));
        let mut review_states = None;
        match tracked_buffer.status {
            TrackedBufferStatus::Deleted => {
                metrics.add_edits(tracked_buffer.unreviewed_edits.edits());
                self.tracked_buffers.remove(&buffer);
                cx.notify();
            }
            _ => {
                let before = ReviewBufferState::capture(tracked_buffer);
                let buffer = buffer.read(cx);
                let buffer_range =
                    buffer_range.start.to_point(buffer)..buffer_range.end.to_point(buffer);
                let mut delta = 0i32;
                let previous_unreviewed_edits = tracked_buffer.unreviewed_edits.clone();
                tracked_buffer.unreviewed_edits.retain_mut(|edit| {
                    edit.old.start = (edit.old.start as i32 + delta) as u32;
                    edit.old.end = (edit.old.end as i32 + delta) as u32;

                    if buffer_range.end.row < edit.new.start
                        || buffer_range.start.row > edit.new.end
                    {
                        true
                    } else {
                        let old_range = tracked_buffer
                            .diff_base
                            .point_to_offset(Point::new(edit.old.start, 0))
                            ..tracked_buffer.diff_base.point_to_offset(cmp::min(
                                Point::new(edit.old.end, 0),
                                tracked_buffer.diff_base.max_point(),
                            ));
                        let new_range = tracked_buffer
                            .snapshot
                            .point_to_offset(Point::new(edit.new.start, 0))
                            ..tracked_buffer.snapshot.point_to_offset(cmp::min(
                                Point::new(edit.new.end, 0),
                                tracked_buffer.snapshot.max_point(),
                            ));
                        tracked_buffer.diff_base.replace(
                            old_range,
                            &tracked_buffer
                                .snapshot
                                .text_for_range(new_range)
                                .collect::<String>(),
                        );
                        delta += edit.new_len() as i32 - edit.old_len() as i32;
                        metrics.add_edit(edit);
                        false
                    }
                });
                if tracked_buffer.unreviewed_edits.is_empty()
                    && let TrackedBufferStatus::Created { .. } = &mut tracked_buffer.status
                {
                    tracked_buffer.status = TrackedBufferStatus::Modified;
                }
                tracked_buffer.review_state_changed_before_recompute |=
                    tracked_buffer.unreviewed_edits != previous_unreviewed_edits;
                tracked_buffer.schedule_diff_update(ChangeAuthor::User, cx);
                let after = ReviewBufferState::capture(tracked_buffer);
                if before.unreviewed_edits != after.unreviewed_edits {
                    review_states = Some((before, after));
                }
            }
        }
        if let Some((before, after)) = review_states {
            let now = cx.background_executor().now();
            let transaction_id =
                buffer.update(cx, |buffer, _cx| buffer.push_empty_undo_transaction(now));
            self.record_review_decision(&buffer, transaction_id, before, after);
        }
        if let Some(telemetry) = telemetry {
            telemetry_report_accepted_edits(&telemetry, metrics);
        }
    }

    pub fn reject_edits_in_ranges(
        &mut self,
        buffer: Entity<Buffer>,
        buffer_ranges: Vec<Range<impl language::ToPoint>>,
        telemetry: Option<ActionLogTelemetry>,
        cx: &mut Context<Self>,
    ) -> (Task<Result<()>>, Option<PerBufferUndo>) {
        let Some(tracked_buffer) = self.tracked_buffers.get_mut(&buffer) else {
            return (Task::ready(Ok(())), None);
        };

        let mut metrics = ActionLogMetrics::for_buffer(buffer.read(cx));
        let mut undo_info: Option<PerBufferUndo> = None;
        let mut review_decision = None;
        let task = match tracked_buffer.status.clone() {
            TrackedBufferStatus::Created {
                existing_file_content,
            } => {
                metrics.add_edits(tracked_buffer.unreviewed_edits.edits());
                let task = if let Some(existing_file_content) = existing_file_content {
                    let before = ReviewBufferState::capture(tracked_buffer);
                    // Capture the agent's content before restoring existing file content
                    let agent_content = buffer.read(cx).text();
                    let buffer_id = buffer.read(cx).remote_id();

                    let transaction_id = buffer.update(cx, |buffer, cx| {
                        buffer.finalize_last_transaction();
                        buffer.start_transaction();
                        buffer.set_text("", cx);
                        for chunk in existing_file_content.chunks() {
                            buffer.append(chunk, cx);
                        }
                        let transaction_id = buffer.end_transaction(cx);
                        buffer.finalize_last_transaction();
                        transaction_id
                    });

                    tracked_buffer.status = TrackedBufferStatus::Modified;
                    tracked_buffer.diff_base = existing_file_content.clone();
                    tracked_buffer.unreviewed_edits.clear();
                    tracked_buffer.snapshot = buffer.read(cx).text_snapshot();
                    tracked_buffer.review_state_changed_before_recompute = true;
                    tracked_buffer.schedule_diff_update(ChangeAuthor::Agent, cx);
                    let after = ReviewBufferState::capture(tracked_buffer);
                    if let Some(transaction_id) = transaction_id {
                        review_decision = Some((transaction_id, before, after));
                    }

                    undo_info = Some(PerBufferUndo {
                        buffer: buffer.downgrade(),
                        edits_to_restore: vec![(
                            Anchor::min_for_buffer(buffer_id)..Anchor::max_for_buffer(buffer_id),
                            agent_content,
                        )],
                        status: UndoBufferStatus::Created {
                            had_existing_content: true,
                        },
                        transaction_id,
                    });

                    self.project
                        .update(cx, |project, cx| project.save_buffer(buffer.clone(), cx))
                } else {
                    // For a file created by AI with no pre-existing content,
                    // only delete the file if we're certain it contains only AI content
                    // with no edits from the user.

                    let initial_version = tracked_buffer.version.clone();
                    let current_version = buffer.read(cx).version();

                    let current_content = buffer.read(cx).text();
                    let tracked_content = tracked_buffer.snapshot.text();

                    let is_ai_only_content =
                        initial_version == current_version && current_content == tracked_content;

                    if is_ai_only_content {
                        let task = buffer
                            .read(cx)
                            .entry_id(cx)
                            .and_then(|entry_id| {
                                self.project
                                    .update(cx, |project, cx| project.delete_entry(entry_id, cx))
                            })
                            .unwrap_or_else(|| Task::ready(Ok(())));

                        cx.background_spawn(async move {
                            task.await?;
                            Ok(())
                        })
                    } else {
                        // Not sure how to disentangle edits made by the user
                        // from edits made by the AI at this point.
                        // For now, preserve both to avoid data loss.
                        //
                        // TODO: Better solution (disable "Reject" after user makes some
                        // edit or find a way to differentiate between AI and user edits)
                        Task::ready(Ok(()))
                    }
                };

                if matches!(tracked_buffer.status, TrackedBufferStatus::Created { .. }) {
                    self.tracked_buffers.remove(&buffer);
                }
                cx.notify();
                task
            }
            TrackedBufferStatus::Deleted => {
                buffer.update(cx, |buffer, cx| {
                    buffer.set_text(tracked_buffer.diff_base.to_string(), cx)
                });
                let save = self
                    .project
                    .update(cx, |project, cx| project.save_buffer(buffer.clone(), cx));

                // Clear all tracked edits for this buffer and start over as if we just read it.
                metrics.add_edits(tracked_buffer.unreviewed_edits.edits());
                self.tracked_buffers.remove(&buffer);
                self.buffer_read(buffer.clone(), cx);
                cx.notify();
                save
            }
            TrackedBufferStatus::Modified => {
                let before = ReviewBufferState::capture(tracked_buffer);
                let (edits_to_restore, transaction_id, remaining_unreviewed_edits) =
                    buffer.update(cx, |buffer, cx| {
                        let mut buffer_row_ranges = buffer_ranges
                            .into_iter()
                            .map(|range| {
                                range.start.to_point(buffer).row..range.end.to_point(buffer).row
                            })
                            .peekable();

                        let mut edits_to_revert = Vec::new();
                        let mut edits_for_undo = Vec::new();
                        let mut remaining_edits = Vec::new();
                        let mut new_row_delta = 0i64;
                        for edit in tracked_buffer.unreviewed_edits.edits() {
                            let new_range = tracked_buffer
                                .snapshot
                                .anchor_before(Point::new(edit.new.start, 0))
                                ..tracked_buffer.snapshot.anchor_after(cmp::min(
                                    Point::new(edit.new.end, 0),
                                    tracked_buffer.snapshot.max_point(),
                                ));
                            let new_row_range = new_range.start.to_point(buffer).row
                                ..new_range.end.to_point(buffer).row;

                            let mut revert = false;
                            while let Some(buffer_row_range) = buffer_row_ranges.peek() {
                                if buffer_row_range.end < new_row_range.start {
                                    buffer_row_ranges.next();
                                } else if buffer_row_range.start > new_row_range.end {
                                    break;
                                } else {
                                    revert = true;
                                    break;
                                }
                            }

                            if revert {
                                metrics.add_edit(edit);
                                let old_range = tracked_buffer
                                    .diff_base
                                    .point_to_offset(Point::new(edit.old.start, 0))
                                    ..tracked_buffer.diff_base.point_to_offset(cmp::min(
                                        Point::new(edit.old.end, 0),
                                        tracked_buffer.diff_base.max_point(),
                                    ));
                                let old_text = tracked_buffer
                                    .diff_base
                                    .chunks_in_range(old_range)
                                    .collect::<String>();

                                // Capture the agent's text before we revert it (for undo)
                                let new_range_offset = new_range.start.to_offset(buffer)
                                    ..new_range.end.to_offset(buffer);
                                let agent_text =
                                    buffer.text_for_range(new_range_offset).collect::<String>();
                                edits_for_undo.push((new_range.clone(), agent_text));

                                edits_to_revert.push((new_range, old_text));
                                new_row_delta += edit.old_len() as i64 - edit.new_len() as i64;
                            } else {
                                let mut remaining_edit = edit.clone();
                                remaining_edit.new.start = (remaining_edit.new.start as i64
                                    + new_row_delta)
                                    .clamp(0, u32::MAX as i64)
                                    as u32;
                                remaining_edit.new.end = (remaining_edit.new.end as i64
                                    + new_row_delta)
                                    .clamp(0, u32::MAX as i64)
                                    as u32;
                                remaining_edits.push(remaining_edit);
                            }
                        }

                        let transaction_id = if edits_to_revert.is_empty() {
                            None
                        } else {
                            buffer.finalize_last_transaction();
                            buffer.start_transaction();
                            buffer.edit(edits_to_revert, None, cx);
                            let transaction_id = buffer.end_transaction(cx);
                            buffer.finalize_last_transaction();
                            transaction_id
                        };
                        (edits_for_undo, transaction_id, Patch::new(remaining_edits))
                    });

                if transaction_id.is_some() {
                    tracked_buffer.unreviewed_edits = remaining_unreviewed_edits;
                    tracked_buffer.snapshot = buffer.read(cx).text_snapshot();
                    tracked_buffer.review_state_changed_before_recompute = true;
                    tracked_buffer.schedule_diff_update(ChangeAuthor::Agent, cx);
                }

                if !edits_to_restore.is_empty() {
                    undo_info = Some(PerBufferUndo {
                        buffer: buffer.downgrade(),
                        edits_to_restore,
                        status: UndoBufferStatus::Modified,
                        transaction_id,
                    });
                }

                if let Some(transaction_id) = transaction_id {
                    let after = ReviewBufferState::capture(tracked_buffer);
                    review_decision = Some((transaction_id, before, after));
                }

                self.project
                    .update(cx, |project, cx| project.save_buffer(buffer.clone(), cx))
            }
        };
        if let Some(telemetry) = telemetry {
            telemetry_report_rejected_edits(&telemetry, metrics);
        }
        if let Some((transaction_id, before, after)) = review_decision {
            self.record_review_decision(&buffer, transaction_id, before, after);
        }
        (task, undo_info)
    }

    pub fn keep_all_edits(
        &mut self,
        telemetry: Option<ActionLogTelemetry>,
        cx: &mut Context<Self>,
    ) {
        let mut review_states = Vec::new();
        self.tracked_buffers.retain(|buffer, tracked_buffer| {
            let mut metrics = ActionLogMetrics::for_buffer(buffer.read(cx));
            metrics.add_edits(tracked_buffer.unreviewed_edits.edits());
            if let Some(telemetry) = telemetry.as_ref() {
                telemetry_report_accepted_edits(telemetry, metrics);
            }
            match tracked_buffer.status {
                TrackedBufferStatus::Deleted => false,
                _ => {
                    let before = ReviewBufferState::capture(tracked_buffer);
                    if let TrackedBufferStatus::Created { .. } = &mut tracked_buffer.status {
                        tracked_buffer.status = TrackedBufferStatus::Modified;
                    }
                    tracked_buffer.review_state_changed_before_recompute |=
                        !tracked_buffer.unreviewed_edits.is_empty();
                    tracked_buffer.unreviewed_edits.clear();
                    tracked_buffer.diff_base = tracked_buffer.snapshot.as_rope().clone();
                    tracked_buffer.schedule_diff_update(ChangeAuthor::User, cx);
                    let after = ReviewBufferState::capture(tracked_buffer);
                    if before.unreviewed_edits != after.unreviewed_edits {
                        review_states.push((buffer.clone(), before, after));
                    }
                    true
                }
            }
        });

        for (buffer, before, after) in review_states {
            let now = cx.background_executor().now();
            let transaction_id =
                buffer.update(cx, |buffer, _cx| buffer.push_empty_undo_transaction(now));
            self.record_review_decision(&buffer, transaction_id, before, after);
        }

        cx.notify();
    }

    pub fn reject_all_edits(
        &mut self,
        telemetry: Option<ActionLogTelemetry>,
        cx: &mut Context<Self>,
    ) -> Task<()> {
        // Clear any previous undo state before starting a new reject operation
        self.last_reject_undo = None;

        let mut undo_buffers = Vec::new();
        let mut futures = Vec::new();

        for buffer in self
            .changed_buffers(cx)
            .map(|(buffer, _)| buffer)
            .collect::<Vec<_>>()
        {
            let buffer_ranges = vec![Anchor::min_max_range_for_buffer(
                buffer.read(cx).remote_id(),
            )];
            let (reject_task, undo_info) =
                self.reject_edits_in_ranges(buffer, buffer_ranges, telemetry.clone(), cx);

            if let Some(undo) = undo_info {
                undo_buffers.push(undo);
            }

            futures.push(async move {
                reject_task.await.log_err();
            });
        }

        // Store the undo information if we have any
        if !undo_buffers.is_empty() {
            self.last_reject_undo = Some(LastRejectUndo {
                buffers: undo_buffers,
            });
        }

        let task = futures::future::join_all(futures);
        cx.background_spawn(async move {
            task.await;
        })
    }

    pub fn has_pending_undo(&self) -> bool {
        self.last_reject_undo.is_some()
    }

    pub fn set_last_reject_undo(&mut self, undo: LastRejectUndo) {
        self.last_reject_undo = Some(undo);
    }

    /// Undoes the most recent reject operation, restoring the rejected agent changes.
    /// This is a best-effort operation: if buffers have been closed or modified externally,
    /// those buffers will be skipped.
    pub fn undo_last_reject(&mut self, cx: &mut Context<Self>) -> Task<()> {
        let Some(undo) = self.last_reject_undo.take() else {
            return Task::ready(());
        };

        let mut save_tasks = Vec::with_capacity(undo.buffers.len());

        for per_buffer_undo in undo.buffers {
            // Skip if the buffer entity has been deallocated
            let Some(buffer) = per_buffer_undo.buffer.upgrade() else {
                continue;
            };

            if let Some(transaction_id) = per_buffer_undo.transaction_id {
                let undone =
                    buffer.update(cx, |buffer, cx| buffer.undo_transaction(transaction_id, cx));
                if undone {
                    let save = self
                        .project
                        .update(cx, |project, cx| project.save_buffer(buffer, cx));
                    save_tasks.push(save);
                    continue;
                }
            }

            buffer.update(cx, |buffer, cx| {
                let mut valid_edits = Vec::new();

                for (anchor_range, text_to_restore) in per_buffer_undo.edits_to_restore {
                    if anchor_range.start.buffer_id == buffer.remote_id()
                        && anchor_range.end.buffer_id == buffer.remote_id()
                    {
                        valid_edits.push((anchor_range, text_to_restore));
                    }
                }

                if !valid_edits.is_empty() {
                    buffer.edit(valid_edits, None, cx);
                }
            });

            if !self.tracked_buffers.contains_key(&buffer) {
                self.buffer_edited(buffer.clone(), cx);
            }

            let save = self
                .project
                .update(cx, |project, cx| project.save_buffer(buffer, cx));
            save_tasks.push(save);
        }

        cx.notify();

        cx.background_spawn(async move {
            futures::future::join_all(save_tasks).await;
        })
    }

    /// Returns the set of buffers that contain edits that haven't been reviewed by the user.
    pub fn changed_buffers(
        &self,
        cx: &App,
    ) -> impl Iterator<Item = (Entity<Buffer>, Entity<BufferDiff>)> {
        self.tracked_buffers
            .iter()
            .filter(|(_, tracked)| tracked.has_edits(cx))
            .map(|(buffer, tracked)| (buffer.clone(), tracked.diff.clone()))
    }

    /// Returns the total number of lines added and removed across all unreviewed buffers.
    pub fn diff_stats(&self, cx: &App) -> DiffStats {
        DiffStats::all_files(self.changed_buffers(cx), cx)
    }

    pub fn diff_load(&self, cx: &App) -> AgentDiffLoad {
        let mut file_count = 0usize;
        let mut complexity = DiffComplexity::default();
        for tracked_buffer in self.tracked_buffers.values() {
            if tracked_buffer.has_edits(cx) {
                file_count += 1;
                complexity += tracked_buffer.diff_complexity;
            }
        }

        let load = AgentDiffLoad::new(file_count, complexity);
        let is_large = load.is_large();
        if self.last_reported_large_diff_state.get() != Some(is_large) {
            self.last_reported_large_diff_state.set(Some(is_large));
            log::debug!(
                "agent diff load changed: large={is_large}, files={file_count}, complexity={complexity:?}"
            );
        }
        load
    }

    pub fn lsp_lease_debug_counters(&self) -> AgentLspLeaseDebugCounters {
        AgentLspLeaseDebugCounters {
            tracked_buffers: self.tracked_buffers.len(),
            read_leases: self.lsp_lease_counters.read_leases.load(Ordering::Relaxed),
            edit_leases: self.lsp_lease_counters.edit_leases.load(Ordering::Relaxed),
            diagnostic_leases: self
                .lsp_lease_counters
                .diagnostic_leases
                .load(Ordering::Relaxed),
            queued_acquisitions: self
                .lsp_lease_counters
                .queued_acquisitions
                .load(Ordering::Relaxed),
        }
    }

    pub fn buffer_diff_load(&self, buffer: &Entity<Buffer>, cx: &App) -> AgentDiffLoad {
        self.tracked_buffers
            .get(buffer)
            .filter(|tracked_buffer| tracked_buffer.has_edits(cx))
            .map_or_else(
                || AgentDiffLoad::new(0, DiffComplexity::default()),
                |tracked_buffer| AgentDiffLoad::new(1, tracked_buffer.diff_complexity),
            )
    }

    /// Iterate over buffers changed since last read or edited by the model
    pub fn stale_buffers<'a>(&'a self, cx: &'a App) -> impl Iterator<Item = &'a Entity<Buffer>> {
        self.tracked_buffers
            .iter()
            .filter(|(buffer, tracked)| {
                let buffer = buffer.read(cx);

                tracked.version != buffer.version
                    && buffer
                        .file()
                        .is_some_and(|file| !file.disk_state().is_deleted())
            })
            .map(|(buffer, _)| buffer)
    }
}

#[derive(Default, Debug, Clone, Copy)]
pub struct DiffStats {
    pub lines_added: u32,
    pub lines_removed: u32,
}

const LARGE_DIFF_FILE_COUNT: usize = 50;
const LARGE_DIFF_CHANGED_ROWS: u64 = 20_000;
const LARGE_DIFF_CHANGED_BYTES: u64 = 4 * 1024 * 1024;
const LARGE_DIFF_HUNK_COUNT: u64 = 2_000;
const LARGE_DIFF_LONGEST_CHANGED_LINE_BYTES: u32 = 256 * 1024;

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct DiffComplexity {
    pub changed_rows: u64,
    pub changed_bytes: u64,
    pub hunk_count: u64,
    pub longest_changed_line_bytes: u32,
}

impl DiffComplexity {
    fn from_diff(
        diff: &BufferDiffSnapshot,
        buffer: &text::BufferSnapshot,
        diff_base: &Rope,
    ) -> Self {
        let (lines_added, lines_removed) = diff.changed_row_counts();
        let mut complexity = Self {
            changed_rows: u64::from(lines_added) + u64::from(lines_removed),
            ..Self::default()
        };

        for hunk in diff.hunks(buffer) {
            complexity.hunk_count = complexity.hunk_count.saturating_add(1);
            let buffer_byte_range =
                hunk.buffer_range.start.to_offset(buffer)..hunk.buffer_range.end.to_offset(buffer);
            complexity.changed_bytes = complexity
                .changed_bytes
                .saturating_add(buffer_byte_range.len() as u64)
                .saturating_add(hunk.diff_base_byte_range.len() as u64);

            let buffer_range = hunk.range;
            for row in buffer_range.start.row..=buffer_range.end.row.min(buffer.max_point().row) {
                complexity.longest_changed_line_bytes = complexity
                    .longest_changed_line_bytes
                    .max(buffer.line_len(row));
            }

            let base_start = diff_base.offset_to_point(hunk.diff_base_byte_range.start);
            let base_end = diff_base.offset_to_point(hunk.diff_base_byte_range.end);
            for row in base_start.row..=base_end.row.min(diff_base.max_point().row) {
                complexity.longest_changed_line_bytes = complexity
                    .longest_changed_line_bytes
                    .max(diff_base.line_len(row));
            }
        }

        complexity
    }
}

impl std::ops::AddAssign for DiffComplexity {
    fn add_assign(&mut self, other: Self) {
        self.changed_rows = self.changed_rows.saturating_add(other.changed_rows);
        self.changed_bytes = self.changed_bytes.saturating_add(other.changed_bytes);
        self.hunk_count = self.hunk_count.saturating_add(other.hunk_count);
        self.longest_changed_line_bytes = self
            .longest_changed_line_bytes
            .max(other.longest_changed_line_bytes);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LargeDiffReason {
    FileCount,
    ChangedRows,
    ChangedBytes,
    HunkCount,
    LongestChangedLine,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentDiffLoad {
    Normal {
        file_count: usize,
        complexity: DiffComplexity,
    },
    Large {
        file_count: usize,
        complexity: DiffComplexity,
        reasons: Vec<LargeDiffReason>,
    },
}

impl AgentDiffLoad {
    fn new(file_count: usize, complexity: DiffComplexity) -> Self {
        let mut reasons = Vec::new();
        if file_count > LARGE_DIFF_FILE_COUNT {
            reasons.push(LargeDiffReason::FileCount);
        }
        if complexity.changed_rows > LARGE_DIFF_CHANGED_ROWS {
            reasons.push(LargeDiffReason::ChangedRows);
        }
        if complexity.changed_bytes > LARGE_DIFF_CHANGED_BYTES {
            reasons.push(LargeDiffReason::ChangedBytes);
        }
        if complexity.hunk_count > LARGE_DIFF_HUNK_COUNT {
            reasons.push(LargeDiffReason::HunkCount);
        }
        if complexity.longest_changed_line_bytes > LARGE_DIFF_LONGEST_CHANGED_LINE_BYTES {
            reasons.push(LargeDiffReason::LongestChangedLine);
        }

        if reasons.is_empty() {
            Self::Normal {
                file_count,
                complexity,
            }
        } else {
            Self::Large {
                file_count,
                complexity,
                reasons,
            }
        }
    }

    pub fn is_large(&self) -> bool {
        matches!(self, Self::Large { .. })
    }

    pub fn file_count(&self) -> usize {
        match self {
            Self::Normal { file_count, .. } | Self::Large { file_count, .. } => *file_count,
        }
    }

    pub fn complexity(&self) -> DiffComplexity {
        match self {
            Self::Normal { complexity, .. } | Self::Large { complexity, .. } => *complexity,
        }
    }
}

impl DiffStats {
    pub fn single_file(diff: &BufferDiff) -> Self {
        let (lines_added, lines_removed) = diff.changed_row_counts();
        DiffStats {
            lines_added,
            lines_removed,
        }
    }

    pub fn all_files(
        changed_buffers: impl IntoIterator<Item = (Entity<Buffer>, Entity<BufferDiff>)>,
        cx: &App,
    ) -> Self {
        let mut total = DiffStats::default();
        for (_, diff) in changed_buffers {
            let stats = DiffStats::single_file(diff.read(cx));
            total.lines_added += stats.lines_added;
            total.lines_removed += stats.lines_removed;
        }
        total
    }
}

#[derive(Clone)]
pub struct ActionLogTelemetry {
    pub agent_telemetry_id: SharedString,
    pub session_id: Arc<str>,
}

struct ActionLogMetrics {
    lines_removed: u32,
    lines_added: u32,
    language: Option<SharedString>,
}

impl ActionLogMetrics {
    fn for_buffer(buffer: &Buffer) -> Self {
        Self {
            language: buffer.language().map(|l| l.name().0),
            lines_removed: 0,
            lines_added: 0,
        }
    }

    fn add_edits(&mut self, edits: &[Edit<u32>]) {
        for edit in edits {
            self.add_edit(edit);
        }
    }

    fn add_edit(&mut self, edit: &Edit<u32>) {
        self.lines_added += edit.new_len();
        self.lines_removed += edit.old_len();
    }
}

fn telemetry_report_accepted_edits(telemetry: &ActionLogTelemetry, metrics: ActionLogMetrics) {
    telemetry::event!(
        "Agent Edits Accepted",
        agent = telemetry.agent_telemetry_id,
        session = telemetry.session_id,
        language = metrics.language,
        lines_added = metrics.lines_added,
        lines_removed = metrics.lines_removed
    );
}

fn telemetry_report_rejected_edits(telemetry: &ActionLogTelemetry, metrics: ActionLogMetrics) {
    telemetry::event!(
        "Agent Edits Rejected",
        agent = telemetry.agent_telemetry_id,
        session = telemetry.session_id,
        language = metrics.language,
        lines_added = metrics.lines_added,
        lines_removed = metrics.lines_removed
    );
}

fn apply_non_conflicting_edits(
    patch: &Patch<u32>,
    edits: Vec<Edit<u32>>,
    old_text: &mut Rope,
    new_text: &Rope,
) -> bool {
    let mut old_edits = patch.edits().iter().cloned().peekable();
    let mut new_edits = edits.into_iter().peekable();
    let mut applied_delta = 0i32;
    let mut rebased_delta = 0i32;
    let mut has_made_changes = false;

    while let Some(mut new_edit) = new_edits.next() {
        let mut conflict = false;

        // Push all the old edits that are before this new edit or that intersect with it.
        while let Some(old_edit) = old_edits.peek() {
            if new_edit.old.end < old_edit.new.start
                || (!old_edit.new.is_empty() && new_edit.old.end == old_edit.new.start)
            {
                break;
            } else if new_edit.old.start > old_edit.new.end
                || (!old_edit.new.is_empty() && new_edit.old.start == old_edit.new.end)
            {
                let old_edit = old_edits.next().unwrap();
                rebased_delta += old_edit.new_len() as i32 - old_edit.old_len() as i32;
            } else {
                conflict = true;
                if new_edits
                    .peek()
                    .is_some_and(|next_edit| next_edit.old.overlaps(&old_edit.new))
                {
                    new_edit = new_edits.next().unwrap();
                } else {
                    let old_edit = old_edits.next().unwrap();
                    rebased_delta += old_edit.new_len() as i32 - old_edit.old_len() as i32;
                }
            }
        }

        if !conflict {
            // This edit doesn't intersect with any old edit, so we can apply it to the old text.
            new_edit.old.start = (new_edit.old.start as i32 + applied_delta - rebased_delta) as u32;
            new_edit.old.end = (new_edit.old.end as i32 + applied_delta - rebased_delta) as u32;
            let old_bytes = old_text.point_to_offset(Point::new(new_edit.old.start, 0))
                ..old_text.point_to_offset(cmp::min(
                    Point::new(new_edit.old.end, 0),
                    old_text.max_point(),
                ));
            let new_bytes = new_text.point_to_offset(Point::new(new_edit.new.start, 0))
                ..new_text.point_to_offset(cmp::min(
                    Point::new(new_edit.new.end, 0),
                    new_text.max_point(),
                ));

            old_text.replace(
                old_bytes,
                &new_text.chunks_in_range(new_bytes).collect::<String>(),
            );
            applied_delta += new_edit.new_len() as i32 - new_edit.old_len() as i32;
            has_made_changes = true;
        }
    }
    has_made_changes
}

fn diff_snapshots(
    old_snapshot: &text::BufferSnapshot,
    new_snapshot: &text::BufferSnapshot,
) -> Vec<Edit<u32>> {
    let mut edits = new_snapshot
        .edits_since::<Point>(&old_snapshot.version)
        .map(|edit| point_to_row_edit(edit, old_snapshot.as_rope(), new_snapshot.as_rope()))
        .peekable();
    let mut row_edits = Vec::new();
    while let Some(mut edit) = edits.next() {
        while let Some(next_edit) = edits.peek() {
            if edit.old.end >= next_edit.old.start {
                edit.old.end = next_edit.old.end;
                edit.new.end = next_edit.new.end;
                edits.next();
            } else {
                break;
            }
        }
        row_edits.push(edit);
    }
    row_edits
}

fn point_to_row_edit(edit: Edit<Point>, old_text: &Rope, new_text: &Rope) -> Edit<u32> {
    if edit.old.start.column == old_text.line_len(edit.old.start.row)
        && new_text
            .chars_at(new_text.point_to_offset(edit.new.start))
            .next()
            == Some('\n')
        && edit.old.start != old_text.max_point()
    {
        Edit {
            old: edit.old.start.row + 1..edit.old.end.row + 1,
            new: edit.new.start.row + 1..edit.new.end.row + 1,
        }
    } else if edit.old.start.column == 0 && edit.old.end.column == 0 && edit.new.end.column == 0 {
        Edit {
            old: edit.old.start.row..edit.old.end.row,
            new: edit.new.start.row..edit.new.end.row,
        }
    } else {
        Edit {
            old: edit.old.start.row..edit.old.end.row + 1,
            new: edit.new.start.row..edit.new.end.row + 1,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum ChangeAuthor {
    User,
    Agent,
}

impl ChangeAuthor {
    fn coalesce(self, other: Self) -> Self {
        if matches!(self, Self::Agent) || matches!(other, Self::Agent) {
            Self::Agent
        } else {
            Self::User
        }
    }
}

struct PendingDiffUpdate {
    generation: u64,
    author: ChangeAuthor,
    snapshot: text::BufferSnapshot,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AgentLspLeaseOwnership {
    Legacy,
    Edit,
    Diagnostic,
}

#[derive(Default)]
struct AgentLspLeaseCounters {
    read_leases: AtomicUsize,
    edit_leases: AtomicUsize,
    diagnostic_leases: AtomicUsize,
    queued_acquisitions: AtomicUsize,
}

const MAX_AGENT_LSP_LEASES_PER_PROJECT: usize = 4;
const AGENT_EDIT_LSP_LEASE_IDLE_TTL: Duration = Duration::from_secs(20);

#[derive(Default)]
struct AgentLspLeaseLimiter {
    state: Mutex<AgentLspLeaseLimiterState>,
}

#[derive(Default)]
struct AgentLspLeaseLimiterState {
    active: usize,
    diagnostic_queue: VecDeque<oneshot::Sender<()>>,
    edit_queue: VecDeque<oneshot::Sender<()>>,
}

struct AgentLspPermit {
    limiter: Arc<AgentLspLeaseLimiter>,
}

impl AgentLspLeaseLimiter {
    async fn acquire(
        self: &Arc<Self>,
        ownership: AgentLspLeaseOwnership,
        counters: &Arc<AgentLspLeaseCounters>,
    ) -> AgentLspPermit {
        let receiver = {
            let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
            if state.active < MAX_AGENT_LSP_LEASES_PER_PROJECT {
                state.active += 1;
                None
            } else {
                let (sender, receiver) = oneshot::channel();
                match ownership {
                    AgentLspLeaseOwnership::Diagnostic => state.diagnostic_queue.push_back(sender),
                    AgentLspLeaseOwnership::Edit | AgentLspLeaseOwnership::Legacy => {
                        state.edit_queue.push_back(sender)
                    }
                }
                counters.queued_acquisitions.fetch_add(1, Ordering::Relaxed);
                Some(receiver)
            }
        };

        if let Some(receiver) = receiver {
            let _queued = QueuedAgentLspAcquisition {
                counters: counters.clone(),
            };
            // The limiter owns the corresponding sender until this request is
            // granted. If the future is cancelled, release() skips the closed
            // sender and transfers the permit to the next waiter.
            receiver.await.ok();
        }

        AgentLspPermit {
            limiter: self.clone(),
        }
    }

    fn release(&self) {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        loop {
            let next = state
                .diagnostic_queue
                .pop_front()
                .or_else(|| state.edit_queue.pop_front());
            let Some(next) = next else {
                state.active = state.active.saturating_sub(1);
                return;
            };
            if next.send(()).is_ok() {
                // The active slot transfers directly to the awakened waiter.
                return;
            }
        }
    }
}

impl Drop for AgentLspPermit {
    fn drop(&mut self) {
        self.limiter.release();
    }
}

struct QueuedAgentLspAcquisition {
    counters: Arc<AgentLspLeaseCounters>,
}

impl Drop for QueuedAgentLspAcquisition {
    fn drop(&mut self) {
        let previous = self
            .counters
            .queued_acquisitions
            .fetch_sub(1, Ordering::Relaxed);
        debug_assert!(
            previous > 0,
            "Agent LSP acquisition queue counter underflow"
        );
    }
}

fn project_lsp_lease_limiter(project_id: EntityId) -> Arc<AgentLspLeaseLimiter> {
    static LIMITERS: OnceLock<Mutex<HashMap<EntityId, SyncWeak<AgentLspLeaseLimiter>>>> =
        OnceLock::new();
    let mut limiters = LIMITERS
        .get_or_init(|| Mutex::new(HashMap::default()))
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    if let Some(limiter) = limiters.get(&project_id).and_then(SyncWeak::upgrade) {
        return limiter;
    }

    let limiter = Arc::new(AgentLspLeaseLimiter::default());
    limiters.insert(project_id, Arc::downgrade(&limiter));
    limiter
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AgentLspLeaseDebugCounters {
    pub tracked_buffers: usize,
    pub read_leases: usize,
    pub edit_leases: usize,
    pub diagnostic_leases: usize,
    pub queued_acquisitions: usize,
}

pub struct AgentLspLease {
    _handle: OpenLspBufferHandle,
    _permit: Option<AgentLspPermit>,
    ownership: AgentLspLeaseOwnership,
    counters: Arc<AgentLspLeaseCounters>,
}

impl AgentLspLease {
    fn new(
        handle: OpenLspBufferHandle,
        ownership: AgentLspLeaseOwnership,
        counters: Arc<AgentLspLeaseCounters>,
        permit: Option<AgentLspPermit>,
    ) -> Self {
        counters.increment(ownership);
        Self {
            _handle: handle,
            _permit: permit,
            ownership,
            counters,
        }
    }
}

impl Drop for AgentLspLease {
    fn drop(&mut self) {
        self.counters.decrement(self.ownership);
    }
}

impl AgentLspLeaseCounters {
    fn counter(&self, ownership: AgentLspLeaseOwnership) -> &AtomicUsize {
        match ownership {
            AgentLspLeaseOwnership::Legacy => &self.read_leases,
            AgentLspLeaseOwnership::Edit => &self.edit_leases,
            AgentLspLeaseOwnership::Diagnostic => &self.diagnostic_leases,
        }
    }

    fn increment(&self, ownership: AgentLspLeaseOwnership) {
        self.counter(ownership).fetch_add(1, Ordering::Relaxed);
    }

    fn decrement(&self, ownership: AgentLspLeaseOwnership) {
        let previous = self.counter(ownership).fetch_sub(1, Ordering::Relaxed);
        debug_assert!(previous > 0, "Agent LSP lease counter underflow");
    }
}

#[derive(Clone, Debug)]
enum TrackedBufferStatus {
    Created { existing_file_content: Option<Rope> },
    Modified,
    Deleted,
}

#[derive(Clone)]
struct ReviewBufferState {
    diff_base: Rope,
    unreviewed_edits: Patch<u32>,
    status: TrackedBufferStatus,
}

impl ReviewBufferState {
    fn capture(tracked_buffer: &TrackedBuffer) -> Self {
        Self {
            diff_base: tracked_buffer.diff_base.clone(),
            unreviewed_edits: tracked_buffer.unreviewed_edits.clone(),
            status: tracked_buffer.status.clone(),
        }
    }

    fn restore(self, tracked_buffer: &mut TrackedBuffer, snapshot: text::BufferSnapshot) {
        tracked_buffer.diff_base = self.diff_base;
        tracked_buffer.unreviewed_edits = self.unreviewed_edits;
        tracked_buffer.status = self.status;
        tracked_buffer.snapshot = snapshot;
        tracked_buffer.review_state_changed_before_recompute = true;
    }
}

struct ReviewDecision {
    buffer: WeakEntity<Buffer>,
    transaction_id: clock::Lamport,
    before: ReviewBufferState,
    after: ReviewBufferState,
}

pub struct TrackedBuffer {
    buffer: Entity<Buffer>,
    diff_base: Rope,
    unreviewed_edits: Patch<u32>,
    status: TrackedBufferStatus,
    version: clock::Global,
    diff: Entity<BufferDiff>,
    snapshot: text::BufferSnapshot,
    diff_update: watch::Sender<()>,
    pending_diff_update: Option<PendingDiffUpdate>,
    in_flight_diff_update: Option<(u64, ChangeAuthor)>,
    diff_generation: u64,
    review_state_changed_before_recompute: bool,
    diff_complexity: DiffComplexity,
    lsp_lease: Option<AgentLspLease>,
    lsp_lease_generation: u64,
    active_edit_sessions: usize,
    lsp_release_task: Option<Task<()>>,
    _maintain_diff: Task<()>,
    _subscription: Subscription,
}

impl TrackedBuffer {
    #[cfg(any(test, feature = "test-support"))]
    pub fn diff(&self) -> &Entity<BufferDiff> {
        &self.diff
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn diff_base_len(&self) -> usize {
        self.diff_base.len()
    }

    fn has_edits(&self, cx: &App) -> bool {
        self.diff
            .read(cx)
            .snapshot(cx)
            .hunks(self.buffer.read(cx))
            .next()
            .is_some()
    }

    fn schedule_diff_update(&mut self, author: ChangeAuthor, cx: &App) {
        let snapshot = self.buffer.read(cx).text_snapshot();
        self.diff_generation = self.diff_generation.saturating_add(1);
        let author = self
            .in_flight_diff_update
            .map(|(_, in_flight_author)| author.coalesce(in_flight_author))
            .unwrap_or(author);

        if let Some(pending) = &mut self.pending_diff_update {
            // Treat a coalesced Agent + User delta as Agent-authored. This may
            // conservatively show a user edit for review, but cannot silently
            // accept an Agent edit into the diff base.
            pending.generation = self.diff_generation;
            pending.author = pending.author.coalesce(author);
            pending.snapshot = snapshot;
        } else {
            self.pending_diff_update = Some(PendingDiffUpdate {
                generation: self.diff_generation,
                author,
                snapshot,
            });
        }
        self.diff_update.send(()).ok();
    }
}

pub struct ChangedBuffer {
    pub diff: Entity<BufferDiff>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use buffer_diff::DiffHunkStatusKind;
    use gpui::{TestAppContext, UpdateGlobal};
    use indoc::indoc;
    use language::Point;
    use project::{FakeFs, Fs, Project, RemoveOptions};
    use rand::prelude::*;

    #[gpui::test]
    async fn test_legacy_lsp_lease_matches_tracked_buffer_lifetime(cx: &mut TestAppContext) {
        init_test(cx);

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(path!("/dir"), json!({"file": "hello"}))
            .await;
        let project = Project::test(fs, [path!("/dir").as_ref()], cx).await;
        let action_log = cx.new(|_| ActionLog::new(project.clone()));
        let file_path = project
            .read_with(cx, |project, cx| project.find_project_path("dir/file", cx))
            .unwrap();
        let buffer = project
            .update(cx, |project, cx| project.open_buffer(file_path, cx))
            .await
            .unwrap();

        action_log.update(cx, |log, cx| log.buffer_read(buffer.clone(), cx));
        action_log.update(cx, |log, cx| log.buffer_read(buffer.clone(), cx));

        assert_eq!(
            action_log.read_with(cx, |log, _cx| log.lsp_lease_debug_counters()),
            AgentLspLeaseDebugCounters {
                tracked_buffers: 1,
                read_leases: 1,
                edit_leases: 0,
                diagnostic_leases: 0,
                queued_acquisitions: 0,
            }
        );
    }

    #[gpui::test]
    async fn test_experimental_lsp_lease_skips_reads_and_acquires_for_edits(
        cx: &mut TestAppContext,
    ) {
        init_test(cx);
        cx.update(|cx| {
            SettingsStore::update_global(cx, |store, cx| {
                store.update_user_settings(cx, |settings| {
                    settings
                        .agent
                        .get_or_insert_default()
                        .experimental_lsp_leases = Some(true);
                });
            });
        });

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(path!("/dir"), json!({"file": "hello"}))
            .await;
        let project = Project::test(fs, [path!("/dir").as_ref()], cx).await;
        let action_log = cx.new(|_| ActionLog::new(project.clone()));
        let file_path = project
            .read_with(cx, |project, cx| project.find_project_path("dir/file", cx))
            .unwrap();
        let buffer = project
            .update(cx, |project, cx| project.open_buffer(file_path, cx))
            .await
            .unwrap();

        action_log.update(cx, |log, cx| log.buffer_read(buffer.clone(), cx));
        assert_eq!(
            action_log.read_with(cx, |log, _cx| log.lsp_lease_debug_counters()),
            AgentLspLeaseDebugCounters {
                tracked_buffers: 1,
                ..Default::default()
            }
        );

        action_log
            .update(cx, |log, cx| log.acquire_edit_lsp_lease(buffer.clone(), cx))
            .await
            .unwrap();
        assert_eq!(
            action_log.read_with(cx, |log, _cx| log.lsp_lease_debug_counters()),
            AgentLspLeaseDebugCounters {
                tracked_buffers: 1,
                edit_leases: 1,
                ..Default::default()
            }
        );

        action_log.update(cx, |log, cx| log.finish_edit_lsp_lease(&buffer, cx));
        cx.executor().advance_clock(Duration::from_secs(19));
        cx.run_until_parked();
        assert_eq!(
            action_log.read_with(cx, |log, _cx| {
                log.lsp_lease_debug_counters().edit_leases
            }),
            1
        );

        action_log
            .update(cx, |log, cx| log.acquire_edit_lsp_lease(buffer.clone(), cx))
            .await
            .unwrap();
        action_log.update(cx, |log, cx| log.finish_edit_lsp_lease(&buffer, cx));
        cx.executor().advance_clock(Duration::from_secs(2));
        cx.run_until_parked();
        assert_eq!(
            action_log.read_with(cx, |log, _cx| {
                log.lsp_lease_debug_counters().edit_leases
            }),
            1,
            "the superseded release timer must not drop a reacquired lease"
        );

        cx.executor().advance_clock(Duration::from_secs(19));
        cx.run_until_parked();
        assert_eq!(
            action_log.read_with(cx, |log, _cx| {
                log.lsp_lease_debug_counters().edit_leases
            }),
            0
        );

        cx.update(|cx| {
            SettingsStore::update_global(cx, |store, cx| {
                store.update_user_settings(cx, |settings| {
                    settings
                        .agent
                        .get_or_insert_default()
                        .experimental_lsp_leases = Some(false);
                });
            });
        });
        cx.run_until_parked();
        assert_eq!(
            action_log.read_with(cx, |log, _cx| log.lsp_lease_debug_counters()),
            AgentLspLeaseDebugCounters {
                tracked_buffers: 1,
                read_leases: 1,
                ..Default::default()
            },
            "disabling the experiment must restore the legacy lease immediately"
        );
    }

    #[gpui::test]
    async fn test_lsp_lease_limiter_prioritizes_diagnostics(cx: &mut TestAppContext) {
        let limiter = Arc::new(AgentLspLeaseLimiter::default());
        let counters = Arc::new(AgentLspLeaseCounters::default());
        let mut active = Vec::new();
        for _ in 0..MAX_AGENT_LSP_LEASES_PER_PROJECT {
            active.push(
                limiter
                    .acquire(AgentLspLeaseOwnership::Edit, &counters)
                    .await,
            );
        }

        let edit_acquired = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let edit_task = cx.executor().spawn({
            let limiter = limiter.clone();
            let counters = counters.clone();
            let edit_acquired = edit_acquired.clone();
            async move {
                let permit = limiter
                    .acquire(AgentLspLeaseOwnership::Edit, &counters)
                    .await;
                edit_acquired.store(true, Ordering::Relaxed);
                permit
            }
        });
        cx.run_until_parked();

        let diagnostic_acquired = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let diagnostic_task = cx.executor().spawn({
            let limiter = limiter.clone();
            let counters = counters.clone();
            let diagnostic_acquired = diagnostic_acquired.clone();
            async move {
                let permit = limiter
                    .acquire(AgentLspLeaseOwnership::Diagnostic, &counters)
                    .await;
                diagnostic_acquired.store(true, Ordering::Relaxed);
                permit
            }
        });
        cx.run_until_parked();
        assert_eq!(counters.queued_acquisitions.load(Ordering::Relaxed), 2);

        active.pop();
        cx.run_until_parked();
        assert!(diagnostic_acquired.load(Ordering::Relaxed));
        assert!(!edit_acquired.load(Ordering::Relaxed));

        let diagnostic_permit = diagnostic_task.await;
        drop(diagnostic_permit);
        cx.run_until_parked();
        assert!(edit_acquired.load(Ordering::Relaxed));
        drop(edit_task.await);
        assert_eq!(counters.queued_acquisitions.load(Ordering::Relaxed), 0);
    }

    #[gpui::test]
    async fn test_concurrent_edit_sessions_share_buffer_lease(cx: &mut TestAppContext) {
        init_test(cx);
        cx.update(|cx| {
            SettingsStore::update_global(cx, |store, cx| {
                store.update_user_settings(cx, |settings| {
                    settings
                        .agent
                        .get_or_insert_default()
                        .experimental_lsp_leases = Some(true);
                });
            });
        });

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(path!("/dir"), json!({"file": "hello"}))
            .await;
        let project = Project::test(fs, [path!("/dir").as_ref()], cx).await;
        let action_log = cx.new(|_| ActionLog::new(project.clone()));
        let file_path = project
            .read_with(cx, |project, cx| project.find_project_path("dir/file", cx))
            .unwrap();
        let buffer = project
            .update(cx, |project, cx| project.open_buffer(file_path, cx))
            .await
            .unwrap();

        let first = action_log.update(cx, |log, cx| log.acquire_edit_lsp_lease(buffer.clone(), cx));
        let second =
            action_log.update(cx, |log, cx| log.acquire_edit_lsp_lease(buffer.clone(), cx));
        futures::future::try_join(first, second).await.unwrap();

        assert_eq!(
            action_log.read_with(cx, |log, _cx| {
                log.lsp_lease_debug_counters().edit_leases
            }),
            1,
            "concurrent sessions for one buffer must share one registration"
        );
        action_log.update(cx, |log, cx| log.finish_edit_lsp_lease(&buffer, cx));
        cx.executor().advance_clock(AGENT_EDIT_LSP_LEASE_IDLE_TTL);
        cx.run_until_parked();
        assert_eq!(
            action_log.read_with(cx, |log, _cx| {
                log.lsp_lease_debug_counters().edit_leases
            }),
            1,
            "the first session must not release the shared lease"
        );

        action_log.update(cx, |log, cx| log.finish_edit_lsp_lease(&buffer, cx));
        cx.executor().advance_clock(AGENT_EDIT_LSP_LEASE_IDLE_TTL);
        cx.run_until_parked();
        assert_eq!(
            action_log.read_with(cx, |log, _cx| {
                log.lsp_lease_debug_counters().edit_leases
            }),
            0
        );
    }

    #[gpui::test]
    async fn test_enabling_experimental_leases_does_not_interrupt_active_legacy_edit(
        cx: &mut TestAppContext,
    ) {
        init_test(cx);
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(path!("/dir"), json!({"file": "hello"}))
            .await;
        let project = Project::test(fs, [path!("/dir").as_ref()], cx).await;
        let action_log = cx.new(|_| ActionLog::new(project.clone()));
        let file_path = project
            .read_with(cx, |project, cx| project.find_project_path("dir/file", cx))
            .unwrap();
        let buffer = project
            .update(cx, |project, cx| project.open_buffer(file_path, cx))
            .await
            .unwrap();

        action_log
            .update(cx, |log, cx| log.acquire_edit_lsp_lease(buffer.clone(), cx))
            .await
            .unwrap();
        cx.update(|cx| {
            SettingsStore::update_global(cx, |store, cx| {
                store.update_user_settings(cx, |settings| {
                    settings
                        .agent
                        .get_or_insert_default()
                        .experimental_lsp_leases = Some(true);
                });
            });
        });
        cx.run_until_parked();
        assert_eq!(
            action_log.read_with(cx, |log, _cx| {
                log.lsp_lease_debug_counters().read_leases
            }),
            1,
            "enabling the experiment must not close LSP during an active legacy edit"
        );

        action_log.update(cx, |log, cx| log.finish_edit_lsp_lease(&buffer, cx));
        assert_eq!(
            action_log.read_with(cx, |log, _cx| {
                log.lsp_lease_debug_counters().read_leases
            }),
            0,
            "the retained legacy lease should be released when that edit finishes"
        );
    }
    use serde_json::json;
    use settings::SettingsStore;
    use std::env;
    use util::{RandomCharIter, path};

    #[test]
    fn test_agent_diff_load_thresholds() {
        assert!(!AgentDiffLoad::new(1, DiffComplexity::default()).is_large());
        assert!(
            AgentDiffLoad::new(LARGE_DIFF_FILE_COUNT + 1, DiffComplexity::default()).is_large()
        );
        assert!(
            AgentDiffLoad::new(
                1,
                DiffComplexity {
                    changed_rows: LARGE_DIFF_CHANGED_ROWS + 1,
                    ..DiffComplexity::default()
                }
            )
            .is_large()
        );
        assert!(
            AgentDiffLoad::new(
                1,
                DiffComplexity {
                    changed_bytes: LARGE_DIFF_CHANGED_BYTES + 1,
                    ..DiffComplexity::default()
                }
            )
            .is_large()
        );
        assert!(
            AgentDiffLoad::new(
                1,
                DiffComplexity {
                    hunk_count: LARGE_DIFF_HUNK_COUNT + 1,
                    ..DiffComplexity::default()
                }
            )
            .is_large()
        );
        assert!(
            AgentDiffLoad::new(
                1,
                DiffComplexity {
                    longest_changed_line_bytes: LARGE_DIFF_LONGEST_CHANGED_LINE_BYTES + 1,
                    ..DiffComplexity::default()
                }
            )
            .is_large()
        );
    }

    #[ctor::ctor(unsafe)]
    fn init_logger() {
        zlog::init_test();
    }

    fn init_test(cx: &mut TestAppContext) {
        cx.update(|cx| {
            let settings_store = SettingsStore::test(cx);
            cx.set_global(settings_store);
        });
    }

    #[gpui::test(iterations = 10)]
    async fn test_keep_edits(cx: &mut TestAppContext) {
        init_test(cx);

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(path!("/dir"), json!({"file": "abc\ndef\nghi\njkl\nmno"}))
            .await;
        let project = Project::test(fs.clone(), [path!("/dir").as_ref()], cx).await;
        let action_log = cx.new(|_| ActionLog::new(project.clone()));
        let file_path = project
            .read_with(cx, |project, cx| project.find_project_path("dir/file", cx))
            .unwrap();
        let buffer = project
            .update(cx, |project, cx| project.open_buffer(file_path, cx))
            .await
            .unwrap();

        cx.update(|cx| {
            action_log.update(cx, |log, cx| log.buffer_read(buffer.clone(), cx));
            buffer.update(cx, |buffer, cx| {
                buffer
                    .edit([(Point::new(1, 1)..Point::new(1, 2), "E")], None, cx)
                    .unwrap()
            });
            buffer.update(cx, |buffer, cx| {
                buffer
                    .edit([(Point::new(4, 2)..Point::new(4, 3), "O")], None, cx)
                    .unwrap()
            });
            action_log.update(cx, |log, cx| log.buffer_edited(buffer.clone(), cx));
        });
        cx.run_until_parked();
        assert_eq!(
            buffer.read_with(cx, |buffer, _| buffer.text()),
            "abc\ndEf\nghi\njkl\nmnO"
        );
        assert_eq!(
            unreviewed_hunks(&action_log, cx),
            vec![(
                buffer.clone(),
                vec![
                    HunkStatus {
                        range: Point::new(1, 0)..Point::new(2, 0),
                        diff_status: DiffHunkStatusKind::Modified,
                        old_text: "def\n".into(),
                    },
                    HunkStatus {
                        range: Point::new(4, 0)..Point::new(4, 3),
                        diff_status: DiffHunkStatusKind::Modified,
                        old_text: "mno".into(),
                    }
                ],
            )]
        );

        action_log.update(cx, |log, cx| {
            log.keep_edits_in_range(buffer.clone(), Point::new(3, 0)..Point::new(4, 3), None, cx)
        });
        cx.run_until_parked();
        assert_eq!(
            unreviewed_hunks(&action_log, cx),
            vec![(
                buffer.clone(),
                vec![HunkStatus {
                    range: Point::new(1, 0)..Point::new(2, 0),
                    diff_status: DiffHunkStatusKind::Modified,
                    old_text: "def\n".into(),
                }],
            )]
        );

        action_log.update(cx, |log, cx| {
            log.keep_edits_in_range(buffer.clone(), Point::new(0, 0)..Point::new(4, 3), None, cx)
        });
        cx.run_until_parked();
        assert_eq!(unreviewed_hunks(&action_log, cx), vec![]);
    }

    #[gpui::test]
    async fn test_undo_and_redo_keep_restore_review_state(cx: &mut TestAppContext) {
        init_test(cx);

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(path!("/dir"), json!({"file": "abc\ndef\nghi\njkl\nmno"}))
            .await;
        let project = Project::test(fs, [path!("/dir").as_ref()], cx).await;
        let action_log = cx.new(|_| ActionLog::new(project.clone()));
        let file_path = project
            .read_with(cx, |project, cx| project.find_project_path("dir/file", cx))
            .unwrap();
        let buffer = project
            .update(cx, |project, cx| project.open_buffer(file_path, cx))
            .await
            .unwrap();

        cx.update(|cx| {
            action_log.update(cx, |log, cx| log.buffer_read(buffer.clone(), cx));
            buffer.update(cx, |buffer, cx| {
                buffer.edit(
                    [
                        (Point::new(1, 0)..Point::new(1, 3), "DEF"),
                        (Point::new(4, 0)..Point::new(4, 3), "MNO"),
                    ],
                    None,
                    cx,
                )
            });
            action_log.update(cx, |log, cx| log.buffer_edited(buffer.clone(), cx));
        });
        cx.run_until_parked();
        assert_eq!(unreviewed_hunks(&action_log, cx)[0].1.len(), 2);

        action_log.update(cx, |log, cx| {
            log.keep_edits_in_range(buffer.clone(), Point::new(1, 0)..Point::new(2, 0), None, cx)
        });
        cx.run_until_parked();
        assert_eq!(unreviewed_hunks(&action_log, cx)[0].1.len(), 1);

        buffer.update(cx, |buffer, cx| buffer.undo(cx));
        cx.run_until_parked();
        assert_eq!(
            buffer.read_with(cx, |buffer, _| buffer.text()),
            "abc\nDEF\nghi\njkl\nMNO"
        );
        assert_eq!(unreviewed_hunks(&action_log, cx)[0].1.len(), 2);

        buffer.update(cx, |buffer, cx| buffer.redo(cx));
        cx.run_until_parked();
        assert_eq!(
            buffer.read_with(cx, |buffer, _| buffer.text()),
            "abc\nDEF\nghi\njkl\nMNO"
        );
        assert_eq!(unreviewed_hunks(&action_log, cx)[0].1.len(), 1);
    }

    #[gpui::test(iterations = 10)]
    async fn test_deletions(cx: &mut TestAppContext) {
        init_test(cx);

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            path!("/dir"),
            json!({"file": "abc\ndef\nghi\njkl\nmno\npqr"}),
        )
        .await;
        let project = Project::test(fs.clone(), [path!("/dir").as_ref()], cx).await;
        let action_log = cx.new(|_| ActionLog::new(project.clone()));
        let file_path = project
            .read_with(cx, |project, cx| project.find_project_path("dir/file", cx))
            .unwrap();
        let buffer = project
            .update(cx, |project, cx| project.open_buffer(file_path, cx))
            .await
            .unwrap();

        cx.update(|cx| {
            action_log.update(cx, |log, cx| log.buffer_read(buffer.clone(), cx));
            buffer.update(cx, |buffer, cx| {
                buffer
                    .edit([(Point::new(1, 0)..Point::new(2, 0), "")], None, cx)
                    .unwrap();
                buffer.finalize_last_transaction();
            });
            buffer.update(cx, |buffer, cx| {
                buffer
                    .edit([(Point::new(3, 0)..Point::new(4, 0), "")], None, cx)
                    .unwrap();
                buffer.finalize_last_transaction();
            });
            action_log.update(cx, |log, cx| log.buffer_edited(buffer.clone(), cx));
        });
        cx.run_until_parked();
        assert_eq!(
            buffer.read_with(cx, |buffer, _| buffer.text()),
            "abc\nghi\njkl\npqr"
        );
        assert_eq!(
            unreviewed_hunks(&action_log, cx),
            vec![(
                buffer.clone(),
                vec![
                    HunkStatus {
                        range: Point::new(1, 0)..Point::new(1, 0),
                        diff_status: DiffHunkStatusKind::Deleted,
                        old_text: "def\n".into(),
                    },
                    HunkStatus {
                        range: Point::new(3, 0)..Point::new(3, 0),
                        diff_status: DiffHunkStatusKind::Deleted,
                        old_text: "mno\n".into(),
                    }
                ],
            )]
        );

        buffer.update(cx, |buffer, cx| buffer.undo(cx));
        cx.run_until_parked();
        assert_eq!(
            buffer.read_with(cx, |buffer, _| buffer.text()),
            "abc\nghi\njkl\nmno\npqr"
        );
        assert_eq!(
            unreviewed_hunks(&action_log, cx),
            vec![(
                buffer.clone(),
                vec![HunkStatus {
                    range: Point::new(1, 0)..Point::new(1, 0),
                    diff_status: DiffHunkStatusKind::Deleted,
                    old_text: "def\n".into(),
                }],
            )]
        );

        action_log.update(cx, |log, cx| {
            log.keep_edits_in_range(buffer.clone(), Point::new(1, 0)..Point::new(1, 0), None, cx)
        });
        cx.run_until_parked();
        assert_eq!(unreviewed_hunks(&action_log, cx), vec![]);
    }

    #[gpui::test(iterations = 10)]
    async fn test_overlapping_user_edits(cx: &mut TestAppContext) {
        init_test(cx);

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(path!("/dir"), json!({"file": "abc\ndef\nghi\njkl\nmno"}))
            .await;
        let project = Project::test(fs.clone(), [path!("/dir").as_ref()], cx).await;
        let action_log = cx.new(|_| ActionLog::new(project.clone()));
        let file_path = project
            .read_with(cx, |project, cx| project.find_project_path("dir/file", cx))
            .unwrap();
        let buffer = project
            .update(cx, |project, cx| project.open_buffer(file_path, cx))
            .await
            .unwrap();

        cx.update(|cx| {
            action_log.update(cx, |log, cx| log.buffer_read(buffer.clone(), cx));
            buffer.update(cx, |buffer, cx| {
                buffer
                    .edit([(Point::new(1, 2)..Point::new(2, 3), "F\nGHI")], None, cx)
                    .unwrap()
            });
            action_log.update(cx, |log, cx| log.buffer_edited(buffer.clone(), cx));
        });
        cx.run_until_parked();
        assert_eq!(
            buffer.read_with(cx, |buffer, _| buffer.text()),
            "abc\ndeF\nGHI\njkl\nmno"
        );
        assert_eq!(
            unreviewed_hunks(&action_log, cx),
            vec![(
                buffer.clone(),
                vec![HunkStatus {
                    range: Point::new(1, 0)..Point::new(3, 0),
                    diff_status: DiffHunkStatusKind::Modified,
                    old_text: "def\nghi\n".into(),
                }],
            )]
        );

        buffer.update(cx, |buffer, cx| {
            buffer.edit(
                [
                    (Point::new(0, 2)..Point::new(0, 2), "X"),
                    (Point::new(3, 0)..Point::new(3, 0), "Y"),
                ],
                None,
                cx,
            )
        });
        cx.run_until_parked();
        assert_eq!(
            buffer.read_with(cx, |buffer, _| buffer.text()),
            "abXc\ndeF\nGHI\nYjkl\nmno"
        );
        assert_eq!(
            unreviewed_hunks(&action_log, cx),
            vec![(
                buffer.clone(),
                vec![HunkStatus {
                    range: Point::new(1, 0)..Point::new(3, 0),
                    diff_status: DiffHunkStatusKind::Modified,
                    old_text: "def\nghi\n".into(),
                }],
            )]
        );

        buffer.update(cx, |buffer, cx| {
            buffer.edit([(Point::new(1, 1)..Point::new(1, 1), "Z")], None, cx)
        });
        cx.run_until_parked();
        assert_eq!(
            buffer.read_with(cx, |buffer, _| buffer.text()),
            "abXc\ndZeF\nGHI\nYjkl\nmno"
        );
        assert_eq!(
            unreviewed_hunks(&action_log, cx),
            vec![(
                buffer.clone(),
                vec![HunkStatus {
                    range: Point::new(1, 0)..Point::new(3, 0),
                    diff_status: DiffHunkStatusKind::Modified,
                    old_text: "def\nghi\n".into(),
                }],
            )]
        );

        action_log.update(cx, |log, cx| {
            log.keep_edits_in_range(buffer.clone(), Point::new(0, 0)..Point::new(1, 0), None, cx)
        });
        cx.run_until_parked();
        assert_eq!(unreviewed_hunks(&action_log, cx), vec![]);
    }

    #[gpui::test(iterations = 10)]
    async fn test_creating_files(cx: &mut TestAppContext) {
        init_test(cx);

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(path!("/dir"), json!({})).await;
        let project = Project::test(fs.clone(), [path!("/dir").as_ref()], cx).await;
        let action_log = cx.new(|_| ActionLog::new(project.clone()));
        let file_path = project
            .read_with(cx, |project, cx| project.find_project_path("dir/file1", cx))
            .unwrap();

        let buffer = project
            .update(cx, |project, cx| project.open_buffer(file_path, cx))
            .await
            .unwrap();
        cx.update(|cx| {
            action_log.update(cx, |log, cx| log.buffer_created(buffer.clone(), cx));
            buffer.update(cx, |buffer, cx| buffer.set_text("lorem", cx));
            action_log.update(cx, |log, cx| log.buffer_edited(buffer.clone(), cx));
        });
        project
            .update(cx, |project, cx| project.save_buffer(buffer.clone(), cx))
            .await
            .unwrap();
        cx.run_until_parked();
        assert_eq!(
            unreviewed_hunks(&action_log, cx),
            vec![(
                buffer.clone(),
                vec![HunkStatus {
                    range: Point::new(0, 0)..Point::new(0, 5),
                    diff_status: DiffHunkStatusKind::Added,
                    old_text: "".into(),
                }],
            )]
        );

        buffer.update(cx, |buffer, cx| buffer.edit([(0..0, "X")], None, cx));
        cx.run_until_parked();
        assert_eq!(
            unreviewed_hunks(&action_log, cx),
            vec![(
                buffer.clone(),
                vec![HunkStatus {
                    range: Point::new(0, 0)..Point::new(0, 6),
                    diff_status: DiffHunkStatusKind::Added,
                    old_text: "".into(),
                }],
            )]
        );

        action_log.update(cx, |log, cx| {
            log.keep_edits_in_range(buffer.clone(), 0..5, None, cx)
        });
        cx.run_until_parked();
        assert_eq!(unreviewed_hunks(&action_log, cx), vec![]);
    }

    #[gpui::test(iterations = 10)]
    async fn test_overwriting_files(cx: &mut TestAppContext) {
        init_test(cx);

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            path!("/dir"),
            json!({
                "file1": "Lorem ipsum dolor"
            }),
        )
        .await;
        let project = Project::test(fs.clone(), [path!("/dir").as_ref()], cx).await;
        let action_log = cx.new(|_| ActionLog::new(project.clone()));
        let file_path = project
            .read_with(cx, |project, cx| project.find_project_path("dir/file1", cx))
            .unwrap();

        let buffer = project
            .update(cx, |project, cx| project.open_buffer(file_path, cx))
            .await
            .unwrap();
        cx.update(|cx| {
            action_log.update(cx, |log, cx| log.buffer_created(buffer.clone(), cx));
            buffer.update(cx, |buffer, cx| buffer.set_text("sit amet consecteur", cx));
            action_log.update(cx, |log, cx| log.buffer_edited(buffer.clone(), cx));
        });
        project
            .update(cx, |project, cx| project.save_buffer(buffer.clone(), cx))
            .await
            .unwrap();
        cx.run_until_parked();
        assert_eq!(
            unreviewed_hunks(&action_log, cx),
            vec![(
                buffer.clone(),
                vec![HunkStatus {
                    range: Point::new(0, 0)..Point::new(0, 19),
                    diff_status: DiffHunkStatusKind::Added,
                    old_text: "".into(),
                }],
            )]
        );

        action_log
            .update(cx, |log, cx| {
                let (task, _) = log.reject_edits_in_ranges(buffer.clone(), vec![2..5], None, cx);
                task
            })
            .await
            .unwrap();
        cx.run_until_parked();
        assert_eq!(unreviewed_hunks(&action_log, cx), vec![]);
        assert_eq!(
            buffer.read_with(cx, |buffer, _cx| buffer.text()),
            "Lorem ipsum dolor"
        );
    }

    #[gpui::test(iterations = 10)]
    async fn test_overwriting_previously_edited_files(cx: &mut TestAppContext) {
        init_test(cx);

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            path!("/dir"),
            json!({
                "file1": "Lorem ipsum dolor"
            }),
        )
        .await;
        let project = Project::test(fs.clone(), [path!("/dir").as_ref()], cx).await;
        let action_log = cx.new(|_| ActionLog::new(project.clone()));
        let file_path = project
            .read_with(cx, |project, cx| project.find_project_path("dir/file1", cx))
            .unwrap();

        let buffer = project
            .update(cx, |project, cx| project.open_buffer(file_path, cx))
            .await
            .unwrap();
        cx.update(|cx| {
            action_log.update(cx, |log, cx| log.buffer_read(buffer.clone(), cx));
            buffer.update(cx, |buffer, cx| buffer.append(" sit amet consecteur", cx));
            action_log.update(cx, |log, cx| log.buffer_edited(buffer.clone(), cx));
        });
        project
            .update(cx, |project, cx| project.save_buffer(buffer.clone(), cx))
            .await
            .unwrap();
        cx.run_until_parked();
        assert_eq!(
            unreviewed_hunks(&action_log, cx),
            vec![(
                buffer.clone(),
                vec![HunkStatus {
                    range: Point::new(0, 0)..Point::new(0, 37),
                    diff_status: DiffHunkStatusKind::Modified,
                    old_text: "Lorem ipsum dolor".into(),
                }],
            )]
        );

        cx.update(|cx| {
            action_log.update(cx, |log, cx| log.buffer_created(buffer.clone(), cx));
            buffer.update(cx, |buffer, cx| buffer.set_text("rewritten", cx));
            action_log.update(cx, |log, cx| log.buffer_edited(buffer.clone(), cx));
        });
        project
            .update(cx, |project, cx| project.save_buffer(buffer.clone(), cx))
            .await
            .unwrap();
        cx.run_until_parked();
        assert_eq!(
            unreviewed_hunks(&action_log, cx),
            vec![(
                buffer.clone(),
                vec![HunkStatus {
                    range: Point::new(0, 0)..Point::new(0, 9),
                    diff_status: DiffHunkStatusKind::Added,
                    old_text: "".into(),
                }],
            )]
        );

        action_log
            .update(cx, |log, cx| {
                let (task, _) = log.reject_edits_in_ranges(buffer.clone(), vec![2..5], None, cx);
                task
            })
            .await
            .unwrap();
        cx.run_until_parked();
        assert_eq!(unreviewed_hunks(&action_log, cx), vec![]);
        assert_eq!(
            buffer.read_with(cx, |buffer, _cx| buffer.text()),
            "Lorem ipsum dolor"
        );
    }

    #[gpui::test(iterations = 10)]
    async fn test_deleting_files(cx: &mut TestAppContext) {
        init_test(cx);

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            path!("/dir"),
            json!({"file1": "lorem\n", "file2": "ipsum\n"}),
        )
        .await;

        let project = Project::test(fs.clone(), [path!("/dir").as_ref()], cx).await;
        let file1_path = project
            .read_with(cx, |project, cx| project.find_project_path("dir/file1", cx))
            .unwrap();
        let file2_path = project
            .read_with(cx, |project, cx| project.find_project_path("dir/file2", cx))
            .unwrap();

        let action_log = cx.new(|_| ActionLog::new(project.clone()));
        let buffer1 = project
            .update(cx, |project, cx| {
                project.open_buffer(file1_path.clone(), cx)
            })
            .await
            .unwrap();
        let buffer2 = project
            .update(cx, |project, cx| {
                project.open_buffer(file2_path.clone(), cx)
            })
            .await
            .unwrap();

        action_log.update(cx, |log, cx| log.will_delete_buffer(buffer1.clone(), cx));
        action_log.update(cx, |log, cx| log.will_delete_buffer(buffer2.clone(), cx));
        project
            .update(cx, |project, cx| {
                project.delete_file(file1_path.clone(), cx)
            })
            .unwrap()
            .await
            .unwrap();
        project
            .update(cx, |project, cx| {
                project.delete_file(file2_path.clone(), cx)
            })
            .unwrap()
            .await
            .unwrap();
        cx.run_until_parked();
        assert_eq!(
            unreviewed_hunks(&action_log, cx),
            vec![
                (
                    buffer1.clone(),
                    vec![HunkStatus {
                        range: Point::new(0, 0)..Point::new(0, 0),
                        diff_status: DiffHunkStatusKind::Deleted,
                        old_text: "lorem\n".into(),
                    }]
                ),
                (
                    buffer2.clone(),
                    vec![HunkStatus {
                        range: Point::new(0, 0)..Point::new(0, 0),
                        diff_status: DiffHunkStatusKind::Deleted,
                        old_text: "ipsum\n".into(),
                    }],
                )
            ]
        );

        // Simulate file1 being recreated externally.
        fs.insert_file(path!("/dir/file1"), "LOREM".as_bytes().to_vec())
            .await;

        // Simulate file2 being recreated by a tool.
        let buffer2 = project
            .update(cx, |project, cx| project.open_buffer(file2_path, cx))
            .await
            .unwrap();
        action_log.update(cx, |log, cx| log.buffer_created(buffer2.clone(), cx));
        buffer2.update(cx, |buffer, cx| buffer.set_text("IPSUM", cx));
        action_log.update(cx, |log, cx| log.buffer_edited(buffer2.clone(), cx));
        project
            .update(cx, |project, cx| project.save_buffer(buffer2.clone(), cx))
            .await
            .unwrap();

        cx.run_until_parked();
        assert_eq!(
            unreviewed_hunks(&action_log, cx),
            vec![(
                buffer2.clone(),
                vec![HunkStatus {
                    range: Point::new(0, 0)..Point::new(0, 5),
                    diff_status: DiffHunkStatusKind::Added,
                    old_text: "".into(),
                }],
            )]
        );

        // Simulate file2 being deleted externally.
        fs.remove_file(path!("/dir/file2").as_ref(), RemoveOptions::default())
            .await
            .unwrap();
        cx.run_until_parked();
        assert_eq!(unreviewed_hunks(&action_log, cx), vec![]);
    }

    #[gpui::test(iterations = 10)]
    async fn test_reject_edits(cx: &mut TestAppContext) {
        init_test(cx);

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(path!("/dir"), json!({"file": "abc\ndef\nghi\njkl\nmno"}))
            .await;
        let project = Project::test(fs.clone(), [path!("/dir").as_ref()], cx).await;
        let action_log = cx.new(|_| ActionLog::new(project.clone()));
        let file_path = project
            .read_with(cx, |project, cx| project.find_project_path("dir/file", cx))
            .unwrap();
        let buffer = project
            .update(cx, |project, cx| project.open_buffer(file_path, cx))
            .await
            .unwrap();

        cx.update(|cx| {
            action_log.update(cx, |log, cx| log.buffer_read(buffer.clone(), cx));
            buffer.update(cx, |buffer, cx| {
                buffer
                    .edit([(Point::new(1, 1)..Point::new(1, 2), "E\nXYZ")], None, cx)
                    .unwrap()
            });
            buffer.update(cx, |buffer, cx| {
                buffer
                    .edit([(Point::new(5, 2)..Point::new(5, 3), "O")], None, cx)
                    .unwrap()
            });
            action_log.update(cx, |log, cx| log.buffer_edited(buffer.clone(), cx));
        });
        cx.run_until_parked();
        assert_eq!(
            buffer.read_with(cx, |buffer, _| buffer.text()),
            "abc\ndE\nXYZf\nghi\njkl\nmnO"
        );
        assert_eq!(
            unreviewed_hunks(&action_log, cx),
            vec![(
                buffer.clone(),
                vec![
                    HunkStatus {
                        range: Point::new(1, 0)..Point::new(3, 0),
                        diff_status: DiffHunkStatusKind::Modified,
                        old_text: "def\n".into(),
                    },
                    HunkStatus {
                        range: Point::new(5, 0)..Point::new(5, 3),
                        diff_status: DiffHunkStatusKind::Modified,
                        old_text: "mno".into(),
                    }
                ],
            )]
        );

        // If the rejected range doesn't overlap with any hunk, we ignore it.
        action_log
            .update(cx, |log, cx| {
                let (task, _) = log.reject_edits_in_ranges(
                    buffer.clone(),
                    vec![Point::new(4, 0)..Point::new(4, 0)],
                    None,
                    cx,
                );
                task
            })
            .await
            .unwrap();
        cx.run_until_parked();
        assert_eq!(
            buffer.read_with(cx, |buffer, _| buffer.text()),
            "abc\ndE\nXYZf\nghi\njkl\nmnO"
        );
        assert_eq!(
            unreviewed_hunks(&action_log, cx),
            vec![(
                buffer.clone(),
                vec![
                    HunkStatus {
                        range: Point::new(1, 0)..Point::new(3, 0),
                        diff_status: DiffHunkStatusKind::Modified,
                        old_text: "def\n".into(),
                    },
                    HunkStatus {
                        range: Point::new(5, 0)..Point::new(5, 3),
                        diff_status: DiffHunkStatusKind::Modified,
                        old_text: "mno".into(),
                    }
                ],
            )]
        );

        action_log
            .update(cx, |log, cx| {
                let (task, _) = log.reject_edits_in_ranges(
                    buffer.clone(),
                    vec![Point::new(0, 0)..Point::new(1, 0)],
                    None,
                    cx,
                );
                task
            })
            .await
            .unwrap();
        cx.run_until_parked();
        assert_eq!(
            buffer.read_with(cx, |buffer, _| buffer.text()),
            "abc\ndef\nghi\njkl\nmnO"
        );
        assert_eq!(
            unreviewed_hunks(&action_log, cx),
            vec![(
                buffer.clone(),
                vec![HunkStatus {
                    range: Point::new(4, 0)..Point::new(4, 3),
                    diff_status: DiffHunkStatusKind::Modified,
                    old_text: "mno".into(),
                }],
            )]
        );

        action_log
            .update(cx, |log, cx| {
                let (task, _) = log.reject_edits_in_ranges(
                    buffer.clone(),
                    vec![Point::new(4, 0)..Point::new(4, 0)],
                    None,
                    cx,
                );
                task
            })
            .await
            .unwrap();
        cx.run_until_parked();
        assert_eq!(
            buffer.read_with(cx, |buffer, _| buffer.text()),
            "abc\ndef\nghi\njkl\nmno"
        );
        assert_eq!(unreviewed_hunks(&action_log, cx), vec![]);
    }

    #[gpui::test]
    async fn test_undo_and_redo_reject_restore_text_and_review_state(cx: &mut TestAppContext) {
        init_test(cx);

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(path!("/dir"), json!({"file": "abc\ndef\nghi\njkl\nmno"}))
            .await;
        let project = Project::test(fs, [path!("/dir").as_ref()], cx).await;
        let action_log = cx.new(|_| ActionLog::new(project.clone()));
        let file_path = project
            .read_with(cx, |project, cx| project.find_project_path("dir/file", cx))
            .unwrap();
        let buffer = project
            .update(cx, |project, cx| project.open_buffer(file_path, cx))
            .await
            .unwrap();

        cx.update(|cx| {
            action_log.update(cx, |log, cx| log.buffer_read(buffer.clone(), cx));
            buffer.update(cx, |buffer, cx| {
                buffer.edit(
                    [
                        (Point::new(1, 0)..Point::new(1, 3), "DEF"),
                        (Point::new(4, 0)..Point::new(4, 3), "MNO"),
                    ],
                    None,
                    cx,
                )
            });
            action_log.update(cx, |log, cx| log.buffer_edited(buffer.clone(), cx));
        });
        cx.run_until_parked();
        assert_eq!(unreviewed_hunks(&action_log, cx)[0].1.len(), 2);

        action_log
            .update(cx, |log, cx| {
                let (task, _) = log.reject_edits_in_ranges(
                    buffer.clone(),
                    vec![Point::new(1, 0)..Point::new(2, 0)],
                    None,
                    cx,
                );
                task
            })
            .await
            .unwrap();
        cx.run_until_parked();
        assert_eq!(
            buffer.read_with(cx, |buffer, _| buffer.text()),
            "abc\ndef\nghi\njkl\nMNO"
        );
        assert_eq!(unreviewed_hunks(&action_log, cx)[0].1.len(), 1);

        buffer.update(cx, |buffer, cx| buffer.undo(cx));
        cx.run_until_parked();
        assert_eq!(
            buffer.read_with(cx, |buffer, _| buffer.text()),
            "abc\nDEF\nghi\njkl\nMNO"
        );
        assert_eq!(unreviewed_hunks(&action_log, cx)[0].1.len(), 2);

        buffer.update(cx, |buffer, cx| buffer.redo(cx));
        cx.run_until_parked();
        assert_eq!(
            buffer.read_with(cx, |buffer, _| buffer.text()),
            "abc\ndef\nghi\njkl\nMNO"
        );
        assert_eq!(unreviewed_hunks(&action_log, cx)[0].1.len(), 1);
    }

    #[gpui::test(iterations = 10)]
    async fn test_reject_multiple_edits(cx: &mut TestAppContext) {
        init_test(cx);

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(path!("/dir"), json!({"file": "abc\ndef\nghi\njkl\nmno"}))
            .await;
        let project = Project::test(fs.clone(), [path!("/dir").as_ref()], cx).await;
        let action_log = cx.new(|_| ActionLog::new(project.clone()));
        let file_path = project
            .read_with(cx, |project, cx| project.find_project_path("dir/file", cx))
            .unwrap();
        let buffer = project
            .update(cx, |project, cx| project.open_buffer(file_path, cx))
            .await
            .unwrap();

        cx.update(|cx| {
            action_log.update(cx, |log, cx| log.buffer_read(buffer.clone(), cx));
            buffer.update(cx, |buffer, cx| {
                buffer
                    .edit([(Point::new(1, 1)..Point::new(1, 2), "E\nXYZ")], None, cx)
                    .unwrap()
            });
            buffer.update(cx, |buffer, cx| {
                buffer
                    .edit([(Point::new(5, 2)..Point::new(5, 3), "O")], None, cx)
                    .unwrap()
            });
            action_log.update(cx, |log, cx| log.buffer_edited(buffer.clone(), cx));
        });
        cx.run_until_parked();
        assert_eq!(
            buffer.read_with(cx, |buffer, _| buffer.text()),
            "abc\ndE\nXYZf\nghi\njkl\nmnO"
        );
        assert_eq!(
            unreviewed_hunks(&action_log, cx),
            vec![(
                buffer.clone(),
                vec![
                    HunkStatus {
                        range: Point::new(1, 0)..Point::new(3, 0),
                        diff_status: DiffHunkStatusKind::Modified,
                        old_text: "def\n".into(),
                    },
                    HunkStatus {
                        range: Point::new(5, 0)..Point::new(5, 3),
                        diff_status: DiffHunkStatusKind::Modified,
                        old_text: "mno".into(),
                    }
                ],
            )]
        );

        action_log.update(cx, |log, cx| {
            let range_1 = buffer.read(cx).anchor_before(Point::new(0, 0))
                ..buffer.read(cx).anchor_before(Point::new(1, 0));
            let range_2 = buffer.read(cx).anchor_before(Point::new(5, 0))
                ..buffer.read(cx).anchor_before(Point::new(5, 3));

            let (task, _) =
                log.reject_edits_in_ranges(buffer.clone(), vec![range_1, range_2], None, cx);
            task.detach();
            assert_eq!(
                buffer.read_with(cx, |buffer, _| buffer.text()),
                "abc\ndef\nghi\njkl\nmno"
            );
        });
        cx.run_until_parked();
        assert_eq!(
            buffer.read_with(cx, |buffer, _| buffer.text()),
            "abc\ndef\nghi\njkl\nmno"
        );
        assert_eq!(unreviewed_hunks(&action_log, cx), vec![]);
    }

    #[gpui::test(iterations = 10)]
    async fn test_reject_deleted_file(cx: &mut TestAppContext) {
        init_test(cx);

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(path!("/dir"), json!({"file": "content"}))
            .await;
        let project = Project::test(fs.clone(), [path!("/dir").as_ref()], cx).await;
        let action_log = cx.new(|_| ActionLog::new(project.clone()));
        let file_path = project
            .read_with(cx, |project, cx| project.find_project_path("dir/file", cx))
            .unwrap();
        let buffer = project
            .update(cx, |project, cx| project.open_buffer(file_path.clone(), cx))
            .await
            .unwrap();

        cx.update(|cx| {
            action_log.update(cx, |log, cx| log.will_delete_buffer(buffer.clone(), cx));
        });
        project
            .update(cx, |project, cx| project.delete_file(file_path.clone(), cx))
            .unwrap()
            .await
            .unwrap();
        cx.run_until_parked();
        assert!(!fs.is_file(path!("/dir/file").as_ref()).await);
        assert_eq!(
            unreviewed_hunks(&action_log, cx),
            vec![(
                buffer.clone(),
                vec![HunkStatus {
                    range: Point::new(0, 0)..Point::new(0, 0),
                    diff_status: DiffHunkStatusKind::Deleted,
                    old_text: "content".into(),
                }]
            )]
        );

        action_log
            .update(cx, |log, cx| {
                let (task, _) = log.reject_edits_in_ranges(
                    buffer.clone(),
                    vec![Point::new(0, 0)..Point::new(0, 0)],
                    None,
                    cx,
                );
                task
            })
            .await
            .unwrap();
        cx.run_until_parked();
        assert_eq!(buffer.read_with(cx, |buffer, _| buffer.text()), "content");
        assert!(fs.is_file(path!("/dir/file").as_ref()).await);
        assert_eq!(unreviewed_hunks(&action_log, cx), vec![]);
    }

    #[gpui::test(iterations = 10)]
    async fn test_reject_created_file(cx: &mut TestAppContext) {
        init_test(cx);

        let fs = FakeFs::new(cx.executor());
        let project = Project::test(fs.clone(), [path!("/dir").as_ref()], cx).await;
        let action_log = cx.new(|_| ActionLog::new(project.clone()));
        let file_path = project
            .read_with(cx, |project, cx| {
                project.find_project_path("dir/new_file", cx)
            })
            .unwrap();
        let buffer = project
            .update(cx, |project, cx| project.open_buffer(file_path, cx))
            .await
            .unwrap();
        cx.update(|cx| {
            action_log.update(cx, |log, cx| log.buffer_created(buffer.clone(), cx));
            buffer.update(cx, |buffer, cx| buffer.set_text("content", cx));
            action_log.update(cx, |log, cx| log.buffer_edited(buffer.clone(), cx));
        });
        project
            .update(cx, |project, cx| project.save_buffer(buffer.clone(), cx))
            .await
            .unwrap();
        assert!(fs.is_file(path!("/dir/new_file").as_ref()).await);
        cx.run_until_parked();
        assert_eq!(
            unreviewed_hunks(&action_log, cx),
            vec![(
                buffer.clone(),
                vec![HunkStatus {
                    range: Point::new(0, 0)..Point::new(0, 7),
                    diff_status: DiffHunkStatusKind::Added,
                    old_text: "".into(),
                }],
            )]
        );

        action_log
            .update(cx, |log, cx| {
                let (task, _) = log.reject_edits_in_ranges(
                    buffer.clone(),
                    vec![Point::new(0, 0)..Point::new(0, 11)],
                    None,
                    cx,
                );
                task
            })
            .await
            .unwrap();
        cx.run_until_parked();
        assert!(!fs.is_file(path!("/dir/new_file").as_ref()).await);
        assert_eq!(unreviewed_hunks(&action_log, cx), vec![]);
    }

    #[gpui::test]
    async fn test_reject_created_file_with_user_edits(cx: &mut TestAppContext) {
        init_test(cx);

        let fs = FakeFs::new(cx.executor());
        let project = Project::test(fs.clone(), [path!("/dir").as_ref()], cx).await;
        let action_log = cx.new(|_| ActionLog::new(project.clone()));

        let file_path = project
            .read_with(cx, |project, cx| {
                project.find_project_path("dir/new_file", cx)
            })
            .unwrap();
        let buffer = project
            .update(cx, |project, cx| project.open_buffer(file_path, cx))
            .await
            .unwrap();

        // AI creates file with initial content
        cx.update(|cx| {
            action_log.update(cx, |log, cx| log.buffer_created(buffer.clone(), cx));
            buffer.update(cx, |buffer, cx| buffer.set_text("ai content", cx));
            action_log.update(cx, |log, cx| log.buffer_edited(buffer.clone(), cx));
        });

        project
            .update(cx, |project, cx| project.save_buffer(buffer.clone(), cx))
            .await
            .unwrap();

        cx.run_until_parked();

        // User makes additional edits
        cx.update(|cx| {
            buffer.update(cx, |buffer, cx| {
                buffer.edit([(10..10, "\nuser added this line")], None, cx);
            });
        });

        project
            .update(cx, |project, cx| project.save_buffer(buffer.clone(), cx))
            .await
            .unwrap();

        assert!(fs.is_file(path!("/dir/new_file").as_ref()).await);

        // Reject all
        action_log
            .update(cx, |log, cx| {
                let (task, _) = log.reject_edits_in_ranges(
                    buffer.clone(),
                    vec![Point::new(0, 0)..Point::new(100, 0)],
                    None,
                    cx,
                );
                task
            })
            .await
            .unwrap();
        cx.run_until_parked();

        // File should still contain all the content
        assert!(fs.is_file(path!("/dir/new_file").as_ref()).await);

        let content = buffer.read_with(cx, |buffer, _| buffer.text());
        assert_eq!(content, "ai content\nuser added this line");
    }

    #[gpui::test]
    async fn test_reject_after_accepting_hunk_on_created_file(cx: &mut TestAppContext) {
        init_test(cx);

        let fs = FakeFs::new(cx.executor());
        let project = Project::test(fs.clone(), [path!("/dir").as_ref()], cx).await;
        let action_log = cx.new(|_| ActionLog::new(project.clone()));

        let file_path = project
            .read_with(cx, |project, cx| {
                project.find_project_path("dir/new_file", cx)
            })
            .unwrap();
        let buffer = project
            .update(cx, |project, cx| project.open_buffer(file_path.clone(), cx))
            .await
            .unwrap();

        // AI creates file with initial content
        cx.update(|cx| {
            action_log.update(cx, |log, cx| log.buffer_created(buffer.clone(), cx));
            buffer.update(cx, |buffer, cx| buffer.set_text("ai content v1", cx));
            action_log.update(cx, |log, cx| log.buffer_edited(buffer.clone(), cx));
        });
        project
            .update(cx, |project, cx| project.save_buffer(buffer.clone(), cx))
            .await
            .unwrap();
        cx.run_until_parked();
        assert_ne!(unreviewed_hunks(&action_log, cx), vec![]);

        // User accepts the single hunk
        action_log.update(cx, |log, cx| {
            let buffer_range = Anchor::min_max_range_for_buffer(buffer.read(cx).remote_id());
            log.keep_edits_in_range(buffer.clone(), buffer_range, None, cx)
        });
        cx.run_until_parked();
        assert_eq!(unreviewed_hunks(&action_log, cx), vec![]);
        assert!(fs.is_file(path!("/dir/new_file").as_ref()).await);

        // AI modifies the file
        cx.update(|cx| {
            buffer.update(cx, |buffer, cx| buffer.set_text("ai content v2", cx));
            action_log.update(cx, |log, cx| log.buffer_edited(buffer.clone(), cx));
        });
        project
            .update(cx, |project, cx| project.save_buffer(buffer.clone(), cx))
            .await
            .unwrap();
        cx.run_until_parked();
        assert_ne!(unreviewed_hunks(&action_log, cx), vec![]);

        // User rejects the hunk
        action_log
            .update(cx, |log, cx| {
                let (task, _) = log.reject_edits_in_ranges(
                    buffer.clone(),
                    vec![Anchor::min_max_range_for_buffer(
                        buffer.read(cx).remote_id(),
                    )],
                    None,
                    cx,
                );
                task
            })
            .await
            .unwrap();
        cx.run_until_parked();
        assert!(fs.is_file(path!("/dir/new_file").as_ref()).await,);
        assert_eq!(
            buffer.read_with(cx, |buffer, _| buffer.text()),
            "ai content v1"
        );
        assert_eq!(unreviewed_hunks(&action_log, cx), vec![]);
    }

    #[gpui::test]
    async fn test_reject_edits_on_previously_accepted_created_file(cx: &mut TestAppContext) {
        init_test(cx);

        let fs = FakeFs::new(cx.executor());
        let project = Project::test(fs.clone(), [path!("/dir").as_ref()], cx).await;
        let action_log = cx.new(|_| ActionLog::new(project.clone()));

        let file_path = project
            .read_with(cx, |project, cx| {
                project.find_project_path("dir/new_file", cx)
            })
            .unwrap();
        let buffer = project
            .update(cx, |project, cx| project.open_buffer(file_path.clone(), cx))
            .await
            .unwrap();

        // AI creates file with initial content
        cx.update(|cx| {
            action_log.update(cx, |log, cx| log.buffer_created(buffer.clone(), cx));
            buffer.update(cx, |buffer, cx| buffer.set_text("ai content v1", cx));
            action_log.update(cx, |log, cx| log.buffer_edited(buffer.clone(), cx));
        });
        project
            .update(cx, |project, cx| project.save_buffer(buffer.clone(), cx))
            .await
            .unwrap();
        cx.run_until_parked();

        // User clicks "Accept All"
        action_log.update(cx, |log, cx| log.keep_all_edits(None, cx));
        cx.run_until_parked();
        assert!(fs.is_file(path!("/dir/new_file").as_ref()).await);
        assert_eq!(unreviewed_hunks(&action_log, cx), vec![]); // Hunks are cleared

        // AI modifies file again
        cx.update(|cx| {
            buffer.update(cx, |buffer, cx| buffer.set_text("ai content v2", cx));
            action_log.update(cx, |log, cx| log.buffer_edited(buffer.clone(), cx));
        });
        project
            .update(cx, |project, cx| project.save_buffer(buffer.clone(), cx))
            .await
            .unwrap();
        cx.run_until_parked();
        assert_ne!(unreviewed_hunks(&action_log, cx), vec![]);

        // User clicks "Reject All"
        action_log
            .update(cx, |log, cx| log.reject_all_edits(None, cx))
            .await;
        cx.run_until_parked();
        assert!(fs.is_file(path!("/dir/new_file").as_ref()).await);
        assert_eq!(
            buffer.read_with(cx, |buffer, _| buffer.text()),
            "ai content v1"
        );
        assert_eq!(unreviewed_hunks(&action_log, cx), vec![]);
    }

    #[gpui::test(iterations = 100)]
    async fn test_random_diffs(mut rng: StdRng, cx: &mut TestAppContext) {
        init_test(cx);

        let operations = env::var("OPERATIONS")
            .map(|i| i.parse().expect("invalid `OPERATIONS` variable"))
            .unwrap_or(20);

        let text = RandomCharIter::new(&mut rng).take(50).collect::<String>();
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(path!("/dir"), json!({"file": text})).await;
        let project = Project::test(fs.clone(), [path!("/dir").as_ref()], cx).await;
        let action_log = cx.new(|_| ActionLog::new(project.clone()));
        let file_path = project
            .read_with(cx, |project, cx| project.find_project_path("dir/file", cx))
            .unwrap();
        let buffer = project
            .update(cx, |project, cx| project.open_buffer(file_path, cx))
            .await
            .unwrap();

        action_log.update(cx, |log, cx| log.buffer_read(buffer.clone(), cx));

        for _ in 0..operations {
            match rng.random_range(0..100) {
                0..25 => {
                    action_log.update(cx, |log, cx| {
                        let range = buffer.read(cx).random_byte_range(0, &mut rng);
                        log::info!("keeping edits in range {:?}", range);
                        log.keep_edits_in_range(buffer.clone(), range, None, cx)
                    });
                }
                25..50 => {
                    action_log
                        .update(cx, |log, cx| {
                            let range = buffer.read(cx).random_byte_range(0, &mut rng);
                            log::info!("rejecting edits in range {:?}", range);
                            let (task, _) =
                                log.reject_edits_in_ranges(buffer.clone(), vec![range], None, cx);
                            task
                        })
                        .await
                        .unwrap();
                }
                _ => {
                    let is_agent_edit = rng.random_bool(0.5);
                    if is_agent_edit {
                        log::info!("agent edit");
                    } else {
                        log::info!("user edit");
                    }
                    cx.update(|cx| {
                        buffer.update(cx, |buffer, cx| buffer.randomly_edit(&mut rng, 1, cx));
                        if is_agent_edit {
                            action_log.update(cx, |log, cx| log.buffer_edited(buffer.clone(), cx));
                        }
                    });
                }
            }

            if rng.random_bool(0.2) {
                quiesce(&action_log, &buffer, cx);
            }
        }

        quiesce(&action_log, &buffer, cx);

        fn quiesce(
            action_log: &Entity<ActionLog>,
            buffer: &Entity<Buffer>,
            cx: &mut TestAppContext,
        ) {
            log::info!("quiescing...");
            cx.run_until_parked();
            action_log.update(cx, |log, cx| {
                let tracked_buffer = log.tracked_buffers.get(buffer).unwrap();
                let mut old_text = tracked_buffer.diff_base.clone();
                let new_text = buffer.read(cx).as_rope();
                for edit in tracked_buffer.unreviewed_edits.edits() {
                    let old_start = old_text.point_to_offset(Point::new(edit.new.start, 0));
                    let old_end = old_text.point_to_offset(cmp::min(
                        Point::new(edit.new.start + edit.old_len(), 0),
                        old_text.max_point(),
                    ));
                    old_text.replace(
                        old_start..old_end,
                        &new_text.slice_rows(edit.new.clone()).to_string(),
                    );
                }
                pretty_assertions::assert_eq!(old_text.to_string(), new_text.to_string());
            })
        }
    }

    #[gpui::test]
    async fn test_keep_edits_on_commit(cx: &mut gpui::TestAppContext) {
        init_test(cx);

        let fs = FakeFs::new(cx.background_executor.clone());
        fs.insert_tree(
            path!("/project"),
            json!({
                ".git": {},
                "file.txt": "a\nb\nc\nd\ne\nf\ng\nh\ni\nj",
            }),
        )
        .await;
        fs.set_head_for_repo(
            path!("/project/.git").as_ref(),
            &[("file.txt", "a\nb\nc\nd\ne\nf\ng\nh\ni\nj".into())],
            "0000000",
        );
        cx.run_until_parked();

        let project = Project::test(fs.clone(), [path!("/project").as_ref()], cx).await;
        let action_log = cx.new(|_| ActionLog::new(project.clone()));

        let file_path = project
            .read_with(cx, |project, cx| {
                project.find_project_path(path!("/project/file.txt"), cx)
            })
            .unwrap();
        let buffer = project
            .update(cx, |project, cx| project.open_buffer(file_path, cx))
            .await
            .unwrap();

        cx.update(|cx| {
            action_log.update(cx, |log, cx| log.buffer_read(buffer.clone(), cx));
            buffer.update(cx, |buffer, cx| {
                buffer.edit(
                    [
                        // Edit at the very start: a -> A
                        (Point::new(0, 0)..Point::new(0, 1), "A"),
                        // Deletion in the middle: remove lines d and e
                        (Point::new(3, 0)..Point::new(5, 0), ""),
                        // Modification: g -> GGG
                        (Point::new(6, 0)..Point::new(6, 1), "GGG"),
                        // Addition: insert new line after h
                        (Point::new(7, 1)..Point::new(7, 1), "\nNEW"),
                        // Edit the very last character: j -> J
                        (Point::new(9, 0)..Point::new(9, 1), "J"),
                    ],
                    None,
                    cx,
                );
            });
            action_log.update(cx, |log, cx| log.buffer_edited(buffer.clone(), cx));
        });
        cx.run_until_parked();
        assert_eq!(
            unreviewed_hunks(&action_log, cx),
            vec![(
                buffer.clone(),
                vec![
                    HunkStatus {
                        range: Point::new(0, 0)..Point::new(1, 0),
                        diff_status: DiffHunkStatusKind::Modified,
                        old_text: "a\n".into()
                    },
                    HunkStatus {
                        range: Point::new(3, 0)..Point::new(3, 0),
                        diff_status: DiffHunkStatusKind::Deleted,
                        old_text: "d\ne\n".into()
                    },
                    HunkStatus {
                        range: Point::new(4, 0)..Point::new(5, 0),
                        diff_status: DiffHunkStatusKind::Modified,
                        old_text: "g\n".into()
                    },
                    HunkStatus {
                        range: Point::new(6, 0)..Point::new(7, 0),
                        diff_status: DiffHunkStatusKind::Added,
                        old_text: "".into()
                    },
                    HunkStatus {
                        range: Point::new(8, 0)..Point::new(8, 1),
                        diff_status: DiffHunkStatusKind::Modified,
                        old_text: "j".into()
                    }
                ]
            )]
        );

        // Simulate a git commit that matches some edits but not others:
        // - Accepts the first edit (a -> A)
        // - Accepts the deletion (remove d and e)
        // - Makes a different change to g (g -> G instead of GGG)
        // - Ignores the NEW line addition
        // - Ignores the last line edit (j stays as j)
        fs.set_head_for_repo(
            path!("/project/.git").as_ref(),
            &[("file.txt", "A\nb\nc\nf\nG\nh\ni\nj".into())],
            "0000001",
        );
        cx.run_until_parked();
        assert_eq!(
            unreviewed_hunks(&action_log, cx),
            vec![(
                buffer.clone(),
                vec![
                    HunkStatus {
                        range: Point::new(4, 0)..Point::new(5, 0),
                        diff_status: DiffHunkStatusKind::Modified,
                        old_text: "g\n".into()
                    },
                    HunkStatus {
                        range: Point::new(6, 0)..Point::new(7, 0),
                        diff_status: DiffHunkStatusKind::Added,
                        old_text: "".into()
                    },
                    HunkStatus {
                        range: Point::new(8, 0)..Point::new(8, 1),
                        diff_status: DiffHunkStatusKind::Modified,
                        old_text: "j".into()
                    }
                ]
            )]
        );

        // Make another commit that accepts the NEW line but with different content
        fs.set_head_for_repo(
            path!("/project/.git").as_ref(),
            &[("file.txt", "A\nb\nc\nf\nGGG\nh\nDIFFERENT\ni\nj".into())],
            "0000002",
        );
        cx.run_until_parked();
        assert_eq!(
            unreviewed_hunks(&action_log, cx),
            vec![(
                buffer,
                vec![
                    HunkStatus {
                        range: Point::new(6, 0)..Point::new(7, 0),
                        diff_status: DiffHunkStatusKind::Added,
                        old_text: "".into()
                    },
                    HunkStatus {
                        range: Point::new(8, 0)..Point::new(8, 1),
                        diff_status: DiffHunkStatusKind::Modified,
                        old_text: "j".into()
                    }
                ]
            )]
        );

        // Final commit that accepts all remaining edits
        fs.set_head_for_repo(
            path!("/project/.git").as_ref(),
            &[("file.txt", "A\nb\nc\nf\nGGG\nh\nNEW\ni\nJ".into())],
            "0000003",
        );
        cx.run_until_parked();
        assert_eq!(unreviewed_hunks(&action_log, cx), vec![]);
    }

    #[gpui::test]
    async fn test_keep_edits_on_commit_with_shifted_diff_boundaries(cx: &mut TestAppContext) {
        init_test(cx);

        let initial_text = indoc! {"
            use crate::{Alpha, Beta};

            fn keep() {
                work();
            }

            fn remove() {
                work();
            }

            fn after() {
                work();
            }
        "};
        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            path!("/project"),
            json!({
                ".git": {},
                "file.rs": initial_text,
            }),
        )
        .await;
        fs.set_head_for_repo(
            path!("/project/.git").as_ref(),
            &[("file.rs", initial_text.into())],
            "0000000",
        );
        cx.run_until_parked();

        let project = Project::test(fs.clone(), [path!("/project").as_ref()], cx).await;
        let action_log = cx.new(|_| ActionLog::new(project.clone()));

        let file_path = project
            .read_with(cx, |project, cx| {
                project.find_project_path(path!("/project/file.rs"), cx)
            })
            .unwrap();
        let buffer = project
            .update(cx, |project, cx| project.open_buffer(file_path, cx))
            .await
            .unwrap();

        let final_text = indoc! {"
            use crate::{Alpha};

            fn keep() {
                work();
            }

            fn after() {
                work();
            }
        "};

        cx.update(|cx| {
            action_log.update(cx, |log, cx| log.buffer_read(buffer.clone(), cx));
            buffer.update(cx, |buffer, cx| {
                buffer.set_text(final_text, cx);
            });
            action_log.update(cx, |log, cx| log.buffer_edited(buffer.clone(), cx));
        });
        cx.run_until_parked();
        assert!(!unreviewed_hunks(&action_log, cx).is_empty());

        fs.set_head_for_repo(
            path!("/project/.git").as_ref(),
            &[("file.rs", final_text.into())],
            "0000001",
        );
        cx.run_until_parked();

        assert_eq!(unreviewed_hunks(&action_log, cx), vec![]);
    }

    /// Regression test: when head_commit updates before the BufferDiff's base
    /// text does, an intermediate DiffChanged (e.g. from a buffer-edit diff
    /// recalculation) must NOT consume the commit signal.  The subscription
    /// should only fire once the base text itself has changed.
    #[gpui::test]
    async fn test_keep_edits_on_commit_with_stale_diff_changed(cx: &mut TestAppContext) {
        init_test(cx);

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            path!("/project"),
            json!({
                ".git": {},
                "file.txt": "aaa\nbbb\nccc\nddd\neee",
            }),
        )
        .await;
        fs.set_head_for_repo(
            path!("/project/.git").as_ref(),
            &[("file.txt", "aaa\nbbb\nccc\nddd\neee".into())],
            "0000000",
        );
        cx.run_until_parked();

        let project = Project::test(fs.clone(), [path!("/project").as_ref()], cx).await;
        let action_log = cx.new(|_| ActionLog::new(project.clone()));

        let file_path = project
            .read_with(cx, |project, cx| {
                project.find_project_path(path!("/project/file.txt"), cx)
            })
            .unwrap();
        let buffer = project
            .update(cx, |project, cx| project.open_buffer(file_path, cx))
            .await
            .unwrap();

        // Agent makes an edit: bbb -> BBB
        cx.update(|cx| {
            action_log.update(cx, |log, cx| log.buffer_read(buffer.clone(), cx));
            buffer.update(cx, |buffer, cx| {
                buffer.edit([(Point::new(1, 0)..Point::new(1, 3), "BBB")], None, cx);
            });
            action_log.update(cx, |log, cx| log.buffer_edited(buffer.clone(), cx));
        });
        cx.run_until_parked();

        // Verify the edit is tracked
        let hunks = unreviewed_hunks(&action_log, cx);
        assert_eq!(hunks.len(), 1);
        let hunk = &hunks[0].1;
        assert_eq!(hunk.len(), 1);
        assert_eq!(hunk[0].old_text, "bbb\n");

        // Simulate the race condition: update only the HEAD SHA first,
        // without changing the committed file contents. This is analogous
        // to compute_snapshot updating head_commit before
        // reload_buffer_diff_bases has loaded the new base text.
        fs.with_git_state(path!("/project/.git").as_ref(), true, |state| {
            state.refs.insert("HEAD".into(), "0000001".into());
        })
        .unwrap();
        cx.run_until_parked();

        // Make a user edit (on a different line) to trigger a buffer diff
        // recalculation.  This fires DiffChanged while the BufferDiff base
        // text is still the OLD text.  With the old head_commit-based
        // subscription this would "consume" the commit detection.
        cx.update(|cx| {
            buffer.update(cx, |buffer, cx| {
                buffer.edit([(Point::new(3, 0)..Point::new(3, 3), "DDD")], None, cx);
            });
            action_log.update(cx, |log, cx| log.buffer_edited(buffer.clone(), cx));
        });
        cx.run_until_parked();

        // Now update the committed file contents to match the buffer
        // (the agent edit was committed). Keep the same SHA so head_commit
        // does NOT change again — this is the second half of the race.
        {
            use git::repository::repo_path;
            fs.with_git_state(path!("/project/.git").as_ref(), true, |state| {
                state
                    .head_contents
                    .insert(repo_path("file.txt"), "aaa\nBBB\nccc\nDDD\neee".into());
            })
            .unwrap();
        }
        cx.run_until_parked();

        // The agent's edit (bbb -> BBB) should be accepted because the
        // committed content now matches. Only the user edit (ddd -> DDD)
        // should remain, but since the user edit is tracked as coming from
        // the user (ChangeAuthor::User) it would have been rebased into
        // the diff base already. So no unreviewed hunks should remain.
        assert_eq!(
            unreviewed_hunks(&action_log, cx),
            vec![],
            "agent edits should have been accepted after the base text update"
        );
    }

    #[gpui::test]
    async fn test_undo_last_reject(cx: &mut TestAppContext) {
        init_test(cx);

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            path!("/dir"),
            json!({
                "file1": "abc\ndef\nghi"
            }),
        )
        .await;
        let project = Project::test(fs.clone(), [path!("/dir").as_ref()], cx).await;
        let action_log = cx.new(|_| ActionLog::new(project.clone()));
        let file_path = project
            .read_with(cx, |project, cx| project.find_project_path("dir/file1", cx))
            .unwrap();

        let buffer = project
            .update(cx, |project, cx| project.open_buffer(file_path, cx))
            .await
            .unwrap();

        // Track the buffer and make an agent edit
        cx.update(|cx| {
            action_log.update(cx, |log, cx| log.buffer_read(buffer.clone(), cx));
            buffer.update(cx, |buffer, cx| {
                buffer
                    .edit(
                        [(Point::new(1, 0)..Point::new(1, 3), "AGENT_EDIT")],
                        None,
                        cx,
                    )
                    .unwrap()
            });
            action_log.update(cx, |log, cx| log.buffer_edited(buffer.clone(), cx));
        });
        cx.run_until_parked();

        // Verify the agent edit is there
        assert_eq!(
            buffer.read_with(cx, |buffer, _| buffer.text()),
            "abc\nAGENT_EDIT\nghi"
        );
        assert!(!unreviewed_hunks(&action_log, cx).is_empty());

        // Reject all edits
        action_log
            .update(cx, |log, cx| log.reject_all_edits(None, cx))
            .await;
        cx.run_until_parked();

        // Verify the buffer is back to original
        assert_eq!(
            buffer.read_with(cx, |buffer, _| buffer.text()),
            "abc\ndef\nghi"
        );
        assert!(unreviewed_hunks(&action_log, cx).is_empty());

        // Verify undo state is available
        assert!(action_log.read_with(cx, |log, _| log.has_pending_undo()));

        // Undo the reject
        action_log
            .update(cx, |log, cx| log.undo_last_reject(cx))
            .await;

        cx.run_until_parked();

        // Verify the agent edit is restored
        assert_eq!(
            buffer.read_with(cx, |buffer, _| buffer.text()),
            "abc\nAGENT_EDIT\nghi"
        );
        assert!(!unreviewed_hunks(&action_log, cx).is_empty());

        // Verify undo state is cleared
        assert!(!action_log.read_with(cx, |log, _| log.has_pending_undo()));
    }

    #[gpui::test]
    async fn test_linked_action_log_buffer_read(cx: &mut TestAppContext) {
        init_test(cx);

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(path!("/dir"), json!({"file": "hello world"}))
            .await;
        let project = Project::test(fs.clone(), [path!("/dir").as_ref()], cx).await;
        let parent_log = cx.new(|_| ActionLog::new(project.clone()));
        let child_log =
            cx.new(|_| ActionLog::new(project.clone()).with_linked_action_log(parent_log.clone()));

        let file_path = project
            .read_with(cx, |project, cx| project.find_project_path("dir/file", cx))
            .unwrap();
        let buffer = project
            .update(cx, |project, cx| project.open_buffer(file_path, cx))
            .await
            .unwrap();

        cx.update(|cx| {
            child_log.update(cx, |log, cx| log.buffer_read(buffer.clone(), cx));
        });

        // Neither log considers the buffer stale immediately after reading it.
        let child_stale = cx.read(|cx| {
            child_log
                .read(cx)
                .stale_buffers(cx)
                .cloned()
                .collect::<Vec<_>>()
        });
        let parent_stale = cx.read(|cx| {
            parent_log
                .read(cx)
                .stale_buffers(cx)
                .cloned()
                .collect::<Vec<_>>()
        });
        assert!(child_stale.is_empty());
        assert!(parent_stale.is_empty());

        // Simulate a user edit after the agent read the file.
        cx.update(|cx| {
            buffer.update(cx, |buffer, cx| {
                buffer.edit([(0..5, "goodbye")], None, cx).unwrap();
            });
        });
        cx.run_until_parked();

        // Both child and parent should see the buffer as stale because both tracked
        // it at the pre-edit version via buffer_read forwarding.
        let child_stale = cx.read(|cx| {
            child_log
                .read(cx)
                .stale_buffers(cx)
                .cloned()
                .collect::<Vec<_>>()
        });
        let parent_stale = cx.read(|cx| {
            parent_log
                .read(cx)
                .stale_buffers(cx)
                .cloned()
                .collect::<Vec<_>>()
        });
        assert_eq!(child_stale, vec![buffer.clone()]);
        assert_eq!(parent_stale, vec![buffer]);
    }

    #[gpui::test]
    async fn test_linked_action_log_buffer_edited(cx: &mut TestAppContext) {
        init_test(cx);

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(path!("/dir"), json!({"file": "abc\ndef\nghi"}))
            .await;
        let project = Project::test(fs.clone(), [path!("/dir").as_ref()], cx).await;
        let parent_log = cx.new(|_| ActionLog::new(project.clone()));
        let child_log =
            cx.new(|_| ActionLog::new(project.clone()).with_linked_action_log(parent_log.clone()));

        let file_path = project
            .read_with(cx, |project, cx| project.find_project_path("dir/file", cx))
            .unwrap();
        let buffer = project
            .update(cx, |project, cx| project.open_buffer(file_path, cx))
            .await
            .unwrap();

        cx.update(|cx| {
            child_log.update(cx, |log, cx| log.buffer_read(buffer.clone(), cx));
            buffer.update(cx, |buffer, cx| {
                buffer
                    .edit([(Point::new(1, 0)..Point::new(1, 3), "DEF")], None, cx)
                    .unwrap();
            });
            child_log.update(cx, |log, cx| log.buffer_edited(buffer.clone(), cx));
        });
        cx.run_until_parked();

        let expected_hunks = vec![(
            buffer,
            vec![HunkStatus {
                range: Point::new(1, 0)..Point::new(2, 0),
                diff_status: DiffHunkStatusKind::Modified,
                old_text: "def\n".into(),
            }],
        )];
        assert_eq!(
            unreviewed_hunks(&child_log, cx),
            expected_hunks,
            "child should track the agent edit"
        );
        assert_eq!(
            unreviewed_hunks(&parent_log, cx),
            expected_hunks,
            "parent should also track the agent edit via linked log forwarding"
        );
    }

    #[gpui::test]
    async fn test_linked_action_log_buffer_created(cx: &mut TestAppContext) {
        init_test(cx);

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(path!("/dir"), json!({})).await;
        let project = Project::test(fs.clone(), [path!("/dir").as_ref()], cx).await;
        let parent_log = cx.new(|_| ActionLog::new(project.clone()));
        let child_log =
            cx.new(|_| ActionLog::new(project.clone()).with_linked_action_log(parent_log.clone()));

        let file_path = project
            .read_with(cx, |project, cx| {
                project.find_project_path("dir/new_file", cx)
            })
            .unwrap();
        let buffer = project
            .update(cx, |project, cx| project.open_buffer(file_path, cx))
            .await
            .unwrap();

        cx.update(|cx| {
            child_log.update(cx, |log, cx| log.buffer_created(buffer.clone(), cx));
            buffer.update(cx, |buffer, cx| buffer.set_text("hello", cx));
            child_log.update(cx, |log, cx| log.buffer_edited(buffer.clone(), cx));
        });
        project
            .update(cx, |project, cx| project.save_buffer(buffer.clone(), cx))
            .await
            .unwrap();
        cx.run_until_parked();

        let expected_hunks = vec![(
            buffer.clone(),
            vec![HunkStatus {
                range: Point::new(0, 0)..Point::new(0, 5),
                diff_status: DiffHunkStatusKind::Added,
                old_text: "".into(),
            }],
        )];
        assert_eq!(
            unreviewed_hunks(&child_log, cx),
            expected_hunks,
            "child should track the created file"
        );
        assert_eq!(
            unreviewed_hunks(&parent_log, cx),
            expected_hunks,
            "parent should also track the created file via linked log forwarding"
        );
    }

    #[gpui::test]
    async fn test_linked_action_log_will_delete_buffer(cx: &mut TestAppContext) {
        init_test(cx);

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(path!("/dir"), json!({"file": "hello\n"}))
            .await;
        let project = Project::test(fs.clone(), [path!("/dir").as_ref()], cx).await;
        let parent_log = cx.new(|_| ActionLog::new(project.clone()));
        let child_log =
            cx.new(|_| ActionLog::new(project.clone()).with_linked_action_log(parent_log.clone()));

        let file_path = project
            .read_with(cx, |project, cx| project.find_project_path("dir/file", cx))
            .unwrap();
        let buffer = project
            .update(cx, |project, cx| project.open_buffer(file_path.clone(), cx))
            .await
            .unwrap();

        cx.update(|cx| {
            child_log.update(cx, |log, cx| log.will_delete_buffer(buffer.clone(), cx));
        });
        project
            .update(cx, |project, cx| project.delete_file(file_path, cx))
            .unwrap()
            .await
            .unwrap();
        cx.run_until_parked();

        let expected_hunks = vec![(
            buffer.clone(),
            vec![HunkStatus {
                range: Point::new(0, 0)..Point::new(0, 0),
                diff_status: DiffHunkStatusKind::Deleted,
                old_text: "hello\n".into(),
            }],
        )];
        assert_eq!(
            unreviewed_hunks(&child_log, cx),
            expected_hunks,
            "child should track the deleted file"
        );
        assert_eq!(
            unreviewed_hunks(&parent_log, cx),
            expected_hunks,
            "parent should also track the deleted file via linked log forwarding"
        );
    }

    /// Simulates the subagent scenario: two child logs linked to the same parent, each
    /// editing a different file. The parent accumulates all edits while each child
    /// only sees its own.
    #[gpui::test]
    async fn test_linked_action_log_independent_tracking(cx: &mut TestAppContext) {
        init_test(cx);

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            path!("/dir"),
            json!({
                "file_a": "content of a",
                "file_b": "content of b",
            }),
        )
        .await;
        let project = Project::test(fs.clone(), [path!("/dir").as_ref()], cx).await;
        let parent_log = cx.new(|_| ActionLog::new(project.clone()));
        let child_log_1 =
            cx.new(|_| ActionLog::new(project.clone()).with_linked_action_log(parent_log.clone()));
        let child_log_2 =
            cx.new(|_| ActionLog::new(project.clone()).with_linked_action_log(parent_log.clone()));

        let file_a_path = project
            .read_with(cx, |project, cx| {
                project.find_project_path("dir/file_a", cx)
            })
            .unwrap();
        let file_b_path = project
            .read_with(cx, |project, cx| {
                project.find_project_path("dir/file_b", cx)
            })
            .unwrap();
        let buffer_a = project
            .update(cx, |project, cx| project.open_buffer(file_a_path, cx))
            .await
            .unwrap();
        let buffer_b = project
            .update(cx, |project, cx| project.open_buffer(file_b_path, cx))
            .await
            .unwrap();

        cx.update(|cx| {
            child_log_1.update(cx, |log, cx| log.buffer_read(buffer_a.clone(), cx));
            buffer_a.update(cx, |buffer, cx| {
                buffer.edit([(0..0, "MODIFIED: ")], None, cx).unwrap();
            });
            child_log_1.update(cx, |log, cx| log.buffer_edited(buffer_a.clone(), cx));

            child_log_2.update(cx, |log, cx| log.buffer_read(buffer_b.clone(), cx));
            buffer_b.update(cx, |buffer, cx| {
                buffer.edit([(0..0, "MODIFIED: ")], None, cx).unwrap();
            });
            child_log_2.update(cx, |log, cx| log.buffer_edited(buffer_b.clone(), cx));
        });
        cx.run_until_parked();

        let child_1_changed: Vec<_> = cx.read(|cx| {
            child_log_1
                .read(cx)
                .changed_buffers(cx)
                .map(|(buffer, _)| buffer)
                .collect()
        });
        let child_2_changed: Vec<_> = cx.read(|cx| {
            child_log_2
                .read(cx)
                .changed_buffers(cx)
                .map(|(buffer, _)| buffer)
                .collect()
        });
        let parent_changed: Vec<_> = cx.read(|cx| {
            parent_log
                .read(cx)
                .changed_buffers(cx)
                .map(|(buffer, _)| buffer)
                .collect()
        });

        assert_eq!(
            child_1_changed,
            vec![buffer_a.clone()],
            "child 1 should only track file_a"
        );
        assert_eq!(
            child_2_changed,
            vec![buffer_b.clone()],
            "child 2 should only track file_b"
        );
        assert_eq!(parent_changed.len(), 2, "parent should track both files");
        assert!(
            parent_changed.contains(&buffer_a) && parent_changed.contains(&buffer_b),
            "parent should contain both buffer_a and buffer_b"
        );
    }

    #[gpui::test]
    async fn test_file_read_time_recorded_on_buffer_read(cx: &mut TestAppContext) {
        init_test(cx);

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(path!("/dir"), json!({"file": "hello world"}))
            .await;
        let project = Project::test(fs.clone(), [path!("/dir").as_ref()], cx).await;
        let action_log = cx.new(|_| ActionLog::new(project.clone()));

        let file_path = project
            .read_with(cx, |project, cx| project.find_project_path("dir/file", cx))
            .unwrap();
        let buffer = project
            .update(cx, |project, cx| project.open_buffer(file_path, cx))
            .await
            .unwrap();

        let abs_path = PathBuf::from(path!("/dir/file"));
        assert!(
            action_log.read_with(cx, |log, _| log.file_read_time(&abs_path).is_none()),
            "file_read_time should be None before buffer_read"
        );

        cx.update(|cx| {
            action_log.update(cx, |log, cx| log.buffer_read(buffer.clone(), cx));
        });

        assert!(
            action_log.read_with(cx, |log, _| log.file_read_time(&abs_path).is_some()),
            "file_read_time should be recorded after buffer_read"
        );
    }

    #[gpui::test]
    async fn test_file_read_time_recorded_on_buffer_edited(cx: &mut TestAppContext) {
        init_test(cx);

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(path!("/dir"), json!({"file": "hello world"}))
            .await;
        let project = Project::test(fs.clone(), [path!("/dir").as_ref()], cx).await;
        let action_log = cx.new(|_| ActionLog::new(project.clone()));

        let file_path = project
            .read_with(cx, |project, cx| project.find_project_path("dir/file", cx))
            .unwrap();
        let buffer = project
            .update(cx, |project, cx| project.open_buffer(file_path, cx))
            .await
            .unwrap();

        let abs_path = PathBuf::from(path!("/dir/file"));
        assert!(
            action_log.read_with(cx, |log, _| log.file_read_time(&abs_path).is_none()),
            "file_read_time should be None before buffer_edited"
        );

        cx.update(|cx| {
            action_log.update(cx, |log, cx| log.buffer_edited(buffer.clone(), cx));
        });

        assert!(
            action_log.read_with(cx, |log, _| log.file_read_time(&abs_path).is_some()),
            "file_read_time should be recorded after buffer_edited"
        );
    }

    #[gpui::test]
    async fn test_file_read_time_recorded_on_buffer_created(cx: &mut TestAppContext) {
        init_test(cx);

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(path!("/dir"), json!({"file": "existing content"}))
            .await;
        let project = Project::test(fs.clone(), [path!("/dir").as_ref()], cx).await;
        let action_log = cx.new(|_| ActionLog::new(project.clone()));

        let file_path = project
            .read_with(cx, |project, cx| project.find_project_path("dir/file", cx))
            .unwrap();
        let buffer = project
            .update(cx, |project, cx| project.open_buffer(file_path, cx))
            .await
            .unwrap();

        let abs_path = PathBuf::from(path!("/dir/file"));
        assert!(
            action_log.read_with(cx, |log, _| log.file_read_time(&abs_path).is_none()),
            "file_read_time should be None before buffer_created"
        );

        cx.update(|cx| {
            action_log.update(cx, |log, cx| log.buffer_created(buffer.clone(), cx));
        });

        assert!(
            action_log.read_with(cx, |log, _| log.file_read_time(&abs_path).is_some()),
            "file_read_time should be recorded after buffer_created"
        );
    }

    #[gpui::test]
    async fn test_file_read_time_removed_on_delete(cx: &mut TestAppContext) {
        init_test(cx);

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(path!("/dir"), json!({"file": "hello world"}))
            .await;
        let project = Project::test(fs.clone(), [path!("/dir").as_ref()], cx).await;
        let action_log = cx.new(|_| ActionLog::new(project.clone()));

        let file_path = project
            .read_with(cx, |project, cx| project.find_project_path("dir/file", cx))
            .unwrap();
        let buffer = project
            .update(cx, |project, cx| project.open_buffer(file_path, cx))
            .await
            .unwrap();

        let abs_path = PathBuf::from(path!("/dir/file"));

        cx.update(|cx| {
            action_log.update(cx, |log, cx| log.buffer_read(buffer.clone(), cx));
        });
        assert!(
            action_log.read_with(cx, |log, _| log.file_read_time(&abs_path).is_some()),
            "file_read_time should exist after buffer_read"
        );

        cx.update(|cx| {
            action_log.update(cx, |log, cx| log.will_delete_buffer(buffer.clone(), cx));
        });
        assert!(
            action_log.read_with(cx, |log, _| log.file_read_time(&abs_path).is_none()),
            "file_read_time should be removed after will_delete_buffer"
        );
    }

    #[gpui::test]
    async fn test_file_read_time_not_forwarded_to_linked_action_log(cx: &mut TestAppContext) {
        init_test(cx);

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(path!("/dir"), json!({"file": "hello world"}))
            .await;
        let project = Project::test(fs.clone(), [path!("/dir").as_ref()], cx).await;
        let parent_log = cx.new(|_| ActionLog::new(project.clone()));
        let child_log =
            cx.new(|_| ActionLog::new(project.clone()).with_linked_action_log(parent_log.clone()));

        let file_path = project
            .read_with(cx, |project, cx| project.find_project_path("dir/file", cx))
            .unwrap();
        let buffer = project
            .update(cx, |project, cx| project.open_buffer(file_path, cx))
            .await
            .unwrap();

        let abs_path = PathBuf::from(path!("/dir/file"));

        cx.update(|cx| {
            child_log.update(cx, |log, cx| log.buffer_read(buffer.clone(), cx));
        });
        assert!(
            child_log.read_with(cx, |log, _| log.file_read_time(&abs_path).is_some()),
            "child should record file_read_time on buffer_read"
        );
        assert!(
            parent_log.read_with(cx, |log, _| log.file_read_time(&abs_path).is_none()),
            "parent should NOT get file_read_time from child's buffer_read"
        );

        cx.update(|cx| {
            child_log.update(cx, |log, cx| log.buffer_edited(buffer.clone(), cx));
        });
        assert!(
            parent_log.read_with(cx, |log, _| log.file_read_time(&abs_path).is_none()),
            "parent should NOT get file_read_time from child's buffer_edited"
        );

        cx.update(|cx| {
            child_log.update(cx, |log, cx| log.buffer_created(buffer.clone(), cx));
        });
        assert!(
            parent_log.read_with(cx, |log, _| log.file_read_time(&abs_path).is_none()),
            "parent should NOT get file_read_time from child's buffer_created"
        );
    }

    #[derive(Debug, PartialEq)]
    struct HunkStatus {
        range: Range<Point>,
        diff_status: DiffHunkStatusKind,
        old_text: String,
    }

    fn unreviewed_hunks(
        action_log: &Entity<ActionLog>,
        cx: &TestAppContext,
    ) -> Vec<(Entity<Buffer>, Vec<HunkStatus>)> {
        cx.read(|cx| {
            action_log
                .read(cx)
                .changed_buffers(cx)
                .map(|(buffer, diff)| {
                    let snapshot = buffer.read(cx).snapshot();
                    (
                        buffer,
                        diff.read(cx)
                            .snapshot(cx)
                            .hunks(&snapshot)
                            .map(|hunk| HunkStatus {
                                diff_status: hunk.status().kind,
                                range: hunk.range,
                                old_text: diff
                                    .read(cx)
                                    .base_text(cx)
                                    .text_for_range(hunk.diff_base_byte_range)
                                    .collect(),
                            })
                            .collect(),
                    )
                })
                .collect()
        })
    }
}
