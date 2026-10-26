mod patch;

use agent_client_protocol::schema::v2 as acp_v2;
use anyhow::{Result, anyhow, ensure};
use buffer_diff::BufferDiff;
use collections::{HashMap, HashSet};
use gpui::{
    App, AppContext, AsyncApp, Context, Entity, SharedString, Subscription, Task, WeakEntity,
};
use itertools::Itertools;
use language::{
    Anchor, Buffer, Capability, LanguageRegistry, OffsetRangeExt as _, Point, TextBuffer,
};
use markdown::Markdown;
use multi_buffer::{MultiBuffer, PathKey, excerpt_context_lines};
use std::{
    cell::OnceCell,
    cmp::Reverse,
    ops::Range,
    path::{Path, PathBuf},
    sync::Arc,
};
use util::ResultExt;

#[derive(Debug)]
pub struct DiffPatch {
    pub files: Vec<DiffPatchFile>,
    pub fallback: Option<Entity<Markdown>>,
}

#[derive(Debug)]
pub struct DiffPatchFile {
    pub change_index: usize,
    pub hunks: Vec<DiffPatchHunk>,
}

#[derive(Debug)]
pub struct DiffPatchHunk {
    pub header: SharedString,
    pub buffer: Entity<MultiBuffer>,
    _update_diff: Task<()>,
}

impl DiffPatch {
    pub(crate) fn new(
        source: &acp_v2::Diff,
        language_registry: &Arc<LanguageRegistry>,
        cx: &mut App,
    ) -> Self {
        let mut files = source
            .changes
            .iter()
            .enumerate()
            .map(|(change_index, _)| DiffPatchFile {
                change_index,
                hunks: Vec::new(),
            })
            .collect::<Vec<_>>();
        if source.patch.is_none() {
            return Self {
                files,
                fallback: None,
            };
        }

        match Self::parse(source) {
            Ok(parsed) => {
                for (change_index, file) in parsed {
                    let path = file.new_path.or(file.old_path).unwrap_or_default();
                    if let Some(render) = files.get_mut(change_index) {
                        render.hunks = file
                            .hunks
                            .into_iter()
                            .map(|hunk| {
                                DiffPatchHunk::new(hunk, &path, language_registry.clone(), cx)
                            })
                            .collect();
                    }
                }
                Self {
                    files,
                    fallback: None,
                }
            }
            Err(error) => {
                let preview = format!(
                    "Diff preview unavailable: {error}\n\n{}",
                    crate::ToolCallContent::patch_preview(source)
                );
                Self {
                    files,
                    fallback: Some(crate::ContentBlock::create_markdown(
                        preview,
                        language_registry,
                        cx,
                    )),
                }
            }
        }
    }

    fn parse(source: &acp_v2::Diff) -> Result<Vec<(usize, patch::PatchFile)>> {
        let patch = source
            .patch
            .as_ref()
            .ok_or_else(|| anyhow!("no patch supplied"))?;
        ensure!(
            patch.format == acp_v2::DiffPatchFormat::GitPatch,
            "unsupported patch format"
        );
        ensure!(
            !source.changes.is_empty(),
            "patch has no reported file changes"
        );
        let mut change_indices: HashMap<_, Option<usize>> = HashMap::default();
        for (index, change) in source.changes.iter().enumerate() {
            if let Some(paths) = diff_change_paths(change) {
                change_indices
                    .entry(paths)
                    .and_modify(|index| *index = None)
                    .or_insert(Some(index));
            }
        }
        let mut seen = HashSet::default();
        patch::parse_patch(&patch.text)?
            .into_iter()
            .map(|mut file| {
                let paths = (
                    file.old_path.as_deref().map(Path::new),
                    file.new_path.as_deref().map(Path::new),
                );
                let anonymous = paths == (None, None);
                let paths = if anonymous {
                    let [change] = source.changes.as_slice() else {
                        return Err(anyhow!(
                            "patch without filenames requires exactly one file change"
                        ));
                    };
                    diff_change_paths(change)
                        .ok_or_else(|| anyhow!("unsupported file operation"))?
                } else {
                    paths
                };
                let index = change_indices
                    .get(&paths)
                    .copied()
                    .flatten()
                    .ok_or_else(|| {
                        anyhow!(
                            "patch paths are ambiguous or do not match the reported file changes"
                        )
                    })?;
                ensure!(seen.insert(index), "repeated patch file");
                let change = source
                    .changes
                    .get(index)
                    .ok_or_else(|| anyhow!("missing file change"))?;
                ensure!(
                    matches!(change.file_type, None | Some(acp_v2::DiffFileType::Text)),
                    "text patch supplied for a non-text file"
                );
                if anonymous && let Some((old, new)) = diff_change_paths(change) {
                    file.old_path = old.map(|path| path.to_string_lossy().into_owned());
                    file.new_path = new.map(|path| path.to_string_lossy().into_owned());
                }
                ensure!(
                    file.hunks.iter().all(|hunk| {
                        (file.old_path.is_some() || hunk.old_count == 0)
                            && (file.new_path.is_some() || hunk.new_count == 0)
                    }),
                    "hunk contents conflict with the reported file operation"
                );
                Ok((index, file))
            })
            .collect()
    }
}

impl DiffPatchHunk {
    fn new(
        hunk: patch::PatchHunk,
        path: &str,
        language_registry: Arc<LanguageRegistry>,
        cx: &mut App,
    ) -> Self {
        let range_label = |start, count| {
            if count == 1 {
                format!("{start}")
            } else {
                format!("{start},{count}")
            }
        };
        let header = format!(
            "@@ -{} +{} @@",
            range_label(hunk.old_start, hunk.old_count),
            range_label(hunk.new_start, hunk.new_count),
        )
        .into();
        let new_buffer = cx.new(|cx| {
            let mut buffer = Buffer::local(hunk.new_text, cx);
            buffer.set_capability(Capability::ReadOnly, cx);
            buffer
        });
        let multibuffer = cx.new(|cx| {
            let mut multibuffer = MultiBuffer::without_headers(Capability::ReadOnly);
            multibuffer.set_excerpts_for_path(
                PathKey::for_buffer(&new_buffer, cx),
                new_buffer.clone(),
                [Point::new(0, 0)..new_buffer.read(cx).max_point()],
                0,
                cx,
            );
            multibuffer
        });
        let task = cx.spawn({
            let multibuffer = multibuffer.clone();
            let path = path.to_owned();
            async move |cx| {
                let language = language_registry
                    .load_language_for_file_path(Path::new(&path))
                    .await
                    .log_err();
                new_buffer.update(cx, |buffer, cx| buffer.set_language(language.clone(), cx));
                let snapshot = new_buffer.read_with(cx, |buffer, _| buffer.snapshot());
                let diff = cx.new(|cx| {
                    // A hunk is only a snippet: it must never offer restore/apply operations.
                    BufferDiff::new(&snapshot, language, Some(language_registry), cx)
                });
                diff.update(cx, |diff, cx| {
                    diff.set_base_text(Some(hunk.old_text.into()), snapshot.text, cx)
                })
                .await;
                multibuffer.update(cx, |buffer, cx| {
                    buffer.add_diff(diff, cx);
                    buffer.set_all_diff_hunks_expanded(cx);
                });
            }
        });
        Self {
            header,
            buffer: multibuffer,
            _update_diff: task,
        }
    }
}

fn diff_change_paths(change: &acp_v2::DiffChange) -> Option<(Option<&Path>, Option<&Path>)> {
    match &change.operation {
        acp_v2::DiffChangeOperation::Add(change) => Some((None, Some(&change.path.0))),
        acp_v2::DiffChangeOperation::Delete(change) => Some((Some(&change.path.0), None)),
        acp_v2::DiffChangeOperation::Modify(change) => {
            Some((Some(&change.path.0), Some(&change.path.0)))
        }
        acp_v2::DiffChangeOperation::Move(change) | acp_v2::DiffChangeOperation::Copy(change) => {
            Some((Some(&change.old_path.0), Some(&change.path.0)))
        }
        _ => None,
    }
}

pub fn diff_change_label(change: &acp_v2::DiffChange) -> String {
    match &change.operation {
        acp_v2::DiffChangeOperation::Add(change) => format!("Added {}", change.path.0.display()),
        acp_v2::DiffChangeOperation::Delete(change) => {
            format!("Deleted {}", change.path.0.display())
        }
        acp_v2::DiffChangeOperation::Modify(change) => {
            format!("Modified {}", change.path.0.display())
        }
        acp_v2::DiffChangeOperation::Move(change) => format!(
            "Moved {} → {}",
            change.old_path.0.display(),
            change.path.0.display()
        ),
        acp_v2::DiffChangeOperation::Copy(change) => format!(
            "Copied {} → {}",
            change.old_path.0.display(),
            change.path.0.display()
        ),
        _ => "Unsupported file operation".to_owned(),
    }
}

pub enum Diff {
    Pending(PendingDiff),
    Finalized(FinalizedDiff),
}

impl Diff {
    pub fn finalized(
        path: String,
        old_text: Option<String>,
        new_text: String,
        language_registry: Arc<LanguageRegistry>,
        cx: &mut Context<Self>,
    ) -> Self {
        Self::Finalized(FinalizedDiff {
            multibuffer: cx.new(|_| MultiBuffer::without_headers(Capability::ReadOnly)),
            path,
            base_text_exists: old_text.is_some(),
            base_text: old_text.unwrap_or_default().into(),
            new_text: new_text.into(),
            language_registry: Some(language_registry),
            resolved: OnceCell::new(),
        })
    }

    pub(crate) fn resolve(&self, cx: &mut Context<Self>) {
        let Self::Finalized(finalized) = self else {
            return;
        };
        // Keep snapshots available to inline review without parsing history
        // until its diff card is opened. The empty multibuffer preserves the
        // fork's accessor API without a panic before resolution.
        finalized.resolved.get_or_init(|| {
            let new_buffer = cx.new(|cx| Buffer::local(finalized.new_text.to_string(), cx));
            let new_buffer_handle = new_buffer.downgrade();
            let base_text_exists = finalized.base_text_exists;
            let base_text = finalized.base_text.clone();
            let language_registry = finalized.language_registry.clone();
            let multibuffer = finalized.multibuffer.clone();
            let path = finalized.path.clone();
            let task = cx.spawn({
                let buffer = new_buffer;
                async move |_, cx| {
                    let language = match language_registry {
                        Some(language_registry) => language_registry
                            .load_language_for_file_path(Path::new(&path))
                            .await
                            .log_err(),
                        None => None,
                    };

                    buffer.update(cx, |buffer, cx| buffer.set_language(language, cx));
                    buffer.update(cx, |buffer, _| buffer.parsing_idle()).await;

                    let diff = build_buffer_diff(base_text, base_text_exists, &buffer, cx).await?;

                    multibuffer.update(cx, |multibuffer, cx| {
                        let hunk_ranges = {
                            let buffer = buffer.read(cx);
                            diff.read(cx)
                                .snapshot(cx)
                                .hunks_intersecting_range(
                                    Anchor::min_for_buffer(buffer.remote_id())
                                        ..Anchor::max_for_buffer(buffer.remote_id()),
                                    buffer,
                                )
                                .map(|diff_hunk| diff_hunk.buffer_range.to_point(buffer))
                                .collect::<Vec<_>>()
                        };

                        multibuffer.set_excerpts_for_path(
                            PathKey::for_buffer(&buffer, cx),
                            buffer.clone(),
                            hunk_ranges,
                            excerpt_context_lines(cx),
                            cx,
                        );
                        multibuffer.add_diff(diff, cx);
                    });

                    anyhow::Ok(())
                }
            });
            ResolvedFinalizedDiff {
                new_buffer: new_buffer_handle,
                _update_diff: task,
            }
        });
    }

    pub fn new(buffer: Entity<Buffer>, cx: &mut Context<Self>) -> Self {
        let buffer_text_snapshot = buffer.read(cx).text_snapshot();
        let language = buffer.read(cx).language().cloned();
        let language_registry = buffer.read(cx).language_registry();
        let buffer_diff = cx.new(|cx| {
            BufferDiff::new_unchanged(&buffer_text_snapshot, language, language_registry, cx)
        });

        let multibuffer = cx.new(|cx| {
            let mut multibuffer = MultiBuffer::without_headers(Capability::ReadOnly);
            multibuffer.add_diff(buffer_diff.clone(), cx);
            multibuffer
        });

        Self::Pending(PendingDiff {
            multibuffer,
            base_text: Arc::from(buffer_text_snapshot.text().as_str()),
            _subscription: cx.observe(&buffer, |this, _, cx| {
                if let Diff::Pending(diff) = this {
                    diff.update(cx);
                }
            }),
            new_buffer: buffer,
            diff: buffer_diff,
            revealed_ranges: Vec::new(),
            update_diff: Task::ready(Ok(())),
        })
    }

    pub fn reveal_range(&mut self, range: Range<Anchor>, cx: &mut Context<Self>) {
        if let Self::Pending(diff) = self {
            diff.reveal_range(range, cx);
        }
    }

    pub fn finalize(&mut self, cx: &mut Context<Self>) {
        if let Self::Pending(diff) = self {
            *self = Self::Finalized(diff.finalize(cx));
        }
    }

    /// Returns the original text before any edits were applied.
    pub fn base_text(&self) -> &Arc<str> {
        match self {
            Self::Pending(PendingDiff { base_text, .. }) => base_text,
            Self::Finalized(FinalizedDiff { base_text, .. }) => base_text,
        }
    }

    /// Returns the buffer being edited, if it is still alive.
    pub fn buffer(&self) -> Option<Entity<Buffer>> {
        match self {
            Self::Pending(PendingDiff { new_buffer, .. }) => Some(new_buffer.clone()),
            Self::Finalized(finalized) => finalized
                .resolved
                .get()
                .and_then(|resolved| resolved.new_buffer.upgrade()),
        }
    }

    pub fn file_path(&self, cx: &App) -> Option<String> {
        match self {
            Self::Pending(PendingDiff { new_buffer, .. }) => new_buffer
                .read(cx)
                .file()
                .map(|file| file.full_path(cx).to_string_lossy().into_owned()),
            Self::Finalized(FinalizedDiff { path, .. }) => Some(path.clone()),
        }
    }

    /// Returns the file snapshot represented by a completed ACP diff.
    pub fn review_snapshot(&self) -> Option<(PathBuf, String, String)> {
        let Self::Finalized(diff) = self else {
            return None;
        };
        if !diff.base_text_exists {
            return None;
        }
        Some((
            PathBuf::from(&diff.path),
            diff.base_text.to_string(),
            diff.new_text.to_string(),
        ))
    }

    pub fn multibuffer(&self) -> &Entity<MultiBuffer> {
        match self {
            Self::Pending(PendingDiff { multibuffer, .. }) => multibuffer,
            Self::Finalized(FinalizedDiff { multibuffer, .. }) => multibuffer,
        }
    }

    pub fn to_markdown(&self, cx: &App) -> String {
        if let Self::Finalized(finalized) = self
            && finalized.resolved.get().is_none()
        {
            return format!(
                "Diff: {}\n```\n{}\n```\n",
                finalized.path, finalized.new_text
            );
        }
        let buffer_text = self
            .multibuffer()
            .read(cx)
            .all_buffers()
            .iter()
            .map(|buffer| buffer.read(cx).text())
            .join("\n");
        let path = match self {
            Diff::Pending(PendingDiff {
                new_buffer: buffer, ..
            }) => buffer
                .read(cx)
                .file()
                .map(|file| file.path().display(file.path_style(cx))),
            Diff::Finalized(FinalizedDiff { path, .. }) => Some(path.as_str().into()),
        };
        format!(
            "Diff: {}\n```\n{}\n```\n",
            path.unwrap_or(MultiBuffer::DEFAULT_TITLE.into()),
            buffer_text
        )
    }

    pub fn has_revealed_range(&self, cx: &App) -> bool {
        !self.multibuffer().read(cx).is_empty()
    }

    pub fn needs_update(&self, old_text: &str, new_text: &str, cx: &App) -> bool {
        match self {
            Diff::Pending(PendingDiff {
                base_text,
                new_buffer,
                ..
            }) => {
                base_text.as_ref() != old_text
                    || !new_buffer.read(cx).as_rope().chunks().equals_str(new_text)
            }
            Self::Finalized(finalized) => {
                finalized.base_text.as_ref() != old_text || finalized.new_text.as_ref() != new_text
            }
        }
    }
}

pub struct PendingDiff {
    multibuffer: Entity<MultiBuffer>,
    base_text: Arc<str>,
    new_buffer: Entity<Buffer>,
    diff: Entity<BufferDiff>,
    revealed_ranges: Vec<Range<Anchor>>,
    _subscription: Subscription,
    update_diff: Task<Result<()>>,
}

impl PendingDiff {
    pub fn update(&mut self, cx: &mut Context<Diff>) {
        let buffer = self.new_buffer.clone();
        let buffer_diff = self.diff.clone();
        let base_text = self.base_text.clone();
        self.update_diff = cx.spawn(async move |diff, cx| {
            let text_snapshot = buffer.read_with(cx, |buffer, _| buffer.text_snapshot());
            let base_text_snapshot = buffer_diff.read_with(cx, |diff, cx| diff.base_text(cx));
            let update = buffer_diff
                .update(cx, |diff, cx| {
                    diff.update_diff(
                        text_snapshot.clone(),
                        &base_text_snapshot,
                        Some(base_text.clone()),
                        cx,
                    )
                })
                .await;
            buffer_diff.update(cx, |diff, cx| {
                diff.set_snapshot(update.clone(), cx);
            });
            diff.update(cx, |diff, cx| {
                if let Diff::Pending(diff) = diff {
                    diff.update_visible_ranges(cx);
                }
            })
        });
    }

    pub fn reveal_range(&mut self, range: Range<Anchor>, cx: &mut Context<Diff>) {
        self.revealed_ranges.push(range);
        self.update_visible_ranges(cx);
    }

    fn finalize(&self, cx: &mut Context<Diff>) -> FinalizedDiff {
        let ranges = self.excerpt_ranges(cx);
        let base_text = self.base_text.clone();
        let new_buffer = self.new_buffer.read(cx);

        let path = new_buffer
            .file()
            .map(|file| file.path().display(file.path_style(cx)))
            .unwrap_or(MultiBuffer::DEFAULT_TITLE.into())
            .into();
        let replica_id = new_buffer.replica_id();

        // Replace the buffer in the multibuffer with the snapshot
        let buffer = cx.new(|cx| {
            let language = self.new_buffer.read(cx).language().cloned();
            let buffer = TextBuffer::new_normalized(
                replica_id,
                cx.entity_id().as_non_zero_u64().into(),
                self.new_buffer.read(cx).line_ending(),
                self.new_buffer.read(cx).as_rope().clone(),
            );
            let mut buffer = Buffer::build(buffer, None, Capability::ReadWrite);
            buffer.set_language(language, cx);
            buffer
        });

        let buffer_diff = cx.spawn({
            let buffer = buffer.clone();
            async move |_this, cx| {
                buffer.update(cx, |buffer, _| buffer.parsing_idle()).await;
                build_buffer_diff(base_text, true, &buffer, cx).await
            }
        });

        let update_diff = cx.spawn(async move |this, cx| {
            let buffer_diff = buffer_diff.await?;
            this.update(cx, |this, cx| {
                this.multibuffer().update(cx, |multibuffer, cx| {
                    let path_key = PathKey::for_buffer(&buffer, cx);
                    multibuffer.clear(cx);
                    multibuffer.set_excerpts_for_path(
                        path_key,
                        buffer,
                        ranges,
                        excerpt_context_lines(cx),
                        cx,
                    );
                    multibuffer.add_diff(buffer_diff.clone(), cx);
                });

                cx.notify();
            })
        });

        FinalizedDiff {
            path,
            base_text: self.base_text.clone(),
            base_text_exists: true,
            new_text: Arc::from(self.new_buffer.read(cx).text()),
            multibuffer: self.multibuffer.clone(),
            language_registry: None,
            resolved: OnceCell::from(ResolvedFinalizedDiff {
                new_buffer: self.new_buffer.downgrade(),
                _update_diff: update_diff,
            }),
        }
    }

    fn update_visible_ranges(&mut self, cx: &mut Context<Diff>) {
        let ranges = self.excerpt_ranges(cx);
        self.multibuffer.update(cx, |multibuffer, cx| {
            multibuffer.set_excerpts_for_path(
                PathKey::for_buffer(&self.new_buffer, cx),
                self.new_buffer.clone(),
                ranges,
                excerpt_context_lines(cx),
                cx,
            );
            let end = multibuffer.len(cx);
            Some(multibuffer.snapshot(cx).offset_to_point(end).row + 1)
        });
        cx.notify();
    }

    fn excerpt_ranges(&self, cx: &App) -> Vec<Range<Point>> {
        let buffer = self.new_buffer.read(cx);
        let mut ranges = self
            .diff
            .read(cx)
            .snapshot(cx)
            .hunks_intersecting_range(
                Anchor::min_for_buffer(buffer.remote_id())
                    ..Anchor::max_for_buffer(buffer.remote_id()),
                buffer,
            )
            .map(|diff_hunk| diff_hunk.buffer_range.to_point(buffer))
            .collect::<Vec<_>>();
        ranges.extend(
            self.revealed_ranges
                .iter()
                .map(|range| range.to_point(buffer)),
        );
        ranges.sort_unstable_by_key(|range| (range.start, Reverse(range.end)));

        // Merge adjacent ranges
        let mut ranges = ranges.into_iter().peekable();
        let mut merged_ranges = Vec::new();
        while let Some(mut range) = ranges.next() {
            while let Some(next_range) = ranges.peek() {
                if range.end >= next_range.start {
                    range.end = range.end.max(next_range.end);
                    ranges.next();
                } else {
                    break;
                }
            }

            merged_ranges.push(range);
        }
        merged_ranges
    }
}

pub struct FinalizedDiff {
    path: String,
    base_text: Arc<str>,
    base_text_exists: bool,
    new_text: Arc<str>,
    multibuffer: Entity<MultiBuffer>,
    language_registry: Option<Arc<LanguageRegistry>>,
    resolved: OnceCell<ResolvedFinalizedDiff>,
}

struct ResolvedFinalizedDiff {
    new_buffer: WeakEntity<Buffer>,
    _update_diff: Task<Result<()>>,
}

async fn build_buffer_diff(
    old_text: Arc<str>,
    base_text_exists: bool,
    buffer: &Entity<Buffer>,
    cx: &mut AsyncApp,
) -> Result<Entity<BufferDiff>> {
    let language = cx.update(|cx| buffer.read(cx).language().cloned());
    let language_registry = cx.update(|cx| buffer.read(cx).language_registry());
    let buffer = cx.update(|cx| buffer.read(cx).snapshot());
    let base_text = base_text_exists.then(|| old_text);

    let diff = cx.new(|cx| BufferDiff::new(&buffer, language, language_registry, cx));
    diff.update(cx, |diff, cx| {
        diff.set_base_text(base_text, buffer.text, cx)
    })
    .await;
    Ok(diff)
}

#[cfg(test)]
mod tests {
    use gpui::{AppContext as _, TestAppContext};
    use language::{Buffer, LanguageRegistry};
    use std::sync::Arc;

    use crate::Diff;

    #[gpui::test]
    async fn test_pending_diff(cx: &mut TestAppContext) {
        let buffer = cx.new(|cx| Buffer::local("hello!", cx));
        let _diff = cx.new(|cx| Diff::new(buffer.clone(), cx));
        buffer.update(cx, |buffer, cx| {
            buffer.set_text("HELLO!", cx);
        });
        cx.run_until_parked();
    }

    #[gpui::test]
    async fn test_finalized_diff_exposes_review_snapshot(cx: &mut TestAppContext) {
        let languages = Arc::new(LanguageRegistry::test(cx.background_executor.clone()));
        let diff = cx.new(|cx| {
            Diff::finalized(
                "README.md".to_owned(),
                Some("before".to_owned()),
                "after".to_owned(),
                languages,
                cx,
            )
        });

        diff.read_with(cx, |diff, cx| {
            assert!(
                diff.buffer().is_none(),
                "review must not materialize a buffer"
            );
            assert!(!diff.has_revealed_range(cx));
            assert!(!diff.needs_update("before", "after", cx));
            assert!(diff.to_markdown(cx).contains("after"));
        });
        let snapshot = diff.read_with(cx, |diff, _cx| diff.review_snapshot());
        assert_eq!(
            snapshot,
            Some(("README.md".into(), "before".to_owned(), "after".to_owned()))
        );

        diff.update(cx, |diff, cx| diff.resolve(cx));
        let buffer_id = diff.read_with(cx, |diff, _| {
            diff.buffer().expect("resolved buffer").entity_id()
        });
        diff.update(cx, |diff, cx| diff.resolve(cx));
        assert_eq!(
            diff.read_with(cx, |diff, _| diff
                .buffer()
                .expect("same resolved buffer")
                .entity_id()),
            buffer_id
        );
        cx.run_until_parked();
        assert_eq!(
            diff.read_with(cx, |diff, _| diff.review_snapshot()),
            snapshot
        );

        let new_file_diff = cx.new(|cx| {
            Diff::finalized(
                "new.rs".to_owned(),
                None,
                "new file".to_owned(),
                Arc::new(LanguageRegistry::test(cx.background_executor().clone())),
                cx,
            )
        });
        assert!(
            new_file_diff
                .read_with(cx, |diff, _cx| diff.review_snapshot())
                .is_none()
        );
    }
}
