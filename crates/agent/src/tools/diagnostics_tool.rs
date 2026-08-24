use crate::{AgentTool, ToolCallEventStream, ToolInput};
use action_log::ActionLog;
use agent_client_protocol::schema::v1 as acp;
use futures::{Future, FutureExt as _};
use gpui::{App, AsyncApp, Entity, Task};
use language::{DiagnosticSeverity, OffsetRangeExt};
use project::Project;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::{fmt::Write, sync::Arc, time::Duration};
use ui::SharedString;
use util::markdown::MarkdownInlineCode;

type Result<T, E = String> = core::result::Result<T, E>;

/// Get errors and warnings for the project or a specific file.
///
/// When a path is provided, shows all diagnostics for that specific file.
/// When no path is provided, shows a summary of error and warning counts for all files in the project.
///
/// This tool attempts to refresh diagnostics before returning.
/// If refreshing diagnostics fails (for example, if the language server does not support pull-based diagnostics), it will return any diagnostics already present.
/// Note that, in this case, the results may be out-of-date, and may or may not reflect the most recent edits.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct DiagnosticsToolInput {
    /// The path to get diagnostics for. If not provided, returns a project-wide summary.
    ///
    /// This path should never be absolute, and the first component
    /// of the path should always be a root directory in a project.
    ///
    /// <example>
    /// If the project has the following root directories:
    ///
    /// - lorem
    /// - ipsum
    ///
    /// If you wanna access diagnostics for `dolor.txt` in `ipsum`, you should use the path `ipsum/dolor.txt`.
    /// </example>
    pub path: Option<String>,
}

pub struct DiagnosticsTool {
    project: Entity<Project>,
    action_log: Entity<ActionLog>,
}

impl DiagnosticsTool {
    pub fn new(project: Entity<Project>, action_log: Entity<ActionLog>) -> Self {
        Self {
            project,
            action_log,
        }
    }
}

async fn with_cancellation<T>(f: impl Future<Output = T>, s: &ToolCallEventStream) -> Result<T> {
    futures::select! {
        result = f.fuse() => Ok(result),
        _ = s.cancelled_by_user().fuse() => {
            Err("Diagnostics cancelled by user".to_string())
        }
    }
}

fn freshness_message(refreshed: DiagnosticsRefresh) -> &'static str {
    match refreshed {
        DiagnosticsRefresh::Refreshed => "Diagnostics successfully refreshed.",
        DiagnosticsRefresh::Failed => "Failed to refresh diagnostics. Diagnostics may be stale.",
        DiagnosticsRefresh::TimedOut => {
            "Diagnostics refresh timed out after 5 seconds. Diagnostics may be stale."
        }
    }
}

#[derive(Clone, Copy)]
enum DiagnosticsRefresh {
    Refreshed,
    Failed,
    TimedOut,
}

async fn pull_diagnostics_with_timeout(
    project: &Entity<Project>,
    action_log: &Entity<ActionLog>,
    path: Option<&Path>,
    event_stream: &ToolCallEventStream,
    cx: &mut AsyncApp,
) -> Result<DiagnosticsRefresh> {
    let timeout = cx.background_executor().timer(Duration::from_secs(5));
    let pull = pull_diagnostics(project, action_log, path, event_stream, cx);
    futures::select! {
        result = pull.fuse() => result,
        _ = timeout.fuse() => {
            if let Some(path) = path {
                log::warn!("Timed out refreshing diagnostics for {}", path.display());
            } else {
                log::warn!("Timed out refreshing workspace diagnostics");
            }
            Ok(DiagnosticsRefresh::TimedOut)
        },
        _ = event_stream.cancelled_by_user().fuse() => {
            Err("Diagnostics cancelled by user".to_string())
        }
    }
}

/// Attempt to pull fresh diagnostics from the LSP before reading them.
///
/// Returns `Ok(true)` if diagnostics were successfully refreshed,
/// `Ok(false)` if the pull failed (callers should fall through to
/// read cached diagnostics), or `Err` if cancelled by the user.
async fn pull_diagnostics(
    project: &Entity<Project>,
    action_log: &Entity<ActionLog>,
    path: Option<&Path>,
    event_stream: &ToolCallEventStream,
    cx: &mut AsyncApp,
) -> Result<DiagnosticsRefresh, String> {
    match path {
        Some(path) => {
            let open_buffer_task = project.update(cx, |project, cx| {
                let Some(project_path) = project.find_project_path(path, cx) else {
                    return Err(format!("Could not find path {} in project", path.display()));
                };
                Ok(project.open_buffer(project_path, cx))
            })?;

            let buffer = with_cancellation(open_buffer_task, event_stream)
                .await?
                .map_err(|e| e.to_string())?;
            let lease_task = action_log.update(cx, |log, cx| {
                log.acquire_diagnostic_lsp_lease(buffer.clone(), cx)
            });
            let lease_result = lease_task.await;
            let _diagnostic_lease = lease_result.map_err(|error| error.to_string())?;

            let lsp_store = project.read_with(cx, |project, _cx| project.lsp_store());
            let pull_task = lsp_store.update(cx, |lsp_store, cx| {
                lsp_store.pull_diagnostics_for_buffer(buffer, cx)
            });
            let pull_result = pull_task.await;
            if let Err(error) = pull_result {
                log::warn!("Failed to pull diagnostics, using cached: {error:#}");
                Ok(DiagnosticsRefresh::Failed)
            } else {
                Ok(DiagnosticsRefresh::Refreshed)
            }
        }
        None => {
            let lsp_store = project.read_with(cx, |project, _cx| project.lsp_store());
            let pull_task = lsp_store.update(cx, |lsp_store, cx| {
                lsp_store.pull_workspace_diagnostics_once(cx)
            });
            let succeeded = pull_task.await;
            if !succeeded {
                log::warn!("Failed to pull workspace diagnostics, using cached");
                Ok(DiagnosticsRefresh::Failed)
            } else {
                Ok(DiagnosticsRefresh::Refreshed)
            }
        }
    }
}

impl AgentTool for DiagnosticsTool {
    type Input = DiagnosticsToolInput;
    type Output = String;

    const NAME: &'static str = "diagnostics";

    fn kind() -> acp::ToolKind {
        acp::ToolKind::Read
    }

    fn initial_title(
        &self,
        input: Result<Self::Input, serde_json::Value>,
        _cx: &mut App,
    ) -> SharedString {
        if let Some(path) = input.ok().and_then(|input| match input.path {
            Some(path) if !path.is_empty() => Some(path),
            _ => None,
        }) {
            format!("Check diagnostics for {}", MarkdownInlineCode(&path)).into()
        } else {
            "Check project diagnostics".into()
        }
    }

    fn run(
        self: Arc<Self>,
        input: ToolInput<Self::Input>,
        event_stream: ToolCallEventStream,
        cx: &mut App,
    ) -> Task<Result<Self::Output, Self::Output>> {
        let project = self.project.clone();
        let action_log = self.action_log.clone();
        cx.spawn(async move |cx| {
            let input = input.recv().await.map_err(|e| e.to_string())?;

            match input.path {
                Some(ref path) if !path.is_empty() => {
                    let refreshed = pull_diagnostics_with_timeout(
                        &project,
                        &action_log,
                        Some(Path::new(path)),
                        &event_stream,
                        cx,
                    )
                    .await?;

                    let open_buffer_task = project.update(cx, |project, cx| {
                        let Some(project_path) = project.find_project_path(path, cx) else {
                            return Err(format!("Could not find path {path} in project"));
                        };
                        Ok(project.open_buffer(project_path, cx))
                    })?;

                    let buffer = with_cancellation(open_buffer_task, &event_stream)
                        .await?
                        .map_err(|e| e.to_string())?;

                    let mut output = String::new();
                    let snapshot = buffer.read_with(cx, |buffer, _cx| buffer.snapshot());

                    for (_, group) in snapshot.diagnostic_groups(None) {
                        let entry = &group.entries[group.primary_ix];
                        let range = entry.range.to_point(&snapshot);
                        let severity = match entry.diagnostic.severity {
                            DiagnosticSeverity::ERROR => "error",
                            DiagnosticSeverity::WARNING => "warning",
                            _ => continue,
                        };

                        writeln!(
                            output,
                            "{} at line {}: {}",
                            severity,
                            range.start.row + 1,
                            entry.diagnostic.message
                        )
                        .ok();
                    }

                    let freshness = freshness_message(refreshed);
                    if output.is_empty() {
                        Ok(format!(
                            "{freshness}\n\nFile doesn't have errors or warnings!"
                        ))
                    } else {
                        Ok(format!("{freshness}\n\n{output}"))
                    }
                }
                _ => {
                    let refreshed = pull_diagnostics_with_timeout(
                        &project,
                        &action_log,
                        None,
                        &event_stream,
                        cx,
                    )
                    .await?;

                    let (output, has_diagnostics) = project.read_with(cx, |project, cx| {
                        let mut output = String::new();
                        let mut has_diagnostics = false;

                        for (project_path, _, summary) in project.diagnostic_summaries(true, cx) {
                            if summary.error_count > 0 || summary.warning_count > 0 {
                                let Some(worktree) =
                                    project.worktree_for_id(project_path.worktree_id, cx)
                                else {
                                    continue;
                                };

                                has_diagnostics = true;
                                output.push_str(&format!(
                                    "{}: {} error(s), {} warning(s)\n",
                                    worktree.read(cx).absolutize(&project_path.path).display(),
                                    summary.error_count,
                                    summary.warning_count
                                ));
                            }
                        }

                        (output, has_diagnostics)
                    });

                    let freshness = freshness_message(refreshed);
                    if has_diagnostics {
                        Ok(format!("{freshness}\n\n{output}"))
                    } else {
                        Ok(format!(
                            "{freshness}\n\nNo errors or warnings found in the project."
                        ))
                    }
                }
            }
        })
    }
}
