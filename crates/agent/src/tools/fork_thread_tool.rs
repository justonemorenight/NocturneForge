use crate::{
    AgentTool, SiblingThreadRequest, Thread, ThreadEnvironment, ToolCallEventStream, ToolInput,
    ToolPermissionContext, ToolPermissionDecision, ZED_AGENT_ID, decide_permission_from_settings,
};
use agent_client_protocol::schema::v1 as acp;
use agent_settings::AgentSettings;
use anyhow::{Result, ensure};
use futures::FutureExt as _;
use gpui::{App, SharedString, Task, WeakEntity};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use settings::Settings as _;
use std::{rc::Rc, sync::Arc};

/// Fork this native conversation for an independent, focused follow-up.
/// Copies history through the previous user turn, NOT the current turn.
/// Include everything from the current request that the fork needs in `prompt`.
/// Prefer staying here for small, tightly coupled changes. Use spawn_agent when
/// you need a worker's result here: forks appear in the sidebar but do not return
/// results to this conversation. Do not wait for a fork or claim it completed.
/// The fork retains this model and profile, but not running workers or approvals.
/// Sharing a worktree shares file edits; do not edit the same files concurrently.
/// A new worktree starts at HEAD and does NOT include uncommitted changes.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ForkThreadToolInput {
    pub title: String,
    /// Bounded follow-up task, acceptance criteria, and any new context.
    pub prompt: String,
    /// Why an independent conversation is preferable to continuing here.
    pub reason: String,
    #[serde(default)]
    pub use_new_worktree: bool,
}

pub struct ForkThreadTool {
    thread: WeakEntity<Thread>,
    environment: Rc<dyn ThreadEnvironment>,
}

impl ForkThreadTool {
    pub fn new(thread: WeakEntity<Thread>, environment: Rc<dyn ThreadEnvironment>) -> Self {
        Self {
            thread,
            environment,
        }
    }
}

impl AgentTool for ForkThreadTool {
    type Input = ForkThreadToolInput;
    type Output = String;
    const NAME: &'static str = "fork_thread";

    fn kind() -> acp::ToolKind {
        acp::ToolKind::Other
    }

    fn initial_title(
        &self,
        input: Result<Self::Input, serde_json::Value>,
        _: &mut App,
    ) -> SharedString {
        input
            .map(|input| format!("Fork conversation: {}", input.title).into())
            .unwrap_or_else(|_| "Fork conversation".into())
    }

    fn run(
        self: Arc<Self>,
        input: ToolInput<Self::Input>,
        event_stream: ToolCallEventStream,
        cx: &mut App,
    ) -> Task<Result<String, String>> {
        cx.spawn(async move |cx| {
            let result: Result<String> = async {
                let input = input.recv().await?;
                ensure!(!input.title.trim().is_empty(), "A fork title is required");
                ensure!(!input.prompt.trim().is_empty(), "A fork task is required");
                ensure!(!input.reason.trim().is_empty(), "Explain why this task should fork");
                let (snapshot, confirmation) = self.thread.read_with(cx, |thread, cx| {
                    Ok::<_, anyhow::Error>((
                        thread.fork_snapshot(input.title.clone(), cx)?,
                        thread.fork_requires_confirmation(),
                    ))
                })??;
                let snapshot = snapshot.await?;
                let authorize = cx.update(|cx| {
                    let context = ToolPermissionContext::new(Self::NAME, vec![input.prompt.clone()]);
                    let title = format!("Fork conversation: {} — {}", input.title, input.reason);
                    // Forced confirmation must not turn an explicit deny into a prompt.
                    if confirmation {
                        if let ToolPermissionDecision::Deny(reason) = decide_permission_from_settings(
                            Self::NAME, &context.input_values, AgentSettings::get_global(cx),
                        ) {
                            return Task::ready(Err(anyhow::anyhow!(reason)));
                        }
                        event_stream.authorize_always_prompt(title, context, cx)
                    } else {
                        event_stream.authorize(title, context, cx)
                    }
                });
                futures::select! {
                    result = authorize.fuse() => result?,
                    _ = event_stream.cancelled_by_user().fuse() => anyhow::bail!("Fork cancelled"),
                }
                ensure!(!event_stream.was_cancelled_by_user(), "Fork cancelled");
                let info = self.environment.create_sibling_thread(SiblingThreadRequest {
                    title: snapshot.title.clone(),
                    prompt: input.prompt,
                    agent_id: Some(ZED_AGENT_ID.to_string()),
                    model: None,
                    use_new_worktree: input.use_new_worktree,
                    worktree_name: None,
                    base_ref: None,
                    fork_snapshot: Some(Arc::new(snapshot)),
                }, cx).await?;
                Ok(format!(
                    "Opened independent conversation {:?} (session {}). It has not necessarily finished. Open it from the sidebar; its results will not be returned here.{}",
                    info.title,
                    info.session_id.map(|id| id.to_string()).unwrap_or_default(),
                    info.warning.map(|warning| format!(" {warning}")).unwrap_or_default(),
                ))
            }.await;
            result.map_err(|error| format!("Failed to fork conversation: {error:#}"))
        })
    }
}
