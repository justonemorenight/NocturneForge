use agent_client_protocol::schema::v1 as acp;
use anyhow::Result;
use gpui::{App, SharedString, Task, WeakEntity};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::{AgentTool, Thread, ToolCallEventStream, ToolInput};

pub(crate) const DEFAULT_TOOL_SEARCH_LIMIT: usize = 8;
pub(crate) const MAX_TOOL_SEARCH_RESULTS: usize = 32;
pub(crate) const MAX_TOOL_SEARCH_DESCRIPTION_BYTES: usize = 512;

/// Searches the optional tool catalog and enables matching tools for the next
/// model request. Use this before calling a capability that is not already in
/// the current tool list.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct ToolSearchToolInput {
    /// A tool name or capability to search for. An empty query lists the most
    /// relevant optional tools that are available to this thread.
    #[serde(default)]
    pub query: String,
    /// Maximum number of tools to return. Values are capped to keep the result
    /// small enough for the next model request.
    #[serde(default)]
    pub limit: Option<usize>,
}

pub struct ToolSearchTool {
    thread: WeakEntity<Thread>,
}

impl ToolSearchTool {
    pub fn new(thread: WeakEntity<Thread>) -> Self {
        Self { thread }
    }
}

impl AgentTool for ToolSearchTool {
    type Input = ToolSearchToolInput;
    type Output = String;

    const NAME: &'static str = "tool_search";

    fn kind() -> acp::ToolKind {
        acp::ToolKind::Other
    }

    fn initial_title(
        &self,
        input: Result<Self::Input, serde_json::Value>,
        _cx: &mut App,
    ) -> SharedString {
        input
            .ok()
            .filter(|input| !input.query.trim().is_empty())
            .map_or_else(
                || "Search tools".into(),
                |input| format!("Search tools: {}", input.query).into(),
            )
    }

    fn run(
        self: Arc<Self>,
        input: ToolInput<Self::Input>,
        event_stream: ToolCallEventStream,
        cx: &mut App,
    ) -> Task<Result<Self::Output, Self::Output>> {
        let thread = self.thread.clone();
        let input = input.recv();
        cx.spawn(async move |cx| {
            let input = input.await.map_err(|error| error.to_string())?;
            let query = input.query;
            let limit = input.limit;
            let result = cx
                .update(|cx| {
                    thread.update(cx, |thread, cx| {
                        thread.search_and_enable_tools(&query, limit, cx)
                    })
                })
                .map_err(|error| error.to_string())?
                .map_err(|error| error.to_string())?;
            event_stream.update_fields(acp::ToolCallUpdateFields::new().title("Tools discovered"));
            Ok(result)
        })
    }
}
