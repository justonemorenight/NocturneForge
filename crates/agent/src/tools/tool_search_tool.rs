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

pub(crate) fn bounded_tool_description(description: &str) -> String {
    let mut bounded =
        String::with_capacity(description.len().min(MAX_TOOL_SEARCH_DESCRIPTION_BYTES));
    for character in description.chars() {
        if bounded.len().saturating_add(character.len_utf8()) > MAX_TOOL_SEARCH_DESCRIPTION_BYTES {
            break;
        }
        bounded.push(if character == '\n' { ' ' } else { character });
    }
    bounded
}

pub(crate) fn tool_search_relevance(
    normalized_name: &str,
    normalized_description: &str,
    query: &str,
) -> Option<u8> {
    if query.is_empty() {
        Some(0)
    } else if normalized_name == query {
        Some(1)
    } else if normalized_name.starts_with(query) {
        Some(2)
    } else if normalized_name.contains(query) {
        Some(3)
    } else if normalized_description.contains(query) {
        Some(4)
    } else {
        None
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_description_normalizes_without_splitting_unicode() {
        let description = format!("first line\n{}", "🦀".repeat(256));
        let bounded = bounded_tool_description(&description);

        assert!(bounded.len() <= MAX_TOOL_SEARCH_DESCRIPTION_BYTES);
        assert!(bounded.starts_with("first line "));
        assert!(!bounded.contains('\n'));
    }

    #[test]
    fn relevance_prefers_names_over_descriptions() {
        assert_eq!(
            tool_search_relevance("edit_file", "write code", "edit_file"),
            Some(1)
        );
        assert_eq!(
            tool_search_relevance("edit_file", "write code", "edit"),
            Some(2)
        );
        assert_eq!(
            tool_search_relevance("streaming_edit", "write code", "edit"),
            Some(3)
        );
        assert_eq!(tool_search_relevance("other", "edit code", "edit"), Some(4));
        assert_eq!(tool_search_relevance("other", "read code", "edit"), None);
    }
}
