use crate::{AgentMessageContent, DbThread, Message};
use agent_client_protocol::schema::v1 as acp;
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ForkOrigin {
    pub session_id: acp::SessionId,
    pub checkpoint_user_message_id: acp_thread::ClientUserMessageId,
    pub inherited_message_count: usize,
    pub inherited_user_message_ids: collections::HashSet<acp_thread::ClientUserMessageId>,
}

pub(crate) fn prepare_fork(
    mut snapshot: DbThread,
    source_session_id: acp::SessionId,
    title: String,
) -> Result<DbThread> {
    // The last user turn owns the fork tool call and must never be copied:
    // its tool results are incomplete and replaying its intent can fork again.
    let boundary = snapshot
        .messages
        .iter()
        .rposition(|message| matches!(message.as_ref(), Message::User(message) if !message.content.is_empty()))
        .ok_or_else(|| anyhow::anyhow!("There is no completed turn to fork"))?;
    let checkpoint_index = snapshot.messages[..boundary]
        .iter()
        .rposition(|message| matches!(message.as_ref(), Message::User(message) if !message.content.is_empty()))
        .ok_or_else(|| anyhow::anyhow!("Finish the first turn before forking its context"))?;
    let Message::User(checkpoint) = snapshot.messages[checkpoint_index].as_ref() else {
        anyhow::bail!("Invalid fork checkpoint");
    };
    let checkpoint = checkpoint.id.clone();
    ensure!(
        snapshot.messages[checkpoint_index + 1..boundary].iter().any(|message| {
            matches!(message.as_ref(), Message::Agent(message)
                if message.content.iter().any(|content| matches!(content, AgentMessageContent::Text(text) if !text.trim().is_empty())))
        }),
        "The checkpoint has no agent response"
    );
    for message in &snapshot.messages[..boundary] {
        if let Message::Agent(message) = message.as_ref() {
            for content in &message.content {
                if let AgentMessageContent::ToolUse(tool_use) = content {
                    ensure!(
                        tool_use.is_input_complete
                            && message.tool_results.contains_key(&tool_use.id),
                        "Cannot fork a checkpoint containing unfinished tool calls"
                    );
                }
            }
        }
    }
    snapshot.messages.truncate(boundary);
    snapshot.fork_origin = Some(ForkOrigin {
        session_id: source_session_id,
        checkpoint_user_message_id: checkpoint,
        inherited_message_count: boundary,
        inherited_user_message_ids: snapshot
            .messages
            .iter()
            .filter_map(|message| match message.as_ref() {
                Message::User(message) => Some(message.id.clone()),
                _ => None,
            })
            .collect(),
    });
    snapshot.title = format!("Fork: {title}").into();
    snapshot.updated_at = chrono::Utc::now();
    snapshot.detailed_summary = None;
    snapshot.cumulative_token_usage = Default::default();
    snapshot.request_token_usage.clear();
    snapshot.plan = None;
    snapshot.proposed_plan = None;
    snapshot.subagent_context = None;
    snapshot.draft_prompt = None;
    snapshot.ui_scroll_position = None;
    snapshot.sandboxed_terminal_temp_dir = None;
    snapshot.sandbox_grants = Default::default();
    snapshot.discovered_tools.clear();
    snapshot.orchestration_run = None;
    snapshot.orchestration_goal = None;
    Ok(snapshot)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AgentMessage, CompactionInfo, UserMessage};
    use acp_thread::ClientUserMessageId;
    use std::sync::Arc;

    fn user() -> Arc<Message> {
        Arc::new(Message::User(UserMessage {
            id: ClientUserMessageId::new(),
            content: Arc::from([crate::UserMessageContent::Text("request".into())]),
        }))
    }

    fn answer(text: &str) -> Arc<Message> {
        Arc::new(Message::Agent(AgentMessage {
            content: vec![AgentMessageContent::Text(text.to_string())],
            ..Default::default()
        }))
    }

    fn snapshot(messages: Vec<Arc<Message>>) -> DbThread {
        serde_json::from_value(serde_json::json!({
            "title": "source", "messages": messages,
            "updated_at": "2026-09-17T00:00:00Z"
        }))
        .expect("legacy thread without optional fields")
    }

    #[test]
    fn fork_excludes_current_turn_and_future_compaction() {
        let messages = vec![
            user(),
            answer("baseline"),
            Arc::new(Message::Compaction(CompactionInfo::Summary(
                "old context".into(),
            ))),
            user(),
            answer("current partial response"),
            Arc::new(Message::Compaction(CompactionInfo::Summary(
                "future context".into(),
            ))),
        ];
        let source = messages.clone();
        let fork = prepare_fork(
            snapshot(messages),
            acp::SessionId::new("parent"),
            "focused".into(),
        )
        .unwrap();
        assert_eq!(fork.messages, source[..3]);
        assert_eq!(source.len(), 6);
        let origin = fork.fork_origin.as_ref().unwrap();
        assert_eq!(origin.session_id, acp::SessionId::new("parent"));
        assert_eq!(origin.inherited_message_count, 3);
        let restored: DbThread =
            serde_json::from_value(serde_json::to_value(&fork).unwrap()).unwrap();
        assert_eq!(restored.fork_origin.unwrap().inherited_message_count, 3);
    }

    #[test]
    fn fork_resets_session_state_but_preserves_restrictions() {
        let mut source = snapshot(vec![user(), answer("baseline"), user()]);
        source.detailed_summary = Some("includes current turn".into());
        source.sandboxed_terminal_temp_dir = Some("/tmp/source-thread".into());
        source.sandbox_grants.network_any_host = true;
        source.draft_prompt = Some(Vec::new());
        source.discovered_tools = vec!["terminal".into()];
        source.tool_filter = Some(vec!["read_file".into()]);
        source.thinking_effort = Some("high".into());
        let fork = prepare_fork(source, acp::SessionId::new("parent"), "focused".into()).unwrap();
        assert!(fork.detailed_summary.is_none());
        assert!(fork.sandboxed_terminal_temp_dir.is_none());
        assert!(!fork.sandbox_grants.network_any_host);
        assert!(fork.draft_prompt.is_none());
        assert!(fork.discovered_tools.is_empty());
        assert_eq!(fork.tool_filter, Some(vec!["read_file".into()]));
        assert_eq!(fork.thinking_effort.as_deref(), Some("high"));
    }

    #[test]
    fn fork_rejects_missing_or_unanswered_checkpoint() {
        for messages in [
            vec![],
            vec![user()],
            vec![user(), user()],
            vec![user(), answer("earlier"), user(), user()],
        ] {
            assert!(
                prepare_fork(
                    snapshot(messages),
                    acp::SessionId::new("parent"),
                    "task".into()
                )
                .is_err()
            );
        }
    }

    #[test]
    fn fork_rejects_unfinished_tool_call() {
        let tool = language_model::LanguageModelToolUse {
            id: "unfinished".into(),
            name: "read_file".into(),
            raw_input: "{}".into(),
            input: language_model::LanguageModelToolUseInput::Json(serde_json::json!({})),
            is_input_complete: true,
            thought_signature: None,
        };
        let messages = vec![
            user(),
            answer("working"),
            Arc::new(Message::Agent(AgentMessage {
                content: vec![AgentMessageContent::ToolUse(tool)],
                ..Default::default()
            })),
            user(),
        ];
        assert!(
            prepare_fork(
                snapshot(messages),
                acp::SessionId::new("parent"),
                "task".into()
            )
            .is_err()
        );
    }

    #[test]
    fn fork_after_manual_compaction_keeps_the_real_checkpoint() {
        let baseline = user();
        let source = snapshot(vec![
            baseline.clone(),
            answer("baseline"),
            Arc::new(Message::User(UserMessage {
                id: ClientUserMessageId::new(),
                content: Arc::from([]),
            })),
            Arc::new(Message::Compaction(CompactionInfo::Summary(
                "baseline summary".into(),
            ))),
            user(),
        ]);
        let fork = prepare_fork(source, acp::SessionId::new("source"), "follow-up".into()).unwrap();
        let Message::User(baseline) = baseline.as_ref() else {
            panic!("expected user")
        };
        assert_eq!(
            fork.fork_origin.unwrap().checkpoint_user_message_id,
            baseline.id
        );
        assert_eq!(fork.messages.len(), 4);
    }
}
