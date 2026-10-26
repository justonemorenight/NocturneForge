use agent_client_protocol::schema::{v1 as acp_v1, v2 as acp};
use anyhow::Context as _;
use chrono::{DateTime, Utc};
use db::kvp::KeyValueStore;
use gpui::{App, AppContext as _, Task};
use project::AgentId;
use serde::{Deserialize, Serialize};
use util::ResultExt as _;

const NAMESPACE: &str = "acp_session_client_state";
const VERSION: u32 = 2;

#[derive(Clone, Serialize)]
struct StateKey {
    agent_id: AgentId,
    session_id: acp::SessionId,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct AcpSessionClientState {
    version: u32,
    draft_prompt: Option<Vec<acp::ContentBlock>>,
    updated_at: DateTime<Utc>,
}

pub fn read_draft_prompt(
    agent_id: &AgentId,
    session_id: &acp::SessionId,
    cx: &App,
) -> Option<Vec<acp::ContentBlock>> {
    read(agent_id, session_id, cx).and_then(|state| state.draft_prompt)
}

fn read(
    agent_id: &AgentId,
    session_id: &acp::SessionId,
    cx: &App,
) -> Option<AcpSessionClientState> {
    let key = key(agent_id, session_id).log_err()?;
    let kvp = KeyValueStore::global(cx);
    let raw = kvp.scoped(NAMESPACE).read(&key).log_err().flatten()?;
    decode_state(&raw).log_err()
}

fn decode_state(raw: &str) -> anyhow::Result<AcpSessionClientState> {
    let value: serde_json::Value = serde_json::from_str(raw)?;
    match value.get("version").and_then(serde_json::Value::as_u64) {
        Some(2) => Ok(serde_json::from_value(value)?),
        Some(1) => {
            #[derive(Deserialize)]
            struct LegacyState {
                draft_prompt: Option<Vec<acp_v1::ContentBlock>>,
                updated_at: DateTime<Utc>,
            }
            let legacy: LegacyState = serde_json::from_value(value)?;
            Ok(AcpSessionClientState {
                version: VERSION,
                draft_prompt: legacy
                    .draft_prompt
                    .map(|blocks| {
                        blocks
                            .into_iter()
                            .map(crate::content::from_v1)
                            .collect::<anyhow::Result<Vec<_>>>()
                    })
                    .transpose()?,
                updated_at: legacy.updated_at,
            })
        }
        version => anyhow::bail!("Unsupported ACP client state version: {version:?}"),
    }
}

pub fn write_draft_prompt(
    agent_id: AgentId,
    session_id: acp::SessionId,
    draft_prompt: Vec<acp::ContentBlock>,
    cx: &App,
) -> Task<anyhow::Result<()>> {
    let kvp = KeyValueStore::global(cx);
    let key = match key(&agent_id, &session_id) {
        Ok(key) => key,
        Err(err) => return Task::ready(Err(err)),
    };
    let payload = match serde_json::to_string(&AcpSessionClientState {
        version: VERSION,
        draft_prompt: Some(draft_prompt),
        updated_at: Utc::now(),
    })
    .context("serializing ACP session client state")
    {
        Ok(payload) => payload,
        Err(err) => return Task::ready(Err(err)),
    };
    cx.background_spawn(async move { kvp.scoped(NAMESPACE).write(key, payload).await })
}

pub fn delete(agent_id: AgentId, session_id: acp::SessionId, cx: &App) -> Task<anyhow::Result<()>> {
    let kvp = KeyValueStore::global(cx);
    let key = match key(&agent_id, &session_id) {
        Ok(key) => key,
        Err(err) => return Task::ready(Err(err)),
    };
    cx.background_spawn(async move { kvp.scoped(NAMESPACE).delete(key).await })
}

fn key(agent_id: &AgentId, session_id: &acp::SessionId) -> anyhow::Result<String> {
    serde_json::to_string(&StateKey {
        agent_id: agent_id.clone(),
        session_id: session_id.clone(),
    })
    .context("serializing ACP session client state key")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn migrates_legacy_drafts_without_losing_metadata() {
        let raw = json!({
            "version": 1,
            "draft_prompt": [{
                "type": "text", "text": "Draft 🌙",
                "annotations": { "audience": ["user"], "priority": 0.7 },
                "_meta": { "owner": "retained" }
            }],
            "updated_at": "2026-10-23T00:00:00Z"
        })
        .to_string();
        let state = decode_state(&raw).expect("migrate legacy draft");
        assert_eq!(state.version, VERSION);
        let blocks = state.draft_prompt.expect("retained draft");
        let acp::ContentBlock::Text(text) = &blocks[0] else {
            panic!("text draft");
        };
        assert_eq!(text.text, "Draft 🌙");
        assert_eq!(
            text.meta.as_ref().expect("metadata").get("owner"),
            Some(&json!("retained"))
        );
        assert_eq!(
            text.annotations.as_ref().expect("annotations").priority,
            Some(0.7)
        );
        let encoded = serde_json::to_string(&AcpSessionClientState {
            version: VERSION,
            draft_prompt: Some(blocks.clone()),
            updated_at: state.updated_at,
        })
        .expect("encode v2 state");
        assert_eq!(
            decode_state(&encoded)
                .expect("decode v2 state")
                .draft_prompt,
            Some(blocks)
        );
    }

    #[test]
    fn rejects_unknown_persisted_state_versions() {
        assert!(
            decode_state(r#"{"version":3,"draft_prompt":[],"updated_at":"2026-10-23T00:00:00Z"}"#)
                .is_err()
        );
    }
}
