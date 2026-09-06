use super::*;

#[test]
fn incremental_checkpoints_are_materialized_and_deduplicated() {
    let store = ContextCheckpointStore::new(ContextCheckpointConfig::default()).unwrap();
    let agent = AgentPath::parse("/root/scout").unwrap();
    store
        .record_full(
            agent.clone(),
            ContextState {
                objective: Some("map parser".to_string()),
                paths: vec!["src/parser.rs".to_string()],
                ..Default::default()
            },
        )
        .unwrap();
    let checkpoint = store
        .record_incremental(
            agent.clone(),
            ContextDelta {
                add_paths: vec!["src/parser.rs".to_string(), "src/lexer.rs".to_string()],
                add_symbols: vec!["Parser::parse".to_string()],
                ..Default::default()
            },
        )
        .unwrap();

    assert_eq!(checkpoint.revision, 2);
    assert_eq!(checkpoint.parent_revision, Some(1));
    assert_eq!(
        checkpoint.state.paths,
        vec!["src/parser.rs".to_string(), "src/lexer.rs".to_string()]
    );
    assert_eq!(store.latest(&agent), Some(checkpoint));
}

#[test]
fn restore_rejects_non_monotonic_revisions() {
    let agent = AgentPath::parse("/root/scout").unwrap();
    let checkpoint = ContextCheckpoint {
        agent_path: agent,
        revision: 1,
        parent_revision: None,
        kind: ContextCheckpointKind::Full,
        state: ContextState::default(),
        captured_at: Utc::now(),
    };
    let snapshot = ContextCheckpointStoreSnapshot {
        checkpoints: vec![checkpoint.clone(), checkpoint],
    };
    assert!(ContextCheckpointStore::restore(snapshot, ContextCheckpointConfig::default()).is_err());
}

#[test]
fn checkpoint_collection_and_byte_limits_are_enforced() {
    let store = ContextCheckpointStore::new(ContextCheckpointConfig {
        max_paths: 1,
        max_text_bytes: 128,
        ..Default::default()
    })
    .unwrap();
    let agent = AgentPath::parse("/root/scout").unwrap();
    assert!(
        store
            .record_full(
                agent.clone(),
                ContextState {
                    paths: vec!["one".to_string(), "two".to_string()],
                    ..Default::default()
                },
            )
            .is_err()
    );
    assert!(
        store
            .record_full(
                agent,
                ContextState {
                    objective: Some("x".repeat(256)),
                    ..Default::default()
                },
            )
            .is_err()
    );
}
