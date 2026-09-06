use super::*;

#[test]
fn idle_agents_are_evicted_and_prepared_for_reload() {
    let manager = AgentResidencyManager::new(AgentResidencyConfig {
        idle_ttl_secs: 10,
        max_records: 2,
    })
    .unwrap();
    let path = AgentPath::parse("/root/scout").unwrap();
    let now = Utc::now();
    manager
        .prepare_execution(path.clone(), None, Some(3), now)
        .unwrap();
    manager.mark_idle(&path, None, now).unwrap();

    assert!(manager.evict_idle(now + Duration::seconds(9)).is_empty());
    assert_eq!(manager.evict_idle(now + Duration::seconds(10)).len(), 1);
    let record = manager
        .prepare_execution(path, None, Some(4), now + Duration::seconds(11))
        .unwrap();
    assert_eq!(record.state, AgentResidencyState::Reloading);
    assert_eq!(record.context_revision, Some(4));
}

#[test]
fn restore_never_claims_that_a_process_local_worker_is_loaded() {
    let path = AgentPath::parse("/root/scout").unwrap();
    let now = Utc::now();
    let manager = AgentResidencyManager::new(AgentResidencyConfig::default()).unwrap();
    manager
        .prepare_execution(path.clone(), None, None, now)
        .unwrap();

    let restored =
        AgentResidencyManager::restore(manager.snapshot(), AgentResidencyConfig::default())
            .unwrap();
    assert_eq!(
        restored.get(&path).map(|record| record.state),
        Some(AgentResidencyState::Unloaded)
    );
}
