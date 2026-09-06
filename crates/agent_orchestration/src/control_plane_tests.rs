use super::*;
use crate::plan_graph::OrchestrationTask;

fn task(id: &str) -> OrchestrationTask {
    OrchestrationTask::new(id, id, "test task")
}

#[test]
fn plan_tasks_receive_stable_unique_canonical_paths() {
    let mut first = task("Code Review");
    first.role = Some("reviewer".to_string());
    let plan = OrchestrationPlan::new("test", vec![first, task("code-review")]);
    let control_plane =
        AgentControlPlane::from_plan(&plan, AgentControlPlaneConfig::default()).unwrap();

    assert_eq!(
        control_plane
            .identity_for_task(&TaskId::new("Code Review"))
            .unwrap()
            .path
            .as_str(),
        "/root/code-review"
    );
    assert_eq!(
        control_plane
            .identity_for_task(&TaskId::new("code-review"))
            .unwrap()
            .path
            .as_str(),
        "/root/code-review-2"
    );
}

#[test]
fn mailbox_rejects_unknown_agents_and_enforces_bounds() {
    let config = AgentControlPlaneConfig {
        max_messages_per_agent: 1,
        max_message_bytes: 5,
        max_mailbox_bytes_per_agent: 5,
        ..Default::default()
    };
    let control_plane = AgentControlPlane::from_plan(
        &OrchestrationPlan::new("test", vec![task("worker")]),
        config,
    )
    .unwrap();
    let root = AgentPath::root();
    let worker = control_plane.resolve(&root, "worker").unwrap();
    let outside = AgentPath::parse("/root/outside").unwrap();

    assert!(
        control_plane
            .send(&outside, &worker, AgentMessageKind::Message, "hello")
            .is_err()
    );
    assert!(
        control_plane
            .send(&root, &worker, AgentMessageKind::Message, "longer")
            .is_err()
    );
    control_plane
        .send(&root, &worker, AgentMessageKind::Message, "hello")
        .unwrap();
    assert!(
        control_plane
            .send(&root, &worker, AgentMessageKind::Message, "again")
            .is_err()
    );
}

#[test]
fn mailbox_coalesces_wakes_and_preserves_order() {
    smol::block_on(async {
        let control_plane = AgentControlPlane::from_plan(
            &OrchestrationPlan::new("test", vec![task("worker")]),
            AgentControlPlaneConfig::default(),
        )
        .unwrap();
        let root = AgentPath::root();
        let worker = control_plane.resolve(&root, "worker").unwrap();
        control_plane
            .send(&root, &worker, AgentMessageKind::Message, "first")
            .unwrap();
        control_plane
            .send(&root, &worker, AgentMessageKind::FollowUp, "second")
            .unwrap();

        let messages = control_plane.wait_for_messages(&worker).await.unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].body, "first");
        assert_eq!(messages[1].body, "second");
        assert!(!messages[0].kind.triggers_turn());
        assert!(messages[1].kind.triggers_turn());
    });
}

#[test]
fn acknowledging_delivery_removes_only_the_delivered_message() {
    let control_plane = AgentControlPlane::from_plan(
        &OrchestrationPlan::new("test", vec![task("worker")]),
        AgentControlPlaneConfig::default(),
    )
    .unwrap();
    let root = AgentPath::root();
    let worker = control_plane.resolve(&root, "worker").unwrap();
    let first = control_plane
        .send(&root, &worker, AgentMessageKind::FollowUp, "first")
        .unwrap();
    let second = control_plane
        .send(&root, &worker, AgentMessageKind::Message, "second")
        .unwrap();

    assert_eq!(
        control_plane
            .acknowledge_delivery(&worker, first.sequence)
            .unwrap(),
        first
    );
    assert_eq!(control_plane.drain(&worker).unwrap(), vec![second]);
}

#[test]
fn rejected_delivery_is_removed_without_draining_other_messages() {
    let control_plane = AgentControlPlane::from_plan(
        &OrchestrationPlan::new("test", vec![task("worker")]),
        AgentControlPlaneConfig::default(),
    )
    .unwrap();
    let root = AgentPath::root();
    let worker = control_plane.resolve(&root, "worker").unwrap();
    let rejected = control_plane
        .send(&root, &worker, AgentMessageKind::FollowUp, "interrupt")
        .unwrap();
    let retained = control_plane
        .send(&root, &worker, AgentMessageKind::Message, "later")
        .unwrap();

    assert_eq!(
        control_plane
            .reject_delivery(&worker, rejected.sequence, "worker finished")
            .unwrap(),
        rejected
    );
    assert_eq!(control_plane.drain(&worker).unwrap(), vec![retained]);
}

#[test]
fn relative_resolution_is_scoped_and_ambiguous_leaf_names_are_rejected() {
    let control_plane = AgentControlPlane::new(AgentControlPlaneConfig::default()).unwrap();
    let root = AgentPath::root();
    let left = control_plane
        .register_child(
            &root,
            TaskId::new("left"),
            "left",
            None,
            WorkerTarget::Native,
        )
        .unwrap();
    let right = control_plane
        .register_child(
            &root,
            TaskId::new("right"),
            "right",
            None,
            WorkerTarget::Native,
        )
        .unwrap();
    control_plane
        .register_child(
            &left,
            TaskId::new("left-scout"),
            "scout",
            None,
            WorkerTarget::Native,
        )
        .unwrap();
    control_plane
        .register_child(
            &right,
            TaskId::new("right-scout"),
            "scout",
            None,
            WorkerTarget::Native,
        )
        .unwrap();

    assert_eq!(
        control_plane.resolve(&left, "scout").unwrap().as_str(),
        "/root/left/scout"
    );
    assert!(control_plane.resolve(&root, "scout").is_err());
}

#[test]
fn unregistering_a_subtree_removes_descendants_and_mailboxes() {
    let control_plane = AgentControlPlane::new(AgentControlPlaneConfig::default()).unwrap();
    let root = AgentPath::root();
    let parent = control_plane
        .register_child(
            &root,
            TaskId::new("parent"),
            "parent",
            None,
            WorkerTarget::Native,
        )
        .unwrap();
    let child = control_plane
        .register_child(
            &parent,
            TaskId::new("child"),
            "child",
            None,
            WorkerTarget::Native,
        )
        .unwrap();

    assert_eq!(control_plane.unregister_subtree(&parent).unwrap(), 2);
    assert!(control_plane.identity(&parent).is_none());
    assert!(control_plane.identity(&child).is_none());
    assert!(control_plane.drain(&child).is_err());
}

#[test]
fn repeated_registration_must_match_the_existing_identity() {
    let control_plane = AgentControlPlane::new(AgentControlPlaneConfig::default()).unwrap();
    let root = AgentPath::root();
    let parent = control_plane
        .register_child(
            &root,
            TaskId::new("parent"),
            "parent",
            None,
            WorkerTarget::Native,
        )
        .unwrap();
    let task_id = TaskId::new("worker");
    control_plane
        .register_child(
            &root,
            task_id.clone(),
            "worker",
            Some("scout".to_string()),
            WorkerTarget::Native,
        )
        .unwrap();

    assert!(
        control_plane
            .register_child(
                &parent,
                task_id,
                "worker",
                Some("scout".to_string()),
                WorkerTarget::Native,
            )
            .is_err()
    );
}

#[test]
fn control_plane_lifecycle_is_recorded_in_the_runtime_event_log() {
    let run_id = RunId::new();
    let event_stream = RuntimeEventStream::new();
    let control_plane = AgentControlPlane::from_plan(
        &OrchestrationPlan::new("test", vec![task("worker")]),
        AgentControlPlaneConfig::default(),
    )
    .unwrap()
    .with_runtime_events(run_id, event_stream.clone());
    let root = AgentPath::root();
    let worker = control_plane.resolve(&root, "worker").unwrap();

    let queued = control_plane
        .send(&root, &worker, AgentMessageKind::FollowUp, "continue")
        .unwrap();
    assert_eq!(control_plane.drain(&worker).unwrap(), vec![queued]);
    assert_eq!(control_plane.unregister_subtree(&worker).unwrap(), 1);

    let history = event_stream.history();
    assert_eq!(
        history
            .iter()
            .filter(|entry| matches!(entry.event, RuntimeEvent::AgentRegistered { .. }))
            .count(),
        2
    );
    assert!(
        history
            .iter()
            .any(|entry| matches!(entry.event, RuntimeEvent::AgentMessageQueued { .. }))
    );
    assert!(history.iter().any(|entry| matches!(
        entry.event,
        RuntimeEvent::AgentMailboxDrained { count: 1, .. }
    )));
    assert!(
        history
            .iter()
            .any(|entry| matches!(entry.event, RuntimeEvent::AgentUnregistered { .. }))
    );
}

#[test]
fn snapshot_restores_dynamic_agents_and_unread_mailbox_messages() {
    let control_plane = AgentControlPlane::new(AgentControlPlaneConfig::default()).unwrap();
    let root = AgentPath::root();
    let parent = control_plane
        .register_child(
            &root,
            TaskId::new("parent"),
            "parent",
            Some("worker".to_string()),
            WorkerTarget::Native,
        )
        .unwrap();
    let child = control_plane
        .register_child(
            &parent,
            TaskId::new("dynamic-child"),
            "scout",
            Some("scout".to_string()),
            WorkerTarget::Native,
        )
        .unwrap();
    let original_message = control_plane
        .send(
            &parent,
            &child,
            AgentMessageKind::FollowUp,
            "inspect parser",
        )
        .unwrap();

    let restored =
        AgentControlPlane::restore(control_plane.snapshot(), AgentControlPlaneConfig::default())
            .unwrap();
    assert_eq!(
        restored
            .identity_for_task(&TaskId::new("dynamic-child"))
            .map(|identity| identity.path),
        Some(child.clone())
    );
    assert_eq!(restored.drain(&child).unwrap(), vec![original_message]);
}

#[test]
fn snapshot_rejects_cross_mailbox_message_injection() {
    let control_plane = AgentControlPlane::from_plan(
        &OrchestrationPlan::new("test", vec![task("first"), task("second")]),
        AgentControlPlaneConfig::default(),
    )
    .unwrap();
    let root = AgentPath::root();
    let first = control_plane.resolve(&root, "first").unwrap();
    let second = control_plane.resolve(&root, "second").unwrap();
    control_plane
        .send(&root, &first, AgentMessageKind::Message, "hello")
        .unwrap();
    let mut snapshot = control_plane.snapshot();
    let mailbox = snapshot
        .mailboxes
        .iter_mut()
        .find(|mailbox| mailbox.recipient == first)
        .unwrap();
    mailbox.messages[0].recipient = second;

    assert!(AgentControlPlane::restore(snapshot, AgentControlPlaneConfig::default()).is_err());
}
