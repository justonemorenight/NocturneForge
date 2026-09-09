use super::*;
use crate::verification::VerificationResult;

#[test]
fn goal_is_achieved_only_when_every_task_is_completed_and_verified() {
    let controller = GoalController::new(
        RunId::new(),
        "ship runtime",
        vec![TaskId::new("task")],
        GoalControllerConfig::default(),
    )
    .unwrap();
    let mut status = TaskStatus::new(TaskId::new("task"));
    status.state = TaskState::Completed;
    status.latest_verification = Some(VerificationResult::pass());

    let snapshot = controller.observe(RunState::Completed, &[status]);
    assert_eq!(snapshot.status, GoalStatus::Achieved);
    assert_eq!(snapshot.verified_tasks, 1);
}

#[test]
fn blocker_requires_repeated_explicit_observations() {
    let controller = GoalController::new(
        RunId::new(),
        "ship runtime",
        vec![TaskId::new("task")],
        GoalControllerConfig {
            repeated_blocker_threshold: 3,
            ..Default::default()
        },
    )
    .unwrap();

    assert_eq!(
        controller
            .record_blocker("missing-input", "waiting for user")
            .unwrap()
            .status,
        GoalStatus::Active
    );
    controller
        .record_blocker("other", "external service")
        .unwrap();
    controller
        .record_blocker("missing-input", "waiting for user")
        .unwrap();
    controller
        .record_blocker("missing-input", "still waiting")
        .unwrap();
    assert_eq!(
        controller
            .record_blocker("missing-input", "waiting remains unresolved")
            .unwrap()
            .status,
        GoalStatus::Blocked
    );
    assert_eq!(controller.clear_blocker().status, GoalStatus::Active);
}

#[test]
fn restore_rejects_impossible_progress() {
    let controller = GoalController::new(
        RunId::new(),
        "ship runtime",
        vec![TaskId::new("task")],
        GoalControllerConfig::default(),
    )
    .unwrap();
    let mut snapshot = controller.snapshot();
    snapshot.completed_tasks = 2;
    assert!(GoalController::restore(snapshot, GoalControllerConfig::default()).is_err());
}

#[test]
fn goal_ignores_statuses_for_tasks_outside_its_plan() {
    let controller = GoalController::new(
        RunId::new(),
        "ship runtime",
        vec![TaskId::new("expected")],
        GoalControllerConfig::default(),
    )
    .unwrap();
    let mut unrelated = TaskStatus::new(TaskId::new("unrelated"));
    unrelated.state = TaskState::Completed;
    unrelated.latest_verification = Some(VerificationResult::pass());

    let snapshot = controller.observe(RunState::Completed, &[unrelated]);
    assert_eq!(snapshot.status, GoalStatus::Failed);
    assert_eq!(snapshot.completed_tasks, 0);
}

#[test]
fn parent_goal_requires_explicit_completion() {
    let controller = GoalController::new(
        RunId::new(),
        "finish the parent turn",
        Vec::new(),
        GoalControllerConfig::default(),
    )
    .unwrap();

    assert_eq!(controller.snapshot().status, GoalStatus::Active);
    assert_eq!(
        controller.mark_achieved().unwrap().status,
        GoalStatus::Achieved
    );
    assert_eq!(
        controller.mark_achieved().unwrap().status,
        GoalStatus::Achieved
    );
}

#[test]
fn failed_parent_goal_cannot_be_completed() {
    let controller = GoalController::new(
        RunId::new(),
        "finish the parent turn",
        Vec::new(),
        GoalControllerConfig::default(),
    )
    .unwrap();

    assert_eq!(
        controller.observe(RunState::Failed, &[]).status,
        GoalStatus::Failed
    );
    assert!(controller.mark_achieved().is_err());
}
