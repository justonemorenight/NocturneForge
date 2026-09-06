use super::*;
use crate::cancellation::CancellationReason;

#[test]
fn permits_are_atomic_per_agent_and_release_on_drop() {
    let limiter = AgentExecutionLimiter::new(AgentExecutionLimiterConfig {
        max_active_turns: 1,
    })
    .unwrap();
    let first = AgentPath::parse("/root/first").unwrap();
    let second = AgentPath::parse("/root/second").unwrap();

    let permit = limiter.try_acquire(&first).unwrap().unwrap();
    assert_eq!(limiter.active_turns(), 1);
    assert!(limiter.try_acquire(&first).is_err());
    assert!(limiter.try_acquire(&second).unwrap().is_none());

    drop(permit);
    assert!(limiter.try_acquire(&second).unwrap().is_some());
}

#[test]
fn queued_acquisition_wakes_after_release() {
    smol::block_on(async {
        let limiter = AgentExecutionLimiter::new(AgentExecutionLimiterConfig {
            max_active_turns: 1,
        })
        .unwrap();
        let first = AgentPath::parse("/root/first").unwrap();
        let second = AgentPath::parse("/root/second").unwrap();
        let first_permit = limiter.try_acquire(&first).unwrap().unwrap();

        let cancellation_token = CancellationToken::new();
        let acquire_second = limiter.acquire(&second, &cancellation_token);
        let release_first = async move {
            drop(first_permit);
        };
        let (_, second_permit) = futures::join!(release_first, acquire_second);
        assert!(second_permit.is_ok());
    });
}

#[test]
fn queued_acquisition_observes_cancellation() {
    smol::block_on(async {
        let limiter = AgentExecutionLimiter::new(AgentExecutionLimiterConfig {
            max_active_turns: 1,
        })
        .unwrap();
        let first = AgentPath::parse("/root/first").unwrap();
        let second = AgentPath::parse("/root/second").unwrap();
        let _permit = limiter.try_acquire(&first).unwrap().unwrap();
        let cancellation_token = CancellationToken::new();
        cancellation_token.cancel(CancellationReason::UserRequested);

        assert!(limiter.acquire(&second, &cancellation_token).await.is_err());
    });
}
