use crate::cancellation::CancellationToken;
use crate::control_plane::AgentPath;
use anyhow::{Result, bail};
use collections::HashSet;
use parking_lot::Mutex;
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AgentExecutionLimiterConfig {
    pub max_active_turns: usize,
}

impl Default for AgentExecutionLimiterConfig {
    fn default() -> Self {
        Self {
            max_active_turns: 4,
        }
    }
}

#[derive(Clone)]
pub struct AgentExecutionLimiter {
    inner: Arc<AgentExecutionLimiterInner>,
}

struct AgentExecutionLimiterInner {
    config: AgentExecutionLimiterConfig,
    active_agents: Mutex<HashSet<AgentPath>>,
    capacity_sender: async_channel::Sender<()>,
    capacity_receiver: async_channel::Receiver<()>,
}

pub struct AgentExecutionPermit {
    owner: AgentPath,
    limiter: AgentExecutionLimiter,
}

impl AgentExecutionLimiter {
    pub fn new(config: AgentExecutionLimiterConfig) -> Result<Self> {
        if config.max_active_turns == 0 {
            bail!("agent execution limit must be greater than zero");
        }
        let (capacity_sender, capacity_receiver) = async_channel::bounded(1);
        Ok(Self {
            inner: Arc::new(AgentExecutionLimiterInner {
                config,
                active_agents: Mutex::new(HashSet::default()),
                capacity_sender,
                capacity_receiver,
            }),
        })
    }

    pub fn max_active_turns(&self) -> usize {
        self.inner.config.max_active_turns
    }

    pub fn active_turns(&self) -> usize {
        self.inner.active_agents.lock().len()
    }

    pub fn try_acquire(&self, owner: &AgentPath) -> Result<Option<AgentExecutionPermit>> {
        let mut active_agents = self.inner.active_agents.lock();
        if active_agents.contains(owner) {
            bail!("agent '{owner}' already has an active turn");
        }
        if active_agents.len() >= self.inner.config.max_active_turns {
            return Ok(None);
        }
        active_agents.insert(owner.clone());
        let has_remaining_capacity = active_agents.len() < self.inner.config.max_active_turns;
        drop(active_agents);
        if has_remaining_capacity {
            self.notify_capacity();
        }
        Ok(Some(AgentExecutionPermit {
            owner: owner.clone(),
            limiter: self.clone(),
        }))
    }

    pub async fn acquire(
        &self,
        owner: &AgentPath,
        cancellation_token: &CancellationToken,
    ) -> Result<AgentExecutionPermit> {
        loop {
            if cancellation_token.is_cancelled() {
                bail!(
                    "agent execution was {}",
                    cancellation_token
                        .reason()
                        .map(|reason| reason.description().to_string())
                        .unwrap_or_else(|| "cancelled".to_string())
                );
            }
            if let Some(permit) = self.try_acquire(owner)? {
                return Ok(permit);
            }

            let capacity = self.inner.capacity_receiver.recv();
            let cancelled = cancellation_token.cancelled();
            futures::pin_mut!(capacity, cancelled);
            match futures::future::select(capacity, cancelled).await {
                futures::future::Either::Left((Ok(()), _)) => {}
                futures::future::Either::Left((Err(_), _)) => {
                    bail!("agent execution limiter is closed")
                }
                futures::future::Either::Right(((), _)) => {}
            }
        }
    }

    fn release(&self, owner: &AgentPath) {
        if self.inner.active_agents.lock().remove(owner) {
            self.notify_capacity();
        }
    }

    fn notify_capacity(&self) {
        match self.inner.capacity_sender.try_send(()) {
            Ok(()) | Err(async_channel::TrySendError::Full(())) => {}
            Err(async_channel::TrySendError::Closed(())) => {
                log::debug!("agent execution limiter capacity channel closed");
            }
        }
    }
}

impl Drop for AgentExecutionPermit {
    fn drop(&mut self) {
        self.limiter.release(&self.owner);
    }
}

#[cfg(test)]
#[path = "execution_limiter_tests.rs"]
mod tests;
