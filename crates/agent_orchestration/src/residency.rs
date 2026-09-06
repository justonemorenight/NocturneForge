use crate::control_plane::AgentPath;
use crate::events::{RuntimeEvent, RuntimeEventStream};
use crate::ids::RunId;
use agent_client_protocol::schema::v1 as acp;
use anyhow::{Result, bail};
use chrono::{DateTime, Duration, Utc};
use collections::HashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AgentResidencyConfig {
    pub idle_ttl_secs: u64,
    pub max_records: usize,
}

impl Default for AgentResidencyConfig {
    fn default() -> Self {
        Self {
            idle_ttl_secs: 15 * 60,
            max_records: 128,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentResidencyState {
    Loaded,
    Idle,
    Unloaded,
    Reloading,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentResidencyRecord {
    pub agent_path: AgentPath,
    pub state: AgentResidencyState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<acp::SessionId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_revision: Option<u64>,
    pub last_active_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentResidencySnapshot {
    pub records: Vec<AgentResidencyRecord>,
}

#[derive(Clone)]
pub struct AgentResidencyManager {
    config: AgentResidencyConfig,
    records: Arc<RwLock<HashMap<AgentPath, AgentResidencyRecord>>>,
    runtime_events: Arc<RwLock<Option<(RunId, RuntimeEventStream)>>>,
}

pub struct AgentResidencyLease {
    manager: AgentResidencyManager,
    agent_path: AgentPath,
}

impl AgentResidencyManager {
    pub fn new(config: AgentResidencyConfig) -> Result<Self> {
        validate_config(&config)?;
        Ok(Self {
            config,
            records: Arc::new(RwLock::new(HashMap::default())),
            runtime_events: Arc::new(RwLock::new(None)),
        })
    }

    pub fn with_runtime_events(self, run_id: RunId, event_stream: RuntimeEventStream) -> Self {
        *self.runtime_events.write() = Some((run_id, event_stream));
        self
    }

    pub fn prepare_execution(
        &self,
        agent_path: AgentPath,
        session_id: Option<acp::SessionId>,
        context_revision: Option<u64>,
        now: DateTime<Utc>,
    ) -> Result<AgentResidencyRecord> {
        let mut records = self.records.write();
        if !records.contains_key(&agent_path) && records.len() >= self.config.max_records {
            bail!(
                "agent residency registry reached its {} record limit",
                self.config.max_records
            );
        }
        let record = records
            .entry(agent_path.clone())
            .or_insert_with(|| AgentResidencyRecord {
                agent_path,
                state: AgentResidencyState::Loaded,
                session_id: session_id.clone(),
                context_revision,
                last_active_at: now,
                updated_at: now,
            });
        record.state = if record.state == AgentResidencyState::Unloaded {
            AgentResidencyState::Reloading
        } else {
            AgentResidencyState::Loaded
        };
        if session_id.is_some() {
            record.session_id = session_id;
        }
        record.context_revision = context_revision.or(record.context_revision);
        record.last_active_at = now;
        record.updated_at = now;
        let record = record.clone();
        drop(records);
        self.emit(record.clone());
        Ok(record)
    }

    pub fn mark_loaded(
        &self,
        agent_path: &AgentPath,
        session_id: Option<acp::SessionId>,
        now: DateTime<Utc>,
    ) -> Result<AgentResidencyRecord> {
        self.transition(agent_path, AgentResidencyState::Loaded, session_id, now)
    }

    pub fn lease(&self, agent_path: &AgentPath) -> Result<AgentResidencyLease> {
        if !self.records.read().contains_key(agent_path) {
            bail!("agent '{agent_path}' has no residency record");
        }
        Ok(AgentResidencyLease {
            manager: self.clone(),
            agent_path: agent_path.clone(),
        })
    }

    pub fn mark_idle(
        &self,
        agent_path: &AgentPath,
        session_id: Option<acp::SessionId>,
        now: DateTime<Utc>,
    ) -> Result<AgentResidencyRecord> {
        self.transition(agent_path, AgentResidencyState::Idle, session_id, now)
    }

    pub fn evict_idle(&self, now: DateTime<Utc>) -> Vec<AgentResidencyRecord> {
        let ttl = Duration::seconds(i64::try_from(self.config.idle_ttl_secs).unwrap_or(i64::MAX));
        let mut records = self.records.write();
        let mut evicted = Vec::new();
        for record in records.values_mut() {
            if record.state == AgentResidencyState::Idle
                && now.signed_duration_since(record.last_active_at) >= ttl
            {
                record.state = AgentResidencyState::Unloaded;
                record.updated_at = now;
                evicted.push(record.clone());
            }
        }
        evicted.sort_by(|left, right| left.agent_path.cmp(&right.agent_path));
        drop(records);
        for record in &evicted {
            self.emit(record.clone());
        }
        evicted
    }

    pub fn get(&self, agent_path: &AgentPath) -> Option<AgentResidencyRecord> {
        self.records.read().get(agent_path).cloned()
    }

    pub fn snapshot(&self) -> AgentResidencySnapshot {
        let mut records = self.records.read().values().cloned().collect::<Vec<_>>();
        records.sort_by(|left, right| left.agent_path.cmp(&right.agent_path));
        AgentResidencySnapshot { records }
    }

    pub fn restore(snapshot: AgentResidencySnapshot, config: AgentResidencyConfig) -> Result<Self> {
        let manager = Self::new(config)?;
        if snapshot.records.len() > manager.config.max_records {
            bail!("agent residency snapshot exceeds its record limit");
        }
        let mut records = manager.records.write();
        for mut record in snapshot.records {
            AgentPath::parse(record.agent_path.as_str())?;
            if matches!(
                record.state,
                AgentResidencyState::Loaded | AgentResidencyState::Reloading
            ) {
                record.state = AgentResidencyState::Unloaded;
                record.updated_at = Utc::now();
            }
            if records.insert(record.agent_path.clone(), record).is_some() {
                bail!("agent residency snapshot contains duplicate paths");
            }
        }
        drop(records);
        Ok(manager)
    }

    fn transition(
        &self,
        agent_path: &AgentPath,
        state: AgentResidencyState,
        session_id: Option<acp::SessionId>,
        now: DateTime<Utc>,
    ) -> Result<AgentResidencyRecord> {
        let mut records = self.records.write();
        let record = records
            .get_mut(agent_path)
            .ok_or_else(|| anyhow::anyhow!("agent '{agent_path}' has no residency record"))?;
        record.state = state;
        if session_id.is_some() {
            record.session_id = session_id;
        }
        record.last_active_at = now;
        record.updated_at = now;
        let record = record.clone();
        drop(records);
        self.emit(record.clone());
        Ok(record)
    }

    fn emit(&self, record: AgentResidencyRecord) {
        if let Some((run_id, event_stream)) = self.runtime_events.read().as_ref() {
            event_stream.emit(RuntimeEvent::AgentResidencyChanged {
                run_id: run_id.clone(),
                record,
            });
        }
    }
}

impl Drop for AgentResidencyLease {
    fn drop(&mut self) {
        if let Err(error) = self.manager.mark_idle(&self.agent_path, None, Utc::now()) {
            log::debug!("failed to mark agent residency idle: {error}");
        }
    }
}

fn validate_config(config: &AgentResidencyConfig) -> Result<()> {
    if config.idle_ttl_secs == 0 || config.max_records == 0 {
        bail!("agent residency limits must be greater than zero");
    }
    Ok(())
}

#[cfg(test)]
#[path = "residency_tests.rs"]
mod tests;
