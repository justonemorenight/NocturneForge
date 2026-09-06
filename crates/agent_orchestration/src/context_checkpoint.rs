use crate::control_plane::AgentPath;
use anyhow::{Result, bail};
use chrono::{DateTime, Utc};
use collections::HashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContextCheckpointConfig {
    pub max_checkpoints_per_agent: usize,
    pub max_paths: usize,
    pub max_symbols: usize,
    pub max_diagnostics: usize,
    pub max_artifact_references: usize,
    pub max_text_bytes: usize,
}

impl Default for ContextCheckpointConfig {
    fn default() -> Self {
        Self {
            max_checkpoints_per_agent: 16,
            max_paths: 256,
            max_symbols: 256,
            max_diagnostics: 128,
            max_artifact_references: 128,
            max_text_bytes: 128 * 1024,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContextCheckpointKind {
    Full,
    Incremental,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextDiagnostic {
    pub path: String,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub severity: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextState {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub objective: Option<String>,
    #[serde(default)]
    pub paths: Vec<String>,
    #[serde(default)]
    pub symbols: Vec<String>,
    #[serde(default)]
    pub diagnostics: Vec<ContextDiagnostic>,
    #[serde(default)]
    pub artifact_references: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub environment: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ContextDelta {
    pub objective: Option<String>,
    pub add_paths: Vec<String>,
    pub add_symbols: Vec<String>,
    pub diagnostics: Option<Vec<ContextDiagnostic>>,
    pub add_artifact_references: Vec<String>,
    pub environment: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextCheckpoint {
    pub agent_path: AgentPath,
    pub revision: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_revision: Option<u64>,
    pub kind: ContextCheckpointKind,
    pub state: ContextState,
    pub captured_at: DateTime<Utc>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextCheckpointStoreSnapshot {
    pub checkpoints: Vec<ContextCheckpoint>,
}

#[derive(Clone)]
pub struct ContextCheckpointStore {
    config: ContextCheckpointConfig,
    checkpoints: Arc<RwLock<HashMap<AgentPath, VecDeque<ContextCheckpoint>>>>,
}

impl ContextCheckpointStore {
    pub fn new(config: ContextCheckpointConfig) -> Result<Self> {
        validate_config(&config)?;
        Ok(Self {
            config,
            checkpoints: Arc::new(RwLock::new(HashMap::default())),
        })
    }

    pub fn record_full(
        &self,
        agent_path: AgentPath,
        state: ContextState,
    ) -> Result<ContextCheckpoint> {
        self.record(agent_path, ContextCheckpointKind::Full, state)
    }

    pub fn record_incremental(
        &self,
        agent_path: AgentPath,
        delta: ContextDelta,
    ) -> Result<ContextCheckpoint> {
        let mut state = self
            .latest(&agent_path)
            .map(|checkpoint| checkpoint.state)
            .unwrap_or_default();
        if let Some(objective) = delta.objective {
            state.objective = Some(objective);
        }
        extend_unique(&mut state.paths, delta.add_paths);
        extend_unique(&mut state.symbols, delta.add_symbols);
        if let Some(diagnostics) = delta.diagnostics {
            state.diagnostics = diagnostics;
        }
        extend_unique(
            &mut state.artifact_references,
            delta.add_artifact_references,
        );
        if let Some(environment) = delta.environment {
            state.environment = Some(environment);
        }
        self.record(agent_path, ContextCheckpointKind::Incremental, state)
    }

    pub fn latest(&self, agent_path: &AgentPath) -> Option<ContextCheckpoint> {
        self.checkpoints
            .read()
            .get(agent_path)
            .and_then(|checkpoints| checkpoints.back())
            .cloned()
    }

    pub fn snapshot(&self) -> ContextCheckpointStoreSnapshot {
        let checkpoints = self
            .checkpoints
            .read()
            .values()
            .flat_map(|checkpoints| checkpoints.iter().cloned())
            .collect();
        ContextCheckpointStoreSnapshot { checkpoints }
    }

    pub fn restore(
        snapshot: ContextCheckpointStoreSnapshot,
        config: ContextCheckpointConfig,
    ) -> Result<Self> {
        let store = Self::new(config)?;
        for checkpoint in snapshot.checkpoints {
            validate_state(&checkpoint.state, &store.config)?;
            if checkpoint.revision == 0 {
                bail!("context checkpoint revision must be greater than zero");
            }
            let mut all = store.checkpoints.write();
            let checkpoints = all.entry(checkpoint.agent_path.clone()).or_default();
            if checkpoints
                .back()
                .is_some_and(|previous| checkpoint.revision <= previous.revision)
            {
                bail!("context checkpoint revisions must be strictly increasing");
            }
            checkpoints.push_back(checkpoint);
            if checkpoints.len() > store.config.max_checkpoints_per_agent {
                checkpoints.pop_front();
            }
        }
        Ok(store)
    }

    fn record(
        &self,
        agent_path: AgentPath,
        kind: ContextCheckpointKind,
        state: ContextState,
    ) -> Result<ContextCheckpoint> {
        validate_state(&state, &self.config)?;
        let mut all = self.checkpoints.write();
        let checkpoints = all.entry(agent_path.clone()).or_default();
        let parent_revision = checkpoints.back().map(|checkpoint| checkpoint.revision);
        let revision = parent_revision.unwrap_or_default().saturating_add(1);
        if revision == u64::MAX && parent_revision == Some(u64::MAX) {
            bail!("context checkpoint revision is exhausted");
        }
        let checkpoint = ContextCheckpoint {
            agent_path,
            revision,
            parent_revision,
            kind,
            state,
            captured_at: Utc::now(),
        };
        checkpoints.push_back(checkpoint.clone());
        if checkpoints.len() > self.config.max_checkpoints_per_agent {
            checkpoints.pop_front();
        }
        Ok(checkpoint)
    }
}

fn extend_unique(values: &mut Vec<String>, additions: Vec<String>) {
    for addition in additions {
        if !values.contains(&addition) {
            values.push(addition);
        }
    }
}

fn validate_config(config: &ContextCheckpointConfig) -> Result<()> {
    if config.max_checkpoints_per_agent == 0 || config.max_text_bytes == 0 {
        bail!("context checkpoint limits must be greater than zero");
    }
    Ok(())
}

fn validate_state(state: &ContextState, config: &ContextCheckpointConfig) -> Result<()> {
    if state.paths.len() > config.max_paths
        || state.symbols.len() > config.max_symbols
        || state.diagnostics.len() > config.max_diagnostics
        || state.artifact_references.len() > config.max_artifact_references
    {
        bail!("context checkpoint exceeds its collection limits");
    }
    let serialized_bytes = serde_json::to_vec(state)?.len();
    if serialized_bytes > config.max_text_bytes {
        bail!(
            "context checkpoint exceeds the {} byte limit",
            config.max_text_bytes
        );
    }
    Ok(())
}

#[cfg(test)]
#[path = "context_checkpoint_tests.rs"]
mod tests;
