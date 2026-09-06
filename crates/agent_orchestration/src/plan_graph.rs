use crate::ids::{PlanId, TaskId};
use crate::worker::{WorkerTarget, WorkspacePolicy};
use collections::{HashMap, HashSet};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::fmt;

const MAX_TASK_RETRIES: u8 = 5;

/// Specification for a single task within an orchestration plan.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct OrchestrationTask {
    pub id: TaskId,
    pub label: String,
    pub description: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_override: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_effort: Option<String>,
    /// Optional tool allowlist for this task.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<String>>,
    /// Predecessor task IDs that must finish successfully before this task starts.
    #[serde(default)]
    pub depends_on: Vec<TaskId>,
    /// Criteria required for this task's output to be considered verified.
    #[serde(default)]
    pub acceptance_criteria: Vec<String>,
    /// Override for maximum retries on failure.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_retries: Option<u8>,
    /// Cumulative provider-reported token ceiling across all attempts, including
    /// input, output, and cache tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token_budget: Option<u64>,
    /// Execution timeout in seconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub time_budget_secs: Option<u64>,
    /// Whether to attempt repair on verification failure.
    #[serde(default = "default_true")]
    pub repair_on_failure: bool,
    /// Context files or paths assigned to this task.
    #[serde(default)]
    pub context_paths: Vec<String>,
    /// Expected format or content structure of task output.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_output: Option<String>,
    /// Whether task output must include valid file citations (evidence).
    #[serde(default)]
    pub evidence_required: bool,
    /// Hard budget on tool calls reported by the task executor.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_budget: Option<u64>,
    /// Stated high-level objective for this task.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub objective: Option<String>,
    /// Comma-separated repository-relative paths or glob patterns defining the
    /// task's primary evidence and affected-file scope.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
    /// Native subagent role. External workers do not interpret this field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub native_role: Option<String>,
    /// Worker execution target: Native or ACP agent.
    #[serde(default)]
    pub target: WorkerTarget,
    /// Mode to request from the worker (e.g. session mode or profile).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
    /// Workspace isolation policy for this task.
    #[serde(default)]
    pub workspace_policy: WorkspacePolicy,
    /// Optional, trusted verification command to run in the worktree and on the
    /// parent checkout after apply. Model-generated task input does not expose
    /// this field; runtime validation rejects shell syntax and non-allowlisted
    /// executables before spawning it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verification_command: Option<String>,
}

fn default_true() -> bool {
    true
}

impl OrchestrationTask {
    pub fn new(
        id: impl Into<TaskId>,
        label: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            description: description.into(),
            role: None,
            model_override: None,
            thinking_effort: None,
            tools: None,
            depends_on: Vec::new(),
            acceptance_criteria: Vec::new(),
            max_retries: None,
            token_budget: None,
            time_budget_secs: None,
            repair_on_failure: true,
            context_paths: Vec::new(),
            expected_output: None,
            evidence_required: false,
            tool_call_budget: None,
            objective: None,
            scope: None,
            native_role: None,
            target: WorkerTarget::Native,
            mode: None,
            workspace_policy: WorkspacePolicy::default(),
            verification_command: None,
        }
    }

    pub fn with_depends_on(mut self, dependencies: Vec<TaskId>) -> Self {
        self.depends_on = dependencies;
        self
    }

    pub fn with_tools(mut self, tools: Vec<String>) -> Self {
        self.tools = Some(tools);
        self
    }

    pub fn with_acceptance_criteria(mut self, criteria: Vec<String>) -> Self {
        self.acceptance_criteria = criteria;
        self
    }

    pub fn with_token_budget(mut self, budget: u64) -> Self {
        self.token_budget = Some(budget);
        self
    }
}

/// An entire orchestration plan composed of interrelated tasks.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct OrchestrationPlan {
    pub id: PlanId,
    pub title: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub explanation: Option<String>,
    pub tasks: Vec<OrchestrationTask>,
}

impl OrchestrationPlan {
    pub fn new(title: impl Into<String>, tasks: Vec<OrchestrationTask>) -> Self {
        Self {
            id: PlanId::new(),
            title: title.into(),
            explanation: None,
            tasks,
        }
    }

    pub fn with_explanation(mut self, explanation: impl Into<String>) -> Self {
        self.explanation = Some(explanation.into());
        self
    }
}

/// Validation errors discovered in a plan dependency graph.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GraphValidationError {
    EmptyTasks,
    DuplicateTaskId(TaskId),
    MissingDependency { task_id: TaskId, dependency: TaskId },
    CyclicDependency(Vec<TaskId>),
    InvalidBudget { task_id: TaskId, reason: String },
    MaxRetriesExceeded { task_id: TaskId, max: u8 },
}

impl fmt::Display for GraphValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyTasks => write!(f, "plan must contain at least one task"),
            Self::DuplicateTaskId(id) => write!(f, "duplicate task ID `{}`", id),
            Self::MissingDependency {
                task_id,
                dependency,
            } => {
                write!(
                    f,
                    "task `{}` depends on unknown task `{}`",
                    task_id, dependency
                )
            }
            Self::CyclicDependency(cycle) => {
                let formatted = cycle
                    .iter()
                    .map(|id| id.as_str())
                    .collect::<Vec<_>>()
                    .join(" -> ");
                write!(f, "dependency graph contains a cycle: {}", formatted)
            }
            Self::InvalidBudget { task_id, reason } => {
                write!(f, "invalid budget for task `{}`: {}", task_id, reason)
            }
            Self::MaxRetriesExceeded { task_id, max } => {
                write!(f, "task `{}` max_retries cannot exceed {}", task_id, max)
            }
        }
    }
}

impl std::error::Error for GraphValidationError {}

/// Validated dependency graph and execution scheduler for a plan.
#[derive(Clone, Debug)]
pub struct PlanGraph {
    plan: OrchestrationPlan,
    tasks_by_id: HashMap<TaskId, OrchestrationTask>,
    waves: Vec<Vec<TaskId>>,
}

impl PlanGraph {
    /// Validates the plan and constructs an executable dependency graph.
    pub fn new(plan: OrchestrationPlan) -> Result<Self, GraphValidationError> {
        if plan.tasks.is_empty() {
            return Err(GraphValidationError::EmptyTasks);
        }

        let mut tasks_by_id = HashMap::default();
        for task in &plan.tasks {
            if tasks_by_id.contains_key(&task.id) {
                return Err(GraphValidationError::DuplicateTaskId(task.id.clone()));
            }
            if let Some(retries) = task.max_retries {
                if retries > MAX_TASK_RETRIES {
                    return Err(GraphValidationError::MaxRetriesExceeded {
                        task_id: task.id.clone(),
                        max: MAX_TASK_RETRIES,
                    });
                }
            }
            if let Some(budget) = task.token_budget {
                if budget == 0 {
                    return Err(GraphValidationError::InvalidBudget {
                        task_id: task.id.clone(),
                        reason: "token budget must be greater than zero".into(),
                    });
                }
            }
            tasks_by_id.insert(task.id.clone(), task.clone());
        }

        for task in &plan.tasks {
            for dep in &task.depends_on {
                if !tasks_by_id.contains_key(dep) {
                    return Err(GraphValidationError::MissingDependency {
                        task_id: task.id.clone(),
                        dependency: dep.clone(),
                    });
                }
                if dep == &task.id {
                    return Err(GraphValidationError::CyclicDependency(vec![
                        task.id.clone(),
                        dep.clone(),
                    ]));
                }
            }
        }

        let waves = Self::compute_waves(&plan.tasks)?;

        Ok(Self {
            plan,
            tasks_by_id,
            waves,
        })
    }

    fn compute_waves(
        tasks: &[OrchestrationTask],
    ) -> Result<Vec<Vec<TaskId>>, GraphValidationError> {
        let mut waves = Vec::new();
        let mut completed = HashSet::default();

        while completed.len() < tasks.len() {
            let current_wave: Vec<TaskId> = tasks
                .iter()
                .filter(|task| {
                    !completed.contains(&task.id)
                        && task.depends_on.iter().all(|dep| completed.contains(dep))
                })
                .map(|task| task.id.clone())
                .collect();

            if current_wave.is_empty() {
                let remaining: Vec<TaskId> = tasks
                    .iter()
                    .filter(|task| !completed.contains(&task.id))
                    .map(|task| task.id.clone())
                    .collect();
                return Err(GraphValidationError::CyclicDependency(remaining));
            }

            for task_id in &current_wave {
                completed.insert(task_id.clone());
            }

            waves.push(current_wave);
        }

        Ok(waves)
    }

    /// Access the underlying plan.
    pub fn plan(&self) -> &OrchestrationPlan {
        &self.plan
    }

    /// Total number of tasks in the graph.
    pub fn task_count(&self) -> usize {
        self.plan.tasks.len()
    }

    /// Retrieve a task definition by its ID.
    pub fn task(&self, id: &TaskId) -> Option<&OrchestrationTask> {
        self.tasks_by_id.get(id)
    }

    /// Pre-computed wave decomposition of tasks.
    pub fn waves(&self) -> &[Vec<TaskId>] {
        &self.waves
    }

    /// Returns task IDs that are ready to run (all dependencies completed, not yet in progress or completed).
    pub fn ready_tasks(
        &self,
        completed: &HashSet<TaskId>,
        in_progress: &HashSet<TaskId>,
        failed_or_cancelled: &HashSet<TaskId>,
    ) -> Vec<TaskId> {
        self.plan
            .tasks
            .iter()
            .filter(|task| {
                !completed.contains(&task.id)
                    && !in_progress.contains(&task.id)
                    && !failed_or_cancelled.contains(&task.id)
                    && task.depends_on.iter().all(|dep| completed.contains(dep))
            })
            .map(|task| task.id.clone())
            .collect()
    }

    /// Checks if all tasks in the plan have finished successfully.
    pub fn is_complete(&self, completed: &HashSet<TaskId>) -> bool {
        self.plan
            .tasks
            .iter()
            .all(|task| completed.contains(&task.id))
    }

    /// Checks if the run can make further progress.
    pub fn can_progress(
        &self,
        completed: &HashSet<TaskId>,
        in_progress: &HashSet<TaskId>,
        failed_or_cancelled: &HashSet<TaskId>,
    ) -> bool {
        if self.is_complete(completed) {
            return false;
        }
        if !in_progress.is_empty() {
            return true;
        }
        !self
            .ready_tasks(completed, in_progress, failed_or_cancelled)
            .is_empty()
    }

    /// Identifies all tasks transitively blocked by the failure or cancellation of a task.
    pub fn blocked_by_failure(&self, failed_task: &TaskId) -> HashSet<TaskId> {
        let mut dependents_map: HashMap<&TaskId, Vec<&TaskId>> = HashMap::default();
        for task in &self.plan.tasks {
            for dep in &task.depends_on {
                dependents_map.entry(dep).or_default().push(&task.id);
            }
        }

        let mut blocked = HashSet::default();
        let mut queue = vec![failed_task];

        while let Some(current) = queue.pop() {
            if let Some(direct_dependents) = dependents_map.get(current) {
                for dep in direct_dependents {
                    if blocked.insert((*dep).clone()) {
                        queue.push(dep);
                    }
                }
            }
        }

        blocked
    }
}
