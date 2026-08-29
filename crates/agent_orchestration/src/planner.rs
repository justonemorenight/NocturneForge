use crate::auto_policy::AutoPolicyDecision;
use crate::auto_policy::AutoPolicyEngine;
use crate::ids::RunId;
use crate::plan_graph::{OrchestrationPlan, OrchestrationTask};
use agent_settings::{AgentAutonomy, AgentExecutionPolicy, AgentExecutionStrategy};
use chrono::{DateTime, Utc};
use collections::HashMap;
use serde::{Deserialize, Serialize};

/// A plan awaiting user approval before its tasks execute.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlanProposal {
    pub run_id: RunId,
    pub plan: OrchestrationPlan,
    pub strategy: AgentExecutionStrategy,
    pub autonomy: AgentAutonomy,
    /// Confidence (0.0-1.0) of the auto-policy engine that produced this proposal.
    #[serde(default)]
    pub confidence: f32,
    /// Human-readable rationale from the policy engine.
    #[serde(default)]
    pub reason: String,
    /// Per-heuristic scores from the policy engine.
    #[serde(default)]
    pub heuristics: HashMap<String, f32>,
    pub proposed_at: DateTime<Utc>,
}

impl PlanProposal {
    /// Records a plan awaiting approval. The strategy is taken from the
    /// resolved policy (an `Auto` policy is expected to have been resolved
    /// to a concrete strategy before proposing).
    pub fn new(run_id: RunId, plan: OrchestrationPlan, policy: AgentExecutionPolicy) -> Self {
        Self {
            run_id,
            plan,
            strategy: policy.strategy,
            autonomy: policy.autonomy,
            confidence: 0.0,
            reason: String::new(),
            heuristics: HashMap::default(),
            proposed_at: Utc::now(),
        }
    }

    /// Records a proposal with the auto-policy decision that produced it.
    pub fn with_decision(
        run_id: RunId,
        plan: OrchestrationPlan,
        policy: AgentExecutionPolicy,
        decision: AutoPolicyDecision,
    ) -> Self {
        Self {
            run_id,
            plan,
            strategy: decision.strategy,
            autonomy: policy.autonomy,
            confidence: decision.confidence,
            reason: decision.reason,
            heuristics: decision.heuristics,
            proposed_at: Utc::now(),
        }
    }
}

/// Builds orchestration plans and resolves execution strategy in `Auto` mode.
pub struct OrchestrationPlanner;

impl OrchestrationPlanner {
    /// Resolves `Auto` strategy to a concrete strategy and launch disposition.
    pub fn resolve_auto(
        prompt: &str,
        task_count: usize,
        autonomy: AgentAutonomy,
    ) -> AutoPolicyDecision {
        AutoPolicyEngine::evaluate(prompt, task_count, 0, autonomy)
    }

    /// Creates a plan from already-decomposed tasks, preserving task metadata.
    pub fn plan_from_tasks(
        title: impl Into<String>,
        tasks: Vec<OrchestrationTask>,
        explanation: Option<String>,
    ) -> OrchestrationPlan {
        let mut plan = OrchestrationPlan::new(title, tasks);
        if let Some(explanation) = explanation {
            plan = plan.with_explanation(explanation);
        }
        plan
    }

    /// Builds a single-task plan. Used as a fallback when `Auto` resolves to
    /// direct execution but a plan is still wanted for the user's approval.
    pub fn single_task_plan(
        title: impl Into<String>,
        task: OrchestrationTask,
    ) -> OrchestrationPlan {
        OrchestrationPlan::new(title, vec![task])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::TaskId;

    #[test]
    fn plan_proposal_records_policy_and_scores() {
        let plan = OrchestrationPlan::new(
            "Parallel delegation",
            vec![
                OrchestrationTask::new(TaskId::new("task-1"), "Search", "Find usages"),
                OrchestrationTask::new(TaskId::new("task-2"), "Implement", "Apply changes"),
            ],
        );
        let policy = AgentExecutionPolicy {
            strategy: AgentExecutionStrategy::Orchestrate,
            autonomy: AgentAutonomy::Manual,
        };
        let proposal = PlanProposal::new(RunId::new(), plan.clone(), policy);
        assert_eq!(proposal.strategy, AgentExecutionStrategy::Orchestrate);
        assert_eq!(proposal.autonomy, AgentAutonomy::Manual);
        assert_eq!(proposal.plan, plan);
        assert_eq!(proposal.confidence, 0.0);
        assert!(proposal.reason.is_empty());
        assert!(proposal.heuristics.is_empty());
    }

    #[test]
    fn resolve_auto_directs_single_step_requests() {
        let decision =
            OrchestrationPlanner::resolve_auto("fix typo in the readme", 1, AgentAutonomy::Manual);
        assert_eq!(decision.strategy, AgentExecutionStrategy::Direct);
        assert!(decision.confidence >= 0.8);
    }

    #[test]
    fn resolve_auto_orchestrates_multi_file_requests() {
        let decision = OrchestrationPlanner::resolve_auto(
            "refactor across all files in parallel with subagents",
            5,
            AgentAutonomy::Manual,
        );
        assert_eq!(decision.strategy, AgentExecutionStrategy::Orchestrate);
    }

    #[test]
    fn plan_from_tasks_preserves_explanation() {
        let plan = OrchestrationPlanner::plan_from_tasks(
            "Refactor",
            vec![OrchestrationTask::new(TaskId::new("t1"), "T1", "Do it")],
            Some("Split into waves".to_string()),
        );
        assert_eq!(plan.explanation.as_deref(), Some("Split into waves"));
        assert_eq!(plan.tasks.len(), 1);
    }

    #[test]
    fn single_task_plan_contains_one_task() {
        let plan = OrchestrationPlanner::single_task_plan(
            "Direct",
            OrchestrationTask::new(TaskId::new("task"), "Task", "Do work"),
        );
        assert_eq!(plan.tasks.len(), 1);
        assert_eq!(plan.tasks[0].id, TaskId::new("task"));
    }
}
