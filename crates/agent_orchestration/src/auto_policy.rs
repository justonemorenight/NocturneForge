use agent_settings::{AgentAutonomy, AgentExecutionStrategy};
use collections::HashMap;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Decision outcome produced by the automatic orchestration policy engine.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AutoPolicyDecision {
    pub strategy: AgentExecutionStrategy,
    pub confidence: f32,
    pub reason: String,
    pub heuristics: HashMap<String, f32>,
}

/// Heuristic-driven policy engine that chooses execution strategy automatically when in `Auto` mode.
pub struct AutoPolicyEngine;

impl AutoPolicyEngine {
    /// Evaluates prompt and context parameters to recommend an execution strategy.
    pub fn evaluate(
        prompt: &str,
        affected_file_count: usize,
        _tool_count: usize,
        _autonomy: AgentAutonomy,
    ) -> AutoPolicyDecision {
        let mut heuristics = HashMap::default();
        let prompt_lower = prompt.to_lowercase();

        // Check planning signals
        let planning_keywords = [
            "plan",
            "how to",
            "architecture",
            "design",
            "explore",
            "investigate",
            "propose",
            "roadmap",
            "rfc",
            "break down",
            "analyze",
        ];
        let mut planning_score: f32 = 0.0;
        for kw in planning_keywords {
            if prompt_lower.contains(kw) {
                planning_score += 0.25;
            }
        }
        planning_score = planning_score.min(1.0);
        heuristics.insert("planning_intent".into(), planning_score);

        // Check multi-task / orchestration signals
        let orchestration_keywords = [
            "refactor across",
            "migrate",
            "all files",
            "multiple",
            "in parallel",
            "step 1",
            "step by step",
            "phase 1",
            "subagent",
            "concurrently",
            "test suite",
        ];
        let mut orchestration_score: f32 = 0.0;
        for kw in orchestration_keywords {
            if prompt_lower.contains(kw) {
                orchestration_score += 0.3;
            }
        }
        if affected_file_count > 3 {
            orchestration_score += 0.35;
        }
        if prompt.lines().count() > 8 {
            orchestration_score += 0.2;
        }
        orchestration_score = orchestration_score.min(1.0);
        heuristics.insert("orchestration_intent".into(), orchestration_score);

        // Check single-action direct signals
        let direct_keywords = [
            "fix typo",
            "read",
            "look at",
            "explain this function",
            "what does",
            "just",
            "simple",
            "quick",
            "one line",
        ];
        let mut direct_score: f32 = 0.0;
        for kw in direct_keywords {
            if prompt_lower.contains(kw) {
                direct_score += 0.4;
            }
        }
        if prompt.len() < 80 {
            direct_score += 0.3;
        }
        direct_score = direct_score.min(1.0);
        heuristics.insert("direct_intent".into(), direct_score);

        let is_single_step = direct_score > 0.5
            || (affected_file_count <= 1
                && prompt.lines().count() <= 3
                && orchestration_score < 0.6);

        // Determine decision based on scores and single-step guardrails
        if is_single_step && direct_score >= planning_score {
            AutoPolicyDecision {
                strategy: AgentExecutionStrategy::Direct,
                confidence: 0.85,
                reason: "Request is single-step or self-contained without decomposable subtasks; executing directly".into(),
                heuristics,
            }
        } else if planning_score > 0.6 && planning_score > orchestration_score {
            AutoPolicyDecision {
                strategy: AgentExecutionStrategy::Plan,
                confidence: planning_score,
                reason: "User prompt focuses on planning, architecture, or exploration without immediate execution".into(),
                heuristics,
            }
        } else if orchestration_score > 0.5 && !is_single_step {
            AutoPolicyDecision {
                strategy: AgentExecutionStrategy::Orchestrate,
                confidence: orchestration_score,
                reason: "Request involves multi-step, multi-file, or parallelizable tasks suitable for orchestration".into(),
                heuristics,
            }
        } else {
            AutoPolicyDecision {
                strategy: AgentExecutionStrategy::Direct,
                confidence: 0.8,
                reason: "Request appears self-contained or single-step, executing directly".into(),
                heuristics,
            }
        }
    }
}
