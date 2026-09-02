use agent_settings::AgentExecutionStrategy;
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum TurnPolicySource {
    Configured,
    Automatic,
    AutomaticFallback,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum AgentToolProfile {
    Direct,
    Plan,
    Orchestrate,
}

impl AgentToolProfile {
    pub fn for_strategy(strategy: AgentExecutionStrategy) -> Self {
        match strategy {
            AgentExecutionStrategy::Direct | AgentExecutionStrategy::Auto => Self::Direct,
            AgentExecutionStrategy::Plan => Self::Plan,
            AgentExecutionStrategy::Orchestrate => Self::Orchestrate,
        }
    }
}

/// Immutable execution contract resolved before a model turn starts.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ResolvedTurnPolicy {
    pub configured_strategy: AgentExecutionStrategy,
    pub strategy: AgentExecutionStrategy,
    pub source: TurnPolicySource,
    pub tool_profile: AgentToolProfile,
    pub auto_decision: Option<AutoPolicyDecision>,
}

impl ResolvedTurnPolicy {
    pub fn configured(configured_strategy: AgentExecutionStrategy) -> Self {
        let (strategy, source) = if configured_strategy == AgentExecutionStrategy::Auto {
            (
                AgentExecutionStrategy::Direct,
                TurnPolicySource::AutomaticFallback,
            )
        } else {
            (configured_strategy, TurnPolicySource::Configured)
        };

        Self {
            configured_strategy,
            strategy,
            source,
            tool_profile: AgentToolProfile::for_strategy(strategy),
            auto_decision: None,
        }
    }

    pub fn automatic(decision: AutoPolicyDecision) -> Self {
        let strategy = match decision.strategy {
            AgentExecutionStrategy::Auto => AgentExecutionStrategy::Direct,
            strategy => strategy,
        };

        Self {
            configured_strategy: AgentExecutionStrategy::Auto,
            strategy,
            source: TurnPolicySource::Automatic,
            tool_profile: AgentToolProfile::for_strategy(strategy),
            auto_decision: Some(decision),
        }
    }
}

/// Runtime capabilities and already-known work shape available to the router.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AutoPolicyContext {
    /// Number of independently executable work items already known by the caller.
    /// This is normally zero before the first model completion and greater than
    /// zero when routing an already-decomposed batch as a compatibility fallback.
    pub work_item_count: usize,
    pub available_tool_count: Option<usize>,
    pub can_orchestrate: bool,
}

/// Tunable routing weights and thresholds owned by the orchestration crate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AutoPolicyConfig {
    pub planning_keyword_weight: f32,
    pub orchestration_keyword_weight: f32,
    pub execution_keyword_weight: f32,
    pub direct_keyword_weight: f32,
    pub short_prompt_direct_bonus: f32,
    pub multi_work_item_bonus: f32,
    pub large_work_item_bonus: f32,
    pub structured_list_bonus: f32,
    pub explicit_plan_score: f32,
    pub orchestration_route_threshold: f32,
    pub planning_route_threshold: f32,
    pub planning_execution_ceiling: f32,
    pub inferred_confidence_floor: f32,
    pub direct_confidence_floor: f32,
    pub explicit_confidence: f32,
    pub capability_fallback_confidence: f32,
    pub requested_plan_confidence: f32,
    pub multi_work_item_threshold: usize,
    pub large_work_item_threshold: usize,
    pub structured_list_item_threshold: usize,
    pub short_prompt_character_limit: usize,
    pub short_prompt_line_limit: usize,
    pub tool_count_normalization_cap: usize,
}

impl Default for AutoPolicyConfig {
    fn default() -> Self {
        Self {
            planning_keyword_weight: 0.22,
            orchestration_keyword_weight: 0.24,
            execution_keyword_weight: 0.2,
            direct_keyword_weight: 0.35,
            short_prompt_direct_bonus: 0.25,
            multi_work_item_bonus: 0.35,
            large_work_item_bonus: 0.65,
            structured_list_bonus: 0.55,
            explicit_plan_score: 0.85,
            orchestration_route_threshold: 0.55,
            planning_route_threshold: 0.55,
            planning_execution_ceiling: 0.4,
            inferred_confidence_floor: 0.72,
            direct_confidence_floor: 0.8,
            explicit_confidence: 0.98,
            capability_fallback_confidence: 0.9,
            requested_plan_confidence: 0.9,
            multi_work_item_threshold: 2,
            large_work_item_threshold: 4,
            structured_list_item_threshold: 3,
            short_prompt_character_limit: 100,
            short_prompt_line_limit: 3,
            tool_count_normalization_cap: 20,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct PromptSignals {
    plan_only: bool,
    direct: bool,
    orchestrate: bool,
    requests_plan: bool,
    executes_after_plan: bool,
}

#[derive(Clone, Copy, Debug, Default)]
struct PolicyScores {
    planning: f32,
    orchestration: f32,
    execution: f32,
    direct: f32,
}

/// Heuristic-driven policy engine that chooses an execution strategy before a turn starts.
pub struct AutoPolicyEngine;

impl AutoPolicyEngine {
    pub fn evaluate(prompt: &str, context: AutoPolicyContext) -> AutoPolicyDecision {
        Self::evaluate_with_config(prompt, context, &AutoPolicyConfig::default())
    }

    pub fn evaluate_with_config(
        prompt: &str,
        context: AutoPolicyContext,
        config: &AutoPolicyConfig,
    ) -> AutoPolicyDecision {
        let prompt_lower = prompt.to_lowercase();
        let signals = PromptSignals::detect(&prompt_lower);
        let scores = PolicyScores::calculate(prompt, &prompt_lower, signals, context, config);
        let heuristics = scores.heuristics(context, config);
        select_decision(signals, scores, context, config, heuristics)
    }
}

impl PromptSignals {
    fn detect(prompt: &str) -> Self {
        Self {
            plan_only: contains_any(prompt, PLAN_ONLY_PHRASES),
            direct: contains_any(prompt, DIRECT_OVERRIDE_PHRASES),
            orchestrate: contains_any(prompt, ORCHESTRATION_OVERRIDE_PHRASES),
            requests_plan: contains_any(prompt, PLAN_REQUEST_PHRASES),
            executes_after_plan: contains_any(prompt, PLAN_THEN_EXECUTE_PHRASES),
        }
    }
}

impl PolicyScores {
    fn calculate(
        original_prompt: &str,
        prompt: &str,
        signals: PromptSignals,
        context: AutoPolicyContext,
        config: &AutoPolicyConfig,
    ) -> Self {
        let planning_floor = if signals.plan_only {
            1.0
        } else if signals.requests_plan && !signals.executes_after_plan {
            config.explicit_plan_score
        } else {
            0.0
        };
        let planning = keyword_score(prompt, PLANNING_KEYWORDS, config.planning_keyword_weight)
            .max(planning_floor);

        let mut orchestration = keyword_score(
            prompt,
            ORCHESTRATION_KEYWORDS,
            config.orchestration_keyword_weight,
        );
        if signals.orchestrate {
            orchestration = 1.0;
        } else {
            orchestration += work_item_bonus(context.work_item_count, config);
            if original_prompt
                .lines()
                .filter(|line| is_list_item(line))
                .count()
                >= config.structured_list_item_threshold
            {
                orchestration += config.structured_list_bonus;
            }
        }

        let execution = keyword_score(prompt, EXECUTION_KEYWORDS, config.execution_keyword_weight);
        let mut direct = keyword_score(prompt, DIRECT_KEYWORDS, config.direct_keyword_weight);
        if original_prompt.chars().count() < config.short_prompt_character_limit
            && original_prompt.lines().count() <= config.short_prompt_line_limit
        {
            direct += config.short_prompt_direct_bonus;
        }
        if signals.direct {
            direct = 1.0;
        }

        Self {
            planning: planning.min(1.0),
            orchestration: orchestration.min(1.0),
            execution: execution.min(1.0),
            direct: direct.min(1.0),
        }
    }

    fn heuristics(
        self,
        context: AutoPolicyContext,
        config: &AutoPolicyConfig,
    ) -> HashMap<String, f32> {
        let mut heuristics = HashMap::default();
        heuristics.insert("planning_intent".into(), self.planning);
        heuristics.insert("orchestration_intent".into(), self.orchestration);
        heuristics.insert("execution_intent".into(), self.execution);
        heuristics.insert("direct_intent".into(), self.direct);
        heuristics.insert(
            "orchestration_capability".into(),
            if context.can_orchestrate { 1.0 } else { 0.0 },
        );
        let normalization_cap = config.tool_count_normalization_cap.max(1);
        if let Some(available_tool_count) = context.available_tool_count {
            heuristics.insert(
                "available_tools".into(),
                (available_tool_count.min(normalization_cap) as f32) / normalization_cap as f32,
            );
        }
        heuristics
    }
}

fn work_item_bonus(work_item_count: usize, config: &AutoPolicyConfig) -> f32 {
    if work_item_count >= config.large_work_item_threshold {
        config.large_work_item_bonus
    } else if work_item_count >= config.multi_work_item_threshold {
        config.multi_work_item_bonus
    } else {
        0.0
    }
}

fn select_decision(
    signals: PromptSignals,
    scores: PolicyScores,
    context: AutoPolicyContext,
    config: &AutoPolicyConfig,
    heuristics: HashMap<String, f32>,
) -> AutoPolicyDecision {
    if signals.plan_only {
        return decision(
            AgentExecutionStrategy::Plan,
            config.explicit_confidence,
            "The user explicitly requested planning or discussion without implementation",
            heuristics,
        );
    }
    if signals.direct {
        return decision(
            AgentExecutionStrategy::Direct,
            config.explicit_confidence,
            "The user explicitly requested direct execution without delegation",
            heuristics,
        );
    }
    if signals.orchestrate {
        return explicit_orchestration_decision(context, config, heuristics);
    }
    if signals.requests_plan && !signals.executes_after_plan {
        return decision(
            AgentExecutionStrategy::Plan,
            config.requested_plan_confidence,
            "The user requested a plan before any implementation begins",
            heuristics,
        );
    }
    if context.can_orchestrate && scores.orchestration >= config.orchestration_route_threshold {
        return decision(
            AgentExecutionStrategy::Orchestrate,
            scores.orchestration.max(config.inferred_confidence_floor),
            "The request contains multiple independent or parallelizable work items",
            heuristics,
        );
    }
    if scores.planning >= config.planning_route_threshold
        && scores.execution < config.planning_execution_ceiling
    {
        return decision(
            AgentExecutionStrategy::Plan,
            scores.planning.max(config.inferred_confidence_floor),
            "The request primarily asks for a plan, design, or proposal before execution",
            heuristics,
        );
    }

    decision(
        AgentExecutionStrategy::Direct,
        scores.direct.max(config.direct_confidence_floor),
        "The request is best handled as one direct agent turn",
        heuristics,
    )
}

fn explicit_orchestration_decision(
    context: AutoPolicyContext,
    config: &AutoPolicyConfig,
    heuristics: HashMap<String, f32>,
) -> AutoPolicyDecision {
    if context.can_orchestrate {
        decision(
            AgentExecutionStrategy::Orchestrate,
            config.explicit_confidence,
            "The user explicitly requested delegation or parallel orchestration",
            heuristics,
        )
    } else {
        decision(
            AgentExecutionStrategy::Direct,
            config.capability_fallback_confidence,
            "The request asks for orchestration, but this turn has no orchestration tool available",
            heuristics,
        )
    }
}

const PLAN_ONLY_PHRASES: &[&str] = &[
    "plan only",
    "only plan",
    "do not code",
    "don't code",
    "no code",
    "do not implement",
    "don't implement",
    "discussion only",
];
const DIRECT_OVERRIDE_PHRASES: &[&str] = &[
    "do not delegate",
    "don't delegate",
    "without delegation",
    "without subagents",
    "without sub-agents",
    "do not use subagents",
    "do not use sub-agents",
    "don't use subagents",
    "don't use sub-agents",
    "do it directly",
    "handle it directly",
    "do it yourself",
    "no subagents",
    "no sub-agents",
];
const ORCHESTRATION_OVERRIDE_PHRASES: &[&str] = &[
    "orchestrate",
    "orchestration",
    "use subagents",
    "use sub-agents",
    "spawn agents",
    "delegate this",
    "delegate the",
    "in parallel",
    "concurrently",
];
const PLAN_REQUEST_PHRASES: &[&str] = &[
    "create a plan",
    "make a plan",
    "draft a plan",
    "plan this",
    "plan the",
];
const PLAN_THEN_EXECUTE_PHRASES: &[&str] = &[
    "then implement",
    "and implement",
    "plan and implement",
    "then execute",
    "and execute",
];
const PLANNING_KEYWORDS: &[&str] = &[
    "plan",
    "architecture",
    "architectural",
    "design",
    "roadmap",
    "rfc",
    "proposal",
    "propose",
    "break down",
];
const ORCHESTRATION_KEYWORDS: &[&str] = &[
    "refactor across",
    "migrate",
    "all files",
    "multiple files",
    "multiple modules",
    "multiple crates",
    "step 1",
    "step 2",
    "phase 1",
    "phase 2",
    "test suite",
    "end to end",
];
const EXECUTION_KEYWORDS: &[&str] = &[
    "implement",
    "fix",
    "refactor",
    "migrate",
    "build",
    "port",
    "update",
    "change",
    "add",
    "remove",
];
const DIRECT_KEYWORDS: &[&str] = &[
    "fix typo",
    "explain this function",
    "what does",
    "one line",
    "quick fix",
    "simple change",
];

fn contains_any(prompt: &str, phrases: &[&str]) -> bool {
    phrases.iter().any(|phrase| prompt.contains(phrase))
}

fn keyword_score(prompt: &str, keywords: &[&str], weight: f32) -> f32 {
    keywords
        .iter()
        .filter(|keyword| contains_keyword(prompt, keyword))
        .fold(0.0_f32, |score, _| score + weight)
        .min(1.0)
}

fn contains_keyword(prompt: &str, keyword: &str) -> bool {
    if keyword.contains(' ')
        || !keyword
            .chars()
            .all(|character| character.is_ascii_alphabetic())
    {
        return prompt.contains(keyword);
    }

    prompt.match_indices(keyword).any(|(start, _)| {
        let end = start + keyword.len();
        let starts_at_boundary = prompt[..start]
            .chars()
            .next_back()
            .is_none_or(|character| !character.is_alphanumeric() && character != '_');
        let ends_at_boundary = prompt[end..]
            .chars()
            .next()
            .is_none_or(|character| !character.is_alphanumeric() && character != '_');
        starts_at_boundary && ends_at_boundary
    })
}

fn is_list_item(line: &str) -> bool {
    let line = line.trim_start();
    line.starts_with("- ")
        || line.starts_with("* ")
        || line
            .split_once('.')
            .or_else(|| line.split_once(')'))
            .is_some_and(|(prefix, _)| {
                !prefix.is_empty() && prefix.chars().all(|c| c.is_ascii_digit())
            })
}

fn decision(
    strategy: AgentExecutionStrategy,
    confidence: f32,
    reason: &str,
    heuristics: HashMap<String, f32>,
) -> AutoPolicyDecision {
    AutoPolicyDecision {
        strategy,
        confidence: confidence.clamp(0.0, 1.0),
        reason: reason.to_string(),
        heuristics,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn capable_context() -> AutoPolicyContext {
        AutoPolicyContext {
            available_tool_count: Some(12),
            can_orchestrate: true,
            ..Default::default()
        }
    }

    #[test]
    fn explicit_direct_request_overrides_complexity_signals() {
        let decision = AutoPolicyEngine::evaluate(
            "Refactor multiple crates, but do not delegate; handle it directly",
            capable_context(),
        );
        assert_eq!(decision.strategy, AgentExecutionStrategy::Direct);
    }

    #[test]
    fn negated_subagent_request_does_not_trigger_orchestration() {
        let decision = AutoPolicyEngine::evaluate(
            "Do not use subagents; implement these changes directly",
            capable_context(),
        );
        assert_eq!(decision.strategy, AgentExecutionStrategy::Direct);
    }

    #[test]
    fn explicit_no_code_request_routes_to_plan() {
        let decision = AutoPolicyEngine::evaluate(
            "Discuss the orchestration architecture only; do not code",
            capable_context(),
        );
        assert_eq!(decision.strategy, AgentExecutionStrategy::Plan);
    }

    #[test]
    fn explicit_parallel_request_routes_to_orchestration() {
        let decision = AutoPolicyEngine::evaluate(
            "Use subagents in parallel to handle these modules",
            capable_context(),
        );
        assert_eq!(decision.strategy, AgentExecutionStrategy::Orchestrate);
    }

    #[test]
    fn unavailable_orchestration_capability_falls_back_to_direct() {
        let decision = AutoPolicyEngine::evaluate(
            "Use subagents in parallel",
            AutoPolicyContext {
                available_tool_count: Some(3),
                can_orchestrate: false,
                ..Default::default()
            },
        );
        assert_eq!(decision.strategy, AgentExecutionStrategy::Direct);
        assert!(decision.reason.contains("no orchestration tool"));
    }

    #[test]
    fn decomposed_batch_routes_to_orchestration() {
        let decision = AutoPolicyEngine::evaluate(
            "Implement the requested changes",
            AutoPolicyContext {
                work_item_count: 4,
                ..capable_context()
            },
        );
        assert_eq!(decision.strategy, AgentExecutionStrategy::Orchestrate);
    }

    #[test]
    fn explicit_plan_request_routes_before_execution() {
        let decision = AutoPolicyEngine::evaluate(
            "Create a plan for improving the Git diff UI",
            capable_context(),
        );
        assert_eq!(decision.strategy, AgentExecutionStrategy::Plan);
    }

    #[test]
    fn plan_then_implement_is_not_trapped_in_plan_mode() {
        let decision = AutoPolicyEngine::evaluate(
            "Create a plan and then implement the Git diff changes",
            capable_context(),
        );
        assert_ne!(decision.strategy, AgentExecutionStrategy::Plan);
    }

    #[test]
    fn structured_work_list_routes_to_orchestration() {
        let decision = AutoPolicyEngine::evaluate(
            "Implement these changes:\n1. Update the parser\n2. Add UI state\n3. Run integration tests",
            capable_context(),
        );
        assert_eq!(decision.strategy, AgentExecutionStrategy::Orchestrate);
    }

    #[test]
    fn keywords_do_not_match_inside_unrelated_words() {
        assert!(!contains_keyword("explain this", "plan"));
        assert!(!contains_keyword("address the issue", "add"));
    }

    #[test]
    fn module_config_can_raise_the_orchestration_threshold() {
        let config = AutoPolicyConfig {
            orchestration_route_threshold: 0.9,
            ..Default::default()
        };
        let decision = AutoPolicyEngine::evaluate_with_config(
            "Handle these items:\n1. Parser\n2. UI state\n3. Integration tests",
            capable_context(),
            &config,
        );
        assert_eq!(decision.strategy, AgentExecutionStrategy::Direct);
    }

    #[test]
    fn configured_strategy_resolves_to_matching_tool_profile() {
        let policy = ResolvedTurnPolicy::configured(AgentExecutionStrategy::Plan);
        assert_eq!(policy.strategy, AgentExecutionStrategy::Plan);
        assert_eq!(policy.source, TurnPolicySource::Configured);
        assert_eq!(policy.tool_profile, AgentToolProfile::Plan);
        assert!(policy.auto_decision.is_none());
    }

    #[test]
    fn automatic_policy_keeps_the_decision_and_concrete_profile() {
        let decision = AutoPolicyEngine::evaluate(
            "Use subagents in parallel to review these modules",
            capable_context(),
        );
        let policy = ResolvedTurnPolicy::automatic(decision.clone());

        assert_eq!(policy.configured_strategy, AgentExecutionStrategy::Auto);
        assert_eq!(policy.strategy, AgentExecutionStrategy::Orchestrate);
        assert_eq!(policy.source, TurnPolicySource::Automatic);
        assert_eq!(policy.tool_profile, AgentToolProfile::Orchestrate);
        assert_eq!(policy.auto_decision.as_ref(), Some(&decision));
    }

    #[test]
    fn unresolved_auto_policy_fails_safe_to_direct() {
        let policy = ResolvedTurnPolicy::configured(AgentExecutionStrategy::Auto);
        assert_eq!(policy.strategy, AgentExecutionStrategy::Direct);
        assert_eq!(policy.source, TurnPolicySource::AutomaticFallback);
        assert_eq!(policy.tool_profile, AgentToolProfile::Direct);
    }
}
