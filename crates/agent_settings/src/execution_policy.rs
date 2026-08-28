use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Controls whether a thread performs work directly or coordinates other agents.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum AgentExecutionStrategy {
    /// Keep the current single-agent behavior.
    #[default]
    Direct,
    /// Produce and maintain a plan without delegating work.
    Plan,
    /// Delegate work according to the thread's orchestration policy.
    Orchestrate,
    /// Let the agent choose between direct execution and orchestration.
    Auto,
}

/// Controls how much confirmation is required while an execution strategy runs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum AgentAutonomy {
    /// Ask before starting delegated work.
    #[default]
    Manual,
    /// Approve the plan once, then allow its tasks to run.
    Supervised,
    /// Run the approved policy without per-task confirmation.
    Autonomous,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct AgentExecutionPolicy {
    #[serde(default)]
    pub strategy: AgentExecutionStrategy,
    #[serde(default)]
    pub autonomy: AgentAutonomy,
}

impl AgentExecutionStrategy {
    /// Returns the additional guidance that turns the selected strategy into
    /// an explicit contract for the model.
    pub fn system_prompt(self) -> Option<&'static str> {
        match self {
            Self::Direct | Self::Auto => None,
            Self::Plan => Some(
                "## Execution strategy: plan\n\n\
                    Stay in a conversational planning mode until the user explicitly ends it.\n\
                    First understand the goal, constraints, current state, and success criteria;\n\
                    then present a decision-complete implementation plan that another engineer\
                    could execute without inventing missing decisions. Treat planning as distinct\
                    from the progress checklist: do not mutate files or perform execution merely\
                    because the user asks for it while this mode is active. Ask only questions\
                    that materially change the plan, and keep the plan synchronized with the\
                    user's refinements. When the plan is ready, use `update_plan` with\
                    `proposal: true`; never use it to report execution progress in this mode.",
            ),
            Self::Orchestrate => Some(
                "## Execution strategy: orchestrate\n\n\
                    Act as an orchestrator: own the complete task from scope through verification.\n\
                    Enumerate the full work surface before dispatch, split substantial or\n\
                    independent units into focused subagent tasks, and dispatch independent work\n\
                    in parallel when possible. Give every task a concrete objective, narrow role,\n\
                    relevant context, and least-privilege tool allowlist; keep the parent thread\n\
                    coordinating rather than duplicating delegated work. Use the batch `tasks`\n\
                    form of `spawn_agent` for independent work so it starts concurrently. Verify\n\
                    each result against\n\
                    acceptance criteria, repair failures, and continue through dependent phases\n\
                    until closure. Report actionable progress and evidence, not just intent. Do not\n\
                    yield at an intermediate phase, and stop only when complete or when a concrete\n\
                    blocker requires the user's input. Respect an explicit request to work directly\n\
                    or not to delegate.",
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_preserve_direct_manual_behavior() {
        let policy = AgentExecutionPolicy::default();
        assert_eq!(policy.strategy, AgentExecutionStrategy::Direct);
        assert_eq!(policy.autonomy, AgentAutonomy::Manual);
    }

    #[test]
    fn policy_round_trips_with_stable_names() {
        let policy = AgentExecutionPolicy {
            strategy: AgentExecutionStrategy::Orchestrate,
            autonomy: AgentAutonomy::Supervised,
        };
        let json = serde_json::to_string(&policy).expect("policy should serialize");
        assert_eq!(
            json,
            r#"{"strategy":"orchestrate","autonomy":"supervised"}"#
        );
        assert_eq!(
            serde_json::from_str::<AgentExecutionPolicy>(&json).expect("policy should deserialize"),
            policy
        );
    }

    #[test]
    fn strategy_prompt_is_opt_in() {
        assert!(AgentExecutionStrategy::Direct.system_prompt().is_none());
        assert!(AgentExecutionStrategy::Auto.system_prompt().is_none());
        assert!(
            AgentExecutionStrategy::Plan
                .system_prompt()
                .is_some_and(|prompt| prompt.contains("decision-complete implementation plan"))
        );
        assert!(
            AgentExecutionStrategy::Orchestrate
                .system_prompt()
                .is_some_and(|prompt| prompt.contains("subagent tasks"))
        );
    }
}
