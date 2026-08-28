use agent_client_protocol::schema::v1 as acp;
use agent_settings::AgentExecutionStrategy;
use anyhow::Result;
use gpui::{App, SharedString, Task, WeakEntity};
use language_model::LanguageModelToolResultContent;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::{
    AgentTool, NativePlan, NativePlanEntry, NativePlanStatus, Thread, ToolCallEventStream,
    ToolInput,
};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct UpdatePlanToolInput {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub explanation: Option<String>,
    #[serde(default)]
    pub proposal: bool,
    pub plan: Vec<UpdatePlanEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub struct UpdatePlanEntry {
    pub step: String,
    pub status: NativePlanStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum UpdatePlanToolOutput {
    Success { message: String },
    Error { error: String },
}

impl From<UpdatePlanToolOutput> for LanguageModelToolResultContent {
    fn from(output: UpdatePlanToolOutput) -> Self {
        match output {
            UpdatePlanToolOutput::Success { message } => message.into(),
            UpdatePlanToolOutput::Error { error } => error.into(),
        }
    }
}

pub struct UpdatePlanTool {
    thread: WeakEntity<Thread>,
}

impl UpdatePlanTool {
    pub fn new(thread: WeakEntity<Thread>) -> Self {
        Self { thread }
    }
}

impl AgentTool for UpdatePlanTool {
    type Input = UpdatePlanToolInput;
    type Output = UpdatePlanToolOutput;

    const NAME: &'static str = "update_plan";

    fn kind() -> acp::ToolKind {
        acp::ToolKind::Other
    }

    fn initial_title(
        &self,
        _input: Result<Self::Input, serde_json::Value>,
        _cx: &mut App,
    ) -> SharedString {
        "Update plan".into()
    }

    fn run(
        self: Arc<Self>,
        input: ToolInput<Self::Input>,
        event_stream: ToolCallEventStream,
        cx: &mut App,
    ) -> Task<Result<Self::Output, Self::Output>> {
        let result = input.recv();
        let thread = self.thread.clone();
        cx.spawn(async move |cx| {
            let input = result.await.map_err(|error| UpdatePlanToolOutput::Error {
                error: error.to_string(),
            })?;
            let proposal = input.proposal;
            let plan = NativePlan {
                explanation: input.explanation,
                entries: input
                    .plan
                    .into_iter()
                    .map(|entry| NativePlanEntry {
                        step: entry.step,
                        status: entry.status,
                    })
                    .collect(),
            };
            let update_result = cx.update(|cx| {
                thread.update(cx, |thread, cx| {
                    if thread.execution_strategy() == AgentExecutionStrategy::Plan && !proposal {
                        anyhow::bail!(
                            "update_plan is a progress checklist and is not available in Plan mode"
                        );
                    }
                    if proposal {
                        thread.propose_plan(plan, cx)
                    } else {
                        thread.update_plan(plan, cx)
                    }
                })
            });
            if let Err(error) = update_result.and_then(|result| result) {
                return Err(UpdatePlanToolOutput::Error {
                    error: error.to_string(),
                });
            }
            let title = if proposal {
                "Plan proposed"
            } else {
                "Plan updated"
            };
            event_stream.update_fields(acp::ToolCallUpdateFields::new().title(title));
            Ok(UpdatePlanToolOutput::Success {
                message: title.to_string(),
            })
        })
    }
}
