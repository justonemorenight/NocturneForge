use acp_thread::{AcpThread, AgentConnection};
use agent_client_protocol::schema::v1 as acp;
use agent_orchestration::{
    CancellationReason, CapabilitySnapshot, IsolatedWorktree, TaskExecutionContext,
    TaskExecutionOutput, WorkerHandle, WorkerHost, WorkerMetadata, WorkerTarget,
};
use anyhow::{Context as _, Result, bail};
use futures::future::LocalBoxFuture;
use gpui::{App, AsyncApp, Entity};
use project::{AgentId, Project};
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

fn task_prompt(
    task: &agent_orchestration::OrchestrationTask,
    dependencies: &[agent_orchestration::DependencyInput],
) -> String {
    let mut sections = vec![
        format!("Task: {}", task.label),
        format!(
            "Objective: {}",
            task.objective
                .as_deref()
                .unwrap_or("Execute the assigned task")
        ),
        format!(
            "Instructions:\n{}",
            crate::tools::task_execution_prompt_with_dependencies(task, dependencies)
        ),
    ];
    if !task.depends_on.is_empty() {
        sections.push(format!(
            "Dependencies: {}",
            task.depends_on
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    if !task.context_paths.is_empty() {
        sections.push(format!("Context paths: {}", task.context_paths.join(", ")));
    }
    if let Some(scope) = &task.scope {
        sections.push(format!("Allowed scope: {scope}"));
    }
    if let Some(tools) = &task.tools {
        sections.push(format!("Allowed tools: {}", tools.join(", ")));
    }
    if task.workspace_policy.read_only {
        sections.push("Workspace policy: read-only; do not modify files.".to_string());
    }
    sections.join("\n\n")
}

/// Resolves an agent identifier or display name from configured external agents.
pub fn resolve_configured_agent(
    available: &[(AgentId, Option<String>)],
    query: &str,
) -> Result<AgentId> {
    let trimmed = query.trim();
    if trimmed.is_empty() {
        bail!("agent query cannot be empty");
    }

    // 1. Exact match by AgentId (case-insensitive)
    let id_matches: Vec<_> = available
        .iter()
        .filter(|(id, _)| id.0.eq_ignore_ascii_case(trimmed))
        .map(|(id, _)| id.clone())
        .collect();
    if id_matches.len() == 1 {
        return id_matches
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("agent id match disappeared during resolution"));
    }

    // 2. Exact match by Display Name (case-insensitive)
    let display_matches: Vec<_> = available
        .iter()
        .filter(|(_, display)| {
            display
                .as_ref()
                .is_some_and(|name| name.eq_ignore_ascii_case(trimmed))
        })
        .map(|(id, _)| id.clone())
        .collect();
    if display_matches.len() == 1 {
        return display_matches
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("agent name match disappeared during resolution"));
    }

    // 3. Normalized alias resolution (e.g. "omp" -> "Oh My Pi", "opencode" -> "OpenCode")
    let alias_matches: Vec<_> = available
        .iter()
        .filter(|(id, display)| {
            let id_lower = id.0.to_lowercase();
            let display_lower = display
                .as_ref()
                .map(|d| d.to_lowercase())
                .unwrap_or_default();
            let q_lower = trimmed.to_lowercase();

            (q_lower == "omp" && (id_lower.contains("omp") || display_lower.contains("oh my pi")))
                || (q_lower == "opencode"
                    && (id_lower.contains("opencode") || display_lower.contains("opencode")))
        })
        .map(|(id, _)| id.clone())
        .collect();

    if alias_matches.len() == 1 {
        return alias_matches
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("agent alias match disappeared during resolution"));
    }

    let available_names: Vec<String> = available
        .iter()
        .map(|(id, display)| match display {
            Some(name) => format!("'{}' (id: '{}')", name, id.0),
            None => format!("'{}'", id.0),
        })
        .collect();

    if id_matches.len() > 1 || display_matches.len() > 1 || alias_matches.len() > 1 {
        bail!(
            "ambiguous agent identifier '{}'; matches multiple configured agents",
            trimmed
        );
    }

    bail!(
        "unknown agent '{}'; available configured agents: [{}]",
        trimmed,
        available_names.join(", ")
    );
}

/// Host for executing orchestration tasks via a live ACP connection (e.g. OMP, OpenCode).
pub struct AcpWorkerHost {
    connection: Rc<dyn AgentConnection>,
    project: Entity<Project>,
    work_dirs: util::path_list::PathList,
    isolated_worktree: Option<IsolatedWorktree>,
    read_only_enforced: bool,
    app: AsyncApp,
    thread_observer: Option<Rc<dyn Fn(Entity<AcpThread>, &mut App)>>,
    process_lease: Option<Rc<WorkerProcessLease>>,
}

struct WorkerProcessLease(Rc<dyn AgentConnection>);

impl Drop for WorkerProcessLease {
    fn drop(&mut self) {
        if let Err(error) = self.0.stop_worker_process() {
            log::error!("failed to stop dedicated ACP worker process: {error}");
        }
    }
}

impl AcpWorkerHost {
    pub fn new(connection: Rc<dyn AgentConnection>, project: Entity<Project>, cx: &App) -> Self {
        let work_dirs = project.read(cx).default_path_list(cx);
        Self {
            connection,
            project,
            work_dirs,
            isolated_worktree: None,
            read_only_enforced: false,
            app: cx.to_async(),
            thread_observer: None,
            process_lease: None,
        }
    }

    pub fn with_process_ownership(mut self) -> Self {
        self.process_lease = Some(Rc::new(WorkerProcessLease(self.connection.clone())));
        self
    }

    pub fn with_isolated_worktree(mut self, worktree: IsolatedWorktree) -> Self {
        self.work_dirs = util::path_list::PathList::new(&[worktree.worktree_path.as_path()]);
        self.isolated_worktree = Some(worktree);
        self
    }

    pub fn with_read_only_enforcement(mut self, enforced: bool) -> Self {
        self.read_only_enforced = enforced;
        self
    }

    pub fn with_thread_observer(
        mut self,
        observer: Rc<dyn Fn(Entity<AcpThread>, &mut App)>,
    ) -> Self {
        self.thread_observer = Some(observer);
        self
    }
}

impl WorkerHost for AcpWorkerHost {
    fn target(&self) -> WorkerTarget {
        WorkerTarget::acp(self.connection.agent_id().0.to_string())
    }

    fn capabilities(&self) -> CapabilitySnapshot {
        CapabilitySnapshot {
            can_resume: self.connection.supports_resume_session(),
            can_load_session: self.connection.supports_load_session(),
            can_cancel: true,
            can_stream_tokens: false,
            can_report_usage: false,
            can_enforce_read_only: self.read_only_enforced,
            can_select_model: true,
            can_select_mode: true,
            supports_worktree_isolation: self.isolated_worktree.is_some(),
            supported_models: Vec::new(),
            supported_modes: Vec::new(),
        }
    }

    fn create_worker(
        &self,
        task: &agent_orchestration::OrchestrationTask,
        _context: &TaskExecutionContext,
    ) -> LocalBoxFuture<'static, Result<Box<dyn WorkerHandle>>> {
        let connection = self.connection.clone();
        let project = self.project.clone();
        let work_dirs = self.work_dirs.clone();
        let isolated_worktree = self.isolated_worktree.clone();
        let target = self.target();
        let task_model = task.model_override.clone();
        let task_mode = task.mode.clone();
        let workspace_policy = task.workspace_policy.clone();
        let capabilities = self.capabilities();
        let app = self.app.clone();
        let thread_observer = self.thread_observer.clone();
        let process_lease = self.process_lease.clone();

        Box::pin(async move {
            let session_task =
                app.update(|cx| connection.clone().new_session(project, work_dirs, cx));

            let thread = session_task
                .await
                .context("failed to create ACP session for worker")?;
            let session_id = app.update(|cx| thread.read(cx).session_id().clone());
            if let Some(observer) = thread_observer {
                app.update(|cx| observer(thread.clone(), cx));
            }

            // Set requested mode if provided
            if let Some(mode) = task_mode.as_ref() {
                let mode_selector = app
                    .update(|cx| connection.session_modes(&session_id, cx))
                    .ok_or_else(|| anyhow::anyhow!("ACP worker does not support session modes"))?;
                let mode_id = acp::SessionModeId::new(mode.as_str());
                let set_mode_task = app.update(|cx| mode_selector.set_mode(mode_id, cx));
                set_mode_task
                    .await
                    .context("failed to set requested ACP session mode")?;
            }

            // Set requested model if provided
            if let Some(model) = task_model.as_ref() {
                let model_selector = connection.model_selector(&session_id).ok_or_else(|| {
                    anyhow::anyhow!("ACP worker does not support model selection")
                })?;
                let model_id = acp_thread::AgentModelId::new(model.as_str());
                let set_model_task = app.update(|cx| model_selector.select_model(model_id, cx));
                set_model_task
                    .await
                    .context("failed to set requested ACP model")?;
            }

            let mut metadata = WorkerMetadata::new(target);
            metadata.model = task_model;
            metadata.mode = task_mode;
            metadata.workspace_policy = workspace_policy;
            metadata.capabilities = capabilities;
            if let Some(worktree) = isolated_worktree {
                metadata.worktree_path = Some(worktree.worktree_path.display().to_string());
                metadata.baseline_commit = Some(worktree.baseline_commit);
            }

            Ok(Box::new(AcpWorkerHandle {
                connection,
                session_id,
                thread,
                metadata,
                app,
                prompt_active: Arc::new(AtomicBool::new(false)),
                _process_lease: process_lease,
            }) as Box<dyn WorkerHandle>)
        })
    }

    fn resume_worker(
        &self,
        session_id: &acp::SessionId,
        task: &agent_orchestration::OrchestrationTask,
        _context: &TaskExecutionContext,
    ) -> LocalBoxFuture<'static, Result<Box<dyn WorkerHandle>>> {
        let connection = self.connection.clone();
        let project = self.project.clone();
        let work_dirs = self.work_dirs.clone();
        let isolated_worktree = self.isolated_worktree.clone();
        let target = self.target();
        let session_id = session_id.clone();
        let task_model = task.model_override.clone();
        let task_mode = task.mode.clone();
        let workspace_policy = task.workspace_policy.clone();
        let capabilities = self.capabilities();
        let app = self.app.clone();
        let thread_observer = self.thread_observer.clone();
        let process_lease = self.process_lease.clone();

        Box::pin(async move {
            if !connection.supports_resume_session() && !connection.supports_load_session() {
                bail!("agent does not support resuming or loading existing sessions");
            }

            let resume_task = app.update(|cx| {
                if connection.supports_resume_session() {
                    connection.clone().resume_session(
                        session_id.clone(),
                        project,
                        work_dirs,
                        None,
                        cx,
                    )
                } else {
                    connection.clone().load_session(
                        session_id.clone(),
                        project,
                        work_dirs,
                        None,
                        cx,
                    )
                }
            });

            let thread = resume_task.await.context("failed to resume ACP session")?;
            if let Some(observer) = thread_observer {
                app.update(|cx| observer(thread.clone(), cx));
            }

            if let Some(mode) = task_mode.as_ref() {
                let mode_selector = app
                    .update(|cx| connection.session_modes(&session_id, cx))
                    .ok_or_else(|| anyhow::anyhow!("ACP worker does not support session modes"))?;
                let set_mode_task = app.update(|cx| {
                    mode_selector.set_mode(acp::SessionModeId::new(mode.as_str()), cx)
                });
                set_mode_task
                    .await
                    .context("failed to restore requested ACP session mode")?;
            }
            if let Some(model) = task_model.as_ref() {
                let model_selector = connection.model_selector(&session_id).ok_or_else(|| {
                    anyhow::anyhow!("ACP worker does not support model selection")
                })?;
                let set_model_task = app.update(|cx| {
                    model_selector.select_model(acp_thread::AgentModelId::new(model.as_str()), cx)
                });
                set_model_task
                    .await
                    .context("failed to restore requested ACP model")?;
            }

            let mut metadata = WorkerMetadata::new(target);
            metadata.model = task_model;
            metadata.mode = task_mode;
            metadata.workspace_policy = workspace_policy;
            metadata.capabilities = capabilities;
            if let Some(worktree) = isolated_worktree {
                metadata.worktree_path = Some(worktree.worktree_path.display().to_string());
                metadata.baseline_commit = Some(worktree.baseline_commit);
            }

            Ok(Box::new(AcpWorkerHandle {
                connection,
                session_id,
                thread,
                metadata,
                app,
                prompt_active: Arc::new(AtomicBool::new(false)),
                _process_lease: process_lease,
            }) as Box<dyn WorkerHandle>)
        })
    }
}

/// Handle to a live ACP session executing tasks.
pub struct AcpWorkerHandle {
    connection: Rc<dyn AgentConnection>,
    session_id: acp::SessionId,
    #[allow(dead_code)]
    thread: Entity<AcpThread>,
    metadata: WorkerMetadata,
    app: AsyncApp,
    prompt_active: Arc<AtomicBool>,
    _process_lease: Option<Rc<WorkerProcessLease>>,
}

impl WorkerHandle for AcpWorkerHandle {
    fn target(&self) -> &WorkerTarget {
        &self.metadata.target
    }

    fn session_id(&self) -> Option<&acp::SessionId> {
        Some(&self.session_id)
    }

    fn metadata(&self) -> &WorkerMetadata {
        &self.metadata
    }

    fn execute(
        &mut self,
        context: TaskExecutionContext,
    ) -> LocalBoxFuture<'static, Result<TaskExecutionOutput>> {
        let connection = self.connection.clone();
        let session_id = self.session_id.clone();
        let thread = self.thread.clone();
        let app = self.app.clone();
        let prompt_active = self.prompt_active.clone();
        let prompt_text = task_prompt(&context.task, &context.dependency_inputs);
        let mut metadata = self.metadata.clone();

        prompt_active.store(true, Ordering::SeqCst);
        Box::pin(async move {
            let result = async {
                metadata.version = connection.agent_version().map(|version| version.to_string());
                if let Some(selector) = connection.model_selector(&session_id) {
                    match app.update(|cx| selector.selected_model(cx)).await {
                        Ok(model) => metadata.model = Some(model.id.to_string()),
                        Err(error) => log::debug!("ACP worker model metadata unavailable: {error}"),
                    }
                }
                metadata.mode = app.update(|cx| {
                    connection.session_modes(&session_id, cx)
                        .map(|modes| modes.current_mode().to_string())
                }).or(metadata.mode);
                context.reporter.report_worker_started(Some(session_id.clone()), metadata.clone());
                let message_start_index = app.update(|cx| thread.read(cx).entries().len());
                let live_metadata = Rc::new(std::cell::RefCell::new(metadata));
                let (budget_sender, budget_receiver) = async_channel::bounded(1);
                let _activity_subscription = app.update(|cx| {
                    let reporter = context.reporter.clone();
                    let live_metadata = live_metadata.clone();
                    let session_id = session_id.clone();
                    let mut seen_tools = std::collections::HashSet::new();
                    let mut nested_sessions = std::collections::HashSet::new();
                    cx.subscribe(&thread, move |thread, event, cx| {
                        use acp_thread::{AcpThreadEvent, AgentThreadEntry, ToolCallStatus};
                        match event {
                            AcpThreadEvent::NewEntry | AcpThreadEvent::EntryUpdated(_) => {
                                let thread = thread.read(cx);
                                let index = match event {
                                    AcpThreadEvent::EntryUpdated(index) => *index,
                                    _ => thread.entries().len().saturating_sub(1),
                                };
                                if index < message_start_index { return; }
                                if let Some(AgentThreadEntry::ToolCall(tool)) = thread.entries().get(index) {
                                    if seen_tools.insert(tool.id.clone()) {
                                        if let Err(error) = reporter.report_tool_call_started(
                                            tool.tool_name.as_deref().unwrap_or("external_tool"),
                                        ) {
                                            if let Err(send_error) = budget_sender.try_send(error) {
                                                log::debug!("ACP budget stop already delivered: {send_error}");
                                            }
                                        }
                                    }
                                    match &tool.status {
                                        ToolCallStatus::WaitingForConfirmation { .. } => reporter.set_phase("awaiting_permission"),
                                        ToolCallStatus::Pending | ToolCallStatus::InProgress => reporter.set_phase("running_tool"),
                                        _ => { reporter.report_tool_call_finished(); reporter.set_phase("running"); }
                                    }
                                }
                            }
                            AcpThreadEvent::ModeUpdated(mode) => {
                                live_metadata.borrow_mut().mode = Some(mode.to_string());
                            }
                            AcpThreadEvent::SubagentSpawned(child_session) => {
                                nested_sessions.insert(child_session.clone());
                                live_metadata.borrow_mut().nested_agent_count = Some(nested_sessions.len() as u64);
                            }
                            AcpThreadEvent::ElicitationRequested(_) => reporter.set_phase("awaiting_user"),
                            AcpThreadEvent::ElicitationResponded(_) | AcpThreadEvent::ToolAuthorizationReceived(_) => reporter.set_phase("running"),
                            _ => return,
                        }
                        let mut metadata = live_metadata.borrow_mut();
                        metadata.last_activity_at = Some(chrono::Utc::now());
                        reporter.report_worker_started(Some(session_id.clone()), metadata.clone());
                    })
                });
                let content = acp::ContentBlock::Text(acp::TextContent::new(prompt_text));
                app.update(|cx| thread.update(cx, |thread, cx| {
                    thread.push_user_content_block(None, content.clone(), cx);
                }));
                let prompt_request = acp::PromptRequest::new(session_id.clone(), vec![content]);

                let prompt_task = app.update(|cx| connection.prompt(prompt_request, cx));
                let budget_stop = budget_receiver.recv();
                futures::pin_mut!(prompt_task, budget_stop);
                let response = match futures::future::select(prompt_task, budget_stop).await {
                    futures::future::Either::Left((response, _)) => response?,
                    futures::future::Either::Right((error, _)) => {
                        app.update(|cx| connection.cancel(&session_id, cx));
                        return Err(error.context("ACP budget monitor closed")?.into());
                    }
                };

                if response.stop_reason == acp::StopReason::Cancelled {
                    bail!("task execution was cancelled by worker");
                }
                anyhow::ensure!(response.stop_reason == acp::StopReason::EndTurn,
                    "ACP worker did not finish its task: {:?}", response.stop_reason);

                let tokens_used = response.usage.as_ref().map(|usage| usage.total_tokens);
                let mut metadata = live_metadata.borrow().clone();
                metadata.capabilities.can_report_usage = tokens_used.is_some();
                app.update(|cx| thread.update(cx, |thread, cx| thread.flush_pending_output(cx)));
                let output_text = app.update(|cx| {
                    let thread = thread.read(cx);
                    let output = thread
                        .entries()
                        .get(message_start_index..)
                        .unwrap_or_default()
                        .iter()
                        .filter_map(|entry| match entry {
                            acp_thread::AgentThreadEntry::AssistantMessage(message) => Some(message),
                            _ => None,
                        })
                        .flat_map(|message| &message.chunks)
                        .filter_map(|chunk| match chunk {
                            acp_thread::AssistantMessageChunk::Message { block, .. } => Some(block.to_markdown(cx)),
                            acp_thread::AssistantMessageChunk::Thought { .. } => None,
                        })
                        .collect::<Vec<_>>()
                        .join("\n")
                        .trim()
                        .to_string();
                    output
                });
                if output_text.is_empty() {
                    bail!(
                        "ACP worker completed with status {:?} but produced no assistant output",
                        response.stop_reason
                    );
                }

                let mut output = TaskExecutionOutput::new(output_text).with_session_id(session_id);

                output.tokens_used = tokens_used;
                output.worker_metadata = Some(metadata);
                Ok(output)
            }
            .await;
            prompt_active.store(false, Ordering::SeqCst);
            result
        })
    }

    fn cancel(&mut self, _reason: CancellationReason) -> LocalBoxFuture<'static, Result<()>> {
        let connection = self.connection.clone();
        let session_id = self.session_id.clone();
        let app = self.app.clone();
        let prompt_active = self.prompt_active.clone();

        Box::pin(async move {
            app.update(|cx| {
                connection.cancel(&session_id, cx);
            });
            prompt_active.store(false, Ordering::SeqCst);
            Ok(())
        })
    }

    fn close(&mut self) -> LocalBoxFuture<'static, Result<()>> {
        let connection = self.connection.clone();
        let session_id = self.session_id.clone();
        let app = self.app.clone();

        Box::pin(async move {
            if connection.supports_close_session() {
                let close_task = app.update(|cx| connection.close_session(&session_id, cx));
                close_task.await?;
            }
            Ok(())
        })
    }

    fn resume(
        &mut self,
        session_id: &acp::SessionId,
        context: TaskExecutionContext,
    ) -> LocalBoxFuture<'static, Result<TaskExecutionOutput>> {
        self.session_id = session_id.clone();
        self.execute(context)
    }
}

impl Drop for AcpWorkerHandle {
    fn drop(&mut self) {
        if self.prompt_active.swap(false, Ordering::SeqCst) {
            let connection = self.connection.clone();
            let session_id = self.session_id.clone();
            self.app.update(|cx| connection.cancel(&session_id, cx));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn task_prompt_includes_the_shared_verification_contract() {
        let mut task =
            agent_orchestration::OrchestrationTask::new("review", "Review", "Inspect code");
        task.acceptance_criteria = vec!["Report concrete evidence".to_string()];
        task.scope = Some("crates/agent".to_string());
        let prompt = task_prompt(&task, &[]);
        assert!(prompt.contains(agent_orchestration::VERIFICATION_START));
        assert!(prompt.contains(agent_orchestration::VERIFICATION_END));
        assert!(prompt.contains("Report concrete evidence"));
        assert!(prompt.contains("Allowed scope: crates/agent"));
    }

    #[test]
    fn test_resolve_configured_agent_by_id_and_name() {
        let available = vec![
            (AgentId("omp".into()), Some("Oh My Pi".into())),
            (AgentId("opencode".into()), Some("OpenCode".into())),
            (AgentId("custom-agent".into()), None),
        ];

        // 1. By exact ID
        assert_eq!(
            resolve_configured_agent(&available, "omp").unwrap(),
            AgentId("omp".into())
        );
        assert_eq!(
            resolve_configured_agent(&available, "opencode").unwrap(),
            AgentId("opencode".into())
        );

        // 2. By display name
        assert_eq!(
            resolve_configured_agent(&available, "Oh My Pi").unwrap(),
            AgentId("omp".into())
        );
        assert_eq!(
            resolve_configured_agent(&available, "OpenCode").unwrap(),
            AgentId("opencode".into())
        );

        // 3. Case-insensitive
        assert_eq!(
            resolve_configured_agent(&available, "oh my pi").unwrap(),
            AgentId("omp".into())
        );
        assert_eq!(
            resolve_configured_agent(&available, "CUSTOM-AGENT").unwrap(),
            AgentId("custom-agent".into())
        );

        // 4. Unknown agent returns clear error listing configured agents
        let err = resolve_configured_agent(&available, "unknown").unwrap_err();
        assert!(err.to_string().contains("unknown agent 'unknown'"));
        assert!(err.to_string().contains("Oh My Pi"));
        assert!(err.to_string().contains("OpenCode"));

        // 5. Empty returns error
        assert!(resolve_configured_agent(&available, "  ").is_err());
    }

    #[test]
    fn test_acp_worker_broker_enforces_read_only_and_feature_flags() {
        let broker_disabled = agent_orchestration::WorkerBroker::new(false);
        let mut acp_task =
            agent_orchestration::OrchestrationTask::new("omp-task", "OMP Task", "Do work in ACP");
        acp_task.target = WorkerTarget::acp("omp");

        // 1. Feature flag off -> rejects before dispatch
        let err = broker_disabled
            .validate_task_parameters(&acp_task)
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("ACP delegation is currently disabled")
        );

        // 2. Feature flag on, but read-only enforcement is false -> rejects read-only task
        let mut broker_enabled = agent_orchestration::WorkerBroker::new(true);

        struct MockOmpHost;
        impl WorkerHost for MockOmpHost {
            fn target(&self) -> WorkerTarget {
                WorkerTarget::acp("omp")
            }
            fn capabilities(&self) -> CapabilitySnapshot {
                CapabilitySnapshot::omp_default()
            }
            fn create_worker(
                &self,
                _task: &agent_orchestration::OrchestrationTask,
                _context: &TaskExecutionContext,
            ) -> LocalBoxFuture<'static, Result<Box<dyn WorkerHandle>>> {
                Box::pin(async { bail!("not implemented for mock") })
            }
            fn resume_worker(
                &self,
                _session_id: &acp::SessionId,
                _task: &agent_orchestration::OrchestrationTask,
                _context: &TaskExecutionContext,
            ) -> LocalBoxFuture<'static, Result<Box<dyn WorkerHandle>>> {
                Box::pin(async { bail!("not implemented for mock") })
            }
        }

        broker_enabled.register_host(WorkerTarget::acp("omp"), Rc::new(MockOmpHost));

        // Read-only task must be blocked because OMP cannot enforce read-only
        acp_task.workspace_policy.read_only = true;
        let err_ro = broker_enabled
            .validate_task_parameters(&acp_task)
            .unwrap_err();
        assert!(
            err_ro
                .to_string()
                .contains("cannot enforce read-only execution")
        );

        // Non read-only task passes validation
        acp_task.workspace_policy.read_only = false;
        acp_task.workspace_policy.isolation =
            agent_orchestration::WorkspaceIsolation::DedicatedWorktree;
        assert!(broker_enabled.validate_task_parameters(&acp_task).is_ok());
    }
}
