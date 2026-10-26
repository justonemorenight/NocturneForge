use super::*;

use acp_thread::{
    AgentConnection, AgentSessionInfo, AgentSessionList, AgentSessionListRequest,
    AgentSessionListResponse, ElicitationStore,
};

use action_log::ActionLog;

use agent_client_protocol::schema::{
    ProtocolVersion,
    v1::{self as acp, ErrorCode},
    v2 as acp_v2,
};

use agent_client_protocol::{
    Agent, Client, ConnectionTo, HandleDispatchFrom, Lines, Responder, V2Builder,
};

use anyhow::anyhow;

use async_channel;

use collections::{HashMap, HashSet};

use feature_flags::{AcpBetaFeatureFlag, FeatureFlagAppExt as _};

use futures::channel::mpsc;

use futures::future::Shared;

use futures::io::BufReader;

use futures::{AsyncBufReadExt as _, Future, FutureExt as _, StreamExt as _};

use project::agent_server_store::{
    AgentServerCommand, AgentServerStore, AllAgentServersSettings, CustomAgentServerSettings,
};

use project::{AgentId, Project};

use remote::remote_client::Interactive;

use serde::{Deserialize, Serialize};

use settings::{AgentConfigOptionValue, SettingsStore};

use std::path::PathBuf;

use std::process::Stdio;

use std::rc::Rc;

use std::sync::{Arc, Mutex};

use std::{any::Any, cell::RefCell};

use task::{Shell, ShellBuilder, SpawnInTerminal};

use util::{ResultExt as _, process::Child};

use util::path_list::PathList;

use anyhow::{Context as _, Result};

use gpui::{App, AppContext as _, AsyncApp, Entity, SharedString, Subscription, Task, WeakEntity};

use acp_thread::{AcpThread, LoadError, TerminalProviderEvent};

use terminal::TerminalBuilder;

use terminal::terminal_settings::{AlternateScroll, CursorShape};

use crate::{CLAUDE_AGENT_ID, CURSOR_ID, GEMINI_ID};

pub const GEMINI_TERMINAL_AUTH_METHOD_ID: &str = "spawn-gemini-cli";

const PARAMETERIZED_MODEL_PICKER_META_KEY: &str = "parameterizedModelPicker";

const CODEX_ACP_AGENT_ID: &str = "codex-acp";

// Older Claude Multi servers emit extension variants alongside the official subagent updates.
#[derive(Clone, Debug, Serialize, Deserialize, agent_client_protocol::JsonRpcNotification)]
#[notification(method = "session/update")]
#[serde(rename_all = "camelCase")]
struct SessionNotificationWithSubagents {
    session_id: acp_v2::SessionId,
    update: serde_json::Value,
}

const EXTERNAL_SUBAGENT_UPDATE_META_KEY: &str = "nocturne_external_subagent_update";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(
    tag = "sessionUpdate",
    rename_all = "snake_case",
    rename_all_fields = "camelCase"
)]
enum ExternalSubagentUpdate {
    SubagentUpdate {
        session_id: acp_v2::SessionId,
        #[serde(flatten)]
        patch: acp_thread::ExternalSubagentPatch,
    },
    SessionMessage {
        #[serde(flatten)]
        message: acp_thread::ExternalSessionMessage,
    },
    SessionMessageChunk {
        message_id: acp_v2::MessageId,
        sender_session_id: Option<acp_v2::SessionId>,
        recipient_session_id: Option<acp_v2::SessionId>,
        content: acp::ContentBlock,
    },
    SubagentSpawned {
        subagent_session_id: acp_v2::SessionId,
        name: String,
        task: String,
        prompt: Option<String>,
    },
    SubagentStateUpdate {
        subagent_session_id: acp_v2::SessionId,
        state: acp_thread::ExternalSubagentState,
    },
}

impl SessionNotificationWithSubagents {
    fn into_notification(self) -> Result<acp::SessionNotification> {
        let update = match self
            .update
            .get("sessionUpdate")
            .and_then(serde_json::Value::as_str)
        {
            Some(
                "subagent_spawned"
                | "subagent_state_update"
                | "subagent_update"
                | "session_message"
                | "session_message_chunk",
            ) => {
                let event: ExternalSubagentUpdate = serde_json::from_value(self.update)?;
                acp::SessionUpdate::SessionInfoUpdate(acp::SessionInfoUpdate::new().meta(
                    acp::Meta::from_iter([(
                        EXTERNAL_SUBAGENT_UPDATE_META_KEY.to_owned(),
                        serde_json::to_value(event)?,
                    )]),
                ))
            }
            _ => serde_json::from_value(self.update)?,
        };
        Ok(acp::SessionNotification::new(
            acp::SessionId::new(self.session_id.0),
            update,
        ))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, agent_client_protocol::JsonRpcNotification)]
#[notification(method = "_claude/sdkMessage")]
#[serde(rename_all = "camelCase")]
struct ClaudeSdkMessageNotification {
    session_id: acp_v2::SessionId,
    message: serde_json::Value,
}

async fn close_session_and_drain(
    connection: &ConnectionTo<Agent>,
    session_id: acp_v2::SessionId,
    supports_close: bool,
    dispatch_tx: &mpsc::UnboundedSender<ForegroundWork>,
) -> Result<()> {
    if supports_close {
        connection
            .send_request(acp::CloseSessionRequest::new(acp::SessionId::new(
                session_id.0,
            )))
            .block_task()
            .await
            .log_err();
    }
    drain_foreground_queue(dispatch_tx).await
}

pub struct AcpConnection {
    id: AgentId,
    telemetry_id: SharedString,
    agent_version: Option<SharedString>,
    claude_config_dir: Option<PathBuf>,
    connection: ConnectionTo<Agent>,
    dispatch_tx: mpsc::UnboundedSender<ForegroundWork>,
    sessions: Rc<RefCell<HashMap<acp_v2::SessionId, AcpSession>>>,
    pending_sessions: Rc<RefCell<HashMap<acp_v2::SessionId, PendingAcpSession>>>,
    session_notification_state: Rc<SessionNotificationState>,
    auth_methods: Vec<acp_v2::AuthMethod>,
    agent_server_store: WeakEntity<AgentServerStore>,
    agent_capabilities: acp::AgentCapabilities,
    request_elicitations: Entity<ElicitationStore>,
    defaults: AcpConnectionDefaults,
    child: RefCell<Option<Child>>,
    _process_sandbox: Option<sandbox::Sandbox>,
    worker_process_owned: bool,
    client_read_only: bool,
    terminal_sandbox_wrap: Option<acp_thread::SandboxWrap>,
    session_list: Option<Rc<AcpSessionList>>,
    debug_log: AcpDebugLog,
    _settings_subscription: Subscription,
    _runtime: runtime::ClientRuntime,
    _wait_task: Task<()>,
    _stderr_task: Shared<Task<()>>,
}

#[derive(Clone, Default)]
struct AcpConnectionDefaults {
    mode: Rc<RefCell<Option<acp_v2::SessionConfigValueId>>>,
    config_options: Rc<RefCell<HashMap<String, AgentConfigOptionValue>>>,
}

impl AcpConnectionDefaults {
    fn new(
        mode: Option<acp_v2::SessionConfigValueId>,
        config_options: HashMap<String, AgentConfigOptionValue>,
    ) -> Self {
        Self {
            mode: Rc::new(RefCell::new(mode)),
            config_options: Rc::new(RefCell::new(config_options)),
        }
    }

    fn mode(&self) -> Option<acp_v2::SessionConfigValueId> {
        self.mode.borrow().clone()
    }

    fn config_option(&self, config_id: &str) -> Option<AgentConfigOptionValue> {
        self.config_options.borrow().get(config_id).cloned()
    }

    fn set(
        &self,
        mode: Option<acp_v2::SessionConfigValueId>,
        config_options: HashMap<String, AgentConfigOptionValue>,
    ) {
        *self.mode.borrow_mut() = mode;
        *self.config_options.borrow_mut() = config_options;
    }

    fn refresh_from_settings(&self, agent_id: &AgentId, cx: &App) {
        let Some(settings_store) = cx.try_global::<SettingsStore>() else {
            self.set(None, HashMap::default());
            return;
        };
        let settings = settings_store.get::<AllAgentServersSettings>(None);
        let Some(agent_settings) = settings.get(agent_id.as_ref()) else {
            self.set(None, HashMap::default());
            return;
        };

        let default_config_options = match agent_settings {
            CustomAgentServerSettings::Custom {
                default_config_options,
                ..
            }
            | CustomAgentServerSettings::Registry {
                default_config_options,
                ..
            } => default_config_options.clone(),
        };
        self.set(
            agent_settings
                .default_mode()
                .map(acp_v2::SessionConfigValueId::new),
            default_config_options,
        );
    }

    fn observe_settings(&self, agent_id: AgentId, cx: &mut App) -> Subscription {
        if cx.try_global::<SettingsStore>().is_none() {
            return Subscription::new(|| {});
        }

        self.refresh_from_settings(&agent_id, cx);
        let defaults = self.clone();
        cx.observe_global::<SettingsStore>(move |cx| {
            defaults.refresh_from_settings(&agent_id, cx);
        })
    }
}

struct PendingAcpSession {
    thread: WeakEntity<AcpThread>,
    task: Shared<Task<Result<(), Arc<anyhow::Error>>>>,
}

struct SessionConfigResponse {
    modes: Option<acp::SessionModeState>,
    config_options: Option<Vec<acp::SessionConfigOption>>,
}

pub struct AcpSessionList {
    connection: ConnectionTo<Agent>,
    supports_delete: bool,
    updates_tx: async_channel::Sender<acp_thread::SessionListUpdate>,
    updates_rx: async_channel::Receiver<acp_thread::SessionListUpdate>,
}

impl AcpSessionList {
    fn new(connection: ConnectionTo<Agent>, supports_delete: bool) -> Self {
        let (tx, rx) = async_channel::unbounded();
        Self {
            connection,
            supports_delete,
            updates_tx: tx,
            updates_rx: rx,
        }
    }

    fn notify_update(&self) {
        self.updates_tx
            .try_send(acp_thread::SessionListUpdate::Refresh)
            .log_err();
    }

    pub(super) fn send_info_update(
        &self,
        session_id: acp_v2::SessionId,
        update: impl Into<acp_thread::AgentSessionInfoUpdate>,
    ) {
        let update = update.into();
        self.updates_tx
            .try_send(acp_thread::SessionListUpdate::SessionInfo { session_id, update })
            .log_err();
    }
}

impl AgentSessionList for AcpSessionList {
    fn list_sessions(
        &self,
        request: AgentSessionListRequest,
        cx: &mut App,
    ) -> Task<Result<AgentSessionListResponse>> {
        let conn = self.connection.clone();
        cx.foreground_executor().spawn(async move {
            let acp_request = acp::ListSessionsRequest::new()
                .cwd(request.cwd)
                .cursor(request.cursor);
            let response = conn
                .send_request(acp_request)
                .block_task()
                .await
                .map_err(map_acp_error)?;
            Ok(AgentSessionListResponse {
                sessions: response
                    .sessions
                    .into_iter()
                    .map(|s| AgentSessionInfo {
                        session_id: acp_v2::SessionId::new(s.session_id.0),
                        work_dirs: Some(work_dirs_from_session_info(
                            s.cwd,
                            s.additional_directories,
                        )),
                        title: s.title.map(Into::into),
                        updated_at: s.updated_at.and_then(|date_str| {
                            chrono::DateTime::parse_from_rfc3339(&date_str)
                                .ok()
                                .map(|dt| dt.with_timezone(&chrono::Utc))
                        }),
                        created_at: None,
                        meta: s.meta,
                    })
                    .collect(),
                next_cursor: response.next_cursor,
                meta: response.meta,
            })
        })
    }

    fn supports_delete(&self) -> bool {
        self.supports_delete
    }

    fn delete_session(&self, session_id: &acp_v2::SessionId, cx: &mut App) -> Task<Result<()>> {
        if !self.supports_delete() {
            return Task::ready(Err(anyhow::anyhow!("delete_session not supported")));
        }

        let conn = self.connection.clone();
        let updates_tx = self.updates_tx.clone();
        let session_id = acp_v2::SessionId::new(session_id.0.clone());
        cx.foreground_executor().spawn(async move {
            conn.send_request(acp::DeleteSessionRequest::new(acp::SessionId::new(
                session_id.0,
            )))
            .block_task()
            .await
            .map_err(map_acp_error)?;
            updates_tx
                .try_send(acp_thread::SessionListUpdate::Refresh)
                .log_err();
            Ok(())
        })
    }

    fn watch(
        &self,
        _cx: &mut App,
    ) -> Option<async_channel::Receiver<acp_thread::SessionListUpdate>> {
        Some(self.updates_rx.clone())
    }

    fn notify_refresh(&self) {
        self.notify_update();
    }

    fn into_any(self: Rc<Self>) -> Rc<dyn Any> {
        self
    }
}

pub async fn connect(
    agent_id: AgentId,
    project: Entity<Project>,
    command: AgentServerCommand,
    agent_server_store: WeakEntity<AgentServerStore>,
    default_mode: Option<acp_v2::SessionConfigValueId>,
    default_config_options: HashMap<String, AgentConfigOptionValue>,
    process_sandbox_policy: Option<sandbox::SandboxPolicy>,
    client_read_only: bool,
    terminal_sandbox_wrap: Option<acp_thread::SandboxWrap>,
    cx: &mut AsyncApp,
) -> Result<Rc<dyn AgentConnection>> {
    let conn = AcpConnection::stdio(
        agent_id,
        project,
        command.clone(),
        agent_server_store,
        default_mode,
        default_config_options,
        process_sandbox_policy,
        client_read_only,
        terminal_sandbox_wrap,
        cx,
    )
    .await?;
    Ok(Rc::new(conn) as _)
}

const MINIMUM_SUPPORTED_VERSION: ProtocolVersion = ProtocolVersion::V1;

pub(super) fn connect_client_future(
    name: &'static str,
    transport: impl agent_client_protocol::ConnectTo<Client> + 'static,
    dispatch_tx: mpsc::UnboundedSender<ForegroundWork>,
    connection_tx: futures::channel::oneshot::Sender<ConnectionTo<Agent>>,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> impl Future<Output = Result<(), acp::Error>> {
    async move {
        // Transport failures can bypass `on_close`; both paths share the same foreground cleanup.
        let cleanup = runtime::disconnect_cleanup(dispatch_tx.clone());
        // Each handler forwards its inputs onto the foreground dispatch queue.
        // The SDK requires the closure to be `Send`, so we move a clone of
        // `dispatch_tx` into each one.
        macro_rules! on_request {
            ($handler:ident) => {{
                let dispatch_tx = dispatch_tx.clone();
                async move |req, responder, _connection| {
                    enqueue_request(&dispatch_tx, req, responder, $handler);
                    Ok(())
                }
            }};
        }
        macro_rules! on_notification {
            ($handler:ident) => {{
                let dispatch_tx = dispatch_tx.clone();
                async move |notif, _connection| {
                    enqueue_notification(&dispatch_tx, notif, $handler);
                    Ok(())
                }
            }};
        }

        let future = Client
            .builder()
            .name(name)
            // --- Request handlers (agent→client) ---
            .on_receive_request(
                on_request!(handle_request_permission),
                agent_client_protocol::on_receive_request!(),
            )
            .on_receive_request(
                on_request!(handle_write_text_file),
                agent_client_protocol::on_receive_request!(),
            )
            .on_receive_request(
                on_request!(handle_read_text_file),
                agent_client_protocol::on_receive_request!(),
            )
            .on_receive_request(
                on_request!(handle_create_terminal),
                agent_client_protocol::on_receive_request!(),
            )
            .on_receive_request(
                on_request!(handle_kill_terminal),
                agent_client_protocol::on_receive_request!(),
            )
            .on_receive_request(
                on_request!(handle_release_terminal),
                agent_client_protocol::on_receive_request!(),
            )
            .on_receive_request(
                on_request!(handle_terminal_output),
                agent_client_protocol::on_receive_request!(),
            )
            .on_receive_request(
                on_request!(handle_wait_for_terminal_exit),
                agent_client_protocol::on_receive_request!(),
            )
            .on_receive_request(
                on_request!(handle_create_elicitation),
                agent_client_protocol::on_receive_request!(),
            )
            // --- Notification handlers (agent→client) ---
            .on_receive_notification(
                on_notification!(handle_session_notification_with_subagents),
                agent_client_protocol::on_receive_notification!(),
            )
            .on_receive_notification(
                on_notification!(handle_complete_elicitation),
                agent_client_protocol::on_receive_notification!(),
            )
            .on_receive_notification(
                on_notification!(handle_claude_sdk_message),
                agent_client_protocol::on_receive_notification!(),
            )
            .on_close({
                let cleanup = cleanup.clone();
                async move |_connection: ConnectionTo<Agent>| cleanup.await
            })
            .connect_with(
                transport,
                move |connection: ConnectionTo<Agent>| async move {
                    runtime::publish_connection(connection, connection_tx, shutdown).await
                },
            );
        runtime::finish_connection(future, cleanup).await
    }
}

fn client_capabilities_for_agent(
    agent_id: &AgentId,
    read_only: bool,
    beta_features_enabled: bool,
) -> acp::ClientCapabilities {
    let mut meta = acp::Meta::from_iter([
        ("terminal_output".into(), true.into()),
        ("terminal-auth".into(), true.into()),
    ]);

    if agent_id.as_ref() == CURSOR_ID {
        meta.insert(PARAMETERIZED_MODEL_PICKER_META_KEY.into(), true.into());
    }
    if agent_id.as_ref() == CODEX_ACP_AGENT_ID {
        meta.insert(
            "jetbrains".into(),
            serde_json::json!({
                "air": {
                    "version": 1,
                    "capabilities": ["agentFileChangeReport"],
                },
            }),
        );
    }

    let mut session_capabilities = acp::ClientSessionCapabilities::new()
        .config_options(
            acp::SessionConfigOptionsCapabilities::new()
                .boolean(acp::BooleanConfigOptionCapabilities::new()),
        )
        .notices(acp::NoticeCapabilities::new());
    if beta_features_enabled {
        session_capabilities = session_capabilities.compaction(acp::CompactionCapabilities::new());
    }

    acp::ClientCapabilities::new()
        .fs(acp::FileSystemCapabilities::new()
            .read_text_file(true)
            .write_text_file(!read_only))
        .terminal(true)
        .subagents(acp::SubagentCapabilities::new())
        .auth(acp::AuthCapabilities::new().terminal(true))
        .session(session_capabilities)
        .elicitation(
            acp::ElicitationCapabilities::new()
                .form(acp::ElicitationFormCapabilities::new())
                .url(acp::ElicitationUrlCapabilities::new()),
        )
        .meta(meta)
}

impl AcpConnection {
    /// Binds supported v2 client handlers to the existing foreground owners.
    /// This does not change the protocol used by the connection's transport.
    pub fn with_v2_client_ingress<Handler: HandleDispatchFrom<Agent>>(
        &self,
        builder: V2Builder<Client, Handler>,
    ) -> V2Builder<Client, impl HandleDispatchFrom<Agent> + use<Handler>> {
        with_v2_client_ingress(builder, self.dispatch_tx.clone())
    }

    pub fn subscribe_debug_messages(
        &self,
    ) -> (
        Vec<AcpDebugMessage>,
        async_channel::Receiver<AcpDebugMessage>,
    ) {
        self.debug_log.subscribe()
    }

    pub async fn stdio(
        agent_id: AgentId,
        project: Entity<Project>,
        command: AgentServerCommand,
        agent_server_store: WeakEntity<AgentServerStore>,
        default_mode: Option<acp_v2::SessionConfigValueId>,
        default_config_options: HashMap<String, AgentConfigOptionValue>,
        process_sandbox_policy: Option<sandbox::SandboxPolicy>,
        client_read_only: bool,
        terminal_sandbox_wrap: Option<acp_thread::SandboxWrap>,
        cx: &mut AsyncApp,
    ) -> Result<Self> {
        let root_dir = project.read_with(cx, |project, cx| {
            project
                .default_path_list(cx)
                .ordered_paths()
                .next()
                .cloned()
        });
        let original_command = command.clone();
        // A worker connection always receives a terminal sandbox wrapper from
        // the orchestration host. Keep ownership explicit so process cleanup
        // does not depend on whether a process-level sandbox was configured.
        let worker_process_owned = terminal_sandbox_wrap.is_some();
        let (path, args, env) = project
            .read_with(cx, |project, cx| {
                project.remote_client().and_then(|client| {
                    let template = client
                        .read(cx)
                        .build_command(
                            Some(command.path.display().to_string()),
                            &command.args,
                            &command.env.clone().into_iter().flatten().collect(),
                            root_dir.as_ref().map(|path| path.display().to_string()),
                            None,
                            Interactive::No,
                        )
                        .log_err()?;
                    Some((template.program, template.args, template.env))
                })
            })
            .unwrap_or_else(|| {
                (
                    command.path.display().to_string(),
                    command.args,
                    command.env.unwrap_or_default(),
                )
            });
        let claude_config_dir = if agent_id.as_ref() == CLAUDE_AGENT_ID
            && project.read_with(cx, |project, _cx| project.is_local())
        {
            Some(
                env.get("CLAUDE_CONFIG_DIR")
                    .filter(|path| !path.is_empty())
                    .map(PathBuf::from)
                    .unwrap_or_else(|| util::paths::home_dir().join(".claude")),
            )
        } else {
            None
        };

        let builder = ShellBuilder::new(&Shell::System, cfg!(windows)).non_interactive();
        let cwd = project.read_with(cx, |project, _cx| {
            project.is_local().then(|| root_dir.clone()).flatten()
        });
        let mut child = builder.build_std_command(Some(path.clone()), &args);
        child.envs(env.clone());
        if let Some(cwd) = &cwd {
            child.current_dir(cwd);
        }
        let mut process_sandbox = process_sandbox_policy
            .map(sandbox::Sandbox::new)
            .transpose()
            .context("failed to initialize ACP worker sandbox")?;
        if let Some(sandbox) = process_sandbox.as_mut() {
            let command = sandbox::CommandAndArgs {
                program: child.get_program().to_string_lossy().into_owned(),
                args: child
                    .get_args()
                    .map(|argument| argument.to_string_lossy().into_owned())
                    .collect(),
                env: env
                    .iter()
                    .map(|(name, value)| (name.clone(), value.clone()))
                    .collect(),
                cwd,
            };
            let wrapped = sandbox
                .wrap(&command)
                .await
                .context("failed to wrap ACP worker process in sandbox")?;
            child = std::process::Command::new(&wrapped.program);
            child.args(&wrapped.args).envs(&wrapped.env);
            if let Some(cwd) = &wrapped.cwd {
                child.current_dir(cwd);
            }
        }
        let mut pending_child = PendingAcpProcess(Some(Child::spawn(
            child,
            Stdio::piped(),
            Stdio::piped(),
            Stdio::piped(),
        )?));
        let child = pending_child
            .0
            .as_mut()
            .context("ACP process was not created")?;

        let stdout = child.stdout.take().context("Failed to take stdout")?;
        let stdin = child.stdin.take().context("Failed to take stdin")?;
        let stderr = child.stderr.take().context("Failed to take stderr")?;
        log::debug!("Spawning external agent server: {:?}, {:?}", path, args);
        log::trace!("Spawned (pid: {})", child.id());

        let sessions = Rc::new(RefCell::new(HashMap::default()));
        let session_notification_state = Rc::new(SessionNotificationState::default());
        let debug_log = AcpDebugLog::default();

        let (release_channel, version): (Option<&str>, String) = cx.update(|cx| {
            (
                release_channel::ReleaseChannel::try_global(cx)
                    .map(|release_channel| release_channel.display_name()),
                release_channel::AppVersion::global(cx).to_string(),
            )
        });

        let client_session_list: Rc<RefCell<Option<Rc<AcpSessionList>>>> =
            Rc::new(RefCell::new(None));
        let request_elicitations = cx.new(|_| ElicitationStore::default());
        let beta_features_enabled = cx.update(|cx| cx.has_flag::<AcpBetaFeatureFlag>());
        let client_capabilities =
            client_capabilities_for_agent(&agent_id, client_read_only, beta_features_enabled);

        let dispatch_context = ClientContext {
            sessions: sessions.clone(),
            session_list: client_session_list.clone(),
            request_elicitations: request_elicitations.clone(),
            active_request_scopes: None,
            session_notification_state: session_notification_state.clone(),
            read_only: client_read_only,
            terminal_sandbox_wrap: terminal_sandbox_wrap.clone(),
            pending_session_updates: None,
        };
        let (mut runtime, shutdown) = runtime::ClientRuntime::new(dispatch_context, cx);
        let dispatch_tx = runtime.sender();

        let incoming_lines = futures::io::BufReader::new(stdout).lines();
        let tapped_incoming = incoming_lines.inspect({
            let debug_log = debug_log.clone();
            move |result| match result {
                Ok(line) => debug_log.record_line(AcpDebugMessageDirection::Incoming, line),
                Err(err) => {
                    log::warn!("ACP transport read error: {err}");
                }
            }
        });

        let tapped_outgoing = futures::sink::unfold(
            (Box::pin(stdin), debug_log.clone()),
            async move |(mut writer, debug_log), line: String| {
                use futures::AsyncWriteExt;
                debug_log.record_line(AcpDebugMessageDirection::Outgoing, &line);
                let mut bytes = line.into_bytes();
                bytes.push(b'\n');
                writer.write_all(&bytes).await?;
                Ok::<_, std::io::Error>((writer, debug_log))
            },
        );

        let transport = Lines::new(tapped_outgoing, tapped_incoming);

        let stderr_task = cx
            .background_spawn({
                let debug_log = debug_log.clone();
                async move {
                    let mut stderr = BufReader::new(stderr);
                    let mut line = String::new();
                    while let Ok(n) = stderr.read_line(&mut line).await
                        && n > 0
                    {
                        let trimmed = line.trim_end_matches(['\n', '\r']);
                        log::warn!("agent stderr: {trimmed}");
                        debug_log.record_line(AcpDebugMessageDirection::Stderr, trimmed);
                        line.clear();
                    }
                }
            })
            .shared();

        let (connection_tx, connection_rx) = futures::channel::oneshot::channel();
        let connection_future = connect_client_future(
            "zed",
            transport,
            dispatch_tx.clone(),
            connection_tx,
            runtime::shutdown_requested(shutdown),
        );
        runtime.start_io(connection_future, cx);

        let connection_rx = async move {
            connection_rx
                .await
                .context("Failed to receive ACP connection handle")
        }
        .boxed_local();
        let status_fut = child
            .status()
            .map(|status| {
                status.map_err(|error| anyhow!("failed to wait for agent server exit: {error}"))
            })
            .boxed_local();
        let (connection, status_fut, _) = runtime::race_startup(
            connection_rx,
            status_fut,
            stderr_task.clone(),
            &debug_log,
            cx,
        )
        .await?;
        let drained = {
            let connection = connection.clone();
            runtime::drain_transport(
                async move { connection.incoming_closed().await },
                dispatch_tx.clone(),
                stderr_task.clone(),
            )
        };

        let initialize_response = connection
            .send_request(
                acp::InitializeRequest::new(ProtocolVersion::V1)
                    .client_capabilities(client_capabilities)
                    .client_info(
                        acp::Implementation::new("zed", version)
                            .title(release_channel.map(ToOwned::to_owned)),
                    ),
            )
            .block_task()
            .map(|response| response.map_err(anyhow::Error::from))
            .boxed_local();
        let (response, status_fut, drained) =
            runtime::race_startup(initialize_response, status_fut, drained, &debug_log, cx).await?;

        if response.protocol_version < MINIMUM_SUPPORTED_VERSION {
            return Err(UnsupportedVersion.into());
        }

        let wait_task = runtime::watch_process(
            status_fut,
            drained,
            debug_log.clone(),
            {
                let sessions = sessions.clone();
                move |load_error, cx| {
                    emit_load_error_to_all_sessions(&sessions, load_error, cx);
                }
            },
            cx,
        );

        let agent_info = response.agent_info;
        let telemetry_id = agent_info
            .as_ref()
            // Use the one the agent provides if we have one
            .map(|info| SharedString::from(info.name.clone()))
            // Otherwise, just use the name
            .unwrap_or_else(|| agent_id.0.clone());
        let agent_version = agent_info
            .and_then(|info| (!info.version.is_empty()).then(|| SharedString::from(info.version)));
        let agent_supports_delete = response
            .agent_capabilities
            .session_capabilities
            .delete
            .is_some();

        let session_list = if response
            .agent_capabilities
            .session_capabilities
            .list
            .is_some()
        {
            let list = Rc::new(AcpSessionList::new(
                connection.clone(),
                agent_supports_delete,
            ));
            *client_session_list.borrow_mut() = Some(list.clone());
            Some(list)
        } else {
            None
        };

        // TODO: Remove this override once Google team releases their official auth methods
        let auth_methods = if agent_id.0.as_ref() == GEMINI_ID {
            let mut gemini_args = original_command.args.clone();
            gemini_args.retain(|a| a != "--experimental-acp" && a != "--acp");
            let value = serde_json::json!({
                "label": "gemini /auth",
                "command": original_command.path.to_string_lossy(),
                "args": gemini_args,
                "env": original_command.env.unwrap_or_default(),
            });
            let meta = acp_v2::Meta::from_iter([("terminal-auth".to_string(), value)]);
            vec![acp_v2::AuthMethod::Agent(
                acp_v2::AuthMethodAgent::new(GEMINI_TERMINAL_AUTH_METHOD_ID, "Login")
                    .description("Login with your Google or Vertex AI account")
                    .meta(meta),
            )]
        } else {
            acp_thread::auth_methods::from_v1(response.auth_methods)?
        };
        let defaults = AcpConnectionDefaults::new(default_mode, default_config_options);
        let settings_subscription = cx.update({
            let agent_id = agent_id.clone();
            let defaults = defaults.clone();
            move |cx| defaults.observe_settings(agent_id, cx)
        });

        Ok(Self {
            id: agent_id,
            auth_methods,
            agent_server_store,
            connection,
            telemetry_id,
            agent_version,
            claude_config_dir,
            sessions,
            dispatch_tx,
            pending_sessions: Rc::new(RefCell::new(HashMap::default())),
            session_notification_state,
            agent_capabilities: response.agent_capabilities,
            request_elicitations,
            defaults,
            session_list,
            debug_log,
            _settings_subscription: settings_subscription,
            _runtime: runtime,
            _wait_task: wait_task,
            _stderr_task: stderr_task,
            child: RefCell::new(pending_child.0.take()),
            _process_sandbox: process_sandbox,
            worker_process_owned,
            client_read_only,
            terminal_sandbox_wrap,
        })
    }

    pub fn prompt_capabilities(&self) -> acp_v2::PromptCapabilities {
        acp_thread::prompt_capabilities::from_v1(
            self.agent_capabilities.prompt_capabilities.clone(),
        )
    }

    #[cfg(any(test, feature = "test-support"))]
    fn new_for_test(
        connection: ConnectionTo<Agent>,
        sessions: Rc<RefCell<HashMap<acp_v2::SessionId, AcpSession>>>,
        session_notification_state: Rc<SessionNotificationState>,
        agent_capabilities: acp::AgentCapabilities,
        request_elicitations: Entity<ElicitationStore>,
        agent_server_store: WeakEntity<AgentServerStore>,
        io_task: Task<()>,
        dispatch_tx: mpsc::UnboundedSender<ForegroundWork>,
        dispatch_task: Task<()>,
        cx: &mut App,
    ) -> Self {
        let agent_id = AgentId::new("test");
        let defaults = AcpConnectionDefaults::default();
        let settings_subscription = defaults.observe_settings(agent_id.clone(), cx);

        Self {
            id: agent_id,
            telemetry_id: "test".into(),
            agent_version: None,
            connection,
            sessions,
            pending_sessions: Rc::new(RefCell::new(HashMap::default())),
            session_notification_state,
            auth_methods: vec![],
            agent_server_store,
            agent_capabilities,
            request_elicitations,
            defaults,
            claude_config_dir: None,
            child: RefCell::new(None),
            _process_sandbox: None,
            worker_process_owned: false,
            client_read_only: false,
            terminal_sandbox_wrap: None,
            session_list: None,
            debug_log: AcpDebugLog::default(),
            _settings_subscription: settings_subscription,
            _runtime: runtime::ClientRuntime::from_test_tasks(
                dispatch_tx.clone(),
                io_task,
                dispatch_task,
            ),
            dispatch_tx,
            _wait_task: Task::ready(()),
            _stderr_task: Task::ready(()).shared(),
        }
    }

    fn session_directories_from_work_dirs(
        &self,
        work_dirs: &PathList,
    ) -> Result<SessionDirectories> {
        let supports_additional_directories = self.supports_session_additional_directories();
        session_directories_from_work_dirs(work_dirs, supports_additional_directories)
    }

    fn register_session(
        self: &Rc<Self>,
        session_id: acp_v2::SessionId,
        thread: &Entity<AcpThread>,
        session_modes: Option<Rc<RefCell<acp::SessionModeState>>>,
        config_options: Option<ConfigOptions>,
        default_config_task: Option<Task<()>>,
        cx: &mut App,
    ) {
        let release_subscription = cx.observe_release(thread, {
            let session_id = session_id.clone();
            let thread = thread.downgrade();
            let this = Rc::downgrade(self);
            move |released_thread, cx| {
                if let Some(this) = this.upgrade() {
                    if released_thread.is_external_subagent() {
                        this.remove_session(&session_id, thread.entity_id());
                        this.session_notification_state.close_session(session_id);
                    } else {
                        this.release_session(session_id, thread, cx);
                    }
                }
            }
        });
        self.sessions.borrow_mut().insert(
            session_id,
            AcpSession {
                thread: thread.downgrade(),
                suppress_abort_err: false,
                session_modes,
                config_options,
                default_config_task,
                _release_subscription: release_subscription,
            },
        );
    }

    fn remove_session(&self, session_id: &acp_v2::SessionId, thread_id: gpui::EntityId) {
        let mut sessions = self.sessions.borrow_mut();
        if sessions
            .get(session_id)
            .is_some_and(|session| session.thread.entity_id() == thread_id)
        {
            sessions.remove(session_id);
        }
    }

    fn remove_pending_session(&self, session_id: &acp_v2::SessionId, thread_id: gpui::EntityId) {
        let mut pending_sessions = self.pending_sessions.borrow_mut();
        if pending_sessions
            .get(session_id)
            .is_some_and(|session| session.thread.entity_id() == thread_id)
        {
            pending_sessions.remove(session_id);
        }
    }

    fn release_session(
        self: &Rc<Self>,
        session_id: acp_v2::SessionId,
        thread: WeakEntity<AcpThread>,
        cx: &mut App,
    ) {
        if self.pending_sessions.borrow().contains_key(&session_id)
            || !self
                .sessions
                .borrow()
                .get(&session_id)
                .is_some_and(|session| session.thread.entity_id() == thread.entity_id())
        {
            return;
        }

        if let Some(session) = self.sessions.borrow_mut().get_mut(&session_id) {
            session.default_config_task.take();
        }

        let task = cx
            .spawn({
                let this = Rc::downgrade(self);
                let connection = self.connection.clone();
                let dispatch_tx = self.dispatch_tx.clone();
                let supports_close = self.agent_supports_session_close();
                let session_id = session_id.clone();
                let thread_id = thread.entity_id();
                async move |_| {
                    let result = close_session_and_drain(
                        &connection,
                        session_id.clone(),
                        supports_close,
                        &dispatch_tx,
                    )
                    .await;
                    if let Some(this) = this.upgrade() {
                        this.remove_session(&session_id, thread_id);
                        this.remove_pending_session(&session_id, thread_id);
                        this.session_notification_state
                            .close_session(session_id.clone());
                    }
                    result.map_err(Arc::new)
                }
            })
            .shared();
        self.pending_sessions
            .borrow_mut()
            .insert(session_id, PendingAcpSession { thread, task });
    }

    fn agent_supports_session_close(&self) -> bool {
        self.agent_capabilities.session_capabilities.close.is_some()
    }

    fn open_or_create_session(
        self: Rc<Self>,
        session_id: acp_v2::SessionId,
        project: Entity<Project>,
        work_dirs: PathList,
        title: Option<SharedString>,
        rpc_call: impl FnOnce(
            ConnectionTo<Agent>,
            acp_v2::SessionId,
            SessionDirectories,
        )
            -> futures::future::LocalBoxFuture<'static, Result<SessionConfigResponse>>
        + 'static,
        cx: &mut App,
    ) -> Task<Result<Entity<AcpThread>>> {
        let pending = self
            .pending_sessions
            .borrow()
            .get(&session_id)
            .map(|pending| (pending.thread.clone(), pending.task.clone()));
        if let Some((thread, task)) = pending {
            if let Some(thread) = thread.upgrade() {
                return cx.foreground_executor().spawn(async move {
                    task.await.map_err(|err| anyhow!(err))?;
                    Ok(thread)
                });
            }
            return cx.spawn(async move |cx| {
                task.await.log_err();
                cx.update(|cx| {
                    self.open_or_create_session(session_id, project, work_dirs, title, rpc_call, cx)
                })
                .await
            });
        }

        let previous_thread = self
            .sessions
            .borrow()
            .get(&session_id)
            .map(|session| session.thread.clone());
        if let Some(thread) = previous_thread {
            if let Some(thread) = thread.upgrade() {
                return Task::ready(Ok(thread));
            }
            self.release_session(session_id.clone(), thread, cx);
            return self
                .open_or_create_session(session_id, project, work_dirs, title, rpc_call, cx);
        }

        let directories = match self.session_directories_from_work_dirs(&work_dirs) {
            Ok(directories) => directories,
            Err(error) => return Task::ready(Err(error)),
        };

        let action_log = cx.new(|_| ActionLog::new(project.clone()));
        let thread = cx.new(|cx| {
            AcpThread::new(
                None,
                title,
                Some(work_dirs),
                self.clone(),
                project,
                action_log,
                session_id.clone(),
                watch::Receiver::constant(self.prompt_capabilities()),
                cx,
            )
        });
        self.session_notification_state
            .register_session(&session_id);
        self.register_session(session_id.clone(), &thread, None, None, None, cx);

        let shared_task: Shared<Task<Result<(), Arc<anyhow::Error>>>> = cx
            .spawn({
                let session_id = session_id.clone();
                let this = Rc::downgrade(&self);
                let thread = thread.downgrade();
                let connection = self.connection.clone();
                let dispatch_tx = self.dispatch_tx.clone();
                let supports_close = self.agent_supports_session_close();
                async move |cx| {
                    if thread.upgrade().is_none() {
                        if let Some(this) = this.upgrade() {
                            this.remove_session(&session_id, thread.entity_id());
                            this.remove_pending_session(&session_id, thread.entity_id());
                            this.session_notification_state
                                .close_session(session_id.clone());
                        }
                        return Ok(());
                    }

                    let response =
                        rpc_call(connection.clone(), session_id.clone(), directories).await;
                    if response.is_err() {
                        drain_foreground_queue(&dispatch_tx).await.log_err();
                    }
                    let Some(connection_state) = this.upgrade() else {
                        return Ok(());
                    };
                    let (modes, config_options) = match response
                        .and_then(|response| config_state(response.modes, response.config_options))
                    {
                        Ok(state) => state,
                        Err(error) => {
                            connection_state.remove_session(&session_id, thread.entity_id());
                            connection_state
                                .remove_pending_session(&session_id, thread.entity_id());
                            connection_state
                                .session_notification_state
                                .close_session(session_id.clone());
                            return Err(Arc::new(error));
                        }
                    };

                    if thread.upgrade().is_some() {
                        {
                            let mut sessions = connection_state.sessions.borrow_mut();
                            let Some(session) = sessions
                                .get_mut(&session_id)
                                .filter(|session| session.thread.entity_id() == thread.entity_id())
                            else {
                                connection_state
                                    .remove_pending_session(&session_id, thread.entity_id());
                                return Err(Arc::new(anyhow!(
                                    "session was closed or replaced before load completed"
                                )));
                            };
                            session.session_modes = modes;
                            session.config_options = config_options.clone();
                            session.default_config_task = None;
                        }
                        connection_state.remove_pending_session(&session_id, thread.entity_id());
                        connection_state.drain_pending_notifications(&session_id, cx);
                        return Ok(());
                    }

                    drop(connection_state);
                    let result = close_session_and_drain(
                        &connection,
                        session_id.clone(),
                        supports_close,
                        &dispatch_tx,
                    )
                    .await;
                    if let Some(this) = this.upgrade() {
                        this.remove_session(&session_id, thread.entity_id());
                        this.remove_pending_session(&session_id, thread.entity_id());
                        this.session_notification_state
                            .close_session(session_id.clone());
                    }
                    result.map_err(Arc::new)
                }
            })
            .shared();

        self.pending_sessions.borrow_mut().insert(
            session_id.clone(),
            PendingAcpSession {
                thread: thread.downgrade(),
                task: shared_task.clone(),
            },
        );

        cx.foreground_executor().spawn(async move {
            shared_task.await.map_err(|err| anyhow!(err))?;
            Ok(thread)
        })
    }

    fn apply_default_config_options(
        &self,
        session_id: &acp_v2::SessionId,
        config_options: &ConfigOptions,
        cx: &mut AsyncApp,
    ) -> Task<()> {
        let id = self.id.clone();
        let defaults = self.defaults.clone();
        let session_id = session_id.clone();
        let config_options = config_options.clone();
        let connection = self.connection.clone();
        cx.spawn(async move |_| {
            let mut attempted_defaults = Vec::new();
            loop {
                let current_options = config_options.config_options.borrow().clone();
                let next_default = current_options.iter().find_map(|option| {
                    // Retry deferred or reset defaults after dependent changes, but
                    // do not repeat an attempt in the same state if defaults conflict.
                    if attempted_defaults.iter().any(|(config_id, options)| {
                        config_id == &option.config_id && options == &current_options
                    }) {
                        return None;
                    }
                    let value = defaults.config_option(option.config_id.0.as_ref())?;
                    let already_set = match &option.kind {
                        acp_v2::SessionConfigKind::Select(select) => {
                            value.as_value_id() == Some(select.current_value.0.as_ref())
                        }
                        acp_v2::SessionConfigKind::Boolean(boolean) => {
                            value.as_bool() == Some(boolean.current_value)
                        }
                        _ => false,
                    };
                    if already_set {
                        return None;
                    }
                    Some((option.clone(), value))
                });
                let Some((config_option, default_value)) = next_default else {
                    break;
                };
                attempted_defaults.push((config_option.config_id.clone(), current_options));

                let value_to_apply = match &config_option.kind {
                    acp_v2::SessionConfigKind::Select(select) => default_value
                        .as_value_id()
                        .and_then(|value_id| match &select.options {
                            acp_v2::SessionConfigSelectOptions::Ungrouped(options) => options
                                .iter()
                                .any(|opt| &*opt.value.0 == value_id)
                                .then(|| {
                                    acp_v2::SessionConfigOptionValue::id(value_id.to_string())
                                }),
                            acp_v2::SessionConfigSelectOptions::Grouped(groups) => groups
                                .iter()
                                .any(|group| {
                                    group.options.iter().any(|opt| &*opt.value.0 == value_id)
                                })
                                .then(|| {
                                    acp_v2::SessionConfigOptionValue::id(value_id.to_string())
                                }),
                            _ => None,
                        }),
                    acp_v2::SessionConfigKind::Boolean(_) => default_value
                        .as_bool()
                        .map(acp_v2::SessionConfigOptionValue::boolean),
                    _ => None,
                };

                let Some(value_to_apply) = value_to_apply else {
                    log::warn!(
                        "`{}` is not a valid value for config option `{}` in {}",
                        default_value,
                        config_option.config_id.0,
                        id
                    );
                    continue;
                };
                config_options
                    .set_config_option(
                        &connection,
                        session_id.clone(),
                        config_option.config_id,
                        value_to_apply,
                    )
                    .await
                    .log_err();
            }
        })
    }

    fn drain_pending_notifications(&self, session_id: &acp_v2::SessionId, cx: &mut AsyncApp) {
        let Some(buffered) = self.session_notification_state.take_pending(session_id) else {
            return;
        };
        let ctx = ClientContext {
            sessions: self.sessions.clone(),
            session_list: Rc::new(RefCell::new(self.session_list.clone())),
            request_elicitations: self.request_elicitations.clone(),
            session_notification_state: self.session_notification_state.clone(),
            read_only: self.client_read_only,
            terminal_sandbox_wrap: self.terminal_sandbox_wrap.clone(),
            active_request_scopes: None,
            pending_session_updates: None,
        };
        for notification in buffered {
            handle_session_notification(notification, cx, &ctx);
        }
    }

    fn session_meta(&self, cx: &App) -> Task<Option<acp::Meta>> {
        if self.id.as_ref() != CLAUDE_AGENT_ID {
            return Task::ready(None);
        }
        let Some(config_dir) = self.claude_config_dir.clone() else {
            return Task::ready(Some(claude_code_completion_meta(None)));
        };
        cx.background_spawn(async move {
            Some(claude_code_completion_meta(claude_code_session_meta(
                &config_dir,
            )))
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct SessionDirectories {
    cwd: PathBuf,
    additional_directories: Vec<PathBuf>,
}

impl SessionDirectories {
    fn into_new_session_request(
        self,
        mcp_servers: Vec<acp::McpServer>,
        meta: Option<acp::Meta>,
    ) -> acp::NewSessionRequest {
        acp::NewSessionRequest::new(self.cwd)
            .additional_directories(self.additional_directories)
            .mcp_servers(mcp_servers)
            .meta(meta)
    }

    fn into_load_session_request(
        self,
        session_id: acp_v2::SessionId,
        mcp_servers: Vec<acp::McpServer>,
        meta: Option<acp::Meta>,
    ) -> acp::LoadSessionRequest {
        acp::LoadSessionRequest::new(acp::SessionId::new(session_id.0), self.cwd)
            .additional_directories(self.additional_directories)
            .mcp_servers(mcp_servers)
            .meta(meta)
    }

    fn into_resume_session_request(
        self,
        session_id: acp_v2::SessionId,
        mcp_servers: Vec<acp::McpServer>,
        meta: Option<acp::Meta>,
    ) -> acp::ResumeSessionRequest {
        acp::ResumeSessionRequest::new(acp::SessionId::new(session_id.0), self.cwd)
            .additional_directories(self.additional_directories)
            .mcp_servers(mcp_servers)
            .meta(meta)
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ClaudeCodeSettings {
    #[serde(default)]
    enabled_plugins: HashMap<String, bool>,
}

#[derive(Deserialize)]
struct ClaudeCodeInstalledPlugins {
    #[serde(default)]
    plugins: HashMap<String, ClaudeCodePluginInstallations>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum ClaudeCodePluginInstallations {
    Multiple(Vec<ClaudeCodePluginInstallation>),
    Single(ClaudeCodePluginInstallation),
}

impl ClaudeCodePluginInstallations {
    fn user_installation(&self) -> Option<&ClaudeCodePluginInstallation> {
        match self {
            Self::Multiple(installations) => installations
                .iter()
                .find(|installation| installation.is_user_scoped()),
            Self::Single(installation) => installation.is_user_scoped().then_some(installation),
        }
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ClaudeCodePluginInstallation {
    #[serde(default)]
    scope: Option<String>,
    install_path: PathBuf,
}

impl ClaudeCodePluginInstallation {
    fn is_user_scoped(&self) -> bool {
        self.scope.as_deref() == Some("user") || self.scope.is_none()
    }
}

fn load_json_file<T: for<'de> Deserialize<'de>>(path: &std::path::Path) -> Option<T> {
    let contents = match std::fs::read_to_string(path) {
        Ok(contents) => contents,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return None,
        Err(error) => {
            log::warn!("Failed to read {}: {error}", path.display());
            return None;
        }
    };

    match serde_json::from_str(&contents) {
        Ok(value) => Some(value),
        Err(error) => {
            log::warn!("Failed to parse {}: {error}", path.display());
            None
        }
    }
}

fn claude_code_completion_meta(plugin_meta: Option<acp::Meta>) -> acp::Meta {
    let mut meta = plugin_meta.unwrap_or_default();
    let claude = meta
        .entry("claudeCode".to_owned())
        .or_insert_with(|| serde_json::json!({}));
    if let Some(claude) = claude.as_object_mut() {
        claude.insert(
            "emitRawSDKMessages".into(),
            serde_json::json!([{ "type": "stream_event" }]),
        );
    }
    meta
}

fn claude_code_session_meta(config_dir: &std::path::Path) -> Option<acp::Meta> {
    let settings: ClaudeCodeSettings = load_json_file(&config_dir.join("settings.json"))?;
    let enabled_plugins = settings
        .enabled_plugins
        .into_iter()
        .filter_map(|(plugin, enabled)| enabled.then_some(plugin))
        .collect::<HashSet<_>>();
    if enabled_plugins.is_empty() {
        return None;
    }

    let installed: ClaudeCodeInstalledPlugins =
        load_json_file(&config_dir.join("plugins").join("installed_plugins.json"))?;
    let mut plugin_paths = enabled_plugins
        .into_iter()
        .filter_map(|plugin| {
            let installations = installed.plugins.get(&plugin)?;
            let installation = installations.user_installation()?;

            if installation.install_path.is_dir() {
                Some(installation.install_path.clone())
            } else {
                log::warn!(
                    "Enabled Claude Code plugin `{plugin}` is missing at {}",
                    installation.install_path.display()
                );
                None
            }
        })
        .collect::<Vec<_>>();

    plugin_paths.sort();
    plugin_paths.dedup();
    if plugin_paths.is_empty() {
        return None;
    }

    Some(acp::Meta::from_iter([(
        "claudeCode".to_string(),
        serde_json::json!({
            "options": {
                "plugins": plugin_paths.into_iter().map(|path| {
                    serde_json::json!({
                        "type": "local",
                        "path": path,
                    })
                }).collect::<Vec<_>>(),
            }
        }),
    )]))
}

fn session_directories_from_work_dirs(
    work_dirs: &PathList,
    supports_additional_directories: bool,
) -> Result<SessionDirectories> {
    let mut ordered_paths = work_dirs.ordered_paths();
    let cwd = ordered_paths
        .next()
        .cloned()
        .ok_or_else(|| anyhow!("Working directory cannot be empty"))?;
    let additional_directories = if supports_additional_directories {
        ordered_paths.cloned().collect()
    } else {
        Vec::new()
    };

    Ok(SessionDirectories {
        cwd,
        additional_directories,
    })
}

fn emit_load_error_to_all_sessions(
    sessions: &Rc<RefCell<HashMap<acp_v2::SessionId, AcpSession>>>,
    error: LoadError,
    cx: &mut AsyncApp,
) {
    let threads: Vec<_> = sessions
        .borrow()
        .values()
        .map(|session| session.thread.clone())
        .collect();

    for thread in threads {
        thread
            .update(cx, |thread, cx| thread.emit_load_error(error.clone(), cx))
            .ok();
    }
}

struct PendingAcpProcess(Option<Child>);

impl Drop for PendingAcpProcess {
    fn drop(&mut self) {
        if let Some(child) = self.0.as_mut() {
            child.kill().log_err();
        }
    }
}

impl Drop for AcpConnection {
    fn drop(&mut self) {
        if let Some(child) = self.child.get_mut().as_mut() {
            child.kill().log_err();
        }
    }
}

/// Used to support the _meta method prior to stabilization
fn meta_terminal_auth_task(
    agent_id: &AgentId,
    method_id: &acp_v2::AuthMethodId,
    method: &acp_v2::AuthMethod,
) -> Option<SpawnInTerminal> {
    #[derive(Deserialize)]
    struct MetaTerminalAuth {
        label: String,
        command: String,
        #[serde(default)]
        args: Vec<String>,
        #[serde(default)]
        env: HashMap<String, String>,
    }

    let meta = method.meta()?;
    let terminal_auth =
        serde_json::from_value::<MetaTerminalAuth>(meta.get("terminal-auth")?.clone()).ok()?;

    Some(acp_thread::build_terminal_auth_task(
        terminal_auth_task_id(agent_id, method_id),
        terminal_auth.label.clone(),
        terminal_auth.command,
        terminal_auth.args,
        terminal_auth.env,
    ))
}

impl AgentConnection for AcpConnection {
    fn stop_worker_process(&self) -> Result<()> {
        anyhow::ensure!(
            self.worker_process_owned,
            "refusing to stop a shared ACP connection"
        );
        let mut child = self.child.borrow_mut();
        if let Some(process) = child.as_mut() {
            process.kill()?;
        }
        child.take();
        Ok(())
    }

    fn agent_id(&self) -> AgentId {
        self.id.clone()
    }

    fn telemetry_id(&self) -> SharedString {
        self.telemetry_id.clone()
    }

    fn agent_version(&self) -> Option<SharedString> {
        self.agent_version.clone()
    }

    fn new_session(
        self: Rc<Self>,
        project: Entity<Project>,
        work_dirs: PathList,
        cx: &mut App,
    ) -> Task<Result<Entity<AcpThread>>> {
        let directories = match self.session_directories_from_work_dirs(&work_dirs) {
            Ok(directories) => directories,
            Err(error) => return Task::ready(Err(error)),
        };
        let name = self.id.0.clone();
        let mcp_servers = mcp_servers_for_project(&project, cx);
        let session_meta = self.session_meta(cx);

        cx.spawn(async move |cx| {
            let session_meta = session_meta.await;
            let pending_new_session = self.session_notification_state.begin_new_session();
            let (response_sender, response_received) = futures::channel::oneshot::channel();
            let (registered, registration_finished) = futures::channel::oneshot::channel();
            let request = self
                .connection
                .prepare_request(directories.into_new_session_request(mcp_servers, session_meta));
            let cancellation = request.cancellation_handle();
            request
                .on_receiving_result({
                    let connection = self.connection.clone();
                    let supports_close = self
                        .agent_capabilities
                        .session_capabilities
                        .close
                        .is_some();
                    async move |result: Result<acp::NewSessionResponse, acp::Error>| {
                        let session_id = result
                            .as_ref()
                            .ok()
                            .map(|response| response.session_id.clone());
                        let registered = response_sender.send(result).is_ok()
                            && registration_finished.await.is_ok();
                        if !registered && supports_close {
                            if let Some(session_id) = session_id {
                                connection
                                    .prepare_request(acp::CloseSessionRequest::new(session_id))
                                    .on_receiving_result(
                                        async move |result: Result<
                                            acp::CloseSessionResponse,
                                            acp::Error,
                                        >| {
                                            if let Err(error) = result {
                                                log::warn!(
                                                    "ACP abandoned-session close failed with code {}",
                                                    error.code
                                                );
                                            }
                                            Ok(())
                                        },
                                    )
                                    .log_err();
                            }
                        }
                        Ok(())
                    }
                })
                .map_err(map_acp_error)?;
            // Early SDK cancellation is not latched, so arm caller-drop cleanup
            // only after the prepared request's publishing call returns.
            let _request_guard = util::defer(move || {
                cancellation.cancel().log_err();
            });
            let response = response_received
                .await
                .context("ACP new-session response callback was cancelled")?
                .map_err(map_acp_error)?;

            let (modes, config_options) = config_state(response.modes, response.config_options)?;
            let session_id = acp_v2::SessionId::new(response.session_id.0.clone());
            let action_log = cx.new(|_| ActionLog::new(project.clone()));
            let thread: Entity<AcpThread> = cx.new(|cx| {
                AcpThread::new(
                    None,
                    None,
                    Some(work_dirs),
                    self.clone(),
                    project,
                    action_log,
                    session_id.clone(),
                    // ACP doesn't currently support per-session prompt capabilities or changing capabilities dynamically.
                    watch::Receiver::constant(self.prompt_capabilities()),
                    cx,
                )
            });
            cx.update(|cx| {
                self.session_notification_state.register_session(&session_id);
                self.register_session(session_id.clone(), &thread, None, None, None, cx);
            });


            let default_mode = self.defaults.mode();
            if let Some(default_mode) = default_mode {
                let default_mode = acp::SessionModeId::new(default_mode.0);
                if let Some(modes) = modes.as_ref() {
                    let mut modes_ref = modes.borrow_mut();
                    let has_mode = modes_ref
                        .available_modes
                        .iter()
                        .any(|mode| mode.id == default_mode);

                    if has_mode {
                        let initial_mode_id = modes_ref.current_mode_id.clone();

                        cx.spawn({
                            let default_mode = default_mode.clone();
                            let session_id = session_id.clone();
                            let modes = modes.clone();
                            let conn = self.connection.clone();
                            async move |_| {
                                let result = conn
                                    .send_request(acp::SetSessionModeRequest::new(
                                        acp::SessionId::new(session_id.0),
                                        default_mode,
                                    ))
                                    .block_task()
                                .await
                                .log_err();

                                if result.is_none() {
                                    modes.borrow_mut().current_mode_id = initial_mode_id;
                                }
                            }
                        })
                        .detach();

                        modes_ref.current_mode_id = default_mode;
                    } else {
                        let available_modes = modes_ref
                            .available_modes
                            .iter()
                            .map(|mode| format!("- `{}`: {}", mode.id, mode.name))
                            .collect::<Vec<_>>()
                            .join("\n");

                        log::warn!(
                            "`{default_mode}` is not valid {name} mode. Available options:\n{available_modes}",
                        );
                    }
                }
            }

            let session_id = acp_v2::SessionId::new(response.session_id.0.clone());
            let default_config_task = config_options.as_ref().map(|options| {
                self.apply_default_config_options(&session_id, options, cx)
            });

            {
                let mut sessions = self.sessions.borrow_mut();
                if let Some(session) = sessions.get_mut(&session_id) {
                    session.session_modes = modes;
                    session.config_options = config_options;
                    session.default_config_task = default_config_task;
                }
            }
            self.drain_pending_notifications(&session_id, cx);
            drop(pending_new_session);

            registered
                .send(())
                .map_err(|_| anyhow!("ACP new-session ordering callback ended before registration"))?;
            Ok(thread)
        })
    }

    fn supports_load_session(&self) -> bool {
        self.agent_capabilities.load_session
    }

    fn supports_resume_session(&self) -> bool {
        self.agent_capabilities
            .session_capabilities
            .resume
            .is_some()
    }

    fn supports_session_additional_directories(&self) -> bool {
        self.agent_capabilities
            .session_capabilities
            .additional_directories
            .is_some()
    }

    fn load_session(
        self: Rc<Self>,
        session_id: acp_v2::SessionId,
        project: Entity<Project>,
        work_dirs: PathList,
        title: Option<SharedString>,
        cx: &mut App,
    ) -> Task<Result<Entity<AcpThread>>> {
        if !self.agent_capabilities.load_session {
            return Task::ready(Err(anyhow!(LoadError::Other(
                "Loading sessions is not supported by this agent.".into()
            ))));
        }

        let mcp_servers = mcp_servers_for_project(&project, cx);
        let session_meta = self.session_meta(cx);
        self.open_or_create_session(
            session_id,
            project,
            work_dirs,
            title,
            move |connection, session_id, directories| {
                Box::pin(async move {
                    let session_meta = session_meta.await;
                    let response = connection
                        .send_request(directories.into_load_session_request(
                            session_id.clone(),
                            mcp_servers,
                            session_meta,
                        ))
                        .block_task()
                        .await
                        .map_err(map_acp_error)?;
                    Ok(SessionConfigResponse {
                        modes: response.modes,
                        config_options: response.config_options,
                    })
                })
            },
            cx,
        )
    }

    fn resume_session(
        self: Rc<Self>,
        session_id: acp_v2::SessionId,
        project: Entity<Project>,
        work_dirs: PathList,
        title: Option<SharedString>,
        cx: &mut App,
    ) -> Task<Result<Entity<AcpThread>>> {
        if self
            .agent_capabilities
            .session_capabilities
            .resume
            .is_none()
        {
            return Task::ready(Err(anyhow!(LoadError::Other(
                "Resuming sessions is not supported by this agent.".into()
            ))));
        }

        let mcp_servers = mcp_servers_for_project(&project, cx);
        let session_meta = self.session_meta(cx);
        self.open_or_create_session(
            session_id,
            project,
            work_dirs,
            title,
            move |connection, session_id, directories| {
                Box::pin(async move {
                    let session_meta = session_meta.await;
                    let response = connection
                        .send_request(directories.into_resume_session_request(
                            session_id.clone(),
                            mcp_servers,
                            session_meta,
                        ))
                        .block_task()
                        .await
                        .map_err(map_acp_error)?;
                    Ok(SessionConfigResponse {
                        modes: response.modes,
                        config_options: response.config_options,
                    })
                })
            },
            cx,
        )
    }

    fn auth_methods(&self) -> &[acp_v2::AuthMethod] {
        &self.auth_methods
    }

    fn terminal_auth_task(
        &self,
        method_id: &acp_v2::AuthMethodId,
        cx: &App,
    ) -> Option<Task<Result<SpawnInTerminal>>> {
        let method = self
            .auth_methods
            .iter()
            .find(|method| method.method_id() == method_id)?;

        match method {
            acp_v2::AuthMethod::Terminal(terminal) => Some(first_class_terminal_auth_task(
                self.id.clone(),
                self.agent_server_store.clone(),
                terminal.clone(),
                cx,
            )),
            acp_v2::AuthMethod::Agent(_) => meta_terminal_auth_task(&self.id, method_id, method)
                .map(|task| Task::ready(Ok(task))),
            _ => None,
        }
    }

    fn authenticate(&self, method_id: acp_v2::AuthMethodId, cx: &mut App) -> Task<Result<()>> {
        if !self.auth_methods.iter().any(|method| {
            matches!(method, acp_v2::AuthMethod::Agent(agent) if agent.method_id == method_id)
        }) {
            return Task::ready(Err(anyhow!(
                "Authentication method is not supported for agent-managed login"
            )));
        }

        let conn = self.connection.clone();
        cx.foreground_executor().spawn(async move {
            conn.send_request(acp::AuthenticateRequest::new(acp::AuthMethodId::new(
                method_id.0,
            )))
            .block_task()
            .await?;
            Ok(())
        })
    }

    fn supports_logout(&self) -> bool {
        self.agent_capabilities.auth.logout.is_some()
    }

    fn logout(&self, cx: &mut App) -> Task<Result<()>> {
        if !self.supports_logout() {
            return Task::ready(Err(anyhow!("Logout is not supported by this agent.")));
        }

        let conn = self.connection.clone();
        cx.foreground_executor().spawn(async move {
            conn.send_request(acp::LogoutRequest::new())
                .block_task()
                .await?;
            Ok(())
        })
    }

    fn validate_prompt_content(&self, content: &[acp_v2::ContentBlock]) -> Result<()> {
        acp_thread::content::validate_prompt_content_for_v1(content)
    }

    fn prompt(
        &self,
        params: acp_v2::PromptRequest,
        cx: &mut App,
    ) -> Task<Result<acp::PromptResponse>> {
        let params = match acp_thread::content::prompt_to_v1(params) {
            Ok(params) => params,
            Err(error) => return Task::ready(Err(error)),
        };
        let conn = self.connection.clone();
        let sessions = self.sessions.clone();
        let session_id = acp_v2::SessionId::new(params.session_id.0.clone());
        let completion_dispatch =
            (self.id.as_ref() == CLAUDE_AGENT_ID).then(|| self.dispatch_tx.clone());
        cx.foreground_executor().spawn(async move {
            let result = conn.send_request(params).block_task().await;
            // The response can arrive before queued stream notifications reach the thread.
            if result.is_ok()
                && let Some(dispatch_tx) = completion_dispatch
            {
                drain_foreground_queue(&dispatch_tx).await?;
            }

            let mut suppress_abort_err = false;

            if let Some(session) = sessions.borrow_mut().get_mut(&session_id) {
                suppress_abort_err = session.suppress_abort_err;
                session.suppress_abort_err = false;
            }

            match result {
                Ok(response) => Ok(response),
                Err(err) => {
                    if err.code != ErrorCode::InternalError {
                        anyhow::bail!(err)
                    }

                    let Some(data) = &err.data else {
                        anyhow::bail!(err)
                    };

                    // Temporary workaround until the following PR is generally available:
                    // https://github.com/google-gemini/gemini-cli/pull/6656

                    #[derive(Deserialize)]
                    #[serde(deny_unknown_fields)]
                    struct ErrorDetails {
                        details: Box<str>,
                    }

                    match serde_json::from_value(data.clone()) {
                        Ok(ErrorDetails { details }) => {
                            if suppress_abort_err
                                && (details.contains("This operation was aborted")
                                    || details.contains("The user aborted a request"))
                            {
                                Ok(acp::PromptResponse::new(acp::StopReason::Cancelled))
                            } else {
                                Err(anyhow!(details))
                            }
                        }
                        Err(_) => Err(anyhow!(err)),
                    }
                }
            }
        })
    }

    fn cancel(&self, session_id: &acp_v2::SessionId, _cx: &mut App) -> Result<()> {
        let params = acp::CancelNotification::new(acp::SessionId::new(session_id.0.clone()));
        self.connection
            .send_notification(params)
            .map_err(map_acp_error)?;
        if let Some(session) = self.sessions.borrow_mut().get_mut(session_id) {
            session.suppress_abort_err = true;
        }
        Ok(())
    }

    fn request_elicitations(&self) -> Option<Entity<ElicitationStore>> {
        Some(self.request_elicitations.clone())
    }

    fn session_modes(
        &self,
        session_id: &acp_v2::SessionId,
        _cx: &App,
    ) -> Option<Rc<dyn acp_thread::AgentSessionModes>> {
        let sessions = self.sessions.clone();
        let sessions_ref = sessions.borrow();
        let Some(session) = sessions_ref.get(session_id) else {
            return None;
        };

        if let Some(modes) = session.session_modes.as_ref() {
            Some(Rc::new(AcpSessionModes {
                connection: self.connection.clone(),
                session_id: session_id.clone(),
                state: modes.clone(),
            }) as _)
        } else {
            None
        }
    }

    fn session_config_options(
        &self,
        session_id: &acp_v2::SessionId,
        _cx: &App,
    ) -> Option<Rc<dyn acp_thread::AgentSessionConfigOptions>> {
        let sessions = self.sessions.borrow();
        let session = sessions.get(session_id)?;

        let config_opts = session.config_options.as_ref()?;

        Some(Rc::new(AcpSessionConfigOptions {
            session_id: session_id.clone(),
            connection: self.connection.clone(),
            config_options: config_opts.clone(),
        }) as _)
    }

    fn session_list(&self, _cx: &mut App) -> Option<Rc<dyn AgentSessionList>> {
        self.session_list.clone().map(|s| s as _)
    }

    fn into_any(self: Rc<Self>) -> Rc<dyn Any> {
        self
    }
}

#[cfg(any(test, feature = "test-support"))]
pub mod test_support {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    use acp_thread::{
        AgentSessionClientUserMessageIds, AgentSessionConfigOptions, AgentSessionModes,
        AgentSessionRetry, AgentSessionSetTitle, AgentSessionTruncate, AgentTelemetry,
    };

    use super::*;

    #[derive(Clone, Default)]
    pub struct FakeAcpAgentServer {
        load_session_count: Arc<AtomicUsize>,
        close_session_count: Arc<AtomicUsize>,
        fail_next_prompt: Arc<AtomicBool>,
        auth_elicitation_request: Arc<Mutex<Option<acp::CreateElicitationRequest>>>,
        auth_elicitation_response:
            Arc<Mutex<Option<async_channel::Sender<acp::CreateElicitationResponse>>>>,
        auth_elicitation_completion: Arc<Mutex<Option<acp::CompleteElicitationNotification>>>,
        exit_status_sender:
            Arc<std::sync::Mutex<Option<async_channel::Sender<std::process::ExitStatus>>>>,
    }

    impl FakeAcpAgentServer {
        pub fn new() -> Self {
            Self::default()
        }

        pub fn load_session_count(&self) -> Arc<AtomicUsize> {
            self.load_session_count.clone()
        }

        pub fn close_session_count(&self) -> Arc<AtomicUsize> {
            self.close_session_count.clone()
        }

        pub fn simulate_server_exit(&self) {
            let sender = self
                .exit_status_sender
                .lock()
                .expect("exit status sender lock should not be poisoned")
                .clone()
                .expect("fake ACP server must be connected before simulating exit");
            sender
                .try_send(std::process::ExitStatus::default())
                .expect("fake ACP server exit receiver should still be alive");
        }

        pub fn fail_next_prompt(&self) {
            self.fail_next_prompt.store(true, Ordering::SeqCst);
        }

        pub fn request_elicitation_during_auth(
            &self,
            request: acp::CreateElicitationRequest,
        ) -> async_channel::Receiver<acp::CreateElicitationResponse> {
            let (response_tx, response_rx) = async_channel::bounded(1);
            *self
                .auth_elicitation_request
                .lock()
                .expect("auth elicitation request lock should not be poisoned") = Some(request);
            *self
                .auth_elicitation_response
                .lock()
                .expect("auth elicitation response lock should not be poisoned") =
                Some(response_tx);
            response_rx
        }
    }

    impl crate::AgentServer for FakeAcpAgentServer {
        fn logo(&self) -> ui::IconName {
            ui::IconName::ZedAgent
        }

        fn agent_id(&self) -> AgentId {
            AgentId::new("Test")
        }

        fn connect(
            &self,
            _delegate: crate::AgentServerDelegate,
            project: Entity<Project>,
            cx: &mut App,
        ) -> Task<anyhow::Result<Rc<dyn AgentConnection>>> {
            let load_session_count = self.load_session_count.clone();
            let close_session_count = self.close_session_count.clone();
            let fail_next_prompt = self.fail_next_prompt.clone();
            let auth_elicitation_request = self.auth_elicitation_request.clone();
            let auth_elicitation_response = self.auth_elicitation_response.clone();
            let auth_elicitation_completion = self.auth_elicitation_completion.clone();
            let exit_status_sender = self.exit_status_sender.clone();
            cx.spawn(async move |cx| {
                let harness = build_fake_acp_connection(
                    project,
                    load_session_count,
                    close_session_count,
                    fail_next_prompt,
                    auth_elicitation_request,
                    auth_elicitation_response,
                    auth_elicitation_completion,
                    cx,
                )
                .await?;
                let (exit_tx, exit_rx) = async_channel::bounded(1);
                *exit_status_sender
                    .lock()
                    .expect("exit status sender lock should not be poisoned") = Some(exit_tx);
                let connection = harness.connection.clone();
                let simulate_exit_task = cx.spawn(async move |cx| {
                    while let Ok(status) = exit_rx.recv().await {
                        emit_load_error_to_all_sessions(
                            &connection.sessions,
                            LoadError::Exited {
                                status,
                                stderr: None,
                            },
                            cx,
                        );
                    }
                    Ok(())
                });
                Ok(Rc::new(FakeAcpAgentConnection {
                    inner: harness.connection,
                    _keep_agent_alive: harness.keep_agent_alive,
                    _simulate_exit_task: simulate_exit_task,
                }) as Rc<dyn AgentConnection>)
            })
        }

        fn into_any(self: Rc<Self>) -> Rc<dyn Any> {
            self
        }
    }

    pub struct FakeAcpConnectionHarness {
        pub connection: Rc<AcpConnection>,
        pub load_session_count: Arc<AtomicUsize>,
        pub close_session_count: Arc<AtomicUsize>,
        pub authenticate_count: Arc<AtomicUsize>,
        pub logout_count: Arc<AtomicUsize>,
        pub keep_agent_alive: Task<anyhow::Result<()>>,
    }

    struct FakeAcpAgentConnection {
        inner: Rc<AcpConnection>,
        _keep_agent_alive: Task<anyhow::Result<()>>,
        _simulate_exit_task: Task<anyhow::Result<()>>,
    }

    impl AgentConnection for FakeAcpAgentConnection {
        fn agent_id(&self) -> AgentId {
            self.inner.agent_id()
        }

        fn telemetry_id(&self) -> SharedString {
            self.inner.telemetry_id()
        }

        fn agent_version(&self) -> Option<SharedString> {
            self.inner.agent_version()
        }

        fn new_session(
            self: Rc<Self>,
            project: Entity<Project>,
            work_dirs: PathList,
            cx: &mut App,
        ) -> Task<Result<Entity<AcpThread>>> {
            self.inner.clone().new_session(project, work_dirs, cx)
        }

        fn supports_load_session(&self) -> bool {
            self.inner.supports_load_session()
        }

        fn load_session(
            self: Rc<Self>,
            session_id: acp_v2::SessionId,
            project: Entity<Project>,
            work_dirs: PathList,
            title: Option<SharedString>,
            cx: &mut App,
        ) -> Task<Result<Entity<AcpThread>>> {
            self.inner
                .clone()
                .load_session(session_id, project, work_dirs, title, cx)
        }

        fn supports_resume_session(&self) -> bool {
            self.inner.supports_resume_session()
        }

        fn supports_session_additional_directories(&self) -> bool {
            self.inner.supports_session_additional_directories()
        }

        fn resume_session(
            self: Rc<Self>,
            session_id: acp_v2::SessionId,
            project: Entity<Project>,
            work_dirs: PathList,
            title: Option<SharedString>,
            cx: &mut App,
        ) -> Task<Result<Entity<AcpThread>>> {
            self.inner
                .clone()
                .resume_session(session_id, project, work_dirs, title, cx)
        }

        fn auth_methods(&self) -> &[acp_v2::AuthMethod] {
            self.inner.auth_methods()
        }

        fn terminal_auth_task(
            &self,
            method: &acp_v2::AuthMethodId,
            cx: &App,
        ) -> Option<Task<Result<SpawnInTerminal>>> {
            self.inner.terminal_auth_task(method, cx)
        }

        fn authenticate(&self, method: acp_v2::AuthMethodId, cx: &mut App) -> Task<Result<()>> {
            self.inner.authenticate(method, cx)
        }

        fn supports_logout(&self) -> bool {
            self.inner.supports_logout()
        }

        fn logout(&self, cx: &mut App) -> Task<Result<()>> {
            self.inner.logout(cx)
        }

        fn client_user_message_ids(
            &self,
            cx: &App,
        ) -> Option<Rc<dyn AgentSessionClientUserMessageIds>> {
            self.inner.client_user_message_ids(cx)
        }

        fn validate_prompt_content(&self, content: &[acp_v2::ContentBlock]) -> Result<()> {
            self.inner.validate_prompt_content(content)
        }

        fn prompt(
            &self,
            params: acp_v2::PromptRequest,
            cx: &mut App,
        ) -> Task<Result<acp::PromptResponse>> {
            self.inner.prompt(params, cx)
        }

        fn retry(
            &self,
            session_id: &acp_v2::SessionId,
            cx: &App,
        ) -> Option<Rc<dyn AgentSessionRetry>> {
            self.inner.retry(session_id, cx)
        }

        fn cancel(&self, session_id: &acp_v2::SessionId, cx: &mut App) -> Result<()> {
            self.inner.cancel(session_id, cx)
        }

        fn request_elicitations(&self) -> Option<Entity<ElicitationStore>> {
            self.inner.request_elicitations()
        }

        fn truncate(
            &self,
            session_id: &acp_v2::SessionId,
            cx: &App,
        ) -> Option<Rc<dyn AgentSessionTruncate>> {
            self.inner.truncate(session_id, cx)
        }

        fn set_title(
            &self,
            session_id: &acp_v2::SessionId,
            cx: &App,
        ) -> Option<Rc<dyn AgentSessionSetTitle>> {
            self.inner.set_title(session_id, cx)
        }

        fn telemetry(&self) -> Option<Rc<dyn AgentTelemetry>> {
            self.inner.telemetry()
        }

        fn session_modes(
            &self,
            session_id: &acp_v2::SessionId,
            cx: &App,
        ) -> Option<Rc<dyn AgentSessionModes>> {
            self.inner.session_modes(session_id, cx)
        }

        fn session_config_options(
            &self,
            session_id: &acp_v2::SessionId,
            cx: &App,
        ) -> Option<Rc<dyn AgentSessionConfigOptions>> {
            self.inner.session_config_options(session_id, cx)
        }

        fn session_list(&self, cx: &mut App) -> Option<Rc<dyn AgentSessionList>> {
            self.inner.session_list(cx)
        }

        fn into_any(self: Rc<Self>) -> Rc<dyn Any> {
            self
        }
    }

    async fn build_fake_acp_connection(
        project: Entity<Project>,
        load_session_count: Arc<AtomicUsize>,
        close_session_count: Arc<AtomicUsize>,
        fail_next_prompt: Arc<AtomicBool>,
        auth_elicitation_request: Arc<Mutex<Option<acp::CreateElicitationRequest>>>,
        auth_elicitation_response: Arc<
            Mutex<Option<async_channel::Sender<acp::CreateElicitationResponse>>>,
        >,
        auth_elicitation_completion: Arc<Mutex<Option<acp::CompleteElicitationNotification>>>,
        cx: &mut AsyncApp,
    ) -> Result<FakeAcpConnectionHarness> {
        let (client_transport, agent_transport) = agent_client_protocol::Channel::duplex();

        let authenticate_count = Arc::new(AtomicUsize::new(0));
        let logout_count = Arc::new(AtomicUsize::new(0));
        let sessions: Rc<RefCell<HashMap<acp_v2::SessionId, AcpSession>>> =
            Rc::new(RefCell::new(HashMap::default()));
        let client_session_list: Rc<RefCell<Option<Rc<AcpSessionList>>>> =
            Rc::new(RefCell::new(None));

        let agent_future = Agent
            .builder()
            .name("fake-agent")
            .on_receive_request(
                async move |req: acp::InitializeRequest, responder, _cx| {
                    responder.respond(
                        acp::InitializeResponse::new(req.protocol_version)
                            .agent_capabilities(
                                acp::AgentCapabilities::default()
                                    .load_session(true)
                                    .session_capabilities(
                                        acp::SessionCapabilities::default()
                                            .close(acp::SessionCloseCapabilities::new()),
                                    ),
                            )
                            .auth_methods(vec![acp::AuthMethod::Agent(acp::AuthMethodAgent::new(
                                "login", "Login",
                            ))]),
                    )
                },
                agent_client_protocol::on_receive_request!(),
            )
            .on_receive_request(
                {
                    let authenticate_count = authenticate_count.clone();
                    let auth_elicitation_request = auth_elicitation_request.clone();
                    let auth_elicitation_response = auth_elicitation_response.clone();
                    let auth_elicitation_completion = auth_elicitation_completion.clone();
                    async move |req: acp::AuthenticateRequest, responder, cx| {
                        assert_eq!(req.method_id, acp::AuthMethodId::new("login"));
                        authenticate_count.fetch_add(1, Ordering::SeqCst);
                        let request = auth_elicitation_request
                            .lock()
                            .expect("auth elicitation request lock should not be poisoned")
                            .take();
                        let response_tx = auth_elicitation_response
                            .lock()
                            .expect("auth elicitation response lock should not be poisoned")
                            .take();
                        let completion = auth_elicitation_completion
                            .lock()
                            .expect("auth elicitation completion lock should not be poisoned")
                            .take();

                        if let Some(request) = request {
                            let client = cx.clone();
                            cx.send_request(request)
                                .on_receiving_result(async move |result| {
                                    if let Ok(response) = result {
                                        if matches!(
                                            &response.action,
                                            acp::ElicitationAction::Accept(_)
                                        ) && let Some(completion) = completion
                                        {
                                            client.send_notification(completion)?;
                                        }
                                        if let Some(response_tx) = response_tx {
                                            response_tx.send(response).await.ok();
                                        }
                                    }
                                    responder.respond(Default::default())
                                })?;
                            Ok(())
                        } else {
                            responder.respond(Default::default())
                        }
                    }
                },
                agent_client_protocol::on_receive_request!(),
            )
            .on_receive_request(
                async move |_req: acp::NewSessionRequest, responder, _cx| {
                    responder.respond(acp::NewSessionResponse::new(acp::SessionId::new("unused")))
                },
                agent_client_protocol::on_receive_request!(),
            )
            .on_receive_request(
                {
                    let fail_next_prompt = fail_next_prompt.clone();
                    async move |_req: acp::PromptRequest, responder, _cx| {
                        if fail_next_prompt.swap(false, Ordering::SeqCst) {
                            responder.respond_with_error(acp::ErrorCode::InternalError.into())
                        } else {
                            responder.respond(acp::PromptResponse::new(acp::StopReason::EndTurn))
                        }
                    }
                },
                agent_client_protocol::on_receive_request!(),
            )
            .on_receive_request(
                {
                    let load_session_count = load_session_count.clone();
                    async move |_req: acp::LoadSessionRequest, responder, _cx| {
                        load_session_count.fetch_add(1, Ordering::SeqCst);
                        responder.respond(acp::LoadSessionResponse::new())
                    }
                },
                agent_client_protocol::on_receive_request!(),
            )
            .on_receive_request(
                {
                    let close_session_count = close_session_count.clone();
                    async move |_req: acp::CloseSessionRequest, responder, _cx| {
                        close_session_count.fetch_add(1, Ordering::SeqCst);
                        responder.respond(acp::CloseSessionResponse::new())
                    }
                },
                agent_client_protocol::on_receive_request!(),
            )
            .on_receive_request(
                {
                    let logout_count = logout_count.clone();
                    async move |_req: acp::LogoutRequest, responder, _cx| {
                        logout_count.fetch_add(1, Ordering::SeqCst);
                        responder.respond(acp::LogoutResponse::new())
                    }
                },
                agent_client_protocol::on_receive_request!(),
            )
            .on_receive_notification(
                async move |_notif: acp::CancelNotification, _cx| Ok(()),
                agent_client_protocol::on_receive_notification!(),
            )
            .connect_to(agent_transport);

        let agent_io_task = cx.background_spawn(agent_future);

        // Wire the production handler set into the fake client so inbound
        // requests/notifications from the fake agent are dispatched the
        // same way the real `stdio` path does.
        let (dispatch_tx, dispatch_rx) = mpsc::unbounded::<ForegroundWork>();

        let (connection_tx, connection_rx) = futures::channel::oneshot::channel();
        let client_future = connect_client_future(
            "zed-test",
            client_transport,
            dispatch_tx.clone(),
            connection_tx,
            futures::future::pending(),
        );
        let client_io_task = cx.background_spawn(async move {
            client_future.await.ok();
        });

        let client_conn: ConnectionTo<Agent> = connection_rx
            .await
            .context("failed to receive fake ACP connection handle")?;

        let response = client_conn
            .send_request(acp::InitializeRequest::new(ProtocolVersion::V1))
            .block_task()
            .await?;

        let agent_capabilities = response.agent_capabilities;
        let auth_methods = acp_thread::auth_methods::from_v1(response.auth_methods)?;

        let request_elicitations = cx.new(|_| ElicitationStore::default());
        let session_notification_state = Rc::new(SessionNotificationState::default());
        let dispatch_context = ClientContext {
            sessions: sessions.clone(),
            session_list: client_session_list.clone(),
            request_elicitations: request_elicitations.clone(),
            session_notification_state: session_notification_state.clone(),
            read_only: false,
            terminal_sandbox_wrap: None,
            active_request_scopes: None,
            pending_session_updates: None,
        };
        let dispatch_task = cx.spawn({
            let mut dispatch_rx = dispatch_rx;
            async move |cx| {
                while let Some(work) = dispatch_rx.next().await {
                    work.run(cx, &dispatch_context);
                }
            }
        });

        let agent_server_store =
            project.read_with(cx, |project, _| project.agent_server_store().downgrade());

        let connection = cx.update(|cx| {
            AcpConnection::new_for_test(
                client_conn,
                sessions,
                session_notification_state,
                agent_capabilities,
                request_elicitations,
                agent_server_store,
                client_io_task,
                dispatch_tx,
                dispatch_task,
                cx,
            )
        });

        let keep_agent_alive = cx.background_spawn(async move {
            agent_io_task.await.ok();
            anyhow::Ok(())
        });

        let mut connection = connection;
        connection.auth_methods = auth_methods;
        Ok(FakeAcpConnectionHarness {
            connection: Rc::new(connection),
            load_session_count,
            close_session_count,
            authenticate_count,
            logout_count,
            keep_agent_alive,
        })
    }

    pub async fn connect_fake_acp_connection(
        project: Entity<Project>,
        cx: &mut gpui::TestAppContext,
    ) -> FakeAcpConnectionHarness {
        cx.update(|cx| {
            let store = settings::SettingsStore::test(cx);
            cx.set_global(store);
        });

        build_fake_acp_connection(
            project,
            Arc::new(AtomicUsize::new(0)),
            Arc::new(AtomicUsize::new(0)),
            Arc::new(AtomicBool::new(false)),
            Arc::new(Mutex::new(None)),
            Arc::new(Mutex::new(None)),
            Arc::new(Mutex::new(None)),
            &mut cx.to_async(),
        )
        .await
        .expect("failed to initialize ACP connection")
    }

    #[cfg(test)]
    pub async fn connect_fake_acp_connection_with_auth_elicitation(
        project: Entity<Project>,
        request: acp::CreateElicitationRequest,
        cx: &mut gpui::TestAppContext,
    ) -> (
        FakeAcpConnectionHarness,
        async_channel::Receiver<acp::CreateElicitationResponse>,
    ) {
        cx.update(|cx| {
            let store = settings::SettingsStore::test(cx);
            cx.set_global(store);
        });

        let (response_tx, response_rx) = async_channel::bounded(1);
        let harness = build_fake_acp_connection(
            project,
            Arc::new(AtomicUsize::new(0)),
            Arc::new(AtomicUsize::new(0)),
            Arc::new(AtomicBool::new(false)),
            Arc::new(Mutex::new(Some(request))),
            Arc::new(Mutex::new(Some(response_tx))),
            Arc::new(Mutex::new(None)),
            &mut cx.to_async(),
        )
        .await
        .expect("failed to initialize ACP connection");

        (harness, response_rx)
    }

    #[cfg(test)]
    pub async fn connect_fake_acp_connection_with_auth_elicitation_completion(
        project: Entity<Project>,
        request: acp::CreateElicitationRequest,
        completion: acp::CompleteElicitationNotification,
        cx: &mut gpui::TestAppContext,
    ) -> (
        FakeAcpConnectionHarness,
        async_channel::Receiver<acp::CreateElicitationResponse>,
    ) {
        cx.update(|cx| {
            let store = settings::SettingsStore::test(cx);
            cx.set_global(store);
        });

        let (response_tx, response_rx) = async_channel::bounded(1);
        let harness = build_fake_acp_connection(
            project,
            Arc::new(AtomicUsize::new(0)),
            Arc::new(AtomicUsize::new(0)),
            Arc::new(AtomicBool::new(false)),
            Arc::new(Mutex::new(Some(request))),
            Arc::new(Mutex::new(Some(response_tx))),
            Arc::new(Mutex::new(Some(completion))),
            &mut cx.to_async(),
        )
        .await
        .expect("failed to initialize ACP connection");

        (harness, response_rx)
    }
}

#[cfg(test)]
mod tests {
    use std::{
        path::Path,
        sync::atomic::{AtomicUsize, Ordering},
    };

    use super::*;
    use feature_flags::{AcpBetaFeatureFlag, FeatureFlag as _};
    use settings::Settings as _;

    #[test]
    fn external_subagents_preserve_initialize_capabilities() -> Result<()> {
        let capabilities =
            client_capabilities_for_agent(&AgentId::new("Claude Multi"), false, false);
        let expected = serde_json::to_value(&capabilities)?;
        let request = serde_json::to_value(
            acp::InitializeRequest::new(ProtocolVersion::V1)
                .client_capabilities(capabilities.clone())
                .client_info(acp::Implementation::new("zed", "test")),
        )?;
        let actual = request["clientCapabilities"].clone();
        assert_eq!(actual["subagents"], serde_json::json!({}));
        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn external_subagents_parse_lifecycle_and_ordinary_updates() -> Result<()> {
        use agent_client_protocol::JsonRpcMessage as _;

        for update in [
            serde_json::json!({"sessionUpdate": "subagent_spawned", "subagentSessionId": "worker", "name": "multi-openai", "task": "Scan frontend", "capabilities": {}}),
            serde_json::json!({"sessionUpdate": "subagent_state_update", "subagentSessionId": "worker", "state": "completed"}),
            serde_json::json!({"sessionUpdate": "subagent_state_update", "subagentSessionId": "worker", "state": "disconnected"}),
        ] {
            let notification = SessionNotificationWithSubagents::parse_message(
                "session/update",
                &serde_json::json!({"sessionId": "parent", "update": update}),
            )?
            .into_notification()?;
            assert_eq!(notification.session_id, acp::SessionId::new("parent"));
            let acp::SessionUpdate::SessionInfoUpdate(info) = notification.update else {
                panic!("expected lifecycle envelope");
            };
            let event =
                info.meta.expect("lifecycle metadata")[EXTERNAL_SUBAGENT_UPDATE_META_KEY].clone();
            assert_eq!(event["sessionUpdate"], update["sessionUpdate"]);
            assert_eq!(event["subagentSessionId"], update["subagentSessionId"]);
        }
        let update = acp::SessionUpdate::AgentMessageChunk(acp::ContentChunk::new(
            acp::ContentBlock::Text(acp::TextContent::new("worker output")),
        ));
        let notification = SessionNotificationWithSubagents::parse_message(
            "session/update",
            &serde_json::json!({"sessionId": "worker", "update": update}),
        )?
        .into_notification()?;
        assert_eq!(notification.session_id, acp::SessionId::new("worker"));
        assert_eq!(
            serde_json::to_value(notification.update)?,
            serde_json::to_value(update)?
        );
        assert!(SessionNotificationWithSubagents {
            session_id: acp_v2::SessionId::new("parent"),
            update: serde_json::json!({"sessionUpdate": "subagent_state_update", "subagentSessionId": "worker", "state": "invalid"}),
        }.into_notification().is_err());
        Ok(())
    }

    #[test]
    fn external_subagents_parse_rfd_patches_without_losing_nulls() -> Result<()> {
        use agent_client_protocol::JsonRpcMessage as _;
        for update in [
            serde_json::json!({"sessionUpdate":"subagent_update", "sessionId":"child", "title":"Worker", "state":{"state":"running"}}),
            serde_json::json!({"sessionUpdate":"subagent_update", "sessionId":"child", "title":null, "state":null, "capabilities":null}),
            serde_json::json!({"sessionUpdate":"subagent_update", "sessionId":"child", "description":"Purpose", "state":{"state":"requires_action"}, "capabilities":{"cancel":{}}}),
            serde_json::json!({"sessionUpdate":"session_message", "messageId":"delegation", "senderSessionId":"parent", "recipientSessionId":"child", "content":[{"type":"text","text":"Inspect sources"}]}),
            serde_json::json!({"sessionUpdate":"session_message", "messageId":"delegation", "content":null}),
            serde_json::json!({"sessionUpdate":"session_message_chunk", "messageId":"delegation", "content":{"type":"text","text":"More"}}),
        ] {
            let notification = SessionNotificationWithSubagents::parse_message(
                "session/update",
                &serde_json::json!({"sessionId":"parent", "update":update}),
            )?
            .into_notification()?;
            let acp::SessionUpdate::SessionInfoUpdate(info) = notification.update else {
                panic!("extension envelope")
            };
            let event = info.meta.expect("metadata")[EXTERNAL_SUBAGENT_UPDATE_META_KEY].clone();
            for (key, value) in update.as_object().expect("wire object") {
                assert_eq!(
                    event.get(key),
                    Some(value),
                    "field {key} must survive normalization"
                );
            }
            if update["sessionUpdate"] == "subagent_update" && update.get("description").is_none() {
                assert!(event.get("description").is_none());
            }
        }
        Ok(())
    }

    #[gpui::test]
    async fn external_subagents_rfd_wire_creates_one_resumable_child(
        cx: &mut gpui::TestAppContext,
    ) {
        init_feature_flags_test(cx);
        let fs = fs::FakeFs::new(cx.executor());
        let project = project::Project::test(fs, [], cx).await;
        let harness = test_support::connect_fake_acp_connection(project.clone(), cx).await;
        let parent = cx
            .update(|cx| {
                harness.connection.clone().new_session(
                    project,
                    PathList::new(&[std::path::Path::new("/test")]),
                    cx,
                )
            })
            .await
            .expect("parent");
        let ctx = ClientContext {
            sessions: harness.connection.sessions.clone(),
            session_list: Rc::new(RefCell::new(None)),
            request_elicitations: harness.connection.request_elicitations.clone(),
            session_notification_state: harness.connection.session_notification_state.clone(),
            read_only: false,
            terminal_sandbox_wrap: None,
            active_request_scopes: None,
            pending_session_updates: None,
        };
        let parent_id = parent.read_with(cx, |thread, _| thread.session_id().clone());
        let child_id = acp_v2::SessionId::new("rfd-worker");
        let mut async_cx = cx.to_async();
        let mut send = |session_id, update| {
            handle_session_notification_with_subagents(
                SessionNotificationWithSubagents { session_id, update },
                &mut async_cx,
                &ctx,
            )
        };
        send(
            parent_id.clone(),
            serde_json::json!({"sessionUpdate":"subagent_update", "sessionId":"rfd-worker", "title":"Review", "state":{"state":"running"}}),
        );
        send(
            child_id.clone(),
            serde_json::json!({"sessionUpdate":"session_message", "messageId":"launch", "senderSessionId":parent_id, "recipientSessionId":"rfd-worker", "content":[{"type":"text","text":"Read one file"}]}),
        );
        send(
            parent_id.clone(),
            serde_json::json!({"sessionUpdate":"subagent_update", "sessionId":"rfd-worker", "state":{"state":"idle", "stopReason":"end_turn"}}),
        );
        send(
            child_id.clone(),
            serde_json::json!({"sessionUpdate":"agent_message_chunk", "content":{"type":"text","text":"Late result"}}),
        );
        send(
            parent_id,
            serde_json::json!({"sessionUpdate":"subagent_update", "sessionId":"rfd-worker", "state":{"state":"running"}}),
        );
        send(
            child_id.clone(),
            serde_json::json!({"sessionUpdate":"session_message", "messageId":"resume", "content":[{"type":"text","text":"Read another file"}]}),
        );
        let child = parent
            .read_with(cx, |thread, _| thread.external_subagent(&child_id))
            .expect("announced child");
        assert_eq!(
            child.read_with(cx, |thread, _| thread.status()),
            acp_thread::ThreadStatus::Generating
        );
        assert_eq!(child.read_with(cx, |thread, _| thread.entries().len()), 3);
        assert_eq!(
            parent.read_with(cx, |thread, _| thread.entries().len()),
            1,
            "reuse the same worker card"
        );
        assert!(harness.connection.sessions.borrow().contains_key(&child_id));
    }

    #[test]
    fn completion_rate_claude_metadata_preserves_plugins_and_requests_stream_events() {
        let plugins = serde_json::json!({"plugins": [{"type": "local", "path": "/plugin"}]});
        let meta = claude_code_completion_meta(Some(acp::Meta::from_iter([(
            "claudeCode".into(),
            serde_json::json!({"options": plugins}),
        )])));
        assert_eq!(
            meta.get("claudeCode")
                .and_then(|value| value.get("options")),
            Some(&plugins)
        );
        assert_eq!(
            meta.get("claudeCode")
                .and_then(|value| value.get("emitRawSDKMessages")),
            Some(&serde_json::json!([{ "type": "stream_event" }]))
        );
        assert!(claude_code_completion_meta(None).contains_key("claudeCode"));
    }

    #[test]
    fn completion_rate_claude_notification_parses_wire_contract() {
        use agent_client_protocol::JsonRpcMessage as _;
        let notification = ClaudeSdkMessageNotification::parse_message("_claude/sdkMessage", &serde_json::json!({
            "sessionId": "session", "message": {"type": "stream_event", "parent_tool_use_id": null, "event": {"type": "message_stop"}},
        })).expect("Claude extension notification");
        assert_eq!(notification.session_id, acp_v2::SessionId::new("session"));
        assert_eq!(
            notification.message.pointer("/event/type"),
            Some(&serde_json::json!("message_stop"))
        );
        assert!(!ClaudeSdkMessageNotification::matches_method(
            "session/update"
        ));
    }

    #[test]
    fn worker_write_paths_reject_escape_and_protected_metadata() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let workspace = directory.path().join("workspace");
        let outside = directory.path().join("outside");
        std::fs::create_dir_all(workspace.join(".git"))?;
        std::fs::create_dir(&outside)?;
        let policy = acp_thread::SandboxWrap {
            writable_paths: vec![workspace.clone()],
            protected_paths: vec![workspace.join(".git")],
            ..Default::default()
        };
        assert!(validate_worker_write_path(&workspace.join("new/file.rs"), &policy).is_ok());
        assert!(validate_worker_write_path(&outside.join("file.rs"), &policy).is_err());
        assert!(
            validate_worker_write_path(&workspace.join("../outside/file.rs"), &policy).is_err()
        );
        assert!(validate_worker_write_path(&workspace.join(".git/config"), &policy).is_err());
        assert!(validate_worker_write_path(std::path::Path::new("relative.rs"), &policy).is_err());
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(&outside, workspace.join("escape"))?;
            std::os::unix::fs::symlink(outside.join("missing"), workspace.join("dangling"))?;
            assert!(
                validate_worker_write_path(&workspace.join("escape/file.rs"), &policy).is_err()
            );
            assert!(
                validate_worker_write_path(&workspace.join("dangling/file.rs"), &policy).is_err()
            );
        }
        Ok(())
    }

    fn init_feature_flags_test(cx: &mut gpui::TestAppContext) {
        cx.update(|cx| {
            let mut settings_store = SettingsStore::test(cx);
            settings_store.register_setting::<feature_flags::FeatureFlagsSettings>();
            cx.set_global(settings_store);
            cx.update_flags(false, vec![]);
        });
    }

    #[gpui::test]
    async fn client_capabilities_include_elicitation_without_acp_beta(
        cx: &mut gpui::TestAppContext,
    ) {
        init_feature_flags_test(cx);
        let capabilities = client_capabilities_for_agent(&AgentId::new("codex-acp"), false, false);
        let elicitation = capabilities
            .elicitation
            .expect("elicitation should always be advertised");

        assert!(elicitation.form.is_some());
        assert!(elicitation.url.is_some());
    }

    #[gpui::test]
    async fn request_scoped_elicitation_during_auth_uses_connection_store(
        cx: &mut gpui::TestAppContext,
    ) {
        init_feature_flags_test(cx);
        cx.update(|cx| {
            cx.update_flags(false, vec![AcpBetaFeatureFlag::NAME.to_string()]);
        });

        let fs = fs::FakeFs::new(cx.executor());
        fs.insert_tree("/", serde_json::json!({ "a": {} })).await;
        let project = project::Project::test(fs, [std::path::Path::new("/a")], cx).await;

        let request_id = acp::RequestId::Number(1);
        let (harness, response_rx) =
            test_support::connect_fake_acp_connection_with_auth_elicitation(
                project,
                acp::CreateElicitationRequest::new(
                    acp::ElicitationFormMode::new(
                        acp::ElicitationRequestScope::new(request_id.clone()),
                        acp::ElicitationSchema::new().string("name", true),
                    ),
                    "Provide a name",
                ),
                cx,
            )
            .await;
        let connection = harness.connection.clone();
        let auth_task =
            cx.update(|cx| connection.authenticate(acp_v2::AuthMethodId::new("login"), cx));
        cx.run_until_parked();

        let store = connection
            .request_elicitations()
            .expect("ACP connections expose request-scoped elicitations");
        let elicitation_id = store.read_with(cx, |store, _| {
            let [elicitation] = store.elicitations() else {
                panic!(
                    "expected one request-scoped elicitation, got {:?}",
                    store.elicitations()
                );
            };
            let acp_v2::ElicitationScope::Request(scope) = elicitation.request.scope() else {
                panic!("expected request-scoped elicitation");
            };
            assert_eq!(scope.request_id.to_string(), request_id.to_string());
            elicitation.id.clone()
        });
        assert!(
            connection.sessions.borrow().is_empty(),
            "auth-time request-scoped elicitations must not require a session"
        );

        let expected_content = std::collections::BTreeMap::from([(
            "name".to_string(),
            acp::ElicitationContentValue::from("Ada"),
        )]);
        store.update(cx, |store, cx| {
            store.respond_to_elicitation(
                &elicitation_id,
                acp_v2::CreateElicitationResponse::new(acp_v2::ElicitationAction::Accept(
                    acp_v2::ElicitationAcceptAction::new().content(
                        serde_json::from_value::<
                            std::collections::BTreeMap<String, acp_v2::ElicitationContentValue>,
                        >(
                            serde_json::to_value(expected_content.clone())
                                .expect("serialize content"),
                        )
                        .expect("convert content"),
                    ),
                )),
                cx,
            );
        });

        let response = response_rx
            .recv()
            .await
            .expect("fake auth flow should receive elicitation response");
        assert_eq!(
            response.action,
            acp::ElicitationAction::Accept(
                acp::ElicitationAcceptAction::new().content(expected_content)
            )
        );
        auth_task.await.expect("auth should complete");
    }

    #[gpui::test]
    async fn request_scoped_url_elicitation_completion_after_create_is_observed(
        cx: &mut gpui::TestAppContext,
    ) {
        init_feature_flags_test(cx);
        cx.update(|cx| {
            cx.update_flags(false, vec![AcpBetaFeatureFlag::NAME.to_string()]);
        });

        let fs = fs::FakeFs::new(cx.executor());
        fs.insert_tree("/", serde_json::json!({ "a": {} })).await;
        let project = project::Project::test(fs, [std::path::Path::new("/a")], cx).await;

        let request_id = acp::RequestId::Number(1);
        let url_elicitation_id = acp::ElicitationId::new("auth-url");
        let (harness, response_rx) =
            test_support::connect_fake_acp_connection_with_auth_elicitation_completion(
                project,
                acp::CreateElicitationRequest::new(
                    acp::ElicitationUrlMode::new(
                        acp::ElicitationRequestScope::new(request_id.clone()),
                        url_elicitation_id.clone(),
                        "https://auth.example.com/device",
                    ),
                    "Authorize Zed in your browser",
                ),
                acp::CompleteElicitationNotification::new(url_elicitation_id),
                cx,
            )
            .await;
        let connection = harness.connection.clone();
        let auth_task =
            cx.update(|cx| connection.authenticate(acp_v2::AuthMethodId::new("login"), cx));
        cx.run_until_parked();

        let store = connection
            .request_elicitations()
            .expect("ACP connections expose request-scoped elicitations");
        let elicitation_id = store.read_with(cx, |store, _| {
            let [elicitation] = store.elicitations() else {
                panic!(
                    "expected one request-scoped elicitation, got {:?}",
                    store.elicitations()
                );
            };
            elicitation.id.clone()
        });
        store.update(cx, |store, cx| {
            store.respond_to_elicitation(
                &elicitation_id,
                acp_v2::CreateElicitationResponse::new(acp_v2::ElicitationAction::Accept(
                    acp_v2::ElicitationAcceptAction::new(),
                )),
                cx,
            );
        });
        let response = response_rx
            .recv()
            .await
            .expect("fake auth flow should receive elicitation response");
        assert_eq!(
            response.action,
            acp::ElicitationAction::Accept(acp::ElicitationAcceptAction::new())
        );
        cx.run_until_parked();

        store.read_with(cx, |store, _| {
            let [elicitation] = store.elicitations() else {
                panic!(
                    "expected one request-scoped elicitation, got {:?}",
                    store.elicitations()
                );
            };
            let acp_v2::ElicitationScope::Request(scope) = elicitation.request.scope() else {
                panic!("expected request-scoped elicitation");
            };
            assert_eq!(scope.request_id.to_string(), request_id.to_string());
            assert!(matches!(
                elicitation.status,
                acp_thread::ElicitationStatus::Completed
            ));
        });

        auth_task.await.expect("auth should complete");
    }

    #[gpui::test]
    async fn request_scoped_elicitation_ignores_open_sessions(cx: &mut gpui::TestAppContext) {
        init_feature_flags_test(cx);
        cx.update(|cx| {
            cx.update_flags(false, vec![AcpBetaFeatureFlag::NAME.to_string()]);
        });

        let fs = fs::FakeFs::new(cx.executor());
        fs.insert_tree("/", serde_json::json!({ "a": {} })).await;
        let project = project::Project::test(fs, [std::path::Path::new("/a")], cx).await;

        let request_id = acp::RequestId::Number(1);
        let (harness, response_rx) =
            test_support::connect_fake_acp_connection_with_auth_elicitation(
                project.clone(),
                acp::CreateElicitationRequest::new(
                    acp::ElicitationFormMode::new(
                        acp::ElicitationRequestScope::new(request_id.clone()),
                        acp::ElicitationSchema::new().string("name", true),
                    ),
                    "Provide a name",
                ),
                cx,
            )
            .await;
        let connection = harness.connection.clone();
        let work_dirs = util::path_list::PathList::new(&[std::path::Path::new("/a")]);

        let first_thread = cx
            .update(|cx| {
                connection.clone().load_session(
                    acp_v2::SessionId::new("session-1"),
                    project.clone(),
                    work_dirs.clone(),
                    None,
                    cx,
                )
            })
            .await
            .expect("first load_session should succeed");
        let second_thread = cx
            .update(|cx| {
                connection.clone().load_session(
                    acp_v2::SessionId::new("session-2"),
                    project,
                    work_dirs,
                    None,
                    cx,
                )
            })
            .await
            .expect("second load_session should succeed");
        cx.run_until_parked();
        assert_eq!(
            connection.sessions.borrow().len(),
            2,
            "test setup should have multiple open sessions"
        );

        let auth_task =
            cx.update(|cx| connection.authenticate(acp_v2::AuthMethodId::new("login"), cx));
        cx.run_until_parked();

        let store = connection
            .request_elicitations()
            .expect("ACP connections expose request-scoped elicitations");
        let elicitation_id = store.read_with(cx, |store, _| {
            let [elicitation] = store.elicitations() else {
                panic!(
                    "expected one request-scoped elicitation, got {:?}",
                    store.elicitations()
                );
            };
            let acp_v2::ElicitationScope::Request(scope) = elicitation.request.scope() else {
                panic!("expected request-scoped elicitation");
            };
            assert_eq!(scope.request_id.to_string(), request_id.to_string());
            elicitation.id.clone()
        });

        for thread in [first_thread, second_thread] {
            thread.read_with(cx, |thread, _| {
                assert!(
                    thread.entries().iter().all(|entry| !matches!(
                        entry,
                        acp_thread::AgentThreadEntry::Elicitation(_)
                    )),
                    "request-scoped elicitation should not be inserted into a session thread"
                );
            });
        }

        store.update(cx, |store, cx| {
            store.respond_to_elicitation(
                &elicitation_id,
                acp_v2::CreateElicitationResponse::new(acp_v2::ElicitationAction::Decline),
                cx,
            );
        });

        let response = response_rx
            .recv()
            .await
            .expect("fake auth flow should receive elicitation response");
        assert_eq!(response.action, acp::ElicitationAction::Decline);
        auth_task.await.expect("auth should complete");
    }

    #[test]
    fn cursor_client_capabilities_include_parameterized_model_picker_meta() {
        let capabilities = client_capabilities_for_agent(&AgentId::new(CURSOR_ID), false, false);
        let meta = capabilities
            .meta
            .expect("expected client capabilities meta");

        assert_eq!(
            meta.get(PARAMETERIZED_MODEL_PICKER_META_KEY),
            Some(&serde_json::json!(true))
        );
        assert_eq!(meta.get("terminal_output"), Some(&serde_json::json!(true)));
        assert_eq!(meta.get("terminal-auth"), Some(&serde_json::json!(true)));
    }

    #[test]
    fn non_cursor_client_capabilities_do_not_include_parameterized_model_picker_meta() {
        let capabilities = client_capabilities_for_agent(&AgentId::new("codex-acp"), false, false);
        let meta = capabilities
            .meta
            .expect("expected client capabilities meta");

        assert!(!meta.contains_key(PARAMETERIZED_MODEL_PICKER_META_KEY));
    }

    #[test]
    fn codex_client_capabilities_advertise_file_change_reports() {
        let capabilities =
            client_capabilities_for_agent(&AgentId::new(CODEX_ACP_AGENT_ID), false, false);
        let meta = capabilities
            .meta
            .expect("expected client capabilities meta");

        assert_eq!(
            meta.get("jetbrains"),
            Some(&serde_json::json!({
                "air": {
                    "version": 1,
                    "capabilities": ["agentFileChangeReport"],
                },
            }))
        );
    }

    #[test]
    fn non_codex_agents_do_not_advertise_codex_file_change_reports() {
        let capabilities =
            client_capabilities_for_agent(&AgentId::new("other-agent"), false, false);
        let meta = capabilities
            .meta
            .expect("expected client capabilities meta");

        assert!(!meta.contains_key("jetbrains"));
    }

    #[test]
    fn client_capabilities_include_boolean_config_options() {
        let capabilities = client_capabilities_for_agent(&AgentId::new("codex-acp"), false, false);

        assert!(
            capabilities
                .session
                .and_then(|session| session.config_options)
                .and_then(|config_options| config_options.boolean)
                .is_some()
        );
    }

    #[test]
    fn client_capabilities_only_include_compaction_with_acp_beta() {
        for (beta_enabled, expected_compaction) in
            [(false, None), (true, Some(serde_json::json!({})))]
        {
            let capabilities =
                client_capabilities_for_agent(&AgentId::new("codex-acp"), false, beta_enabled);
            let capabilities =
                serde_json::to_value(capabilities).expect("client capabilities should serialize");
            let session = capabilities
                .get("session")
                .expect("session capabilities should be advertised");
            assert_eq!(session.get("compaction"), expected_compaction.as_ref());
        }
    }

    #[gpui::test]
    async fn connection_routes_terminal_auth_without_acp_beta(cx: &mut gpui::TestAppContext) {
        init_feature_flags_test(cx);

        let fs = fs::FakeFs::new(cx.executor());
        fs.insert_tree("/", serde_json::json!({ "project": {} }))
            .await;
        let project = project::Project::test(fs, [std::path::Path::new("/project")], cx).await;
        let mut harness = test_support::connect_fake_acp_connection(project.clone(), cx).await;
        cx.update(|cx| {
            AllAgentServersSettings::override_global(
                AllAgentServersSettings(HashMap::from_iter([(
                    "test".to_string(),
                    settings::CustomAgentServerSettings::Custom {
                        path: PathBuf::from("test-agent"),
                        args: vec!["--acp".into(), "--verbose".into()],
                        env: HashMap::from_iter([
                            ("BASE".into(), "1".into()),
                            ("SHARED".into(), "base".into()),
                        ]),
                        default_mode: None,
                        default_config_options: HashMap::default(),
                        favorite_config_option_values: HashMap::default(),
                    }
                    .into(),
                )])),
                cx,
            );
        });
        cx.run_until_parked();
        assert_eq!(
            harness.connection.auth_methods(),
            &[acp_v2::AuthMethod::Agent(acp_v2::AuthMethodAgent::new(
                "login", "Login"
            ))]
        );
        let method_id = acp_v2::AuthMethodId::new("login");
        let mut methods = acp_thread::auth_methods::from_v1(vec![acp::AuthMethod::Terminal(
            acp::AuthMethodTerminal::new(method_id.0.clone(), "First-class login")
                .args(vec!["first-class-auth".into(), "".into()])
                .env(std::collections::HashMap::from_iter([
                    ("SHARED".into(), "override".into()),
                    ("EMPTY".into(), "".into()),
                ]))
                .meta(acp::Meta::from_iter([(
                    "terminal-auth".to_string(),
                    serde_json::json!({
                        "label": "Legacy login",
                        "command": "legacy-agent",
                        "args": ["legacy-auth"],
                    }),
                )])),
        )])
        .expect("terminal authentication should adapt");
        let Some(acp_v2::AuthMethod::Terminal(terminal)) = methods.first_mut() else {
            panic!("expected a shared terminal authentication method");
        };
        terminal.env.push(
            acp_v2::EnvVariable::new("EXTRA", "2").meta(acp_v2::Meta::from_iter([(
                "extension".into(),
                serde_json::json!({"nested": [null, true]}),
            )])),
        );
        let advertised_methods = methods.clone();
        Rc::get_mut(&mut harness.connection)
            .expect("test harness should have the only connection handle")
            .auth_methods = methods;

        let terminal_task = cx
            .update(|cx| {
                cx.update_flags(true, Vec::new());
                feature_flags::FeatureFlagsSettings::override_global(
                    feature_flags::FeatureFlagsSettings {
                        overrides: HashMap::from_iter([(
                            AcpBetaFeatureFlag::NAME.into(),
                            "off".into(),
                        )]),
                    },
                    cx,
                );
                assert!(!cx.has_flag::<AcpBetaFeatureFlag>());
                harness.connection.terminal_auth_task(&method_id, cx)
            })
            .expect("first-class terminal auth should be routed without ACP beta");
        let terminal_task = terminal_task
            .await
            .expect("first-class routing should resolve the test agent's external command");
        assert_eq!(terminal_task.command.as_deref(), Some("test-agent"));
        assert_eq!(
            terminal_task.args,
            vec!["--acp", "--verbose", "first-class-auth", ""]
        );
        assert_eq!(terminal_task.env.get("BASE").map(String::as_str), Some("1"));
        assert_eq!(
            terminal_task.env.get("SHARED").map(String::as_str),
            Some("override")
        );
        assert_eq!(terminal_task.env.get("EMPTY").map(String::as_str), Some(""));
        assert_eq!(
            terminal_task.env.get("EXTRA").map(String::as_str),
            Some("2")
        );
        assert_eq!(terminal_task.label, "First-class login");
        assert_eq!(harness.connection.auth_methods(), advertised_methods);
        cx.update(|cx| harness.connection.authenticate(method_id.clone(), cx))
            .await
            .expect_err("terminal auth must not use the agent-managed login RPC");

        let unknown = serde_json::json!({
            "type": "_future_login",
            "methodId": "unknown",
            "name": "Unknown login",
            "custom": {"nested": [null, true]},
            "_meta": {"terminal-auth": {"label": "Not executable", "command": "agent"}}
        });
        Rc::get_mut(&mut harness.connection)
            .expect("test harness should have the only connection handle")
            .auth_methods = vec![serde_json::from_value(unknown.clone()).expect("unknown method")];
        let unknown_id = acp_v2::AuthMethodId::new("unknown");
        cx.update(|cx| {
            assert!(
                harness
                    .connection
                    .terminal_auth_task(&unknown_id, cx)
                    .is_none()
            );
            harness.connection.authenticate(unknown_id, cx)
        })
        .await
        .expect_err("unknown auth must not use the agent-managed login RPC");
        assert_eq!(
            serde_json::to_value(harness.connection.auth_methods()).expect("shared methods"),
            serde_json::json!([unknown])
        );

        Rc::get_mut(&mut harness.connection)
            .expect("test harness should have the only connection handle")
            .auth_methods
            .clear();
        cx.update(|cx| harness.connection.authenticate(method_id, cx))
            .await
            .expect_err("unadvertised auth must not use the agent-managed login RPC");

        assert_eq!(harness.authenticate_count.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn terminal_auth_task_builds_spawn_from_prebuilt_command() {
        let command = AgentServerCommand {
            path: "/path/to/agent".into(),
            args: vec!["--acp".into(), "--verbose".into(), "/auth".into()],
            env: Some(HashMap::from_iter([
                ("BASE".into(), "1".into()),
                ("SHARED".into(), "override".into()),
                ("EXTRA".into(), "2".into()),
            ])),
        };
        let method = acp_v2::AuthMethodTerminal::new("login", "Login");

        let task = terminal_auth_task(&command, &AgentId::new("test-agent"), &method);

        assert_eq!(task.command.as_deref(), Some("/path/to/agent"));
        assert_eq!(task.args, vec!["--acp", "--verbose", "/auth"]);
        assert_eq!(
            task.env,
            HashMap::from_iter([
                ("BASE".into(), "1".into()),
                ("SHARED".into(), "override".into()),
                ("EXTRA".into(), "2".into()),
            ])
        );
        assert_eq!(task.label, "Login");
        assert_eq!(task.command_label, "Login");
    }

    #[test]
    fn legacy_terminal_auth_task_parses_meta_and_retries_session() {
        let method_id = acp_v2::AuthMethodId::new("legacy-login");
        let method = acp_v2::AuthMethod::Agent(
            acp_v2::AuthMethodAgent::new(method_id.clone(), "Login").meta(acp_v2::Meta::from_iter(
                [(
                    "terminal-auth".to_string(),
                    serde_json::json!({
                        "label": "legacy /auth",
                        "command": "legacy-agent",
                        "args": ["auth", "--interactive"],
                        "env": {
                            "AUTH_MODE": "interactive",
                        },
                    }),
                )],
            )),
        );

        let task = meta_terminal_auth_task(&AgentId::new("test-agent"), &method_id, &method)
            .expect("expected legacy terminal auth task");

        assert_eq!(task.id.0, "external-agent-test-agent-legacy-login-login");
        assert_eq!(task.command.as_deref(), Some("legacy-agent"));
        assert_eq!(task.args, vec!["auth", "--interactive"]);
        assert_eq!(
            task.env,
            HashMap::from_iter([("AUTH_MODE".into(), "interactive".into())])
        );
        assert_eq!(task.label, "legacy /auth");
    }

    #[test]
    fn legacy_terminal_auth_task_returns_none_for_invalid_meta() {
        let method_id = acp_v2::AuthMethodId::new("legacy-login");
        let method = acp_v2::AuthMethod::Agent(
            acp_v2::AuthMethodAgent::new(method_id.clone(), "Login").meta(acp_v2::Meta::from_iter(
                [(
                    "terminal-auth".to_string(),
                    serde_json::json!({
                        "label": "legacy /auth",
                    }),
                )],
            )),
        );

        assert!(
            meta_terminal_auth_task(&AgentId::new("test-agent"), &method_id, &method).is_none()
        );
    }

    #[test]
    fn first_class_terminal_auth_takes_precedence_over_legacy_meta() {
        let method_id = acp_v2::AuthMethodId::new("login");
        let method = acp_v2::AuthMethod::Terminal(
            acp_v2::AuthMethodTerminal::new(method_id, "Login")
                .args(vec!["/auth".into()])
                .env(vec![acp_v2::EnvVariable::new("AUTH_MODE", "first-class")])
                .meta(acp_v2::Meta::from_iter([(
                    "terminal-auth".to_string(),
                    serde_json::json!({
                        "label": "legacy /auth",
                        "command": "legacy-agent",
                        "args": ["legacy-auth"],
                        "env": {
                            "AUTH_MODE": "legacy",
                        },
                    }),
                )])),
        );

        let command = AgentServerCommand {
            path: "/path/to/agent".into(),
            args: vec!["--acp".into(), "/auth".into()],
            env: Some(HashMap::from_iter([
                ("BASE".into(), "1".into()),
                ("AUTH_MODE".into(), "first-class".into()),
            ])),
        };

        let task = match &method {
            acp_v2::AuthMethod::Terminal(terminal) => {
                terminal_auth_task(&command, &AgentId::new("test-agent"), terminal)
            }
            _ => unreachable!(),
        };

        assert_eq!(task.command.as_deref(), Some("/path/to/agent"));
        assert_eq!(task.args, vec!["--acp", "/auth"]);
        assert_eq!(
            task.env,
            HashMap::from_iter([
                ("BASE".into(), "1".into()),
                ("AUTH_MODE".into(), "first-class".into()),
            ])
        );
        assert_eq!(task.label, "Login");
    }

    #[test]
    fn trailing_stderr_only_uses_final_stderr_block() {
        let debug_log = AcpDebugLog::default();
        debug_log.record_line(AcpDebugMessageDirection::Stderr, "stale stderr");
        debug_log.record_line(
            AcpDebugMessageDirection::Incoming,
            r#"{"method":"initialized"}"#,
        );

        assert_eq!(debug_log.trailing_stderr(), None);

        debug_log.record_line(AcpDebugMessageDirection::Stderr, "recent stderr");
        assert_eq!(
            debug_log.trailing_stderr().as_deref(),
            Some("recent stderr")
        );
    }

    #[test]
    fn debug_log_records_each_json_rpc_batch_entry() {
        let debug_log = AcpDebugLog::default();
        debug_log.record_line(
            AcpDebugMessageDirection::Incoming,
            r#"{"jsonrpc":"2.0","method":"legacy/update"}"#,
        );
        debug_log.record_line(
            AcpDebugMessageDirection::Incoming,
            r#"[
                {"jsonrpc":"2.0","method":"session/update","params":{"value":1}},
                null,
                [{"jsonrpc":"2.0","method":"nested/update"}],
                {"jsonrpc":"2.0","id":1,"method":"session/one","params":{"value":2}},
                {"jsonrpc":"2.0","id":{"invalid":true},"method":"invalid/id"}
            ]"#,
        );
        debug_log.record_line(
            AcpDebugMessageDirection::Outgoing,
            r#"[
                {"jsonrpc":"2.0","id":1,"result":{"accepted":true}},
                {"jsonrpc":"2.0","id":null,"error":{"code":-32600,"message":"Invalid Request"}}
            ]"#,
        );

        let (messages, _receiver) = debug_log.subscribe();
        let mut messages = messages.iter();

        assert!(matches!(
            messages.next(),
            Some(AcpDebugMessage {
                direction: AcpDebugMessageDirection::Incoming,
                message: AcpDebugMessageContent::Notification { method, .. },
            }) if method.as_ref() == "legacy/update"
        ));
        assert!(matches!(
            messages.next(),
            Some(AcpDebugMessage {
                direction: AcpDebugMessageDirection::Incoming,
                message: AcpDebugMessageContent::Notification { method, .. },
            }) if method.as_ref() == "session/update"
        ));
        assert!(matches!(
            messages.next(),
            Some(AcpDebugMessage {
                direction: AcpDebugMessageDirection::Incoming,
                message: AcpDebugMessageContent::Request { id, method, .. },
            }) if id == &acp::RequestId::Number(1) && method.as_ref() == "session/one"
        ));
        assert!(matches!(
            messages.next(),
            Some(AcpDebugMessage {
                direction: AcpDebugMessageDirection::Outgoing,
                message: AcpDebugMessageContent::Response {
                    id,
                    result: Ok(Some(_)),
                },
            }) if id == &acp::RequestId::Number(1)
        ));
        assert!(matches!(
            messages.next(),
            Some(AcpDebugMessage {
                direction: AcpDebugMessageDirection::Outgoing,
                message: AcpDebugMessageContent::Response {
                    id,
                    result: Err(_),
                },
            }) if id == &acp::RequestId::Null
        ));
        assert!(messages.next().is_none());
    }

    #[test]
    fn session_directories_use_ordered_paths_when_supported() {
        let work_dirs = PathList::new(&[
            std::path::PathBuf::from("/workspace-b"),
            std::path::PathBuf::from("/workspace-a"),
            std::path::PathBuf::from("/workspace-c"),
        ]);

        let directories =
            session_directories_from_work_dirs(&work_dirs, true).expect("work dirs should convert");

        assert_eq!(
            directories,
            SessionDirectories {
                cwd: std::path::PathBuf::from("/workspace-b"),
                additional_directories: vec![
                    std::path::PathBuf::from("/workspace-a"),
                    std::path::PathBuf::from("/workspace-c")
                ],
            }
        );

        let session_id = acp_v2::SessionId::new("session-1");
        let new_session_request = directories
            .clone()
            .into_new_session_request(Vec::new(), None);
        let load_session_request =
            directories
                .clone()
                .into_load_session_request(session_id.clone(), Vec::new(), None);
        let resume_session_request =
            directories.into_resume_session_request(session_id, Vec::new(), None);

        assert_eq!(
            new_session_request.cwd,
            std::path::PathBuf::from("/workspace-b")
        );
        assert_eq!(
            new_session_request.additional_directories,
            vec![
                std::path::PathBuf::from("/workspace-a"),
                std::path::PathBuf::from("/workspace-c")
            ]
        );
        assert_eq!(
            load_session_request.additional_directories,
            new_session_request.additional_directories
        );
        assert_eq!(
            resume_session_request.additional_directories,
            new_session_request.additional_directories
        );
    }

    #[test]
    fn session_directories_drop_additional_paths_when_unsupported() {
        let work_dirs = PathList::new(&[
            std::path::PathBuf::from("/workspace-b"),
            std::path::PathBuf::from("/workspace-a"),
        ]);

        let directories = session_directories_from_work_dirs(&work_dirs, false)
            .expect("work dirs should convert");

        assert_eq!(
            directories,
            SessionDirectories {
                cwd: std::path::PathBuf::from("/workspace-b"),
                additional_directories: Vec::new(),
            }
        );
    }

    #[test]
    fn session_info_work_dirs_preserve_cwd_then_additional_directories() {
        let work_dirs = work_dirs_from_session_info(
            std::path::PathBuf::from("/workspace-b"),
            vec![
                std::path::PathBuf::from("/workspace-a"),
                std::path::PathBuf::from("/workspace-c"),
            ],
        );

        assert_eq!(
            work_dirs.ordered_paths().cloned().collect::<Vec<_>>(),
            vec![
                std::path::PathBuf::from("/workspace-b"),
                std::path::PathBuf::from("/workspace-a"),
                std::path::PathBuf::from("/workspace-c"),
            ]
        );
    }

    #[test]
    fn session_info_work_dirs_deduplicate_cwd_and_additional_directories() {
        let work_dirs = work_dirs_from_session_info(
            std::path::PathBuf::from("/workspace-b"),
            vec![
                std::path::PathBuf::from("/workspace-a"),
                std::path::PathBuf::from("/workspace-b"),
                std::path::PathBuf::from("/workspace-a"),
                std::path::PathBuf::from("/workspace-c"),
            ],
        );

        assert_eq!(
            work_dirs.ordered_paths().cloned().collect::<Vec<_>>(),
            vec![
                std::path::PathBuf::from("/workspace-b"),
                std::path::PathBuf::from("/workspace-a"),
                std::path::PathBuf::from("/workspace-c"),
            ]
        );
    }

    #[gpui::test]
    async fn session_list_includes_additional_directories_in_work_dirs(
        cx: &mut gpui::TestAppContext,
    ) {
        let connection = connect_session_list_test_agent(
            vec![
                acp::SessionInfo::new("session-1", "/workspace-b").additional_directories(vec![
                    std::path::PathBuf::from("/workspace-a"),
                    std::path::PathBuf::from("/workspace-b"),
                    std::path::PathBuf::from("/workspace-a"),
                    std::path::PathBuf::from("/workspace-c"),
                ]),
            ],
            cx,
        )
        .await;
        let session_list = AcpSessionList::new(connection, false);

        let response = cx
            .update(|cx| session_list.list_sessions(AgentSessionListRequest::default(), cx))
            .await
            .expect("session list should load");
        let session = response
            .sessions
            .first()
            .expect("session list should include the returned session");
        let work_dirs = session
            .work_dirs
            .as_ref()
            .expect("session should include work dirs");

        assert_eq!(
            work_dirs.ordered_paths().cloned().collect::<Vec<_>>(),
            vec![
                std::path::PathBuf::from("/workspace-b"),
                std::path::PathBuf::from("/workspace-a"),
                std::path::PathBuf::from("/workspace-c"),
            ]
        );
    }

    async fn connect_session_list_test_agent(
        sessions: Vec<acp::SessionInfo>,
        cx: &mut gpui::TestAppContext,
    ) -> ConnectionTo<Agent> {
        let (client_transport, agent_transport) = agent_client_protocol::Channel::duplex();
        let sessions = Arc::new(sessions);

        cx.background_spawn(
            Agent
                .builder()
                .name("list-test-agent")
                .on_receive_request(
                    {
                        let sessions = sessions.clone();
                        async move |_request: acp::ListSessionsRequest, responder, _cx| {
                            responder.respond(acp::ListSessionsResponse::new((*sessions).clone()))
                        }
                    },
                    agent_client_protocol::on_receive_request!(),
                )
                .connect_to(agent_transport),
        )
        .detach();

        let (connection_tx, connection_rx) = futures::channel::oneshot::channel();
        cx.background_spawn(Client.builder().name("list-test-client").connect_with(
            client_transport,
            move |connection: ConnectionTo<Agent>| async move {
                connection_tx.send(connection).ok();
                futures::future::pending::<Result<(), acp::Error>>().await
            },
        ))
        .detach();

        connection_rx
            .await
            .expect("failed to receive ACP connection")
    }

    #[gpui::test]
    async fn additional_directories_support_respects_agent_capability(
        cx: &mut gpui::TestAppContext,
    ) {
        cx.update(|cx| {
            let store = settings::SettingsStore::test(cx);
            cx.set_global(store);
        });

        let fs = fs::FakeFs::new(cx.executor());
        fs.insert_tree("/", serde_json::json!({ "a": {}, "b": {} }))
            .await;
        let project = project::Project::test(fs, [std::path::Path::new("/a")], cx).await;
        let mut harness = test_support::connect_fake_acp_connection(project, cx).await;

        let work_dirs = PathList::new(&[
            std::path::PathBuf::from("/workspace-b"),
            std::path::PathBuf::from("/workspace-a"),
        ]);

        let missing_capability = harness
            .connection
            .session_directories_from_work_dirs(&work_dirs)
            .expect("work dirs should convert");
        assert!(missing_capability.additional_directories.is_empty());

        Rc::get_mut(&mut harness.connection)
            .expect("test harness should own the only ACP connection handle")
            .agent_capabilities
            .session_capabilities
            .additional_directories = Some(acp::SessionAdditionalDirectoriesCapabilities::new());

        let supported = harness
            .connection
            .session_directories_from_work_dirs(&work_dirs)
            .expect("work dirs should convert");
        assert_eq!(
            supported,
            SessionDirectories {
                cwd: std::path::PathBuf::from("/workspace-b"),
                additional_directories: vec![std::path::PathBuf::from("/workspace-a")],
            }
        );
    }

    async fn connect_session_delete_test_agent(
        deleted_sessions: Arc<std::sync::Mutex<Vec<acp_v2::SessionId>>>,
        cx: &mut gpui::TestAppContext,
    ) -> ConnectionTo<Agent> {
        let (client_transport, agent_transport) = agent_client_protocol::Channel::duplex();

        cx.background_spawn(
            Agent
                .builder()
                .name("delete-test-agent")
                .on_receive_request(
                    {
                        let deleted_sessions = deleted_sessions.clone();
                        async move |request: acp::DeleteSessionRequest, responder, _cx| {
                            deleted_sessions
                                .lock()
                                .expect("deleted sessions lock should not be poisoned")
                                .push(acp_v2::SessionId::new(request.session_id.0));
                            responder.respond(acp::DeleteSessionResponse::default())
                        }
                    },
                    agent_client_protocol::on_receive_request!(),
                )
                .connect_to(agent_transport),
        )
        .detach();

        let (connection_tx, connection_rx) = futures::channel::oneshot::channel();
        cx.background_spawn(Client.builder().name("delete-test-client").connect_with(
            client_transport,
            move |connection: ConnectionTo<Agent>| async move {
                connection_tx.send(connection).ok();
                futures::future::pending::<Result<(), acp::Error>>().await
            },
        ))
        .detach();

        connection_rx
            .await
            .expect("failed to receive ACP connection")
    }

    #[gpui::test]
    async fn settings_changes_refresh_active_connection_defaults(cx: &mut gpui::TestAppContext) {
        cx.update(|cx| {
            let store = settings::SettingsStore::test(cx);
            cx.set_global(store);
        });

        let fs = fs::FakeFs::new(cx.executor());
        fs.insert_tree("/", serde_json::json!({ "a": {} })).await;
        let project = project::Project::test(fs, [std::path::Path::new("/a")], cx).await;
        let harness = test_support::connect_fake_acp_connection(project, cx).await;

        cx.update(|cx| {
            AllAgentServersSettings::override_global(
                AllAgentServersSettings(HashMap::from_iter([(
                    "test".to_string(),
                    settings::CustomAgentServerSettings::Custom {
                        path: PathBuf::from("test-agent"),
                        args: Vec::new(),
                        env: HashMap::default(),
                        default_mode: Some("manual".to_string()),
                        default_config_options: HashMap::from_iter([(
                            "mode".to_string(),
                            AgentConfigOptionValue::from("manual"),
                        )]),
                        favorite_config_option_values: HashMap::default(),
                    }
                    .into(),
                )])),
                cx,
            );
        });
        cx.run_until_parked();

        assert_eq!(
            harness.connection.defaults.mode(),
            Some(acp_v2::SessionConfigValueId::new("manual"))
        );
        assert_eq!(
            harness
                .connection
                .defaults
                .config_option("mode")
                .as_ref()
                .and_then(AgentConfigOptionValue::as_value_id),
            Some("manual"),
        );

        cx.update(|cx| {
            AllAgentServersSettings::override_global(
                AllAgentServersSettings(HashMap::default()),
                cx,
            );
        });
        cx.run_until_parked();

        assert_eq!(harness.connection.defaults.mode(), None);
        assert_eq!(harness.connection.defaults.config_option("mode"), None);
    }

    #[gpui::test]
    async fn loaded_sessions_do_not_apply_config_defaults(cx: &mut gpui::TestAppContext) {
        cx.update(|cx| {
            let store = SettingsStore::test(cx);
            cx.set_global(store);
        });
        let fs = fs::FakeFs::new(cx.executor());
        fs.insert_tree("/", serde_json::json!({"a": {}})).await;
        let project = project::Project::test(fs, [std::path::Path::new("/a")], cx).await;
        let (connection, requests) = connect_config_defaults_test_agent(cx).await;
        connection.defaults.set(
            None,
            HashMap::from_iter([("web_search".into(), AgentConfigOptionValue::Boolean(true))]),
        );
        let connection = Rc::new(connection);
        let returned = vec![acp::SessionConfigOption::boolean(
            "web_search",
            "Web Search",
            false,
        )];
        let session_id = acp_v2::SessionId::new("saved-config");
        let thread = cx
            .update(|cx| {
                connection.clone().open_or_create_session(
                    session_id.clone(),
                    project,
                    PathList::new(&[std::path::Path::new("/a")]),
                    None,
                    {
                        let returned = returned.clone();
                        |_, _, _| {
                            async move {
                                Ok(SessionConfigResponse {
                                    modes: None,
                                    config_options: Some(returned),
                                })
                            }
                            .boxed_local()
                        }
                    },
                    cx,
                )
            })
            .await
            .expect("load preserves saved settings");
        cx.run_until_parked();
        assert!(
            requests.lock().expect("config requests").is_empty(),
            "load must not send saved defaults"
        );
        let options = cx
            .read(|cx| connection.session_config_options(thread.read(cx).session_id(), cx))
            .expect("loaded config");
        assert_eq!(
            options.config_options(),
            acp_thread::config_options::from_v1(returned).expect("v2 projection")
        );
    }

    #[gpui::test]
    async fn default_config_options_apply_boolean_defaults(cx: &mut gpui::TestAppContext) {
        let (connection, set_config_requests) = connect_config_defaults_test_agent_with_handler(
            |_| {
                Ok(acp::SetSessionConfigOptionResponse::new(vec![
                    acp::SessionConfigOption::boolean("web_search", "Web Search", true),
                ]))
            },
            cx,
        )
        .await;
        connection.defaults.set(
            None,
            HashMap::from_iter([(
                "web_search".to_string(),
                AgentConfigOptionValue::Boolean(true),
            )]),
        );
        let config_options = ConfigOptions::new(Rc::new(RefCell::new(vec![
            acp_v2::SessionConfigOption::boolean("web_search", "Web Search", false),
        ])));

        let mut async_cx = cx.to_async();
        let _default_config_task = connection.apply_default_config_options(
            &acp_v2::SessionId::new("session-config-defaults"),
            &config_options,
            &mut async_cx,
        );
        drop(async_cx);
        cx.run_until_parked();

        let requests = set_config_requests
            .lock()
            .expect("set config requests mutex poisoned");
        assert_eq!(
            requests.as_slice(),
            &[serde_json::json!({
                "sessionId": "session-config-defaults", "configId": "web_search",
                "type": "boolean", "value": true
            })]
        );

        let options = config_options.config_options.borrow();
        assert!(
            matches!(&options[0].kind, acp_v2::SessionConfigKind::Boolean(boolean) if boolean.current_value)
        );
    }

    #[gpui::test]
    async fn default_config_options_refresh_dependent_options(cx: &mut gpui::TestAppContext) {
        for (initial_efforts, updated_efforts, default_effort, reasoning_first) in [
            (vec!["medium", "high"], vec!["medium", "ultra"], None, false),
            (
                vec!["medium", "high"],
                vec!["medium", "ultra"],
                Some("ultra"),
                false,
            ),
            (vec![], vec!["medium", "ultra"], Some("ultra"), false),
            (
                vec!["medium", "high"],
                vec!["medium", "ultra"],
                Some("high"),
                false,
            ),
            (vec!["medium", "high"], vec![], Some("high"), false),
            (
                vec!["medium", "high"],
                vec!["medium", "ultra"],
                Some("ultra"),
                true,
            ),
            (
                vec!["medium", "ultra"],
                vec!["medium", "ultra"],
                Some("ultra"),
                true,
            ),
        ] {
            let expected_effort = if default_effort == Some("ultra") {
                "ultra"
            } else {
                "medium"
            };
            let mut initial_options =
                model_config_options("initial-model", "medium", &initial_efforts);
            if reasoning_first {
                initial_options.reverse();
            }
            let initially_supported = reasoning_first && initial_efforts.contains(&"ultra");
            let expected_options =
                model_config_options("new-model", expected_effort, &updated_efforts);
            let selected_model = Mutex::new("initial-model");
            let (connection, requests) = connect_config_defaults_test_agent_with_handler(
                move |request| {
                    let mut selected_model = selected_model.lock().expect("selected model");
                    let effort = if request.config_id == acp::SessionConfigId::new("model") {
                        assert_eq!(
                            request.value,
                            acp::SessionConfigOptionValue::value_id("new-model")
                        );
                        *selected_model = "new-model";
                        "medium"
                    } else {
                        assert_eq!(
                            request.config_id,
                            acp::SessionConfigId::new("reasoning_effort")
                        );
                        assert_eq!(
                            request.value,
                            acp::SessionConfigOptionValue::value_id("ultra")
                        );
                        "ultra"
                    };
                    let efforts = if *selected_model == "initial-model" {
                        &initial_efforts
                    } else {
                        &updated_efforts
                    };
                    Ok(acp::SetSessionConfigOptionResponse::new(
                        model_config_options(*selected_model, effort, efforts),
                    ))
                },
                cx,
            )
            .await;
            let mut defaults = HashMap::from_iter([(
                "model".to_string(),
                AgentConfigOptionValue::from("new-model"),
            )]);
            if let Some(effort) = default_effort {
                defaults.insert(
                    "reasoning_effort".to_string(),
                    AgentConfigOptionValue::from(effort),
                );
            }
            connection.defaults.set(None, defaults);
            let (_, config_options) =
                config_state(None, Some(initial_options)).expect("initial config");
            let config_options = config_options.expect("config options");
            let mut receiver = config_options.rx.clone();
            let _default_config_task = connection.apply_default_config_options(
                &acp_v2::SessionId::new("session"),
                &config_options,
                &mut cx.to_async(),
            );
            cx.run_until_parked();

            assert_eq!(
                *config_options.config_options.borrow(),
                acp_thread::config_options::from_v1(expected_options).expect("expected config"),
            );
            assert_eq!(receiver.changed().now_or_never(), Some(Ok(())));
            let mut expected_requests = vec![serde_json::json!({
                "sessionId": "session", "configId": "model", "value": "new-model"
            })];
            if initially_supported {
                expected_requests.insert(
                    0,
                    serde_json::json!({
                        "sessionId": "session", "configId": "reasoning_effort", "value": "ultra"
                    }),
                );
            }
            if default_effort == Some("ultra") {
                expected_requests.push(serde_json::json!({
                    "sessionId": "session", "configId": "reasoning_effort", "value": "ultra"
                }));
            }
            assert_eq!(*requests.lock().expect("requests"), expected_requests);
        }
    }

    fn model_config_options(
        model: &str,
        effort: &str,
        efforts: &[&str],
    ) -> Vec<acp::SessionConfigOption> {
        let mut options = vec![serde_json::json!({
            "id": "model", "name": "Model", "category": "model", "type": "select",
            "currentValue": model,
            "options": [
                {"value": "initial-model", "name": "Initial model"},
                {"value": "new-model", "name": "New model"}
            ]
        })];
        if !efforts.is_empty() {
            options.push(serde_json::json!({
                "id": "reasoning_effort", "name": "Reasoning effort", "category": "thought_level",
                "type": "select", "currentValue": effort,
                "options": efforts.iter().map(|value| serde_json::json!({
                    "value": value, "name": value
                })).collect::<Vec<_>>()
            }));
        }
        serde_json::from_value(serde_json::json!(options)).expect("model config options")
    }

    #[gpui::test]
    async fn default_config_options_preserve_state_on_error(cx: &mut gpui::TestAppContext) {
        let (connection, requests) = connect_config_defaults_test_agent_with_handler(
            |_| Err(acp::Error::invalid_params()),
            cx,
        )
        .await;
        connection.defaults.set(
            None,
            HashMap::from_iter([(
                "web_search".to_string(),
                AgentConfigOptionValue::Boolean(true),
            )]),
        );
        let original_options = vec![acp_v2::SessionConfigOption::boolean(
            "web_search",
            "Web Search",
            false,
        )];
        let config_options = ConfigOptions::new(Rc::new(RefCell::new(original_options.clone())));
        let mut receiver = config_options.rx.clone();
        let _default_config_task = connection.apply_default_config_options(
            &acp_v2::SessionId::new("session"),
            &config_options,
            &mut cx.to_async(),
        );
        cx.run_until_parked();

        assert_eq!(*config_options.config_options.borrow(), original_options);
        assert!(receiver.changed().now_or_never().is_none());
        assert_eq!(requests.lock().expect("requests").len(), 1);
    }

    #[gpui::test]
    async fn default_config_options_stop_when_defaults_conflict(cx: &mut gpui::TestAppContext) {
        let (connection, requests) = connect_config_defaults_test_agent_with_handler(
            |request| {
                Ok(acp::SetSessionConfigOptionResponse::new(vec![
                    acp::SessionConfigOption::boolean(
                        "first",
                        "First",
                        request.config_id.0.as_ref() == "first",
                    ),
                    acp::SessionConfigOption::boolean(
                        "second",
                        "Second",
                        request.config_id.0.as_ref() == "second",
                    ),
                ]))
            },
            cx,
        )
        .await;
        connection.defaults.set(
            None,
            HashMap::from_iter([
                ("first".to_string(), AgentConfigOptionValue::Boolean(true)),
                ("second".to_string(), AgentConfigOptionValue::Boolean(true)),
            ]),
        );
        let config_options = ConfigOptions::new(Rc::new(RefCell::new(vec![
            acp_v2::SessionConfigOption::boolean("first", "First", false),
            acp_v2::SessionConfigOption::boolean("second", "Second", false),
        ])));
        let _default_config_task = connection.apply_default_config_options(
            &acp_v2::SessionId::new("session"),
            &config_options,
            &mut cx.to_async(),
        );
        cx.run_until_parked();
        assert_eq!(requests.lock().expect("requests").len(), 3);
    }

    #[gpui::test]
    async fn default_config_options_cancel_on_session_release(cx: &mut gpui::TestAppContext) {
        cx.update(|cx| {
            let store = settings::SettingsStore::test(cx);
            cx.set_global(store);
        });
        let fs = fs::FakeFs::new(cx.executor());
        fs.insert_tree("/", serde_json::json!({"a": {}})).await;
        let project = project::Project::test(fs, [std::path::Path::new("/a")], cx).await;
        let (response_sender, response_receiver) = async_channel::bounded::<()>(1);
        let (connection, requests) = connect_config_defaults_test_agent_with_async_handler(
            move |request| {
                let response_receiver = response_receiver.clone();
                async move {
                    assert_eq!(request.config_id, acp::SessionConfigId::new("model"));
                    response_receiver
                        .recv()
                        .await
                        .expect("release model response");
                    Ok(acp::SetSessionConfigOptionResponse::new(
                        model_config_options("new-model", "medium", &["medium", "ultra"]),
                    ))
                }
            },
            cx,
        )
        .await;
        let connection = Rc::new(connection);
        connection.defaults.set(
            None,
            HashMap::from_iter([
                (
                    "model".to_string(),
                    AgentConfigOptionValue::from("new-model"),
                ),
                (
                    "reasoning_effort".to_string(),
                    AgentConfigOptionValue::from("ultra"),
                ),
            ]),
        );
        let session_id = acp_v2::SessionId::new("session");
        let initial_options = model_config_options("initial-model", "medium", &["medium"]);
        let thread = cx
            .update(|cx| {
                connection.clone().open_or_create_session(
                    session_id.clone(),
                    project.clone(),
                    PathList::new(&[std::path::Path::new("/a")]),
                    None,
                    move |_, _, _| {
                        async move {
                            Ok(SessionConfigResponse {
                                modes: None,
                                config_options: Some(initial_options),
                            })
                        }
                        .boxed_local()
                    },
                    cx,
                )
            })
            .await
            .expect("load initial session");
        // Exercise the new-session task lifetime without making load apply defaults.
        let options = connection
            .sessions
            .borrow()
            .get(&session_id)
            .and_then(|session| session.config_options.clone())
            .expect("session config");
        let defaults =
            connection.apply_default_config_options(&session_id, &options, &mut cx.to_async());
        connection
            .sessions
            .borrow_mut()
            .get_mut(&session_id)
            .expect("registered session")
            .default_config_task = Some(defaults);
        cx.run_until_parked();
        assert_eq!(requests.lock().expect("requests").len(), 1);
        drop(thread);
        release_dropped_entities(cx);

        let replacement_options = vec![acp::SessionConfigOption::boolean(
            "web_search",
            "Web Search",
            false,
        )];
        let expected_options = acp_thread::config_options::from_v1(replacement_options.clone())
            .expect("replacement config");
        let _replacement_thread = cx
            .update(|cx| {
                connection.clone().open_or_create_session(
                    session_id.clone(),
                    project,
                    PathList::new(&[std::path::Path::new("/a")]),
                    None,
                    move |_, _, _| {
                        async move {
                            Ok(SessionConfigResponse {
                                modes: None,
                                config_options: Some(replacement_options),
                            })
                        }
                        .boxed_local()
                    },
                    cx,
                )
            })
            .await
            .expect("load replacement session");
        response_sender
            .send(())
            .await
            .expect("unblock stale response");
        cx.run_until_parked();

        assert_eq!(requests.lock().expect("requests").len(), 1);
        let sessions = connection.sessions.borrow();
        let config_options = sessions
            .get(&session_id)
            .expect("replacement session")
            .config_options
            .as_ref()
            .expect("replacement config");
        assert_eq!(*config_options.config_options.borrow(), expected_options);
    }

    #[gpui::test]
    async fn shared_config_values_use_v1_wire_shapes(cx: &mut gpui::TestAppContext) {
        use acp_thread::AgentSessionConfigOptions;

        let (connection, requests) = connect_config_defaults_test_agent(cx).await;
        let config = ConfigOptions::new(Rc::new(RefCell::new(vec![
            acp_v2::SessionConfigOption::boolean("config", "Config", false),
        ])));
        let options = AcpSessionConfigOptions {
            session_id: acp_v2::SessionId::new("session"),
            connection: connection.connection.clone(),
            config_options: config,
        };
        let original = options.config_options();
        let unknown =
            acp_v2::SessionConfigOptionValue::Other(acp_v2::OtherSessionConfigOptionValue::new(
                "_custom",
                serde_json::json!("private-value"),
                Default::default(),
            ));
        assert!(
            cx.update(|cx| options.set_config_option("config".into(), unknown, cx))
                .await
                .is_err()
        );
        assert_eq!(options.config_options(), original);
        assert!(requests.lock().expect("requests").is_empty());

        for value in [
            acp_v2::SessionConfigOptionValue::id("choice"),
            acp_v2::SessionConfigOptionValue::boolean(true),
        ] {
            let updated = cx
                .update(|cx| options.set_config_option("config".into(), value, cx))
                .await
                .expect("supported v1 value");
            assert!(updated.is_empty());
            assert_eq!(options.config_options(), updated);
        }
        assert_eq!(
            requests.lock().expect("requests").as_slice(),
            &[
                serde_json::json!({"sessionId": "session", "configId": "config", "value": "choice"}),
                serde_json::json!({
                    "sessionId": "session", "configId": "config", "type": "boolean", "value": true
                })
            ]
        );
    }

    async fn connect_config_defaults_test_agent(
        cx: &mut gpui::TestAppContext,
    ) -> (AcpConnection, Arc<Mutex<Vec<serde_json::Value>>>) {
        connect_config_defaults_test_agent_with_handler(
            |_| Ok(acp::SetSessionConfigOptionResponse::new(Vec::new())),
            cx,
        )
        .await
    }

    async fn connect_config_defaults_test_agent_with_handler(
        handler: impl Fn(
            acp::SetSessionConfigOptionRequest,
        ) -> Result<acp::SetSessionConfigOptionResponse, acp::Error>
        + Send
        + Sync
        + 'static,
        cx: &mut gpui::TestAppContext,
    ) -> (AcpConnection, Arc<Mutex<Vec<serde_json::Value>>>) {
        connect_config_defaults_test_agent_with_async_handler(
            move |request| futures::future::ready(handler(request)),
            cx,
        )
        .await
    }

    async fn connect_config_defaults_test_agent_with_async_handler<
        F: Future<Output = Result<acp::SetSessionConfigOptionResponse, acp::Error>> + Send + 'static,
    >(
        handler: impl Fn(acp::SetSessionConfigOptionRequest) -> F + Send + Sync + 'static,
        cx: &mut gpui::TestAppContext,
    ) -> (AcpConnection, Arc<Mutex<Vec<serde_json::Value>>>) {
        let set_config_requests = Arc::new(Mutex::new(Vec::new()));
        let (client_transport, client_inspection) = agent_client_protocol::Channel::duplex();
        let (agent_inspection, agent_transport) = agent_client_protocol::Channel::duplex();
        cx.background_spawn({
            let set_config_requests = set_config_requests.clone();
            async move {
                agent_client_protocol::Channel::bridge_with_inspection(
                    client_inspection,
                    agent_inspection,
                    move |message| {
                        if let agent_client_protocol::RawJsonRpcMessage::Request(request) = message
                            && request.method.as_ref() == "session/set_config_option"
                        {
                            set_config_requests
                                .lock()
                                .expect("set config requests mutex poisoned")
                                .push(
                                    serde_json::to_value(&request.params)
                                        .expect("raw config request parameters"),
                                );
                        }
                        Ok(())
                    },
                    |_| Ok(()),
                )
                .await
                .log_err();
            }
        })
        .detach();

        cx.background_spawn(
            Agent
                .builder()
                .name("config-defaults-test-agent")
                .on_receive_request(
                    async move |request: acp::SetSessionConfigOptionRequest, responder, _cx| {
                        match handler(request).await {
                            Ok(response) => responder.respond(response),
                            Err(error) => responder.respond_with_error(error),
                        }
                    },
                    agent_client_protocol::on_receive_request!(),
                )
                .connect_to(agent_transport),
        )
        .detach();

        let (connection_tx, connection_rx) = futures::channel::oneshot::channel();
        let client_io_task = cx.background_spawn(async move {
            Client
                .builder()
                .name("config-defaults-test-client")
                .connect_with(
                    client_transport,
                    move |connection: ConnectionTo<Agent>| async move {
                        connection_tx.send(connection).ok();
                        futures::future::pending::<Result<(), acp::Error>>().await
                    },
                )
                .await
                .ok();
        });

        let client_conn = connection_rx
            .await
            .expect("failed to receive ACP connection");
        let sessions = Rc::new(RefCell::new(HashMap::default()));

        let connection = cx.update(|cx| {
            let request_elicitations = cx.new(|_| ElicitationStore::default());
            AcpConnection::new_for_test(
                client_conn,
                sessions,
                Rc::new(SessionNotificationState::default()),
                acp::AgentCapabilities::default(),
                request_elicitations,
                WeakEntity::new_invalid(),
                client_io_task,
                mpsc::unbounded().0,
                Task::ready(()),
                cx,
            )
        });

        (connection, set_config_requests)
    }

    #[gpui::test]
    async fn session_list_delete_sends_session_delete_when_supported(
        cx: &mut gpui::TestAppContext,
    ) {
        let deleted_sessions = Arc::new(std::sync::Mutex::new(Vec::new()));
        let connection = connect_session_delete_test_agent(deleted_sessions.clone(), cx).await;
        let session_list = AcpSessionList::new(connection, true);
        let session_id = acp_v2::SessionId::new("session-to-delete");

        cx.update(|cx| session_list.delete_session(&session_id, cx))
            .await
            .expect("delete_session failed");

        assert_eq!(
            *deleted_sessions
                .lock()
                .expect("deleted sessions lock should not be poisoned"),
            vec![session_id]
        );
    }

    #[gpui::test]
    async fn session_list_delete_does_not_send_when_unsupported(cx: &mut gpui::TestAppContext) {
        let deleted_sessions = Arc::new(std::sync::Mutex::new(Vec::new()));
        let connection = connect_session_delete_test_agent(deleted_sessions.clone(), cx).await;
        let session_list = AcpSessionList::new(connection, false);
        let session_id = acp_v2::SessionId::new("session-to-delete");

        let error = cx
            .update(|cx| session_list.delete_session(&session_id, cx))
            .await
            .expect_err("delete_session should fail when unsupported");

        assert!(
            error.to_string().contains("delete_session not supported"),
            "unexpected error: {error}"
        );
        assert!(
            deleted_sessions
                .lock()
                .expect("deleted sessions lock should not be poisoned")
                .is_empty()
        );
    }

    #[cfg(not(windows))]
    #[gpui::test]
    async fn startup_returns_error_when_agent_exits_before_initialization(
        cx: &mut gpui::TestAppContext,
    ) {
        cx.update(|cx| {
            let store = settings::SettingsStore::test(cx);
            cx.set_global(store);
        });
        cx.executor().allow_parking();

        let temp_dir = tempfile::tempdir().unwrap();
        let project = project::Project::example([temp_dir.path()], &mut cx.to_async()).await;
        let agent_server_store =
            project.read_with(cx, |project, _| project.agent_server_store().downgrade());
        let command = AgentServerCommand {
            path: "/bin/sh".into(),
            args: vec![
                "-c".into(),
                r#"printf '%s\n' 'npm error code ETARGET' 'npm error notarget No matching version found for @agentclientprotocol/claude-agent-acp@0.32.0 with a date before 4/28/2026, 12:11:38 PM.' >&2; exit 1"#.into(),
            ],
            env: None,
        };

        let mut async_cx = cx.to_async();
        let startup = AcpConnection::stdio(
            AgentId::new("test-agent"),
            project,
            command,
            agent_server_store,
            None,
            HashMap::default(),
            None,
            false,
            None,
            &mut async_cx,
        )
        .fuse();
        let timeout = cx
            .background_executor
            .timer(std::time::Duration::from_secs(5))
            .fuse();
        futures::pin_mut!(startup, timeout);

        let result = futures::select! {
            result = startup => result,
            _ = timeout => panic!("timed out waiting for failed ACP startup"),
        };

        let Err(error) = result else {
            panic!("expected ACP startup to fail");
        };
        let load_error = error
            .downcast::<LoadError>()
            .expect("startup failure should preserve the typed load error");
        match load_error {
            LoadError::Exited { status, stderr } => {
                assert!(!status.success(), "expected non-zero exit status");
                assert_eq!(
                    stderr.as_deref(),
                    Some(
                        "npm error code ETARGET\nnpm error notarget No matching version found for @agentclientprotocol/claude-agent-acp@0.32.0 with a date before 4/28/2026, 12:11:38 PM."
                    ),
                    "startup failure should retain the agent's final stderr"
                );
            }
            error => panic!("expected exited load error, got: {error:?}"),
        };
    }

    #[cfg(unix)]
    #[gpui::test]
    async fn agent_exit_preserves_final_updates_and_diagnostics(cx: &mut gpui::TestAppContext) {
        assert_agent_exit_preserves_final_output(false, cx).await;
    }

    #[cfg(unix)]
    #[gpui::test]
    async fn agent_exit_with_inherited_handles_preserves_final_output(
        cx: &mut gpui::TestAppContext,
    ) {
        assert_agent_exit_preserves_final_output(true, cx).await;
    }

    #[cfg(unix)]
    async fn assert_agent_exit_preserves_final_output(
        inherit_handles: bool,
        cx: &mut gpui::TestAppContext,
    ) {
        cx.update(|cx| {
            let store = settings::SettingsStore::test(cx);
            cx.set_global(store);
        });
        cx.executor().allow_parking();

        let directory = tempfile::tempdir().expect("create working directory");
        let project = project::Project::example([directory.path()], &mut cx.to_async()).await;
        let agent_server_store =
            project.read_with(cx, |project, _| project.agent_server_store().downgrade());
        let session_id = acp_v2::SessionId::new("exiting-agent");
        let initialize_response =
            serde_json::to_string(&acp::InitializeResponse::new(ProtocolVersion::V1))
                .expect("serialize initialization");
        let session_response =
            serde_json::to_string(&acp::NewSessionResponse::new(session_id.0.clone()))
                .expect("serialize session");
        let prompt_response =
            serde_json::to_string(&acp::PromptResponse::new(acp::StopReason::EndTurn))
                .expect("serialize prompt response");
        let notification = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": acp::SessionNotification::new(
                session_id.0,
                acp::SessionUpdate::AgentMessageChunk(acp::ContentChunk::new(
                    "final answer before exit".into(),
                )),
            ),
        });
        let inherited_handle_script = inherit_handles.then_some("sleep 2 &");
        let script = format!(
            r#"
read_request() {{
    IFS= read -r request || exit 2
    request_id=$(printf '%s' "$request" | sed -n 's/.*"id":\("[^"]*"\).*/\1/p')
    test -n "$request_id" || exit 3
}}
respond() {{
    printf '{{"jsonrpc":"2.0","id":%s,"result":%s}}\n' "$request_id" "$1"
}}
read_request
respond '{initialize_response}'
read_request
respond '{session_response}'
read_request
printf '%s\n' '{notification}'
respond '{prompt_response}'
printf '%s\n' 'final diagnostic before exit' >&2
{}
exit 7
"#,
            inherited_handle_script.unwrap_or_default(),
        );
        let command = AgentServerCommand {
            path: "/bin/sh".into(),
            args: vec!["-c".into(), script],
            env: None,
        };
        let timeout = cx
            .background_executor
            .timer(std::time::Duration::from_secs(5))
            .fuse();
        let exchange = async {
            let connection = Rc::new(
                AcpConnection::stdio(
                    AgentId::new("exiting-agent"),
                    project.clone(),
                    command,
                    agent_server_store,
                    None,
                    HashMap::default(),
                    None,
                    false,
                    None,
                    &mut cx.to_async(),
                )
                .await
                .expect("initialize scripted agent"),
            );
            let thread = cx
                .update(|cx| {
                    connection.clone().new_session(
                        project,
                        PathList::new(&[directory.path()]),
                        cx,
                    )
                })
                .await
                .expect("create scripted session");
            let (exit_sender, exit_receiver) = async_channel::unbounded();
            let _subscription = cx.update(|cx| {
                let debug_log = connection.debug_log.clone();
                let protocol_connection = connection.connection.clone();
                cx.subscribe(&thread, move |thread, event, cx| {
                    if let acp_thread::AcpThreadEvent::LoadError(error) = event {
                        let (messages, _receiver) = debug_log.subscribe();
                        let stderr_captured = messages.iter().any(|message| {
                            matches!(
                                &message.message,
                                AcpDebugMessageContent::Stderr { line }
                                    if line.as_ref() == "final diagnostic before exit"
                            )
                        });
                        exit_sender
                            .try_send((
                                thread.read(cx).to_markdown(cx),
                                error.clone(),
                                protocol_connection.is_incoming_closed(),
                                stderr_captured,
                            ))
                            .expect("capture exit event");
                    }
                })
            });
            let response = thread
                .update(cx, |thread, cx| thread.send_raw("Respond, then exit", cx))
                .await
                .expect("the final prompt response should arrive before closure")
                .expect("prompt should finish");
            assert_eq!(response.stop_reason, acp::StopReason::EndTurn);

            let (transcript_at_exit, error, incoming_closed_at_exit, stderr_captured_at_exit) =
                exit_receiver.recv().await.expect("receive process exit");
            if inherit_handles {
                assert!(
                    !incoming_closed_at_exit,
                    "the descendant should still hold the agent's stdout open"
                );
            } else {
                assert!(
                    incoming_closed_at_exit,
                    "normal process exit should reach clean stdout EOF before the exit event"
                );
            }
            cx.run_until_parked();
            let final_transcript = thread.read_with(cx, |thread, cx| thread.to_markdown(cx));
            assert!(
                final_transcript.contains("final answer before exit"),
                "the scripted agent's final notification should reach the thread"
            );
            assert!(
                transcript_at_exit.contains("final answer before exit"),
                "final notification should be applied before reporting exit: at exit {transcript_at_exit:?}, after settling {final_transcript:?}"
            );
            let LoadError::Exited { status, .. } = error else {
                panic!("expected process exit, got {error:?}");
            };
            assert_eq!(status.code(), Some(7));
            assert!(
                stderr_captured_at_exit,
                "the final diagnostic should be logged before reporting exit"
            );
            if inherit_handles {
                connection.connection.incoming_closed().await;
            }
        }
        .fuse();
        futures::pin_mut!(exchange, timeout);
        futures::select! {
            _ = exchange => {}
            _ = timeout => panic!("timed out waiting for scripted agent exit"),
        }
    }

    async fn connect_fake_agent(
        close_session_gate: Option<async_channel::Receiver<()>>,
        cx: &mut gpui::TestAppContext,
    ) -> (
        Rc<AcpConnection>,
        Entity<project::Project>,
        Arc<AtomicUsize>,
        Arc<AtomicUsize>,
        Arc<std::sync::Mutex<Vec<acp::SessionUpdate>>>,
        Arc<std::sync::Mutex<Option<async_channel::Receiver<()>>>>,
        Task<anyhow::Result<()>>,
    ) {
        connect_fake_agent_with_handle(close_session_gate, None, cx).await
    }

    async fn connect_fake_agent_with_handle(
        close_session_gate: Option<async_channel::Receiver<()>>,
        agent_sender: Option<futures::channel::oneshot::Sender<ConnectionTo<Client>>>,
        cx: &mut gpui::TestAppContext,
    ) -> (
        Rc<AcpConnection>,
        Entity<project::Project>,
        Arc<AtomicUsize>,
        Arc<AtomicUsize>,
        Arc<std::sync::Mutex<Vec<acp::SessionUpdate>>>,
        Arc<std::sync::Mutex<Option<async_channel::Receiver<()>>>>,
        Task<anyhow::Result<()>>,
    ) {
        cx.update(|cx| {
            let store = settings::SettingsStore::test(cx);
            cx.set_global(store);
        });

        let fs = fs::FakeFs::new(cx.executor());
        fs.insert_tree("/", serde_json::json!({ "a": {} })).await;
        let project = project::Project::test(fs, [std::path::Path::new("/a")], cx).await;

        let load_count = Arc::new(AtomicUsize::new(0));
        let close_count = Arc::new(AtomicUsize::new(0));
        let load_session_updates: Arc<std::sync::Mutex<Vec<acp::SessionUpdate>>> =
            Arc::new(std::sync::Mutex::new(Vec::new()));
        let load_session_gate: Arc<std::sync::Mutex<Option<async_channel::Receiver<()>>>> =
            Arc::new(std::sync::Mutex::new(None));
        let close_session_gate = Arc::new(std::sync::Mutex::new(close_session_gate));

        let (client_transport, agent_transport) = agent_client_protocol::Channel::duplex();

        let sessions: Rc<RefCell<HashMap<acp_v2::SessionId, AcpSession>>> =
            Rc::new(RefCell::new(HashMap::default()));
        let client_session_list: Rc<RefCell<Option<Rc<AcpSessionList>>>> =
            Rc::new(RefCell::new(None));

        // Build the fake agent side. It handles the requests issued by
        // `AcpConnection` during the test and tracks load/close counts.
        let agent_future = Agent
            .builder()
            .name("fake-agent")
            .on_receive_request(
                async move |req: acp::InitializeRequest, responder, _cx| {
                    responder.respond(
                        acp::InitializeResponse::new(req.protocol_version).agent_capabilities(
                            acp::AgentCapabilities::default()
                                .load_session(true)
                                .session_capabilities(
                                    acp::SessionCapabilities::default()
                                        .close(acp::SessionCloseCapabilities::new()),
                                ),
                        ),
                    )
                },
                agent_client_protocol::on_receive_request!(),
            )
            .on_receive_request(
                async move |_req: acp::AuthenticateRequest, responder, _cx| {
                    responder.respond(Default::default())
                },
                agent_client_protocol::on_receive_request!(),
            )
            .on_receive_request(
                async move |_req: acp::NewSessionRequest, responder, _cx| {
                    responder.respond(acp::NewSessionResponse::new(acp::SessionId::new("unused")))
                },
                agent_client_protocol::on_receive_request!(),
            )
            .on_receive_request(
                async move |_req: acp::PromptRequest, responder, _cx| {
                    responder.respond(acp::PromptResponse::new(acp::StopReason::EndTurn))
                },
                agent_client_protocol::on_receive_request!(),
            )
            .on_receive_request(
                {
                    let load_count = load_count.clone();
                    let load_session_updates = load_session_updates.clone();
                    let load_session_gate = load_session_gate.clone();
                    async move |req: acp::LoadSessionRequest, responder, cx| {
                        load_count.fetch_add(1, Ordering::SeqCst);

                        // Simulate spec-compliant history replay: send
                        // notifications to the client before responding to the
                        // load request.
                        let updates = std::mem::take(
                            &mut *load_session_updates
                                .lock()
                                .expect("load_session_updates mutex poisoned"),
                        );
                        for update in updates {
                            cx.send_notification(acp::SessionNotification::new(
                                req.session_id.clone(),
                                update,
                            ))?;
                        }

                        // If a gate was installed, park on it before responding
                        // so tests can interleave other work (e.g.
                        // `close_session`) with an in-flight load.
                        let gate = load_session_gate
                            .lock()
                            .expect("load_session_gate mutex poisoned")
                            .take();
                        if let Some(gate) = gate {
                            gate.recv().await.ok();
                        }

                        responder.respond(acp::LoadSessionResponse::new())
                    }
                },
                agent_client_protocol::on_receive_request!(),
            )
            .on_receive_request(
                {
                    let close_count = close_count.clone();
                    let close_session_gate = close_session_gate.clone();
                    async move |_req: acp::CloseSessionRequest, responder, _cx| {
                        close_count.fetch_add(1, Ordering::SeqCst);
                        let gate = close_session_gate
                            .lock()
                            .expect("close_session_gate mutex poisoned")
                            .take();
                        if let Some(gate) = gate {
                            gate.recv().await.ok();
                        }
                        responder.respond(acp::CloseSessionResponse::new())
                    }
                },
                agent_client_protocol::on_receive_request!(),
            )
            .on_receive_notification(
                async move |_notif: acp::CancelNotification, _cx| Ok(()),
                agent_client_protocol::on_receive_notification!(),
            )
            .connect_with(
                agent_transport,
                move |connection: ConnectionTo<Client>| async move {
                    if let Some(agent_sender) = agent_sender {
                        agent_sender
                            .send(connection)
                            .map_err(|_| acp::Error::internal_error())?;
                    }
                    futures::future::pending::<Result<(), acp::Error>>().await
                },
            );

        let agent_io_task = cx.background_spawn(agent_future);

        // Wire the production handler set into the fake client so inbound
        // requests/notifications from the fake agent reach the same
        // dispatcher that the real `stdio` path uses.
        let (dispatch_tx, dispatch_rx) = mpsc::unbounded::<ForegroundWork>();

        let (connection_tx, connection_rx) = futures::channel::oneshot::channel();
        let client_future = connect_client_future(
            "zed-test",
            client_transport,
            dispatch_tx.clone(),
            connection_tx,
            futures::future::pending(),
        );
        let client_io_task = cx.background_spawn(async move {
            client_future.await.ok();
        });

        let client_conn: ConnectionTo<Agent> = connection_rx
            .await
            .expect("failed to receive ACP connection handle");

        let response = client_conn
            .send_request(acp::InitializeRequest::new(ProtocolVersion::V1))
            .block_task()
            .await
            .expect("failed to initialize ACP connection");

        let agent_capabilities = response.agent_capabilities;

        let request_elicitations = cx.new(|_| ElicitationStore::default());
        let session_notification_state = Rc::new(SessionNotificationState::default());
        let dispatch_context = ClientContext {
            sessions: sessions.clone(),
            session_list: client_session_list.clone(),
            request_elicitations: request_elicitations.clone(),
            session_notification_state: session_notification_state.clone(),
            read_only: false,
            terminal_sandbox_wrap: None,
            active_request_scopes: None,
            pending_session_updates: None,
        };
        // `TestAppContext::spawn` hands out an `AsyncApp` by value, whereas the
        // production path uses `Context::spawn` which hands out `&mut AsyncApp`.
        // Bind the value-form to a local and take `&mut` of it to reuse the
        // same dispatch loop shape.
        let dispatch_task = cx.spawn({
            let mut dispatch_rx = dispatch_rx;
            move |cx| async move {
                let mut cx = cx;
                while let Some(work) = dispatch_rx.next().await {
                    work.run(&mut cx, &dispatch_context);
                }
            }
        });

        let agent_server_store =
            project.read_with(cx, |project, _| project.agent_server_store().downgrade());

        let connection = cx.update(|cx| {
            AcpConnection::new_for_test(
                client_conn,
                sessions,
                session_notification_state,
                agent_capabilities,
                request_elicitations,
                agent_server_store,
                client_io_task,
                dispatch_tx,
                dispatch_task,
                cx,
            )
        });

        let keep_agent_alive = cx.background_spawn(async move {
            agent_io_task.await.ok();
            anyhow::Ok(())
        });

        (
            Rc::new(connection),
            project,
            load_count,
            close_count,
            load_session_updates,
            load_session_gate,
            keep_agent_alive,
        )
    }

    fn drain_sdk_background_tasks(cx: &gpui::TestAppContext) {
        let executor = cx.executor();
        let dispatcher = executor
            .dispatcher()
            .as_test()
            .expect("deterministic test dispatcher");
        for _ in 0..1000 {
            if !dispatcher.tick(true) {
                return;
            }
        }
        panic!("SDK dispatch did not drain within bounded ticks");
    }

    #[gpui::test]
    async fn eof_cancels_elicitations_for_retained_idle_owners(cx: &mut gpui::TestAppContext) {
        let (agent_sender, agent_receiver) = futures::channel::oneshot::channel();
        let (connection, project, _, _, _, _, keep_agent_alive) =
            connect_fake_agent_with_handle(None, Some(agent_sender), cx).await;
        let agent = agent_receiver.await.expect("fake agent handle");
        let session_id = acp_v2::SessionId::new("retained-idle-session");
        let thread = cx
            .update(|cx| {
                connection.clone().load_session(
                    session_id.clone(),
                    project,
                    PathList::new(&[std::path::Path::new("/a")]),
                    None,
                    cx,
                )
            })
            .await
            .expect("load retained idle session");
        let store = connection
            .request_elicitations()
            .expect("connection elicitation store");
        let session_request = acp::CreateElicitationRequest::new(
            acp::ElicitationFormMode::new(
                acp::ElicitationSessionScope::new(session_id.0.clone()),
                acp::ElicitationSchema::new().string("name", true),
            ),
            "Session question",
        );
        let request_request = acp::CreateElicitationRequest::new(
            acp::ElicitationFormMode::new(
                acp::ElicitationRequestScope::new(acp::RequestId::Number(1)),
                acp::ElicitationSchema::new().string("name", true),
            ),
            "Connection question",
        );
        let session_rpc = agent.send_request(session_request.clone());
        let request_rpc = agent.send_request(request_request.clone());
        cx.run_until_parked();

        let session_entry = thread.read_with(cx, |thread, _| {
            assert_eq!(thread.status(), acp_thread::ThreadStatus::Idle);
            thread
                .entries()
                .iter()
                .find_map(|entry| match entry {
                    acp_thread::AgentThreadEntry::Elicitation(id) => Some(id.clone()),
                    _ => None,
                })
                .expect("session elicitation")
        });
        let request_entry = store.read_with(cx, |store, _| {
            let [elicitation] = store.elicitations() else {
                panic!("expected one connection elicitation");
            };
            assert!(matches!(
                elicitation.status,
                acp_thread::ElicitationStatus::Pending { .. }
            ));
            elicitation.id.clone()
        });
        let session_waiter = thread.update(cx, |thread, cx| {
            thread
                .request_elicitation(
                    acp_thread::elicitation::request_from_v1(session_request.clone())
                        .expect("convert request"),
                    cx,
                )
                .expect("session response waiter")
        });
        let request_waiter = store.update(cx, |store, cx| {
            store
                .request_elicitation(
                    acp_thread::elicitation::request_from_v1(request_request.clone())
                        .expect("convert request"),
                    cx,
                )
                .expect("connection response waiter")
        });
        let unrelated_store = cx.new(|_| ElicitationStore::default());
        let unrelated_waiter = unrelated_store.update(cx, |store, cx| {
            store
                .request_elicitation(
                    acp_thread::elicitation::request_from_v1(request_request.clone())
                        .expect("convert request"),
                    cx,
                )
                .expect("unrelated response waiter")
        });
        let queued_session_rpc = agent.send_request(session_request);
        let queued_request_rpc = agent.send_request(request_request);
        drain_sdk_background_tasks(cx);
        assert_eq!(
            thread.read_with(cx, |thread, _| {
                thread
                    .entries()
                    .iter()
                    .filter(|entry| matches!(entry, acp_thread::AgentThreadEntry::Elicitation(_)))
                    .count()
            }),
            2,
            "queued session request must not be applied before foreground dispatch"
        );
        assert_eq!(
            store.read_with(cx, |store, _| store.elicitations().len()),
            2
        );

        drop(keep_agent_alive);
        connection.connection.incoming_closed().await;
        cx.run_until_parked();

        thread.read_with(cx, |thread, _| {
            let (_, elicitation) = thread.elicitation(&session_entry).expect("retained entry");
            assert!(matches!(
                elicitation.status,
                acp_thread::ElicitationStatus::Canceled
            ));
            let entries = thread
                .entries()
                .iter()
                .filter_map(|entry| match entry {
                    acp_thread::AgentThreadEntry::Elicitation(id) => {
                        Some(thread.elicitation(id).expect("elicitation entry").1)
                    }
                    _ => None,
                })
                .collect::<Vec<_>>();
            assert_eq!(
                entries.len(),
                3,
                "queued request must register before cleanup"
            );
            assert!(
                entries
                    .iter()
                    .all(|entry| matches!(entry.status, acp_thread::ElicitationStatus::Canceled))
            );
            assert_eq!(thread.status(), acp_thread::ThreadStatus::Idle);
        });
        store.read_with(cx, |store, _| {
            let (_, elicitation) = store.elicitation(&request_entry).expect("retained entry");
            assert!(matches!(
                elicitation.status,
                acp_thread::ElicitationStatus::Canceled
            ));
            assert_eq!(store.elicitations().len(), 3);
            assert!(
                store
                    .elicitations()
                    .iter()
                    .all(|entry| matches!(entry.status, acp_thread::ElicitationStatus::Canceled))
            );
        });
        assert_eq!(
            session_waiter.await.action,
            acp_v2::ElicitationAction::Cancel
        );
        assert_eq!(
            request_waiter.await.action,
            acp_v2::ElicitationAction::Cancel
        );
        unrelated_store.update(cx, |store, cx| {
            assert!(matches!(
                store
                    .elicitations()
                    .first()
                    .expect("unrelated entry")
                    .status,
                acp_thread::ElicitationStatus::Pending { .. }
            ));
            store.cancel_all(cx);
        });
        assert_eq!(
            unrelated_waiter.await.action,
            acp_v2::ElicitationAction::Cancel
        );
        drop((
            session_rpc,
            request_rpc,
            queued_session_rpc,
            queued_request_rpc,
        ));
    }

    #[gpui::test]
    async fn eof_preserves_queued_elicitation_completion(cx: &mut gpui::TestAppContext) {
        let (agent_sender, agent_receiver) = futures::channel::oneshot::channel();
        let (connection, project, _, _, _, _, keep_agent_alive) =
            connect_fake_agent_with_handle(None, Some(agent_sender), cx).await;
        let agent = agent_receiver.await.expect("fake agent handle");
        let session_id = acp_v2::SessionId::new("completion-session");
        let thread = cx
            .update(|cx| {
                connection.clone().load_session(
                    session_id.clone(),
                    project,
                    PathList::new(&[std::path::Path::new("/a")]),
                    None,
                    cx,
                )
            })
            .await
            .expect("load completion session");
        let store = connection
            .request_elicitations()
            .expect("connection elicitation store");
        let session_scope =
            acp_v2::ElicitationScope::Session(acp_v2::ElicitationSessionScope::new(session_id.0));
        let request_scope = acp_v2::ElicitationScope::Request(
            acp_v2::ElicitationRequestScope::new(acp_v2::RequestId::Number(1)),
        );
        let url_request = |scope: acp_v2::ElicitationScope, id: &str| {
            acp_v2::CreateElicitationRequest::new(
                acp_v2::ElicitationUrlMode::new(
                    scope,
                    acp_v2::ElicitationId::new(id),
                    "https://auth.example.com/device",
                ),
                "Authorize in your browser",
            )
        };
        let accept = || {
            acp_v2::CreateElicitationResponse::new(acp_v2::ElicitationAction::Accept(
                acp_v2::ElicitationAcceptAction::new(),
            ))
        };
        let mut session_entries = Vec::new();
        for id in ["session-completed", "session-incomplete"] {
            let (entry, response) = thread.update(cx, |thread, cx| {
                thread
                    .request_elicitation_with_id(url_request(session_scope.clone(), id), cx)
                    .expect("session URL elicitation")
            });
            thread.update(cx, |thread, cx| {
                thread.respond_to_elicitation(&entry, accept(), cx);
            });
            assert_eq!(response.await, accept());
            session_entries.push(entry);
        }
        let mut request_entries = Vec::new();
        for id in ["request-completed", "request-incomplete"] {
            let (entry, response) = store.update(cx, |store, cx| {
                store
                    .request_elicitation_with_id(url_request(request_scope.clone(), id), cx)
                    .expect("request URL elicitation")
            });
            store.update(cx, |store, cx| {
                store.respond_to_elicitation(&entry, accept(), cx);
            });
            assert_eq!(response.await, accept());
            request_entries.push(entry);
        }
        cx.run_until_parked();

        for id in ["session-completed", "request-completed"] {
            agent
                .send_notification(acp_v2::CompleteElicitationNotification::new(
                    acp_v2::ElicitationId::new(id),
                ))
                .expect("send final completion");
        }
        drain_sdk_background_tasks(cx);
        drop(keep_agent_alive);
        drain_sdk_background_tasks(cx);
        cx.run_until_parked();
        connection.connection.incoming_closed().await;

        let [session_completed, session_incomplete] = session_entries.as_slice() else {
            panic!("expected two session URL elicitations");
        };
        thread.read_with(cx, |thread, _| {
            assert!(matches!(
                thread
                    .elicitation(session_completed)
                    .expect("completed URL")
                    .1
                    .status,
                acp_thread::ElicitationStatus::Completed
            ));
            assert!(matches!(
                thread
                    .elicitation(session_incomplete)
                    .expect("incomplete URL")
                    .1
                    .status,
                acp_thread::ElicitationStatus::Canceled
            ));
        });
        let [request_completed, request_incomplete] = request_entries.as_slice() else {
            panic!("expected two connection URL elicitations");
        };
        store.read_with(cx, |store, _| {
            assert!(matches!(
                store
                    .elicitation(request_completed)
                    .expect("completed URL")
                    .1
                    .status,
                acp_thread::ElicitationStatus::Completed
            ));
            assert!(matches!(
                store
                    .elicitation(request_incomplete)
                    .expect("incomplete URL")
                    .1
                    .status,
                acp_thread::ElicitationStatus::Canceled
            ));
        });
    }

    #[gpui::test]
    async fn transport_error_cancels_retained_elicitations(cx: &mut gpui::TestAppContext) {
        let store = cx.new(|_| ElicitationStore::default());
        let context = ClientContext {
            sessions: Rc::new(RefCell::new(HashMap::default())),
            session_list: Rc::new(RefCell::new(None)),
            request_elicitations: store.clone(),
            session_notification_state: Rc::new(SessionNotificationState::default()),
            read_only: false,
            terminal_sandbox_wrap: None,
            active_request_scopes: None,
            pending_session_updates: None,
        };
        let (dispatch_tx, mut dispatch_rx) = mpsc::unbounded::<ForegroundWork>();
        let dispatch_task = cx.spawn(move |mut cx| async move {
            while let Some(work) = dispatch_rx.next().await {
                work.run(&mut cx, &context);
            }
        });
        let (incoming_tx, incoming_rx) = mpsc::unbounded::<std::io::Result<String>>();
        let outgoing = futures::sink::unfold((), async move |(), _line: String| {
            Ok::<_, std::io::Error>(())
        });
        let (connection_tx, connection_rx) = futures::channel::oneshot::channel();
        let io_task = cx.background_spawn(connect_client_future(
            "failing-transport",
            Lines::new(Box::pin(outgoing), incoming_rx),
            dispatch_tx,
            connection_tx,
            futures::future::pending(),
        ));
        let _connection = connection_rx.await.expect("SDK connection handle");
        let (entry, response) = store.update(cx, |store, cx| {
            store
                .request_elicitation_with_id(
                    acp_v2::CreateElicitationRequest::new(
                        acp_v2::ElicitationFormMode::new(
                            acp_v2::ElicitationRequestScope::new(acp_v2::RequestId::Number(1)),
                            acp_v2::ElicitationSchema::new().string("name", true),
                        ),
                        "Retained question",
                    ),
                    cx,
                )
                .expect("retained elicitation")
        });
        incoming_tx
            .unbounded_send(Err(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "test transport failure",
            )))
            .expect("inject transport error");
        assert!(io_task.await.is_err(), "transport error must propagate");
        store.read_with(cx, |store, _| {
            assert!(matches!(
                store.elicitation(&entry).expect("retained entry").1.status,
                acp_thread::ElicitationStatus::Canceled
            ));
        });
        assert_eq!(response.await.action, acp_v2::ElicitationAction::Cancel);
        drop(dispatch_task);
    }

    #[test]
    fn disconnect_cleanup_reports_queue_failures() {
        let (sender, receiver) = mpsc::unbounded::<ForegroundWork>();
        drop(receiver);
        let error = futures::executor::block_on(settle_after_disconnect(sender))
            .expect_err("closed dispatch queue must reject cleanup");
        assert_eq!(error.code, ErrorCode::InternalError);

        let (sender, mut receiver) = mpsc::unbounded::<ForegroundWork>();
        let mut cleanup = Box::pin(settle_after_disconnect(sender));
        assert!(cleanup.as_mut().now_or_never().is_none());
        let accepted = futures::executor::block_on(receiver.next()).expect("accepted cleanup");
        drop(accepted);
        let error = futures::executor::block_on(cleanup)
            .expect_err("dropped cleanup acknowledgment must fail");
        assert_eq!(error.code, ErrorCode::InternalError);
    }

    #[gpui::test]
    async fn disconnect_cleanup_runs_once_for_both_completion_paths(cx: &mut gpui::TestAppContext) {
        let context = ClientContext {
            sessions: Rc::new(RefCell::new(HashMap::default())),
            session_list: Rc::new(RefCell::new(None)),
            request_elicitations: cx.new(|_| ElicitationStore::default()),
            session_notification_state: Rc::new(SessionNotificationState::default()),
            read_only: false,
            terminal_sandbox_wrap: None,
            active_request_scopes: None,
            pending_session_updates: None,
        };
        let (sender, mut receiver) = mpsc::unbounded::<ForegroundWork>();
        let cleanup = settle_after_disconnect(sender.clone()).boxed().shared();
        let mut close_callback = Box::pin(cleanup.clone());
        let mut connection_finished = Box::pin(cleanup.clone());
        assert!(close_callback.as_mut().now_or_never().is_none());
        assert!(connection_finished.as_mut().now_or_never().is_none());
        let work = receiver.next().await.expect("one cleanup work item");
        assert!(
            receiver.next().now_or_never().is_none(),
            "both completion paths must share one queued cleanup"
        );
        work.run(&mut cx.to_async(), &context);
        close_callback.await.expect("close callback cleanup");
        connection_finished
            .await
            .expect("connection completion cleanup");
        assert!(receiver.next().now_or_never().is_none());
    }

    #[gpui::test]
    async fn test_loaded_sessions_keep_state_until_last_handle_drops(
        cx: &mut gpui::TestAppContext,
    ) {
        let (
            connection,
            project,
            load_count,
            close_count,
            _load_session_updates,
            _load_session_gate,
            _keep_agent_alive,
        ) = connect_fake_agent(None, cx).await;

        let session_id = acp_v2::SessionId::new("session-1");
        let work_dirs = util::path_list::PathList::new(&[std::path::Path::new("/a")]);

        // Load the same session twice concurrently — the second call should join
        // the pending task rather than issuing a second ACP load_session RPC.
        let first_load = cx.update(|cx| {
            connection.clone().load_session(
                session_id.clone(),
                project.clone(),
                work_dirs.clone(),
                None,
                cx,
            )
        });
        let second_load = cx.update(|cx| {
            connection.clone().load_session(
                session_id.clone(),
                project.clone(),
                work_dirs.clone(),
                None,
                cx,
            )
        });

        let first_thread = first_load.await.expect("first load failed");
        let second_thread = second_load.await.expect("second load failed");
        cx.run_until_parked();

        assert_eq!(
            first_thread.entity_id(),
            second_thread.entity_id(),
            "concurrent loads for the same session should share one AcpThread"
        );
        assert_eq!(
            load_count.load(Ordering::SeqCst),
            1,
            "underlying ACP load_session should be called exactly once for concurrent loads"
        );

        drop(first_thread);
        release_dropped_entities(cx);

        assert_eq!(
            close_count.load(Ordering::SeqCst),
            0,
            "ACP close_session should not be sent while a thread handle is alive"
        );
        assert!(
            connection.sessions.borrow().contains_key(&session_id),
            "session should still be tracked while a thread handle is alive"
        );

        drop(second_thread);
        release_dropped_entities(cx);

        assert_eq!(
            close_count.load(Ordering::SeqCst),
            1,
            "ACP close_session should be sent exactly once when the last handle drops"
        );
        assert!(
            !connection.sessions.borrow().contains_key(&session_id),
            "session should be removed after the last handle drops"
        );
    }

    // Regression test: per the ACP spec, an agent replays the entire conversation
    // history as `session/update` notifications *before* responding to the
    // `session/load` request. These notifications must be applied to the
    // reconstructed thread, not dropped because the session hasn't been
    // registered yet.
    #[gpui::test]
    async fn test_load_session_replays_notifications_sent_before_response(
        cx: &mut gpui::TestAppContext,
    ) {
        let (
            connection,
            project,
            _load_count,
            _close_count,
            load_session_updates,
            _load_session_gate,
            _keep_agent_alive,
        ) = connect_fake_agent(None, cx).await;

        // Queue up some history updates that the fake agent will stream to
        // the client during the `load_session` call, before responding.
        *load_session_updates
            .lock()
            .expect("load_session_updates mutex poisoned") = vec![
            acp::SessionUpdate::UserMessageChunk(acp::ContentChunk::new(acp::ContentBlock::Text(
                acp::TextContent::new(String::from("hello agent")),
            ))),
            acp::SessionUpdate::AgentMessageChunk(acp::ContentChunk::new(acp::ContentBlock::Text(
                acp::TextContent::new(String::from("hi user")),
            ))),
        ];

        let session_id = acp_v2::SessionId::new("session-replay");
        let work_dirs = util::path_list::PathList::new(&[std::path::Path::new("/a")]);

        let thread = cx
            .update(|cx| {
                connection.clone().load_session(
                    session_id.clone(),
                    project.clone(),
                    work_dirs,
                    None,
                    cx,
                )
            })
            .await
            .expect("load_session failed");
        cx.run_until_parked();

        let entries = thread.read_with(cx, |thread, _| {
            thread
                .entries()
                .iter()
                .map(|entry| match entry {
                    acp_thread::AgentThreadEntry::UserMessage(_) => "user",
                    acp_thread::AgentThreadEntry::SessionMessage(_) => "session_message",
                    acp_thread::AgentThreadEntry::AssistantMessage(_) => "assistant",
                    acp_thread::AgentThreadEntry::ToolCall(_) => "tool_call",
                    acp_thread::AgentThreadEntry::Elicitation(_) => "elicitation",
                    acp_thread::AgentThreadEntry::ContextCompaction(_) => "compaction",
                })
                .collect::<Vec<_>>()
        });

        assert_eq!(
            entries,
            vec!["user", "assistant"],
            "replayed notifications should be applied to the thread"
        );
    }

    #[gpui::test]
    async fn test_dropping_loader_during_in_flight_load_closes_session(
        cx: &mut gpui::TestAppContext,
    ) {
        let (
            connection,
            project,
            load_count,
            close_count,
            _load_session_updates,
            load_session_gate,
            _keep_agent_alive,
        ) = connect_fake_agent(None, cx).await;

        // Install a gate so the fake agent's `load_session` handler parks
        // before sending its response. We'll close the session while the
        // load is parked.
        let (gate_tx, gate_rx) = async_channel::bounded::<()>(1);
        *load_session_gate
            .lock()
            .expect("load_session_gate mutex poisoned") = Some(gate_rx);

        let session_id = acp_v2::SessionId::new("session-close-during-load");
        let work_dirs = util::path_list::PathList::new(&[std::path::Path::new("/a")]);

        let load_task = cx.update(|cx| {
            connection.clone().load_session(
                session_id.clone(),
                project.clone(),
                work_dirs,
                None,
                cx,
            )
        });

        // Let the load RPC reach the agent and park on the gate.
        cx.run_until_parked();
        assert_eq!(
            load_count.load(Ordering::SeqCst),
            1,
            "load_session RPC should have been dispatched"
        );
        assert!(
            connection
                .pending_sessions
                .borrow()
                .contains_key(&session_id),
            "pending_sessions entry should exist while load is in flight"
        );
        assert!(
            connection.sessions.borrow().contains_key(&session_id),
            "sessions entry should be pre-registered to receive replay notifications"
        );

        drop(load_task);
        cx.run_until_parked();
        assert_eq!(
            close_count.load(Ordering::SeqCst),
            0,
            "ACP close_session must not be sent while the load RPC is in flight"
        );

        // Release the gate so the load RPC can finally respond.
        gate_tx.send(()).await.expect("gate send failed");
        drop(gate_tx);
        cx.run_until_parked();
        release_dropped_entities(cx);

        assert_eq!(
            close_count.load(Ordering::SeqCst),
            1,
            "ACP close_session should be sent exactly once"
        );
        assert!(
            !connection.sessions.borrow().contains_key(&session_id),
            "sessions entry should be removed once the loaded thread has no holders"
        );
        assert!(
            !connection
                .pending_sessions
                .borrow()
                .contains_key(&session_id),
            "pending_sessions entry should be removed once the load resolves"
        );
    }

    #[gpui::test]
    async fn test_abandoned_load_does_not_retain_connection_or_thread(
        cx: &mut gpui::TestAppContext,
    ) {
        let (
            connection,
            project,
            load_count,
            close_count,
            _load_session_updates,
            load_session_gate,
            keep_agent_alive,
        ) = connect_fake_agent(None, cx).await;
        let (gate_tx, gate_rx) = async_channel::bounded::<()>(1);
        *load_session_gate
            .lock()
            .expect("load_session_gate mutex poisoned") = Some(gate_rx);

        let session_id = acp_v2::SessionId::new("session-abandoned-during-load");
        let load_task = cx.update(|cx| {
            connection.clone().load_session(
                session_id.clone(),
                project,
                PathList::new(&[Path::new("/a")]),
                None,
                cx,
            )
        });
        cx.run_until_parked();
        assert_eq!(load_count.load(Ordering::SeqCst), 1);

        let weak_thread = connection
            .sessions
            .borrow()
            .get(&session_id)
            .expect("in-flight load should register a thread")
            .thread
            .clone();
        let weak_connection = Rc::downgrade(&connection);
        drop(load_task);
        drop(connection);
        release_dropped_entities(cx);

        let connection_retained = weak_connection.upgrade().is_some();
        let thread_retained = weak_thread.upgrade().is_some();
        let close_count_before_response = close_count.load(Ordering::SeqCst);
        assert_eq!(close_count_before_response, 0);

        if let Err(error) = gate_tx.try_send(()) {
            assert!(error.is_closed(), "unexpected gate send failure: {error}");
        }
        drop(gate_tx);
        release_dropped_entities(cx);
        drop(keep_agent_alive);
        cx.run_until_parked();

        assert!(
            !connection_retained && !thread_retained,
            "abandoned load retained objects: connection={connection_retained}, thread={thread_retained}"
        );
    }

    #[gpui::test]
    async fn test_retry_after_failed_load_discards_old_replay(cx: &mut gpui::TestAppContext) {
        let (connection, project, load_count, close_count, _, _, _keep_agent_alive) =
            connect_fake_agent(None, cx).await;
        let session_id = acp_v2::SessionId::new("session-failed-load");
        let work_dirs = PathList::new(&[Path::new("/a")]);
        let (response_tx, response_rx) = futures::channel::oneshot::channel::<()>();
        let first_load = cx.update(|cx| {
            connection.clone().open_or_create_session(
                session_id.clone(),
                project.clone(),
                work_dirs.clone(),
                None,
                {
                    let dispatch_tx = connection.dispatch_tx.clone();
                    move |_, session_id, _| {
                        async move {
                            response_rx.await?;
                            enqueue_notification(
                                &dispatch_tx,
                                acp::SessionNotification::new(
                                    session_id.0,
                                    acp::SessionUpdate::AgentMessageChunk(acp::ContentChunk::new(
                                        "stale replay".into(),
                                    )),
                                ),
                                handle_session_notification,
                            );
                            Err(anyhow!("load failed"))
                        }
                        .boxed_local()
                    }
                },
                cx,
            )
        });
        cx.run_until_parked();

        let old_thread = connection
            .sessions
            .borrow()
            .get(&session_id)
            .expect("first load should register a thread")
            .thread
            .clone();
        drop(first_load);
        release_dropped_entities(cx);
        assert!(old_thread.upgrade().is_none());

        let replacement_load = cx.update(|cx| {
            connection
                .clone()
                .load_session(session_id.clone(), project, work_dirs, None, cx)
        });
        cx.run_until_parked();
        assert_eq!(load_count.load(Ordering::SeqCst), 0);
        response_tx
            .send(())
            .expect("failed load should still be waiting");

        let replacement = replacement_load
            .await
            .expect("replacement load should succeed");
        cx.run_until_parked();
        assert_ne!(replacement.entity_id(), old_thread.entity_id());
        assert!(replacement.read_with(cx, |thread, _| thread.entries().is_empty()));
        assert_eq!(load_count.load(Ordering::SeqCst), 1);
        assert_eq!(close_count.load(Ordering::SeqCst), 0);
        assert!(
            !connection
                .pending_sessions
                .borrow()
                .contains_key(&session_id)
        );
        assert_eq!(
            connection
                .sessions
                .borrow()
                .get(&session_id)
                .expect("replacement session")
                .thread
                .entity_id(),
            replacement.entity_id()
        );
    }

    #[gpui::test]
    async fn test_dropping_one_loader_during_load_preserves_other_concurrent_loader(
        cx: &mut gpui::TestAppContext,
    ) {
        let (
            connection,
            project,
            load_count,
            close_count,
            _load_session_updates,
            load_session_gate,
            _keep_agent_alive,
        ) = connect_fake_agent(None, cx).await;

        let (gate_tx, gate_rx) = async_channel::bounded::<()>(1);
        *load_session_gate
            .lock()
            .expect("load_session_gate mutex poisoned") = Some(gate_rx);

        let session_id = acp_v2::SessionId::new("session-concurrent-close");
        let work_dirs = util::path_list::PathList::new(&[std::path::Path::new("/a")]);

        // Kick off two concurrent loads; the second must join the first's pending
        // task rather than issuing a second RPC.
        let first_load = cx.update(|cx| {
            connection.clone().load_session(
                session_id.clone(),
                project.clone(),
                work_dirs.clone(),
                None,
                cx,
            )
        });
        let second_load = cx.update(|cx| {
            connection.clone().load_session(
                session_id.clone(),
                project.clone(),
                work_dirs.clone(),
                None,
                cx,
            )
        });

        cx.run_until_parked();
        assert_eq!(
            load_count.load(Ordering::SeqCst),
            1,
            "load_session RPC should only be dispatched once for concurrent loads"
        );

        drop(first_load);
        cx.run_until_parked();
        assert_eq!(
            close_count.load(Ordering::SeqCst),
            0,
            "close_session RPC must not be dispatched while another loader remains"
        );

        // Release the gate so the load RPC can finally respond.
        gate_tx.send(()).await.expect("gate send failed");
        drop(gate_tx);

        let second_thread = second_load.await.expect("second load should still succeed");
        cx.run_until_parked();

        assert_eq!(
            close_count.load(Ordering::SeqCst),
            0,
            "close_session RPC must not be dispatched while a thread handle is alive"
        );
        assert!(
            connection.sessions.borrow().contains_key(&session_id),
            "session must remain tracked while a thread handle is alive"
        );
        assert!(
            !connection
                .pending_sessions
                .borrow()
                .contains_key(&session_id),
            "pending_sessions entry should be cleared once the load resolves"
        );

        drop(second_thread);
        release_dropped_entities(cx);
        assert_eq!(
            close_count.load(Ordering::SeqCst),
            1,
            "close_session RPC should fire exactly once when the last handle is released"
        );
        assert!(
            !connection.sessions.borrow().contains_key(&session_id),
            "session should be removed after the last handle is released"
        );
    }

    #[gpui::test]
    async fn test_load_initializes_config_before_draining_queued_updates(
        cx: &mut gpui::TestAppContext,
    ) {
        let (connection, project, _, _, _, _, _keep_agent_alive) =
            connect_fake_agent(None, cx).await;
        let session_id = acp_v2::SessionId::new("session-config-update");
        let updated_options = vec![acp::SessionConfigOption::boolean(
            "web_search",
            "Web Search",
            true,
        )];
        let thread = cx
            .update(|cx| {
                connection.clone().open_or_create_session(
                    session_id.clone(),
                    project,
                    PathList::new(&[std::path::Path::new("/a")]),
                    None,
                    {
                        let dispatch_tx = connection.dispatch_tx.clone();
                        let updated_options = updated_options.clone();
                        move |_, session_id, _| {
                            async move {
                                // Later notifications can already be queued when the response is ready.
                                enqueue_notification(
                                    &dispatch_tx,
                                    acp::SessionNotification::new(
                                        session_id.0,
                                        acp::SessionUpdate::ConfigOptionUpdate(
                                            acp::ConfigOptionUpdate::new(updated_options),
                                        ),
                                    ),
                                    handle_session_notification,
                                );
                                Ok(SessionConfigResponse {
                                    modes: None,
                                    config_options: Some(vec![acp::SessionConfigOption::boolean(
                                        "web_search",
                                        "Web Search",
                                        false,
                                    )]),
                                })
                            }
                            .boxed_local()
                        }
                    },
                    cx,
                )
            })
            .await
            .expect("session should load");
        cx.run_until_parked();

        let sessions = connection.sessions.borrow();
        let session = sessions.get(&session_id).expect("loaded session");
        assert_eq!(session.thread.entity_id(), thread.entity_id());
        let options = session.config_options.as_ref().expect("loaded config");
        assert_eq!(
            *options.config_options.borrow(),
            vec![acp_v2::SessionConfigOption::boolean(
                "web_search",
                "Web Search",
                true,
            )]
        );
    }

    #[gpui::test]
    async fn test_abandoned_load_releases_connection_before_rpc_response(
        cx: &mut gpui::TestAppContext,
    ) {
        for poll_load in [false, true] {
            let (
                connection,
                project,
                load_count,
                close_count,
                _load_session_updates,
                load_session_gate,
                keep_agent_alive,
            ) = connect_fake_agent(None, cx).await;

            let (gate_tx, gate_rx) = async_channel::bounded::<()>(1);
            *load_session_gate
                .lock()
                .expect("load_session_gate mutex poisoned") = Some(gate_rx);

            let session_id = acp_v2::SessionId::new("session-abandoned-during-load");
            let work_dirs = util::path_list::PathList::new(&[std::path::Path::new("/a")]);
            let load_task = cx.update(|cx| {
                connection.clone().load_session(
                    session_id.clone(),
                    project.clone(),
                    work_dirs,
                    None,
                    cx,
                )
            });

            if poll_load {
                cx.run_until_parked();
                assert_eq!(load_count.load(Ordering::SeqCst), 1);
            }
            let weak_thread = connection
                .sessions
                .borrow()
                .get(&session_id)
                .expect("in-flight load should register its thread")
                .thread
                .clone();
            let weak_connection = Rc::downgrade(&connection);

            drop(load_task);
            drop(connection);
            release_dropped_entities(cx);
            let connection_retained_before_response = weak_connection.upgrade().is_some();
            let thread_retained_before_response = weak_thread.upgrade().is_some();
            let close_count_before_response = close_count.load(Ordering::SeqCst);

            // Releasing the connection can also tear down the peer's gated handler.
            if let Err(error) = gate_tx.try_send(()) {
                assert!(error.is_closed(), "unexpected gate send failure: {error}");
            }
            drop(gate_tx);
            release_dropped_entities(cx);
            drop(keep_agent_alive);
            cx.run_until_parked();

            assert_eq!(
                close_count_before_response, 0,
                "closing the ACP session must wait for the in-flight load response"
            );
            assert!(
                !connection_retained_before_response && !thread_retained_before_response,
                "abandoned load retained objects while the fake agent held its response: connection={connection_retained_before_response}, thread={thread_retained_before_response}"
            );
        }
    }

    #[gpui::test]
    async fn test_reopening_session_waits_for_close_acknowledgement(cx: &mut gpui::TestAppContext) {
        for abandon_load in [false, true] {
            let (close_gate_tx, close_gate_rx) = async_channel::bounded::<()>(1);
            let (
                connection,
                project,
                load_count,
                close_count,
                load_session_updates,
                load_session_gate,
                _keep_agent_alive,
            ) = connect_fake_agent(Some(close_gate_rx), cx).await;
            let (load_gate_tx, load_gate_rx) = async_channel::bounded::<()>(1);
            *load_session_gate
                .lock()
                .expect("load_session_gate mutex poisoned") = Some(load_gate_rx);
            *load_session_updates
                .lock()
                .expect("load_session_updates mutex poisoned") =
                vec![acp::SessionUpdate::AgentMessageChunk(
                    acp::ContentChunk::new(acp::ContentBlock::Text(acp::TextContent::new(
                        "old".to_string(),
                    ))),
                )];

            let session_id = acp_v2::SessionId::new("session-reopen-abandoned-load");
            let work_dirs = util::path_list::PathList::new(&[std::path::Path::new("/a")]);
            let first_load = cx.update(|cx| {
                connection.clone().load_session(
                    session_id.clone(),
                    project.clone(),
                    work_dirs.clone(),
                    None,
                    cx,
                )
            });
            cx.run_until_parked();
            assert_eq!(load_count.load(Ordering::SeqCst), 1);
            let old_thread = connection
                .sessions
                .borrow()
                .get(&session_id)
                .expect("first load should register its replay recipient")
                .thread
                .clone();
            let old_thread_id = old_thread.entity_id();

            if abandon_load {
                drop(first_load);
            } else {
                load_gate_tx.send(()).await.expect("load gate send failed");
                let loaded_thread = first_load.await.expect("initial load failed");
                drop(loaded_thread);
            }
            release_dropped_entities(cx);
            assert!(
                old_thread.upgrade().is_none(),
                "old owner should release its thread"
            );
            *load_session_updates
                .lock()
                .expect("load_session_updates mutex poisoned") =
                vec![acp::SessionUpdate::AgentMessageChunk(
                    acp::ContentChunk::new(acp::ContentBlock::Text(acp::TextContent::new(
                        "new".to_string(),
                    ))),
                )];
            let replacement_load = cx.update(|cx| {
                connection.clone().load_session(
                    session_id.clone(),
                    project.clone(),
                    work_dirs.clone(),
                    None,
                    cx,
                )
            });
            cx.run_until_parked();
            assert_eq!(load_count.load(Ordering::SeqCst), 1);
            assert_eq!(
                close_count.load(Ordering::SeqCst),
                usize::from(!abandon_load)
            );

            if abandon_load {
                load_gate_tx.send(()).await.expect("load gate send failed");
            }
            drop(load_gate_tx);
            release_dropped_entities(cx);
            assert_eq!(
                close_count.load(Ordering::SeqCst),
                1,
                "old session should close before its replacement starts"
            );
            assert_eq!(
                load_count.load(Ordering::SeqCst),
                1,
                "replacement must wait for the old close acknowledgement"
            );
            assert!(
                connection
                    .sessions
                    .borrow()
                    .get(&session_id)
                    .is_none_or(|session| session.thread.entity_id() == old_thread_id),
                "replacement thread must not be registered before close acknowledgement"
            );

            close_gate_tx
                .send(())
                .await
                .expect("close gate send failed");
            drop(close_gate_tx);
            let timeout = cx
                .background_executor
                .timer(std::time::Duration::from_secs(5))
                .fuse();
            let replacement_load = replacement_load.fuse();
            futures::pin_mut!(replacement_load, timeout);
            let replacement = futures::select! {
                result = replacement_load => result.expect("replacement load failed"),
                _ = timeout => panic!("timed out waiting for replacement load"),
            };
            cx.run_until_parked();
            assert_ne!(replacement.entity_id(), old_thread_id);
            assert_eq!(load_count.load(Ordering::SeqCst), 2);
            assert_eq!(close_count.load(Ordering::SeqCst), 1);
            assert_eq!(
                connection
                    .sessions
                    .borrow()
                    .get(&session_id)
                    .map(|session| session.thread.entity_id()),
                Some(replacement.entity_id()),
                "old cleanup must not unregister the replacement"
            );
            let transcript = replacement.read_with(cx, |thread, cx| thread.to_markdown(cx));
            assert_eq!(transcript, "## Assistant\n\nnew\n\n");

            drop(replacement);
            release_dropped_entities(cx);
            assert_eq!(close_count.load(Ordering::SeqCst), 2);
            assert!(!connection.sessions.borrow().contains_key(&session_id));
            assert!(
                !connection
                    .pending_sessions
                    .borrow()
                    .contains_key(&session_id)
            );
        }
    }

    #[gpui::test]
    async fn test_reopening_after_failed_load_discards_old_replay(cx: &mut gpui::TestAppContext) {
        let (connection, project, load_count, close_count, _, _, _keep_agent_alive) =
            connect_fake_agent(None, cx).await;
        let session_id = acp_v2::SessionId::new("session-failed-load");
        let work_dirs = PathList::new(&[std::path::Path::new("/a")]);
        let (response_tx, response_rx) = futures::channel::oneshot::channel::<()>();
        let first_load = cx.update(|cx| {
            connection.clone().open_or_create_session(
                session_id.clone(),
                project.clone(),
                work_dirs.clone(),
                None,
                {
                    let dispatch_tx = connection.dispatch_tx.clone();
                    move |_, session_id, _| {
                        async move {
                            response_rx.await?;
                            enqueue_notification(
                                &dispatch_tx,
                                acp::SessionNotification::new(
                                    session_id.0,
                                    acp::SessionUpdate::AgentMessageChunk(acp::ContentChunk::new(
                                        "old replay".into(),
                                    )),
                                ),
                                handle_session_notification,
                            );
                            Err(anyhow!("load failed"))
                        }
                        .boxed_local()
                    }
                },
                cx,
            )
        });
        cx.run_until_parked();
        let old_thread = connection
            .sessions
            .borrow()
            .get(&session_id)
            .expect("loading session")
            .thread
            .clone();
        drop(first_load);
        release_dropped_entities(cx);
        assert!(old_thread.upgrade().is_none());

        let replacement_load = cx.update(|cx| {
            connection
                .clone()
                .load_session(session_id.clone(), project, work_dirs, None, cx)
        });
        cx.run_until_parked();
        assert_eq!(load_count.load(Ordering::SeqCst), 0);
        response_tx
            .send(())
            .expect("failed load should still be waiting");

        let replacement = replacement_load.await.expect("replacement should load");
        cx.run_until_parked();
        assert_ne!(replacement.entity_id(), old_thread.entity_id());
        assert!(replacement.read_with(cx, |thread, _| thread.entries().is_empty()));
        assert_eq!(load_count.load(Ordering::SeqCst), 1);
        assert_eq!(close_count.load(Ordering::SeqCst), 0);
        assert!(
            !connection
                .pending_sessions
                .borrow()
                .contains_key(&session_id)
        );
        assert_eq!(
            connection
                .sessions
                .borrow()
                .get(&session_id)
                .expect("replacement session")
                .thread
                .entity_id(),
            replacement.entity_id()
        );
    }

    fn release_dropped_entities(cx: &mut gpui::TestAppContext) {
        cx.run_until_parked();
        cx.update(|_| ());
        cx.run_until_parked();
    }
}

fn mcp_servers_for_project(project: &Entity<Project>, cx: &App) -> Vec<acp::McpServer> {
    mcp_configurations_for_project(project, cx)
        .into_iter()
        .map(|(name, configuration)| match &*configuration {
            project::context_server_store::ContextServerConfiguration::Custom {
                command, ..
            }
            | project::context_server_store::ContextServerConfiguration::Extension {
                command,
                ..
            } => acp::McpServer::Stdio(
                acp::McpServerStdio::new(name, &command.path)
                    .args(command.args.clone())
                    .env(if let Some(env) = command.env.as_ref() {
                        env.iter()
                            .map(|(name, value)| acp::EnvVariable::new(name, value))
                            .collect()
                    } else {
                        vec![]
                    }),
            ),
            project::context_server_store::ContextServerConfiguration::Http {
                url,
                headers,
                timeout: _,
                oauth: _,
            } => acp::McpServer::Http(
                acp::McpServerHttp::new(name, url.to_string()).headers(
                    headers
                        .iter()
                        .map(|(name, value)| acp::HttpHeader::new(name, value))
                        .collect(),
                ),
            ),
        })
        .collect()
}

fn config_state(
    modes: Option<acp::SessionModeState>,
    config_options: Option<Vec<acp::SessionConfigOption>>,
) -> Result<(
    Option<Rc<RefCell<acp::SessionModeState>>>,
    Option<ConfigOptions>,
)> {
    if let Some(opts) = config_options {
        let opts = acp_thread::config_options::from_v1(opts)?;
        let opts = ConfigOptions::new(Rc::new(RefCell::new(opts)));
        return Ok((None, Some(opts)));
    }

    let modes = modes.map(|modes| Rc::new(RefCell::new(modes)));
    Ok((modes, None))
}

struct AcpSessionModes {
    session_id: acp_v2::SessionId,
    connection: ConnectionTo<Agent>,
    state: Rc<RefCell<acp::SessionModeState>>,
}

impl acp_thread::AgentSessionModes for AcpSessionModes {
    fn current_mode(&self) -> acp_v2::SessionConfigValueId {
        acp_v2::SessionConfigValueId::new(self.state.borrow().current_mode_id.0.clone())
    }

    fn all_modes(&self) -> Vec<acp_v2::SessionConfigSelectOption> {
        self.state
            .borrow()
            .available_modes
            .iter()
            .map(mode_from_v1)
            .collect()
    }

    fn set_mode(&self, mode_id: acp_v2::SessionConfigValueId, cx: &mut App) -> Task<Result<()>> {
        let mode_id = acp::SessionModeId::new(mode_id.0);
        let connection = self.connection.clone();
        let session_id = acp::SessionId::new(self.session_id.0.clone());
        let old_mode_id;
        {
            let mut state = self.state.borrow_mut();
            old_mode_id = state.current_mode_id.clone();
            state.current_mode_id = mode_id.clone();
        };
        let state = self.state.clone();
        cx.foreground_executor().spawn(async move {
            let result = connection
                .send_request(acp::SetSessionModeRequest::new(session_id, mode_id))
                .block_task()
                .await;

            if result.is_err() {
                state.borrow_mut().current_mode_id = old_mode_id;
            }

            result?;

            Ok(())
        })
    }
}

struct AcpSessionConfigOptions {
    session_id: acp_v2::SessionId,
    connection: ConnectionTo<Agent>,
    config_options: ConfigOptions,
}

impl acp_thread::AgentSessionConfigOptions for AcpSessionConfigOptions {
    fn config_options(&self) -> Vec<acp_v2::SessionConfigOption> {
        self.config_options.config_options.borrow().clone()
    }

    fn set_config_option(
        &self,
        config_id: acp_v2::SessionConfigId,
        value: acp_v2::SessionConfigOptionValue,
        cx: &mut App,
    ) -> Task<Result<Vec<acp_v2::SessionConfigOption>>> {
        let connection = self.connection.clone();
        let session_id = self.session_id.clone();
        let config_options = self.config_options.clone();

        cx.foreground_executor().spawn(async move {
            config_options
                .set_config_option(&connection, session_id, config_id, value)
                .await
        })
    }

    fn watch(&self, _cx: &mut App) -> Option<watch::Receiver<()>> {
        Some(self.config_options.rx.clone())
    }
}

fn handle_request_permission(
    args: acp::RequestPermissionRequest,
    responder: Responder<acp::RequestPermissionResponse>,
    cx: &mut AsyncApp,
    ctx: &ClientContext,
) {
    let thread = match session_thread(ctx, &acp_v2::SessionId::new(args.session_id.0)) {
        Ok(t) => t,
        Err(e) => return respond_err(responder, e),
    };

    let cancellation = responder.cancellation();
    cx.spawn(async move |cx| {
        let options = match args
            .options
            .into_iter()
            .map(permission_option_from_v1)
            .collect::<Result<Vec<_>, _>>()
        {
            Ok(options) => options,
            Err(error) => return respond_err(responder, error),
        };
        let (request_id, task) = match thread
            .update(cx, |thread, cx| {
                thread.request_tool_call_authorization_with_id(
                    args.tool_call,
                    acp_thread::PermissionOptions::Flat(options),
                    acp_thread::AuthorizationKind::PermissionGrant,
                    cx,
                )
            })
            .flatten_acp()
        {
            Ok(request) => request,
            Err(error) => return respond_err(responder, error),
        };
        let result = cancellation
            .run_until_cancelled(async { Ok(task.await) })
            .await;

        match result {
            Ok(outcome) => {
                responder
                    .respond(acp::RequestPermissionResponse::new(
                        permission_outcome_to_v1(outcome),
                    ))
                    .log_err();
            }
            Err(e) => {
                if e.code == ErrorCode::RequestCancelled {
                    thread
                        .update(cx, |thread, cx| {
                            thread.cancel_permission_request(request_id, cx)
                        })
                        .log_err();
                }
                respond_err(responder, e)
            }
        }
    })
    .detach();
}

fn handle_create_elicitation(
    args: acp::CreateElicitationRequest,
    responder: Responder<acp::CreateElicitationResponse>,
    cx: &mut AsyncApp,
    ctx: &ClientContext,
) {
    let args = match acp_thread::elicitation::request_from_v1(args) {
        Ok(args) => args,
        Err(error) => return respond_err(responder, acp::Error::from(error)),
    };
    let responder = responder.wrap_params(|_, response| {
        response.and_then(|response| {
            acp_thread::elicitation::response_to_v1(response).map_err(acp::Error::from)
        })
    });
    handle_v2_create_elicitation(args, responder, cx, ctx);
}

fn handle_complete_elicitation(
    args: acp::CompleteElicitationNotification,
    cx: &mut AsyncApp,
    ctx: &ClientContext,
) {
    handle_v2_complete_elicitation(
        acp_v2::CompleteElicitationNotification::new(acp_v2::ElicitationId::new(
            args.elicitation_id.0,
        ))
        .meta(args.meta),
        cx,
        ctx,
    );
}

fn handle_write_text_file(
    args: acp::WriteTextFileRequest,
    responder: Responder<acp::WriteTextFileResponse>,
    cx: &mut AsyncApp,
    ctx: &ClientContext,
) {
    if ctx.read_only {
        return respond_err(
            responder,
            acp::Error::invalid_params()
                .data("write_text_file is disabled for this read-only ACP worker"),
        );
    }
    if let Some(policy) = &ctx.terminal_sandbox_wrap
        && let Err(error) = validate_worker_write_path(&args.path, policy)
    {
        return respond_err(
            responder,
            acp::Error::invalid_params().data(error.to_string()),
        );
    }
    let thread = match session_thread(ctx, &acp_v2::SessionId::new(args.session_id.0.clone())) {
        Ok(t) => t,
        Err(e) => return respond_err(responder, e),
    };

    cx.spawn(async move |cx| {
        let result: Result<_, acp::Error> = async {
            thread
                .update(cx, |thread, cx| {
                    thread.write_text_file(args.path, args.content, cx)
                })
                .map_err(acp::Error::from)?
                .await?;
            Ok(())
        }
        .await;

        match result {
            Ok(()) => {
                responder
                    .respond(acp::WriteTextFileResponse::default())
                    .log_err();
            }
            Err(e) => respond_err(responder, e),
        }
    })
    .detach();
}

fn validate_worker_write_path(
    path: &std::path::Path,
    policy: &acp_thread::SandboxWrap,
) -> Result<()> {
    anyhow::ensure!(
        path.is_absolute(),
        "ACP worker writes require an absolute path"
    );
    anyhow::ensure!(
        !path
            .components()
            .any(|component| matches!(component, std::path::Component::ParentDir)),
        "ACP worker write path cannot contain parent traversal"
    );
    let mut ancestor = path;
    let mut missing = Vec::new();
    let path = loop {
        match std::fs::canonicalize(ancestor) {
            Ok(mut resolved) => {
                for component in missing.iter().rev() {
                    resolved.push(component);
                }
                break resolved;
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                // A dangling symlink is not a missing file: following it later
                // could redirect the write outside the approved workspace.
                anyhow::ensure!(
                    std::fs::symlink_metadata(ancestor).is_err(),
                    "ACP worker cannot write through a dangling symlink"
                );
                missing.push(ancestor.file_name().context("invalid worker write path")?);
                ancestor = ancestor
                    .parent()
                    .context("worker write path has no existing root")?;
            }
            Err(error) => return Err(error.into()),
        }
    };
    let writable = policy
        .writable_paths
        .iter()
        .any(|root| std::fs::canonicalize(root).is_ok_and(|root| path.starts_with(root)));
    anyhow::ensure!(
        writable,
        "ACP worker write is outside its managed workspace"
    );
    for protected in &policy.protected_paths {
        let protected = std::fs::canonicalize(protected)?;
        anyhow::ensure!(
            !path.starts_with(protected),
            "ACP worker cannot write protected workspace metadata"
        );
    }
    Ok(())
}

fn handle_read_text_file(
    args: acp::ReadTextFileRequest,
    responder: Responder<acp::ReadTextFileResponse>,
    cx: &mut AsyncApp,
    ctx: &ClientContext,
) {
    let thread = match session_thread(ctx, &acp_v2::SessionId::new(args.session_id.0.clone())) {
        Ok(t) => t,
        Err(e) => return respond_err(responder, e),
    };

    cx.spawn(async move |cx| {
        let cancellation = responder.cancellation();
        let result = cancellation
            .run_until_cancelled(async {
                thread
                    .update(cx, |thread, cx| {
                        thread.read_text_file(args.path, args.line, args.limit, false, cx)
                    })
                    .map_err(acp::Error::from)?
                    .await
            })
            .await;

        respond_result(responder, result.map(acp::ReadTextFileResponse::new));
    })
    .detach();
}

fn handle_claude_sdk_message(
    notification: ClaudeSdkMessageNotification,
    cx: &mut AsyncApp,
    ctx: &ClientContext,
) {
    let thread = ctx
        .sessions
        .borrow()
        .get(&notification.session_id)
        .map(|session| session.thread.clone());
    if let Some(thread) = thread {
        thread
            .update(cx, |thread, cx| {
                thread.observe_claude_completion_message(&notification.message, cx);
            })
            .log_err();
    }
}

fn handle_session_notification_with_subagents(
    notification: SessionNotificationWithSubagents,
    cx: &mut AsyncApp,
    ctx: &ClientContext,
) {
    match notification.into_notification() {
        Ok(notification) => handle_session_notification(notification, cx, ctx),
        Err(error) => log::warn!("Unsupported ACP session update: {error:#}"),
    }
}

fn handle_session_notification(
    notification: acp::SessionNotification,
    cx: &mut AsyncApp,
    ctx: &ClientContext,
) {
    // Extract everything we need from the session while briefly borrowing.
    let (thread, session_modes, config_opts_data) = {
        let sessions = ctx.sessions.borrow();
        let Some(session) =
            sessions.get(&acp_v2::SessionId::new(notification.session_id.0.clone()))
        else {
            drop(sessions);
            if ctx
                .session_notification_state
                .should_buffer(&acp_v2::SessionId::new(notification.session_id.0.clone()))
            {
                log::debug!(
                    "Buffering session update while a new session is registering: {:?}",
                    notification.session_id
                );
                ctx.session_notification_state.buffer(notification);
            } else {
                log::debug!(
                    "Dropping session update for inactive session: {:?}",
                    notification.session_id
                );
            }
            return;
        };
        (
            session.thread.clone(),
            session.session_modes.clone(),
            session
                .config_options
                .as_ref()
                .map(|opts| (opts.config_options.clone(), opts.tx.clone())),
        )
    };

    if let acp::SessionUpdate::SessionInfoUpdate(info) = &notification.update
        && let Some(event) = info
            .meta
            .as_ref()
            .and_then(|meta| meta.get(EXTERNAL_SUBAGENT_UPDATE_META_KEY))
    {
        let result = (|| -> Result<()> {
            let event: ExternalSubagentUpdate = serde_json::from_value(event.clone())?;
            match event {
                ExternalSubagentUpdate::SubagentUpdate { session_id, patch } => {
                    cx.update(|cx| -> Result<()> {
                        if let Some(existing) = ctx.sessions.borrow().get(&session_id) {
                            let owned = existing.thread.read_with(cx, |child, _| {
                                child.parent_session_id()
                                    == Some(&acp_v2::SessionId::new(
                                        notification.session_id.0.clone(),
                                    ))
                            })?;
                            anyhow::ensure!(
                                owned,
                                "Subagent {session_id} already belongs to another parent"
                            );
                        }
                        let child = thread.update(cx, |thread, cx| {
                            thread.upsert_external_subagent(session_id.clone(), patch, cx)
                        })??;
                        let connection = child
                            .read(cx)
                            .connection()
                            .clone()
                            .into_any()
                            .downcast::<AcpConnection>()
                            .map_err(|_| anyhow!("External subagent requires an ACP connection"))?;
                        connection
                            .session_notification_state
                            .register_session(&session_id);
                        if !ctx.sessions.borrow().contains_key(&session_id) {
                            connection.register_session(
                                session_id.clone(),
                                &child,
                                None,
                                None,
                                None,
                                cx,
                            );
                        }
                        Ok(())
                    })?;
                    if let Some(buffered) = ctx.session_notification_state.take_pending(&session_id)
                    {
                        for update in buffered {
                            handle_session_notification(update, cx, ctx);
                        }
                    }
                }
                ExternalSubagentUpdate::SessionMessage { message } => {
                    thread.update(cx, |thread, cx| {
                        thread.upsert_external_session_message(message, false, cx)
                    })??;
                }
                ExternalSubagentUpdate::SessionMessageChunk {
                    message_id,
                    sender_session_id,
                    recipient_session_id,
                    content,
                } => {
                    thread.update(cx, |thread, cx| {
                        thread.upsert_external_session_message(
                            acp_thread::ExternalSessionMessage {
                                message_id,
                                sender_session_id,
                                recipient_session_id,
                                content: Some(Some(vec![acp_thread::content::from_v1(content)?])),
                                meta: None,
                            },
                            true,
                            cx,
                        )
                    })??;
                }
                ExternalSubagentUpdate::SubagentSpawned {
                    subagent_session_id,
                    name,
                    task,
                    prompt,
                } => {
                    cx.update(|cx| -> Result<()> {
                        let child = thread.update(cx, |thread, cx| {
                            thread.spawn_external_subagent(
                                subagent_session_id.clone(),
                                name,
                                task,
                                prompt,
                                cx,
                            )
                        })??;
                        let connection = child
                            .read(cx)
                            .connection()
                            .clone()
                            .into_any()
                            .downcast::<AcpConnection>()
                            .map_err(|_| anyhow!("External subagent requires an ACP connection"))?;
                        connection
                            .session_notification_state
                            .register_session(&subagent_session_id);
                        connection.register_session(
                            subagent_session_id.clone(),
                            &child,
                            None,
                            None,
                            None,
                            cx,
                        );
                        Ok(())
                    })?;
                    if let Some(buffered) = ctx
                        .session_notification_state
                        .take_pending(&subagent_session_id)
                    {
                        for update in buffered {
                            handle_session_notification(update, cx, ctx);
                        }
                    }
                }
                ExternalSubagentUpdate::SubagentStateUpdate {
                    subagent_session_id,
                    state,
                } => {
                    thread.update(cx, |thread, cx| {
                        thread.update_external_subagent_state(&subagent_session_id, state, cx)
                    })??;
                }
            }
            Ok(())
        })();
        if let Err(error) = result {
            log::error!("Failed to handle external subagent update: {error:#}");
        }
        return;
    }
    // Borrow is dropped here.

    // Apply mode/config/session_list updates without holding the borrow.
    if let acp::SessionUpdate::CurrentModeUpdate(acp::CurrentModeUpdate {
        current_mode_id, ..
    }) = &notification.update
    {
        if let Some(session_modes) = &session_modes {
            session_modes.borrow_mut().current_mode_id = current_mode_id.clone();
        }
    }

    if let acp::SessionUpdate::ConfigOptionUpdate(acp::ConfigOptionUpdate {
        config_options, ..
    }) = &notification.update
    {
        if let Some((config_opts_cell, tx_cell)) = &config_opts_data {
            let Some(config_options) =
                acp_thread::config_options::from_v1(config_options.clone()).log_err()
            else {
                return;
            };
            *config_opts_cell.borrow_mut() = config_options;
            tx_cell.borrow_mut().send(()).ok();
        }
    }

    if let acp::SessionUpdate::SessionInfoUpdate(info_update) = &notification.update
        && let Some(session_list) = ctx.session_list.borrow().as_ref()
    {
        session_list.send_info_update(
            acp_v2::SessionId::new(notification.session_id.0.clone()),
            info_update.clone(),
        );
    }

    // A few ACP adapters attach terminal metadata to the first update rather than the
    // initial tool call, so accept it at either point in the tool-call lifecycle.
    let terminal_metadata = match &notification.update {
        acp::SessionUpdate::ToolCall(tool_call) => tool_call
            .meta
            .as_ref()
            .map(|meta| (meta, tool_call.title.clone())),
        acp::SessionUpdate::ToolCallUpdate(tool_call_update) => {
            tool_call_update.meta.as_ref().map(|meta| {
                (
                    meta,
                    tool_call_update
                        .fields
                        .title
                        .clone()
                        .unwrap_or_else(|| "Terminal".to_owned()),
                )
            })
        }
        _ => None,
    };
    if let Some((meta, label)) = terminal_metadata
        && let Some(terminal_info) = meta.get("terminal_info")
        && let Some(id_str) = terminal_info
            .get("terminal_id")
            .and_then(|value| value.as_str())
    {
        let terminal_id = acp_v2::TerminalId::new(id_str);
        let cwd = terminal_info
            .get("cwd")
            .and_then(|value| value.as_str().map(PathBuf::from));

        thread
            .update(cx, |thread, cx| {
                if thread.terminal(terminal_id.clone()).is_ok() {
                    return;
                }
                let builder = TerminalBuilder::new_display_only(
                    CursorShape::default(),
                    AlternateScroll::On,
                    None,
                    0,
                    cx.background_executor(),
                    thread.project().read(cx).path_style(cx),
                );
                let lower = cx.new(|cx| builder.subscribe(cx));
                thread.on_terminal_provider_event(
                    TerminalProviderEvent::Created {
                        terminal_id,
                        label,
                        cwd,
                        output_byte_limit: None,
                        terminal: lower,
                    },
                    cx,
                );
            })
            .log_err();
    }

    // Forward the update to the acp_thread as usual.
    if let Err(err) = thread
        .update(cx, |thread, cx| {
            thread.handle_session_update(notification.update.clone(), cx)
        })
        .flatten_acp()
    {
        log::error!(
            "Failed to handle session update for {:?}: {err:?}",
            notification.session_id
        );
    }

    // Post-handle: stream terminal output/exit if present on ToolCallUpdate meta.
    if let acp::SessionUpdate::ToolCallUpdate(tcu) = &notification.update {
        if let Some(meta) = &tcu.meta {
            if let Some(term_out) = meta.get("terminal_output") {
                if let Some(id_str) = term_out.get("terminal_id").and_then(|v| v.as_str()) {
                    let terminal_id = acp_v2::TerminalId::new(id_str);
                    if let Some(s) = term_out.get("data").and_then(|v| v.as_str()) {
                        let data = s.as_bytes().to_vec();
                        thread
                            .update(cx, |thread, cx| {
                                thread.on_terminal_provider_event(
                                    TerminalProviderEvent::Output { terminal_id, data },
                                    cx,
                                );
                            })
                            .log_err();
                    }
                }
            }

            if let Some(term_exit) = meta.get("terminal_exit") {
                if let Some(id_str) = term_exit.get("terminal_id").and_then(|v| v.as_str()) {
                    let terminal_id = acp_v2::TerminalId::new(id_str);
                    let status = acp::TerminalExitStatus::new()
                        .exit_code(
                            term_exit
                                .get("exit_code")
                                .and_then(|v| v.as_u64())
                                .map(|i| i as u32),
                        )
                        .signal(
                            term_exit
                                .get("signal")
                                .and_then(|v| v.as_str().map(|s| s.to_string())),
                        );

                    thread
                        .update(cx, |thread, cx| {
                            thread.on_terminal_provider_event(
                                TerminalProviderEvent::Exit {
                                    terminal_id,
                                    status,
                                },
                                cx,
                            );
                        })
                        .log_err();
                }
            }
        }
    }
}

fn handle_create_terminal(
    args: acp::CreateTerminalRequest,
    responder: Responder<acp::CreateTerminalResponse>,
    cx: &mut AsyncApp,
    ctx: &ClientContext,
) {
    let thread = match session_thread(ctx, &acp_v2::SessionId::new(args.session_id.0.clone())) {
        Ok(t) => t,
        Err(e) => return respond_err(responder, e),
    };
    let terminal_sandbox_wrap = ctx.terminal_sandbox_wrap.clone();

    cx.spawn(async move |cx| {
        let result: Result<_, acp::Error> = async {
            let terminal_task = thread
                .update(cx, |thread, cx| {
                    thread.create_terminal(
                        args.command,
                        args.args,
                        args.env
                            .into_iter()
                            .map(|variable| (variable.name, variable.value))
                            .collect(),
                        args.cwd,
                        args.output_byte_limit,
                        terminal_sandbox_wrap,
                        cx,
                    )
                })
                .map_err(acp::Error::from)?;
            let terminal_entity = terminal_task.await?;
            let terminal_id = terminal_entity.read_with(cx, |terminal, _| terminal.id().clone());
            Ok(terminal_id)
        }
        .await;

        match result {
            Ok(terminal_id) => {
                responder
                    .respond(acp::CreateTerminalResponse::new(acp::TerminalId::new(
                        terminal_id.0,
                    )))
                    .log_err();
            }
            Err(e) => respond_err(responder, e),
        }
    })
    .detach();
}

fn handle_kill_terminal(
    args: acp::KillTerminalRequest,
    responder: Responder<acp::KillTerminalResponse>,
    cx: &mut AsyncApp,
    ctx: &ClientContext,
) {
    let thread = match session_thread(ctx, &acp_v2::SessionId::new(args.session_id.0)) {
        Ok(t) => t,
        Err(e) => return respond_err(responder, e),
    };

    match thread
        .update(cx, |thread, cx| {
            thread.kill_terminal(acp_v2::TerminalId::new(args.terminal_id.0), cx)
        })
        .flatten_acp()
    {
        Ok(()) => {
            responder
                .respond(acp::KillTerminalResponse::default())
                .log_err();
        }
        Err(e) => respond_err(responder, e),
    }
}

fn handle_release_terminal(
    args: acp::ReleaseTerminalRequest,
    responder: Responder<acp::ReleaseTerminalResponse>,
    cx: &mut AsyncApp,
    ctx: &ClientContext,
) {
    let thread = match session_thread(ctx, &acp_v2::SessionId::new(args.session_id.0)) {
        Ok(t) => t,
        Err(e) => return respond_err(responder, e),
    };

    match thread
        .update(cx, |thread, cx| {
            thread.release_terminal(acp_v2::TerminalId::new(args.terminal_id.0), cx)
        })
        .flatten_acp()
    {
        Ok(()) => {
            responder
                .respond(acp::ReleaseTerminalResponse::default())
                .log_err();
        }
        Err(e) => respond_err(responder, e),
    }
}

fn handle_terminal_output(
    args: acp::TerminalOutputRequest,
    responder: Responder<acp::TerminalOutputResponse>,
    cx: &mut AsyncApp,
    ctx: &ClientContext,
) {
    let thread = match session_thread(ctx, &acp_v2::SessionId::new(args.session_id.0)) {
        Ok(t) => t,
        Err(e) => return respond_err(responder, e),
    };

    match thread
        .read_with(cx, |thread, cx| -> anyhow::Result<_> {
            let out = thread
                .terminal(acp_v2::TerminalId::new(args.terminal_id.0))?
                .read(cx)
                .current_output(cx);
            Ok(out)
        })
        .flatten_acp()
    {
        Ok(output) => {
            responder.respond(process_output_to_v1(output)).log_err();
        }
        Err(e) => respond_err(responder, e),
    }
}

fn handle_wait_for_terminal_exit(
    args: acp::WaitForTerminalExitRequest,
    responder: Responder<acp::WaitForTerminalExitResponse>,
    cx: &mut AsyncApp,
    ctx: &ClientContext,
) {
    let thread = match session_thread(ctx, &acp_v2::SessionId::new(args.session_id.0)) {
        Ok(t) => t,
        Err(e) => return respond_err(responder, e),
    };

    cx.spawn(async move |cx| {
        let cancellation = responder.cancellation();
        let result = cancellation
            .run_until_cancelled(async {
                let exit_status = thread
                    .update(cx, |thread, cx| {
                        thread
                            .terminal(acp_v2::TerminalId::new(args.terminal_id.0))?
                            .read(cx)
                            .wait_for_exit()
                    })
                    .flatten_acp()?
                    .await;
                Ok(exit_status)
            })
            .await;

        respond_result(
            responder,
            result.map(|status| {
                acp::WaitForTerminalExitResponse::new(terminal_exit_status_to_v1(status))
            }),
        );
    })
    .detach();
}

fn mode_from_v1(mode: &acp::SessionMode) -> acp_v2::SessionConfigSelectOption {
    acp_v2::SessionConfigSelectOption::new(
        acp_v2::SessionConfigValueId::new(mode.id.0.clone()),
        mode.name.clone(),
    )
    .description(mode.description.clone())
    .meta(mode.meta.clone())
}

fn permission_option_from_v1(
    option: acp::PermissionOption,
) -> Result<acp_v2::PermissionOption, acp::Error> {
    let kind = match option.kind {
        acp::PermissionOptionKind::AllowOnce => acp_v2::PermissionOptionKind::AllowOnce,
        acp::PermissionOptionKind::AllowAlways => acp_v2::PermissionOptionKind::AllowAlways,
        acp::PermissionOptionKind::RejectOnce => acp_v2::PermissionOptionKind::RejectOnce,
        acp::PermissionOptionKind::RejectAlways => acp_v2::PermissionOptionKind::RejectAlways,
        _ => {
            return Err(acp::Error::invalid_params().data("unsupported permission option kind"));
        }
    };
    Ok(acp_v2::PermissionOption::new(
        acp_v2::PermissionOptionId::new(option.option_id.0),
        option.name,
        kind,
    )
    .meta(option.meta))
}

fn permission_outcome_to_v1(
    outcome: acp_thread::RequestPermissionOutcome,
) -> acp::RequestPermissionOutcome {
    match outcome {
        acp_thread::RequestPermissionOutcome::Cancelled
        | acp_thread::RequestPermissionOutcome::InterruptedByFollowUp => {
            acp::RequestPermissionOutcome::Cancelled
        }
        acp_thread::RequestPermissionOutcome::Selected(outcome) => {
            acp::RequestPermissionOutcome::Selected(acp::SelectedPermissionOutcome::new(
                acp::PermissionOptionId::new(outcome.option_id.0),
            ))
        }
    }
}

fn process_output_to_v1(output: acp_thread::ProcessOutput) -> acp::TerminalOutputResponse {
    acp::TerminalOutputResponse::new(output.output, output.truncated)
        .exit_status(output.exit_status.map(terminal_exit_status_to_v1))
}

fn terminal_exit_status_to_v1(status: acp_v2::TerminalExitStatus) -> acp::TerminalExitStatus {
    acp::TerminalExitStatus::new()
        .exit_code(status.exit_code)
        .signal(status.signal)
        .meta(status.meta)
}
