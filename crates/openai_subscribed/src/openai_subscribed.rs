use anyhow::{Context as _, Result, anyhow};
use base64::Engine as _;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use credentials_provider::CredentialsProvider;
use futures::{
    FutureExt, StreamExt,
    future::{AbortHandle, Abortable, BoxFuture, Shared},
};
use gpui::{App, AsyncApp, BackgroundExecutor, Context, Entity, SharedString, Task, WeakEntity};
use http_client::{
    AsyncBody, CustomHeaders, HttpClient, Method, Request as HttpRequest, RequestBuilderExt,
    http::{HeaderName, HeaderValue},
};
use language_model::{
    CompactionResult, LanguageModel, LanguageModelCompletionError, LanguageModelCompletionEvent,
    LanguageModelEffortLevel, LanguageModelId, LanguageModelName, LanguageModelProviderId,
    LanguageModelProviderName, LanguageModelRequest, LanguageModelToolChoice,
    ProviderErrorCategory, RateLimiter,
};
use open_ai::{
    ReasoningEffort,
    responses::{ResponseInputItem, stream_response, stream_response_with_body},
};
use parking_lot::Mutex;
use rand::RngCore as _;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::sync::{
    Arc,
    atomic::{AtomicU64, AtomicUsize, Ordering},
};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use url::form_urlencoded;
use util::ResultExt as _;

use open_ai::completion::{OpenAiResponseEventMapper, into_open_ai_response_with_account_scope};

pub const PROVIDER_ID: LanguageModelProviderId = LanguageModelProviderId::new("openai-subscribed");
pub const PROVIDER_NAME: LanguageModelProviderName =
    LanguageModelProviderName::new("ChatGPT Subscription");

const CODEX_BASE_URL: &str = "https://chatgpt.com/backend-api/codex";
const CHATGPT_BACKEND_BASE_URL: &str = "https://chatgpt.com/backend-api";
const OPENAI_TOKEN_URL: &str = "https://auth.openai.com/oauth/token";
const OPENAI_AUTHORIZE_URL: &str = "https://auth.openai.com/oauth/authorize";
const CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";

const LEGACY_CREDENTIALS_KEY: &str = "https://chatgpt.com/backend-api/codex";
const ACCOUNT_MANIFEST_KEY: &str = "https://chatgpt.com/backend-api/codex/accounts";
const ACCOUNT_CREDENTIALS_PREFIX: &str = "https://chatgpt.com/backend-api/codex/account/";
const MAX_ACCOUNT_SESSIONS: usize = 5;
const QUOTA_REFRESH_INTERVAL: Duration = Duration::from_secs(60);
const QUOTA_STALE_AFTER: Duration = Duration::from_secs(15 * 60);
const TOKEN_REFRESH_BUFFER_MS: u64 = 5 * 60 * 1000;
/// Requests the complete account catalog without Codex CLI version filtering.
///
/// The backend treats this exact version as an ungated sentinel. Other versions
/// are compared with each model's `minimal_client_version`.
const UNGATED_MODEL_CATALOG_CLIENT_VERSION: &str = "0.0.0";
const RESPONSE_STREAM_IDLE_TIMEOUT: Duration = Duration::from_secs(5 * 60);
const COMPACTION_ATTEMPT_TIMEOUT: Duration = Duration::from_secs(100);
const COMPACTION_OPERATION_TIMEOUT: Duration = Duration::from_secs(120);
const COMPACTION_MAX_ATTEMPTS: usize = 3;
const COMPACTION_RETRY_DELAYS: [Duration; COMPACTION_MAX_ATTEMPTS - 1] =
    [Duration::from_millis(500), Duration::from_secs(1)];
const MODEL_CATALOG_READ_TIMEOUT: Duration = Duration::from_secs(30);
const OAUTH_FLOW_TIMEOUT: Duration = Duration::from_secs(5 * 60);

#[derive(Serialize, Deserialize, Clone, Debug)]
struct CodexCredentials {
    access_token: String,
    refresh_token: String,
    expires_at_ms: u64,
    #[serde(default)]
    account_id: Option<String>,
    #[serde(default)]
    email: Option<String>,
    #[serde(default)]
    user_id: Option<String>,
    #[serde(default)]
    plan_type: Option<String>,
}

impl CodexCredentials {
    fn is_expired(&self) -> bool {
        let now = now_ms();
        now + TOKEN_REFRESH_BUFFER_MS >= self.expires_at_ms
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
struct AccountManifest {
    #[serde(default = "account_manifest_version")]
    version: u32,
    #[serde(default)]
    active_session_id: Option<String>,
    #[serde(default)]
    sessions: Vec<AccountSessionMetadata>,
}

fn account_manifest_version() -> u32 {
    1
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct AccountSessionMetadata {
    session_id: String,
    email: Option<String>,
    user_id: Option<String>,
    display_name: Option<String>,
    image_url: Option<String>,
    last_used_at_ms: u64,
    #[serde(default)]
    token_expires_at_ms: Option<u64>,
    #[serde(default)]
    selected_workspace_account_id: Option<String>,
    #[serde(default)]
    workspaces: Vec<WorkspaceMetadata>,
    quota: Option<QuotaSnapshot>,
    quota_fetched_at_ms: Option<u64>,
    #[serde(default)]
    reauthentication_required: bool,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct WorkspaceMetadata {
    account_id: String,
    name: Option<String>,
    image_url: Option<String>,
    kind: Option<String>,
    plan_type: Option<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct QuotaWindow {
    pub used_percent: f64,
    pub window_minutes: Option<i64>,
    pub resets_at: Option<i64>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct QuotaSnapshot {
    pub primary: Option<QuotaWindow>,
    pub secondary: Option<QuotaWindow>,
    pub credits_has_credits: bool,
    pub credits_unlimited: bool,
    pub credits_balance: Option<String>,
    pub limit_name: Option<String>,
    pub plan_type: Option<String>,
    pub captured_at_ms: u64,
}

#[derive(Clone, Debug)]
pub struct AccountSummary {
    pub session_id: SharedString,
    pub email: Option<SharedString>,
    pub display_name: Option<SharedString>,
    pub workspace_name: Option<SharedString>,
    pub workspace_kind: Option<SharedString>,
    pub plan_type: Option<SharedString>,
    pub token_expires_at_ms: Option<u64>,
    pub quota: Option<QuotaSnapshot>,
    pub quota_stale: bool,
    pub reauthentication_required: bool,
    pub is_active: bool,
    pub is_busy: bool,
}

/// Whether the persisted account manifest was read successfully during the
/// initial load. While `Pending` or `Unavailable`, background code must not
/// overwrite the manifest on disk: a transient keychain failure or a corrupt
/// manifest must not silently drop known accounts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ManifestLoadState {
    Pending,
    Loaded,
    Unavailable,
}

/// Tracks the number of in-flight model operations and pokes the `State`
/// entity when the count transitions between zero and non-zero so busy UI can
/// re-render without the guard touching GPUI from `Drop`.
struct BusyNotifier {
    counter: AtomicUsize,
    notify_tx: async_channel::Sender<()>,
    receiver: Mutex<Option<async_channel::Receiver<()>>>,
}
impl BusyNotifier {
    fn new() -> Self {
        let (notify_tx, receiver) = async_channel::unbounded();
        Self {
            counter: AtomicUsize::new(0),
            notify_tx,
            receiver: Mutex::new(Some(receiver)),
        }
    }

    /// Registers an in-flight operation. The first transition out of idle also
    /// starts the watcher task that forwards busy notifications to the state.
    fn enter(&self, state: &WeakEntity<State>, cx: &AsyncApp) {
        if self.counter.fetch_add(1, Ordering::Relaxed) == 0 {
            if let Some(receiver) = self.receiver.lock().take() {
                let state = state.clone();
                cx.spawn(async move |cx| {
                    while receiver.recv().await.is_ok() {
                        state.update(cx, |_, cx| cx.notify()).ok();
                    }
                })
                .detach();
            }
            let _ = self.notify_tx.try_send(());
        }
    }

    /// Unregisters an in-flight operation. Only called from `Drop`, so it
    /// never touches GPUI; the watcher task performs the notify instead.
    fn exit(&self) {
        if self.counter.fetch_sub(1, Ordering::Relaxed) == 1 {
            let _ = self.notify_tx.try_send(());
        }
    }

    fn is_busy(&self) -> bool {
        self.counter.load(Ordering::Relaxed) > 0
    }
}

pub struct State {
    manifest: AccountManifest,
    credentials: Option<CodexCredentials>,
    active_session_id: Option<String>,
    sign_in_task: Option<Task<()>>,
    sign_in_abort_handle: Option<AbortHandle>,
    sign_out_task: Option<Task<()>>,
    account_mutation_in_progress: bool,
    refresh_task: Option<Shared<Task<Result<CodexCredentials, Arc<anyhow::Error>>>>>,
    load_task: Option<Shared<Task<Result<(), Arc<anyhow::Error>>>>>,
    credentials_provider: Arc<dyn CredentialsProvider>,
    http_client: Arc<dyn HttpClient>,
    auth_generation: u64,
    /// Monotonic sequence of accepted account mutations. A switch whose
    /// captured sequence is no longer current discards its result, so the
    /// account clicked last is the one that stays active.
    account_mutation_seq: u64,
    last_auth_error: Option<SharedString>,
    models: Vec<ChatGptModel>,
    model_fetch_task: Option<Task<()>>,
    /// A single serialized monitor owns both the immediate refresh and the
    /// periodic refreshes. Keeping one task prevents overlapping quota scans
    /// when switching accounts or re-authenticating while a cycle is running.
    quota_refresh_task: Option<Task<()>>,
    active_operations: Arc<BusyNotifier>,
    manifest_load_state: ManifestLoadState,
    client_version: String,
}

#[derive(Debug)]
enum RefreshError {
    Fatal(anyhow::Error),
    Transient(anyhow::Error),
}

impl std::fmt::Display for RefreshError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RefreshError::Fatal(e) => write!(f, "{e}"),
            RefreshError::Transient(e) => write!(f, "{e}"),
        }
    }
}

impl State {
    /// Creates state and starts loading persisted credentials.
    ///
    /// Model discovery requests the ungated account catalog because host
    /// application versions are unrelated to Codex CLI compatibility versions.
    ///
    /// [`State::load_task`] resolves once the load finishes.
    pub fn new(
        http_client: Arc<dyn HttpClient>,
        credentials_provider: Arc<dyn CredentialsProvider>,
        cx: &mut Context<Self>,
    ) -> Self {
        Self::new_with_client_version(
            http_client,
            credentials_provider,
            UNGATED_MODEL_CATALOG_CLIENT_VERSION.to_string(),
            cx,
        )
    }

    fn new_with_client_version(
        http_client: Arc<dyn HttpClient>,
        credentials_provider: Arc<dyn CredentialsProvider>,
        client_version: String,
        cx: &mut Context<Self>,
    ) -> Self {
        let load_task = cx
            .spawn({
                let credentials_provider = credentials_provider.clone();
                async move |this, cx| {
                    let generation = this
                        .read_with(cx, |state, _| state.auth_generation)
                        .unwrap_or(0);

                    enum ManifestLoad {
                        Loaded(AccountManifest),
                        Missing,
                        Unavailable,
                    }
                    let load = match credentials_provider
                        .read_credentials(ACCOUNT_MANIFEST_KEY, cx)
                        .await
                    {
                        Ok(Some((_, bytes))) => match serde_json::from_slice(&bytes) {
                            Ok(manifest) => ManifestLoad::Loaded(manifest),
                            Err(err) => {
                                log::error!(
                                    "ChatGPT subscription account manifest is corrupt: {err}"
                                );
                                ManifestLoad::Unavailable
                            }
                        },
                        Ok(None) => ManifestLoad::Missing,
                        Err(err) => {
                            log::error!(
                                "Failed to read ChatGPT subscription account manifest from the \
                                 credential store: {err:#}"
                            );
                            ManifestLoad::Unavailable
                        }
                    };

                    // A manifest that could not be read must not fall back to the
                    // legacy single-account key: that would resurrect accounts the
                    // user signed out, and a later write could clobber the real
                    // manifest. Only a manifest that is genuinely missing can be
                    // migrated.
                    let migrate_legacy = matches!(&load, ManifestLoad::Missing);
                    let manifest_load_state = match &load {
                        ManifestLoad::Loaded(_) => ManifestLoadState::Loaded,
                        ManifestLoad::Missing => ManifestLoadState::Loaded,
                        ManifestLoad::Unavailable => ManifestLoadState::Unavailable,
                    };
                    let mut manifest = match load {
                        ManifestLoad::Loaded(manifest) => manifest,
                        ManifestLoad::Missing | ManifestLoad::Unavailable => {
                            AccountManifest::default()
                        }
                    };
                    if manifest.version == 0 {
                        manifest.version = 1;
                    }

                    let mut active_session_id = manifest.active_session_id.clone().or_else(|| {
                        manifest
                            .sessions
                            .iter()
                            .max_by_key(|session| session.last_used_at_ms)
                            .map(|session| session.session_id.clone())
                    });
                    manifest.active_session_id = active_session_id.clone();
                    let mut credentials = None;
                    let mut credentials_unavailable = false;
                    if let Some(session_id) = active_session_id.as_deref() {
                        match credentials_provider
                            .read_credentials(&account_credentials_key(session_id), cx)
                            .await
                        {
                            Ok(Some((_, bytes))) => {
                                credentials =
                                    serde_json::from_slice::<CodexCredentials>(&bytes).ok();
                                if credentials.is_none() {
                                    log::error!(
                                        "ChatGPT subscription account credentials are corrupt"
                                    );
                                    credentials_unavailable = true;
                                }
                            }
                            Ok(None) => {}
                            Err(err) => {
                                log::warn!(
                                    "Failed to read ChatGPT subscription account credentials: \
                                     {err:#}"
                                );
                                credentials_unavailable = true;
                            }
                        }
                    }

                    // Migrate the old single-account credential on first load. The
                    // legacy key is only removed once the account-scoped key and
                    // the manifest have both persisted successfully; a failed
                    // persist retries the migration on the next launch.
                    if migrate_legacy
                        && !credentials_unavailable
                        && let Ok(Some((_, bytes))) = credentials_provider
                            .read_credentials(LEGACY_CREDENTIALS_KEY, cx)
                            .await
                    {
                        match serde_json::from_slice::<CodexCredentials>(&bytes) {
                            Ok(creds) => {
                                let session_id = session_id_for_credentials(&creds);
                                active_session_id = Some(session_id.clone());
                                credentials = Some(creds.clone());
                                upsert_manifest_session(&mut manifest, &session_id, &creds);
                                manifest.active_session_id = Some(session_id.clone());
                                let persist = async {
                                    let json = serde_json::to_vec(&creds)?;
                                    credentials_provider
                                        .write_credentials(
                                            &account_credentials_key(&session_id),
                                            "Bearer",
                                            &json,
                                            cx,
                                        )
                                        .await?;
                                    let json = serde_json::to_vec(&manifest)?;
                                    credentials_provider
                                        .write_credentials(ACCOUNT_MANIFEST_KEY, "json", &json, cx)
                                        .await?;
                                    anyhow::Ok(())
                                };
                                match persist.await {
                                    Ok(()) => {
                                        credentials_provider
                                            .delete_credentials(LEGACY_CREDENTIALS_KEY, cx)
                                            .await
                                            .log_err();
                                    }
                                    Err(err) => {
                                        log::error!(
                                            "Failed to migrate legacy ChatGPT subscription \
                                             credentials: {err:#}"
                                        );
                                    }
                                }
                            }
                            Err(err) => {
                                log::warn!(
                                    "Failed to deserialize ChatGPT subscription credentials: \
                                     {err}"
                                );
                            }
                        }
                    }

                    this.update(cx, |state, cx| {
                        // A sign-in/out or switch that happened while loading
                        // owns the identity now; discard the loaded result.
                        state.load_task = None;
                        if state.auth_generation != generation {
                            return;
                        }
                        state.manifest = manifest;
                        state.manifest_load_state = manifest_load_state;
                        state.active_session_id = active_session_id;
                        state.credentials = credentials;
                        if state.credentials.is_some() {
                            state.restart_model_fetch(cx);
                        }
                        state.start_quota_monitor(cx);
                        cx.notify();
                    })?;
                    Ok::<(), Arc<anyhow::Error>>(())
                }
            })
            .shared();

        Self {
            manifest: AccountManifest {
                version: 1,
                ..Default::default()
            },
            credentials: None,
            active_session_id: None,
            sign_in_task: None,
            sign_in_abort_handle: None,
            sign_out_task: None,
            account_mutation_in_progress: false,
            refresh_task: None,
            load_task: Some(load_task),
            credentials_provider,
            http_client,
            auth_generation: 0,
            account_mutation_seq: 0,
            last_auth_error: None,
            models: ChatGptModel::fallback_models(),
            model_fetch_task: None,
            quota_refresh_task: None,
            active_operations: Arc::new(BusyNotifier::new()),
            manifest_load_state: ManifestLoadState::Pending,
            client_version,
        }
    }

    pub fn is_authenticated(&self) -> bool {
        self.credentials.is_some()
    }

    pub fn email(&self) -> Option<&str> {
        self.credentials.as_ref().and_then(|c| c.email.as_deref())
    }

    /// Returns the accounts known to this installation. The active account is
    /// always first so compact pickers can render it without another sort.
    pub fn account_summaries(&self) -> Vec<AccountSummary> {
        let now = now_ms();
        let stale_after_ms = QUOTA_STALE_AFTER.as_millis() as u64;
        let mut accounts = self
            .manifest
            .sessions
            .iter()
            .map(|session| {
                let workspace = session
                    .selected_workspace_account_id
                    .as_deref()
                    .and_then(|id| {
                        session
                            .workspaces
                            .iter()
                            .find(|workspace| workspace.account_id == id)
                    })
                    .or_else(|| session.workspaces.first());
                let plan_type = workspace
                    .and_then(|workspace| workspace.plan_type.clone())
                    .or_else(|| {
                        session
                            .quota
                            .as_ref()
                            .and_then(|quota| quota.plan_type.clone())
                    })
                    .or_else(|| {
                        self.credentials_for_session(&session.session_id)
                            .and_then(|creds| creds.plan_type)
                    });
                AccountSummary {
                    session_id: session.session_id.clone().into(),
                    email: session.email.clone().map(Into::into),
                    display_name: session.display_name.clone().map(Into::into),
                    workspace_name: workspace
                        .and_then(|workspace| workspace.name.clone())
                        .map(Into::into),
                    workspace_kind: workspace
                        .and_then(|workspace| workspace.kind.clone())
                        .map(Into::into),
                    plan_type: plan_type.map(Into::into),
                    token_expires_at_ms: session.token_expires_at_ms,
                    quota: session.quota.clone(),
                    quota_stale: session
                        .quota_fetched_at_ms
                        .is_none_or(|fetched_at| now.saturating_sub(fetched_at) > stale_after_ms),
                    reauthentication_required: session.reauthentication_required,
                    is_active: self.active_session_id.as_deref()
                        == Some(session.session_id.as_str()),
                    is_busy: self.active_operations.is_busy() || self.account_mutation_in_progress,
                }
            })
            .collect::<Vec<_>>();
        accounts.sort_by_key(|account| !account.is_active);
        accounts
    }

    pub fn active_account_id(&self) -> Option<SharedString> {
        self.active_session_id.clone().map(Into::into)
    }

    pub fn is_busy(&self) -> bool {
        self.active_operations.is_busy() || self.account_mutation_in_progress
    }

    fn begin_account_mutation(&mut self) -> Result<()> {
        if self.is_busy() {
            return Err(anyhow!(
                "Cannot change ChatGPT accounts while a request or account operation is active"
            ));
        }
        self.account_mutation_in_progress = true;
        Ok(())
    }

    fn begin_account_switch(&mut self) -> Result<()> {
        self.begin_account_mutation()
    }

    fn finish_account_mutation(&mut self) {
        self.account_mutation_in_progress = false;
    }

    pub fn switch_account(
        &mut self,
        session_id: SharedString,
        cx: &mut Context<Self>,
    ) -> Task<Result<()>> {
        let session_id = session_id.to_string();
        if self.active_session_id.as_deref() == Some(session_id.as_str())
            && self.credentials.is_some()
        {
            return Task::ready(Ok(()));
        }
        if !self
            .manifest
            .sessions
            .iter()
            .any(|session| session.session_id == session_id)
        {
            return Task::ready(Err(anyhow!("Unknown ChatGPT account session")));
        }
        if let Err(error) = self.begin_account_switch() {
            return Task::ready(Err(error));
        }
        cx.notify();

        self.account_mutation_seq = self.account_mutation_seq.wrapping_add(1);
        let mutation_seq = self.account_mutation_seq;
        let provider = self.credentials_provider.clone();
        let key = account_credentials_key(&session_id);
        cx.spawn(async move |this, cx| {
            let operation_result: Result<Option<(AccountManifest, CodexCredentials)>> = async {
                let (_, bytes) = provider
                    .read_credentials(&key, cx)
                    .await?
                    .ok_or_else(|| anyhow!("ChatGPT account credentials are missing"))?;
                let credentials = serde_json::from_slice::<CodexCredentials>(&bytes)
                    .context("Failed to deserialize ChatGPT account credentials")?;
                let manifest = this.read_with(cx, |state, _| {
                    if state.account_mutation_seq != mutation_seq
                        || !state.account_mutation_in_progress
                    {
                        return None;
                    }
                    Some(state.manifest.clone())
                })?;
                let Some(mut manifest) = manifest else {
                    return Ok(None);
                };
                manifest.active_session_id = Some(session_id.clone());
                if let Some(session) = manifest
                    .sessions
                    .iter_mut()
                    .find(|session| session.session_id == session_id)
                {
                    session.last_used_at_ms = now_ms();
                    session.reauthentication_required = false;
                }
                write_manifest(&provider, &manifest, cx).await?;
                Ok(Some((manifest, credentials)))
            }
            .await;

            this.update(cx, |state, cx| {
                let is_current = state.account_mutation_seq == mutation_seq;
                state.finish_account_mutation();
                if is_current && let Ok(Some((manifest, credentials))) = &operation_result {
                    state.auth_generation = state.auth_generation.wrapping_add(1);
                    state.active_session_id = Some(session_id);
                    state.manifest = manifest.clone();
                    state.credentials = Some(credentials.clone());
                    state.refresh_task = None;
                    state.restart_model_fetch(cx);
                    state.last_auth_error = None;
                } else if is_current && operation_result.is_err() {
                    state.last_auth_error =
                        Some("Failed to switch ChatGPT accounts. Please try again.".into());
                }
                cx.notify();
            })?;
            operation_result.map(|_| ())
        })
    }

    /// Starts the all-account monitor. The first scan runs immediately after
    /// the manifest is loaded, then repeats every minute without requiring an
    /// account switch.
    fn start_quota_monitor(&mut self, cx: &mut Context<Self>) {
        if self.quota_refresh_task.is_some()
            || self.manifest_load_state != ManifestLoadState::Loaded
        {
            return;
        }
        self.quota_refresh_task = Some(cx.spawn(async move |this, cx| {
            loop {
                if let Err(error) = refresh_all_account_quotas(&this, cx).await {
                    // A transient keychain, network, or persistence error must not disable the
                    // monitor for the rest of the app lifetime.
                    log::warn!("ChatGPT all-account quota cycle failed: {error:#}");
                }
                cx.background_executor().timer(QUOTA_REFRESH_INTERVAL).await;
            }
        }));
    }

    /// Requests an immediate all-account scan. The monitor remains serialized,
    /// so restarting it cancels any sleeping cycle before beginning a new one.
    pub fn refresh_quota(&mut self, cx: &mut Context<Self>) {
        self.quota_refresh_task.take();
        self.start_quota_monitor(cx);
    }

    fn credentials_for_session(&self, session_id: &str) -> Option<CodexCredentials> {
        if self.active_session_id.as_deref() == Some(session_id) {
            self.credentials.clone()
        } else {
            None
        }
    }

    pub fn is_signing_in(&self) -> bool {
        self.sign_in_task.is_some()
    }

    pub fn last_auth_error(&self) -> Option<SharedString> {
        self.last_auth_error.clone()
    }

    pub fn models(&self) -> Vec<ChatGptModel> {
        self.models.clone()
    }

    /// The in-flight task loading persisted credentials, or `None` once the
    /// initial load has finished.
    pub fn load_task(&self) -> Option<Shared<Task<Result<(), Arc<anyhow::Error>>>>> {
        self.load_task.clone()
    }

    /// Starts the browser-based OAuth sign-in flow. No-op while a sign-in is
    /// already in progress; observe the entity to react to the outcome.
    pub fn sign_in(&mut self, cx: &mut Context<Self>) -> Task<Result<()>> {
        if self.is_signing_in() {
            return Task::ready(Ok(()));
        }
        if let Err(error) = self.begin_account_mutation() {
            return Task::ready(Err(error));
        }

        let http_client = self.http_client.clone();
        let (abort_handle, abort_registration) = AbortHandle::new_pair();
        self.sign_in_abort_handle = Some(abort_handle);
        let task = cx
            .spawn(async move |this, cx| {
                // Let the initial load settle so the new session is persisted
                // against the manifest that is actually on disk.
                if let Some(load_task) =
                    this.read_with(cx, |state, _| state.load_task.clone()).unwrap_or(None)
                {
                    load_task.await.log_err();
                }
                let generation = this
                    .read_with(cx, |state, _| state.auth_generation)
                    .unwrap_or(0);
                let oauth_result = {
                    let oauth_flow =
                        Abortable::new(do_oauth_flow(http_client, cx), abort_registration).fuse();
                    let timeout = cx
                        .background_executor()
                        .timer(OAUTH_FLOW_TIMEOUT)
                        .fuse();
                    futures::pin_mut!(oauth_flow, timeout);
                    futures::select! {
                        result = oauth_flow => result.ok(),
                        _ = timeout => Some(Err(anyhow!(
                            "ChatGPT sign-in timed out after {OAUTH_FLOW_TIMEOUT:?}"
                        ))),
                    }
                };
                let Some(oauth_result) = oauth_result else {
                    return Ok::<(), Arc<anyhow::Error>>(());
                };
                this.update(cx, |state, cx| {
                    state.sign_in_abort_handle = None;
                    cx.notify();
                })
                .log_err();
                match oauth_result {
                Ok(creds) => {
                    let session_id = session_id_for_credentials(&creds);
                    // Discard the result if the user signed out or switched
                    // while the browser flow was open.
                    let generation_current = this
                        .read_with(cx, |state, _| state.auth_generation == generation)
                        .unwrap_or(false);
                    if !generation_current {
                        this.update(cx, |state, cx| {
                            state.sign_in_task = None;
                            state.sign_in_abort_handle = None;
                            state.finish_account_mutation();
                            cx.notify();
                        })
                        .log_err();
                        return Ok(());
                    }
                    let persist_result = async {
                        let credentials_provider =
                            this.read_with(cx, |state, _| state.credentials_provider.clone())?;
                        let json = serde_json::to_vec(&creds)?;
                        let mut manifest = this.read_with(cx, |state, _| state.manifest.clone())?;
                        if manifest
                            .sessions
                            .iter()
                            .all(|session| session.session_id != session_id)
                            && manifest.sessions.len() >= MAX_ACCOUNT_SESSIONS
                        {
                            return Err(anyhow!(
                                "ChatGPT supports at most {MAX_ACCOUNT_SESSIONS} saved accounts"
                            ));
                        }
                        credentials_provider
                            .write_credentials(
                                &account_credentials_key(&session_id),
                                "Bearer",
                                &json,
                                cx,
                            )
                            .await?;
                        upsert_manifest_session(&mut manifest, &session_id, &creds);
                        manifest.active_session_id = Some(session_id.clone());
                        credentials_provider
                            .write_credentials(
                                ACCOUNT_MANIFEST_KEY,
                                "json",
                                &serde_json::to_vec(&manifest)?,
                                cx,
                            )
                            .await?;
                        anyhow::Ok(())
                    }
                    .await;

                    match persist_result {
                        Ok(()) => {
                            this.update(cx, |state, cx| {
                                if state.auth_generation != generation {
                                    state.sign_in_task = None;
                                    return;
                                }
                                upsert_manifest_session(
                                    &mut state.manifest,
                                    &session_id,
                                    &creds,
                                );
                                state.manifest.active_session_id = Some(session_id.clone());
                                state.active_session_id = Some(session_id);
                                state.credentials = Some(creds);
                                state.auth_generation = state.auth_generation.wrapping_add(1);
                                state.account_mutation_seq =
                                    state.account_mutation_seq.wrapping_add(1);
                                state.refresh_task = None;
                                state.manifest_load_state = ManifestLoadState::Loaded;
                                state.sign_in_task = None;
                                state.sign_in_abort_handle = None;
                                state.finish_account_mutation();
                                state.last_auth_error = None;
                                state.refresh_quota(cx);
                                state.restart_model_fetch(cx);
                                cx.notify();
                            })
                            .log_err();
                        }
                        Err(err) => {
                            log::error!(
                                "ChatGPT subscription sign-in failed to persist credentials: {err:?}"
                            );
                            this.update(cx, |state, cx| {
                                state.sign_in_task = None;
                                state.sign_in_abort_handle = None;
                                state.finish_account_mutation();
                                state.last_auth_error =
                                    Some("Failed to save credentials. Please try again.".into());
                                cx.notify();
                            })
                            .log_err();
                        }
                    }
                }
                Err(err) => {
                    log::error!("ChatGPT subscription sign-in failed: {err:?}");
                    this.update(cx, |state, cx| {
                        state.sign_in_task = None;
                        state.sign_in_abort_handle = None;
                        state.finish_account_mutation();
                        state.last_auth_error = Some("Sign-in failed. Please try again.".into());
                        cx.notify();
                    })
                    .log_err();
                }
            }
            Ok::<(), Arc<anyhow::Error>>(())
        })
        .shared();

        self.last_auth_error = None;
        // Keep the task alive after the caller drops their handle and resolve
        // the returned task once the flow has fully completed.
        let sign_in_watch = task.clone();
        self.sign_in_task = Some(cx.spawn(async move |this, cx| {
            sign_in_watch.await.log_err();
            this.update(cx, |_, cx| cx.notify()).ok();
        }));
        cx.notify();
        cx.spawn(async move |_this, _cx| {
            task.await.log_err();
            anyhow::Ok(())
        })
    }

    pub fn cancel_sign_in(&mut self, cx: &mut Context<Self>) {
        let Some(abort_handle) = self.sign_in_abort_handle.take() else {
            return;
        };
        abort_handle.abort();
        self.auth_generation = self.auth_generation.wrapping_add(1);
        self.sign_in_task = None;
        self.finish_account_mutation();
        self.last_auth_error = Some("Sign-in cancelled. You can try again.".into());
        cx.notify();
    }

    pub fn can_cancel_sign_in(&self) -> bool {
        self.sign_in_abort_handle.is_some()
    }

    /// Prevents new model operations immediately, then commits the persisted
    /// account removal before exposing the next account to observers.
    pub fn sign_out(&mut self, cx: &mut Context<Self>) -> Task<Result<()>> {
        if let Err(error) = self.begin_account_mutation() {
            return Task::ready(Err(error));
        }
        // Even without an active session this invalidates any in-flight load
        // or sign-in so a sign-out is never reverted by a pending result.
        self.auth_generation = self.auth_generation.wrapping_add(1);
        let initial_removed_session_id = self.active_session_id.clone();
        self.account_mutation_seq = self.account_mutation_seq.wrapping_add(1);
        let original_manifest = self.manifest.clone();
        let original_active_session_id = self.active_session_id.clone();
        let original_credentials = self.credentials.take();
        let original_models = self.models.clone();
        let original_last_auth_error = self.last_auth_error.take();
        self.refresh_task = None;
        self.model_fetch_task = None;
        self.models = ChatGptModel::fallback_models();
        cx.notify();

        let credentials_provider = self.credentials_provider.clone();
        let generation = self.auth_generation;
        let load_task = self.load_task.clone();
        let task = cx
            .spawn(async move |this, cx| {
                let operation_result: Result<(
                    AccountManifest,
                    Option<String>,
                    Option<CodexCredentials>,
                    Option<String>,
                )> = async {
                    if let Some(load_task) = load_task {
                        load_task.await.map_err(|error| anyhow!("{error}"))?;
                    }
                    let still_current = this
                        .read_with(cx, |state, _| state.auth_generation == generation)
                        .unwrap_or(false);
                    if !still_current {
                        return Err(anyhow!("ChatGPT account changed during sign-out"));
                    }

                    let mut manifest = match credentials_provider
                        .read_credentials(ACCOUNT_MANIFEST_KEY, cx)
                        .await
                        .context("Failed to read the ChatGPT account manifest")?
                    {
                        Some((_, bytes)) => serde_json::from_slice(&bytes)
                            .context("The ChatGPT account manifest is corrupt")?,
                        None => original_manifest.clone(),
                    };
                    let legacy_credentials = match credentials_provider
                        .read_credentials(LEGACY_CREDENTIALS_KEY, cx)
                        .await
                        .context("Failed to inspect the legacy ChatGPT credential")?
                    {
                        Some((_, bytes)) => match serde_json::from_slice::<CodexCredentials>(&bytes)
                        {
                            Ok(credentials) => Some(credentials),
                            Err(error) => {
                                log::warn!(
                                    "Failed to deserialize the legacy ChatGPT credential: {error}"
                                );
                                None
                            }
                        },
                        None => None,
                    };
                    let removed_session_id = initial_removed_session_id
                        .clone()
                        .or_else(|| manifest.active_session_id.clone())
                        .or_else(|| {
                            legacy_credentials
                                .as_ref()
                                .map(session_id_for_credentials)
                        });
                    let Some(removed_session_id) = removed_session_id else {
                        manifest.active_session_id = None;
                        write_manifest(&credentials_provider, &manifest, cx).await?;
                        return Ok((manifest, None, None, None));
                    };
                    manifest
                        .sessions
                        .retain(|session| session.session_id != removed_session_id);
                    if legacy_credentials.as_ref().is_some_and(|credentials| {
                        session_id_for_credentials(credentials) == removed_session_id
                    }) {
                        credentials_provider
                            .delete_credentials(LEGACY_CREDENTIALS_KEY, cx)
                            .await
                            .context("Failed to delete the legacy ChatGPT credential")?;
                    }

                    let mut candidates = manifest.sessions.iter().collect::<Vec<_>>();
                    candidates.sort_by_key(|session| std::cmp::Reverse(session.last_used_at_ms));
                    let mut next_account = None;
                    for session in candidates {
                        let key = account_credentials_key(&session.session_id);
                        match credentials_provider.read_credentials(&key, cx).await {
                            Ok(Some((_, bytes))) => {
                                match serde_json::from_slice::<CodexCredentials>(&bytes) {
                                    Ok(credentials) => {
                                        next_account =
                                            Some((session.session_id.clone(), credentials));
                                        break;
                                    }
                                    Err(error) => log::warn!(
                                        "Failed to deserialize saved ChatGPT account {}: {error}",
                                        session.session_id
                                    ),
                                }
                            }
                            Ok(None) => {}
                            Err(error) => log::warn!(
                                "Failed to load saved ChatGPT account {}: {error:#}",
                                session.session_id
                            ),
                        }
                    }
                    let (next_session_id, next_credentials) = match next_account {
                        Some((session_id, credentials)) => {
                            (Some(session_id), Some(credentials))
                        }
                        None => (None, None),
                    };
                    manifest.active_session_id = next_session_id.clone();
                    write_manifest(&credentials_provider, &manifest, cx)
                        .await
                        .context("Failed to update the ChatGPT account manifest")?;

                    let cleanup_error = credentials_provider
                        .delete_credentials(&account_credentials_key(&removed_session_id), cx)
                        .await
                        .context("Failed to delete ChatGPT subscription credentials")
                        .err()
                        .map(|error| format!("{error:#}"));
                    Ok((
                        manifest,
                        next_session_id,
                        next_credentials,
                        cleanup_error,
                    ))
                }
                .await;

                let update_result = this.update(cx, |state, cx| {
                    state.sign_out_task = None;
                    state.finish_account_mutation();
                    match &operation_result {
                        Ok((manifest, next_session_id, next_credentials, cleanup_error)) => {
                            state.manifest = manifest.clone();
                            state.active_session_id = next_session_id.clone();
                            state.credentials = next_credentials.clone();
                            state.last_auth_error = cleanup_error.as_ref().map(|_| {
                                "Signed out, but failed to remove cached credentials. Please try \
                                 signing out again."
                                    .into()
                            });
                            if state.credentials.is_some() {
                                state.restart_model_fetch(cx);
                            }
                        }
                        Err(error) => {
                            state.manifest = original_manifest;
                            state.active_session_id = original_active_session_id;
                            state.credentials = original_credentials;
                            state.models = original_models;
                            state.last_auth_error = original_last_auth_error;
                            log::error!("ChatGPT sign-out failed: {error:#}");
                        }
                    }
                    cx.notify();
                });
                update_result.map_err(Arc::new)?;
                match operation_result {
                    Ok((_, _, _, Some(cleanup_error))) => Err(Arc::new(anyhow!(cleanup_error))),
                    Ok(_) => Ok(()),
                    Err(error) => Err(Arc::new(error)),
                }
            })
            .shared();

        let sign_out_watch = task.clone();
        self.sign_out_task = Some(cx.spawn(async move |_this, _cx| {
            sign_out_watch.await.log_err();
        }));
        cx.notify();
        cx.spawn(async move |_this, _cx| task.await.map_err(|error| anyhow!("{error}")))
    }

    fn restart_model_fetch(&mut self, cx: &mut Context<Self>) {
        self.model_fetch_task = None;

        let http_client = self.http_client.clone();
        let client_version = self.client_version.clone();
        let auth_generation = self.auth_generation;
        self.model_fetch_task = Some(cx.spawn(async move |this, cx| {
            let result = async {
                let credentials = get_fresh_credentials(&this, &http_client, cx).await?;
                let fetch = fetch_codex_models(
                    http_client.as_ref(),
                    &credentials,
                    &client_version,
                )
                .fuse();
                let timeout = FutureExt::fuse(
                    cx.background_executor().timer(MODEL_CATALOG_READ_TIMEOUT),
                );
                futures::pin_mut!(fetch, timeout);
                let models = futures::select! {
                    result = fetch => result.map_err(LanguageModelCompletionError::Other)?,
                    _ = timeout => return Err(LanguageModelCompletionError::Other(anyhow!(
                        "ChatGPT subscription model catalog timed out after {MODEL_CATALOG_READ_TIMEOUT:?}"
                    ))),
                };
                Ok::<_, LanguageModelCompletionError>(models)
            }
            .await;

            this.update(cx, |state, cx| {
                state.model_fetch_task = None;
                if state.auth_generation != auth_generation || state.credentials.is_none() {
                    return;
                }

                match result {
                    Ok(models) if !models.is_empty() => {
                        state.install_models(models);
                        cx.notify();
                    }
                    Ok(_) => {
                        log::warn!("ChatGPT subscription returned an empty model catalog");
                    }
                    Err(error) => {
                        log::warn!(
                            "Failed to refresh ChatGPT subscription model catalog: {error:#}"
                        );
                    }
                }
            })
            .log_err();
        }));
    }

    fn install_models(&mut self, mut models: Vec<ChatGptModel>) {
        for model in &mut models {
            if let Some(existing) = self
                .models
                .iter()
                .find(|existing| existing.id() == model.id())
            {
                existing
                    .context_window
                    .store(model.max_token_count(), Ordering::Relaxed);
                model.context_window = existing.context_window.clone();
            }
        }
        self.models = models;
    }
}

//
#[derive(Clone, Debug)]
pub struct ChatGptModel {
    id: Arc<str>,
    display_name: Arc<str>,
    context_window: Arc<AtomicU64>,
    default_reasoning_effort: Option<ReasoningEffort>,
    supported_reasoning_efforts: Arc<[ReasoningEffort]>,
    supports_images: bool,
    supports_parallel_tool_calls: bool,
    supports_priority: bool,
}

impl ChatGptModel {
    fn fallback_context_window(model_id: &str) -> u64 {
        if model_id.starts_with("gpt-5.6") {
            299_000
        } else {
            272_000
        }
    }

    pub fn fallback_models() -> Vec<Self> {
        [
            ("gpt-5.6-sol", "GPT-5.6 Sol", ReasoningEffort::Low, true),
            (
                "gpt-5.6-terra",
                "GPT-5.6 Terra",
                ReasoningEffort::Medium,
                true,
            ),
            (
                "gpt-5.6-luna",
                "GPT-5.6 Luna",
                ReasoningEffort::Medium,
                true,
            ),
            ("gpt-5.5", "GPT-5.5", ReasoningEffort::Medium, true),
        ]
        .into_iter()
        .map(
            |(id, display_name, default_reasoning_effort, supports_priority)| Self {
                id: id.into(),
                display_name: display_name.into(),
                context_window: Arc::new(AtomicU64::new(Self::fallback_context_window(id))),
                default_reasoning_effort: Some(default_reasoning_effort),
                supported_reasoning_efforts: if id.starts_with("gpt-5.6") {
                    Arc::from([
                        ReasoningEffort::Low,
                        ReasoningEffort::Medium,
                        ReasoningEffort::High,
                        ReasoningEffort::XHigh,
                        ReasoningEffort::Max,
                    ])
                } else {
                    Arc::from([
                        ReasoningEffort::Low,
                        ReasoningEffort::Medium,
                        ReasoningEffort::High,
                        ReasoningEffort::XHigh,
                    ])
                },
                supports_images: true,
                supports_parallel_tool_calls: true,
                supports_priority,
            },
        )
        .collect()
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn display_name(&self) -> &str {
        &self.display_name
    }

    fn max_token_count(&self) -> u64 {
        self.context_window.load(Ordering::Relaxed)
    }

    fn max_output_tokens(&self) -> Option<u64> {
        None
    }

    fn supports_images(&self) -> bool {
        self.supports_images
    }

    fn default_reasoning_effort(&self) -> Option<ReasoningEffort> {
        self.default_reasoning_effort
    }

    fn supported_reasoning_efforts(&self) -> &[ReasoningEffort] {
        &self.supported_reasoning_efforts
    }

    fn supports_parallel_tool_calls(&self) -> bool {
        self.supports_parallel_tool_calls
    }

    fn supports_prompt_cache_key(&self) -> bool {
        true
    }

    pub fn supports_priority(&self) -> bool {
        self.supports_priority
    }

    fn from_catalog(model: CodexModelInfo) -> Option<Self> {
        if model.visibility != "list" {
            return None;
        }

        let context_window = model
            .context_window
            .or(model.max_context_window)
            .and_then(|value| u64::try_from(value).ok())
            .filter(|value| *value > 0)
            .unwrap_or_else(|| Self::fallback_context_window(&model.slug));
        let mut supported_reasoning_efforts = model
            .supported_reasoning_levels
            .into_iter()
            .filter_map(|level| reasoning_effort_from_catalog(&level.effort))
            .collect::<Vec<_>>();
        supported_reasoning_efforts.dedup();
        if supported_reasoning_efforts.is_empty() {
            supported_reasoning_efforts.extend([
                ReasoningEffort::Low,
                ReasoningEffort::Medium,
                ReasoningEffort::High,
                ReasoningEffort::XHigh,
            ]);
        }

        let default_reasoning_effort = model
            .default_reasoning_level
            .as_deref()
            .and_then(reasoning_effort_from_catalog)
            .filter(|effort| supported_reasoning_efforts.contains(effort))
            .or_else(|| supported_reasoning_efforts.first().copied());
        let supports_priority = model
            .additional_speed_tiers
            .iter()
            .any(|tier| tier == "priority")
            || model.service_tiers.iter().any(|tier| tier.id == "priority");
        let supports_images = model.input_modalities.is_empty()
            || model
                .input_modalities
                .iter()
                .any(|modality| modality == "image");

        Some(Self {
            display_name: if model.display_name.trim().is_empty() {
                model.slug.clone().into()
            } else {
                model.display_name.into()
            },
            id: model.slug.into(),
            context_window: Arc::new(AtomicU64::new(context_window)),
            default_reasoning_effort,
            supported_reasoning_efforts: supported_reasoning_efforts.into(),
            supports_images,
            supports_parallel_tool_calls: model.supports_parallel_tool_calls,
            supports_priority,
        })
    }
}

#[derive(Deserialize)]
struct CodexModelsResponse {
    models: Vec<CodexModelInfo>,
}

#[derive(Deserialize)]
struct CodexModelInfo {
    slug: String,
    display_name: String,
    #[serde(default)]
    default_reasoning_level: Option<String>,
    #[serde(default)]
    supported_reasoning_levels: Vec<CodexReasoningLevel>,
    #[serde(default)]
    visibility: String,
    #[serde(default)]
    priority: i32,
    #[serde(default)]
    additional_speed_tiers: Vec<String>,
    #[serde(default)]
    service_tiers: Vec<CodexServiceTier>,
    #[serde(default)]
    supports_parallel_tool_calls: bool,
    #[serde(default)]
    context_window: Option<i64>,
    #[serde(default)]
    max_context_window: Option<i64>,
    #[serde(default)]
    input_modalities: Vec<String>,
}

#[derive(Deserialize)]
struct CodexReasoningLevel {
    effort: String,
}

#[derive(Deserialize)]
struct CodexServiceTier {
    id: String,
}

fn reasoning_effort_from_catalog(effort: &str) -> Option<ReasoningEffort> {
    match effort {
        "none" => Some(ReasoningEffort::None),
        "minimal" => Some(ReasoningEffort::Minimal),
        "low" => Some(ReasoningEffort::Low),
        "medium" => Some(ReasoningEffort::Medium),
        "high" => Some(ReasoningEffort::High),
        "xhigh" => Some(ReasoningEffort::XHigh),
        "max" | "ultra" => Some(ReasoningEffort::Max),
        _ => None,
    }
}

/// Creates a [`LanguageModel`] for `model` that authenticates through
/// `state`'s credentials, refreshing them as needed.
pub fn create_language_model(
    model: ChatGptModel,
    state: &Entity<State>,
    cx: &App,
) -> Arc<dyn LanguageModel> {
    Arc::new(OpenAiSubscribedLanguageModel {
        id: LanguageModelId::from(model.id().to_string()),
        http_client: state.read(cx).http_client.clone(),
        model,
        state: state.clone(),
        request_limiter: RateLimiter::new(4),
    })
}

struct OpenAiSubscribedLanguageModel {
    id: LanguageModelId,
    model: ChatGptModel,
    state: Entity<State>,
    http_client: Arc<dyn HttpClient>,
    request_limiter: RateLimiter,
}

impl LanguageModel for OpenAiSubscribedLanguageModel {
    fn cache_warming_scope(&self, cx: &App) -> Option<String> {
        let state = self.state.read(cx);
        if state.account_mutation_in_progress || state.credentials.is_none() {
            return None;
        }
        Some(format!(
            "{}:{}:{}",
            state.active_session_id.as_deref()?,
            state.auth_generation,
            self.model.id()
        ))
    }

    fn id(&self) -> LanguageModelId {
        self.id.clone()
    }

    fn name(&self) -> LanguageModelName {
        LanguageModelName::from(self.model.display_name().to_string())
    }

    fn provider_id(&self) -> LanguageModelProviderId {
        PROVIDER_ID
    }

    fn provider_name(&self) -> LanguageModelProviderName {
        PROVIDER_NAME
    }

    fn supports_tools(&self) -> bool {
        true
    }

    fn supports_images(&self) -> bool {
        self.model.supports_images()
    }

    fn supports_tool_choice(&self, _choice: LanguageModelToolChoice) -> bool {
        true
    }

    fn supports_streaming_tools(&self) -> bool {
        true
    }

    fn supports_thinking(&self) -> bool {
        true
    }

    fn supports_fast_mode(&self) -> bool {
        self.model.supports_priority()
    }

    fn supports_server_side_compaction(&self) -> bool {
        true
    }

    fn supported_effort_levels(&self) -> Vec<LanguageModelEffortLevel> {
        let default_effort = self.model.default_reasoning_effort();
        self.model
            .supported_reasoning_efforts()
            .iter()
            .copied()
            .filter_map(|effort| {
                let (name, value) = match effort {
                    ReasoningEffort::None => return None,
                    ReasoningEffort::Minimal => ("Minimal", "minimal"),
                    ReasoningEffort::Low => ("Low", "low"),
                    ReasoningEffort::Medium => ("Medium", "medium"),
                    ReasoningEffort::High => ("High", "high"),
                    ReasoningEffort::XHigh => ("Extra High", "xhigh"),
                    ReasoningEffort::Max => ("Max", "max"),
                };

                Some(LanguageModelEffortLevel {
                    name: name.into(),
                    value: value.into(),
                    is_default: Some(effort) == default_effort,
                })
            })
            .collect()
    }

    fn telemetry_id(&self) -> String {
        format!("openai-subscribed/{}", self.model.id())
    }

    fn supports_explicit_compaction(&self) -> bool {
        true
    }

    fn compact(
        &self,
        mut request: LanguageModelRequest,
        cx: &AsyncApp,
    ) -> BoxFuture<'static, Result<CompactionResult, LanguageModelCompletionError>> {
        if !self.model.supports_priority() {
            request.speed = None;
        }
        let model_id = self.model.id().to_string();
        let supports_parallel_tool_calls = self.model.supports_parallel_tool_calls();
        let supports_prompt_cache_key = self.model.supports_prompt_cache_key();
        let default_reasoning_effort = self.model.default_reasoning_effort();
        let supports_none_reasoning_effort = self
            .model
            .supported_reasoning_efforts()
            .contains(&ReasoningEffort::None);
        let state = self.state.downgrade();
        let http_client = self.http_client.clone();
        let request_limiter = self.request_limiter.clone();
        let future = cx.spawn(async move |cx| {
            // Mark the whole operation busy up front, including credential
            // refresh and request preparation, so account mutations are
            // blocked for the full lifetime of the compaction.
            let active_operations = state
                .read_with(&*cx, |state, _| {
                    if state.account_mutation_in_progress {
                        return Err(anyhow!("A ChatGPT account operation is in progress"));
                    }
                    Ok(state.active_operations.clone())
                })
                .map_err(LanguageModelCompletionError::Other)?
                .map_err(LanguageModelCompletionError::Other)?;
            active_operations.enter(&state, &cx);
            let operation_guard = ActiveOperationGuard(active_operations);
            let auth_generation = state
                .read_with(&*cx, |state, _| state.auth_generation)
                .map_err(LanguageModelCompletionError::Other)?;
            let creds = get_fresh_credentials(&state, &http_client, cx).await?;
            let (account_scope, current_generation) = state
                .read_with(&*cx, |state, _| {
                    (state.active_session_id.clone(), state.auth_generation)
                })
                .map_err(LanguageModelCompletionError::Other)?;
            if current_generation != auth_generation {
                return Err(LanguageModelCompletionError::Other(anyhow!(
                    "ChatGPT account changed while preparing compaction"
                )));
            }
            let mut responses_request = into_open_ai_response_with_account_scope(
                request,
                &model_id,
                supports_parallel_tool_calls,
                supports_prompt_cache_key,
                None,
                default_reasoning_effort,
                supports_none_reasoning_effort,
                &PROVIDER_ID,
                account_scope.as_deref(),
            )
            .map_err(LanguageModelCompletionError::Other)?;
            responses_request.store = Some(false);
            responses_request.instructions.get_or_insert_default();
            responses_request.context_management = None;
            responses_request
                .input
                .push(ResponseInputItem::CompactionTrigger);
            let request_body = serde_json::to_string(&responses_request)
                .map_err(|error| LanguageModelCompletionError::Other(error.into()))?;
            let is_streaming = responses_request.stream;
            let extra_headers = codex_headers(&creds, responses_request.prompt_cache_key.as_deref());
            let access_token = creds.access_token.clone();
            let provider_name = PROVIDER_NAME.0.to_string();
            let background_executor = cx.background_executor().clone();

            let result = async {
                let operation_started_at = Instant::now();
                for attempt in 1..=COMPACTION_MAX_ATTEMPTS {
                    let remaining_operation_time = COMPACTION_OPERATION_TIMEOUT
                        .saturating_sub(operation_started_at.elapsed());
                    if remaining_operation_time.is_zero() {
                        return Err(LanguageModelCompletionError::Other(anyhow!(
                            "ChatGPT subscription compaction timed out after {COMPACTION_OPERATION_TIMEOUT:?}"
                        )));
                    }
                    let started_at = Instant::now();
                    let attempt_body = request_body.clone();
                    let attempt_headers = extra_headers.clone();
                    let attempt_access_token = access_token.clone();
                    let attempt_http_client = http_client.clone();
                    let attempt_provider_name = provider_name.clone();
                    let attempt_account_scope = account_scope.clone();
                    let attempt_background_executor = background_executor.clone();
                    let attempt_request = request_limiter
                        .run(async move {
                            let response_stream = stream_response_with_body(
                                attempt_http_client.as_ref(),
                                attempt_provider_name.as_str(),
                                CODEX_BASE_URL,
                                &attempt_access_token,
                                attempt_body,
                                is_streaming,
                                &attempt_headers,
                            )
                            .await
                            .map_err(LanguageModelCompletionError::from)?;
                            let mapper = OpenAiResponseEventMapper::new_with_account_scope(
                                PROVIDER_ID,
                                attempt_account_scope,
                            );
                            let mut event_stream = mapper.map_stream(response_stream.boxed());
                            let mut compacted_context = None;
                            let mut usage = language_model::TokenUsage::default();

                            while let Some(event) = event_stream.next().await {
                                match event? {
                                    LanguageModelCompletionEvent::Compaction(
                                        language_model::CompactionUpdate::Finished(context),
                                    ) => {
                                        if compacted_context.replace(context).is_some() {
                                            return Err(LanguageModelCompletionError::Other(anyhow!(
                                                "ChatGPT subscription compaction returned multiple replacement contexts"
                                            )));
                                        }
                                    }
                                    LanguageModelCompletionEvent::UsageUpdate(updated_usage) => {
                                        usage = updated_usage;
                                    }
                                    _ => {}
                                }
                            }

                            let context = compacted_context.ok_or_else(|| {
                                LanguageModelCompletionError::Other(anyhow!(
                                    "ChatGPT subscription compaction returned no replacement context"
                                ))
                            })?;
                            Ok(CompactionResult { context, usage })
                        })
                        .fuse();
                    let attempt_timeout = COMPACTION_ATTEMPT_TIMEOUT.min(remaining_operation_time);
                    let timeout = attempt_background_executor.timer(attempt_timeout).fuse();
                    futures::pin_mut!(attempt_request, timeout);
                    let attempt_result = futures::select! {
                        result = attempt_request => result,
                        _ = timeout => Err(LanguageModelCompletionError::Other(anyhow!(
                            "ChatGPT subscription compaction attempt timed out after {attempt_timeout:?}"
                        ))),
                    };
                    let elapsed_ms = started_at.elapsed().as_millis();

                    match attempt_result {
                        Ok(result) => {
                            log::info!(
                                "ChatGPT subscription compaction attempt={} elapsed_ms={} result_class=success",
                                attempt,
                                elapsed_ms,
                            );
                            return Ok(result);
                        }
                        Err(error) => {
                            let result_class = compaction_error_class(&error);
                            let retryable = is_retryable_compaction_error(&error);
                            log::warn!(
                                "ChatGPT subscription compaction attempt={} elapsed_ms={} result_class={} retryable={}",
                                attempt,
                                elapsed_ms,
                                result_class,
                                retryable,
                            );
                            if !retryable || attempt == COMPACTION_MAX_ATTEMPTS {
                                return Err(error);
                            }
                            let delay = compaction_retry_delay(&error, attempt);
                            log::debug!(
                                "ChatGPT subscription compaction retrying attempt={} delay_ms={}",
                                attempt + 1,
                                delay.as_millis(),
                            );
                            if COMPACTION_OPERATION_TIMEOUT
                                .saturating_sub(operation_started_at.elapsed())
                                <= delay
                            {
                                return Err(error);
                            }
                            background_executor.timer(delay).await;
                        }
                    }
                }
                Err(LanguageModelCompletionError::Other(anyhow!(
                    "ChatGPT subscription compaction ended without a result"
                )))
            }
            .await;
            drop(operation_guard);
            result
        });

        future.boxed()
    }

    fn max_token_count(&self) -> u64 {
        self.model.max_token_count()
    }

    fn max_output_tokens(&self) -> Option<u64> {
        self.model.max_output_tokens()
    }

    fn stream_completion(
        &self,
        mut request: LanguageModelRequest,
        cx: &AsyncApp,
    ) -> BoxFuture<
        'static,
        Result<
            futures::stream::BoxStream<
                'static,
                Result<LanguageModelCompletionEvent, LanguageModelCompletionError>,
            >,
            LanguageModelCompletionError,
        >,
    > {
        if !self.model.supports_priority() {
            request.speed = None;
        }

        // The Codex backend rejects `max_output_tokens` (`Unsupported parameter`),
        // unlike the public OpenAI Responses API. Pass `None` so the field is
        // omitted from the serialized request body entirely.
        let model_id = self.model.id().to_string();
        let supports_parallel_tool_calls = self.model.supports_parallel_tool_calls();
        let supports_prompt_cache_key = self.model.supports_prompt_cache_key();
        let default_reasoning_effort = self.model.default_reasoning_effort();
        let supports_none_reasoning_effort = self
            .model
            .supported_reasoning_efforts()
            .contains(&ReasoningEffort::None);
        let state = self.state.downgrade();
        let http_client = self.http_client.clone();
        let request_limiter = self.request_limiter.clone();
        let background_executor = cx.background_executor().clone();

        let expected_auth_generation = state.read_with(cx, |state, _| state.auth_generation);
        let future = cx.spawn(async move |cx| {
            let expected_auth_generation = expected_auth_generation
                .map_err(LanguageModelCompletionError::Other)?;
            // Mark the whole operation busy up front, including credential
            // refresh and request preparation, so account mutations are
            // blocked for the full lifetime of the stream.
            let active_operations = state
                .read_with(&*cx, |state, _| {
                    if state.account_mutation_in_progress {
                        return Err(anyhow!("A ChatGPT account operation is in progress"));
                    }
                    if state.auth_generation != expected_auth_generation {
                        return Err(anyhow!("ChatGPT account changed before request started"));
                    }
                    Ok(state.active_operations.clone())
                })
                .map_err(LanguageModelCompletionError::Other)?
                .map_err(LanguageModelCompletionError::Other)?;
            active_operations.enter(&state, &cx);
            let operation_guard = ActiveOperationGuard(active_operations);
            let auth_generation = state
                .read_with(&*cx, |state, _| state.auth_generation)
                .map_err(LanguageModelCompletionError::Other)?;
            let creds = get_fresh_credentials(&state, &http_client, cx).await?;
            let (account_scope, current_generation) = state
                .read_with(&*cx, |state, _| {
                    (state.active_session_id.clone(), state.auth_generation)
                })
                .map_err(LanguageModelCompletionError::Other)?;
            if current_generation != auth_generation {
                return Err(LanguageModelCompletionError::Other(anyhow!(
                    "ChatGPT account changed while preparing request"
                )));
            }
            let mut responses_request = into_open_ai_response_with_account_scope(
                request,
                &model_id,
                supports_parallel_tool_calls,
                supports_prompt_cache_key,
                None,
                default_reasoning_effort,
                supports_none_reasoning_effort,
                &PROVIDER_ID,
                account_scope.as_deref(),
            )
            .map_err(LanguageModelCompletionError::Other)?;
            responses_request.store = Some(false);

            // `into_open_ai_response` already hoists system messages into
            // `instructions`, which is the only form the Codex backend accepts.
            responses_request.instructions.get_or_insert_default();

            let extra_headers = codex_headers(&creds, responses_request.prompt_cache_key.as_deref());
            let access_token = creds.access_token.clone();
            let timeout = cx
                .background_executor()
                .timer(RESPONSE_STREAM_IDLE_TIMEOUT);
            request_limiter
                .stream(async move {
                    let provider_name = PROVIDER_NAME;
                    let response = stream_response(
                        http_client.as_ref(),
                        provider_name.0.as_str(),
                        CODEX_BASE_URL,
                        &access_token,
                        responses_request,
                        &extra_headers,
                    )
                    .fuse();
                    let timeout = FutureExt::fuse(timeout);
                    futures::pin_mut!(response, timeout);
                    futures::select! {
                        result = response => result.map_err(LanguageModelCompletionError::from),
                        _ = timeout => Err(LanguageModelCompletionError::Other(anyhow!(
                            "ChatGPT subscription response did not start within {RESPONSE_STREAM_IDLE_TIMEOUT:?}"
                        ))),
                    }
                })
                .await
                .map(|stream| (stream, account_scope, operation_guard))
        });

        async move {
            let (stream, account_scope, operation_guard) = future.await?;
            let mapper =
                OpenAiResponseEventMapper::new_with_account_scope(PROVIDER_ID, account_scope);
            let stream = language_model::stream_in_background(
                mapper.map_stream(stream.boxed()).boxed(),
                background_executor.clone(),
            );
            Ok(stream_with_idle_timeout(
                stream,
                background_executor,
                operation_guard,
            ))
        }
        .boxed()
    }
}

fn codex_headers(creds: &CodexCredentials, routing_cache_key: Option<&str>) -> CustomHeaders {
    let mut header_pairs: Vec<(HeaderName, HeaderValue)> = vec![
        (
            HeaderName::from_static("originator"),
            HeaderValue::from_static("zed"),
        ),
        (
            HeaderName::from_static("openai-beta"),
            HeaderValue::from_static("responses=experimental"),
        ),
    ];
    if let Some(ref id) = creds.account_id
        && !id.is_empty()
        && let Ok(value) = HeaderValue::from_str(id)
    {
        header_pairs.push((HeaderName::from_static("chatgpt-account-id"), value));
    }
    if let Some(routing_cache_key) = routing_cache_key
        && !routing_cache_key.is_empty()
        && let Ok(value) = HeaderValue::from_str(routing_cache_key)
    {
        header_pairs.push((HeaderName::from_static("session-id"), value.clone()));
        header_pairs.push((HeaderName::from_static("thread-id"), value));
    }
    CustomHeaders::new(header_pairs)
}

async fn fetch_codex_models(
    http_client: &dyn HttpClient,
    credentials: &CodexCredentials,
    client_version: &str,
) -> Result<Vec<ChatGptModel>> {
    let mut url = url::Url::parse(&format!("{CODEX_BASE_URL}/models"))?;
    url.query_pairs_mut()
        .append_pair("client_version", client_version);
    let headers = codex_headers(credentials, None);
    let request = HttpRequest::builder()
        .method(Method::GET)
        .uri(url.as_str())
        .header(
            "Authorization",
            format!("Bearer {}", credentials.access_token.trim()),
        )
        .header("Accept", "application/json")
        .extra_headers(&headers)
        .body(AsyncBody::empty())?;

    let mut response = http_client.send(request).await?;
    let status = response.status();
    let mut body = String::new();
    smol::io::AsyncReadExt::read_to_string(response.body_mut(), &mut body).await?;
    if !status.is_success() {
        return Err(anyhow!(
            "ChatGPT subscription model catalog request failed (HTTP {status}): {body}"
        ));
    }

    let mut response: CodexModelsResponse =
        serde_json::from_str(&body).context("Failed to parse ChatGPT model catalog")?;
    response.models.sort_by_key(|model| model.priority);
    Ok(response
        .models
        .into_iter()
        .filter_map(ChatGptModel::from_catalog)
        .collect())
}

struct ActiveOperationGuard(Arc<BusyNotifier>);

impl Drop for ActiveOperationGuard {
    fn drop(&mut self) {
        self.0.exit();
    }
}

/// Fetches usage for every persisted account in one serialized cycle. Quota is
/// account-scoped, so it must not depend on which account happens to be active
/// in the UI. Credentials are read from the secure store for each session and
/// the whole result is discarded if an account mutation races the cycle.
async fn refresh_all_account_quotas(state: &WeakEntity<State>, cx: &mut AsyncApp) -> Result<()> {
    let Some((provider, http_client, session_ids, generation)) = state
        .read_with(cx, |state, _| {
            if state.manifest_load_state != ManifestLoadState::Loaded {
                return None;
            }
            Some((
                state.credentials_provider.clone(),
                state.http_client.clone(),
                state
                    .manifest
                    .sessions
                    .iter()
                    .map(|session| session.session_id.clone())
                    .collect::<Vec<_>>(),
                state.auth_generation,
            ))
        })
        .ok()
        .flatten()
    else {
        return Ok(());
    };

    let mut updates = Vec::new();
    let mut credential_updates = Vec::new();
    let mut reauthentication_required = Vec::new();
    for session_id in session_ids {
        let mut credentials = match provider
            .read_credentials(&account_credentials_key(&session_id), cx)
            .await
        {
            Ok(Some((_, bytes))) => match serde_json::from_slice::<CodexCredentials>(&bytes) {
                Ok(credentials) => credentials,
                Err(error) => {
                    log::warn!(
                        "ChatGPT quota check skipped account {session_id}: invalid credentials: {error}"
                    );
                    continue;
                }
            },
            Ok(None) => {
                log::warn!("ChatGPT quota check skipped account {session_id}: credentials missing");
                continue;
            }
            Err(error) => {
                log::warn!(
                    "ChatGPT quota check skipped account {session_id}: keychain read failed: {error:#}"
                );
                continue;
            }
        };

        // The usage endpoint is account-scoped and requires a current access token. Refresh an
        // expired credential before querying so a background account receives the same accurate
        // quota as the active one.
        if credentials.is_expired() {
            match refresh_quota_credentials(
                state,
                generation,
                &provider,
                &http_client,
                &session_id,
                &credentials,
                cx,
            )
            .await
            {
                Ok(refreshed) => {
                    credential_updates.push((
                        session_id.clone(),
                        credentials.refresh_token.clone(),
                        refreshed.clone(),
                    ));
                    credentials = refreshed;
                }
                Err(RefreshError::Fatal(error)) => {
                    log::warn!(
                        "ChatGPT quota check requires sign-in again for account {session_id}: {error:#}"
                    );
                    reauthentication_required.push(session_id);
                    continue;
                }
                Err(RefreshError::Transient(error)) => {
                    log::debug!(
                        "ChatGPT quota token refresh failed transiently for account {session_id}: {error:#}"
                    );
                    continue;
                }
            }
        }

        let mut result = fetch_quota(http_client.as_ref(), &credentials).await;
        // ChatGPT can invalidate an access token before its JWT expiry. Match Cockpit's recovery
        // behavior: force exactly one refresh-token exchange on 401, persist rotated tokens, then
        // retry the official usage endpoint once.
        if matches!(result, Err(QuotaFetchError::Unauthorized(_))) {
            match refresh_quota_credentials(
                state,
                generation,
                &provider,
                &http_client,
                &session_id,
                &credentials,
                cx,
            )
            .await
            {
                Ok(refreshed) => {
                    credential_updates.push((
                        session_id.clone(),
                        credentials.refresh_token.clone(),
                        refreshed.clone(),
                    ));
                    credentials = refreshed;
                    result = fetch_quota(http_client.as_ref(), &credentials).await;
                }
                Err(RefreshError::Fatal(error)) => {
                    log::warn!(
                        "ChatGPT quota check requires sign-in again for account {session_id}: {error:#}"
                    );
                    reauthentication_required.push(session_id);
                    continue;
                }
                Err(RefreshError::Transient(error)) => {
                    log::debug!(
                        "ChatGPT quota token recovery failed transiently for account {session_id}: {error:#}"
                    );
                    continue;
                }
            }
        }

        match result {
            Ok(quota) => updates.push((session_id, quota)),
            Err(error) => {
                log::debug!("ChatGPT quota check failed for account {session_id}: {error}")
            }
        }
    }

    if updates.is_empty() && credential_updates.is_empty() && reauthentication_required.is_empty() {
        return Ok(());
    }
    let updated = state.update(cx, |state, cx| {
        if state.auth_generation != generation
            || state.manifest_load_state != ManifestLoadState::Loaded
        {
            return false;
        }
        let fetched_at = now_ms();
        for (session_id, quota) in updates {
            if let Some(session) = state
                .manifest
                .sessions
                .iter_mut()
                .find(|session| session.session_id == session_id)
            {
                session.quota = Some(quota);
                session.quota_fetched_at_ms = Some(fetched_at);
                session.reauthentication_required = false;
            }
        }
        for (session_id, old_refresh_token, credentials) in credential_updates {
            if state.active_session_id.as_deref() == Some(session_id.as_str())
                && state
                    .credentials
                    .as_ref()
                    .is_some_and(|current| current.refresh_token == old_refresh_token)
            {
                state.credentials = Some(credentials.clone());
            }
            if let Some(session) = state
                .manifest
                .sessions
                .iter_mut()
                .find(|session| session.session_id == session_id)
            {
                session.token_expires_at_ms = Some(credentials.expires_at_ms);
            }
        }
        for session_id in reauthentication_required {
            if let Some(session) = state
                .manifest
                .sessions
                .iter_mut()
                .find(|session| session.session_id == session_id)
            {
                session.reauthentication_required = true;
            }
        }
        cx.notify();
        true
    })?;
    if updated {
        persist_manifest(&provider, state, cx).await?;
    } else {
        log::debug!("Discarded stale ChatGPT quota cycle after account mutation");
    }
    Ok(())
}

async fn refresh_quota_credentials(
    state: &WeakEntity<State>,
    generation: u64,
    provider: &Arc<dyn CredentialsProvider>,
    http_client: &Arc<dyn HttpClient>,
    session_id: &str,
    credentials: &CodexCredentials,
    cx: &AsyncApp,
) -> Result<CodexCredentials, RefreshError> {
    let mut refreshed = refresh_token(http_client, &credentials.refresh_token).await?;
    refreshed.account_id = refreshed.account_id.or(credentials.account_id.clone());
    refreshed.email = refreshed.email.or(credentials.email.clone());
    refreshed.user_id = refreshed.user_id.or(credentials.user_id.clone());
    refreshed.plan_type = refreshed.plan_type.or(credentials.plan_type.clone());

    let key = account_credentials_key(session_id);
    let session_is_current = state
        .read_with(cx, |state, _| {
            state.auth_generation == generation
                && state
                    .manifest
                    .sessions
                    .iter()
                    .any(|session| session.session_id == session_id)
        })
        .unwrap_or(false);
    if !session_is_current {
        return Err(RefreshError::Transient(anyhow!(
            "ChatGPT account changed during quota token refresh"
        )));
    }

    // A foreground request may have rotated this credential while the quota request was in
    // flight. Never overwrite that newer token; use it for the quota retry instead.
    if let Some((_, bytes)) = provider
        .read_credentials(&key, cx)
        .await
        .map_err(RefreshError::Transient)?
    {
        let current = serde_json::from_slice::<CodexCredentials>(&bytes)
            .map_err(|error| RefreshError::Transient(error.into()))?;
        if current.refresh_token != credentials.refresh_token
            || current.access_token != credentials.access_token
        {
            return Ok(current);
        }
    }

    let json =
        serde_json::to_vec(&refreshed).map_err(|error| RefreshError::Transient(error.into()))?;
    provider
        .write_credentials(&key, "Bearer", &json, cx)
        .await
        .map_err(RefreshError::Transient)?;

    // Sign-out can race the keychain write after the guard above. If the account was removed,
    // delete only the stale scoped credential so a background refresh cannot resurrect it.
    let session_still_current = state
        .read_with(cx, |state, _| {
            state.auth_generation == generation
                && state
                    .manifest
                    .sessions
                    .iter()
                    .any(|session| session.session_id == session_id)
        })
        .unwrap_or(false);
    if !session_still_current {
        let account_was_removed = state
            .read_with(cx, |state, _| {
                !state
                    .manifest
                    .sessions
                    .iter()
                    .any(|session| session.session_id == session_id)
            })
            .unwrap_or(true);
        if account_was_removed {
            provider.delete_credentials(&key, cx).await.log_err();
        }
        return Err(RefreshError::Transient(anyhow!(
            "ChatGPT account changed during quota credential persistence"
        )));
    }
    Ok(refreshed)
}

/// Persists the in-memory manifest, refusing to overwrite a manifest that
/// could not be loaded and converging on the newest identity if a background
/// write races a newer account mutation (a stale write must never leave a
/// different account active on disk).
async fn persist_manifest(
    credentials_provider: &Arc<dyn CredentialsProvider>,
    state: &WeakEntity<State>,
    cx: &mut AsyncApp,
) -> Result<()> {
    for _ in 0..4 {
        let Some((manifest, generation, load_ready)) = state
            .read_with(cx, |s, _| {
                (
                    s.manifest.clone(),
                    s.auth_generation,
                    s.manifest_load_state == ManifestLoadState::Loaded,
                )
            })
            .ok()
        else {
            return Ok(());
        };
        if !load_ready {
            return Ok(());
        }
        write_manifest(credentials_provider, &manifest, cx).await?;
        let still_current = state
            .read_with(cx, |s, _| s.auth_generation == generation)
            .unwrap_or(false);
        if still_current {
            return Ok(());
        }
    }
    Err(anyhow!(
        "ChatGPT subscription manifest writes kept racing account mutations"
    ))
}

async fn write_manifest(
    credentials_provider: &Arc<dyn CredentialsProvider>,
    manifest: &AccountManifest,
    cx: &AsyncApp,
) -> Result<()> {
    if manifest.sessions.is_empty() {
        credentials_provider
            .delete_credentials(ACCOUNT_MANIFEST_KEY, cx)
            .await
    } else {
        let json = serde_json::to_vec(manifest)?;
        credentials_provider
            .write_credentials(ACCOUNT_MANIFEST_KEY, "json", &json, cx)
            .await
    }
}

fn stream_with_idle_timeout(
    stream: futures::stream::BoxStream<
        'static,
        Result<LanguageModelCompletionEvent, LanguageModelCompletionError>,
    >,
    background_executor: BackgroundExecutor,
    operation_guard: ActiveOperationGuard,
) -> futures::stream::BoxStream<
    'static,
    Result<LanguageModelCompletionEvent, LanguageModelCompletionError>,
> {
    futures::stream::unfold(
        (stream, background_executor, false, operation_guard),
        |(mut stream, background_executor, timed_out, operation_guard)| async move {
            if timed_out {
                return None;
            }

            let outcome = {
                let next_event = stream.next().fuse();
                let timeout = FutureExt::fuse(
                    background_executor.timer(RESPONSE_STREAM_IDLE_TIMEOUT),
                );
                futures::pin_mut!(next_event, timeout);
                futures::select! {
                    event = next_event => Ok(event),
                    _ = timeout => Err(()),
                }
            };

            match outcome {
                Ok(Some(event)) => Some((
                    event,
                    (stream, background_executor, false, operation_guard),
                )),
                Ok(None) => None,
                Err(()) => Some((
                    Err(LanguageModelCompletionError::Other(anyhow!(
                        "ChatGPT subscription response was idle for {RESPONSE_STREAM_IDLE_TIMEOUT:?}"
                    ))),
                    (stream, background_executor, true, operation_guard),
                )),
            }
        },
    )
    .boxed()
}

fn compaction_error_class(error: &LanguageModelCompletionError) -> &'static str {
    match error {
        LanguageModelCompletionError::ProviderRejection {
            category: ProviderErrorCategory::RateLimit,
            ..
        } => "rate_limit",
        LanguageModelCompletionError::ProviderRejection {
            category: ProviderErrorCategory::Overloaded | ProviderErrorCategory::InternalServer,
            ..
        } => "server_error",
        LanguageModelCompletionError::ProviderRejection {
            category: ProviderErrorCategory::Authentication,
            ..
        } => "auth",
        LanguageModelCompletionError::ProviderRejection {
            category: ProviderErrorCategory::Timeout,
            ..
        } => "timeout",
        LanguageModelCompletionError::ApiReadResponseError { .. }
        | LanguageModelCompletionError::HttpSend { .. } => "transport",
        LanguageModelCompletionError::Other(error) if is_transport_error(error) => "transport",
        LanguageModelCompletionError::Other(error) if is_timeout_error(error) => "timeout",
        LanguageModelCompletionError::ProviderRejection {
            category:
                ProviderErrorCategory::PromptTooLarge { .. }
                | ProviderErrorCategory::RequestPayloadTooLarge,
            ..
        } => "prompt_too_large",
        LanguageModelCompletionError::ProviderRejection {
            category: ProviderErrorCategory::InvalidRequest,
            ..
        } => "bad_request",
        _ => "non_retryable",
    }
}

fn is_retryable_compaction_error(error: &LanguageModelCompletionError) -> bool {
    matches!(
        compaction_error_class(error),
        "rate_limit" | "server_error" | "transport" | "timeout"
    )
}

fn compaction_retry_delay(
    error: &LanguageModelCompletionError,
    completed_attempt: usize,
) -> Duration {
    let provider_delay = match error {
        LanguageModelCompletionError::ProviderRejection { retry_after, .. } => *retry_after,
        _ => None,
    };

    provider_delay.unwrap_or_else(|| {
        COMPACTION_RETRY_DELAYS
            .get(completed_attempt.saturating_sub(1))
            .copied()
            .unwrap_or(Duration::from_secs(1))
    })
}

fn is_timeout_error(error: &anyhow::Error) -> bool {
    let message = error.to_string().to_ascii_lowercase();
    message.contains("timed out") || message.contains("timeout")
}

fn is_transport_error(error: &anyhow::Error) -> bool {
    let message = error.to_string().to_ascii_lowercase();
    [
        "connection",
        "disconnect",
        "broken pipe",
        "reset by peer",
        "network",
        "unexpected eof",
        "end of file",
        "failed to fill whole buffer",
        "incomplete message",
        "connection closed",
        "stream closed",
        "socket closed",
    ]
    .iter()
    .any(|marker| message.contains(marker))
}

async fn get_fresh_credentials(
    state: &WeakEntity<State>,
    http_client: &Arc<dyn HttpClient>,
    cx: &mut AsyncApp,
) -> Result<CodexCredentials, LanguageModelCompletionError> {
    let (creds, existing_task) = state
        .read_with(&*cx, |s, _| (s.credentials.clone(), s.refresh_task.clone()))
        .map_err(LanguageModelCompletionError::Other)?;

    let creds = creds.ok_or(LanguageModelCompletionError::NoApiKey {
        provider: PROVIDER_NAME,
    })?;

    if !creds.is_expired() {
        return Ok(creds);
    }

    // If another caller is already refreshing, await their result.
    if let Some(shared_task) = existing_task {
        return shared_task
            .await
            .map_err(|e| LanguageModelCompletionError::Other(anyhow::anyhow!("{e}")));
    }

    // We are the first caller to notice expiry — spawn the refresh task.
    let http_client_clone = http_client.clone();
    let state_clone = state.clone();
    let refresh_token_value = creds.refresh_token.clone();

    // Capture the identity so the refreshed credential is written back under
    // the same account, even if the user switches while the refresh runs.
    let (generation, session_id) = state
        .read_with(&*cx, |s, _| {
            (s.auth_generation, s.active_session_id.clone())
        })
        .map_err(LanguageModelCompletionError::Other)?;
    let Some(session_id) = session_id else {
        return Err(LanguageModelCompletionError::NoApiKey {
            provider: PROVIDER_NAME,
        });
    };
    let account_key = account_credentials_key(&session_id);

    let shared_task = cx
        .spawn(async move |cx| {
            let result = refresh_token(&http_client_clone, &refresh_token_value).await;

            match result {
                Ok(mut refreshed) => {
                    // Some refresh responses omit identity claims. Keep the
                    // account routing metadata from the previous token so a
                    // refresh never silently drops chatgpt-account-id.
                    refreshed.account_id = refreshed.account_id.or(creds.account_id.clone());
                    refreshed.email = refreshed.email.or(creds.email.clone());
                    refreshed.user_id = refreshed.user_id.or(creds.user_id.clone());
                    refreshed.plan_type = refreshed.plan_type.or(creds.plan_type.clone());
                    let persist_result: Result<CodexCredentials, Arc<anyhow::Error>> = async {
                        // Only apply the result if this refresh still belongs
                        // to the current identity (a sign-in/out or switch
                        // during the refresh discards it).
                        let (current_generation, current_session_id) = state_clone
                            .read_with(&*cx, |s, _| {
                                (s.auth_generation, s.active_session_id.clone())
                            })
                            .map_err(|e| Arc::new(e))?;
                        if current_generation != generation
                            || current_session_id.as_deref() != Some(session_id.as_str())
                        {
                            return Err(Arc::new(anyhow!(
                                "ChatGPT account changed during token refresh"
                            )));
                        }

                        let credentials_provider = state_clone
                            .read_with(&*cx, |s, _| s.credentials_provider.clone())
                            .map_err(|e| Arc::new(e))?;

                        let json =
                            serde_json::to_vec(&refreshed).map_err(|e| Arc::new(e.into()))?;

                        // The refreshed credential belongs to the account the
                        // refresh started for, regardless of what is active now.
                        credentials_provider
                            .write_credentials(&account_key, "Bearer", &json, &*cx)
                            .await
                            .map_err(|e| Arc::new(e))?;

                        // Re-check identity after the keychain write so the
                        // in-memory state can never be clobbered by a stale
                        // refresh that raced a switch.
                        let (current_generation, current_session_id) = state_clone
                            .read_with(&*cx, |s, _| {
                                (s.auth_generation, s.active_session_id.clone())
                            })
                            .map_err(|e| Arc::new(e))?;
                        if current_generation != generation
                            || current_session_id.as_deref() != Some(session_id.as_str())
                        {
                            return Err(Arc::new(anyhow!(
                                "ChatGPT account changed during token refresh"
                            )));
                        }
                        state_clone
                            .update(cx, |s, cx| {
                                s.credentials = Some(refreshed.clone());
                                if let Some(session) = s
                                    .manifest
                                    .sessions
                                    .iter_mut()
                                    .find(|session| session.session_id == session_id)
                                {
                                    session.token_expires_at_ms = Some(refreshed.expires_at_ms);
                                }
                                s.refresh_task = None;
                                cx.notify();
                            })
                            .map_err(|e| Arc::new(e))?;
                        // Persist the refreshed expiry so it survives a restart.
                        persist_manifest(&credentials_provider, &state_clone, cx)
                            .await
                            .log_err();

                        Ok(refreshed)
                    }
                    .await;

                    // Clear refresh_task on failure too.
                    if persist_result.is_err() {
                        state_clone
                            .update(cx, |s, _| {
                                s.refresh_task = None;
                            })
                            .ok();
                    }

                    persist_result
                }
                Err(RefreshError::Fatal(e)) => {
                    log::error!("ChatGPT subscription token refresh failed fatally: {e:?}");
                    let (current_generation, current_session_id) = state_clone
                        .read_with(&*cx, |s, _| {
                            (s.auth_generation, s.active_session_id.clone())
                        })
                        .unwrap_or((u64::MAX, None));
                    if current_generation == generation
                        && current_session_id.as_deref() == Some(session_id.as_str())
                    {
                        // Mark the account as needing re-authentication, but
                        // keep the session metadata and the keychain entry so
                        // the account remains visible and can be signed back
                        // into. A different account must never be affected.
                        state_clone
                            .update(cx, |s, cx| {
                                s.refresh_task = None;
                                s.credentials = None;
                                if let Some(session) = s
                                    .manifest
                                    .sessions
                                    .iter_mut()
                                    .find(|session| session.session_id == session_id)
                                {
                                    session.reauthentication_required = true;
                                }
                                s.last_auth_error =
                                    Some("Your session has expired. Please sign in again.".into());
                                cx.notify();
                            })
                            .ok();
                        if let Ok(credentials_provider) =
                            state_clone.read_with(&*cx, |s, _| s.credentials_provider.clone())
                        {
                            persist_manifest(&credentials_provider, &state_clone, cx)
                                .await
                                .log_err();
                        }
                    } else {
                        log::warn!(
                            "ChatGPT subscription token refresh failed after the account \
                             changed; leaving the current account untouched"
                        );
                        state_clone
                            .update(cx, |s, _| {
                                s.refresh_task = None;
                            })
                            .ok();
                    }
                    Err(Arc::new(e))
                }
                Err(RefreshError::Transient(e)) => {
                    log::warn!("ChatGPT subscription token refresh failed transiently: {e:?}");
                    state_clone
                        .update(cx, |s, _| {
                            s.refresh_task = None;
                        })
                        .ok();
                    Err(Arc::new(e))
                }
            }
        })
        .shared();

    // Store the shared task so concurrent callers can join on it.
    state
        .update(cx, |s, _| {
            s.refresh_task = Some(shared_task.clone());
        })
        .map_err(LanguageModelCompletionError::Other)?;

    shared_task
        .await
        .map_err(|e| LanguageModelCompletionError::Other(anyhow::anyhow!("{e}")))
}

#[derive(Deserialize)]
struct TokenResponse {
    access_token: String,
    refresh_token: String,
    #[serde(default)]
    id_token: Option<String>,
    expires_in: u64,
    #[serde(default)]
    email: Option<String>,
}

// The OAuth client registered for `CLIENT_ID` (the Codex CLI's client) only allows
// `http://localhost:1455/auth/callback` and `http://localhost:1457/auth/callback`
// as redirect URIs; using anything else (different host, port, or path) causes
// auth.openai.com to reject the authorize request with a generic `unknown_error`
// before redirecting back. Keep these in sync with the Codex CLI's redirect URI
// allow-list (see codex-rs/login/src/server.rs in openai/codex).
const CODEX_CALLBACK_HOST: &str = "localhost";
const CODEX_CALLBACK_PORT: u16 = 1455;
const CODEX_CALLBACK_FALLBACK_PORT: u16 = 1457;
const CODEX_CALLBACK_PATH: &str = "/auth/callback";

async fn do_oauth_flow(
    http_client: Arc<dyn HttpClient>,
    cx: &AsyncApp,
) -> Result<CodexCredentials> {
    // Start the callback server FIRST so the redirect URI is ready
    let (redirect_uri, callback_rx) =
        oauth_callback_server::start_oauth_callback_server_with_config(
            oauth_callback_server::OAuthCallbackServerConfig {
                host: CODEX_CALLBACK_HOST,
                preferred_port: CODEX_CALLBACK_PORT,
                fallback_port: Some(CODEX_CALLBACK_FALLBACK_PORT),
                path: CODEX_CALLBACK_PATH,
            },
        )
        .context("Failed to start OAuth callback server")?;

    // PKCE verifier: 32 random bytes → base64url (no padding)
    let mut verifier_bytes = [0u8; 32];
    rand::rng().fill_bytes(&mut verifier_bytes);
    let verifier = URL_SAFE_NO_PAD.encode(verifier_bytes);

    // PKCE challenge: SHA-256(verifier) → base64url
    let mut hasher = Sha256::new();
    hasher.update(verifier.as_bytes());
    let challenge = URL_SAFE_NO_PAD.encode(hasher.finalize().as_slice());

    // CSRF state: 16 random bytes → hex string
    let mut state_bytes = [0u8; 16];
    rand::rng().fill_bytes(&mut state_bytes);
    let oauth_state: String = state_bytes.iter().map(|b| format!("{b:02x}")).collect();

    let mut auth_url = url::Url::parse(OPENAI_AUTHORIZE_URL).expect("valid base URL");
    auth_url
        .query_pairs_mut()
        .append_pair("client_id", CLIENT_ID)
        .append_pair("redirect_uri", &redirect_uri)
        // Deliberately excludes `api.connectors.read api.connectors.invoke`
        // (which Codex CLI requests): extra scopes inflate the
        // access-token JWT, and the serialized credentials must fit within
        // Windows Credential Manager's 2560-byte blob limit
        // (CRED_MAX_CREDENTIAL_BLOB_SIZE). See #58541.
        .append_pair("scope", "openid profile email offline_access")
        .append_pair("response_type", "code")
        .append_pair("code_challenge", &challenge)
        .append_pair("code_challenge_method", "S256")
        .append_pair("id_token_add_organizations", "true")
        .append_pair("state", &oauth_state)
        .append_pair("codex_cli_simplified_flow", "true")
        .append_pair("originator", "zed");

    // Open browser AFTER the listener is ready
    cx.update(|cx| cx.open_url(auth_url.as_str()));

    // Await the callback
    let callback = callback_rx
        .await
        .map_err(|_| anyhow!("OAuth callback was cancelled"))?
        .context("OAuth callback failed")?;

    // Validate CSRF state
    if callback.state != oauth_state {
        return Err(anyhow!("OAuth state mismatch"));
    }

    let tokens = exchange_code(&http_client, &callback.code, &verifier, &redirect_uri)
        .await
        .context("Token exchange failed")?;

    let jwt = tokens
        .id_token
        .as_deref()
        .unwrap_or(tokens.access_token.as_str());
    let claims = extract_jwt_claims(jwt);

    Ok(CodexCredentials {
        access_token: tokens.access_token,
        refresh_token: tokens.refresh_token,
        expires_at_ms: now_ms() + tokens.expires_in * 1000,
        account_id: claims.account_id,
        email: claims.email.or(tokens.email),
        user_id: claims.user_id,
        plan_type: claims.plan_type,
    })
}

async fn exchange_code(
    client: &Arc<dyn HttpClient>,
    code: &str,
    verifier: &str,
    redirect_uri: &str,
) -> Result<TokenResponse> {
    let body = form_urlencoded::Serializer::new(String::new())
        .append_pair("grant_type", "authorization_code")
        .append_pair("client_id", CLIENT_ID)
        .append_pair("code", code)
        .append_pair("redirect_uri", redirect_uri)
        .append_pair("code_verifier", verifier)
        .finish();

    let request = HttpRequest::builder()
        .method(Method::POST)
        .uri(OPENAI_TOKEN_URL)
        .header("Content-Type", "application/x-www-form-urlencoded")
        .body(AsyncBody::from(body))?;

    let mut response = client.send(request).await?;
    let mut body = String::new();
    smol::io::AsyncReadExt::read_to_string(response.body_mut(), &mut body).await?;

    if !response.status().is_success() {
        return Err(anyhow!(
            "Token exchange failed (HTTP {}): {body}",
            response.status()
        ));
    }

    serde_json::from_str::<TokenResponse>(&body).context("Failed to parse token response")
}

async fn refresh_token(
    client: &Arc<dyn HttpClient>,
    refresh_token: &str,
) -> Result<CodexCredentials, RefreshError> {
    let body = form_urlencoded::Serializer::new(String::new())
        .append_pair("grant_type", "refresh_token")
        .append_pair("client_id", CLIENT_ID)
        .append_pair("refresh_token", refresh_token)
        .finish();

    let request = HttpRequest::builder()
        .method(Method::POST)
        .uri(OPENAI_TOKEN_URL)
        .header("Content-Type", "application/x-www-form-urlencoded")
        .body(AsyncBody::from(body))
        .map_err(|e| RefreshError::Transient(e.into()))?;

    let mut response = client
        .send(request)
        .await
        .map_err(|e| RefreshError::Transient(e))?;
    let status = response.status();
    let mut body = String::new();
    smol::io::AsyncReadExt::read_to_string(response.body_mut(), &mut body)
        .await
        .map_err(|e| RefreshError::Transient(e.into()))?;

    if !status.is_success() {
        let err = anyhow!("Token refresh failed (HTTP {}): {body}", status);
        // 400/401/403 indicate a revoked or invalid refresh token.
        // 5xx and other errors are treated as transient.
        if status == http_client::StatusCode::BAD_REQUEST
            || status == http_client::StatusCode::UNAUTHORIZED
            || status == http_client::StatusCode::FORBIDDEN
        {
            return Err(RefreshError::Fatal(err));
        }
        return Err(RefreshError::Transient(err));
    }

    let tokens: TokenResponse =
        serde_json::from_str(&body).map_err(|e| RefreshError::Transient(e.into()))?;
    let jwt = tokens
        .id_token
        .as_deref()
        .unwrap_or(tokens.access_token.as_str());
    let claims = extract_jwt_claims(jwt);

    Ok(CodexCredentials {
        access_token: tokens.access_token,
        refresh_token: tokens.refresh_token,
        expires_at_ms: now_ms() + tokens.expires_in * 1000,
        account_id: claims.account_id,
        email: claims.email.or(tokens.email),
        user_id: claims.user_id,
        plan_type: claims.plan_type,
    })
}

struct JwtClaims {
    account_id: Option<String>,
    email: Option<String>,
    user_id: Option<String>,
    plan_type: Option<String>,
}

/// Extract claims from a JWT payload (base64url middle segment).
/// Extracts `chatgpt_account_id` from three possible locations (matching Roo Code's
/// implementation) and the `email` claim.
fn extract_jwt_claims(jwt: &str) -> JwtClaims {
    let Some(payload_b64) = jwt.split('.').nth(1) else {
        return JwtClaims {
            account_id: None,
            email: None,
            user_id: None,
            plan_type: None,
        };
    };
    let Ok(payload) = URL_SAFE_NO_PAD.decode(payload_b64) else {
        return JwtClaims {
            account_id: None,
            email: None,
            user_id: None,
            plan_type: None,
        };
    };
    let Ok(claims) = serde_json::from_slice::<serde_json::Value>(&payload) else {
        return JwtClaims {
            account_id: None,
            email: None,
            user_id: None,
            plan_type: None,
        };
    };

    let account_id = claims
        .get("chatgpt_account_id")
        .and_then(|v| v.as_str())
        .or_else(|| {
            claims
                .get("https://api.openai.com/auth")
                .and_then(|v| v.get("chatgpt_account_id"))
                .and_then(|v| v.as_str())
        })
        .or_else(|| {
            claims
                .get("organizations")
                .and_then(|v| v.as_array())
                .and_then(|arr| arr.first())
                .and_then(|org| org.get("id"))
                .and_then(|v| v.as_str())
        })
        .map(|s| s.to_owned());

    let email = claims
        .get("email")
        .and_then(|v| v.as_str())
        .map(|s| s.to_owned());

    let user_id = claims
        .get("sub")
        .and_then(|v| v.as_str())
        .map(str::to_owned);
    let plan_type = claims
        .get("chatgpt_plan_type")
        .and_then(|v| v.as_str())
        .or_else(|| {
            claims
                .get("https://api.openai.com/auth")
                .and_then(|v| v.get("chatgpt_plan_type"))
                .and_then(|v| v.as_str())
        })
        .map(str::to_owned);

    JwtClaims {
        account_id,
        email,
        user_id,
        plan_type,
    }
}

fn account_credentials_key(session_id: &str) -> String {
    format!("{ACCOUNT_CREDENTIALS_PREFIX}{session_id}")
}

fn session_id_for_credentials(credentials: &CodexCredentials) -> String {
    let stable_identity = credentials
        .user_id
        .as_deref()
        .or(credentials.account_id.as_deref())
        .or(credentials.email.as_deref())
        .unwrap_or(&credentials.refresh_token);
    let mut hasher = Sha256::new();
    hasher.update(stable_identity.as_bytes());
    let digest = hasher.finalize();
    digest[..16]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn upsert_manifest_session(
    manifest: &mut AccountManifest,
    session_id: &str,
    credentials: &CodexCredentials,
) {
    let existing = manifest
        .sessions
        .iter_mut()
        .find(|session| session.session_id == session_id);
    if let Some(session) = existing {
        session.email = credentials.email.clone().or(session.email.clone());
        session.user_id = credentials.user_id.clone().or(session.user_id.clone());
        session.last_used_at_ms = now_ms();
        session.token_expires_at_ms = Some(credentials.expires_at_ms);
        session.reauthentication_required = false;
        if let Some(account_id) = credentials.account_id.clone() {
            // The workspace the credential is actually scoped to is the
            // selected one, even when it differs from a previously selected
            // workspace for the same OAuth identity.
            session.selected_workspace_account_id = Some(account_id.clone());
            if let Some(workspace) = session
                .workspaces
                .iter_mut()
                .find(|workspace| workspace.account_id == account_id)
            {
                workspace.plan_type = credentials
                    .plan_type
                    .clone()
                    .or(workspace.plan_type.clone());
            } else {
                session.workspaces.push(WorkspaceMetadata {
                    account_id,
                    name: None,
                    image_url: None,
                    kind: None,
                    plan_type: credentials.plan_type.clone(),
                });
            }
        }
        return;
    }

    let workspaces = credentials
        .account_id
        .clone()
        .map(|account_id| {
            vec![WorkspaceMetadata {
                account_id,
                name: None,
                image_url: None,
                kind: None,
                plan_type: credentials.plan_type.clone(),
            }]
        })
        .unwrap_or_default();
    manifest.sessions.push(AccountSessionMetadata {
        session_id: session_id.to_string(),
        email: credentials.email.clone(),
        user_id: credentials.user_id.clone(),
        display_name: None,
        image_url: None,
        last_used_at_ms: now_ms(),
        token_expires_at_ms: Some(credentials.expires_at_ms),
        selected_workspace_account_id: credentials.account_id.clone(),
        workspaces,
        quota: None,
        quota_fetched_at_ms: None,
        reauthentication_required: false,
    });
}

#[derive(Debug)]
enum QuotaFetchError {
    Unauthorized(anyhow::Error),
    Other(anyhow::Error),
}

impl std::fmt::Display for QuotaFetchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unauthorized(error) | Self::Other(error) => write!(f, "{error:#}"),
        }
    }
}

async fn fetch_quota(
    http_client: &dyn HttpClient,
    credentials: &CodexCredentials,
) -> Result<QuotaSnapshot, QuotaFetchError> {
    let request = HttpRequest::builder()
        .method(Method::GET)
        .uri(format!("{CHATGPT_BACKEND_BASE_URL}/wham/usage"))
        .header(
            "Authorization",
            format!("Bearer {}", credentials.access_token.trim()),
        )
        .header("Accept", "application/json")
        .extra_headers(&codex_headers(credentials, None))
        .body(AsyncBody::empty())
        .map_err(|error| QuotaFetchError::Other(error.into()))?;
    let mut response = http_client
        .send(request)
        .await
        .map_err(QuotaFetchError::Other)?;
    let status = response.status();
    let mut body = String::new();
    smol::io::AsyncReadExt::read_to_string(response.body_mut(), &mut body)
        .await
        .map_err(|error| QuotaFetchError::Other(error.into()))?;
    if !status.is_success() {
        let error = anyhow!("ChatGPT usage request failed (HTTP {status}): {body}");
        if status == http_client::StatusCode::UNAUTHORIZED {
            return Err(QuotaFetchError::Unauthorized(error));
        }
        return Err(QuotaFetchError::Other(error));
    }
    let value = serde_json::from_str::<serde_json::Value>(&body)
        .context("Failed to parse ChatGPT usage response")
        .map_err(QuotaFetchError::Other)?;
    quota_snapshot_from_value(&value).map_err(QuotaFetchError::Other)
}

fn quota_snapshot_from_value(value: &serde_json::Value) -> Result<QuotaSnapshot> {
    let rate_limit = value.get("rate_limit").unwrap_or(value);
    let primary = quota_window_from_value(rate_limit.get("primary_window"));
    let secondary = quota_window_from_value(rate_limit.get("secondary_window"));
    let credits = rate_limit.get("credits").or_else(|| value.get("credits"));
    if primary.is_none() && secondary.is_none() && credits.is_none() {
        return Err(anyhow!(
            "ChatGPT usage response did not contain rate-limit or credit data"
        ));
    }
    Ok(QuotaSnapshot {
        primary,
        secondary,
        credits_has_credits: credits
            .and_then(|credits| credits.get("has_credits"))
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false),
        credits_unlimited: credits
            .and_then(|credits| credits.get("unlimited"))
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false),
        credits_balance: credits
            .and_then(|credits| credits.get("balance"))
            .map(|balance| balance.to_string().trim_matches('"').to_string()),
        limit_name: rate_limit
            .get("limit_name")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned),
        plan_type: value
            .get("plan_type")
            .or_else(|| rate_limit.get("plan_type"))
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned),
        captured_at_ms: now_ms(),
    })
}

fn quota_window_from_value(value: Option<&serde_json::Value>) -> Option<QuotaWindow> {
    let value = value?;
    let used_percent = value
        .get("used_percent")
        .and_then(serde_json::Value::as_f64)?;
    if !used_percent.is_finite() {
        return None;
    }
    let window_minutes = value
        .get("limit_window_seconds")
        .and_then(serde_json::Value::as_i64)
        .map(|seconds| seconds / 60);
    let resets_at = value
        .get("reset_at")
        .and_then(serde_json::Value::as_i64)
        .or_else(|| {
            value
                .get("reset_after_seconds")
                .and_then(serde_json::Value::as_i64)
                .map(|seconds| (now_ms() / 1000) as i64 + seconds)
        });
    Some(QuotaWindow {
        used_percent: used_percent.clamp(0.0, 100.0),
        window_minutes,
        resets_at,
    })
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or_else(|err| {
            log::error!("System clock is before UNIX epoch: {err}");
            0
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::{AppContext as _, TestAppContext};
    use http_client::FakeHttpClient;
    use parking_lot::Mutex;
    use std::collections::HashMap;
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[gpui::test]
    async fn cache_warming_scope_tracks_account_and_auth_generation(cx: &mut TestAppContext) {
        let credentials = make_fresh_credentials();
        let session_id = session_id_for_credentials(&credentials);
        let http: Arc<dyn HttpClient> = FakeHttpClient::create(|_| async {
            Ok(http_client::Response::builder()
                .status(200)
                .body(http_client::AsyncBody::default())?)
        });
        let state = make_state(http, Some(credentials), cx);
        state.update(cx, |state, _| {
            state.active_session_id = Some(session_id);
        });
        let catalog_model = ChatGptModel::fallback_models()
            .into_iter()
            .next()
            .expect("fallback catalog must contain a model");
        let model = cx.read(|cx| create_language_model(catalog_model, &state, cx));

        let initial_scope = cx
            .read(|cx| model.cache_warming_scope(cx))
            .expect("authenticated model should expose a warming scope");
        state.update(cx, |state, _| {
            state.auth_generation = state.auth_generation.wrapping_add(1);
        });
        let next_scope = cx
            .read(|cx| model.cache_warming_scope(cx))
            .expect("authenticated model should expose a warming scope");
        assert_ne!(initial_scope, next_scope);

        state.update(cx, |state, _| {
            state.account_mutation_in_progress = true;
        });
        assert!(cx.read(|cx| model.cache_warming_scope(cx)).is_none());
    }

    #[test]
    fn test_compaction_retry_policy() {
        let rate_limit = LanguageModelCompletionError::from_provider_response(
            PROVIDER_NAME,
            None,
            None,
            "too many requests".to_string(),
            Some(Duration::from_secs(7)),
            ProviderErrorCategory::RateLimit,
        );
        assert!(is_retryable_compaction_error(&rate_limit));
        assert_eq!(
            compaction_retry_delay(&rate_limit, 1),
            Duration::from_secs(7)
        );

        let server_error = LanguageModelCompletionError::from_provider_response(
            PROVIDER_NAME,
            Some(http_client::StatusCode::BAD_GATEWAY),
            None,
            "bad gateway".to_string(),
            None,
            ProviderErrorCategory::InternalServer,
        );
        assert!(is_retryable_compaction_error(&server_error));
        assert_eq!(
            compaction_retry_delay(&server_error, 1),
            Duration::from_millis(500)
        );

        assert!(is_retryable_compaction_error(
            &LanguageModelCompletionError::Other(anyhow!("connection reset by peer"))
        ));
        assert!(!is_retryable_compaction_error(
            &LanguageModelCompletionError::from_provider_response(
                PROVIDER_NAME,
                None,
                None,
                "prompt too large".to_string(),
                None,
                ProviderErrorCategory::PromptTooLarge { tokens: None },
            )
        ));
        assert!(!is_retryable_compaction_error(
            &LanguageModelCompletionError::from_provider_response(
                PROVIDER_NAME,
                None,
                None,
                "expired".to_string(),
                None,
                ProviderErrorCategory::Authentication,
            )
        ));
    }

    #[test]
    fn test_codex_headers_only_include_valid_routing_key() {
        let credentials = make_fresh_credentials();
        let has_header = |headers: &CustomHeaders, name: &str| {
            headers
                .iter()
                .any(|(header_name, _)| header_name.as_str() == name)
        };

        let without_routing_key = codex_headers(&credentials, None);
        assert!(!has_header(&without_routing_key, "session-id"));
        assert!(!has_header(&without_routing_key, "thread-id"));

        let with_empty_routing_key = codex_headers(&credentials, Some(""));
        assert!(!has_header(&with_empty_routing_key, "session-id"));
        assert!(!has_header(&with_empty_routing_key, "thread-id"));

        let with_invalid_routing_key = codex_headers(&credentials, Some("thread\n123"));
        assert!(!has_header(&with_invalid_routing_key, "session-id"));
        assert!(!has_header(&with_invalid_routing_key, "thread-id"));

        let with_routing_key = codex_headers(&credentials, Some("thread-123"));
        assert!(has_header(&with_routing_key, "session-id"));
        assert!(has_header(&with_routing_key, "thread-id"));
        assert_eq!(
            with_routing_key
                .iter()
                .find(|(name, _)| name.as_str() == "session-id")
                .and_then(|(_, value)| value.to_str().ok()),
            Some("thread-123")
        );
        assert_eq!(
            with_routing_key
                .iter()
                .find(|(name, _)| name.as_str() == "thread-id")
                .and_then(|(_, value)| value.to_str().ok()),
            Some("thread-123")
        );
    }

    #[gpui::test]
    async fn test_auto_compaction_streams_from_codex_responses_with_account_scope(
        cx: &mut TestAppContext,
    ) {
        let request_count = Arc::new(AtomicUsize::new(0));
        let request_count_for_handler = request_count.clone();
        let http_client = FakeHttpClient::create(move |request| {
            let request_count = request_count_for_handler.clone();
            async move {
                assert_eq!(
                    request.uri().to_string(),
                    "https://chatgpt.com/backend-api/codex/responses"
                );
                assert_eq!(
                    request
                        .headers()
                        .get("session-id")
                        .and_then(|value| value.to_str().ok()),
                    Some("thread-123")
                );
                assert_eq!(
                    request
                        .headers()
                        .get("thread-id")
                        .and_then(|value| value.to_str().ok()),
                    Some("thread-123")
                );
                let mut body = String::new();
                smol::io::AsyncReadExt::read_to_string(&mut request.into_body(), &mut body).await?;
                let body: serde_json::Value = serde_json::from_str(&body)?;
                assert_eq!(
                    body["context_management"],
                    serde_json::json!([{
                        "type": "compaction",
                        "compact_threshold": 100_000,
                    }])
                );
                request_count.fetch_add(1, Ordering::SeqCst);
                Ok(http_client::Response::builder()
                    .status(200)
                    .body(http_client::AsyncBody::from(compaction_response_stream()))?)
            }
        });

        let http: Arc<dyn HttpClient> = http_client;
        let mut credentials = make_fresh_credentials();
        credentials.account_id = Some("workspace-account".to_string());
        let state = make_state(http, Some(credentials), cx);
        state.update(cx, |state, _| {
            state.active_session_id = Some("session-auto".to_string());
        });
        let model = cx.read(|cx| {
            let model = ChatGptModel::fallback_models()
                .into_iter()
                .find(|model| model.id() == "gpt-5.5")
                .expect("fallback gpt-5.5 model");
            create_language_model(model, &state, cx)
        });
        assert!(model.supports_server_side_compaction());

        let request = LanguageModelRequest {
            messages: vec![language_model::LanguageModelRequestMessage {
                role: language_model::Role::User,
                content: vec![language_model::MessageContent::Text("Hello".into())],
                cache: false,
                reasoning_details: None,
            }],
            compact_at_tokens: Some(100_000),
            thread_id: Some("thread-123".to_string()),
            ..Default::default()
        };
        let events = model
            .stream_completion(request, &cx.to_async())
            .await
            .expect("the response stream should start")
            .collect::<Vec<_>>()
            .await;

        assert_eq!(request_count.load(Ordering::SeqCst), 1);
        assert!(events.iter().any(|event| {
            matches!(
                event,
                Ok(LanguageModelCompletionEvent::Compaction(
                    language_model::CompactionUpdate::Started
                ))
            )
        }));
        let state = events.iter().find_map(|event| match event {
            Ok(LanguageModelCompletionEvent::Compaction(
                language_model::CompactionUpdate::Finished(
                    language_model::CompactedContext::ProviderState(state),
                ),
            )) => Some(state),
            _ => None,
        });
        let state = state.expect("expected the streamed provider compaction state");
        assert_eq!(state.provider_id(), &PROVIDER_ID);
        assert_eq!(state.account_scope(), Some("session-auto"));
        assert_eq!(
            open_ai::responses::provider_compaction_items_with_scope(
                state,
                &PROVIDER_ID,
                Some("session-auto"),
            )
            .expect("the compacted state should parse"),
            Some(vec![serde_json::json!({
                "type": "compaction",
                "id": "cmp_1",
                "encrypted_content": "opaque-state",
            })])
        );
    }

    #[gpui::test]
    async fn test_explicit_compaction_streams_with_codex_compaction_trigger(
        cx: &mut TestAppContext,
    ) {
        let http_client = FakeHttpClient::create(move |request| async move {
            assert_eq!(
                request.uri().to_string(),
                "https://chatgpt.com/backend-api/codex/responses"
            );
            assert_eq!(
                request
                    .headers()
                    .get("session-id")
                    .and_then(|value| value.to_str().ok()),
                Some("thread-123")
            );
            assert_eq!(
                request
                    .headers()
                    .get("thread-id")
                    .and_then(|value| value.to_str().ok()),
                Some("thread-123")
            );
            let mut body = String::new();
            smol::io::AsyncReadExt::read_to_string(&mut request.into_body(), &mut body).await?;
            let body: serde_json::Value = serde_json::from_str(&body)?;
            assert!(body.get("context_management").is_none());
            assert_eq!(
                body["input"].as_array().and_then(|input| input.last()),
                Some(&serde_json::json!({"type": "compaction_trigger"}))
            );
            Ok(http_client::Response::builder()
                .status(200)
                .body(http_client::AsyncBody::from(compaction_response_stream()))?)
        });

        let http: Arc<dyn HttpClient> = http_client;
        let state = make_state(http, Some(make_fresh_credentials()), cx);
        state.update(cx, |state, _| {
            state.active_session_id = Some("session-explicit".to_string());
        });
        let model = cx.read(|cx| {
            let model = ChatGptModel::fallback_models()
                .into_iter()
                .find(|model| model.id() == "gpt-5.5")
                .expect("fallback gpt-5.5 model");
            create_language_model(model, &state, cx)
        });
        let request = LanguageModelRequest {
            messages: vec![language_model::LanguageModelRequestMessage {
                role: language_model::Role::User,
                content: vec![language_model::MessageContent::Text("Hello".into())],
                cache: false,
                reasoning_details: None,
            }],
            compact_at_tokens: Some(100_000),
            thread_id: Some("thread-123".to_string()),
            ..Default::default()
        };

        let result = model
            .compact(request, &cx.to_async())
            .await
            .expect("manual compaction should succeed");
        let language_model::CompactedContext::ProviderState(state) = result.context else {
            panic!("expected provider compaction state");
        };
        assert_eq!(state.provider_id(), &PROVIDER_ID);
        assert_eq!(state.account_scope(), Some("session-explicit"));
        assert_eq!(
            open_ai::responses::provider_compaction_items_with_scope(
                &state,
                &PROVIDER_ID,
                Some("session-explicit"),
            )
            .expect("the compacted state should parse"),
            Some(vec![serde_json::json!({
                "type": "compaction",
                "id": "cmp_1",
                "encrypted_content": "opaque-state",
            })])
        );
    }

    fn compaction_response_stream() -> String {
        let compaction_item = serde_json::json!({
            "type": "compaction",
            "id": "cmp_1",
            "encrypted_content": "opaque-state",
        });
        [
            serde_json::json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": compaction_item,
            }),
            serde_json::json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": compaction_item,
            }),
            serde_json::json!({
                "type": "response.completed",
                "response": {
                    "status": "completed",
                    "output": [],
                },
            }),
        ]
        .into_iter()
        .map(|event| format!("data: {event}\n\n"))
        .collect()
    }

    #[gpui::test]
    async fn test_concurrent_refresh_deduplicates(cx: &mut TestAppContext) {
        let refresh_count = Arc::new(AtomicUsize::new(0));
        let refresh_count_clone = refresh_count.clone();

        let http_client = FakeHttpClient::create(move |_request| {
            let refresh_count = refresh_count_clone.clone();
            async move {
                refresh_count.fetch_add(1, Ordering::SeqCst);
                let body = fake_token_response();
                Ok(http_client::Response::builder()
                    .status(200)
                    .body(http_client::AsyncBody::from(body))?)
            }
        });

        let http: Arc<dyn HttpClient> = http_client;
        let state = make_state(http.clone(), Some(make_expired_credentials()), cx);
        let session_id = session_id_for_credentials(&make_expired_credentials());
        state.update(cx, |state, _| {
            state.active_session_id = Some(session_id.clone());
        });

        let weak_state = cx.read(|_cx| state.downgrade());

        // Spawn two concurrent refresh attempts.
        let weak1 = weak_state.clone();
        let http1 = http.clone();
        let task1 =
            cx.spawn(async move |mut cx| get_fresh_credentials(&weak1, &http1, &mut cx).await);

        let weak2 = weak_state.clone();
        let http2 = http.clone();
        let task2 =
            cx.spawn(async move |mut cx| get_fresh_credentials(&weak2, &http2, &mut cx).await);

        // Drive both to completion.
        cx.run_until_parked();
        let result1 = task1.await;
        let result2 = task2.await;

        assert!(result1.is_ok(), "first refresh should succeed");
        assert!(result2.is_ok(), "second refresh should succeed");
        assert_eq!(result1.unwrap().access_token, "fresh_access");
        assert_eq!(result2.unwrap().access_token, "fresh_access");
        assert_eq!(
            refresh_count.load(Ordering::SeqCst),
            1,
            "refresh_token should only be called once despite two concurrent callers"
        );
    }

    #[gpui::test]
    async fn test_fresh_credentials_skip_refresh(cx: &mut TestAppContext) {
        let refresh_count = Arc::new(AtomicUsize::new(0));
        let refresh_count_clone = refresh_count.clone();

        let http_client = FakeHttpClient::create(move |_request| {
            let refresh_count = refresh_count_clone.clone();
            async move {
                refresh_count.fetch_add(1, Ordering::SeqCst);
                let body = fake_token_response();
                Ok(http_client::Response::builder()
                    .status(200)
                    .body(http_client::AsyncBody::from(body))?)
            }
        });

        let http: Arc<dyn HttpClient> = http_client;
        let state = make_state(http.clone(), Some(make_fresh_credentials()), cx);

        let weak_state = cx.read(|_cx| state.downgrade());

        let weak = weak_state.clone();
        let http_clone = http.clone();
        let result = cx
            .spawn(async move |mut cx| get_fresh_credentials(&weak, &http_clone, &mut cx).await)
            .await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap().access_token, "fresh_access");
        assert_eq!(
            refresh_count.load(Ordering::SeqCst),
            0,
            "no refresh should happen when credentials are fresh"
        );
    }

    #[gpui::test]
    async fn test_no_credentials_returns_no_api_key(cx: &mut TestAppContext) {
        let http_client = FakeHttpClient::create(|_| async {
            Ok(http_client::Response::builder()
                .status(200)
                .body(http_client::AsyncBody::default())?)
        });

        let http: Arc<dyn HttpClient> = http_client;
        let state = make_state(http.clone(), None, cx);

        let weak_state = cx.read(|_cx| state.downgrade());

        let weak = weak_state.clone();
        let http_clone = http.clone();
        let result = cx
            .spawn(async move |mut cx| get_fresh_credentials(&weak, &http_clone, &mut cx).await)
            .await;

        assert!(matches!(
            result,
            Err(LanguageModelCompletionError::NoApiKey { .. })
        ));
    }

    #[gpui::test]
    async fn test_fatal_refresh_marks_reauth_and_keeps_account(cx: &mut TestAppContext) {
        let http_client = FakeHttpClient::create(move |_request| async move {
            Ok(http_client::Response::builder()
                .status(401)
                .body(http_client::AsyncBody::from(r#"{"error":"invalid_grant"}"#))?)
        });

        let http: Arc<dyn HttpClient> = http_client;
        let creds_provider = Arc::new(FakeCredentialsProvider::new());
        let session_id = session_id_for_credentials(&make_expired_credentials());
        creds_provider.insert(
            &account_credentials_key(&session_id),
            "Bearer",
            serde_json::to_vec(&make_expired_credentials()).unwrap(),
        );
        let state = make_state_with_credentials_provider(
            http.clone(),
            Some(make_expired_credentials()),
            creds_provider.clone(),
            cx,
        );
        state.update(cx, |state, _| {
            state.active_session_id = Some(session_id.clone());
            state.manifest = AccountManifest {
                version: 1,
                active_session_id: Some(session_id.clone()),
                sessions: vec![AccountSessionMetadata {
                    session_id: session_id.clone(),
                    email: None,
                    user_id: None,
                    display_name: None,
                    image_url: None,
                    last_used_at_ms: now_ms(),
                    token_expires_at_ms: Some(0),
                    selected_workspace_account_id: None,
                    workspaces: Vec::new(),
                    quota: None,
                    quota_fetched_at_ms: None,
                    reauthentication_required: false,
                }],
            };
        });

        let weak_state = cx.read(|_cx| state.downgrade());

        let weak = weak_state.clone();
        let http_clone = http.clone();
        let result = cx
            .spawn(async move |mut cx| get_fresh_credentials(&weak, &http_clone, &mut cx).await)
            .await;

        cx.run_until_parked();

        assert!(result.is_err(), "fatal refresh should return an error");
        cx.read(|cx| {
            let s = state.read(cx);
            assert!(
                s.credentials.is_none(),
                "credentials should be cleared on fatal refresh failure"
            );
            assert!(
                s.last_auth_error.is_some(),
                "last_auth_error should be set on fatal refresh failure"
            );
            let session = s
                .manifest
                .sessions
                .iter()
                .find(|session| session.session_id == session_id)
                .expect("the expired account must stay in the manifest");
            assert!(
                session.reauthentication_required,
                "the expired account should be marked as needing re-authentication"
            );
            assert_eq!(
                s.active_session_id.as_deref(),
                Some(session_id.as_str()),
                "the expired account stays the active session"
            );
        });
        assert!(
            creds_provider
                .storage
                .lock()
                .contains_key(&account_credentials_key(&session_id)),
            "the keychain entry must be kept so the account can be signed back into"
        );
    }

    #[gpui::test]
    async fn test_transient_refresh_keeps_credentials(cx: &mut TestAppContext) {
        let http_client = FakeHttpClient::create(move |_request| async move {
            Ok(http_client::Response::builder()
                .status(500)
                .body(http_client::AsyncBody::from("Internal Server Error"))?)
        });

        let http: Arc<dyn HttpClient> = http_client;
        let state = make_state(http.clone(), Some(make_expired_credentials()), cx);
        let session_id = session_id_for_credentials(&make_expired_credentials());
        state.update(cx, |state, _| {
            state.active_session_id = Some(session_id.clone());
        });

        let weak_state = cx.read(|_cx| state.downgrade());

        let weak = weak_state.clone();
        let http_clone = http.clone();
        let result = cx
            .spawn(async move |mut cx| get_fresh_credentials(&weak, &http_clone, &mut cx).await)
            .await;

        cx.run_until_parked();

        assert!(result.is_err(), "transient refresh should return an error");
        cx.read(|cx| {
            let s = state.read(cx);
            assert!(
                s.credentials.is_some(),
                "credentials should be kept on transient refresh failure"
            );
            assert!(
                s.last_auth_error.is_none(),
                "last_auth_error should not be set on transient refresh failure"
            );
        });
    }

    #[gpui::test]
    async fn test_sign_out_during_refresh_discards_result(cx: &mut TestAppContext) {
        let (gate_tx, gate_rx) = futures::channel::oneshot::channel::<()>();
        let gate_rx = Arc::new(Mutex::new(Some(gate_rx)));
        let gate_rx_clone = gate_rx.clone();

        let http_client = FakeHttpClient::create(move |_request| {
            let gate_rx = gate_rx_clone.clone();
            async move {
                // Wait until the gate is opened, simulating a slow network.
                let rx = gate_rx.lock().take();
                if let Some(rx) = rx {
                    let _ = rx.await;
                }
                let body = fake_token_response();
                Ok(http_client::Response::builder()
                    .status(200)
                    .body(http_client::AsyncBody::from(body))?)
            }
        });

        let http: Arc<dyn HttpClient> = http_client;
        let state = make_state(http.clone(), Some(make_expired_credentials()), cx);
        let session_id = session_id_for_credentials(&make_expired_credentials());
        state.update(cx, |state, _| {
            state.active_session_id = Some(session_id.clone());
        });

        let weak_state = cx.read(|_cx| state.downgrade());

        // Start a refresh
        let weak = weak_state.clone();
        let http_clone = http.clone();
        let refresh_task =
            cx.spawn(async move |mut cx| get_fresh_credentials(&weak, &http_clone, &mut cx).await);

        cx.run_until_parked();

        // Sign out while the refresh is in-flight
        state.update(cx, |state, cx| {
            state.sign_out(cx).detach();
        });
        cx.run_until_parked();

        // Now let the refresh respond by opening the gate
        let _ = gate_tx.send(());
        cx.run_until_parked();

        let result = refresh_task.await;
        assert!(result.is_err(), "refresh should fail after sign-out");

        cx.read(|cx| {
            let s = state.read(cx);
            assert!(
                s.credentials.is_none(),
                "sign-out should have cleared credentials"
            );
        });
    }

    #[gpui::test]
    async fn test_sign_out_completes_fully(cx: &mut TestAppContext) {
        let creds = make_fresh_credentials();
        let session_id = session_id_for_credentials(&creds);
        let creds_provider = Arc::new(FakeCredentialsProvider::new());
        creds_provider.insert(
            &account_credentials_key(&session_id),
            "Bearer",
            serde_json::to_vec(&creds).unwrap(),
        );
        creds_provider.insert(
            ACCOUNT_MANIFEST_KEY,
            "json",
            serde_json::to_vec(&AccountManifest {
                version: 1,
                active_session_id: Some(session_id.clone()),
                sessions: vec![AccountSessionMetadata {
                    session_id: session_id.clone(),
                    email: None,
                    user_id: None,
                    display_name: None,
                    image_url: None,
                    last_used_at_ms: now_ms(),
                    token_expires_at_ms: Some(creds.expires_at_ms),
                    selected_workspace_account_id: None,
                    workspaces: Vec::new(),
                    quota: None,
                    quota_fetched_at_ms: None,
                    reauthentication_required: false,
                }],
            })
            .unwrap(),
        );

        let http: Arc<dyn HttpClient> = FakeHttpClient::create(|_| async {
            Ok(http_client::Response::builder()
                .status(200)
                .body(http_client::AsyncBody::default())?)
        });
        let state = make_state_with_credentials_provider(
            http,
            Some(make_fresh_credentials()),
            creds_provider.clone(),
            cx,
        );
        state.update(cx, |state, _| {
            state.active_session_id = Some(session_id.clone());
        });

        let sign_out_task = state.update(cx, |state, cx| state.sign_out(cx));

        cx.run_until_parked();
        sign_out_task.await.expect("sign-out should succeed");

        assert!(
            creds_provider.storage.lock().is_empty(),
            "credential store should be empty after sign-out"
        );
        cx.read(|cx| {
            assert!(
                !state.read(cx).is_authenticated(),
                "state should show not authenticated"
            );
        });
    }

    #[gpui::test]
    async fn test_sign_out_blocks_new_account_mutations(cx: &mut TestAppContext) {
        let credentials = make_fresh_credentials();
        let session_id = session_id_for_credentials(&credentials);
        let manifest = AccountManifest {
            version: 1,
            active_session_id: Some(session_id.clone()),
            sessions: vec![session_metadata(&session_id, now_ms(), None)],
        };
        let credentials_provider = Arc::new(FakeCredentialsProvider::new());
        credentials_provider.insert(
            &account_credentials_key(&session_id),
            "Bearer",
            serde_json::to_vec(&credentials).unwrap(),
        );
        credentials_provider.insert(
            ACCOUNT_MANIFEST_KEY,
            "json",
            serde_json::to_vec(&manifest).unwrap(),
        );
        let gate_tx = credentials_provider.gate_reads_for(ACCOUNT_MANIFEST_KEY);
        let http: Arc<dyn HttpClient> = FakeHttpClient::create(|_| async {
            Ok(http_client::Response::builder()
                .status(200)
                .body(http_client::AsyncBody::default())?)
        });
        let state =
            make_state_with_credentials_provider(http, Some(credentials), credentials_provider, cx);
        state.update(cx, |state, _| {
            state.manifest = manifest;
            state.active_session_id = Some(session_id);
        });

        let sign_out_task = state.update(cx, |state, cx| state.sign_out(cx));
        cx.run_until_parked();
        let sign_in_task = state.update(cx, |state, cx| state.sign_in(cx));
        assert!(
            sign_in_task.await.is_err(),
            "sign-in must remain blocked until sign-out persistence completes"
        );

        gate_tx.send(()).expect("release manifest read");
        cx.run_until_parked();
        sign_out_task.await.expect("sign-out should succeed");
    }

    #[gpui::test]
    async fn test_sign_out_manifest_read_error_restores_active_account(cx: &mut TestAppContext) {
        let credentials = make_fresh_credentials();
        let session_id = session_id_for_credentials(&credentials);
        let manifest = AccountManifest {
            version: 1,
            active_session_id: Some(session_id.clone()),
            sessions: vec![session_metadata(&session_id, now_ms(), None)],
        };
        let credentials_provider = Arc::new(FakeCredentialsProvider::new());
        credentials_provider.insert(
            &account_credentials_key(&session_id),
            "Bearer",
            serde_json::to_vec(&credentials).unwrap(),
        );
        credentials_provider.insert(
            ACCOUNT_MANIFEST_KEY,
            "json",
            serde_json::to_vec(&manifest).unwrap(),
        );
        credentials_provider.fail_reads_for(ACCOUNT_MANIFEST_KEY);
        let http: Arc<dyn HttpClient> = FakeHttpClient::create(|_| async {
            Ok(http_client::Response::builder()
                .status(200)
                .body(http_client::AsyncBody::default())?)
        });
        let state = make_state_with_credentials_provider(
            http,
            Some(credentials),
            credentials_provider.clone(),
            cx,
        );
        state.update(cx, |state, _| {
            state.manifest = manifest;
            state.active_session_id = Some(session_id.clone());
        });

        let sign_out_task = state.update(cx, |state, cx| state.sign_out(cx));
        cx.run_until_parked();
        assert!(sign_out_task.await.is_err());
        cx.read(|cx| {
            let state = state.read(cx);
            assert!(state.is_authenticated());
            assert_eq!(
                state.active_session_id.as_deref(),
                Some(session_id.as_str())
            );
            assert!(!state.is_busy());
        });
        assert!(
            credentials_provider
                .storage
                .lock()
                .contains_key(&account_credentials_key(&session_id)),
            "a failed sign-out must not delete the active credential"
        );
    }

    #[gpui::test]
    async fn test_switch_manifest_write_error_keeps_current_account(cx: &mut TestAppContext) {
        let account_a = make_credentials("a");
        let account_b = make_credentials("b");
        let account_a_id = session_id_for_credentials(&account_a);
        let account_b_id = session_id_for_credentials(&account_b);
        let manifest = AccountManifest {
            version: 1,
            active_session_id: Some(account_a_id.clone()),
            sessions: vec![
                session_metadata(&account_a_id, 2, None),
                session_metadata(&account_b_id, 1, None),
            ],
        };
        let credentials_provider = Arc::new(FakeCredentialsProvider::new());
        credentials_provider.insert(
            &account_credentials_key(&account_b_id),
            "Bearer",
            serde_json::to_vec(&account_b).unwrap(),
        );
        credentials_provider.insert(
            ACCOUNT_MANIFEST_KEY,
            "json",
            serde_json::to_vec(&manifest).unwrap(),
        );
        credentials_provider.fail_writes_for(ACCOUNT_MANIFEST_KEY);
        let http: Arc<dyn HttpClient> = FakeHttpClient::create(|_| async {
            Ok(http_client::Response::builder()
                .status(200)
                .body(http_client::AsyncBody::default())?)
        });
        let state = make_state_with_credentials_provider(
            http,
            Some(account_a.clone()),
            credentials_provider,
            cx,
        );
        state.update(cx, |state, _| {
            state.manifest = manifest;
            state.active_session_id = Some(account_a_id.clone());
        });

        let switch_task = state.update(cx, |state, cx| {
            state.switch_account(account_b_id.into(), cx)
        });
        cx.run_until_parked();
        assert!(switch_task.await.is_err());
        cx.read(|cx| {
            let state = state.read(cx);
            assert_eq!(
                state.active_session_id.as_deref(),
                Some(account_a_id.as_str())
            );
            assert_eq!(
                state
                    .credentials
                    .as_ref()
                    .map(|value| value.access_token.as_str()),
                Some(account_a.access_token.as_str())
            );
            assert!(!state.is_busy());
        });
    }

    #[gpui::test]
    async fn test_sign_out_does_not_activate_account_without_credentials(cx: &mut TestAppContext) {
        let account_a = make_credentials("a");
        let account_a_id = session_id_for_credentials(&account_a);
        let missing_account_id = "missing-account".to_string();
        let manifest = AccountManifest {
            version: 1,
            active_session_id: Some(account_a_id.clone()),
            sessions: vec![
                session_metadata(&account_a_id, 1, None),
                session_metadata(&missing_account_id, 2, None),
            ],
        };
        let credentials_provider = Arc::new(FakeCredentialsProvider::new());
        credentials_provider.insert(
            &account_credentials_key(&account_a_id),
            "Bearer",
            serde_json::to_vec(&account_a).unwrap(),
        );
        credentials_provider.insert(
            ACCOUNT_MANIFEST_KEY,
            "json",
            serde_json::to_vec(&manifest).unwrap(),
        );
        let http: Arc<dyn HttpClient> = FakeHttpClient::create(|_| async {
            Ok(http_client::Response::builder()
                .status(200)
                .body(http_client::AsyncBody::default())?)
        });
        let state =
            make_state_with_credentials_provider(http, Some(account_a), credentials_provider, cx);
        state.update(cx, |state, _| {
            state.manifest = manifest;
            state.active_session_id = Some(account_a_id);
        });

        let sign_out_task = state.update(cx, |state, cx| state.sign_out(cx));
        cx.run_until_parked();
        sign_out_task.await.expect("sign-out should succeed");
        cx.read(|cx| {
            let state = state.read(cx);
            assert_eq!(state.active_session_id, None);
            assert!(state.credentials.is_none());
        });
        let retry = state.update(cx, |state, cx| {
            state.switch_account(missing_account_id.into(), cx)
        });
        assert!(
            retry.await.is_err(),
            "switch must retry the missing credential"
        );
    }

    #[gpui::test]
    async fn test_sign_out_surfaces_credential_cleanup_failure(cx: &mut TestAppContext) {
        let credentials = make_fresh_credentials();
        let session_id = session_id_for_credentials(&credentials);
        let credential_key = account_credentials_key(&session_id);
        let manifest = AccountManifest {
            version: 1,
            active_session_id: Some(session_id.clone()),
            sessions: vec![session_metadata(&session_id, now_ms(), None)],
        };
        let credentials_provider = Arc::new(FakeCredentialsProvider::new());
        credentials_provider.insert(
            &credential_key,
            "Bearer",
            serde_json::to_vec(&credentials).unwrap(),
        );
        credentials_provider.insert(
            ACCOUNT_MANIFEST_KEY,
            "json",
            serde_json::to_vec(&manifest).unwrap(),
        );
        credentials_provider.fail_deletes_for(&credential_key);
        let http: Arc<dyn HttpClient> = FakeHttpClient::create(|_| async {
            Ok(http_client::Response::builder()
                .status(200)
                .body(http_client::AsyncBody::default())?)
        });
        let state = make_state_with_credentials_provider(
            http,
            Some(credentials),
            credentials_provider.clone(),
            cx,
        );
        state.update(cx, |state, _| {
            state.manifest = manifest;
            state.active_session_id = Some(session_id);
        });

        let sign_out_task = state.update(cx, |state, cx| state.sign_out(cx));
        cx.run_until_parked();
        assert!(sign_out_task.await.is_err());
        cx.read(|cx| {
            let state = state.read(cx);
            assert!(!state.is_authenticated());
            assert!(state.last_auth_error.is_some());
        });
        assert!(
            credentials_provider
                .storage
                .lock()
                .contains_key(&credential_key)
        );
    }

    #[gpui::test]
    async fn test_cancel_sign_in_releases_account_mutation(cx: &mut TestAppContext) {
        let http: Arc<dyn HttpClient> = FakeHttpClient::create(|_| async {
            Ok(http_client::Response::builder()
                .status(200)
                .body(http_client::AsyncBody::default())?)
        });
        let state = make_state(http, None, cx);
        let (abort_handle, abort_registration) = AbortHandle::new_pair();
        state.update(cx, |state, cx| {
            state.account_mutation_in_progress = true;
            state.sign_in_abort_handle = Some(abort_handle);
            state.sign_in_task = Some(Task::ready(()));
            state.cancel_sign_in(cx);
        });

        cx.read(|cx| {
            let state = state.read(cx);
            assert!(!state.is_busy());
            assert!(!state.is_signing_in());
            assert!(!state.can_cancel_sign_in());
        });
        let aborted = Abortable::new(futures::future::pending::<()>(), abort_registration).await;
        assert!(aborted.is_err(), "the OAuth future must be aborted");
    }

    #[gpui::test]
    async fn test_initial_load_restores_persisted_credentials(cx: &mut TestAppContext) {
        let creds = make_fresh_credentials();
        let creds_json = serde_json::to_vec(&creds).unwrap();
        let creds_provider = Arc::new(FakeCredentialsProvider::new());
        // The legacy single-account key triggers the migration path, which
        // restores the account-scoped key and the manifest.
        creds_provider.insert(LEGACY_CREDENTIALS_KEY, "Bearer", creds_json);

        let http: Arc<dyn HttpClient> = FakeHttpClient::create(|_| async {
            Ok(http_client::Response::builder()
                .status(200)
                .body(http_client::AsyncBody::default())?)
        });

        let state = cx.new(|cx| State::new(http.clone(), creds_provider.clone(), cx));

        cx.read(|cx| {
            assert_eq!(
                state.read(cx).client_version,
                UNGATED_MODEL_CATALOG_CLIENT_VERSION
            );
        });

        let load_task = cx
            .read(|cx| state.read(cx).load_task())
            .expect("constructor should start the credentials load");

        cx.run_until_parked();
        load_task.await.expect("load should succeed");

        cx.read(|cx| {
            let state = state.read(cx);
            assert!(state.is_authenticated());
            assert!(state.load_task().is_none());
        });
    }

    #[gpui::test]
    async fn test_legacy_migration_deletes_legacy_key_after_success(cx: &mut TestAppContext) {
        let creds = make_fresh_credentials();
        let creds_provider = Arc::new(FakeCredentialsProvider::new());
        creds_provider.insert(
            LEGACY_CREDENTIALS_KEY,
            "Bearer",
            serde_json::to_vec(&creds).unwrap(),
        );

        let http: Arc<dyn HttpClient> = FakeHttpClient::create(|_| async {
            Ok(http_client::Response::builder()
                .status(200)
                .body(http_client::AsyncBody::default())?)
        });
        let state = cx.new(|cx| {
            State::new_with_client_version(http, creds_provider.clone(), "test".to_string(), cx)
        });
        let load_task = cx
            .read(|cx| state.read(cx).load_task())
            .expect("constructor should start the credentials load");

        cx.run_until_parked();
        load_task.await.expect("load should succeed");

        cx.read(|cx| assert!(state.read(cx).is_authenticated()));
        let storage = creds_provider.storage.lock();
        let session_id = session_id_for_credentials(&creds);
        assert!(
            storage.contains_key(&account_credentials_key(&session_id)),
            "the account-scoped key should be written"
        );
        assert!(
            storage.contains_key(ACCOUNT_MANIFEST_KEY),
            "the manifest should be written"
        );
        assert!(
            !storage.contains_key(LEGACY_CREDENTIALS_KEY),
            "the legacy key should be deleted once the migration persisted"
        );
    }

    #[gpui::test]
    async fn test_legacy_migration_keeps_legacy_when_persist_fails(cx: &mut TestAppContext) {
        let creds = make_fresh_credentials();
        let creds_provider = Arc::new(FakeCredentialsProvider::new());
        creds_provider.insert(
            LEGACY_CREDENTIALS_KEY,
            "Bearer",
            serde_json::to_vec(&creds).unwrap(),
        );
        creds_provider.fail_writes_for(ACCOUNT_MANIFEST_KEY);

        let http: Arc<dyn HttpClient> = FakeHttpClient::create(|_| async {
            Ok(http_client::Response::builder()
                .status(200)
                .body(http_client::AsyncBody::default())?)
        });
        let state = cx.new(|cx| {
            State::new_with_client_version(http, creds_provider.clone(), "test".to_string(), cx)
        });
        let load_task = cx
            .read(|cx| state.read(cx).load_task())
            .expect("constructor should start the credentials load");

        cx.run_until_parked();
        load_task.await.expect("load should succeed");

        cx.read(|cx| assert!(state.read(cx).is_authenticated()));
        assert!(
            creds_provider
                .storage
                .lock()
                .contains_key(LEGACY_CREDENTIALS_KEY),
            "the legacy key must survive a failed persist so the migration retries"
        );
    }

    #[gpui::test]
    async fn test_sign_out_of_migrated_account_does_not_resurrect(cx: &mut TestAppContext) {
        let creds = make_fresh_credentials();
        let creds_provider = Arc::new(FakeCredentialsProvider::new());
        creds_provider.insert(
            LEGACY_CREDENTIALS_KEY,
            "Bearer",
            serde_json::to_vec(&creds).unwrap(),
        );

        let http: Arc<dyn HttpClient> = FakeHttpClient::create(|_| async {
            Ok(http_client::Response::builder()
                .status(200)
                .body(http_client::AsyncBody::default())?)
        });
        let state = cx.new(|cx| {
            State::new_with_client_version(
                http.clone(),
                creds_provider.clone(),
                "test".to_string(),
                cx,
            )
        });
        let load_task = cx
            .read(|cx| state.read(cx).load_task())
            .expect("constructor should start the credentials load");
        cx.run_until_parked();
        load_task.await.expect("load should succeed");

        let sign_out_task = state.update(cx, |state, cx| state.sign_out(cx));
        cx.run_until_parked();
        sign_out_task.await.expect("sign-out should succeed");
        assert!(
            creds_provider.storage.lock().is_empty(),
            "sign-out should remove the migrated account everywhere, including the legacy key"
        );

        // A fresh launch must not restore the signed-out account.
        let state2 = cx
            .new(|cx| State::new_with_client_version(http, creds_provider, "test".to_string(), cx));
        let load_task2 = cx
            .read(|cx| state2.read(cx).load_task())
            .expect("constructor should start the credentials load");
        cx.run_until_parked();
        load_task2.await.expect("load should succeed");
        cx.read(|cx| {
            assert!(
                !state2.read(cx).is_authenticated(),
                "the signed-out account must not come back after a restart"
            );
        });
    }

    #[gpui::test]
    async fn test_fatal_refresh_after_switch_does_not_wipe_new_account(cx: &mut TestAppContext) {
        let (gate_tx, gate_rx) = futures::channel::oneshot::channel::<()>();
        let gate_rx = Arc::new(Mutex::new(Some(gate_rx)));
        let gate_rx_clone = gate_rx.clone();
        let http_client = FakeHttpClient::create(move |_request| {
            let gate_rx = gate_rx_clone.clone();
            async move {
                let rx = gate_rx.lock().take();
                if let Some(rx) = rx {
                    let _ = rx.await;
                }
                Ok(http_client::Response::builder()
                    .status(401)
                    .body(http_client::AsyncBody::from(r#"{"error":"invalid_grant"}"#))?)
            }
        });
        let http: Arc<dyn HttpClient> = http_client;

        let a_creds = make_expired_credentials();
        let b_creds = make_credentials("b");
        let a_id = session_id_for_credentials(&a_creds);
        let b_id = session_id_for_credentials(&b_creds);
        let creds_provider = Arc::new(FakeCredentialsProvider::new());
        creds_provider.insert(
            &account_credentials_key(&b_id),
            "Bearer",
            serde_json::to_vec(&b_creds).unwrap(),
        );
        let state = make_state_with_credentials_provider(
            http.clone(),
            Some(a_creds.clone()),
            creds_provider.clone(),
            cx,
        );
        state.update(cx, |state, _| {
            state.active_session_id = Some(a_id.clone());
            state.manifest = AccountManifest {
                version: 1,
                active_session_id: Some(a_id.clone()),
                sessions: vec![
                    session_metadata(&a_id, 2, Some(0)),
                    session_metadata(&b_id, 1, None),
                ],
            };
        });

        let weak_state = cx.read(|_cx| state.downgrade());
        let weak = weak_state.clone();
        let http_clone = http.clone();
        let refresh_task =
            cx.spawn(async move |mut cx| get_fresh_credentials(&weak, &http_clone, &mut cx).await);
        cx.run_until_parked();

        // Switch to B while A's fatal refresh is in flight.
        let switch_task = state.update(cx, |state, cx| {
            state.switch_account(b_id.clone().into(), cx)
        });
        cx.run_until_parked();
        switch_task.await.expect("switch should succeed");

        let _ = gate_tx.send(());
        cx.run_until_parked();
        assert!(
            refresh_task.await.is_err(),
            "the refresh should fail after the account changed"
        );

        cx.read(|cx| {
            let s = state.read(cx);
            assert_eq!(s.active_session_id.as_deref(), Some(b_id.as_str()));
            assert_eq!(
                s.credentials
                    .as_ref()
                    .map(|creds| creds.access_token.as_str()),
                Some("b"),
                "B's credentials must not be wiped by A's stale refresh"
            );
            let a_session = s
                .manifest
                .sessions
                .iter()
                .find(|session| session.session_id == a_id)
                .expect("A stays in the manifest");
            assert!(
                !a_session.reauthentication_required,
                "a stale fatal refresh must not mark the account it no longer belongs to"
            );
        });
        assert!(
            creds_provider
                .storage
                .lock()
                .contains_key(&account_credentials_key(&b_id)),
            "B's keychain entry must be untouched"
        );
    }

    #[gpui::test]
    async fn test_concurrent_switch_is_rejected(cx: &mut TestAppContext) {
        let http: Arc<dyn HttpClient> = FakeHttpClient::create(|_| async {
            Ok(http_client::Response::builder()
                .status(200)
                .body(http_client::AsyncBody::default())?)
        });
        let b_creds = make_credentials("b");
        let c_creds = make_credentials("c");
        let b_id = session_id_for_credentials(&b_creds);
        let c_id = session_id_for_credentials(&c_creds);
        let creds_provider = Arc::new(FakeCredentialsProvider::new());
        creds_provider.insert(
            &account_credentials_key(&b_id),
            "Bearer",
            serde_json::to_vec(&b_creds).unwrap(),
        );
        creds_provider.insert(
            &account_credentials_key(&c_id),
            "Bearer",
            serde_json::to_vec(&c_creds).unwrap(),
        );
        let state = make_state_with_credentials_provider(http, None, creds_provider.clone(), cx);
        state.update(cx, |state, _| {
            state.manifest = AccountManifest {
                version: 1,
                active_session_id: None,
                sessions: vec![
                    session_metadata(&b_id, 2, None),
                    session_metadata(&c_id, 1, None),
                ],
            };
        });

        // B's credential read is slow, so another switch must not start until
        // its manifest transaction has completed.
        let gate_tx = creds_provider.gate_reads_for(&account_credentials_key(&b_id));
        let switch_b = state.update(cx, |state, cx| {
            state.switch_account(b_id.clone().into(), cx)
        });
        cx.run_until_parked();
        let switch_c = state.update(cx, |state, cx| {
            state.switch_account(c_id.clone().into(), cx)
        });
        assert!(
            switch_c.await.is_err(),
            "concurrent switch must be rejected"
        );

        let _ = gate_tx.send(());
        cx.run_until_parked();
        switch_b.await.expect("switch to B should complete");

        cx.read(|cx| {
            let s = state.read(cx);
            assert_eq!(
                s.active_session_id.as_deref(),
                Some(b_id.as_str()),
                "the serialized switch must stay active"
            );
            assert_eq!(
                s.manifest.active_session_id.as_deref(),
                Some(b_id.as_str()),
                "the in-memory manifest must match the active account"
            );
        });
        let storage = creds_provider.storage.lock();
        let (_, manifest_bytes) = storage
            .get(ACCOUNT_MANIFEST_KEY)
            .expect("the manifest should be persisted");
        let persisted: AccountManifest = serde_json::from_slice(manifest_bytes).unwrap();
        assert_eq!(
            persisted.active_session_id.as_deref(),
            Some(b_id.as_str()),
            "the persisted manifest must agree with the completed switch"
        );
    }

    #[gpui::test]
    async fn test_busy_blocks_sign_in_and_sign_out(cx: &mut TestAppContext) {
        let http: Arc<dyn HttpClient> = FakeHttpClient::create(|_| async {
            Ok(http_client::Response::builder()
                .status(200)
                .body(http_client::AsyncBody::default())?)
        });
        let state = make_state(http, Some(make_fresh_credentials()), cx);
        state.update(cx, |state, _| {
            state
                .active_operations
                .counter
                .fetch_add(1, Ordering::Relaxed);
        });

        let sign_in_task = state.update(cx, |state, cx| state.sign_in(cx));
        assert!(
            sign_in_task.await.is_err(),
            "sign-in must be blocked while a request is active"
        );
        let sign_out_task = state.update(cx, |state, cx| state.sign_out(cx));
        assert!(
            sign_out_task.await.is_err(),
            "sign-out must be blocked while a request is active"
        );

        state.update(cx, |state, _| {
            state
                .active_operations
                .counter
                .fetch_sub(1, Ordering::Relaxed);
        });
    }

    #[gpui::test]
    async fn test_refresh_after_switch_discards_result(cx: &mut TestAppContext) {
        let (gate_tx, gate_rx) = futures::channel::oneshot::channel::<()>();
        let gate_rx = Arc::new(Mutex::new(Some(gate_rx)));
        let gate_rx_clone = gate_rx.clone();
        let http_client = FakeHttpClient::create(move |_request| {
            let gate_rx = gate_rx_clone.clone();
            async move {
                let rx = gate_rx.lock().take();
                if let Some(rx) = rx {
                    let _ = rx.await;
                }
                let body = fake_token_response();
                Ok(http_client::Response::builder()
                    .status(200)
                    .body(http_client::AsyncBody::from(body))?)
            }
        });
        let http: Arc<dyn HttpClient> = http_client;

        let a_creds = make_expired_credentials();
        let b_creds = make_credentials("b");
        let a_id = session_id_for_credentials(&a_creds);
        let b_id = session_id_for_credentials(&b_creds);
        let creds_provider = Arc::new(FakeCredentialsProvider::new());
        creds_provider.insert(
            &account_credentials_key(&a_id),
            "Bearer",
            serde_json::to_vec(&a_creds).unwrap(),
        );
        creds_provider.insert(
            &account_credentials_key(&b_id),
            "Bearer",
            serde_json::to_vec(&b_creds).unwrap(),
        );
        let state = make_state_with_credentials_provider(
            http.clone(),
            Some(a_creds.clone()),
            creds_provider.clone(),
            cx,
        );
        state.update(cx, |state, _| {
            state.active_session_id = Some(a_id.clone());
            state.manifest = AccountManifest {
                version: 1,
                active_session_id: Some(a_id.clone()),
                sessions: vec![
                    session_metadata(&a_id, 2, Some(0)),
                    session_metadata(&b_id, 1, None),
                ],
            };
        });

        let weak_state = cx.read(|_cx| state.downgrade());
        let weak = weak_state.clone();
        let http_clone = http.clone();
        let refresh_task =
            cx.spawn(async move |mut cx| get_fresh_credentials(&weak, &http_clone, &mut cx).await);
        cx.run_until_parked();

        let switch_task = state.update(cx, |state, cx| {
            state.switch_account(b_id.clone().into(), cx)
        });
        cx.run_until_parked();
        switch_task.await.expect("switch should succeed");

        let _ = gate_tx.send(());
        cx.run_until_parked();
        assert!(
            refresh_task.await.is_err(),
            "a refresh superseded by a switch must fail"
        );

        cx.read(|cx| {
            let s = state.read(cx);
            assert_eq!(s.active_session_id.as_deref(), Some(b_id.as_str()));
            assert_eq!(
                s.credentials
                    .as_ref()
                    .map(|creds| creds.access_token.as_str()),
                Some("b"),
                "B's credentials must not be replaced by A's refreshed token"
            );
        });
        let storage = creds_provider.storage.lock();
        let (_, a_bytes) = storage
            .get(&account_credentials_key(&a_id))
            .expect("A's keychain entry");
        let a_persisted: CodexCredentials = serde_json::from_slice(a_bytes).unwrap();
        assert_eq!(
            a_persisted.access_token, "old_access",
            "the stale refresh must not write A's refreshed token after the switch"
        );
    }

    #[gpui::test]
    async fn test_corrupt_manifest_skips_migration(cx: &mut TestAppContext) {
        let creds = make_fresh_credentials();
        let creds_provider = Arc::new(FakeCredentialsProvider::new());
        creds_provider.insert(ACCOUNT_MANIFEST_KEY, "json", b"not a manifest".to_vec());
        creds_provider.insert(
            LEGACY_CREDENTIALS_KEY,
            "Bearer",
            serde_json::to_vec(&creds).unwrap(),
        );

        let http: Arc<dyn HttpClient> = FakeHttpClient::create(|_| async {
            Ok(http_client::Response::builder()
                .status(200)
                .body(http_client::AsyncBody::default())?)
        });
        let state = cx.new(|cx| {
            State::new_with_client_version(http, creds_provider.clone(), "test".to_string(), cx)
        });
        let load_task = cx
            .read(|cx| state.read(cx).load_task())
            .expect("constructor should start the credentials load");
        cx.run_until_parked();
        load_task.await.expect("load should succeed");

        cx.read(|cx| {
            assert!(
                !state.read(cx).is_authenticated(),
                "a corrupt manifest must not fall back to the legacy key"
            );
        });
        let storage = creds_provider.storage.lock();
        assert!(
            storage.contains_key(LEGACY_CREDENTIALS_KEY),
            "the legacy key must be left in place"
        );
        assert_eq!(
            storage.get(ACCOUNT_MANIFEST_KEY).unwrap().1,
            b"not a manifest",
            "the corrupt manifest must not be overwritten by a default"
        );
    }

    #[gpui::test]
    async fn test_manifest_read_error_skips_migration(cx: &mut TestAppContext) {
        let creds = make_fresh_credentials();
        let creds_provider = Arc::new(FakeCredentialsProvider::new());
        creds_provider.fail_reads_for(ACCOUNT_MANIFEST_KEY);
        creds_provider.insert(
            LEGACY_CREDENTIALS_KEY,
            "Bearer",
            serde_json::to_vec(&creds).unwrap(),
        );

        let http: Arc<dyn HttpClient> = FakeHttpClient::create(|_| async {
            Ok(http_client::Response::builder()
                .status(200)
                .body(http_client::AsyncBody::default())?)
        });
        let state = cx.new(|cx| {
            State::new_with_client_version(http, creds_provider.clone(), "test".to_string(), cx)
        });
        let load_task = cx
            .read(|cx| state.read(cx).load_task())
            .expect("constructor should start the credentials load");
        cx.run_until_parked();
        load_task.await.expect("load should succeed");

        cx.read(|cx| {
            assert!(
                !state.read(cx).is_authenticated(),
                "a keychain read error must not fall back to the legacy key"
            );
        });
        assert!(
            creds_provider
                .storage
                .lock()
                .contains_key(LEGACY_CREDENTIALS_KEY),
            "the legacy key must be left in place"
        );
    }

    #[gpui::test]
    async fn test_refresh_persists_token_expiry(cx: &mut TestAppContext) {
        let http_client = FakeHttpClient::create(move |_request| async move {
            Ok(http_client::Response::builder()
                .status(200)
                .body(http_client::AsyncBody::from(fake_token_response()))?)
        });
        let http: Arc<dyn HttpClient> = http_client;
        let creds = make_expired_credentials();
        let session_id = session_id_for_credentials(&creds);
        let creds_provider = Arc::new(FakeCredentialsProvider::new());
        creds_provider.insert(
            &account_credentials_key(&session_id),
            "Bearer",
            serde_json::to_vec(&creds).unwrap(),
        );
        let state = make_state_with_credentials_provider(
            http.clone(),
            Some(creds),
            creds_provider.clone(),
            cx,
        );
        state.update(cx, |state, _| {
            state.active_session_id = Some(session_id.clone());
            state.manifest = AccountManifest {
                version: 1,
                active_session_id: Some(session_id.clone()),
                sessions: vec![session_metadata(&session_id, now_ms(), Some(0))],
            };
        });

        let weak_state = cx.read(|_cx| state.downgrade());
        let weak = weak_state.clone();
        let http_clone = http.clone();
        let result = cx
            .spawn(async move |mut cx| get_fresh_credentials(&weak, &http_clone, &mut cx).await)
            .await;
        let refreshed = result.expect("refresh should succeed");
        cx.run_until_parked();

        cx.read(|cx| {
            let s = state.read(cx);
            let session = s
                .manifest
                .sessions
                .iter()
                .find(|session| session.session_id == session_id)
                .expect("session stays in the manifest");
            assert_eq!(
                session.token_expires_at_ms,
                Some(refreshed.expires_at_ms),
                "the in-memory expiry should be updated"
            );
        });
        let storage = creds_provider.storage.lock();
        let (_, manifest_bytes) = storage
            .get(ACCOUNT_MANIFEST_KEY)
            .expect("the manifest should be persisted");
        let persisted: AccountManifest = serde_json::from_slice(manifest_bytes).unwrap();
        let session = persisted
            .sessions
            .iter()
            .find(|session| session.session_id == session_id)
            .expect("session stays in the persisted manifest");
        assert_eq!(
            session.token_expires_at_ms,
            Some(refreshed.expires_at_ms),
            "the refreshed expiry must survive a restart"
        );
    }

    #[gpui::test]
    async fn test_sign_out_before_load_completes_does_not_resurrect(cx: &mut TestAppContext) {
        let creds = make_fresh_credentials();
        let creds_provider = Arc::new(FakeCredentialsProvider::new());
        creds_provider.insert(
            LEGACY_CREDENTIALS_KEY,
            "Bearer",
            serde_json::to_vec(&creds).unwrap(),
        );
        let gate_tx = creds_provider.gate_reads_for(ACCOUNT_MANIFEST_KEY);

        let http: Arc<dyn HttpClient> = FakeHttpClient::create(|_| async {
            Ok(http_client::Response::builder()
                .status(200)
                .body(http_client::AsyncBody::default())?)
        });
        let state = cx.new(|cx| {
            State::new_with_client_version(
                http.clone(),
                creds_provider.clone(),
                "test".to_string(),
                cx,
            )
        });

        // Let the load start (and capture the identity) before signing out.
        cx.run_until_parked();

        // Sign out while the initial load is still reading the manifest.
        let sign_out_task = state.update(cx, |state, cx| state.sign_out(cx));
        let _ = gate_tx.send(());
        cx.run_until_parked();
        assert!(sign_out_task.await.is_ok());
        cx.read(|cx| {
            let s = state.read(cx);
            assert!(
                !s.is_authenticated(),
                "a sign-out must not be reverted by the pending load"
            );
            assert!(s.load_task().is_none(), "the load should settle");
        });
        assert!(
            creds_provider.storage.lock().is_empty(),
            "sign-out during initial load must remove migrated credentials"
        );

        let restarted = cx
            .new(|cx| State::new_with_client_version(http, creds_provider, "test".to_string(), cx));
        let restarted_load = cx
            .read(|cx| restarted.read(cx).load_task())
            .expect("restart should load credentials");
        cx.run_until_parked();
        restarted_load.await.expect("restart load should finish");
        cx.read(|cx| {
            assert!(
                !restarted.read(cx).is_authenticated(),
                "the account must not return after restart"
            );
        });
    }

    #[gpui::test]
    async fn test_busy_transitions_notify(cx: &mut TestAppContext) {
        let http: Arc<dyn HttpClient> = FakeHttpClient::create(|_| async {
            Ok(http_client::Response::builder()
                .status(200)
                .body(http_client::AsyncBody::default())?)
        });
        let state = make_state(http, None, cx);
        let notifies = Arc::new(AtomicUsize::new(0));
        let notifies_clone = notifies.clone();
        let _subscription = cx.update(|cx| {
            cx.observe(&state, move |_, _| {
                notifies_clone.fetch_add(1, Ordering::SeqCst);
            })
        });

        let weak = cx.read(|_cx| state.downgrade());
        let async_cx = cx.to_async();

        cx.read(|cx| state.read(cx).active_operations.enter(&weak, &async_cx));
        cx.run_until_parked();
        assert_eq!(notifies.load(Ordering::SeqCst), 1, "0→1 must notify");
        cx.read(|cx| state.read(cx).active_operations.exit());
        cx.run_until_parked();
        assert_eq!(notifies.load(Ordering::SeqCst), 2, "1→0 must notify");

        // Nested operations only notify once on the transition out of idle.
        cx.read(|cx| state.read(cx).active_operations.enter(&weak, &async_cx));
        cx.read(|cx| state.read(cx).active_operations.enter(&weak, &async_cx));
        cx.run_until_parked();
        assert_eq!(notifies.load(Ordering::SeqCst), 3);
        cx.read(|cx| state.read(cx).active_operations.exit());
        cx.run_until_parked();
        assert_eq!(notifies.load(Ordering::SeqCst), 3, "2→1 must not notify");
        cx.read(|cx| state.read(cx).active_operations.exit());
        cx.run_until_parked();
        assert_eq!(notifies.load(Ordering::SeqCst), 4);

        drop(state);
        cx.run_until_parked();
    }

    #[test]
    fn test_upsert_manifest_session_updates_selected_workspace() {
        let session_id = "session";
        let mut manifest = AccountManifest {
            version: 1,
            active_session_id: Some(session_id.to_string()),
            sessions: vec![AccountSessionMetadata {
                session_id: session_id.to_string(),
                email: None,
                user_id: None,
                display_name: None,
                image_url: None,
                last_used_at_ms: 0,
                token_expires_at_ms: None,
                selected_workspace_account_id: Some("ws1".to_string()),
                workspaces: vec![WorkspaceMetadata {
                    account_id: "ws1".to_string(),
                    name: None,
                    image_url: None,
                    kind: None,
                    plan_type: None,
                }],
                quota: None,
                quota_fetched_at_ms: None,
                reauthentication_required: false,
            }],
        };
        let creds = CodexCredentials {
            access_token: "token".to_string(),
            refresh_token: "refresh".to_string(),
            expires_at_ms: now_ms() + 3_600_000,
            account_id: Some("ws2".to_string()),
            email: None,
            user_id: None,
            plan_type: Some("pro".to_string()),
        };
        upsert_manifest_session(&mut manifest, session_id, &creds);
        let session = manifest
            .sessions
            .iter()
            .find(|session| session.session_id == session_id)
            .expect("session stays");
        assert_eq!(
            session.selected_workspace_account_id.as_deref(),
            Some("ws2"),
            "the selected workspace must follow the credential's account id"
        );
        assert_eq!(session.workspaces.len(), 2);
        assert!(
            session
                .workspaces
                .iter()
                .any(|workspace| workspace.account_id == "ws2"
                    && workspace.plan_type.as_deref() == Some("pro"))
        );
    }

    fn make_credentials(access_token: &str) -> CodexCredentials {
        CodexCredentials {
            access_token: access_token.to_string(),
            refresh_token: format!("refresh_{access_token}"),
            expires_at_ms: now_ms() + 3_600_000,
            account_id: None,
            email: None,
            user_id: None,
            plan_type: None,
        }
    }

    fn session_metadata(
        session_id: &str,
        last_used_at_ms: u64,
        token_expires_at_ms: Option<u64>,
    ) -> AccountSessionMetadata {
        AccountSessionMetadata {
            session_id: session_id.to_string(),
            email: None,
            user_id: None,
            display_name: None,
            image_url: None,
            last_used_at_ms,
            token_expires_at_ms,
            selected_workspace_account_id: None,
            workspaces: Vec::new(),
            quota: None,
            quota_fetched_at_ms: None,
            reauthentication_required: false,
        }
    }

    struct FakeCredentialsProvider {
        storage: Mutex<HashMap<String, (String, Vec<u8>)>>,
        /// Keys whose reads fail with an error, simulating transient keychain
        /// failures.
        failing_reads: Mutex<Vec<String>>,
        /// Keys whose writes fail, simulating persist failures.
        failing_writes: Mutex<Vec<String>>,
        /// Keys whose deletes fail, simulating keychain cleanup failures.
        failing_deletes: Mutex<Vec<String>>,
        /// Keys whose next read waits on a oneshot so tests can control the
        /// ordering of concurrent operations.
        read_gates: Mutex<HashMap<String, futures::channel::oneshot::Receiver<()>>>,
    }

    impl FakeCredentialsProvider {
        fn new() -> Self {
            Self {
                storage: Mutex::new(HashMap::new()),
                failing_reads: Mutex::new(Vec::new()),
                failing_writes: Mutex::new(Vec::new()),
                failing_deletes: Mutex::new(Vec::new()),
                read_gates: Mutex::new(HashMap::new()),
            }
        }

        fn insert(&self, key: &str, username: &str, password: Vec<u8>) {
            self.storage
                .lock()
                .insert(key.to_string(), (username.to_string(), password));
        }

        fn fail_reads_for(&self, key: &str) {
            self.failing_reads.lock().push(key.to_string());
        }

        fn fail_writes_for(&self, key: &str) {
            self.failing_writes.lock().push(key.to_string());
        }

        fn fail_deletes_for(&self, key: &str) {
            self.failing_deletes.lock().push(key.to_string());
        }

        fn gate_reads_for(&self, key: &str) -> futures::channel::oneshot::Sender<()> {
            let (tx, rx) = futures::channel::oneshot::channel();
            self.read_gates.lock().insert(key.to_string(), rx);
            tx
        }
    }

    impl CredentialsProvider for FakeCredentialsProvider {
        fn read_credentials<'a>(
            &'a self,
            url: &'a str,
            _cx: &'a AsyncApp,
        ) -> Pin<Box<dyn Future<Output = Result<Option<(String, Vec<u8>)>>> + 'a>> {
            Box::pin(async move {
                let gate = self.read_gates.lock().remove(url);
                if let Some(gate) = gate {
                    let _ = gate.await;
                }
                if self.failing_reads.lock().iter().any(|key| key == url) {
                    return Err(anyhow!("simulated keychain read failure"));
                }
                Ok(self.storage.lock().get(url).cloned())
            })
        }

        fn write_credentials<'a>(
            &'a self,
            url: &'a str,
            username: &'a str,
            password: &'a [u8],
            _cx: &'a AsyncApp,
        ) -> Pin<Box<dyn Future<Output = Result<()>> + 'a>> {
            if self.failing_writes.lock().iter().any(|key| key == url) {
                return Box::pin(async move { Err(anyhow!("simulated keychain write failure")) });
            }
            let username = username.to_string();
            let password = password.to_vec();
            Box::pin(async move {
                self.storage
                    .lock()
                    .insert(url.to_string(), (username, password));
                Ok(())
            })
        }

        fn delete_credentials<'a>(
            &'a self,
            url: &'a str,
            _cx: &'a AsyncApp,
        ) -> Pin<Box<dyn Future<Output = Result<()>> + 'a>> {
            let url = url.to_string();
            if self.failing_deletes.lock().iter().any(|key| key == &url) {
                return Box::pin(async move { Err(anyhow!("simulated keychain delete failure")) });
            }
            Box::pin(async move {
                self.storage.lock().remove(&url);
                Ok(())
            })
        }
    }

    fn make_state(
        http_client: Arc<dyn HttpClient>,
        credentials: Option<CodexCredentials>,
        cx: &mut TestAppContext,
    ) -> Entity<State> {
        make_state_with_credentials_provider(
            http_client,
            credentials,
            Arc::new(FakeCredentialsProvider::new()),
            cx,
        )
    }

    fn make_state_with_credentials_provider(
        http_client: Arc<dyn HttpClient>,
        credentials: Option<CodexCredentials>,
        credentials_provider: Arc<dyn CredentialsProvider>,
        cx: &mut TestAppContext,
    ) -> Entity<State> {
        cx.new(|_cx| State {
            manifest: AccountManifest {
                version: 1,
                ..Default::default()
            },
            credentials,
            active_session_id: None,
            sign_in_task: None,
            sign_in_abort_handle: None,
            sign_out_task: None,
            account_mutation_in_progress: false,
            refresh_task: None,
            load_task: None,
            credentials_provider,
            http_client,
            auth_generation: 0,
            account_mutation_seq: 0,
            last_auth_error: None,
            models: ChatGptModel::fallback_models(),
            model_fetch_task: None,
            quota_refresh_task: None,
            active_operations: Arc::new(BusyNotifier::new()),
            manifest_load_state: ManifestLoadState::Loaded,
            client_version: "test".to_string(),
        })
    }

    fn make_expired_credentials() -> CodexCredentials {
        CodexCredentials {
            access_token: "old_access".to_string(),
            refresh_token: "old_refresh".to_string(),
            expires_at_ms: 0,
            account_id: None,
            email: None,
            user_id: None,
            plan_type: None,
        }
    }

    fn make_fresh_credentials() -> CodexCredentials {
        CodexCredentials {
            access_token: "fresh_access".to_string(),
            refresh_token: "fresh_refresh".to_string(),
            expires_at_ms: now_ms() + 3_600_000,
            account_id: None,
            email: None,
            user_id: None,
            plan_type: None,
        }
    }

    fn fake_token_response() -> String {
        serde_json::json!({
            "access_token": "fresh_access",
            "refresh_token": "fresh_refresh",
            "expires_in": 3600
        })
        .to_string()
    }
}
