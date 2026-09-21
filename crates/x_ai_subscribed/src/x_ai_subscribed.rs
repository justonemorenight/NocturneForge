use anyhow::{Context as _, Result, anyhow};
use base64::Engine as _;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use credentials_provider::CredentialsProvider;
use futures::{FutureExt, StreamExt, future::BoxFuture, future::Shared};
use gpui::{App, AsyncApp, Context, Entity, SharedString, Task, WeakEntity};
use http_client::{AsyncBody, CustomHeaders, HttpClient, Method, Request as HttpRequest};
use language_model::{
    LanguageModel, LanguageModelCompletionError, LanguageModelCompletionEvent,
    LanguageModelEffortLevel, LanguageModelId, LanguageModelName, LanguageModelProviderId,
    LanguageModelProviderName, LanguageModelRequest, LanguageModelToolChoice,
    ProviderErrorCategory, RateLimiter,
};
use open_ai::completion::OpenAiEventMapper;
use open_ai::{ReasoningEffort, ResponseStreamEvent};
use rand::RngCore as _;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use url::form_urlencoded;
use util::ResultExt as _;

pub const PROVIDER_ID: LanguageModelProviderId = LanguageModelProviderId::new("x_ai_subscribed");
pub const PROVIDER_NAME: LanguageModelProviderName = LanguageModelProviderName::new("SuperGrok");

pub use x_ai::XAI_API_URL;
const XAI_AUTHORIZE_URL: &str = "https://auth.x.ai/oauth2/authorize";
const XAI_TOKEN_URL: &str = "https://auth.x.ai/oauth2/token";
const CLIENT_ID: &str = "b1a00492-073a-47ea-816f-4c329264a828";
const OAUTH_SCOPE: &str = "openid profile email offline_access grok-cli:access api:access";

/// Keychain slot. Must not collide with the xAI API-key provider (`https://api.x.ai/v1`).
const CREDENTIALS_KEY: &str = "https://auth.x.ai/zed-supergrok";
const ACCOUNT_MANIFEST_KEY: &str = "https://auth.x.ai/zed-supergrok/accounts";
const ACCOUNT_CREDENTIALS_PREFIX: &str = "https://auth.x.ai/zed-supergrok/account/";
const MAX_ACCOUNT_SESSIONS: usize = 5;
const TOKEN_REFRESH_BUFFER_MS: u64 = Duration::from_secs(120).as_millis() as u64;
const QUOTA_REFRESH_INTERVAL: Duration = Duration::from_secs(60);
const QUOTA_STALE_AFTER: Duration = Duration::from_secs(15 * 60);
pub const QUOTA_EXHAUSTION_THRESHOLD_PERCENT: f64 = 100.0;
const GROK_CLI_BILLING_URL: &str = "https://cli-chat-proxy.grok.com/v1/billing?format=credits";
const GROK_CREDITS_GRPC_URL: &str =
    "https://grok.com/grok_api_v2.GrokBuildBilling/GetGrokCreditsConfig";

const CALLBACK_HOST: &str = "127.0.0.1";
const CALLBACK_PORT: u16 = 56121;
const CALLBACK_PATH: &str = "/callback";

const INFERENCE_FORBIDDEN_MESSAGE: &str = "Login succeeded, but this Grok account cannot use the API \
    (HTTP 403). Some plans do not include this access. You can also use the separate xAI provider \
    with an API key from console.x.ai.";

#[derive(Serialize, Deserialize, Clone, Debug)]
struct SuperGrokCredentials {
    access_token: String,
    refresh_token: String,
    expires_at_ms: u64,
    email: Option<String>,
    #[serde(default)]
    account_id: Option<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
struct AccountManifest {
    #[serde(default)]
    active_session_id: Option<String>,
    #[serde(default)]
    sessions: Vec<AccountSessionMetadata>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub enum AccountExclusionScope {
    Account,
    Model(String),
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub enum AccountExclusion {
    ReauthenticationRequired,
    Forbidden {
        scope: AccountExclusionScope,
    },
    RateLimited {
        retry_at_ms: u64,
        scope: AccountExclusionScope,
    },
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct AccountSessionMetadata {
    pub session_id: String,
    pub email: Option<String>,
    #[serde(default)]
    pub login_order: u32,
    pub last_used_at_ms: u64,
    #[serde(default)]
    pub reauthentication_required: bool,
    #[serde(default)]
    pub quota: Option<QuotaSnapshot>,
    #[serde(default)]
    pub quota_fetched_at_ms: Option<u64>,
    #[serde(default)]
    pub plan_type: Option<String>,
    #[serde(default)]
    pub exclusion: Option<AccountExclusion>,
    #[serde(default)]
    pub exclusions: Vec<AccountExclusion>,
}

impl AccountSessionMetadata {
    pub fn is_excluded_for(&self, model_id: &str, now_ms: u64) -> bool {
        if self.reauthentication_required {
            return true;
        }
        if let Some(quota) = &self.quota {
            if quota.used_percent >= QUOTA_EXHAUSTION_THRESHOLD_PERCENT {
                if let Some(reset_ms) = quota
                    .resets_at
                    .map(|seconds| (seconds as u64).saturating_mul(1000))
                {
                    if now_ms < reset_ms {
                        return true;
                    }
                }
            }
        }
        self.all_exclusions().into_iter().any(|ex| match ex {
            AccountExclusion::ReauthenticationRequired => self.reauthentication_required,
            AccountExclusion::Forbidden { scope } => match scope {
                AccountExclusionScope::Account => true,
                AccountExclusionScope::Model(m) => m == model_id,
            },
            AccountExclusion::RateLimited { retry_at_ms, scope } => {
                if now_ms >= retry_at_ms {
                    false
                } else {
                    match scope {
                        AccountExclusionScope::Account => true,
                        AccountExclusionScope::Model(m) => m == model_id,
                    }
                }
            }
        })
    }

    pub fn is_eligible_for(&self, model_id: &str, now_ms: u64) -> bool {
        !self.is_excluded_for(model_id, now_ms)
    }

    pub fn clean_expired_exclusions(&mut self, now_ms: u64) {
        if let Some(ex) = self.exclusion.take() {
            if !self.exclusions.contains(&ex) {
                self.exclusions.push(ex);
            }
        }
        self.exclusions.retain(|ex| match ex {
            AccountExclusion::RateLimited { retry_at_ms, .. } => *retry_at_ms > now_ms,
            _ => true,
        });
        self.exclusion = self.exclusions.last().cloned();
    }

    pub fn add_exclusion(&mut self, exclusion: AccountExclusion) {
        if let Some(ex) = self.exclusion.take() {
            if !self.exclusions.contains(&ex) {
                self.exclusions.push(ex);
            }
        }
        match &exclusion {
            AccountExclusion::Forbidden {
                scope: AccountExclusionScope::Account,
            } => {
                self.exclusions
                    .retain(|ex| !matches!(ex, AccountExclusion::Forbidden { .. }));
            }
            AccountExclusion::Forbidden {
                scope: AccountExclusionScope::Model(m),
            } => {
                self.exclusions.retain(|ex| match ex {
                    AccountExclusion::Forbidden {
                        scope: AccountExclusionScope::Model(existing),
                    } if existing == m => false,
                    _ => true,
                });
            }
            AccountExclusion::RateLimited {
                scope: AccountExclusionScope::Account,
                ..
            } => {
                self.exclusions
                    .retain(|ex| !matches!(ex, AccountExclusion::RateLimited { .. }));
            }
            AccountExclusion::RateLimited {
                scope: AccountExclusionScope::Model(m),
                ..
            } => {
                self.exclusions.retain(|ex| match ex {
                    AccountExclusion::RateLimited {
                        scope: AccountExclusionScope::Model(existing),
                        ..
                    } if existing == m => false,
                    _ => true,
                });
            }
            AccountExclusion::ReauthenticationRequired => {
                self.reauthentication_required = true;
            }
        }
        self.exclusions.push(exclusion.clone());
        self.exclusion = Some(exclusion);
    }

    pub fn clear_rate_limits(&mut self) {
        if let Some(ex) = self.exclusion.take() {
            if !self.exclusions.contains(&ex) {
                self.exclusions.push(ex);
            }
        }
        self.exclusions
            .retain(|ex| !matches!(ex, AccountExclusion::RateLimited { .. }));
        self.exclusion = self.exclusions.last().cloned();
    }

    pub fn clear_account_rate_limits(&mut self) {
        if let Some(ex) = self.exclusion.take() {
            if !self.exclusions.contains(&ex) {
                self.exclusions.push(ex);
            }
        }
        self.exclusions.retain(|ex| {
            !matches!(
                ex,
                AccountExclusion::RateLimited {
                    scope: AccountExclusionScope::Account,
                    ..
                }
            )
        });
        self.exclusion = self.exclusions.last().cloned();
    }

    pub fn clear_reauthentication_required(&mut self) {
        self.reauthentication_required = false;
        if let Some(ex) = self.exclusion.take() {
            if !self.exclusions.contains(&ex) {
                self.exclusions.push(ex);
            }
        }
        self.exclusions
            .retain(|ex| !matches!(ex, AccountExclusion::ReauthenticationRequired));
        self.exclusion = self.exclusions.last().cloned();
    }

    pub fn clear_all_exclusions(&mut self) {
        self.exclusion = None;
        self.exclusions.clear();
        self.reauthentication_required = false;
    }

    fn all_exclusions(&self) -> Vec<AccountExclusion> {
        let mut list = self.exclusions.clone();
        if let Some(ex) = &self.exclusion {
            if !list.contains(ex) {
                list.push(ex.clone());
            }
        }
        list
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct QuotaSnapshot {
    pub used_percent: f64,
    pub resets_at: Option<i64>,
    pub plan_type: Option<String>,
    pub captured_at_ms: u64,
}

#[derive(Clone, Debug)]
pub struct AccountSummary {
    pub session_id: SharedString,
    pub email: Option<SharedString>,
    pub plan_type: Option<SharedString>,
    pub quota: Option<QuotaSnapshot>,
    pub quota_stale: bool,
    pub reauthentication_required: bool,
    pub is_active: bool,
    pub is_busy: bool,
    pub exclusion: Option<AccountExclusion>,
    pub exclusions: Vec<AccountExclusion>,
}

impl AccountSummary {
    pub fn is_excluded_for(&self, model_id: &str, now_ms: u64) -> bool {
        if self.reauthentication_required {
            return true;
        }
        if let Some(quota) = &self.quota {
            if quota.used_percent >= QUOTA_EXHAUSTION_THRESHOLD_PERCENT {
                if let Some(reset_ms) = quota
                    .resets_at
                    .map(|seconds| (seconds as u64).saturating_mul(1000))
                {
                    if now_ms < reset_ms {
                        return true;
                    }
                }
            }
        }
        self.exclusions.iter().any(|ex| match ex {
            AccountExclusion::ReauthenticationRequired => self.reauthentication_required,
            AccountExclusion::Forbidden { scope } => match scope {
                AccountExclusionScope::Account => true,
                AccountExclusionScope::Model(m) => m == model_id,
            },
            AccountExclusion::RateLimited { retry_at_ms, scope } => {
                if now_ms >= *retry_at_ms {
                    false
                } else {
                    match scope {
                        AccountExclusionScope::Account => true,
                        AccountExclusionScope::Model(m) => m == model_id,
                    }
                }
            }
        })
    }

    pub fn is_eligible_for(&self, model_id: &str, now_ms: u64) -> bool {
        !self.is_excluded_for(model_id, now_ms)
    }
}

#[derive(Clone, Default)]
struct SessionInFlightTracker {
    counts: Arc<Mutex<HashMap<String, usize>>>,
}

impl SessionInFlightTracker {
    fn acquire(&self, session_id: &str) -> SessionLease {
        if let Ok(mut counts) = self.counts.lock() {
            *counts.entry(session_id.to_string()).or_insert(0) += 1;
        }
        SessionLease {
            session_id: session_id.to_string(),
            tracker: self.clone(),
        }
    }

    fn release(&self, session_id: &str) {
        if let Ok(mut counts) = self.counts.lock() {
            if let Some(count) = counts.get_mut(session_id) {
                *count = count.saturating_sub(1);
            }
        }
    }

    fn count(&self, session_id: &str) -> usize {
        self.counts
            .lock()
            .map(|counts| counts.get(session_id).copied().unwrap_or(0))
            .unwrap_or(0)
    }
}

struct SessionLease {
    session_id: String,
    tracker: SessionInFlightTracker,
}

impl Drop for SessionLease {
    fn drop(&mut self) {
        self.tracker.release(&self.session_id);
    }
}

impl SuperGrokCredentials {
    fn is_expired(&self) -> bool {
        now_ms() + TOKEN_REFRESH_BUFFER_MS >= self.expires_at_ms
    }
}

pub struct State {
    manifest: AccountManifest,
    credentials: Option<SuperGrokCredentials>,
    cached_credentials: HashMap<String, SuperGrokCredentials>,
    active_session_id: Option<String>,
    sign_in_task: Option<Task<Result<()>>>,
    refresh_task: Option<Shared<Task<Result<SuperGrokCredentials, Arc<anyhow::Error>>>>>,
    refresh_tasks: HashMap<String, Shared<Task<Result<SuperGrokCredentials, Arc<anyhow::Error>>>>>,
    load_task: Option<Shared<Task<Result<(), Arc<anyhow::Error>>>>>,
    credentials_provider: Arc<dyn CredentialsProvider>,
    http_client: Arc<dyn HttpClient>,
    auth_generation: u64,
    routing_generation: u64,
    last_auth_error: Option<SharedString>,
    active_operations: Arc<AtomicUsize>,
    in_flight_tracker: SessionInFlightTracker,
    consecutive_rate_limits: HashMap<String, u32>,
    account_mutation_in_progress: bool,
    account_mutation_seq: u64,
    quota_refresh_task: Option<Task<()>>,
}

#[derive(Debug)]
enum RefreshError {
    Fatal(anyhow::Error),
    Transient(anyhow::Error),
}

impl std::fmt::Display for RefreshError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RefreshError::Fatal(error) => write!(f, "{error}"),
            RefreshError::Transient(error) => write!(f, "{error}"),
        }
    }
}

impl State {
    pub fn new(
        http_client: Arc<dyn HttpClient>,
        credentials_provider: Arc<dyn CredentialsProvider>,
        cx: &mut Context<Self>,
    ) -> Self {
        let load_task = cx
            .spawn({
                let credentials_provider = credentials_provider.clone();
                async move |this, cx| {
                    let manifest = credentials_provider
                        .read_credentials(ACCOUNT_MANIFEST_KEY, cx)
                        .await
                        .ok()
                        .flatten()
                        .and_then(|(_, bytes)| {
                            serde_json::from_slice::<AccountManifest>(&bytes).ok()
                        });
                    let legacy = credentials_provider
                        .read_credentials(CREDENTIALS_KEY, cx)
                        .await
                        .ok()
                        .flatten()
                        .and_then(|(_, bytes)| {
                            serde_json::from_slice::<SuperGrokCredentials>(&bytes).ok()
                        });
                    let mut cached_credentials = HashMap::new();
                    let mut manifest = manifest.unwrap_or_default();
                    let mut active_credentials = None;
                    let mut active_session_id = None;
                    let mut needs_persist = false;
                    if manifest.sessions.is_empty() {
                        if let Some(mut credentials) = legacy {
                            let session_id = credentials
                                .account_id
                                .clone()
                                .unwrap_or_else(|| account_session_id(&credentials));
                            credentials.account_id = Some(session_id.clone());
                            manifest.sessions.push(AccountSessionMetadata {
                                session_id: session_id.clone(),
                                email: credentials.email.clone(),
                                login_order: 1,
                                last_used_at_ms: now_ms(),
                                reauthentication_required: false,
                                ..Default::default()
                            });
                            manifest.active_session_id = Some(session_id.clone());
                            active_session_id = Some(session_id.clone());
                            cached_credentials.insert(session_id, credentials.clone());
                            active_credentials = Some(credentials);
                            needs_persist = true;
                        }
                    } else {
                        let mut max_order = manifest
                            .sessions
                            .iter()
                            .map(|s| s.login_order)
                            .max()
                            .unwrap_or(0);
                        for session in &mut manifest.sessions {
                            if session.login_order == 0 {
                                max_order += 1;
                                session.login_order = max_order;
                                needs_persist = true;
                            }
                        }

                        let manifest_active_session_id = manifest.active_session_id.clone();
                        let mut session_ids = Vec::with_capacity(manifest.sessions.len());
                        if let Some(session_id) = manifest_active_session_id.as_ref() {
                            session_ids.push(session_id.clone());
                        }
                        let mut other_sessions = manifest
                            .sessions
                            .iter()
                            .filter(|session| {
                                Some(session.session_id.as_str())
                                    != manifest_active_session_id.as_deref()
                            })
                            .collect::<Vec<_>>();
                        other_sessions.sort_by(|left, right| {
                            right.last_used_at_ms.cmp(&left.last_used_at_ms)
                        });
                        session_ids.extend(
                            other_sessions
                                .into_iter()
                                .map(|session| session.session_id.clone()),
                        );

                        for session_id in session_ids {
                            let key = account_credentials_key(&session_id);
                            let stored_credentials = match credentials_provider
                                .read_credentials(&key, cx)
                                .await
                            {
                                Ok(stored_credentials) => stored_credentials,
                                Err(error) => {
                                    log::warn!(
                                        "Failed to read SuperGrok credentials for session {session_id}: {error:#}"
                                    );
                                    continue;
                                }
                            };
                            let Some((_, bytes)) = stored_credentials else {
                                if let Some(account) = manifest
                                    .sessions
                                    .iter_mut()
                                    .find(|account| account.session_id == session_id)
                                {
                                    if !account.reauthentication_required {
                                        account.reauthentication_required = true;
                                        needs_persist = true;
                                    }
                                }
                                continue;
                            };
                            let mut credentials = match serde_json::from_slice::<
                                SuperGrokCredentials,
                            >(&bytes)
                            {
                                Ok(credentials) => credentials,
                                Err(error) => {
                                    log::warn!(
                                        "Invalid SuperGrok credentials for session {session_id}: {error:#}"
                                    );
                                    if let Some(account) = manifest
                                        .sessions
                                        .iter_mut()
                                        .find(|account| account.session_id == session_id)
                                    {
                                        if !account.reauthentication_required {
                                            account.reauthentication_required = true;
                                            needs_persist = true;
                                        }
                                    }
                                    continue;
                                }
                            };
                            credentials.account_id = Some(session_id.clone());
                            cached_credentials.insert(session_id.clone(), credentials.clone());

                            if active_credentials.is_none() {
                                if manifest.active_session_id.as_deref() != Some(session_id.as_str()) {
                                    manifest.active_session_id = Some(session_id.clone());
                                    needs_persist = true;
                                }
                                active_session_id = Some(session_id);
                                active_credentials = Some(credentials);
                            }
                        }

                        if active_credentials.is_none() && manifest.active_session_id.take().is_some()
                        {
                            needs_persist = true;
                        }
                    }
                    let credentials_to_persist = active_credentials.clone();
                    let session_to_persist = active_session_id.clone();
                    let manifest_for_state = manifest.clone();
                    let cached_credentials_for_state = cached_credentials.clone();
                    this.update(cx, |state, cx| {
                        state.credentials = active_credentials;
                        state.cached_credentials = cached_credentials_for_state;
                        state.active_session_id = active_session_id;
                        if state.credentials.is_some() {
                            state.auth_generation = state.auth_generation.wrapping_add(1);
                        }
                        state.manifest = manifest_for_state;
                        state.load_task = None;
                        state.start_quota_monitor(cx);
                        cx.notify();
                    })?;
                    if needs_persist {
                        if let (Some(credentials), Some(session_id)) =
                            (credentials_to_persist, session_to_persist)
                        {
                            credentials_provider
                                .write_credentials(
                                    &account_credentials_key(&session_id),
                                    "Bearer",
                                    &serde_json::to_vec(&credentials)
                                        .map_err(|error| Arc::new(anyhow!(error)))?,
                                    cx,
                                )
                                .await?;
                        }
                        credentials_provider
                            .write_credentials(
                                ACCOUNT_MANIFEST_KEY,
                                "manifest",
                                &serde_json::to_vec(&manifest)
                                    .map_err(|error| Arc::new(anyhow!(error)))?,
                                cx,
                            )
                            .await?;
                    }
                    Ok::<(), Arc<anyhow::Error>>(())
                }
            })
            .shared();

        Self {
            manifest: AccountManifest::default(),
            credentials: None,
            cached_credentials: HashMap::new(),
            active_session_id: None,
            sign_in_task: None,
            refresh_task: None,
            refresh_tasks: HashMap::new(),
            load_task: Some(load_task),
            credentials_provider,
            http_client,
            auth_generation: 0,
            routing_generation: 0,
            last_auth_error: None,
            active_operations: Arc::new(AtomicUsize::new(0)),
            in_flight_tracker: SessionInFlightTracker::default(),
            consecutive_rate_limits: HashMap::new(),
            account_mutation_in_progress: false,
            account_mutation_seq: 0,
            quota_refresh_task: None,
        }
    }

    pub fn is_authenticated(&self) -> bool {
        self.credentials.is_some()
    }

    pub fn email(&self) -> Option<&str> {
        self.credentials.as_ref().and_then(|c| c.email.as_deref())
    }

    pub fn is_signing_in(&self) -> bool {
        self.sign_in_task.is_some()
    }

    pub fn last_auth_error(&self) -> Option<SharedString> {
        self.last_auth_error.clone()
    }

    pub fn load_task(&self) -> Option<Shared<Task<Result<(), Arc<anyhow::Error>>>>> {
        self.load_task.clone()
    }

    pub fn account_summaries(&self) -> Vec<AccountSummary> {
        let now = now_ms();
        let stale_after_ms = QUOTA_STALE_AFTER.as_millis() as u64;
        self.manifest
            .sessions
            .iter()
            .map(|account| AccountSummary {
                session_id: account.session_id.clone().into(),
                email: account.email.clone().map(Into::into),
                plan_type: account.plan_type.clone().map(Into::into),
                quota: account.quota.clone(),
                quota_stale: account
                    .quota_fetched_at_ms
                    .is_none_or(|fetched_at| now.saturating_sub(fetched_at) > stale_after_ms),
                reauthentication_required: account.reauthentication_required,
                is_active: self.active_session_id.as_deref() == Some(account.session_id.as_str()),
                is_busy: self.in_flight_tracker.count(&account.session_id) > 0
                    || self.account_mutation_in_progress,
                exclusion: account.exclusion.clone(),
                exclusions: account.all_exclusions(),
            })
            .collect()
    }

    pub fn eligible_accounts(&self, model_id: &str) -> Vec<AccountSessionMetadata> {
        let now = now_ms();
        let mut accounts = Vec::new();

        if let Some(active_id) = self.active_session_id.as_deref() {
            if let Some(active_session) = self
                .manifest
                .sessions
                .iter()
                .find(|s| s.session_id == active_id)
            {
                if active_session.is_eligible_for(model_id, now) {
                    accounts.push(active_session.clone());
                }
            }
        }

        let mut others: Vec<_> = self
            .manifest
            .sessions
            .iter()
            .filter(|s| {
                Some(s.session_id.as_str()) != self.active_session_id.as_deref()
                    && s.is_eligible_for(model_id, now)
            })
            .cloned()
            .collect();

        others.sort_by(|a, b| {
            a.login_order
                .cmp(&b.login_order)
                .then_with(|| a.session_id.cmp(&b.session_id))
        });

        accounts.extend(others);
        accounts
    }

    pub fn record_session_success(&mut self, session_id: &str) {
        if let Some(session) = self
            .manifest
            .sessions
            .iter_mut()
            .find(|s| s.session_id == session_id)
        {
            session.last_used_at_ms = now_ms();
        }
        self.consecutive_rate_limits.remove(session_id);
    }

    pub fn mark_reauthentication_required(&mut self, session_id: &str) {
        if let Some(session) = self
            .manifest
            .sessions
            .iter_mut()
            .find(|s| s.session_id == session_id)
        {
            session.reauthentication_required = true;
            session.add_exclusion(AccountExclusion::ReauthenticationRequired);
        }
        if self.active_session_id.as_deref() == Some(session_id) {
            self.credentials = None;
            self.last_auth_error =
                Some("Your SuperGrok session has expired. Sign in again.".into());
        }
        self.cached_credentials.remove(session_id);
        self.routing_generation = self.routing_generation.wrapping_add(1);
    }

    pub fn exclude_session(&mut self, session_id: &str, exclusion: AccountExclusion) {
        if let Some(session) = self
            .manifest
            .sessions
            .iter_mut()
            .find(|s| s.session_id == session_id)
        {
            session.add_exclusion(exclusion);
        }
        self.routing_generation = self.routing_generation.wrapping_add(1);
    }

    pub fn calculate_rate_limit_retry(
        &mut self,
        session_id: &str,
        model_id: &str,
        error: &LanguageModelCompletionError,
    ) -> (u64, AccountExclusionScope) {
        let now = now_ms();
        let consecutive = self
            .consecutive_rate_limits
            .get(session_id)
            .copied()
            .unwrap_or(0);
        self.consecutive_rate_limits
            .insert(session_id.to_string(), consecutive + 1);

        if let LanguageModelCompletionError::ProviderRejection {
            retry_after: Some(retry_after),
            ..
        } = error
        {
            let retry_at_ms = now + retry_after.as_millis() as u64;
            return (
                retry_at_ms,
                AccountExclusionScope::Model(model_id.to_string()),
            );
        }

        if let Some(session) = self
            .manifest
            .sessions
            .iter()
            .find(|s| s.session_id == session_id)
        {
            if let Some(quota) = &session.quota {
                if quota.used_percent >= QUOTA_EXHAUSTION_THRESHOLD_PERCENT {
                    if let Some(resets_at) = quota.resets_at {
                        let reset_ms = (resets_at as u64).saturating_mul(1000);
                        if reset_ms > now {
                            return (reset_ms, AccountExclusionScope::Account);
                        }
                    }
                }
            }
        }

        let cooldown_ms = match consecutive {
            0 => 30_000,
            1 => 60_000,
            _ => 300_000,
        };
        (
            now + cooldown_ms,
            AccountExclusionScope::Model(model_id.to_string()),
        )
    }

    pub fn can_cancel_sign_in(&self) -> bool {
        self.sign_in_task.is_some()
    }

    pub fn cancel_sign_in(&mut self, cx: &mut Context<Self>) {
        self.sign_in_task = None;
        cx.notify();
    }

    fn start_quota_monitor(&mut self, cx: &mut Context<Self>) {
        if self.quota_refresh_task.is_some() || self.manifest.sessions.is_empty() {
            return;
        }
        self.quota_refresh_task = Some(cx.spawn(async move |this, cx| {
            loop {
                if let Err(error) = refresh_all_account_quotas(&this, cx).await {
                    log::warn!("SuperGrok all-account quota cycle failed: {error:#}");
                }
                cx.background_executor().timer(QUOTA_REFRESH_INTERVAL).await;
            }
        }));
    }

    pub fn is_busy(&self) -> bool {
        self.active_operations.load(Ordering::Acquire) > 0 || self.account_mutation_in_progress
    }

    pub fn switch_account(
        &mut self,
        session_id: SharedString,
        cx: &mut Context<Self>,
    ) -> Task<Result<()>> {
        if self.is_busy() || self.is_signing_in() {
            return Task::ready(Err(anyhow!(
                "Cannot change SuperGrok accounts while another account operation or request is active"
            )));
        }
        let session_id = session_id.to_string();
        if self.active_session_id.as_deref() == Some(session_id.as_str()) {
            return Task::ready(Ok(()));
        }
        self.account_mutation_in_progress = true;
        self.account_mutation_seq = self.account_mutation_seq.wrapping_add(1);
        self.auth_generation = self.auth_generation.wrapping_add(1);
        let mutation_seq = self.account_mutation_seq;
        let generation = self.auth_generation;
        cx.notify();
        let credentials_provider = self.credentials_provider.clone();
        cx.spawn(async move |this, cx| {
            let key = account_credentials_key(&session_id);
            let stored = credentials_provider.read_credentials(&key, cx).await;
            let credentials = match stored {
                Ok(Some((_, bytes))) => {
                    match serde_json::from_slice::<SuperGrokCredentials>(&bytes) {
                        Ok(mut credentials) => {
                            credentials.account_id = Some(session_id.clone());
                            credentials
                        }
                        Err(e) => {
                            this.update(cx, |state, cx| {
                                if state.account_mutation_seq == mutation_seq {
                                    state.account_mutation_in_progress = false;
                                    cx.notify();
                                }
                            })
                            .ok();
                            return Err(anyhow!("Invalid SuperGrok credentials: {e:#}"));
                        }
                    }
                }
                Ok(None) => {
                    this.update(cx, |state, cx| {
                        if state.account_mutation_seq == mutation_seq {
                            state.account_mutation_in_progress = false;
                            cx.notify();
                        }
                    })
                    .ok();
                    return Err(anyhow!("SuperGrok account credentials not found"));
                }
                Err(e) => {
                    this.update(cx, |state, cx| {
                        if state.account_mutation_seq == mutation_seq {
                            state.account_mutation_in_progress = false;
                            cx.notify();
                        }
                    })
                    .ok();
                    return Err(e);
                }
            };

            let manifest_bytes = match this.read_with(cx, |state, _| {
                let mut manifest = state.manifest.clone();
                manifest.active_session_id = Some(session_id.clone());
                if let Some(account) = manifest
                    .sessions
                    .iter_mut()
                    .find(|account| account.session_id == session_id)
                {
                    account.last_used_at_ms = now_ms();
                }
                serde_json::to_vec(&manifest)
            }) {
                Ok(Ok(bytes)) => bytes,
                Ok(Err(e)) => {
                    this.update(cx, |state, cx| {
                        if state.account_mutation_seq == mutation_seq {
                            state.account_mutation_in_progress = false;
                            cx.notify();
                        }
                    })
                    .ok();
                    return Err(e.into());
                }
                Err(e) => {
                    this.update(cx, |state, cx| {
                        if state.account_mutation_seq == mutation_seq {
                            state.account_mutation_in_progress = false;
                            cx.notify();
                        }
                    })
                    .ok();
                    return Err(e);
                }
            };

            if let Err(write_err) = credentials_provider
                .write_credentials(ACCOUNT_MANIFEST_KEY, "manifest", &manifest_bytes, cx)
                .await
            {
                this.update(cx, |state, cx| {
                    if state.account_mutation_seq == mutation_seq {
                        state.account_mutation_in_progress = false;
                        cx.notify();
                    }
                })
                .ok();
                return Err(write_err);
            }

            this.update(cx, |state, cx| {
                if state.account_mutation_seq == mutation_seq {
                    state.account_mutation_in_progress = false;
                    state.manifest.active_session_id = Some(session_id.clone());
                    if let Some(account) = state
                        .manifest
                        .sessions
                        .iter_mut()
                        .find(|account| account.session_id == session_id)
                    {
                        account.last_used_at_ms = now_ms();
                    }
                    state.credentials = Some(credentials.clone());
                    state
                        .cached_credentials
                        .insert(session_id.clone(), credentials);
                    state.active_session_id = Some(session_id);
                    state.auth_generation = generation;
                    state.routing_generation = state.routing_generation.wrapping_add(1);
                    state.refresh_task = None;
                    state.start_quota_monitor(cx);
                    cx.notify();
                }
                Ok(())
            })?
        })
    }

    pub fn http_client(&self) -> Arc<dyn HttpClient> {
        self.http_client.clone()
    }

    pub fn sign_in(&mut self, cx: &mut Context<Self>) {
        if self.is_signing_in() || self.is_busy() {
            return;
        }

        self.account_mutation_in_progress = true;
        self.account_mutation_seq = self.account_mutation_seq.wrapping_add(1);
        self.auth_generation = self.auth_generation.wrapping_add(1);
        let mutation_seq = self.account_mutation_seq;
        let generation = self.auth_generation;
        let http_client = self.http_client.clone();
        let task = cx.spawn(async move |this, cx| {
            let oauth_result = do_oauth_flow(http_client, cx).await;

            match oauth_result {
                Ok(creds) => {
                    let credentials_provider =
                        this.read_with(cx, |state, _| state.credentials_provider.clone())?;
                    let mut creds = creds;
                    let session_id = creds
                        .account_id
                        .clone()
                        .unwrap_or_else(|| account_session_id(&creds));
                    creds.account_id = Some(session_id.clone());

                    let (creds_json, manifest_bytes) = match this.read_with(cx, |state, _| {
                        let has = state
                            .manifest
                            .sessions
                            .iter()
                            .any(|account| account.session_id == session_id);
                        if !has && state.manifest.sessions.len() >= MAX_ACCOUNT_SESSIONS {
                            return Err(anyhow!(
                                "Maximum of {MAX_ACCOUNT_SESSIONS} SuperGrok accounts reached"
                            ));
                        }
                        let mut manifest = state.manifest.clone();
                        if let Some(account) = manifest
                            .sessions
                            .iter_mut()
                            .find(|account| account.session_id == session_id)
                        {
                            account.email = creds.email.clone();
                            account.last_used_at_ms = now_ms();
                            account.clear_all_exclusions();
                        } else {
                            let next_order = manifest
                                .sessions
                                .iter()
                                .map(|s| s.login_order)
                                .max()
                                .unwrap_or(0)
                                + 1;
                            manifest.sessions.push(AccountSessionMetadata {
                                session_id: session_id.clone(),
                                email: creds.email.clone(),
                                login_order: next_order,
                                last_used_at_ms: now_ms(),
                                reauthentication_required: false,
                                ..Default::default()
                            });
                        }
                        manifest.active_session_id = Some(session_id.clone());
                        let creds_bytes = serde_json::to_vec(&creds)?;
                        let manifest_bytes = serde_json::to_vec(&manifest)?;
                        anyhow::Ok((creds_bytes, manifest_bytes))
                    }) {
                        Ok(Ok(pair)) => pair,
                        Ok(Err(e)) => {
                            this.update(cx, |state, cx| {
                                if state.account_mutation_seq == mutation_seq {
                                    state.account_mutation_in_progress = false;
                                    state.sign_in_task = None;
                                    state.last_auth_error = Some(SharedString::from(e.to_string()));
                                    cx.notify();
                                }
                            })
                            .ok();
                            return Err(e);
                        }
                        Err(e) => {
                            this.update(cx, |state, cx| {
                                if state.account_mutation_seq == mutation_seq {
                                    state.account_mutation_in_progress = false;
                                    state.sign_in_task = None;
                                    state.last_auth_error =
                                        Some("Failed to sign in. Please try again.".into());
                                    cx.notify();
                                }
                            })
                            .ok();
                            return Err(e);
                        }
                    };

                    let account_key = account_credentials_key(&session_id);
                    if let Err(write_err) = credentials_provider
                        .write_credentials(&account_key, "Bearer", &creds_json, cx)
                        .await
                    {
                        this.update(cx, |state, cx| {
                            if state.account_mutation_seq == mutation_seq {
                                state.account_mutation_in_progress = false;
                                state.sign_in_task = None;
                                state.last_auth_error =
                                    Some("Failed to save credentials. Please try again.".into());
                                cx.notify();
                            }
                        })
                        .ok();
                        return Err(write_err);
                    }

                    if let Err(write_err) = credentials_provider
                        .write_credentials(ACCOUNT_MANIFEST_KEY, "manifest", &manifest_bytes, cx)
                        .await
                    {
                        credentials_provider
                            .delete_credentials(&account_key, cx)
                            .await
                            .log_err();
                        this.update(cx, |state, cx| {
                            if state.account_mutation_seq == mutation_seq {
                                state.account_mutation_in_progress = false;
                                state.sign_in_task = None;
                                state.last_auth_error = Some(
                                    "Failed to save account manifest. Please try again.".into(),
                                );
                                cx.notify();
                            }
                        })
                        .ok();
                        return Err(write_err);
                    }

                    this.update(cx, |state, cx| {
                        if state.account_mutation_seq == mutation_seq {
                            state.account_mutation_in_progress = false;
                            state.auth_generation = state.auth_generation.max(generation);
                            if let Some(account) = state
                                .manifest
                                .sessions
                                .iter_mut()
                                .find(|account| account.session_id == session_id)
                            {
                                account.email = creds.email.clone();
                                account.last_used_at_ms = now_ms();
                                account.clear_all_exclusions();
                            } else {
                                let next_order = state
                                    .manifest
                                    .sessions
                                    .iter()
                                    .map(|s| s.login_order)
                                    .max()
                                    .unwrap_or(0)
                                    + 1;
                                state.manifest.sessions.push(AccountSessionMetadata {
                                    session_id: session_id.clone(),
                                    email: creds.email.clone(),
                                    login_order: next_order,
                                    last_used_at_ms: now_ms(),
                                    reauthentication_required: false,
                                    ..Default::default()
                                });
                            }
                            state.manifest.active_session_id = Some(session_id.clone());
                            state.active_session_id = Some(session_id.clone());
                            state.credentials = Some(creds.clone());
                            state.cached_credentials.insert(session_id.clone(), creds);
                            state.consecutive_rate_limits.remove(&session_id);
                            state.routing_generation = state.routing_generation.wrapping_add(1);
                            state.last_auth_error = None;
                            state.sign_in_task = None;
                            state.start_quota_monitor(cx);
                            cx.notify();
                        }
                    })?;
                    Ok(())
                }
                Err(err) => {
                    log::error!("SuperGrok sign-in failed: {err:?}");
                    this.update(cx, |state, cx| {
                        if state.account_mutation_seq == mutation_seq {
                            state.account_mutation_in_progress = false;
                            state.sign_in_task = None;
                            state.last_auth_error =
                                Some("Failed to sign in. Please try again.".into());
                            cx.notify();
                        }
                    })
                    .log_err();
                    Err(err)
                }
            }
        });

        self.last_auth_error = None;
        self.sign_in_task = Some(task);
        cx.notify();
    }

    pub fn sign_out(&mut self, cx: &mut Context<Self>) -> Task<Result<()>> {
        let Some(active_session_id) = self.active_session_id.clone() else {
            return Task::ready(Ok(()));
        };
        self.sign_out_account(active_session_id.into(), cx)
    }

    pub fn sign_out_account(
        &mut self,
        session_id: SharedString,
        cx: &mut Context<Self>,
    ) -> Task<Result<()>> {
        if self.is_busy() || self.is_signing_in() {
            return Task::ready(Err(anyhow!(
                "Cannot sign out while another account operation or request is active"
            )));
        }
        self.account_mutation_in_progress = true;
        self.account_mutation_seq = self.account_mutation_seq.wrapping_add(1);
        let mutation_seq = self.account_mutation_seq;
        let removed_session_id = session_id.to_string();
        let was_active = self.active_session_id.as_deref() == Some(removed_session_id.as_str());

        let mut manifest = self.manifest.clone();
        manifest
            .sessions
            .retain(|account| account.session_id != removed_session_id);

        let next_session_id = if was_active {
            manifest
                .sessions
                .iter()
                .max_by_key(|account| account.last_used_at_ms)
                .map(|account| account.session_id.clone())
        } else {
            manifest.active_session_id.clone()
        };
        manifest.active_session_id = next_session_id.clone();

        self.cached_credentials.remove(&removed_session_id);
        self.consecutive_rate_limits.remove(&removed_session_id);
        self.refresh_tasks.remove(&removed_session_id);
        self.routing_generation = self.routing_generation.wrapping_add(1);
        self.auth_generation = self.auth_generation.wrapping_add(1);
        if was_active {
            self.credentials = None;
            self.active_session_id = None;
            self.sign_in_task = None;
            self.refresh_task = None;
            self.last_auth_error = None;
        }
        self.manifest = manifest.clone();
        if self.manifest.sessions.is_empty() {
            self.quota_refresh_task = None;
        }
        cx.notify();

        let credentials_provider = self.credentials_provider.clone();
        cx.spawn(async move |this, cx| {
            let result = async {
                credentials_provider
                    .delete_credentials(&account_credentials_key(&removed_session_id), cx)
                    .await?;
                if was_active {
                    credentials_provider
                        .delete_credentials(CREDENTIALS_KEY, cx)
                        .await?;
                }
                if let Some(session_id) = next_session_id.clone() {
                    let credentials = if was_active {
                        let (_, bytes) = credentials_provider
                            .read_credentials(&account_credentials_key(&session_id), cx)
                            .await?
                            .ok_or_else(|| anyhow!("SuperGrok account credentials not found"))?;
                        Some(serde_json::from_slice::<SuperGrokCredentials>(&bytes)?)
                    } else {
                        None
                    };
                    credentials_provider
                        .write_credentials(
                            ACCOUNT_MANIFEST_KEY,
                            "manifest",
                            &serde_json::to_vec(&manifest)?,
                            cx,
                        )
                        .await?;
                    if was_active {
                        this.update(cx, |state, cx| {
                            state.credentials = credentials;
                            state.active_session_id = Some(session_id);
                            state.auth_generation = state.auth_generation.wrapping_add(1);
                            cx.notify();
                        })?;
                    }
                } else {
                    credentials_provider
                        .delete_credentials(ACCOUNT_MANIFEST_KEY, cx)
                        .await?;
                }
                anyhow::Ok(())
            }
            .await;

            this.update(cx, |state, cx| {
                if state.account_mutation_seq == mutation_seq {
                    state.account_mutation_in_progress = false;
                    cx.notify();
                }
                result
            })?
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SuperGrokModel {
    Grok46,
    Grok45,
    GrokBuild01,
}

impl SuperGrokModel {
    pub fn all() -> Vec<Self> {
        vec![Self::Grok46, Self::Grok45, Self::GrokBuild01]
    }

    fn x_ai_model(&self) -> Option<x_ai::Model> {
        match self {
            Self::Grok46 => Some(x_ai::Model::Grok46),
            Self::Grok45 => Some(x_ai::Model::Grok45),
            // grok-build-0.1 is SuperGrok-only; it is not in the BYOK catalog.
            Self::GrokBuild01 => None,
        }
    }

    pub fn id(&self) -> &str {
        match self {
            Self::Grok46 => "grok-4.6",
            Self::Grok45 => "grok-4.5",
            Self::GrokBuild01 => "grok-build-0.1",
        }
    }

    pub fn display_name(&self) -> &str {
        match self {
            Self::Grok46 => "Grok 4.6",
            Self::Grok45 => "Grok 4.5",
            Self::GrokBuild01 => "Grok Build 0.1",
        }
    }

    pub fn max_token_count(&self) -> u64 {
        self.x_ai_model()
            .map(|model| model.max_token_count())
            .unwrap_or(256_000)
    }

    pub fn max_output_tokens(&self) -> Option<u64> {
        self.x_ai_model()
            .map(|model| model.max_output_tokens())
            .unwrap_or(Some(64_000))
    }

    pub fn supports_images(&self) -> bool {
        self.x_ai_model()
            .map(|model| model.supports_images())
            .unwrap_or(true)
    }

    pub fn supports_tools(&self) -> bool {
        self.x_ai_model()
            .map(|model| model.supports_tool())
            .unwrap_or(true)
    }

    pub fn supports_parallel_tool_calls(&self) -> bool {
        self.x_ai_model()
            .map(|model| model.supports_parallel_tool_calls())
            .unwrap_or(true)
    }

    pub fn supports_reasoning_effort(&self) -> bool {
        self.x_ai_model()
            .map(|model| model.supports_reasoning_effort())
            .unwrap_or(false)
    }
}

pub fn create_language_model(
    model: SuperGrokModel,
    state: &Entity<State>,
    cx: &App,
) -> Arc<dyn LanguageModel> {
    let state_read = state.read(cx);
    let http_client = state_read.http_client();
    let in_flight_tracker = state_read.in_flight_tracker.clone();
    let active_operations = state_read.active_operations.clone();
    Arc::new(SuperGrokLanguageModel {
        id: LanguageModelId::from(model.id().to_string()),
        http_client,
        model,
        state: state.clone(),
        api_url: XAI_API_URL.into(),
        extra_headers: CustomHeaders::default(),
        request_limiter: RateLimiter::new(4),
        active_operations,
        in_flight_tracker,
    })
}

struct SuperGrokLanguageModel {
    id: LanguageModelId,
    model: SuperGrokModel,
    state: Entity<State>,
    http_client: Arc<dyn HttpClient>,
    api_url: SharedString,
    extra_headers: CustomHeaders,
    request_limiter: RateLimiter,
    active_operations: Arc<AtomicUsize>,
    in_flight_tracker: SessionInFlightTracker,
}

fn advertised_reasoning_efforts(model: &SuperGrokModel) -> &'static [ReasoningEffort] {
    // xAI rejects `reasoning_effort: "none"` on grok-4.5/4.6. Compact and title
    // requests disable thinking, so we omit the field instead of sending none.
    match model.x_ai_model() {
        Some(x_ai::Model::Grok45) => &[
            ReasoningEffort::Low,
            ReasoningEffort::Medium,
            ReasoningEffort::High,
        ],
        Some(x_ai::Model::Grok46) => &[
            ReasoningEffort::Low,
            ReasoningEffort::Medium,
            ReasoningEffort::High,
            ReasoningEffort::XHigh,
        ],
        _ => &[],
    }
}

fn default_thinking_reasoning_effort(model: &SuperGrokModel) -> Option<ReasoningEffort> {
    match model.x_ai_model() {
        Some(x_ai::Model::Grok45 | x_ai::Model::Grok46) => Some(ReasoningEffort::High),
        _ => None,
    }
}

fn reasoning_effort_for_request(
    request: &LanguageModelRequest,
    model: &SuperGrokModel,
) -> Option<ReasoningEffort> {
    let supported_efforts = advertised_reasoning_efforts(model);
    if supported_efforts.is_empty() {
        return None;
    }

    if request.thinking_allowed {
        request
            .thinking_effort
            .as_deref()
            .and_then(|effort| effort.parse::<ReasoningEffort>().ok())
            .filter(|effort| supported_efforts.contains(effort))
            .filter(|effort| *effort != ReasoningEffort::None)
            .or_else(|| default_thinking_reasoning_effort(model))
    } else {
        None
    }
}

fn supported_thinking_effort_levels(model: &SuperGrokModel) -> Vec<LanguageModelEffortLevel> {
    let default_effort = default_thinking_reasoning_effort(model);
    advertised_reasoning_efforts(model)
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
                ReasoningEffort::Max => return None,
            };

            Some(LanguageModelEffortLevel {
                name: name.into(),
                value: value.into(),
                is_default: Some(effort) == default_effort,
            })
        })
        .collect()
}

fn map_completion_error(error: LanguageModelCompletionError) -> LanguageModelCompletionError {
    match error {
        LanguageModelCompletionError::ProviderRejection {
            provider,
            status,
            code,
            retry_after,
            category: ProviderErrorCategory::Permission,
            ..
        } if provider == PROVIDER_NAME => LanguageModelCompletionError::ProviderRejection {
            provider,
            status,
            code,
            message: INFERENCE_FORBIDDEN_MESSAGE.to_string(),
            retry_after,
            category: ProviderErrorCategory::Permission,
        },
        other => other,
    }
}

struct ActiveOperationGuard(Arc<AtomicUsize>);

impl Drop for ActiveOperationGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

fn classify_forbidden_scope(
    error: &LanguageModelCompletionError,
    model_id: &str,
) -> AccountExclusionScope {
    let message = match error {
        LanguageModelCompletionError::ProviderRejection { message, .. } => message.as_str(),
        _ => "",
    };
    let lower = message.to_ascii_lowercase();
    let model_lower = model_id.to_ascii_lowercase();
    if lower.contains(&model_lower) || lower.contains("model") {
        AccountExclusionScope::Model(model_id.to_string())
    } else {
        AccountExclusionScope::Account
    }
}

fn is_authentication_error(error: &LanguageModelCompletionError) -> bool {
    matches!(
        error,
        LanguageModelCompletionError::ProviderRejection {
            status: Some(http_client::StatusCode::UNAUTHORIZED),
            ..
        } | LanguageModelCompletionError::ProviderRejection {
            category: ProviderErrorCategory::Authentication,
            ..
        }
    )
}

fn is_forbidden_error(error: &LanguageModelCompletionError) -> bool {
    matches!(
        error,
        LanguageModelCompletionError::ProviderRejection {
            status: Some(http_client::StatusCode::FORBIDDEN),
            ..
        } | LanguageModelCompletionError::ProviderRejection {
            category: ProviderErrorCategory::Permission,
            ..
        }
    )
}

fn is_rate_limit_error(error: &LanguageModelCompletionError) -> bool {
    matches!(
        error,
        LanguageModelCompletionError::ProviderRejection {
            status: Some(http_client::StatusCode::TOO_MANY_REQUESTS),
            ..
        } | LanguageModelCompletionError::ProviderRejection {
            category: ProviderErrorCategory::RateLimit,
            ..
        }
    )
}

fn persist_manifest_in_background(state: &WeakEntity<State>, cx: &AsyncApp) {
    let state = state.clone();
    cx.spawn(async move |cx| {
        let Some(provider) = state
            .read_with(cx, |s, _| s.credentials_provider.clone())
            .ok()
        else {
            return;
        };
        persist_manifest(&provider, &state, cx).await.log_err();
    })
    .detach();
}

impl SuperGrokLanguageModel {
    fn stream_open_ai_completion(
        &self,
        request: open_ai::Request,
        cx: &AsyncApp,
    ) -> BoxFuture<
        'static,
        Result<
            futures::stream::BoxStream<'static, Result<ResponseStreamEvent>>,
            LanguageModelCompletionError,
        >,
    > {
        let http_client = self.http_client.clone();
        let api_url = self.api_url.clone();
        let extra_headers = self.extra_headers.clone();
        let state = self.state.downgrade();
        let request_limiter = self.request_limiter.clone();
        let active_operations = self.active_operations.clone();
        let in_flight_tracker = self.in_flight_tracker.clone();
        let model_id = self.model.id().to_string();
        active_operations.fetch_add(1, Ordering::AcqRel);
        let guard = ActiveOperationGuard(active_operations);

        let future = cx.spawn(async move |cx| {
            let mut attempted_sessions = HashSet::new();

            loop {
                let eligible = state
                    .read_with(cx, |s, _| s.eligible_accounts(&model_id))
                    .map_err(LanguageModelCompletionError::Other)?;

                let candidate = eligible
                    .into_iter()
                    .find(|s| !attempted_sessions.contains(&s.session_id));

                let Some(candidate) = candidate else {
                    let no_accounts = state
                        .read_with(cx, |s, _| s.manifest.sessions.is_empty())
                        .unwrap_or(true);
                    if no_accounts {
                        return Err(LanguageModelCompletionError::NoApiKey {
                            provider: PROVIDER_NAME,
                        });
                    }

                    let fallback_err = state
                        .read_with(cx, |s, _| {
                            let now = now_ms();
                            let mut earliest_rate_limit: Option<u64> = None;
                            let mut all_reauth = true;
                            let mut all_forbidden = true;

                            for session in &s.manifest.sessions {
                                if !session.reauthentication_required {
                                    all_reauth = false;
                                }
                                let is_forbidden =
                                    session.all_exclusions().iter().any(|ex| match ex {
                                        AccountExclusion::Forbidden { scope } => match scope {
                                            AccountExclusionScope::Account => true,
                                            AccountExclusionScope::Model(m) => m == &model_id,
                                        },
                                        _ => false,
                                    });
                                if !is_forbidden {
                                    all_forbidden = false;
                                }
                                for ex in session.all_exclusions() {
                                    if let AccountExclusion::RateLimited {
                                        retry_at_ms,
                                        scope,
                                    } = ex
                                    {
                                        let applies = match scope {
                                            AccountExclusionScope::Account => true,
                                            AccountExclusionScope::Model(m) => m == model_id,
                                        };
                                        if applies && retry_at_ms > now {
                                            earliest_rate_limit = Some(
                                                earliest_rate_limit
                                                    .map(|current| current.min(retry_at_ms))
                                                    .unwrap_or(retry_at_ms),
                                            );
                                        }
                                    }
                                }
                            }

                            if let Some(retry_at) = earliest_rate_limit {
                                let retry_after =
                                    Duration::from_millis(retry_at.saturating_sub(now));
                                LanguageModelCompletionError::ProviderRejection {
                                    provider: PROVIDER_NAME,
                                    status: Some(http_client::StatusCode::TOO_MANY_REQUESTS),
                                    code: Some("rate_limit_exceeded".to_string()),
                                    message:
                                        "All SuperGrok accounts are currently rate-limited. Please try again shortly."
                                            .to_string(),
                                    retry_after: Some(retry_after),
                                    category: ProviderErrorCategory::RateLimit,
                                }
                            } else if all_reauth {
                                LanguageModelCompletionError::ProviderRejection {
                                    provider: PROVIDER_NAME,
                                    status: Some(http_client::StatusCode::UNAUTHORIZED),
                                    code: Some("reauthentication_required".to_string()),
                                    message:
                                        "All SuperGrok accounts require signing in again."
                                            .to_string(),
                                    retry_after: None,
                                    category: ProviderErrorCategory::Authentication,
                                }
                            } else if all_forbidden {
                                LanguageModelCompletionError::ProviderRejection {
                                    provider: PROVIDER_NAME,
                                    status: Some(http_client::StatusCode::FORBIDDEN),
                                    code: Some("forbidden".to_string()),
                                    message: INFERENCE_FORBIDDEN_MESSAGE.to_string(),
                                    retry_after: None,
                                    category: ProviderErrorCategory::Permission,
                                }
                            } else {
                                LanguageModelCompletionError::ProviderRejection {
                                    provider: PROVIDER_NAME,
                                    status: Some(http_client::StatusCode::SERVICE_UNAVAILABLE),
                                    code: Some("no_eligible_accounts".to_string()),
                                    message:
                                        "No eligible SuperGrok accounts available for this request."
                                            .to_string(),
                                    retry_after: None,
                                    category: ProviderErrorCategory::Other,
                                }
                            }
                        })
                        .map_err(LanguageModelCompletionError::Other)?;

                    return Err(fallback_err);
                };

                let session_id = candidate.session_id.clone();
                attempted_sessions.insert(session_id.clone());
                let lease = in_flight_tracker.acquire(&session_id);

                let credentials = match get_fresh_credentials_for_session(
                    &state,
                    &http_client,
                    &session_id,
                    cx,
                )
                .await
                {
                    Ok(creds) => creds,
                    Err(err) => {
                        drop(lease);
                        if matches!(err, LanguageModelCompletionError::NoApiKey { .. }) {
                            state
                                .update(cx, |s, cx| {
                                    s.mark_reauthentication_required(&session_id);
                                    cx.notify();
                                })
                                .ok();
                            persist_manifest_in_background(&state, cx);
                        }
                        continue;
                    }
                };

                let request_clone = request.clone();
                let access_token = credentials.access_token.clone();
                let client = http_client.clone();
                let url = api_url.clone();
                let headers = extra_headers.clone();

                let stream_result = request_limiter
                    .stream(async move {
                        open_ai::stream_completion(
                            client.as_ref(),
                            PROVIDER_NAME.0.as_str(),
                            url.as_ref(),
                            &access_token,
                            request_clone,
                            &headers,
                        )
                        .await
                        .map_err(LanguageModelCompletionError::from)
                    })
                    .await;

                match stream_result {
                    Ok(stream) => {
                        state
                            .update(cx, |s, _| {
                                s.record_session_success(&session_id);
                            })
                            .ok();
                        return Ok((stream.boxed(), guard, lease));
                    }
                    Err(completion_error) => {
                        drop(lease);

                        if is_authentication_error(&completion_error) {
                            match force_refresh_session_credentials(
                                &state,
                                &http_client,
                                &session_id,
                                cx,
                            )
                            .await
                            {
                                Ok(refreshed_creds) => {
                                    let retry_lease = in_flight_tracker.acquire(&session_id);
                                    let request_clone = request.clone();
                                    let access_token = refreshed_creds.access_token.clone();
                                    let client = http_client.clone();
                                    let url = api_url.clone();
                                    let headers = extra_headers.clone();

                                    let retry_result = request_limiter
                                        .stream(async move {
                                            open_ai::stream_completion(
                                                client.as_ref(),
                                                PROVIDER_NAME.0.as_str(),
                                                url.as_ref(),
                                                &access_token,
                                                request_clone,
                                                &headers,
                                            )
                                            .await
                                            .map_err(LanguageModelCompletionError::from)
                                        })
                                        .await;

                                    match retry_result {
                                        Ok(stream) => {
                                            state
                                                .update(cx, |s, _| {
                                                    s.record_session_success(&session_id);
                                                })
                                                .ok();
                                            return Ok((stream.boxed(), guard, retry_lease));
                                        }
                                        Err(retry_comp_err) => {
                                            drop(retry_lease);
                                            if is_authentication_error(&retry_comp_err) {
                                                state
                                                    .update(cx, |s, cx| {
                                                        s.mark_reauthentication_required(
                                                            &session_id,
                                                        );
                                                        cx.notify();
                                                    })
                                                    .ok();
                                                persist_manifest_in_background(&state, cx);
                                                continue;
                                            } else if is_forbidden_error(&retry_comp_err) {
                                                let scope = classify_forbidden_scope(
                                                    &retry_comp_err,
                                                    &model_id,
                                                );
                                                state
                                                    .update(cx, |s, cx| {
                                                        s.exclude_session(
                                                            &session_id,
                                                            AccountExclusion::Forbidden { scope },
                                                        );
                                                        cx.notify();
                                                    })
                                                    .ok();
                                                persist_manifest_in_background(&state, cx);
                                                continue;
                                            } else if is_rate_limit_error(&retry_comp_err) {
                                                state
                                                    .update(cx, |s, cx| {
                                                        let (retry_at_ms, scope) = s
                                                            .calculate_rate_limit_retry(
                                                                &session_id,
                                                                &model_id,
                                                                &retry_comp_err,
                                                            );
                                                        s.exclude_session(
                                                            &session_id,
                                                            AccountExclusion::RateLimited {
                                                                retry_at_ms,
                                                                scope,
                                                            },
                                                        );
                                                        cx.notify();
                                                    })
                                                    .ok();
                                                persist_manifest_in_background(&state, cx);
                                                continue;
                                            } else {
                                                return Err(map_completion_error(retry_comp_err));
                                            }
                                        }
                                    }
                                }
                                Err(RefreshError::Fatal(_)) => {
                                    state
                                        .update(cx, |s, cx| {
                                            s.mark_reauthentication_required(&session_id);
                                            cx.notify();
                                        })
                                        .ok();
                                    persist_manifest_in_background(&state, cx);
                                    continue;
                                }
                                Err(RefreshError::Transient(_)) => {
                                    continue;
                                }
                            }
                        } else if is_forbidden_error(&completion_error) {
                            let scope = classify_forbidden_scope(&completion_error, &model_id);
                            state
                                .update(cx, |s, cx| {
                                    s.exclude_session(
                                        &session_id,
                                        AccountExclusion::Forbidden { scope },
                                    );
                                    cx.notify();
                                })
                                .ok();
                            persist_manifest_in_background(&state, cx);
                            continue;
                        } else if is_rate_limit_error(&completion_error) {
                            state
                                .update(cx, |s, cx| {
                                    let (retry_at_ms, scope) = s.calculate_rate_limit_retry(
                                        &session_id,
                                        &model_id,
                                        &completion_error,
                                    );
                                    s.exclude_session(
                                        &session_id,
                                        AccountExclusion::RateLimited { retry_at_ms, scope },
                                    );
                                    cx.notify();
                                })
                                .ok();
                            persist_manifest_in_background(&state, cx);
                            continue;
                        } else {
                            return Err(map_completion_error(completion_error));
                        }
                    }
                }
            }
        });

        async move {
            let (stream, guard, lease) = future.await?;
            let stream = futures::stream::unfold(
                (stream, guard, lease),
                |(mut stream, guard, lease)| async move {
                    let event = stream.next().await?;
                    Some((event, (stream, guard, lease)))
                },
            );
            Ok(stream.boxed())
        }
        .boxed()
    }
}

impl LanguageModel for SuperGrokLanguageModel {
    fn cache_warming_scope(&self, cx: &App) -> Option<String> {
        let state = self.state.read(cx);
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
        self.model.supports_tools()
    }

    fn supports_images(&self) -> bool {
        self.model.supports_images()
    }

    fn supports_streaming_tools(&self) -> bool {
        true
    }

    fn supports_tool_choice(&self, choice: LanguageModelToolChoice) -> bool {
        match choice {
            LanguageModelToolChoice::Auto
            | LanguageModelToolChoice::Any
            | LanguageModelToolChoice::None => true,
        }
    }

    fn supports_thinking(&self) -> bool {
        self.model.supports_reasoning_effort()
    }

    fn supported_effort_levels(&self) -> Vec<LanguageModelEffortLevel> {
        supported_thinking_effort_levels(&self.model)
    }

    fn telemetry_id(&self) -> String {
        format!("x_ai_subscribed/{}", self.model.id())
    }

    fn max_token_count(&self) -> u64 {
        self.model.max_token_count()
    }

    fn max_output_tokens(&self) -> Option<u64> {
        self.model.max_output_tokens()
    }

    fn supports_split_token_display(&self) -> bool {
        true
    }

    fn stream_completion(
        &self,
        request: LanguageModelRequest,
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
        let reasoning_effort = reasoning_effort_for_request(&request, &self.model);
        let request = match open_ai::completion::into_open_ai(
            request,
            self.model.id(),
            self.model.supports_parallel_tool_calls(),
            false,
            self.max_output_tokens(),
            open_ai::completion::ChatCompletionMaxTokensParameter::MaxCompletionTokens,
            reasoning_effort,
            false,
        ) {
            Ok(request) => request,
            Err(error) => return async move { Err(error.into()) }.boxed(),
        };
        let completions = self.stream_open_ai_completion(request, cx);
        let executor = cx.background_executor().clone();
        async move {
            let mapper = OpenAiEventMapper::new();
            Ok(language_model::stream_in_background(
                mapper.map_stream(completions.await?).boxed(),
                executor,
            ))
        }
        .boxed()
    }
}

#[allow(dead_code)]
async fn get_fresh_credentials(
    state: &WeakEntity<State>,
    http_client: &Arc<dyn HttpClient>,
    cx: &mut AsyncApp,
) -> Result<SuperGrokCredentials, LanguageModelCompletionError> {
    let session_id = state
        .read_with(&*cx, |s, _| s.active_session_id.clone())
        .map_err(LanguageModelCompletionError::Other)?
        .ok_or(LanguageModelCompletionError::NoApiKey {
            provider: PROVIDER_NAME,
        })?;
    get_fresh_credentials_for_session(state, http_client, &session_id, cx).await
}

async fn get_fresh_credentials_for_session(
    state: &WeakEntity<State>,
    http_client: &Arc<dyn HttpClient>,
    session_id: &str,
    cx: &mut AsyncApp,
) -> Result<SuperGrokCredentials, LanguageModelCompletionError> {
    let (creds, existing_task, generation) = state
        .read_with(&*cx, |s, _| {
            (
                s.cached_credentials.get(session_id).cloned().or_else(|| {
                    if s.active_session_id.as_deref() == Some(session_id) {
                        s.credentials.clone()
                    } else {
                        None
                    }
                }),
                s.refresh_tasks.get(session_id).cloned().or_else(|| {
                    if s.active_session_id.as_deref() == Some(session_id) {
                        s.refresh_task.clone()
                    } else {
                        None
                    }
                }),
                s.auth_generation,
            )
        })
        .map_err(LanguageModelCompletionError::Other)?;

    let creds = match creds {
        Some(c) => c,
        None => {
            let provider = state
                .read_with(&*cx, |s, _| s.credentials_provider.clone())
                .map_err(LanguageModelCompletionError::Other)?;
            let key = account_credentials_key(session_id);
            let stored = provider
                .read_credentials(&key, cx)
                .await
                .map_err(LanguageModelCompletionError::Other)?;
            let Some((_, bytes)) = stored else {
                return Err(LanguageModelCompletionError::NoApiKey {
                    provider: PROVIDER_NAME,
                });
            };
            let session_is_in_manifest = state
                .read_with(&*cx, |s, _| {
                    s.manifest
                        .sessions
                        .iter()
                        .any(|sess| sess.session_id == session_id)
                })
                .map_err(LanguageModelCompletionError::Other)?;
            if !session_is_in_manifest {
                return Err(LanguageModelCompletionError::NoApiKey {
                    provider: PROVIDER_NAME,
                });
            }
            let loaded: SuperGrokCredentials = serde_json::from_slice(&bytes)
                .map_err(|e| LanguageModelCompletionError::Other(e.into()))?;
            state
                .update(cx, |s, _| {
                    s.cached_credentials
                        .insert(session_id.to_string(), loaded.clone());
                })
                .ok();
            loaded
        }
    };

    if !creds.is_expired() {
        return Ok(creds);
    }

    if let Some(shared_task) = existing_task {
        return shared_task
            .await
            .map_err(|e| LanguageModelCompletionError::Other(anyhow!("{e}")));
    }

    perform_token_refresh(state, http_client, session_id, &creds, generation, cx).await
}

async fn perform_token_refresh(
    state: &WeakEntity<State>,
    http_client: &Arc<dyn HttpClient>,
    session_id: &str,
    creds: &SuperGrokCredentials,
    generation: u64,
    cx: &mut AsyncApp,
) -> Result<SuperGrokCredentials, LanguageModelCompletionError> {
    let http_client_clone = http_client.clone();
    let state_clone = state.clone();
    let previous_refresh_token = creds.refresh_token.clone();
    let previous_email = creds.email.clone();
    let session_id_str = session_id.to_string();

    let shared_task = cx
        .spawn(async move |cx| {
            let result = refresh_token(&http_client_clone, &previous_refresh_token).await;

            match result {
                Ok(tokens) => {
                    let persist_result: Result<SuperGrokCredentials, Arc<anyhow::Error>> = async {
                        let (current_generation, session_still_present) = state_clone
                            .read_with(&*cx, |s, _| {
                                (
                                    s.auth_generation,
                                    s.manifest
                                        .sessions
                                        .iter()
                                        .any(|sess| sess.session_id == session_id_str),
                                )
                            })
                            .map_err(|e| Arc::new(e))?;
                        if current_generation != generation || !session_still_present {
                            return Err(Arc::new(anyhow!(
                                "Sign-out occurred during token refresh"
                            )));
                        }

                        let claims = tokens
                            .id_token
                            .as_deref()
                            .map(extract_email_claim)
                            .unwrap_or(None);
                        let refreshed = SuperGrokCredentials {
                            access_token: tokens.access_token,
                            refresh_token: tokens
                                .refresh_token
                                .unwrap_or(previous_refresh_token.clone()),
                            expires_at_ms: now_ms() + tokens.expires_in * 1000,
                            email: claims.or(tokens.email).or(previous_email.clone()),
                            account_id: Some(session_id_str.clone()),
                        };

                        let credentials_provider = state_clone
                            .read_with(&*cx, |s, _| s.credentials_provider.clone())
                            .map_err(|e| Arc::new(e))?;

                        let json =
                            serde_json::to_vec(&refreshed).map_err(|e| Arc::new(e.into()))?;

                        credentials_provider
                            .write_credentials(
                                &account_credentials_key(&session_id_str),
                                "Bearer",
                                &json,
                                &*cx,
                            )
                            .await
                            .map_err(|e| Arc::new(e))?;

                        let (still_current, still_in_manifest) = state_clone
                            .read_with(&*cx, |s, _| {
                                (
                                    s.auth_generation == generation,
                                    s.manifest
                                        .sessions
                                        .iter()
                                        .any(|sess| sess.session_id == session_id_str),
                                )
                            })
                            .map_err(|e| Arc::new(e))?;

                        if !still_in_manifest {
                            credentials_provider
                                .delete_credentials(&account_credentials_key(&session_id_str), &*cx)
                                .await
                                .log_err();
                            state_clone
                                .update(cx, |s, _| {
                                    s.cached_credentials.remove(&session_id_str);
                                    if s.active_session_id.as_deref() == Some(&session_id_str) {
                                        s.refresh_task = None;
                                    }
                                    s.refresh_tasks.remove(&session_id_str);
                                })
                                .ok();
                            return Err(Arc::new(anyhow!(
                                "Session was signed out during token refresh persistence"
                            )));
                        }

                        if !still_current {
                            state_clone
                                .update(cx, |s, _| {
                                    s.cached_credentials
                                        .insert(session_id_str.clone(), refreshed.clone());
                                    if s.active_session_id.as_deref() == Some(&session_id_str) {
                                        s.credentials = Some(refreshed.clone());
                                        s.refresh_task = None;
                                    }
                                    s.refresh_tasks.remove(&session_id_str);
                                })
                                .map_err(|e| Arc::new(e))?;
                            return Err(Arc::new(anyhow!(
                                "Account context changed during token refresh persistence"
                            )));
                        }

                        state_clone
                            .update(cx, |s, _| {
                                s.cached_credentials
                                    .insert(session_id_str.clone(), refreshed.clone());
                                if s.active_session_id.as_deref() == Some(&session_id_str) {
                                    s.credentials = Some(refreshed.clone());
                                    s.refresh_task = None;
                                }
                                s.refresh_tasks.remove(&session_id_str);
                            })
                            .map_err(|e| Arc::new(e))?;

                        Ok(refreshed)
                    }
                    .await;

                    if persist_result.is_err() {
                        state_clone
                            .update(cx, |s, _| {
                                if s.active_session_id.as_deref() == Some(&session_id_str) {
                                    s.refresh_task = None;
                                }
                                s.refresh_tasks.remove(&session_id_str);
                            })
                            .ok();
                    }

                    persist_result
                }
                Err(RefreshError::Fatal(e)) => {
                    log::error!("SuperGrok token refresh failed fatally: {e:?}");
                    let still_current_generation = state_clone
                        .read_with(&*cx, |s, _| s.auth_generation == generation)
                        .unwrap_or(false);
                    if still_current_generation {
                        let manifest_to_persist = state_clone
                            .update(cx, |s, cx| {
                                if s.active_session_id.as_deref() == Some(&session_id_str) {
                                    s.refresh_task = None;
                                    s.credentials = None;
                                    s.last_auth_error = Some(
                                        "Your SuperGrok session has expired. Sign in again.".into(),
                                    );
                                }
                                s.refresh_tasks.remove(&session_id_str);
                                s.cached_credentials.remove(&session_id_str);
                                if let Some(account) = s
                                    .manifest
                                    .sessions
                                    .iter_mut()
                                    .find(|account| account.session_id == session_id_str)
                                {
                                    account.reauthentication_required = true;
                                    account
                                        .add_exclusion(AccountExclusion::ReauthenticationRequired);
                                }
                                s.routing_generation = s.routing_generation.wrapping_add(1);
                                cx.notify();
                                s.manifest.clone()
                            })
                            .ok();
                        if let Ok(credentials_provider) =
                            state_clone.read_with(&*cx, |s, _| s.credentials_provider.clone())
                        {
                            credentials_provider
                                .delete_credentials(&account_credentials_key(&session_id_str), &*cx)
                                .await
                                .log_err();
                            if let Some(manifest) = manifest_to_persist {
                                if let Ok(json) = serde_json::to_vec(&manifest) {
                                    credentials_provider
                                        .write_credentials(
                                            ACCOUNT_MANIFEST_KEY,
                                            "manifest",
                                            &json,
                                            &*cx,
                                        )
                                        .await
                                        .log_err();
                                }
                            }
                        }
                    } else {
                        state_clone
                            .update(cx, |s, _| {
                                if s.active_session_id.as_deref() == Some(&session_id_str) {
                                    s.refresh_task = None;
                                }
                                s.refresh_tasks.remove(&session_id_str);
                            })
                            .ok();
                    }
                    Err(Arc::new(e))
                }
                Err(RefreshError::Transient(e)) => {
                    log::warn!("SuperGrok token refresh failed transiently: {e:?}");
                    state_clone
                        .update(cx, |s, _| {
                            if s.active_session_id.as_deref() == Some(&session_id_str) {
                                s.refresh_task = None;
                            }
                            s.refresh_tasks.remove(&session_id_str);
                        })
                        .ok();
                    Err(Arc::new(e))
                }
            }
        })
        .shared();

    let session_key = session_id.to_string();
    let is_active = state
        .read_with(&*cx, |s, _| {
            s.active_session_id.as_deref() == Some(&session_key)
        })
        .unwrap_or(false);

    state
        .update(cx, |s, _| {
            if is_active {
                s.refresh_task = Some(shared_task.clone());
            }
            s.refresh_tasks.insert(session_key, shared_task.clone());
        })
        .map_err(LanguageModelCompletionError::Other)?;

    shared_task
        .await
        .map_err(|e| LanguageModelCompletionError::Other(anyhow!("{e}")))
}

async fn force_refresh_session_credentials(
    state: &WeakEntity<State>,
    http_client: &Arc<dyn HttpClient>,
    session_id: &str,
    cx: &mut AsyncApp,
) -> Result<SuperGrokCredentials, RefreshError> {
    let (creds, generation) = state
        .read_with(&*cx, |s, _| {
            (
                s.cached_credentials.get(session_id).cloned().or_else(|| {
                    if s.active_session_id.as_deref() == Some(session_id) {
                        s.credentials.clone()
                    } else {
                        None
                    }
                }),
                s.auth_generation,
            )
        })
        .map_err(RefreshError::Transient)?;

    let creds = match creds {
        Some(c) => c,
        None => {
            let provider = state
                .read_with(&*cx, |s, _| s.credentials_provider.clone())
                .map_err(RefreshError::Transient)?;
            let key = account_credentials_key(session_id);
            let stored = provider
                .read_credentials(&key, cx)
                .await
                .map_err(RefreshError::Transient)?;
            let Some((_, bytes)) = stored else {
                return Err(RefreshError::Fatal(anyhow!("Account credentials missing")));
            };
            serde_json::from_slice(&bytes).map_err(|e| RefreshError::Fatal(e.into()))?
        }
    };

    let session_id_str = session_id.to_string();
    let state_clone = state.clone();
    let http_client = http_client.clone();
    let previous_refresh = creds.refresh_token.clone();
    let previous_email = creds.email.clone();

    let tokens = match refresh_token(&http_client, &previous_refresh).await {
        Ok(tokens) => tokens,
        Err(e) => {
            if matches!(e, RefreshError::Fatal(_)) {
                let still_present = state_clone
                    .read_with(&*cx, |s, _| {
                        s.auth_generation == generation
                            && s.manifest
                                .sessions
                                .iter()
                                .any(|sess| sess.session_id == session_id_str)
                    })
                    .unwrap_or(false);
                if still_present {
                    state_clone
                        .update(cx, |s, cx| {
                            s.mark_reauthentication_required(&session_id_str);
                            cx.notify();
                        })
                        .ok();
                    if let Ok(provider) =
                        state_clone.read_with(&*cx, |s, _| s.credentials_provider.clone())
                    {
                        provider
                            .delete_credentials(&account_credentials_key(&session_id_str), &*cx)
                            .await
                            .log_err();
                    }
                }
            }
            return Err(e);
        }
    };

    let claims = tokens
        .id_token
        .as_deref()
        .map(extract_email_claim)
        .unwrap_or(None);
    let refreshed = SuperGrokCredentials {
        access_token: tokens.access_token,
        refresh_token: tokens
            .refresh_token
            .filter(|token| !token.is_empty())
            .unwrap_or(previous_refresh),
        expires_at_ms: now_ms() + tokens.expires_in * 1000,
        email: claims.or(tokens.email).or(previous_email),
        account_id: Some(session_id_str.clone()),
    };

    let credentials_provider = state_clone
        .read_with(&*cx, |s, _| s.credentials_provider.clone())
        .map_err(RefreshError::Transient)?;
    let json = serde_json::to_vec(&refreshed).map_err(|e| RefreshError::Transient(e.into()))?;

    let (current_gen, session_still_present) = state_clone
        .read_with(&*cx, |s, _| {
            (
                s.auth_generation,
                s.manifest
                    .sessions
                    .iter()
                    .any(|sess| sess.session_id == session_id_str),
            )
        })
        .map_err(RefreshError::Transient)?;
    if current_gen != generation || !session_still_present {
        return Err(RefreshError::Transient(anyhow!(
            "Sign-out occurred during token refresh"
        )));
    }

    credentials_provider
        .write_credentials(
            &account_credentials_key(&session_id_str),
            "Bearer",
            &json,
            &*cx,
        )
        .await
        .map_err(RefreshError::Transient)?;

    let (still_current, still_in_manifest) = state_clone
        .read_with(&*cx, |s, _| {
            (
                s.auth_generation == generation,
                s.manifest
                    .sessions
                    .iter()
                    .any(|sess| sess.session_id == session_id_str),
            )
        })
        .map_err(RefreshError::Transient)?;

    if !still_in_manifest {
        credentials_provider
            .delete_credentials(&account_credentials_key(&session_id_str), &*cx)
            .await
            .log_err();
        state_clone
            .update(cx, |s, _| {
                s.cached_credentials.remove(&session_id_str);
            })
            .ok();
        return Err(RefreshError::Transient(anyhow!(
            "Session was signed out during token refresh persistence"
        )));
    }

    if !still_current {
        state_clone
            .update(cx, |s, _| {
                s.cached_credentials
                    .insert(session_id_str.clone(), refreshed.clone());
            })
            .map_err(RefreshError::Transient)?;
        return Err(RefreshError::Transient(anyhow!(
            "Account context changed during token refresh persistence"
        )));
    }

    state_clone
        .update(cx, |s, _| {
            s.cached_credentials
                .insert(session_id_str.clone(), refreshed.clone());
            if s.active_session_id.as_deref() == Some(&session_id_str) {
                s.credentials = Some(refreshed.clone());
            }
        })
        .map_err(RefreshError::Transient)?;

    Ok(refreshed)
}

#[derive(Deserialize)]
struct TokenResponse {
    access_token: String,
    #[serde(default)]
    refresh_token: Option<String>,
    #[serde(default)]
    id_token: Option<String>,
    expires_in: u64,
    #[serde(default)]
    email: Option<String>,
    #[serde(default)]
    sub: Option<String>,
}

struct PkceAuthorizeRequest {
    authorize_url: String,
    verifier: String,
    state: String,
    redirect_uri: String,
}

fn pkce_challenge(verifier: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(verifier.as_bytes());
    URL_SAFE_NO_PAD.encode(hasher.finalize().as_slice())
}

fn build_authorize_request(
    redirect_uri: &str,
    verifier: &str,
    oauth_state: &str,
) -> Result<String> {
    let challenge = pkce_challenge(verifier);
    let mut auth_url = url::Url::parse(XAI_AUTHORIZE_URL).context("invalid authorize URL")?;
    auth_url
        .query_pairs_mut()
        .append_pair("client_id", CLIENT_ID)
        .append_pair("redirect_uri", redirect_uri)
        .append_pair("scope", OAUTH_SCOPE)
        .append_pair("response_type", "code")
        .append_pair("code_challenge", &challenge)
        .append_pair("code_challenge_method", "S256")
        .append_pair("state", oauth_state)
        .append_pair("nonce", oauth_state);
    Ok(auth_url.to_string())
}

fn new_pkce_authorize_request(redirect_uri: String) -> Result<PkceAuthorizeRequest> {
    let mut verifier_bytes = [0u8; 32];
    rand::rng().fill_bytes(&mut verifier_bytes);
    let verifier = URL_SAFE_NO_PAD.encode(verifier_bytes);

    let mut state_bytes = [0u8; 16];
    rand::rng().fill_bytes(&mut state_bytes);
    let oauth_state: String = state_bytes.iter().map(|b| format!("{b:02x}")).collect();

    let authorize_url = build_authorize_request(&redirect_uri, &verifier, &oauth_state)?;
    Ok(PkceAuthorizeRequest {
        authorize_url,
        verifier,
        state: oauth_state,
        redirect_uri,
    })
}

async fn do_oauth_flow(
    http_client: Arc<dyn HttpClient>,
    cx: &AsyncApp,
) -> Result<SuperGrokCredentials> {
    let (redirect_uri, callback_rx) =
        oauth_callback_server::start_oauth_callback_server_with_config(
            oauth_callback_server::OAuthCallbackServerConfig {
                host: CALLBACK_HOST,
                preferred_port: CALLBACK_PORT,
                fallback_port: None,
                path: CALLBACK_PATH,
            },
        )
        .context("Failed to start OAuth callback server")?;

    let pkce = new_pkce_authorize_request(redirect_uri)?;
    cx.update(|cx| cx.open_url(&pkce.authorize_url));

    let callback = callback_rx
        .await
        .map_err(|_| anyhow!("OAuth callback was cancelled"))?
        .context("OAuth callback failed")?;

    if callback.state != pkce.state {
        return Err(anyhow!("OAuth state mismatch"));
    }

    let tokens = exchange_code(
        &http_client,
        &callback.code,
        &pkce.verifier,
        &pkce.redirect_uri,
    )
    .await
    .context("Token exchange failed")?;

    let refresh_token = tokens
        .refresh_token
        .filter(|token| !token.is_empty())
        .context("Token response did not include a refresh_token")?;
    let email = tokens
        .id_token
        .as_deref()
        .and_then(extract_email_claim)
        .or(tokens.email);
    let account_id = tokens
        .id_token
        .as_deref()
        .and_then(|token| extract_claim(token, "sub"))
        .or(tokens.sub);

    Ok(SuperGrokCredentials {
        access_token: tokens.access_token,
        refresh_token,
        expires_at_ms: now_ms() + tokens.expires_in * 1000,
        email,
        account_id,
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
        .uri(XAI_TOKEN_URL)
        .header("Content-Type", "application/x-www-form-urlencoded")
        .body(AsyncBody::from(body))?;

    let mut response = client.send(request).await?;
    let mut body = String::new();
    smol::io::AsyncReadExt::read_to_string(response.body_mut(), &mut body).await?;

    if !response.status().is_success() {
        return Err(anyhow!(
            "Token exchange failed (HTTP {}): {}",
            response.status(),
            redact_token_body(&body)
        ));
    }

    serde_json::from_str::<TokenResponse>(&body).context("Failed to parse token response")
}

async fn refresh_token(
    client: &Arc<dyn HttpClient>,
    refresh_token: &str,
) -> Result<TokenResponse, RefreshError> {
    let body = form_urlencoded::Serializer::new(String::new())
        .append_pair("grant_type", "refresh_token")
        .append_pair("client_id", CLIENT_ID)
        .append_pair("refresh_token", refresh_token)
        .finish();

    let request = HttpRequest::builder()
        .method(Method::POST)
        .uri(XAI_TOKEN_URL)
        .header("Content-Type", "application/x-www-form-urlencoded")
        .body(AsyncBody::from(body))
        .map_err(|e| RefreshError::Transient(e.into()))?;

    let mut response = client
        .send(request)
        .await
        .map_err(RefreshError::Transient)?;
    let status = response.status();
    let mut body = String::new();
    smol::io::AsyncReadExt::read_to_string(response.body_mut(), &mut body)
        .await
        .map_err(|e| RefreshError::Transient(e.into()))?;

    if !status.is_success() {
        let err = anyhow!(
            "Token refresh failed (HTTP {}): {}",
            status,
            redact_token_body(&body)
        );
        if status == http_client::StatusCode::BAD_REQUEST
            || status == http_client::StatusCode::UNAUTHORIZED
            || status == http_client::StatusCode::FORBIDDEN
        {
            return Err(RefreshError::Fatal(err));
        }
        return Err(RefreshError::Transient(err));
    }

    serde_json::from_str(&body).map_err(|e| RefreshError::Transient(e.into()))
}

fn extract_email_claim(jwt: &str) -> Option<String> {
    let payload_b64 = jwt.split('.').nth(1)?;
    let payload = URL_SAFE_NO_PAD.decode(payload_b64).ok()?;
    let claims = serde_json::from_slice::<serde_json::Value>(&payload).ok()?;
    claims
        .get("email")
        .and_then(|value| value.as_str())
        .map(str::to_owned)
}

fn extract_claim(jwt: &str, name: &str) -> Option<String> {
    let payload_b64 = jwt.split('.').nth(1)?;
    let payload = URL_SAFE_NO_PAD.decode(payload_b64).ok()?;
    let claims = serde_json::from_slice::<serde_json::Value>(&payload).ok()?;
    claims
        .get(name)
        .and_then(|value| value.as_str())
        .map(str::to_owned)
}

fn account_credentials_key(session_id: &str) -> String {
    format!("{ACCOUNT_CREDENTIALS_PREFIX}{session_id}")
}

fn account_session_id(credentials: &SuperGrokCredentials) -> String {
    credentials.email.clone().unwrap_or_else(|| {
        let mut hasher = Sha256::new();
        hasher.update(credentials.refresh_token.as_bytes());
        format!("grok-{:x}", hasher.finalize())
    })
}

fn redact_token_body(body: &str) -> String {
    const MAX_LEN: usize = 240;
    if body.len() <= MAX_LEN {
        body.to_string()
    } else {
        format!("{}…", &body[..MAX_LEN])
    }
}

pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or_else(|err| {
            log::error!("System clock is before UNIX epoch: {err}");
            0
        })
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

async fn refresh_all_account_quotas(state: &WeakEntity<State>, cx: &mut AsyncApp) -> Result<()> {
    let Some((provider, http_client, session_ids, generation)) = state
        .read_with(cx, |state, _| {
            if state.manifest.sessions.is_empty()
                || state.account_mutation_in_progress
                || state.is_signing_in()
            {
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
            Ok(Some((_, bytes))) => match serde_json::from_slice::<SuperGrokCredentials>(&bytes) {
                Ok(credentials) => credentials,
                Err(error) => {
                    log::warn!(
                        "SuperGrok quota check skipped account {session_id}: invalid credentials: {error}"
                    );
                    continue;
                }
            },
            Ok(None) => {
                log::warn!(
                    "SuperGrok quota check skipped account {session_id}: credentials missing"
                );
                continue;
            }
            Err(error) => {
                log::warn!(
                    "SuperGrok quota check skipped account {session_id}: keychain read failed: {error:#}"
                );
                continue;
            }
        };

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
                        "SuperGrok quota check requires sign-in again for account {session_id}: {error:#}"
                    );
                    reauthentication_required.push(session_id);
                    continue;
                }
                Err(RefreshError::Transient(error)) => {
                    log::debug!(
                        "SuperGrok quota token refresh failed transiently for account {session_id}: {error:#}"
                    );
                    continue;
                }
            }
        }

        let mut result = fetch_quota(http_client.as_ref(), &credentials).await;
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
                        "SuperGrok quota check requires sign-in again for account {session_id}: {error:#}"
                    );
                    reauthentication_required.push(session_id);
                    continue;
                }
                Err(RefreshError::Transient(error)) => {
                    log::debug!(
                        "SuperGrok quota token recovery failed transiently for account {session_id}: {error:#}"
                    );
                    continue;
                }
            }
        }

        match result {
            Ok(quota) => updates.push((session_id, quota)),
            Err(error) => {
                log::debug!("SuperGrok quota check failed for account {session_id}: {error}")
            }
        }
    }

    if updates.is_empty() && credential_updates.is_empty() && reauthentication_required.is_empty() {
        return Ok(());
    }
    let updated = state.update(cx, |state, cx| {
        if state.auth_generation != generation || state.account_mutation_in_progress {
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
                if let Some(plan_type) = quota.plan_type.clone() {
                    session.plan_type = Some(plan_type);
                }
                session.clean_expired_exclusions(fetched_at);
                if quota.used_percent >= QUOTA_EXHAUSTION_THRESHOLD_PERCENT {
                    let retry_at_ms = quota
                        .resets_at
                        .map(|s| (s as u64).saturating_mul(1000))
                        .filter(|ms| *ms > fetched_at)
                        .unwrap_or(fetched_at + 300_000);
                    session.add_exclusion(AccountExclusion::RateLimited {
                        retry_at_ms,
                        scope: AccountExclusionScope::Account,
                    });
                } else {
                    session.clear_account_rate_limits();
                }
                session.quota = Some(quota);
                session.quota_fetched_at_ms = Some(fetched_at);
            }
        }
        for (session_id, old_refresh_token, credentials) in credential_updates {
            if state.active_session_id.as_deref() == Some(session_id.as_str())
                && state
                    .credentials
                    .as_ref()
                    .is_some_and(|current| current.refresh_token == old_refresh_token)
            {
                state.credentials = Some(credentials);
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
                session.add_exclusion(AccountExclusion::ReauthenticationRequired);
            }
        }
        state.routing_generation = state.routing_generation.wrapping_add(1);
        cx.notify();
        true
    })?;
    if updated {
        persist_manifest(&provider, state, cx).await?;
    } else {
        log::debug!("Discarded stale SuperGrok quota cycle after account mutation");
    }
    Ok(())
}

async fn refresh_quota_credentials(
    state: &WeakEntity<State>,
    generation: u64,
    provider: &Arc<dyn CredentialsProvider>,
    http_client: &Arc<dyn HttpClient>,
    session_id: &str,
    credentials: &SuperGrokCredentials,
    cx: &AsyncApp,
) -> Result<SuperGrokCredentials, RefreshError> {
    let tokens = refresh_token(http_client, &credentials.refresh_token).await?;
    let claims = tokens
        .id_token
        .as_deref()
        .map(extract_email_claim)
        .unwrap_or(None);
    let refreshed = SuperGrokCredentials {
        access_token: tokens.access_token,
        refresh_token: tokens
            .refresh_token
            .filter(|token| !token.is_empty())
            .unwrap_or_else(|| credentials.refresh_token.clone()),
        expires_at_ms: now_ms() + tokens.expires_in * 1000,
        email: claims.or(tokens.email).or(credentials.email.clone()),
        account_id: Some(session_id.to_string()),
    };

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
            "SuperGrok account changed during quota token refresh"
        )));
    }

    if let Some((_, bytes)) = provider
        .read_credentials(&key, cx)
        .await
        .map_err(RefreshError::Transient)?
    {
        let current = serde_json::from_slice::<SuperGrokCredentials>(&bytes)
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
            "SuperGrok account changed during quota credential persistence"
        )));
    }
    Ok(refreshed)
}

async fn persist_manifest(
    credentials_provider: &Arc<dyn CredentialsProvider>,
    state: &WeakEntity<State>,
    cx: &mut AsyncApp,
) -> Result<()> {
    for _ in 0..4 {
        let Some((manifest, generation)) = state
            .read_with(cx, |state, _| {
                (state.manifest.clone(), state.auth_generation)
            })
            .ok()
        else {
            return Ok(());
        };
        if manifest.sessions.is_empty() {
            credentials_provider
                .delete_credentials(ACCOUNT_MANIFEST_KEY, cx)
                .await?;
            let still_current = state
                .read_with(cx, |state, _| state.auth_generation == generation)
                .unwrap_or(false);
            if still_current {
                return Ok(());
            }
            continue;
        }
        credentials_provider
            .write_credentials(
                ACCOUNT_MANIFEST_KEY,
                "manifest",
                &serde_json::to_vec(&manifest)?,
                cx,
            )
            .await?;
        let still_current = state
            .read_with(cx, |state, _| state.auth_generation == generation)
            .unwrap_or(false);
        if still_current {
            return Ok(());
        }
    }
    Err(anyhow!(
        "SuperGrok subscription manifest writes kept racing account mutations"
    ))
}

async fn fetch_quota(
    http_client: &dyn HttpClient,
    credentials: &SuperGrokCredentials,
) -> Result<QuotaSnapshot, QuotaFetchError> {
    match fetch_quota_from_cli_billing(http_client, credentials).await {
        Ok(quota) => return Ok(quota),
        Err(QuotaFetchError::Unauthorized(error)) => {
            return Err(QuotaFetchError::Unauthorized(error));
        }
        Err(error) => {
            log::debug!("SuperGrok CLI billing quota failed, trying grok.com credits: {error}");
        }
    }
    fetch_quota_from_grok_credits(http_client, credentials).await
}

async fn fetch_quota_from_cli_billing(
    http_client: &dyn HttpClient,
    credentials: &SuperGrokCredentials,
) -> Result<QuotaSnapshot, QuotaFetchError> {
    let request = HttpRequest::builder()
        .method(Method::GET)
        .uri(GROK_CLI_BILLING_URL)
        .header(
            "Authorization",
            format!("Bearer {}", credentials.access_token.trim()),
        )
        .header("Accept", "application/json")
        .header("x-xai-token-auth", "xai-grok-cli")
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
        let error = anyhow!("SuperGrok billing request failed (HTTP {status}): {body}");
        if status == http_client::StatusCode::UNAUTHORIZED {
            return Err(QuotaFetchError::Unauthorized(error));
        }
        return Err(QuotaFetchError::Other(error));
    }
    let value = serde_json::from_str::<serde_json::Value>(&body)
        .context("Failed to parse SuperGrok billing response")
        .map_err(QuotaFetchError::Other)?;
    quota_snapshot_from_billing_value(&value).map_err(QuotaFetchError::Other)
}

async fn fetch_quota_from_grok_credits(
    http_client: &dyn HttpClient,
    credentials: &SuperGrokCredentials,
) -> Result<QuotaSnapshot, QuotaFetchError> {
    let request = HttpRequest::builder()
        .method(Method::POST)
        .uri(GROK_CREDITS_GRPC_URL)
        .header(
            "Authorization",
            format!("Bearer {}", credentials.access_token.trim()),
        )
        .header("Content-Type", "application/grpc-web+proto")
        .header("Accept", "application/grpc-web+proto")
        .header("X-Grpc-Web", "1")
        .header("Origin", "https://grok.com")
        .header("Referer", "https://grok.com/")
        .body(AsyncBody::from(vec![0u8, 0, 0, 0, 0]))
        .map_err(|error| QuotaFetchError::Other(error.into()))?;
    let mut response = http_client
        .send(request)
        .await
        .map_err(QuotaFetchError::Other)?;
    let status = response.status();
    let mut body = Vec::new();
    smol::io::AsyncReadExt::read_to_end(response.body_mut(), &mut body)
        .await
        .map_err(|error| QuotaFetchError::Other(error.into()))?;
    if !status.is_success() {
        let preview = String::from_utf8_lossy(&body);
        let error = anyhow!("SuperGrok credits request failed (HTTP {status}): {preview}");
        if status == http_client::StatusCode::UNAUTHORIZED {
            return Err(QuotaFetchError::Unauthorized(error));
        }
        return Err(QuotaFetchError::Other(error));
    }
    quota_snapshot_from_grok_credits(&body).map_err(QuotaFetchError::Other)
}

fn quota_snapshot_from_billing_value(value: &serde_json::Value) -> Result<QuotaSnapshot> {
    let config = value.get("config").unwrap_or(value);
    let used_percent = json_f64(config, "creditUsagePercent")
        .or_else(|| json_f64(config, "credit_usage_percent"))
        .or_else(|| json_f64(value, "creditUsagePercent"))
        .or_else(|| json_f64(value, "credit_usage_percent"))
        .or_else(|| {
            let used = json_f64(config, "onDemandUsed")
                .or_else(|| nested_json_f64(config, &["onDemandUsed", "val"]))
                .or_else(|| json_f64(value, "used"));
            let cap = json_f64(config, "onDemandCap")
                .or_else(|| nested_json_f64(config, &["onDemandCap", "val"]))
                .or_else(|| json_f64(value, "monthlyLimit"));
            match (used, cap) {
                (Some(used), Some(cap)) if cap > 0.0 => Some((used / cap) * 100.0),
                (Some(_), Some(0.0)) => Some(0.0),
                _ => None,
            }
        })
        .ok_or_else(|| anyhow!("SuperGrok billing response did not contain usage percent"))?;
    if !used_percent.is_finite() {
        return Err(anyhow!("SuperGrok billing usage percent was not finite"));
    }
    let resets_at = json_timestamp_seconds(
        config
            .get("currentPeriod")
            .and_then(|period| period.get("end").or_else(|| period.get("billingPeriodEnd"))),
    )
    .or_else(|| json_timestamp_seconds(config.get("billingPeriodEnd")))
    .or_else(|| json_timestamp_seconds(value.get("billingPeriodEnd")));
    let plan_type = config
        .get("subscription_tier_display")
        .or_else(|| config.get("plan"))
        .or_else(|| value.get("subscription_tier_display"))
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);
    Ok(QuotaSnapshot {
        used_percent: used_percent.clamp(0.0, 100.0),
        resets_at,
        plan_type,
        captured_at_ms: now_ms(),
    })
}

fn json_f64(value: &serde_json::Value, key: &str) -> Option<f64> {
    json_number(value.get(key))
}

fn nested_json_f64(value: &serde_json::Value, path: &[&str]) -> Option<f64> {
    let mut current = value;
    for key in path {
        current = current.get(*key)?;
    }
    json_number(Some(current))
}

fn json_number(value: Option<&serde_json::Value>) -> Option<f64> {
    let value = value?;
    if let Some(number) = value.as_f64() {
        return number.is_finite().then_some(number);
    }
    if let Some(number) = value.as_i64() {
        return Some(number as f64);
    }
    if let Some(number) = value.as_u64() {
        return Some(number as f64);
    }
    value
        .as_str()?
        .parse()
        .ok()
        .filter(|number: &f64| number.is_finite())
}

fn json_timestamp_seconds(value: Option<&serde_json::Value>) -> Option<i64> {
    let value = value?;
    if let Some(seconds) = value.as_i64() {
        return Some(seconds);
    }
    if let Some(seconds) = value.as_u64() {
        return Some(seconds as i64);
    }
    if let Some(seconds) = value.as_f64() {
        return seconds.is_finite().then_some(seconds as i64);
    }
    if let Some(text) = value.as_str() {
        if let Ok(seconds) = text.parse::<i64>() {
            return Some(seconds);
        }
        return chrono::DateTime::parse_from_rfc3339(text)
            .ok()
            .map(|datetime| datetime.timestamp());
    }
    None
}

fn quota_snapshot_from_grok_credits(body: &[u8]) -> Result<QuotaSnapshot> {
    let message = decode_grpc_web_message(body)?;
    let config = protobuf_first_bytes(&message, 1)
        .ok_or_else(|| anyhow!("GetGrokCreditsConfigResponse missing config field"))?;
    let mut used_percent = 0.0;
    let mut resets_at = None;
    let mut saw_percent = false;
    for (number, wire, value) in protobuf_fields(config)? {
        match (number, wire) {
            (1, 5) => {
                if let Ok(bytes) = <[u8; 4]>::try_from(value) {
                    used_percent = f32::from_le_bytes(bytes) as f64;
                    saw_percent = true;
                }
            }
            (5, 2) => {
                resets_at = protobuf_timestamp_seconds(value);
            }
            _ => {}
        }
    }
    if !saw_percent && resets_at.is_none() && config.is_empty() {
        return Err(anyhow!(
            "SuperGrok credits response did not contain usage data"
        ));
    }
    if !used_percent.is_finite() {
        return Err(anyhow!("SuperGrok credits usage percent was not finite"));
    }
    Ok(QuotaSnapshot {
        used_percent: used_percent.clamp(0.0, 100.0),
        resets_at,
        plan_type: None,
        captured_at_ms: now_ms(),
    })
}

fn decode_grpc_web_message(body: &[u8]) -> Result<&[u8]> {
    if body.is_empty() {
        return Err(anyhow!(
            "Empty gRPC-web response from grok.com credits endpoint"
        ));
    }
    let mut offset = 0;
    let mut messages = Vec::new();
    while offset < body.len() {
        if offset + 5 > body.len() {
            return Err(anyhow!("Malformed gRPC-web frame: truncated header"));
        }
        let flags = body[offset];
        let length = u32::from_be_bytes(
            body[offset + 1..offset + 5]
                .try_into()
                .map_err(|_| anyhow!("Malformed gRPC-web frame: invalid length"))?,
        ) as usize;
        offset += 5;
        if offset + length > body.len() {
            return Err(anyhow!("Malformed gRPC-web frame: truncated payload"));
        }
        let payload = &body[offset..offset + length];
        offset += length;
        if flags & 0x80 != 0 {
            continue;
        }
        messages.push(payload);
    }
    messages
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("gRPC-web response contained no message frames"))
}

fn protobuf_fields(data: &[u8]) -> Result<Vec<(u32, u32, &[u8])>> {
    let mut fields = Vec::new();
    let mut offset = 0;
    while offset < data.len() {
        let (key, next) = protobuf_varint(data, offset)?;
        offset = next;
        let number = (key >> 3) as u32;
        let wire = (key & 0x07) as u32;
        match wire {
            0 => {
                let start = offset;
                let (_, next) = protobuf_varint(data, offset)?;
                fields.push((number, wire, &data[start..next]));
                offset = next;
            }
            1 => {
                if offset + 8 > data.len() {
                    return Err(anyhow!("Malformed protobuf: truncated fixed64"));
                }
                fields.push((number, wire, &data[offset..offset + 8]));
                offset += 8;
            }
            2 => {
                let (length, next) = protobuf_varint(data, offset)?;
                offset = next;
                let length = length as usize;
                if offset + length > data.len() {
                    return Err(anyhow!("Malformed protobuf: truncated bytes"));
                }
                fields.push((number, wire, &data[offset..offset + length]));
                offset += length;
            }
            5 => {
                if offset + 4 > data.len() {
                    return Err(anyhow!("Malformed protobuf: truncated fixed32"));
                }
                fields.push((number, wire, &data[offset..offset + 4]));
                offset += 4;
            }
            other => return Err(anyhow!("Unsupported protobuf wire type {other}")),
        }
    }
    Ok(fields)
}

fn protobuf_first_bytes(data: &[u8], field: u32) -> Option<&[u8]> {
    protobuf_fields(data)
        .ok()?
        .into_iter()
        .find(|(number, wire, _)| *number == field && *wire == 2)
        .map(|(_, _, value)| value)
}

fn protobuf_timestamp_seconds(data: &[u8]) -> Option<i64> {
    let mut seconds = None;
    for (number, wire, value) in protobuf_fields(data).ok()? {
        if number == 1 && wire == 0 {
            let (parsed, _) = protobuf_varint(value, 0).ok()?;
            seconds = Some(parsed as i64);
        }
    }
    seconds
}

fn protobuf_varint(data: &[u8], mut offset: usize) -> Result<(u64, usize)> {
    let mut value = 0u64;
    let mut shift = 0;
    loop {
        let byte = *data
            .get(offset)
            .ok_or_else(|| anyhow!("Malformed protobuf: truncated varint"))?;
        offset += 1;
        value |= u64::from(byte & 0x7F) << shift;
        if byte & 0x80 == 0 {
            return Ok((value, offset));
        }
        shift += 7;
        if shift > 70 {
            return Err(anyhow!("Malformed protobuf: varint too long"));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::{AppContext as _, TestAppContext};
    use http_client::FakeHttpClient;
    use parking_lot::Mutex;
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn grok_46_supports_xhigh_and_defaults_to_high() {
        let effort_levels = supported_thinking_effort_levels(&SuperGrokModel::Grok46);
        let values = effort_levels
            .iter()
            .map(|level| level.value.as_ref())
            .collect::<Vec<_>>();

        assert_eq!(values, ["low", "medium", "high", "xhigh"]);
        assert_eq!(
            effort_levels
                .iter()
                .find(|level| level.is_default)
                .map(|level| level.value.as_ref()),
            Some("high")
        );
    }

    #[test]
    fn grok_46_request_uses_selected_reasoning_effort() {
        let request = LanguageModelRequest {
            thinking_allowed: true,
            thinking_effort: Some("xhigh".to_string()),
            ..Default::default()
        };

        assert_eq!(
            reasoning_effort_for_request(&request, &SuperGrokModel::Grok46),
            Some(ReasoningEffort::XHigh)
        );
    }

    #[test]
    fn grok_46_omits_reasoning_effort_when_thinking_is_disabled() {
        let request = LanguageModelRequest {
            thinking_allowed: false,
            thinking_effort: Some("medium".to_string()),
            ..Default::default()
        };

        assert_eq!(
            reasoning_effort_for_request(&request, &SuperGrokModel::Grok46),
            None
        );
    }

    #[test]
    fn grok_build_omits_reasoning_effort() {
        let request = LanguageModelRequest {
            thinking_allowed: true,
            thinking_effort: Some("medium".to_string()),
            ..Default::default()
        };

        assert_eq!(
            reasoning_effort_for_request(&request, &SuperGrokModel::GrokBuild01),
            None
        );
    }

    #[test]
    fn grok_45_does_not_advertise_xhigh() {
        let effort_levels = supported_thinking_effort_levels(&SuperGrokModel::Grok45);
        let values = effort_levels
            .iter()
            .map(|level| level.value.as_ref())
            .collect::<Vec<_>>();
        assert_eq!(values, ["low", "medium", "high"]);
        assert_eq!(
            effort_levels
                .iter()
                .find(|level| level.is_default)
                .map(|level| level.value.as_ref()),
            Some("high")
        );
    }

    #[test]
    fn grok_46_and_45_omit_max_output_tokens() {
        assert_eq!(SuperGrokModel::Grok46.max_output_tokens(), None);
        assert_eq!(SuperGrokModel::Grok45.max_output_tokens(), None);
        assert_eq!(
            SuperGrokModel::GrokBuild01.max_output_tokens(),
            Some(64_000)
        );
    }

    #[test]
    fn authorize_url_uses_public_client_and_s256() {
        let verifier = "test-verifier";
        let url = build_authorize_request("http://127.0.0.1:56121/callback", verifier, "abc123")
            .expect("url should build");
        let parsed = url::Url::parse(&url).unwrap();
        let pairs: Vec<(String, String)> = parsed
            .query_pairs()
            .map(|(k, v)| (k.into_owned(), v.into_owned()))
            .collect();

        assert_eq!(parsed.as_str().split('?').next(), Some(XAI_AUTHORIZE_URL));
        assert!(pairs.contains(&("client_id".into(), CLIENT_ID.into())));
        assert!(pairs.contains(&(
            "redirect_uri".into(),
            "http://127.0.0.1:56121/callback".into()
        )));
        assert!(pairs.contains(&("scope".into(), OAUTH_SCOPE.into())));
        assert!(pairs.contains(&("response_type".into(), "code".into())));
        assert!(pairs.contains(&("code_challenge_method".into(), "S256".into())));
        assert!(pairs.contains(&("code_challenge".into(), pkce_challenge(verifier))));
        assert!(pairs.contains(&("state".into(), "abc123".into())));
        assert!(pairs.contains(&("nonce".into(), "abc123".into())));
    }

    #[gpui::test]
    async fn test_concurrent_refresh_deduplicates(cx: &mut TestAppContext) {
        let refresh_count = Arc::new(AtomicUsize::new(0));
        let refresh_count_clone = refresh_count.clone();

        let http_client = FakeHttpClient::create(move |_request| {
            let refresh_count = refresh_count_clone.clone();
            async move {
                refresh_count.fetch_add(1, Ordering::SeqCst);
                let body = fake_token_response(true);
                Ok(http_client::Response::builder()
                    .status(200)
                    .body(http_client::AsyncBody::from(body))?)
            }
        });

        let http: Arc<dyn HttpClient> = http_client;
        let state = make_state(http.clone(), Some(make_expired_credentials()), cx);
        let weak_state = cx.read(|_cx| state.downgrade());

        let weak1 = weak_state.clone();
        let http1 = http.clone();
        let task1 =
            cx.spawn(async move |mut cx| get_fresh_credentials(&weak1, &http1, &mut cx).await);

        let weak2 = weak_state.clone();
        let http2 = http.clone();
        let task2 =
            cx.spawn(async move |mut cx| get_fresh_credentials(&weak2, &http2, &mut cx).await);

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
                let body = fake_token_response(true);
                Ok(http_client::Response::builder()
                    .status(200)
                    .body(http_client::AsyncBody::from(body))?)
            }
        });

        let http: Arc<dyn HttpClient> = http_client;
        let state = make_state(http.clone(), Some(make_fresh_credentials()), cx);
        let weak_state = cx.read(|_cx| state.downgrade());
        let http_clone = http.clone();
        let result = cx
            .spawn(async move |mut cx| {
                get_fresh_credentials(&weak_state, &http_clone, &mut cx).await
            })
            .await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap().access_token, "fresh_access");
        assert_eq!(refresh_count.load(Ordering::SeqCst), 0);
    }

    #[gpui::test]
    async fn test_no_credentials_returns_no_api_key(cx: &mut TestAppContext) {
        let http: Arc<dyn HttpClient> = FakeHttpClient::create(|_| async {
            Ok(http_client::Response::builder()
                .status(200)
                .body(http_client::AsyncBody::default())?)
        });
        let state = make_state(http.clone(), None, cx);
        let weak_state = cx.read(|_cx| state.downgrade());
        let result = cx
            .spawn(async move |mut cx| get_fresh_credentials(&weak_state, &http, &mut cx).await)
            .await;

        assert!(matches!(
            result,
            Err(LanguageModelCompletionError::NoApiKey { .. })
        ));
    }

    #[gpui::test]
    async fn test_fatal_refresh_clears_auth_state(cx: &mut TestAppContext) {
        let http: Arc<dyn HttpClient> = FakeHttpClient::create(move |_request| async move {
            Ok(http_client::Response::builder()
                .status(401)
                .body(http_client::AsyncBody::from(r#"{"error":"invalid_grant"}"#))?)
        });
        let state = make_state(http.clone(), Some(make_expired_credentials()), cx);
        let weak_state = cx.read(|_cx| state.downgrade());
        let result = cx
            .spawn(async move |mut cx| get_fresh_credentials(&weak_state, &http, &mut cx).await)
            .await;

        cx.run_until_parked();
        assert!(result.is_err());
        cx.read(|cx| {
            let state = state.read(cx);
            assert!(state.credentials.is_none());
            assert!(state.last_auth_error.is_some());
        });
    }

    #[gpui::test]
    async fn test_fatal_refresh_persists_reauth_in_manifest(cx: &mut TestAppContext) {
        let http: Arc<dyn HttpClient> = FakeHttpClient::create(move |_request| async move {
            Ok(http_client::Response::builder()
                .status(401)
                .body(http_client::AsyncBody::from(r#"{"error":"invalid_grant"}"#))?)
        });
        let credentials_provider = Arc::new(FakeCredentialsProvider::new());
        let mut expired = make_expired_credentials();
        expired.account_id = Some("test-session".to_string());

        credentials_provider
            .write_credentials(
                &account_credentials_key("test-session"),
                "Bearer",
                &serde_json::to_vec(&expired).unwrap(),
                &cx.to_async(),
            )
            .await
            .unwrap();

        let manifest = AccountManifest {
            active_session_id: Some("test-session".to_string()),
            sessions: vec![AccountSessionMetadata {
                session_id: "test-session".to_string(),
                email: Some("test@example.com".to_string()),
                last_used_at_ms: now_ms(),
                reauthentication_required: false,
                ..Default::default()
            }],
        };
        credentials_provider
            .write_credentials(
                ACCOUNT_MANIFEST_KEY,
                "manifest",
                &serde_json::to_vec(&manifest).unwrap(),
                &cx.to_async(),
            )
            .await
            .unwrap();

        let state = cx.new(|_cx| State {
            manifest: manifest.clone(),
            credentials: Some(expired),
            cached_credentials: HashMap::new(),
            active_session_id: Some("test-session".to_string()),
            sign_in_task: None,
            refresh_task: None,
            refresh_tasks: HashMap::new(),
            load_task: None,
            credentials_provider: credentials_provider.clone(),
            http_client: http.clone(),
            auth_generation: 1,
            routing_generation: 0,
            last_auth_error: None,
            active_operations: Arc::new(AtomicUsize::new(0)),
            in_flight_tracker: SessionInFlightTracker::default(),
            consecutive_rate_limits: HashMap::new(),
            account_mutation_in_progress: false,
            account_mutation_seq: 0,
            quota_refresh_task: None,
        });

        let weak_state = cx.read(|_cx| state.downgrade());
        let result = cx
            .spawn(async move |mut cx| get_fresh_credentials(&weak_state, &http, &mut cx).await)
            .await;

        cx.run_until_parked();
        assert!(result.is_err());

        // Manifest in state must have reauthentication_required = true
        cx.read(|cx| {
            let state = state.read(cx);
            assert!(state.credentials.is_none());
            assert!(state.manifest.sessions[0].reauthentication_required);
        });

        // Manifest in persistent storage must also have reauthentication_required = true
        let stored_manifest_bytes = credentials_provider
            .read_credentials(ACCOUNT_MANIFEST_KEY, &cx.to_async())
            .await
            .unwrap()
            .expect("manifest must be persisted");
        let stored_manifest: AccountManifest =
            serde_json::from_slice(&stored_manifest_bytes.1).unwrap();
        assert!(stored_manifest.sessions[0].reauthentication_required);

        // Account credentials must be deleted from storage
        let stored_creds = credentials_provider
            .read_credentials(&account_credentials_key("test-session"), &cx.to_async())
            .await
            .unwrap();
        assert!(stored_creds.is_none());
    }

    #[gpui::test]
    async fn test_transient_refresh_keeps_credentials(cx: &mut TestAppContext) {
        let http: Arc<dyn HttpClient> = FakeHttpClient::create(move |_request| async move {
            Ok(http_client::Response::builder()
                .status(500)
                .body(http_client::AsyncBody::from("Internal Server Error"))?)
        });
        let state = make_state(http.clone(), Some(make_expired_credentials()), cx);
        let weak_state = cx.read(|_cx| state.downgrade());
        let result = cx
            .spawn(async move |mut cx| get_fresh_credentials(&weak_state, &http, &mut cx).await)
            .await;

        cx.run_until_parked();
        assert!(result.is_err());
        cx.read(|cx| {
            let state = state.read(cx);
            assert!(state.credentials.is_some());
            assert!(state.last_auth_error.is_none());
        });
    }

    #[gpui::test]
    async fn test_refresh_keeps_previous_refresh_token_when_omitted(cx: &mut TestAppContext) {
        let http: Arc<dyn HttpClient> = FakeHttpClient::create(move |_request| async move {
            let body = fake_token_response(false);
            Ok(http_client::Response::builder()
                .status(200)
                .body(http_client::AsyncBody::from(body))?)
        });
        let state = make_state(http.clone(), Some(make_expired_credentials()), cx);
        let weak_state = cx.read(|_cx| state.downgrade());
        let result = cx
            .spawn(async move |mut cx| get_fresh_credentials(&weak_state, &http, &mut cx).await)
            .await
            .expect("refresh should succeed");

        assert_eq!(result.access_token, "fresh_access");
        assert_eq!(result.refresh_token, "old_refresh");
    }

    #[gpui::test]
    async fn test_sign_out_during_refresh_discards_result(cx: &mut TestAppContext) {
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
                let body = fake_token_response(true);
                Ok(http_client::Response::builder()
                    .status(200)
                    .body(http_client::AsyncBody::from(body))?)
            }
        });

        let http: Arc<dyn HttpClient> = http_client;
        let state = make_state(http.clone(), Some(make_expired_credentials()), cx);
        let weak_state = cx.read(|_cx| state.downgrade());
        let refresh_task =
            cx.spawn(async move |mut cx| get_fresh_credentials(&weak_state, &http, &mut cx).await);

        cx.run_until_parked();
        state.update(cx, |state, cx| {
            state.sign_out(cx).detach();
        });
        cx.run_until_parked();
        let _ = gate_tx.send(());
        cx.run_until_parked();

        assert!(refresh_task.await.is_err());
        cx.read(|cx| {
            assert!(state.read(cx).credentials.is_none());
        });
    }

    #[gpui::test]
    async fn test_fatal_refresh_after_sign_out_keeps_new_session(cx: &mut TestAppContext) {
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

        let creds_provider = Arc::new(FakeCredentialsProvider::new());
        let http: Arc<dyn HttpClient> = http_client;
        let state = make_state_with_credentials_provider(
            http.clone(),
            Some(make_expired_credentials()),
            creds_provider.clone(),
            cx,
        );
        let weak_state = cx.read(|_cx| state.downgrade());
        let refresh_task =
            cx.spawn(async move |mut cx| get_fresh_credentials(&weak_state, &http, &mut cx).await);

        cx.run_until_parked();
        state.update(cx, |state, cx| {
            state.sign_out(cx).detach();
        });
        cx.run_until_parked();

        let new_creds = make_fresh_credentials();
        let new_creds_json = serde_json::to_vec(&new_creds).unwrap();
        creds_provider.storage.lock().insert(
            CREDENTIALS_KEY.to_string(),
            ("Bearer".to_string(), new_creds_json),
        );
        state.update(cx, |state, cx| {
            state.auth_generation = state.auth_generation.wrapping_add(1);
            state.credentials = Some(new_creds);
            cx.notify();
        });

        let _ = gate_tx.send(());
        cx.run_until_parked();

        assert!(refresh_task.await.is_err());
        cx.read(|cx| {
            let state = state.read(cx);
            assert!(state.is_authenticated());
            assert_eq!(
                state.credentials.as_ref().unwrap().access_token,
                "fresh_access"
            );
            assert!(state.last_auth_error.is_none());
        });
        assert!(creds_provider.storage.lock().contains_key(CREDENTIALS_KEY));
    }

    #[gpui::test]
    async fn test_sign_out_completes_fully(cx: &mut TestAppContext) {
        let creds_provider = Arc::new(FakeCredentialsProvider::new());
        creds_provider.storage.lock().insert(
            CREDENTIALS_KEY.to_string(),
            ("Bearer".to_string(), b"some-creds".to_vec()),
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

        let sign_out_task = state.update(cx, |state, cx| state.sign_out(cx));
        cx.run_until_parked();
        sign_out_task.await.expect("sign-out should succeed");

        assert!(!creds_provider.storage.lock().contains_key(CREDENTIALS_KEY));
        cx.read(|cx| {
            assert!(!state.read(cx).is_authenticated());
        });
    }

    #[gpui::test]
    async fn test_initial_load_restores_persisted_credentials(cx: &mut TestAppContext) {
        let creds = make_fresh_credentials();
        let creds_json = serde_json::to_vec(&creds).unwrap();
        let creds_provider = Arc::new(FakeCredentialsProvider::new());
        creds_provider.storage.lock().insert(
            CREDENTIALS_KEY.to_string(),
            ("Bearer".to_string(), creds_json),
        );

        let http: Arc<dyn HttpClient> = FakeHttpClient::create(|_| async {
            Ok(http_client::Response::builder()
                .status(200)
                .body(http_client::AsyncBody::default())?)
        });
        let state = cx.new(|cx| State::new(http, creds_provider, cx));
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
    async fn test_initial_load_falls_back_to_valid_account(cx: &mut TestAppContext) {
        let fallback_credentials = SuperGrokCredentials {
            access_token: "fallback_access".to_string(),
            refresh_token: "fallback_refresh".to_string(),
            expires_at_ms: now_ms() + 3_600_000,
            email: Some("fallback@example.com".to_string()),
            account_id: Some("session-fallback".to_string()),
        };
        let manifest = AccountManifest {
            active_session_id: Some("session-missing".to_string()),
            sessions: vec![
                AccountSessionMetadata {
                    session_id: "session-missing".to_string(),
                    email: Some("missing@example.com".to_string()),
                    last_used_at_ms: now_ms(),
                    reauthentication_required: false,
                    ..Default::default()
                },
                AccountSessionMetadata {
                    session_id: "session-fallback".to_string(),
                    email: fallback_credentials.email.clone(),
                    last_used_at_ms: now_ms() - 1000,
                    reauthentication_required: false,
                    ..Default::default()
                },
            ],
        };
        let credentials_provider = Arc::new(FakeCredentialsProvider::new());
        credentials_provider.storage.lock().insert(
            account_credentials_key("session-fallback"),
            (
                "Bearer".to_string(),
                serde_json::to_vec(&fallback_credentials).unwrap(),
            ),
        );
        credentials_provider.storage.lock().insert(
            ACCOUNT_MANIFEST_KEY.to_string(),
            (
                "manifest".to_string(),
                serde_json::to_vec(&manifest).unwrap(),
            ),
        );

        let http: Arc<dyn HttpClient> = FakeHttpClient::create(|_| async {
            Ok(http_client::Response::builder()
                .status(200)
                .body(http_client::AsyncBody::default())?)
        });
        let state = cx.new(|cx| State::new(http, credentials_provider.clone(), cx));
        let load_task = cx
            .read(|cx| state.read(cx).load_task())
            .expect("constructor should start the credentials load");

        cx.run_until_parked();
        load_task.await.expect("load should succeed");

        cx.read(|cx| {
            let state = state.read(cx);
            assert_eq!(state.active_session_id.as_deref(), Some("session-fallback"));
            assert_eq!(
                state
                    .credentials
                    .as_ref()
                    .map(|credentials| credentials.access_token.as_str()),
                Some("fallback_access")
            );
            assert_eq!(
                state.manifest.active_session_id.as_deref(),
                Some("session-fallback")
            );
            assert!(state.manifest.sessions[0].reauthentication_required);
        });
    }

    struct FakeCredentialsProvider {
        storage: Mutex<std::collections::HashMap<String, (String, Vec<u8>)>>,
        fail_write: std::sync::atomic::AtomicBool,
    }

    impl FakeCredentialsProvider {
        fn new() -> Self {
            Self {
                storage: Mutex::new(std::collections::HashMap::new()),
                fail_write: std::sync::atomic::AtomicBool::new(false),
            }
        }
    }

    impl CredentialsProvider for FakeCredentialsProvider {
        fn read_credentials<'a>(
            &'a self,
            url: &'a str,
            _cx: &'a AsyncApp,
        ) -> Pin<Box<dyn Future<Output = Result<Option<(String, Vec<u8>)>>> + 'a>> {
            let result = self.storage.lock().get(url).cloned();
            Box::pin(async move { Ok(result) })
        }

        fn write_credentials<'a>(
            &'a self,
            url: &'a str,
            username: &'a str,
            password: &'a [u8],
            _cx: &'a AsyncApp,
        ) -> Pin<Box<dyn Future<Output = Result<()>> + 'a>> {
            if self.fail_write.load(Ordering::SeqCst) {
                return Box::pin(async { Err(anyhow!("Keychain write failed")) });
            }
            self.storage
                .lock()
                .insert(url.to_string(), (username.to_string(), password.to_vec()));
            Box::pin(async { Ok(()) })
        }

        fn delete_credentials<'a>(
            &'a self,
            url: &'a str,
            _cx: &'a AsyncApp,
        ) -> Pin<Box<dyn Future<Output = Result<()>> + 'a>> {
            self.storage.lock().remove(url);
            Box::pin(async { Ok(()) })
        }
    }

    fn make_state(
        http_client: Arc<dyn HttpClient>,
        credentials: Option<SuperGrokCredentials>,
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
        credentials: Option<SuperGrokCredentials>,
        credentials_provider: Arc<dyn CredentialsProvider>,
        cx: &mut TestAppContext,
    ) -> Entity<State> {
        cx.new(|_cx| {
            let active_session_id = credentials.as_ref().map(|credentials| {
                credentials
                    .account_id
                    .clone()
                    .unwrap_or_else(|| account_session_id(credentials))
            });
            let mut manifest = AccountManifest::default();
            if let Some(session_id) = &active_session_id {
                manifest.active_session_id = Some(session_id.clone());
                manifest.sessions.push(AccountSessionMetadata {
                    session_id: session_id.clone(),
                    email: credentials.as_ref().and_then(|c| c.email.clone()),
                    login_order: 1,
                    last_used_at_ms: now_ms(),
                    reauthentication_required: false,
                    ..Default::default()
                });
            }
            State {
                manifest,
                credentials,
                cached_credentials: HashMap::new(),
                active_session_id,
                sign_in_task: None,
                refresh_task: None,
                refresh_tasks: HashMap::new(),
                load_task: None,
                credentials_provider,
                http_client,
                auth_generation: 0,
                routing_generation: 0,
                last_auth_error: None,
                active_operations: Arc::new(AtomicUsize::new(0)),
                in_flight_tracker: SessionInFlightTracker::default(),
                consecutive_rate_limits: HashMap::new(),
                account_mutation_in_progress: false,
                account_mutation_seq: 0,
                quota_refresh_task: None,
            }
        })
    }

    fn make_expired_credentials() -> SuperGrokCredentials {
        SuperGrokCredentials {
            access_token: "old_access".to_string(),
            refresh_token: "old_refresh".to_string(),
            expires_at_ms: 0,
            email: None,
            account_id: None,
        }
    }

    fn make_fresh_credentials() -> SuperGrokCredentials {
        SuperGrokCredentials {
            access_token: "fresh_access".to_string(),
            refresh_token: "fresh_refresh".to_string(),
            expires_at_ms: now_ms() + 3_600_000,
            email: None,
            account_id: None,
        }
    }

    fn fake_token_response(include_refresh_token: bool) -> String {
        let mut value = serde_json::json!({
            "access_token": "fresh_access",
            "expires_in": 3600
        });
        if include_refresh_token {
            value["refresh_token"] = serde_json::json!("fresh_refresh");
        }
        value.to_string()
    }

    #[test]
    fn test_active_operation_guard_tracks_busy_count() {
        let active_operations = Arc::new(AtomicUsize::new(0));
        assert_eq!(active_operations.load(Ordering::Acquire), 0);

        active_operations.fetch_add(1, Ordering::AcqRel);
        let guard1 = ActiveOperationGuard(active_operations.clone());
        assert_eq!(active_operations.load(Ordering::Acquire), 1);

        active_operations.fetch_add(1, Ordering::AcqRel);
        let guard2 = ActiveOperationGuard(active_operations.clone());
        assert_eq!(active_operations.load(Ordering::Acquire), 2);

        drop(guard1);
        assert_eq!(active_operations.load(Ordering::Acquire), 1);

        drop(guard2);
        assert_eq!(active_operations.load(Ordering::Acquire), 0);
    }

    #[gpui::test]
    async fn test_cancel_sign_in(cx: &mut TestAppContext) {
        let http: Arc<dyn HttpClient> = FakeHttpClient::create(|_| async {
            Ok(http_client::Response::builder()
                .status(200)
                .body(http_client::AsyncBody::default())?)
        });
        let state = make_state(http, None, cx);

        cx.update(|cx| {
            state.update(cx, |state, cx| {
                state.sign_in_task = Some(cx.spawn(async move |_, _| Ok::<(), anyhow::Error>(())));
                assert!(state.can_cancel_sign_in());
                state.cancel_sign_in(cx);
                assert!(!state.can_cancel_sign_in());
                assert!(!state.is_signing_in());
            });
        });
    }

    #[gpui::test]
    async fn test_sign_out_account_removes_inactive_account(cx: &mut TestAppContext) {
        let http: Arc<dyn HttpClient> = FakeHttpClient::create(|_| async {
            Ok(http_client::Response::builder()
                .status(200)
                .body(http_client::AsyncBody::default())?)
        });
        let credentials_provider = Arc::new(FakeCredentialsProvider::new());
        let active_creds = SuperGrokCredentials {
            access_token: "active_token".to_string(),
            refresh_token: "active_refresh".to_string(),
            expires_at_ms: now_ms() + 3_600_000,
            email: Some("active@example.com".to_string()),
            account_id: Some("session-active".to_string()),
        };
        let inactive_creds = SuperGrokCredentials {
            access_token: "inactive_token".to_string(),
            refresh_token: "inactive_refresh".to_string(),
            expires_at_ms: now_ms() + 3_600_000,
            email: Some("inactive@example.com".to_string()),
            account_id: Some("session-inactive".to_string()),
        };

        credentials_provider
            .write_credentials(
                &account_credentials_key("session-active"),
                "Bearer",
                &serde_json::to_vec(&active_creds).unwrap(),
                &cx.to_async(),
            )
            .await
            .unwrap();
        credentials_provider
            .write_credentials(
                &account_credentials_key("session-inactive"),
                "Bearer",
                &serde_json::to_vec(&inactive_creds).unwrap(),
                &cx.to_async(),
            )
            .await
            .unwrap();

        let manifest = AccountManifest {
            active_session_id: Some("session-active".to_string()),
            sessions: vec![
                AccountSessionMetadata {
                    session_id: "session-active".to_string(),
                    email: Some("active@example.com".to_string()),
                    last_used_at_ms: now_ms(),
                    reauthentication_required: false,
                    ..Default::default()
                },
                AccountSessionMetadata {
                    session_id: "session-inactive".to_string(),
                    email: Some("inactive@example.com".to_string()),
                    last_used_at_ms: now_ms() - 1000,
                    reauthentication_required: false,
                    ..Default::default()
                },
            ],
        };
        credentials_provider
            .write_credentials(
                ACCOUNT_MANIFEST_KEY,
                "manifest",
                &serde_json::to_vec(&manifest).unwrap(),
                &cx.to_async(),
            )
            .await
            .unwrap();

        let state = cx.new(|_cx| State {
            manifest: manifest.clone(),
            credentials: Some(active_creds),
            cached_credentials: HashMap::new(),
            active_session_id: Some("session-active".to_string()),
            sign_in_task: None,
            refresh_task: None,
            refresh_tasks: HashMap::new(),
            load_task: None,
            credentials_provider: credentials_provider.clone(),
            http_client: http,
            auth_generation: 1,
            routing_generation: 0,
            last_auth_error: None,
            active_operations: Arc::new(AtomicUsize::new(0)),
            in_flight_tracker: SessionInFlightTracker::default(),
            consecutive_rate_limits: HashMap::new(),
            account_mutation_in_progress: false,
            account_mutation_seq: 0,
            quota_refresh_task: None,
        });

        let task = cx.update(|cx| {
            state.update(cx, |state, cx| {
                state.sign_out_account("session-inactive".into(), cx)
            })
        });
        cx.run_until_parked();
        task.await.unwrap();

        cx.read(|cx| {
            let state = state.read(cx);
            assert_eq!(state.active_session_id.as_deref(), Some("session-active"));
            assert_eq!(state.account_summaries().len(), 1);
            assert_eq!(
                state.account_summaries()[0].session_id.as_ref(),
                "session-active"
            );
        });

        // Inactive account credentials must be deleted from storage
        let inactive_in_store = credentials_provider
            .read_credentials(&account_credentials_key("session-inactive"), &cx.to_async())
            .await
            .unwrap();
        assert!(inactive_in_store.is_none());
    }

    #[gpui::test]
    async fn test_account_switch_is_serialized(cx: &mut TestAppContext) {
        let http: Arc<dyn HttpClient> = FakeHttpClient::create(|_| async {
            Ok(http_client::Response::builder()
                .status(200)
                .body(http_client::AsyncBody::default())?)
        });
        let credentials_provider = Arc::new(FakeCredentialsProvider::new());
        let account_a = SuperGrokCredentials {
            access_token: "a_access".to_string(),
            refresh_token: "a_refresh".to_string(),
            expires_at_ms: now_ms() + 3_600_000,
            email: Some("a@example.com".to_string()),
            account_id: Some("session-a".to_string()),
        };
        let account_b = SuperGrokCredentials {
            access_token: "b_access".to_string(),
            refresh_token: "b_refresh".to_string(),
            expires_at_ms: now_ms() + 3_600_000,
            email: Some("b@example.com".to_string()),
            account_id: Some("session-b".to_string()),
        };
        credentials_provider.storage.lock().insert(
            account_credentials_key("session-b"),
            (
                "Bearer".to_string(),
                serde_json::to_vec(&account_b).unwrap(),
            ),
        );
        let manifest = AccountManifest {
            active_session_id: Some("session-a".to_string()),
            sessions: vec![
                AccountSessionMetadata {
                    session_id: "session-a".to_string(),
                    email: account_a.email.clone(),
                    last_used_at_ms: now_ms(),
                    reauthentication_required: false,
                    ..Default::default()
                },
                AccountSessionMetadata {
                    session_id: "session-b".to_string(),
                    email: account_b.email.clone(),
                    last_used_at_ms: now_ms() - 1000,
                    reauthentication_required: false,
                    ..Default::default()
                },
            ],
        };
        let state = cx.new(|_cx| State {
            manifest,
            credentials: Some(account_a),
            cached_credentials: HashMap::new(),
            active_session_id: Some("session-a".to_string()),
            sign_in_task: None,
            refresh_task: None,
            refresh_tasks: HashMap::new(),
            load_task: None,
            credentials_provider,
            http_client: http,
            auth_generation: 0,
            routing_generation: 0,
            last_auth_error: None,
            active_operations: Arc::new(AtomicUsize::new(0)),
            in_flight_tracker: SessionInFlightTracker::default(),
            consecutive_rate_limits: HashMap::new(),
            account_mutation_in_progress: false,
            account_mutation_seq: 0,
            quota_refresh_task: None,
        });

        let first_switch =
            state.update(cx, |state, cx| state.switch_account("session-b".into(), cx));
        assert!(cx.read(|cx| state.read(cx).is_busy()));
        let second_switch =
            state.update(cx, |state, cx| state.switch_account("session-a".into(), cx));
        assert!(second_switch.await.is_err());

        cx.run_until_parked();
        first_switch
            .await
            .expect("first account switch should succeed");
        cx.read(|cx| {
            let state = state.read(cx);
            assert_eq!(state.active_session_id.as_deref(), Some("session-b"));
            assert!(!state.is_busy());
        });
    }

    #[test]
    fn quota_snapshot_from_billing_percent() {
        let value = serde_json::json!({
            "config": {
                "creditUsagePercent": 35.4,
                "currentPeriod": { "end": "2026-05-31T19:00:00-05:00" },
                "subscription_tier_display": "SuperGrok"
            }
        });
        let quota = quota_snapshot_from_billing_value(&value).unwrap();
        assert!((quota.used_percent - 35.4).abs() < f64::EPSILON);
        assert_eq!(quota.plan_type.as_deref(), Some("SuperGrok"));
        assert_eq!(
            quota.resets_at,
            Some(
                chrono::DateTime::parse_from_rfc3339("2026-05-31T19:00:00-05:00")
                    .unwrap()
                    .timestamp()
            )
        );
    }

    #[test]
    fn quota_snapshot_from_billing_on_demand_ratio() {
        let value = serde_json::json!({
            "config": {
                "onDemandUsed": { "val": 25 },
                "onDemandCap": { "val": 100 }
            }
        });
        let quota = quota_snapshot_from_billing_value(&value).unwrap();
        assert!((quota.used_percent - 25.0).abs() < f64::EPSILON);
    }

    #[test]
    fn quota_snapshot_from_grok_credits_protobuf() {
        let payload = grok_credits_payload(42.5, Some(1_780_272_000));
        let quota = quota_snapshot_from_grok_credits(&payload).unwrap();
        assert!((quota.used_percent - 42.5).abs() < 0.01);
        assert_eq!(quota.resets_at, Some(1_780_272_000));
    }

    #[gpui::test]
    async fn test_fetch_quota_uses_cli_billing_json(cx: &mut TestAppContext) {
        let http: Arc<dyn HttpClient> =
            FakeHttpClient::create(|request| async move {
                assert_eq!(request.uri().host(), Some("cli-chat-proxy.grok.com"));
                assert_eq!(
                    request
                        .headers()
                        .get("x-xai-token-auth")
                        .map(|value| value.as_bytes()),
                    Some(&b"xai-grok-cli"[..])
                );
                Ok(http_client::Response::builder().status(200).body(
                    http_client::AsyncBody::from(r#"{"config":{"creditUsagePercent":12.0}}"#),
                )?)
            });
        let credentials = make_fresh_credentials();
        let quota = cx
            .spawn(async move |_cx| fetch_quota(http.as_ref(), &credentials).await)
            .await
            .unwrap();
        assert!((quota.used_percent - 12.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_sticky_priority_failover_account_order() {
        let now = now_ms();
        let session_1 = AccountSessionMetadata {
            session_id: "session-1".to_string(),
            login_order: 1,
            last_used_at_ms: now,
            reauthentication_required: false,
            ..Default::default()
        };
        let session_2 = AccountSessionMetadata {
            session_id: "session-2".to_string(),
            login_order: 2,
            last_used_at_ms: now,
            reauthentication_required: false,
            ..Default::default()
        };
        let session_3 = AccountSessionMetadata {
            session_id: "session-3".to_string(),
            login_order: 3,
            last_used_at_ms: now,
            reauthentication_required: false,
            ..Default::default()
        };

        let manifest = AccountManifest {
            active_session_id: Some("session-2".to_string()),
            sessions: vec![session_1, session_2, session_3],
        };

        let mut state = State {
            manifest,
            credentials: None,
            cached_credentials: HashMap::new(),
            active_session_id: Some("session-2".to_string()),
            sign_in_task: None,
            refresh_task: None,
            refresh_tasks: HashMap::new(),
            load_task: None,
            credentials_provider: Arc::new(FakeCredentialsProvider::new()),
            http_client: FakeHttpClient::create(|_| async {
                Ok(http_client::Response::builder()
                    .status(200)
                    .body(Default::default())?)
            }),
            auth_generation: 0,
            routing_generation: 0,
            last_auth_error: None,
            active_operations: Arc::new(AtomicUsize::new(0)),
            in_flight_tracker: SessionInFlightTracker::default(),
            consecutive_rate_limits: HashMap::new(),
            account_mutation_in_progress: false,
            account_mutation_seq: 0,
            quota_refresh_task: None,
        };

        // 1. Preferred active account ("session-2") must come first
        let eligible = state.eligible_accounts("grok-4.6");
        assert_eq!(eligible.len(), 3);
        assert_eq!(eligible[0].session_id, "session-2");
        assert_eq!(eligible[1].session_id, "session-1");
        assert_eq!(eligible[2].session_id, "session-3");

        // 2. Exclude session-2 -> failover chooses session-1 (lowest login_order)
        state.exclude_session(
            "session-2",
            AccountExclusion::RateLimited {
                retry_at_ms: now + 60_000,
                scope: AccountExclusionScope::Account,
            },
        );
        let eligible = state.eligible_accounts("grok-4.6");
        assert_eq!(eligible.len(), 2);
        assert_eq!(eligible[0].session_id, "session-1");
        assert_eq!(eligible[1].session_id, "session-3");

        // 3. Exclude session-1 -> failover chooses session-3
        state.exclude_session(
            "session-1",
            AccountExclusion::Forbidden {
                scope: AccountExclusionScope::Account,
            },
        );
        let eligible = state.eligible_accounts("grok-4.6");
        assert_eq!(eligible.len(), 1);
        assert_eq!(eligible[0].session_id, "session-3");

        // 4. Rate limit on session-2 expires -> session-2 becomes preferred active again
        state.manifest.sessions[1].clean_expired_exclusions(now + 70_000);
        let eligible = state.eligible_accounts("grok-4.6");
        assert_eq!(eligible.len(), 2);
        assert_eq!(eligible[0].session_id, "session-2");
        assert_eq!(eligible[1].session_id, "session-3");
    }

    #[test]
    fn test_exclusion_scope_model_vs_account() {
        let error_model = LanguageModelCompletionError::ProviderRejection {
            provider: PROVIDER_NAME,
            status: Some(http_client::StatusCode::FORBIDDEN),
            code: Some("forbidden".to_string()),
            message: "The model grok-4.6 is restricted on your plan".to_string(),
            retry_after: None,
            category: ProviderErrorCategory::Permission,
        };
        assert_eq!(
            classify_forbidden_scope(&error_model, "grok-4.6"),
            AccountExclusionScope::Model("grok-4.6".to_string())
        );

        let error_account = LanguageModelCompletionError::ProviderRejection {
            provider: PROVIDER_NAME,
            status: Some(http_client::StatusCode::FORBIDDEN),
            code: Some("forbidden".to_string()),
            message: "Inference access not enabled for this account".to_string(),
            retry_after: None,
            category: ProviderErrorCategory::Permission,
        };
        assert_eq!(
            classify_forbidden_scope(&error_account, "grok-4.6"),
            AccountExclusionScope::Account
        );

        let mut session = AccountSessionMetadata {
            session_id: "test-session".to_string(),
            login_order: 1,
            last_used_at_ms: now_ms(),
            reauthentication_required: false,
            ..Default::default()
        };

        // Model-specific exclusion
        session.add_exclusion(AccountExclusion::Forbidden {
            scope: AccountExclusionScope::Model("grok-4.6".to_string()),
        });
        assert!(session.is_excluded_for("grok-4.6", now_ms()));
        assert!(!session.is_excluded_for("grok-build-0.1", now_ms()));

        // Account-wide exclusion
        session.add_exclusion(AccountExclusion::Forbidden {
            scope: AccountExclusionScope::Account,
        });
        assert!(session.is_excluded_for("grok-4.6", now_ms()));
        assert!(session.is_excluded_for("grok-build-0.1", now_ms()));
    }

    #[test]
    fn test_failover_rate_limit_calculation() {
        let mut state = state_placeholder();
        let now = now_ms();

        // 1. Retry-After header
        let err_retry_after = LanguageModelCompletionError::ProviderRejection {
            provider: PROVIDER_NAME,
            status: Some(http_client::StatusCode::TOO_MANY_REQUESTS),
            code: Some("rate_limit_exceeded".to_string()),
            message: "Rate limit exceeded".to_string(),
            retry_after: Some(Duration::from_secs(15)),
            category: ProviderErrorCategory::RateLimit,
        };
        let (retry_at, scope) =
            state.calculate_rate_limit_retry("session-1", "grok-4.6", &err_retry_after);
        assert!((retry_at as i64 - (now + 15_000) as i64).abs() < 500);
        assert_eq!(scope, AccountExclusionScope::Model("grok-4.6".to_string()));

        // 2. Quota exhausted (used_percent >= QUOTA_EXHAUSTION_THRESHOLD_PERCENT)
        let reset_timestamp = (now / 1000 + 3600) as i64;
        state.manifest.sessions.push(AccountSessionMetadata {
            session_id: "session-exhausted".to_string(),
            login_order: 1,
            last_used_at_ms: now,
            reauthentication_required: false,
            quota: Some(QuotaSnapshot {
                used_percent: 100.0,
                resets_at: Some(reset_timestamp),
                plan_type: Some("SuperGrok".to_string()),
                captured_at_ms: now,
            }),
            ..Default::default()
        });

        let err_no_header = LanguageModelCompletionError::ProviderRejection {
            provider: PROVIDER_NAME,
            status: Some(http_client::StatusCode::TOO_MANY_REQUESTS),
            code: Some("rate_limit_exceeded".to_string()),
            message: "Rate limit exceeded".to_string(),
            retry_after: None,
            category: ProviderErrorCategory::RateLimit,
        };
        let (retry_at, scope) =
            state.calculate_rate_limit_retry("session-exhausted", "grok-4.6", &err_no_header);
        assert_eq!(retry_at, (reset_timestamp as u64) * 1000);
        assert_eq!(scope, AccountExclusionScope::Account);

        // 3. Fallback exponential backoff ladder: 30s -> 60s -> 300s
        let (retry_1, _) =
            state.calculate_rate_limit_retry("session-fallback", "grok-4.6", &err_no_header);
        assert!((retry_1 as i64 - (now + 30_000) as i64).abs() < 500);

        let (retry_2, _) =
            state.calculate_rate_limit_retry("session-fallback", "grok-4.6", &err_no_header);
        assert!((retry_2 as i64 - (now + 60_000) as i64).abs() < 500);

        let (retry_3, _) =
            state.calculate_rate_limit_retry("session-fallback", "grok-4.6", &err_no_header);
        assert!((retry_3 as i64 - (now + 300_000) as i64).abs() < 500);
    }

    #[gpui::test]
    async fn test_401_inference_refreshes_and_retries_same_account(cx: &mut TestAppContext) {
        let auth_calls = Arc::new(AtomicUsize::new(0));
        let completion_calls = Arc::new(AtomicUsize::new(0));

        let auth_calls_clone = auth_calls.clone();
        let completion_calls_clone = completion_calls.clone();

        let http: Arc<dyn HttpClient> = FakeHttpClient::create(move |request| {
            let auth_calls = auth_calls_clone.clone();
            let completion_calls = completion_calls_clone.clone();
            async move {
                let uri = request.uri().to_string();
                if uri.contains("/oauth2/token") || uri.contains("/oauth/token") {
                    auth_calls.fetch_add(1, Ordering::SeqCst);
                    let body = serde_json::json!({
                        "access_token": "refreshed_access_token",
                        "refresh_token": "same_refresh_token",
                        "expires_in": 3600
                    });
                    Ok(http_client::Response::builder().status(200).body(
                        http_client::AsyncBody::from(serde_json::to_vec(&body).unwrap()),
                    )?)
                } else if uri.contains("/chat/completions") {
                    let count = completion_calls.fetch_add(1, Ordering::SeqCst);
                    if count == 0 {
                        // First inference call returns 401 Unauthorized
                        Ok(http_client::Response::builder().status(401).body(
                            http_client::AsyncBody::from(r#"{"error":{"message":"Unauthorized"}}"#),
                        )?)
                    } else {
                        // Retry with refreshed token succeeds!
                        let auth_header = request
                            .headers()
                            .get("authorization")
                            .and_then(|v| v.to_str().ok());
                        assert_eq!(auth_header, Some("Bearer refreshed_access_token"));
                        Ok(http_client::Response::builder()
                            .status(200)
                            .body(http_client::AsyncBody::from(
                                "data: {\"choices\":[{\"delta\":{\"content\":\"Hello from refreshed account\"}}]}\n\ndata: [DONE]\n\n"
                            ))?)
                    }
                } else {
                    Ok(http_client::Response::builder()
                        .status(200)
                        .body(Default::default())?)
                }
            }
        });

        let credentials_provider = Arc::new(FakeCredentialsProvider::new());
        let account_a = SuperGrokCredentials {
            access_token: "old_access_token".to_string(),
            refresh_token: "refresh_a".to_string(),
            expires_at_ms: now_ms() + 3600_000,
            email: Some("a@example.com".to_string()),
            account_id: Some("session-a".to_string()),
        };
        let account_b = SuperGrokCredentials {
            access_token: "access_b".to_string(),
            refresh_token: "refresh_b".to_string(),
            expires_at_ms: now_ms() + 3600_000,
            email: Some("b@example.com".to_string()),
            account_id: Some("session-b".to_string()),
        };

        credentials_provider.storage.lock().insert(
            account_credentials_key("session-a"),
            (
                "Bearer".to_string(),
                serde_json::to_vec(&account_a).unwrap(),
            ),
        );
        credentials_provider.storage.lock().insert(
            account_credentials_key("session-b"),
            (
                "Bearer".to_string(),
                serde_json::to_vec(&account_b).unwrap(),
            ),
        );

        let manifest = AccountManifest {
            active_session_id: Some("session-a".to_string()),
            sessions: vec![
                AccountSessionMetadata {
                    session_id: "session-a".to_string(),
                    login_order: 1,
                    last_used_at_ms: now_ms(),
                    reauthentication_required: false,
                    ..Default::default()
                },
                AccountSessionMetadata {
                    session_id: "session-b".to_string(),
                    login_order: 2,
                    last_used_at_ms: now_ms() - 1000,
                    reauthentication_required: false,
                    ..Default::default()
                },
            ],
        };

        let mut cached_creds = HashMap::new();
        cached_creds.insert("session-a".to_string(), account_a.clone());
        cached_creds.insert("session-b".to_string(), account_b.clone());

        let state = cx.new(|_cx| State {
            manifest,
            credentials: Some(account_a),
            cached_credentials: cached_creds,
            active_session_id: Some("session-a".to_string()),
            sign_in_task: None,
            refresh_task: None,
            refresh_tasks: HashMap::new(),
            load_task: None,
            credentials_provider,
            http_client: http.clone(),
            auth_generation: 0,
            routing_generation: 0,
            last_auth_error: None,
            active_operations: Arc::new(AtomicUsize::new(0)),
            in_flight_tracker: SessionInFlightTracker::default(),
            consecutive_rate_limits: HashMap::new(),
            account_mutation_in_progress: false,
            account_mutation_seq: 0,
            quota_refresh_task: None,
        });

        let model = cx.read(|cx| create_language_model(SuperGrokModel::Grok46, &state, cx));
        let request = LanguageModelRequest {
            messages: vec![language_model::LanguageModelRequestMessage {
                role: language_model::Role::User,
                content: vec!["Hello".into()],
                cache: false,
                reasoning_details: None,
            }],
            ..Default::default()
        };

        let stream = cx
            .spawn({
                let model = model.clone();
                async move |cx| model.stream_completion(request, &cx).await
            })
            .await
            .unwrap();

        let events: Vec<_> = stream.collect().await;
        assert_eq!(completion_calls.load(Ordering::SeqCst), 2);
        assert_eq!(auth_calls.load(Ordering::SeqCst), 1);
        assert!(!events.is_empty());

        // Account A remains active and not excluded
        cx.read(|cx| {
            let state = state.read(cx);
            assert_eq!(state.active_session_id.as_deref(), Some("session-a"));
            assert!(!state.manifest.sessions[0].reauthentication_required);
        });
    }

    #[gpui::test]
    async fn test_401_fatal_refresh_marks_reauth_and_fails_over(cx: &mut TestAppContext) {
        let auth_calls = Arc::new(AtomicUsize::new(0));
        let completion_calls = Arc::new(AtomicUsize::new(0));

        let auth_calls_clone = auth_calls.clone();
        let completion_calls_clone = completion_calls.clone();

        let http: Arc<dyn HttpClient> = FakeHttpClient::create(move |request| {
            let auth_calls = auth_calls_clone.clone();
            let completion_calls = completion_calls_clone.clone();
            async move {
                let uri = request.uri().to_string();
                if uri.contains("/oauth2/token") || uri.contains("/oauth/token") {
                    auth_calls.fetch_add(1, Ordering::SeqCst);
                    // Fatal invalid_grant
                    Ok(http_client::Response::builder()
                        .status(400)
                        .body(http_client::AsyncBody::from(r#"{"error":"invalid_grant"}"#))?)
                } else if uri.contains("/chat/completions") {
                    let count = completion_calls.fetch_add(1, Ordering::SeqCst);
                    let auth_header = request
                        .headers()
                        .get("authorization")
                        .and_then(|v| v.to_str().ok());
                    if count == 0 {
                        // First call on Account A returns 401
                        assert_eq!(auth_header, Some("Bearer access_a"));
                        Ok(http_client::Response::builder().status(401).body(
                            http_client::AsyncBody::from(r#"{"error":{"message":"Unauthorized"}}"#),
                        )?)
                    } else {
                        // Failover call on Account B succeeds!
                        assert_eq!(auth_header, Some("Bearer access_b"));
                        Ok(http_client::Response::builder()
                            .status(200)
                            .body(http_client::AsyncBody::from(
                                "data: {\"choices\":[{\"delta\":{\"content\":\"Hello from Account B\"}}]}\n\ndata: [DONE]\n\n"
                            ))?)
                    }
                } else {
                    Ok(http_client::Response::builder()
                        .status(200)
                        .body(Default::default())?)
                }
            }
        });

        let credentials_provider = Arc::new(FakeCredentialsProvider::new());
        let account_a = SuperGrokCredentials {
            access_token: "access_a".to_string(),
            refresh_token: "refresh_a".to_string(),
            expires_at_ms: now_ms() + 3600_000,
            email: Some("a@example.com".to_string()),
            account_id: Some("session-a".to_string()),
        };
        let account_b = SuperGrokCredentials {
            access_token: "access_b".to_string(),
            refresh_token: "refresh_b".to_string(),
            expires_at_ms: now_ms() + 3600_000,
            email: Some("b@example.com".to_string()),
            account_id: Some("session-b".to_string()),
        };

        credentials_provider.storage.lock().insert(
            account_credentials_key("session-a"),
            (
                "Bearer".to_string(),
                serde_json::to_vec(&account_a).unwrap(),
            ),
        );
        credentials_provider.storage.lock().insert(
            account_credentials_key("session-b"),
            (
                "Bearer".to_string(),
                serde_json::to_vec(&account_b).unwrap(),
            ),
        );

        let manifest = AccountManifest {
            active_session_id: Some("session-a".to_string()),
            sessions: vec![
                AccountSessionMetadata {
                    session_id: "session-a".to_string(),
                    login_order: 1,
                    last_used_at_ms: now_ms(),
                    reauthentication_required: false,
                    ..Default::default()
                },
                AccountSessionMetadata {
                    session_id: "session-b".to_string(),
                    login_order: 2,
                    last_used_at_ms: now_ms() - 1000,
                    reauthentication_required: false,
                    ..Default::default()
                },
            ],
        };

        let mut cached_creds = HashMap::new();
        cached_creds.insert("session-a".to_string(), account_a.clone());
        cached_creds.insert("session-b".to_string(), account_b.clone());

        let state = cx.new(|_cx| State {
            manifest,
            credentials: Some(account_a),
            cached_credentials: cached_creds,
            active_session_id: Some("session-a".to_string()),
            sign_in_task: None,
            refresh_task: None,
            refresh_tasks: HashMap::new(),
            load_task: None,
            credentials_provider,
            http_client: http.clone(),
            auth_generation: 0,
            routing_generation: 0,
            last_auth_error: None,
            active_operations: Arc::new(AtomicUsize::new(0)),
            in_flight_tracker: SessionInFlightTracker::default(),
            consecutive_rate_limits: HashMap::new(),
            account_mutation_in_progress: false,
            account_mutation_seq: 0,
            quota_refresh_task: None,
        });

        let model = cx.read(|cx| create_language_model(SuperGrokModel::Grok46, &state, cx));
        let request = LanguageModelRequest {
            messages: vec![language_model::LanguageModelRequestMessage {
                role: language_model::Role::User,
                content: vec!["Hello".into()],
                cache: false,
                reasoning_details: None,
            }],
            ..Default::default()
        };

        let stream = cx
            .spawn({
                let model = model.clone();
                async move |cx| model.stream_completion(request, &cx).await
            })
            .await
            .unwrap();

        let events: Vec<_> = stream.collect().await;
        assert_eq!(completion_calls.load(Ordering::SeqCst), 2);
        assert_eq!(auth_calls.load(Ordering::SeqCst), 1);
        assert!(!events.is_empty());

        // Account A was marked reauthentication_required
        cx.read(|cx| {
            let state = state.read(cx);
            assert_eq!(state.active_session_id.as_deref(), Some("session-a"));
            assert!(state.manifest.sessions[0].reauthentication_required);
            assert!(!state.manifest.sessions[1].reauthentication_required);
        });
    }

    #[gpui::test]
    async fn test_403_and_429_failover_to_backup_account(cx: &mut TestAppContext) {
        let completion_calls = Arc::new(AtomicUsize::new(0));
        let completion_calls_clone = completion_calls.clone();

        let http: Arc<dyn HttpClient> = FakeHttpClient::create(move |request| {
            let completion_calls = completion_calls_clone.clone();
            async move {
                let uri = request.uri().to_string();
                if uri.contains("/chat/completions") {
                    let count = completion_calls.fetch_add(1, Ordering::SeqCst);
                    let auth_header = request
                        .headers()
                        .get("authorization")
                        .and_then(|v| v.to_str().ok());
                    if count == 0 {
                        assert_eq!(auth_header, Some("Bearer access_a"));
                        // 429 Too Many Requests
                        Ok(http_client::Response::builder()
                            .status(429)
                            .header("Retry-After", "30")
                            .body(http_client::AsyncBody::from(
                                r#"{"error":{"message":"Rate limited"}}"#,
                            ))?)
                    } else {
                        // Failover to Account B succeeds
                        assert_eq!(auth_header, Some("Bearer access_b"));
                        Ok(http_client::Response::builder()
                            .status(200)
                            .body(http_client::AsyncBody::from(
                                "data: {\"choices\":[{\"delta\":{\"content\":\"Hello from B\"}}]}\n\ndata: [DONE]\n\n"
                            ))?)
                    }
                } else {
                    Ok(http_client::Response::builder()
                        .status(200)
                        .body(Default::default())?)
                }
            }
        });

        let credentials_provider = Arc::new(FakeCredentialsProvider::new());
        let account_a = SuperGrokCredentials {
            access_token: "access_a".to_string(),
            refresh_token: "refresh_a".to_string(),
            expires_at_ms: now_ms() + 3600_000,
            email: Some("a@example.com".to_string()),
            account_id: Some("session-a".to_string()),
        };
        let account_b = SuperGrokCredentials {
            access_token: "access_b".to_string(),
            refresh_token: "refresh_b".to_string(),
            expires_at_ms: now_ms() + 3600_000,
            email: Some("b@example.com".to_string()),
            account_id: Some("session-b".to_string()),
        };

        credentials_provider.storage.lock().insert(
            account_credentials_key("session-a"),
            (
                "Bearer".to_string(),
                serde_json::to_vec(&account_a).unwrap(),
            ),
        );
        credentials_provider.storage.lock().insert(
            account_credentials_key("session-b"),
            (
                "Bearer".to_string(),
                serde_json::to_vec(&account_b).unwrap(),
            ),
        );

        let manifest = AccountManifest {
            active_session_id: Some("session-a".to_string()),
            sessions: vec![
                AccountSessionMetadata {
                    session_id: "session-a".to_string(),
                    login_order: 1,
                    last_used_at_ms: now_ms(),
                    reauthentication_required: false,
                    ..Default::default()
                },
                AccountSessionMetadata {
                    session_id: "session-b".to_string(),
                    login_order: 2,
                    last_used_at_ms: now_ms() - 1000,
                    reauthentication_required: false,
                    ..Default::default()
                },
            ],
        };

        let mut cached_creds = HashMap::new();
        cached_creds.insert("session-a".to_string(), account_a.clone());
        cached_creds.insert("session-b".to_string(), account_b.clone());

        let state = cx.new(|_cx| State {
            manifest,
            credentials: Some(account_a),
            cached_credentials: cached_creds,
            active_session_id: Some("session-a".to_string()),
            sign_in_task: None,
            refresh_task: None,
            refresh_tasks: HashMap::new(),
            load_task: None,
            credentials_provider,
            http_client: http.clone(),
            auth_generation: 0,
            routing_generation: 0,
            last_auth_error: None,
            active_operations: Arc::new(AtomicUsize::new(0)),
            in_flight_tracker: SessionInFlightTracker::default(),
            consecutive_rate_limits: HashMap::new(),
            account_mutation_in_progress: false,
            account_mutation_seq: 0,
            quota_refresh_task: None,
        });

        let model = cx.read(|cx| create_language_model(SuperGrokModel::Grok46, &state, cx));
        let request = LanguageModelRequest {
            messages: vec![language_model::LanguageModelRequestMessage {
                role: language_model::Role::User,
                content: vec!["Hello".into()],
                cache: false,
                reasoning_details: None,
            }],
            ..Default::default()
        };

        let stream = cx
            .spawn({
                let model = model.clone();
                async move |cx| model.stream_completion(request, &cx).await
            })
            .await
            .unwrap();

        let events: Vec<_> = stream.collect().await;
        assert_eq!(completion_calls.load(Ordering::SeqCst), 2);
        assert!(!events.is_empty());

        // Account A is rate limited, Account B served request, active UI session remains session-a
        cx.read(|cx| {
            let state = state.read(cx);
            assert_eq!(state.active_session_id.as_deref(), Some("session-a"));
            assert!(state.manifest.sessions[0].is_excluded_for("grok-4.6", now_ms()));
            assert!(!state.manifest.sessions[1].is_excluded_for("grok-4.6", now_ms()));
        });
    }

    #[gpui::test]
    async fn test_quota_monitor_readmits_rate_limited_account(cx: &mut TestAppContext) {
        let http: Arc<dyn HttpClient> =
            FakeHttpClient::create(|_| async {
                Ok(http_client::Response::builder().status(200).body(
                    http_client::AsyncBody::from(r#"{"config":{"creditUsagePercent":35.0}}"#),
                )?)
            });

        let credentials_provider = Arc::new(FakeCredentialsProvider::new());
        let account_a = SuperGrokCredentials {
            access_token: "access_a".to_string(),
            refresh_token: "refresh_a".to_string(),
            expires_at_ms: now_ms() + 3600_000,
            email: Some("a@example.com".to_string()),
            account_id: Some("session-a".to_string()),
        };
        credentials_provider.storage.lock().insert(
            account_credentials_key("session-a"),
            (
                "Bearer".to_string(),
                serde_json::to_vec(&account_a).unwrap(),
            ),
        );

        let mut session_a = AccountSessionMetadata {
            session_id: "session-a".to_string(),
            login_order: 1,
            last_used_at_ms: now_ms(),
            reauthentication_required: false,
            ..Default::default()
        };
        session_a.add_exclusion(AccountExclusion::RateLimited {
            retry_at_ms: now_ms() + 300_000,
            scope: AccountExclusionScope::Account,
        });

        let manifest = AccountManifest {
            active_session_id: Some("session-a".to_string()),
            sessions: vec![session_a],
        };

        let state = cx.new(|_cx| State {
            manifest,
            credentials: Some(account_a.clone()),
            cached_credentials: HashMap::new(),
            active_session_id: Some("session-a".to_string()),
            sign_in_task: None,
            refresh_task: None,
            refresh_tasks: HashMap::new(),
            load_task: None,
            credentials_provider,
            http_client: http,
            auth_generation: 0,
            routing_generation: 0,
            last_auth_error: None,
            active_operations: Arc::new(AtomicUsize::new(0)),
            in_flight_tracker: SessionInFlightTracker::default(),
            consecutive_rate_limits: HashMap::new(),
            account_mutation_in_progress: false,
            account_mutation_seq: 0,
            quota_refresh_task: None,
        });

        cx.read(|cx| {
            let state = state.read(cx);
            assert!(state.manifest.sessions[0].is_excluded_for("grok-4.6", now_ms()));
        });

        let weak_state = cx.read(|_cx| state.downgrade());
        cx.spawn(async move |mut cx| {
            refresh_all_account_quotas(&weak_state, &mut cx)
                .await
                .unwrap();
        })
        .await;

        cx.read(|cx| {
            let state = state.read(cx);
            assert!(!state.manifest.sessions[0].is_excluded_for("grok-4.6", now_ms()));
            assert_eq!(
                state.manifest.sessions[0]
                    .quota
                    .as_ref()
                    .map(|q| q.used_percent),
                Some(35.0)
            );
        });
    }

    #[test]
    fn test_clear_reauth_restores_eligibility() {
        let mut session = AccountSessionMetadata {
            session_id: "test-session".to_string(),
            login_order: 1,
            last_used_at_ms: now_ms(),
            reauthentication_required: true,
            ..Default::default()
        };
        session.add_exclusion(AccountExclusion::ReauthenticationRequired);
        assert!(session.is_excluded_for("grok-4.6", now_ms()));

        session.clear_reauthentication_required();
        assert!(!session.reauthentication_required);
        assert!(!session.is_excluded_for("grok-4.6", now_ms()));
    }

    #[gpui::test]
    async fn test_switch_account_does_not_clear_reauth(cx: &mut TestAppContext) {
        let http: Arc<dyn HttpClient> = FakeHttpClient::create(|_| async {
            Ok(http_client::Response::builder()
                .status(200)
                .body(Default::default())?)
        });
        let credentials_provider = Arc::new(FakeCredentialsProvider::new());
        let account_b = SuperGrokCredentials {
            access_token: "access_b".to_string(),
            refresh_token: "refresh_b".to_string(),
            expires_at_ms: now_ms() + 3600_000,
            email: Some("b@example.com".to_string()),
            account_id: Some("session-b".to_string()),
        };
        credentials_provider.storage.lock().insert(
            account_credentials_key("session-b"),
            (
                "Bearer".to_string(),
                serde_json::to_vec(&account_b).unwrap(),
            ),
        );

        let manifest = AccountManifest {
            active_session_id: Some("session-a".to_string()),
            sessions: vec![
                AccountSessionMetadata {
                    session_id: "session-a".to_string(),
                    login_order: 1,
                    last_used_at_ms: now_ms(),
                    reauthentication_required: false,
                    ..Default::default()
                },
                AccountSessionMetadata {
                    session_id: "session-b".to_string(),
                    login_order: 2,
                    last_used_at_ms: now_ms() - 1000,
                    reauthentication_required: true,
                    exclusion: Some(AccountExclusion::ReauthenticationRequired),
                    exclusions: vec![AccountExclusion::ReauthenticationRequired],
                    ..Default::default()
                },
            ],
        };

        let state = cx.new(|_cx| State {
            manifest,
            credentials: None,
            cached_credentials: HashMap::new(),
            active_session_id: Some("session-a".to_string()),
            sign_in_task: None,
            refresh_task: None,
            refresh_tasks: HashMap::new(),
            load_task: None,
            credentials_provider,
            http_client: http,
            auth_generation: 0,
            routing_generation: 0,
            last_auth_error: None,
            active_operations: Arc::new(AtomicUsize::new(0)),
            in_flight_tracker: SessionInFlightTracker::default(),
            consecutive_rate_limits: HashMap::new(),
            account_mutation_in_progress: false,
            account_mutation_seq: 0,
            quota_refresh_task: None,
        });

        // Switch to session-b
        state
            .update(cx, |state, cx| state.switch_account("session-b".into(), cx))
            .await
            .unwrap();

        cx.read(|cx| {
            let state = state.read(cx);
            assert_eq!(state.active_session_id.as_deref(), Some("session-b"));
            // Reauth requirement must NOT be cleared by switch_account!
            assert!(state.manifest.sessions[1].reauthentication_required);
            assert!(state.manifest.sessions[1].is_excluded_for("grok-4.6", now_ms()));
        });
    }

    #[gpui::test]
    async fn test_switch_account_propagates_keychain_write_failure(cx: &mut TestAppContext) {
        let http: Arc<dyn HttpClient> = FakeHttpClient::create(|_| async {
            Ok(http_client::Response::builder()
                .status(200)
                .body(Default::default())?)
        });
        let credentials_provider = Arc::new(FakeCredentialsProvider::new());
        let account_b = SuperGrokCredentials {
            access_token: "access_b".to_string(),
            refresh_token: "refresh_b".to_string(),
            expires_at_ms: now_ms() + 3600_000,
            email: Some("b@example.com".to_string()),
            account_id: Some("session-b".to_string()),
        };
        credentials_provider.storage.lock().insert(
            account_credentials_key("session-b"),
            (
                "Bearer".to_string(),
                serde_json::to_vec(&account_b).unwrap(),
            ),
        );

        let manifest = AccountManifest {
            active_session_id: Some("session-a".to_string()),
            sessions: vec![
                AccountSessionMetadata {
                    session_id: "session-a".to_string(),
                    login_order: 1,
                    last_used_at_ms: now_ms(),
                    reauthentication_required: false,
                    ..Default::default()
                },
                AccountSessionMetadata {
                    session_id: "session-b".to_string(),
                    login_order: 2,
                    last_used_at_ms: now_ms() - 1000,
                    reauthentication_required: false,
                    ..Default::default()
                },
            ],
        };

        let state = cx.new(|_cx| State {
            manifest,
            credentials: None,
            cached_credentials: HashMap::new(),
            active_session_id: Some("session-a".to_string()),
            sign_in_task: None,
            refresh_task: None,
            refresh_tasks: HashMap::new(),
            load_task: None,
            credentials_provider: credentials_provider.clone(),
            http_client: http,
            auth_generation: 0,
            routing_generation: 0,
            last_auth_error: None,
            active_operations: Arc::new(AtomicUsize::new(0)),
            in_flight_tracker: SessionInFlightTracker::default(),
            consecutive_rate_limits: HashMap::new(),
            account_mutation_in_progress: false,
            account_mutation_seq: 0,
            quota_refresh_task: None,
        });

        // Force keychain write failure
        credentials_provider
            .fail_write
            .store(true, Ordering::SeqCst);

        // Switch to session-b should fail and propagate error
        let switch_result = state
            .update(cx, |state, cx| state.switch_account("session-b".into(), cx))
            .await;
        assert!(switch_result.is_err());

        // In-memory state must NOT be changed!
        cx.read(|cx| {
            let state = state.read(cx);
            assert_eq!(state.active_session_id.as_deref(), Some("session-a"));
            assert!(!state.account_mutation_in_progress);
        });
    }

    #[test]
    fn test_quota_exhaustion_immediately_excludes_account() {
        let now = now_ms();
        let session = AccountSessionMetadata {
            session_id: "session-exhausted".to_string(),
            login_order: 1,
            last_used_at_ms: now,
            reauthentication_required: false,
            quota: Some(QuotaSnapshot {
                used_percent: 100.0,
                resets_at: Some((now / 1000 + 3600) as i64),
                plan_type: Some("SuperGrok".to_string()),
                captured_at_ms: now,
            }),
            ..Default::default()
        };

        assert!(session.is_excluded_for("grok-4.6", now));
        assert!(!session.is_eligible_for("grok-4.6", now));
    }

    #[test]
    fn test_quota_exhaustion_threshold_100_percent() {
        let now = now_ms();
        let session_99_5 = AccountSessionMetadata {
            session_id: "session-almost-full".to_string(),
            login_order: 1,
            last_used_at_ms: now,
            reauthentication_required: false,
            quota: Some(QuotaSnapshot {
                used_percent: 99.5,
                resets_at: Some((now / 1000 + 3600) as i64),
                plan_type: Some("SuperGrok".to_string()),
                captured_at_ms: now,
            }),
            ..Default::default()
        };
        // Under 100.0% threshold, 99.5% used is still eligible to consume the last 0.5%!
        assert!(session_99_5.is_eligible_for("grok-4.6", now));

        let session_100 = AccountSessionMetadata {
            session_id: "session-fully-exhausted".to_string(),
            login_order: 2,
            last_used_at_ms: now,
            reauthentication_required: false,
            quota: Some(QuotaSnapshot {
                used_percent: 100.0,
                resets_at: Some((now / 1000 + 3600) as i64),
                plan_type: Some("SuperGrok".to_string()),
                captured_at_ms: now,
            }),
            ..Default::default()
        };
        // At 100.0%, it is immediately excluded!
        assert!(session_100.is_excluded_for("grok-4.6", now));
        assert!(!session_100.is_eligible_for("grok-4.6", now));
    }

    #[test]
    fn test_quota_without_reset_uses_explicit_cooldown() {
        let now = now_ms();
        let mut session = AccountSessionMetadata {
            session_id: "session-without-reset".to_string(),
            login_order: 1,
            last_used_at_ms: now,
            quota: Some(QuotaSnapshot {
                used_percent: 100.0,
                resets_at: None,
                plan_type: None,
                captured_at_ms: now,
            }),
            ..Default::default()
        };

        assert!(session.is_eligible_for("grok-4.6", now));
        session.add_exclusion(AccountExclusion::RateLimited {
            retry_at_ms: now + 300_000,
            scope: AccountExclusionScope::Account,
        });
        assert!(!session.is_eligible_for("grok-4.6", now));
        assert!(session.is_eligible_for("grok-4.6", now + 300_001));
    }

    #[gpui::test]
    async fn test_token_refresh_during_sign_out_does_not_resurrect(cx: &mut TestAppContext) {
        let http: Arc<dyn HttpClient> = FakeHttpClient::create(|request| async move {
            let uri = request.uri().to_string();
            if uri.contains("/oauth2/token") || uri.contains("/oauth/token") {
                let body = serde_json::json!({
                    "access_token": "resurrected_access_token",
                    "refresh_token": "resurrected_refresh_token",
                    "expires_in": 3600
                });
                Ok(http_client::Response::builder().status(200).body(
                    http_client::AsyncBody::from(serde_json::to_vec(&body).unwrap()),
                )?)
            } else {
                Ok(http_client::Response::builder()
                    .status(200)
                    .body(Default::default())?)
            }
        });

        let credentials_provider = Arc::new(FakeCredentialsProvider::new());
        let account_a = SuperGrokCredentials {
            access_token: "old_token".to_string(),
            refresh_token: "old_refresh".to_string(),
            expires_at_ms: now_ms() - 1000, // Expired!
            email: Some("a@example.com".to_string()),
            account_id: Some("session-a".to_string()),
        };
        credentials_provider.storage.lock().insert(
            account_credentials_key("session-a"),
            (
                "Bearer".to_string(),
                serde_json::to_vec(&account_a).unwrap(),
            ),
        );

        let manifest = AccountManifest {
            active_session_id: Some("session-a".to_string()),
            sessions: vec![AccountSessionMetadata {
                session_id: "session-a".to_string(),
                login_order: 1,
                last_used_at_ms: now_ms(),
                reauthentication_required: false,
                ..Default::default()
            }],
        };

        let mut cached_creds = HashMap::new();
        cached_creds.insert("session-a".to_string(), account_a.clone());

        let state = cx.new(|_cx| State {
            manifest,
            credentials: Some(account_a),
            cached_credentials: cached_creds,
            active_session_id: Some("session-a".to_string()),
            sign_in_task: None,
            refresh_task: None,
            refresh_tasks: HashMap::new(),
            load_task: None,
            credentials_provider: credentials_provider.clone(),
            http_client: http.clone(),
            auth_generation: 0,
            routing_generation: 0,
            last_auth_error: None,
            active_operations: Arc::new(AtomicUsize::new(0)),
            in_flight_tracker: SessionInFlightTracker::default(),
            consecutive_rate_limits: HashMap::new(),
            account_mutation_in_progress: false,
            account_mutation_seq: 0,
            quota_refresh_task: None,
        });

        // Simulate user signing out session-a!
        state
            .update(cx, |state, cx| {
                state.sign_out_account("session-a".into(), cx)
            })
            .await
            .unwrap();

        // Now if a background refresh task tries to force refresh session-a:
        let weak_state = cx.read(|_cx| state.downgrade());
        let refresh_result = cx
            .spawn(async move |mut cx| {
                force_refresh_session_credentials(&weak_state, &http, "session-a", &mut cx).await
            })
            .await;

        // Refresh must fail because session-a was signed out!
        assert!(refresh_result.is_err());

        // Credential must NOT be resurrected into cache!
        cx.read(|cx| {
            let state = state.read(cx);
            assert!(!state.cached_credentials.contains_key("session-a"));
            assert!(state.credentials.is_none());
        });

        // Credential must NOT be in credentials_provider storage!
        assert!(
            !credentials_provider
                .storage
                .lock()
                .contains_key(&account_credentials_key("session-a"))
        );
    }

    #[gpui::test]
    async fn test_sign_out_inactive_account_bumps_auth_generation(cx: &mut TestAppContext) {
        let manifest = AccountManifest {
            active_session_id: Some("session-a".to_string()),
            sessions: vec![
                AccountSessionMetadata {
                    session_id: "session-a".to_string(),
                    login_order: 1,
                    last_used_at_ms: now_ms(),
                    reauthentication_required: false,
                    ..Default::default()
                },
                AccountSessionMetadata {
                    session_id: "session-b".to_string(),
                    login_order: 2,
                    last_used_at_ms: now_ms(),
                    reauthentication_required: false,
                    ..Default::default()
                },
            ],
        };

        let state = cx.new(|_cx| State {
            manifest,
            credentials: None,
            cached_credentials: HashMap::new(),
            active_session_id: Some("session-a".to_string()),
            sign_in_task: None,
            refresh_task: None,
            refresh_tasks: HashMap::new(),
            load_task: None,
            credentials_provider: Arc::new(FakeCredentialsProvider::new()),
            http_client: FakeHttpClient::create(|_| async {
                Ok(http_client::Response::builder()
                    .status(200)
                    .body(Default::default())?)
            }),
            auth_generation: 0,
            routing_generation: 0,
            last_auth_error: None,
            active_operations: Arc::new(AtomicUsize::new(0)),
            in_flight_tracker: SessionInFlightTracker::default(),
            consecutive_rate_limits: HashMap::new(),
            account_mutation_in_progress: false,
            account_mutation_seq: 0,
            quota_refresh_task: None,
        });

        let initial_auth_generation = cx.read(|cx| state.read(cx).auth_generation);
        state
            .update(cx, |state, cx| {
                state.sign_out_account("session-b".into(), cx)
            })
            .await
            .unwrap();

        // Auth generation must be bumped even for inactive account sign-out!
        cx.read(|cx| {
            let state = state.read(cx);
            assert_eq!(state.auth_generation, initial_auth_generation + 1);
            assert_eq!(state.manifest.sessions.len(), 1);
            assert_eq!(state.manifest.sessions[0].session_id, "session-a");
        });
    }

    #[gpui::test]
    async fn test_quota_monitor_preserves_model_level_retry_after(cx: &mut TestAppContext) {
        let http: Arc<dyn HttpClient> =
            FakeHttpClient::create(|_| async {
                Ok(http_client::Response::builder().status(200).body(
                    http_client::AsyncBody::from(r#"{"config":{"creditUsagePercent":35.0}}"#),
                )?)
            });

        let credentials_provider = Arc::new(FakeCredentialsProvider::new());
        let account_a = SuperGrokCredentials {
            access_token: "access_a".to_string(),
            refresh_token: "refresh_a".to_string(),
            expires_at_ms: now_ms() + 3600_000,
            email: Some("a@example.com".to_string()),
            account_id: Some("session-a".to_string()),
        };
        credentials_provider.storage.lock().insert(
            account_credentials_key("session-a"),
            (
                "Bearer".to_string(),
                serde_json::to_vec(&account_a).unwrap(),
            ),
        );

        let mut session_a = AccountSessionMetadata {
            session_id: "session-a".to_string(),
            login_order: 1,
            last_used_at_ms: now_ms(),
            reauthentication_required: false,
            ..Default::default()
        };
        // Model-scoped 429 Retry-After cooldown (5 minutes)
        session_a.add_exclusion(AccountExclusion::RateLimited {
            retry_at_ms: now_ms() + 300_000,
            scope: AccountExclusionScope::Model("grok-4.6".to_string()),
        });

        let manifest = AccountManifest {
            active_session_id: Some("session-a".to_string()),
            sessions: vec![session_a],
        };

        let state = cx.new(|_cx| State {
            manifest,
            credentials: Some(account_a.clone()),
            cached_credentials: HashMap::new(),
            active_session_id: Some("session-a".to_string()),
            sign_in_task: None,
            refresh_task: None,
            refresh_tasks: HashMap::new(),
            load_task: None,
            credentials_provider,
            http_client: http,
            auth_generation: 0,
            routing_generation: 0,
            last_auth_error: None,
            active_operations: Arc::new(AtomicUsize::new(0)),
            in_flight_tracker: SessionInFlightTracker::default(),
            consecutive_rate_limits: HashMap::new(),
            account_mutation_in_progress: false,
            account_mutation_seq: 0,
            quota_refresh_task: None,
        });

        let weak_state = cx.read(|_cx| state.downgrade());
        cx.spawn(async move |mut cx| {
            refresh_all_account_quotas(&weak_state, &mut cx)
                .await
                .unwrap();
        })
        .await;

        // Model-level rate limit must still be preserved after quota refresh!
        cx.read(|cx| {
            let state = state.read(cx);
            assert!(state.manifest.sessions[0].is_excluded_for("grok-4.6", now_ms()));
            // But it is not excluded for other models!
            assert!(!state.manifest.sessions[0].is_excluded_for("grok-build-0.1", now_ms()));
        });
    }

    #[test]
    fn test_concurrency_lease_tracks_active_sessions() {
        let tracker = SessionInFlightTracker::default();
        assert_eq!(tracker.count("session-1"), 0);

        let lease_1 = tracker.acquire("session-1");
        assert_eq!(tracker.count("session-1"), 1);

        let lease_2 = tracker.acquire("session-1");
        assert_eq!(tracker.count("session-1"), 2);

        drop(lease_1);
        assert_eq!(tracker.count("session-1"), 1);

        drop(lease_2);
        assert_eq!(tracker.count("session-1"), 0);
    }

    fn state_placeholder() -> State {
        State {
            manifest: AccountManifest::default(),
            credentials: None,
            cached_credentials: HashMap::new(),
            active_session_id: None,
            sign_in_task: None,
            refresh_task: None,
            refresh_tasks: HashMap::new(),
            load_task: None,
            credentials_provider: Arc::new(FakeCredentialsProvider::new()),
            http_client: FakeHttpClient::create(|_| async {
                Ok(http_client::Response::builder()
                    .status(200)
                    .body(Default::default())?)
            }),
            auth_generation: 0,
            routing_generation: 0,
            last_auth_error: None,
            active_operations: Arc::new(AtomicUsize::new(0)),
            in_flight_tracker: SessionInFlightTracker::default(),
            consecutive_rate_limits: HashMap::new(),
            account_mutation_in_progress: false,
            account_mutation_seq: 0,
            quota_refresh_task: None,
        }
    }

    fn grok_credits_payload(used_percent: f32, resets_at: Option<i64>) -> Vec<u8> {
        let mut config = Vec::new();
        config.push((1 << 3) | 5);
        config.extend_from_slice(&used_percent.to_le_bytes());
        if let Some(seconds) = resets_at {
            let mut timestamp = Vec::new();
            timestamp.push(1 << 3);
            encode_varint(seconds as u64, &mut timestamp);
            config.push((5 << 3) | 2);
            encode_varint(timestamp.len() as u64, &mut config);
            config.extend_from_slice(&timestamp);
        }
        let mut message = Vec::new();
        message.push((1 << 3) | 2);
        encode_varint(config.len() as u64, &mut message);
        message.extend_from_slice(&config);
        let mut frame = vec![0];
        frame.extend_from_slice(&(message.len() as u32).to_be_bytes());
        frame.extend_from_slice(&message);
        frame
    }

    fn encode_varint(mut value: u64, out: &mut Vec<u8>) {
        loop {
            let mut byte = (value & 0x7F) as u8;
            value >>= 7;
            if value != 0 {
                byte |= 0x80;
            }
            out.push(byte);
            if value == 0 {
                break;
            }
        }
    }
}
