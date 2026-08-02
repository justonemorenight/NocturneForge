use anyhow::Result;
use credentials_provider::CredentialsProvider;
use futures::future::Shared;
use gpui::{App, Context, Entity, SharedString, Task, Window};
use http_client::HttpClient;
use language_model::{
    AuthenticateError, FastModeConfirmation, IconOrSvg, InlineDescription, LanguageModel,
    LanguageModelProvider, LanguageModelProviderId, LanguageModelProviderName,
    LanguageModelProviderState, ProviderAccountSummary, ProviderSettingsView,
};
use openai_subscribed::{PROVIDER_ID, PROVIDER_NAME, State, create_language_model};
use release_channel::AppVersion;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use ui::{ConfiguredApiCard, prelude::*};

const SUBSCRIPTION_DESCRIPTION: &str =
    "Sign in with your ChatGPT Plus or Pro subscription to use OpenAI models in Zed's agent.";

pub struct OpenAiSubscribedProvider {
    state: Entity<State>,
}

impl OpenAiSubscribedProvider {
    pub fn new(
        http_client: Arc<dyn HttpClient>,
        credentials_provider: Arc<dyn CredentialsProvider>,
        cx: &mut App,
    ) -> Self {
        // The Codex models endpoint caps `client_version` at 32 characters.
        // Dev builds include the full commit SHA in semver build metadata, so
        // send the whole semantic version just like Codex CLI does.
        let version = AppVersion::global(cx);
        let client_version = format!("{}.{}.{}", version.major, version.minor, version.patch);
        let state = cx.new(|cx| State::new(http_client, credentials_provider, client_version, cx));
        Self { state }
    }
}

impl LanguageModelProviderState for OpenAiSubscribedProvider {
    type ObservableEntity = State;

    fn observable_entity(&self) -> Option<Entity<Self::ObservableEntity>> {
        Some(self.state.clone())
    }
}

impl LanguageModelProvider for OpenAiSubscribedProvider {
    fn id(&self) -> LanguageModelProviderId {
        PROVIDER_ID
    }

    fn name(&self) -> LanguageModelProviderName {
        PROVIDER_NAME
    }

    fn icon(&self) -> IconOrSvg {
        IconOrSvg::Icon(IconName::AiOpenAiGptSub)
    }

    fn default_model(&self, cx: &App) -> Option<Arc<dyn LanguageModel>> {
        let models = self.state.read(cx).models();
        let model = models
            .iter()
            .find(|model| model.id() == "gpt-5.5")
            .or_else(|| models.first())?
            .clone();
        Some(create_language_model(model, &self.state, cx))
    }

    fn default_fast_model(&self, cx: &App) -> Option<Arc<dyn LanguageModel>> {
        let models = self.state.read(cx).models();
        let model = models
            .iter()
            .find(|model| model.id() == "gpt-5.6-luna")
            .or_else(|| models.first())?
            .clone();
        Some(create_language_model(model, &self.state, cx))
    }

    fn provided_models(&self, cx: &App) -> Vec<Arc<dyn LanguageModel>> {
        self.state
            .read(cx)
            .models()
            .into_iter()
            .map(|model| create_language_model(model, &self.state, cx))
            .collect()
    }

    fn is_authenticated(&self, cx: &App) -> bool {
        self.state.read(cx).is_authenticated()
    }

    fn account_summaries(&self, cx: &App) -> Vec<ProviderAccountSummary> {
        self.state
            .read(cx)
            .account_summaries()
            .into_iter()
            .map(|account| {
                let label = account
                    .email
                    .clone()
                    .or(account.display_name.clone())
                    .unwrap_or_else(|| account.session_id.clone());
                let mut detail = account
                    .plan_type
                    .clone()
                    .or(account.workspace_name.clone())
                    .or(account.workspace_kind.clone());
                if let Some(expires_at_ms) = account.token_expires_at_ms {
                    let now_ms = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .map(|duration| duration.as_millis() as u64)
                        .unwrap_or_default();
                    let refresh_label = if expires_at_ms <= now_ms {
                        "refreshing token…".to_string()
                    } else {
                        format!("token refresh in {}m", (expires_at_ms - now_ms) / 60_000)
                    };
                    detail = Some(match detail {
                        Some(detail) => format!("{detail} · {refresh_label}").into(),
                        None => refresh_label.into(),
                    });
                }
                let quota = account.quota.as_ref().and_then(|quota| {
                    quota.primary.as_ref().map(|window| {
                        let stale = if account.quota_stale { " · stale" } else { "" };
                        format!("{:.0}% used{}", window.used_percent, stale).into()
                    })
                });
                ProviderAccountSummary {
                    id: account.session_id,
                    label,
                    detail,
                    quota,
                    is_active: account.is_active,
                    is_busy: account.is_busy,
                    reauthentication_required: account.reauthentication_required,
                }
            })
            .collect()
    }

    fn switch_account(&self, account_id: SharedString, cx: &mut App) -> Task<Result<()>> {
        self.state
            .update(cx, |state, cx| state.switch_account(account_id, cx))
    }

    fn add_account(&self, cx: &mut App) -> Task<Result<()>> {
        self.state.update(cx, |state, cx| state.sign_in(cx))
    }

    fn can_cancel_account_sign_in(&self, cx: &App) -> bool {
        self.state.read(cx).can_cancel_sign_in()
    }

    fn cancel_account_sign_in(&self, cx: &mut App) {
        self.state.update(cx, |state, cx| state.cancel_sign_in(cx));
    }

    fn authenticate(&self, cx: &mut App) -> Task<Result<(), AuthenticateError>> {
        if self.is_authenticated(cx) {
            return Task::ready(Ok(()));
        }
        let load_task: Option<Shared<_>> = self.state.read(cx).load_task();
        if let Some(load_task) = load_task {
            let weak_state = self.state.downgrade();
            cx.spawn(async move |cx| {
                let _ = load_task.await;
                let is_auth = weak_state
                    .read_with(&*cx, |state, _| state.is_authenticated())
                    .unwrap_or(false);
                if is_auth {
                    Ok(())
                } else {
                    Err(AuthenticateError::CredentialsNotFound)
                }
            })
        } else {
            Task::ready(Err(AuthenticateError::CredentialsNotFound))
        }
    }

    fn settings_view(&self, cx: &mut App) -> Option<ProviderSettingsView> {
        let is_authenticated = self.state.read(cx).is_authenticated();
        let title = if is_authenticated {
            None
        } else {
            Some("Configure ChatGPT".into())
        };
        let description = if is_authenticated {
            None
        } else {
            Some(InlineDescription::Text(SUBSCRIPTION_DESCRIPTION.into()))
        };

        Some(ProviderSettingsView::Inline(
            language_model::InlineProviderSettings {
                title,
                description,
                create_view: Arc::new({
                    let state = self.state.clone();
                    move |_window, cx| {
                        cx.new(|_cx| ConfigurationView {
                            state: state.clone(),
                            compact: true,
                        })
                        .into()
                    }
                }),
            },
        ))
    }

    fn authentication_error_message(&self) -> SharedString {
        "Your ChatGPT subscription session is invalid or has expired. \
        Sign in again via Settings > AI > LLM Providers to continue."
            .into()
    }

    fn missing_credentials_error_message(&self) -> SharedString {
        "You are not signed in to your ChatGPT account. \
        Sign in via Settings > AI > LLM Providers to continue."
            .into()
    }

    fn fast_mode_confirmation(&self, _cx: &App) -> Option<FastModeConfirmation> {
        Some(FastModeConfirmation {
            title: "Enable Fast Mode for OpenAI?".into(),
            message: "Fast mode sends requests using OpenAI's Priority processing tier, which \
                targets significantly lower latency than the standard tier and is billed at a \
                premium per-token rate."
                .into(),
        })
    }
}

struct ConfigurationView {
    state: Entity<State>,
    /// When `true`, the description is rendered elsewhere (the settings row's
    /// left column), so it's omitted here to avoid duplication.
    compact: bool,
}

impl Render for ConfigurationView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let state = self.state.read(cx);
        let accounts = state.account_summaries();
        let is_busy = state.is_busy();
        let can_cancel_sign_in = state.can_cancel_sign_in();

        // Account management stays available even when the active credential
        // is invalid: a fatal refresh marks the account for re-authentication
        // instead of hiding the other sessions.
        if !accounts.is_empty() {
            let last_auth_error = state.last_auth_error();
            let state_entity = self.state.clone();
            let cards = accounts.into_iter().map(|account| {
                let mut label = account
                    .email
                    .clone()
                    .unwrap_or_else(|| "ChatGPT account".into());
                if let Some(plan) = account.plan_type.as_ref() {
                    label = format!("{label} · {plan}").into();
                }
                if let Some(quota) = account
                    .quota
                    .as_ref()
                    .and_then(|quota| quota.primary.as_ref())
                {
                    label = format!("{label} · {:.0}% used", quota.used_percent).into();
                }
                if account.reauthentication_required {
                    label = format!("{label} · sign in required").into();
                }
                let account_id = account.session_id.clone();
                let action_label = if account.is_active {
                    "Sign Out"
                } else {
                    "Switch"
                };
                let is_active = account.is_active;
                let state_entity = state_entity.clone();
                ConfiguredApiCard::new(
                    format!("openai-subscribed-account-{}", account.session_id),
                    label,
                )
                .button_label(action_label)
                .disabled(is_busy)
                .on_click(move |_, _window, cx| {
                    if is_active {
                        state_entity
                            .update(cx, |state, cx| state.sign_out(cx))
                            .detach_and_log_err(cx);
                    } else {
                        state_entity
                            .update(cx, |state, cx| state.switch_account(account_id.clone(), cx))
                            .detach_and_log_err(cx);
                    }
                })
            });

            let state_entity = self.state.clone();
            let sign_in_state = state_entity.clone();
            let add_account_label = if can_cancel_sign_in {
                "Cancel Sign In"
            } else {
                "Add Account"
            };
            return v_flex()
                .gap_2()
                .children(cards)
                .child(
                    h_flex()
                        .gap_2()
                        .when(!state.is_authenticated(), |this| {
                            let sign_in_state = sign_in_state.clone();
                            this.child(
                                Button::new("openai-subscribed-sign-in", "Sign In")
                                    .style(ButtonStyle::Outlined)
                                    .size(ButtonSize::Medium)
                                    .disabled(is_busy)
                                    .on_click(cx.listener(move |_this, _, _window, cx| {
                                        sign_in_state
                                            .update(cx, |state, cx| state.sign_in(cx))
                                            .detach_and_log_err(cx);
                                    })),
                            )
                        })
                        .child(
                            Button::new("openai-subscribed-add-account", add_account_label)
                                .style(ButtonStyle::Outlined)
                                .size(ButtonSize::Medium)
                                .disabled(is_busy && !can_cancel_sign_in)
                                .on_click(cx.listener(move |_this, _, _window, cx| {
                                    if can_cancel_sign_in {
                                        state_entity.update(cx, |state, cx| {
                                            state.cancel_sign_in(cx);
                                        });
                                    } else {
                                        state_entity
                                            .update(cx, |state, cx| state.sign_in(cx))
                                            .detach_and_log_err(cx);
                                    }
                                })),
                        ),
                )
                .when_some(last_auth_error, |this, error| {
                    this.child(
                        h_flex()
                            .gap_1()
                            .child(
                                Icon::new(IconName::XCircle)
                                    .color(Color::Error)
                                    .size(IconSize::Small),
                            )
                            .child(Label::new(error).color(Color::Muted)),
                    )
                })
                .into_any_element();
        }

        let last_auth_error = state.last_auth_error();
        let provider_state = self.state.clone();

        let is_signing_in = state.is_signing_in();
        let button_label = if can_cancel_sign_in {
            "Cancel Sign In"
        } else if is_signing_in {
            "Signing in…"
        } else {
            "Sign In"
        };

        v_flex()
            .gap_2()
            .when(!self.compact, |this| {
                this.child(Label::new(SUBSCRIPTION_DESCRIPTION))
            })
            .child(
                Button::new("sign-in", button_label)
                    .when(!self.compact, |this| this.full_width())
                    .style(ButtonStyle::Outlined)
                    .size(ButtonSize::Medium)
                    .loading(is_signing_in && !can_cancel_sign_in)
                    .disabled((is_signing_in || is_busy) && !can_cancel_sign_in)
                    .on_click(move |_, _window, cx| {
                        if can_cancel_sign_in {
                            provider_state.update(cx, |state, cx| state.cancel_sign_in(cx));
                        } else {
                            provider_state
                                .update(cx, |state, cx| state.sign_in(cx))
                                .detach_and_log_err(cx);
                        }
                    }),
            )
            .when_some(last_auth_error, |this, error| {
                this.child(
                    h_flex()
                        .gap_1()
                        .justify_center()
                        .child(
                            Icon::new(IconName::XCircle)
                                .color(Color::Error)
                                .size(IconSize::Small),
                        )
                        .child(Label::new(error).color(Color::Muted)),
                )
            })
            .into_any_element()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::{AsyncApp, TestAppContext};
    use http_client::FakeHttpClient;
    use parking_lot::Mutex;
    use std::collections::HashMap;
    use std::future::Future;
    use std::pin::Pin;

    #[gpui::test]
    async fn test_authenticate_awaits_initial_load(cx: &mut TestAppContext) {
        let creds_json = serde_json::json!({
            "access_token": "fresh_access",
            "refresh_token": "fresh_refresh",
            "expires_at_ms": u64::MAX,
            "account_id": null,
            "email": null,
        });
        let creds_provider = Arc::new(FakeCredentialsProvider::new());
        // The legacy single-account key triggers the migration path during the
        // initial load; must match `LEGACY_CREDENTIALS_KEY` in openai_subscribed.
        creds_provider.insert(
            "https://chatgpt.com/backend-api/codex",
            "Bearer",
            serde_json::to_vec(&creds_json).unwrap(),
        );

        let http_client = FakeHttpClient::create(|_| async {
            Ok(http_client::Response::builder()
                .status(200)
                .body(http_client::AsyncBody::default())?)
        });

        let provider =
            cx.update(|cx| OpenAiSubscribedProvider::new(http_client, creds_provider, cx));

        // Before load completes, authenticate should still await the load.
        let auth_task = cx.update(|cx| provider.authenticate(cx));

        // Drive the load to completion.
        cx.run_until_parked();

        let result = auth_task.await;
        assert!(
            result.is_ok(),
            "authenticate should succeed after load completes with valid credentials"
        );
    }

    struct FakeCredentialsProvider {
        storage: Mutex<HashMap<String, (String, Vec<u8>)>>,
    }

    impl FakeCredentialsProvider {
        fn new() -> Self {
            Self {
                storage: Mutex::new(HashMap::new()),
            }
        }

        fn insert(&self, key: &str, username: &str, password: Vec<u8>) {
            self.storage
                .lock()
                .insert(key.to_string(), (username.to_string(), password));
        }
    }

    impl CredentialsProvider for FakeCredentialsProvider {
        fn read_credentials<'a>(
            &'a self,
            url: &'a str,
            _cx: &'a AsyncApp,
        ) -> Pin<Box<dyn Future<Output = Result<Option<(String, Vec<u8>)>>> + 'a>> {
            Box::pin(async move { Ok(self.storage.lock().get(url).cloned()) })
        }

        fn write_credentials<'a>(
            &'a self,
            url: &'a str,
            username: &'a str,
            password: &'a [u8],
            _cx: &'a AsyncApp,
        ) -> Pin<Box<dyn Future<Output = Result<()>> + 'a>> {
            let url = url.to_string();
            let username = username.to_string();
            let password = password.to_vec();
            Box::pin(async move {
                self.storage.lock().insert(url, (username, password));
                Ok(())
            })
        }

        fn delete_credentials<'a>(
            &'a self,
            url: &'a str,
            _cx: &'a AsyncApp,
        ) -> Pin<Box<dyn Future<Output = Result<()>> + 'a>> {
            let url = url.to_string();
            Box::pin(async move {
                self.storage.lock().remove(&url);
                Ok(())
            })
        }
    }
}
