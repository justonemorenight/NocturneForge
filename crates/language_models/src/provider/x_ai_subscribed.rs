use anyhow::{Result, anyhow};
use credentials_provider::CredentialsProvider;
use futures::future::Shared;
use gpui::{App, Context, Entity, SharedString, Task, Window};
use http_client::HttpClient;
use language_model::{
    AuthenticateError, IconOrSvg, InlineDescription, LanguageModel, LanguageModelProvider,
    LanguageModelProviderId, LanguageModelProviderName, LanguageModelProviderState,
    ProviderAccountSummary, ProviderSettingsView, QuotaPoolAccountSummary, QuotaPoolSummary,
};
use std::sync::Arc;
use ui::{ConfiguredApiCard, prelude::*};
use x_ai_subscribed::{PROVIDER_ID, PROVIDER_NAME, State, SuperGrokModel, create_language_model};

const SUBSCRIPTION_DESCRIPTION: &str =
    "Sign in with your SuperGrok subscription to use Grok models in Zed's agent.";

fn remaining_percent(used_percent: f64) -> f64 {
    100.0 - used_percent.clamp(0.0, 100.0)
}

pub struct XAiSubscribedProvider {
    state: Entity<State>,
}

impl XAiSubscribedProvider {
    pub fn new(
        http_client: Arc<dyn HttpClient>,
        credentials_provider: Arc<dyn CredentialsProvider>,
        cx: &mut App,
    ) -> Self {
        let state = cx.new(|cx| State::new(http_client, credentials_provider, cx));
        Self { state }
    }

    fn create_model(&self, model: SuperGrokModel, cx: &App) -> Arc<dyn LanguageModel> {
        create_language_model(model, &self.state, cx)
    }
}

impl LanguageModelProviderState for XAiSubscribedProvider {
    type ObservableEntity = State;

    fn observable_entity(&self) -> Option<Entity<Self::ObservableEntity>> {
        Some(self.state.clone())
    }
}

impl LanguageModelProvider for XAiSubscribedProvider {
    fn id(&self) -> LanguageModelProviderId {
        PROVIDER_ID
    }

    fn name(&self) -> LanguageModelProviderName {
        PROVIDER_NAME
    }

    fn icon(&self) -> IconOrSvg {
        IconOrSvg::Icon(IconName::AiXAi)
    }

    fn default_model(&self, cx: &App) -> Option<Arc<dyn LanguageModel>> {
        Some(self.create_model(SuperGrokModel::Grok46, cx))
    }

    fn default_fast_model(&self, cx: &App) -> Option<Arc<dyn LanguageModel>> {
        Some(self.create_model(SuperGrokModel::GrokBuild01, cx))
    }

    fn provided_models(&self, cx: &App) -> Vec<Arc<dyn LanguageModel>> {
        if !self.is_authenticated(cx) {
            return Vec::new();
        }
        SuperGrokModel::all()
            .into_iter()
            .map(|model| self.create_model(model, cx))
            .collect()
    }

    fn recommended_models(&self, cx: &App) -> Vec<Arc<dyn LanguageModel>> {
        self.default_model(cx).into_iter().collect()
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
                let quota = account.quota.as_ref().map(|quota| {
                    let stale = if account.quota_stale { " · stale" } else { "" };
                    format!(
                        "{:.0}% remaining{stale}",
                        remaining_percent(quota.used_percent),
                    )
                    .into()
                });
                ProviderAccountSummary {
                    id: account.session_id,
                    label: account.email.unwrap_or_else(|| "SuperGrok account".into()),
                    detail: account.plan_type,
                    quota,
                    is_active: account.is_active,
                    is_busy: account.is_busy,
                    reauthentication_required: account.reauthentication_required,
                }
            })
            .collect()
    }

    fn quota_pool(&self, model_id: Option<&str>, cx: &App) -> Option<QuotaPoolSummary> {
        let state = self.state.read(cx);
        let summaries = state.account_summaries();
        if summaries.is_empty() {
            return None;
        }

        let now = x_ai_subscribed::now_ms();
        let target_model = model_id.unwrap_or("grok-4.6");
        let eligible_session_ids: std::collections::HashSet<_> = state
            .eligible_accounts(target_model)
            .into_iter()
            .map(|s| s.session_id)
            .collect();
        let eligible_count = eligible_session_ids.len();

        let mut total_known_remaining = 0.0;
        let mut has_unknown_eligible = false;
        let mut pool_has_stale = false;
        let mut accounts = Vec::new();

        for account in &summaries {
            let is_eligible = eligible_session_ids.contains(account.session_id.as_ref());
            let is_reauth = account.reauthentication_required;
            let is_stale = account.quota_stale;
            let is_rate_limited = account.exclusions.iter().any(|ex| match ex {
                x_ai_subscribed::AccountExclusion::RateLimited { retry_at_ms, scope } => {
                    let applies = match scope {
                        x_ai_subscribed::AccountExclusionScope::Account => true,
                        x_ai_subscribed::AccountExclusionScope::Model(m) => m == target_model,
                    };
                    applies && *retry_at_ms > now
                }
                _ => false,
            });
            let is_forbidden = account.exclusions.iter().any(|ex| match ex {
                x_ai_subscribed::AccountExclusion::Forbidden { scope } => match scope {
                    x_ai_subscribed::AccountExclusionScope::Account => true,
                    x_ai_subscribed::AccountExclusionScope::Model(m) => m == target_model,
                },
                _ => false,
            });

            let rem = if !is_eligible {
                Some(0.0)
            } else if let Some(quota) = &account.quota {
                if is_stale {
                    pool_has_stale = true;
                }
                Some(remaining_percent(quota.used_percent))
            } else {
                has_unknown_eligible = true;
                None
            };

            if let Some(r) = rem {
                total_known_remaining += r;
            }

            accounts.push(QuotaPoolAccountSummary {
                id: account.session_id.clone(),
                label: account
                    .email
                    .clone()
                    .unwrap_or_else(|| "SuperGrok account".into()),
                remaining_percent: rem,
                is_stale,
                is_active: account.is_active,
                is_eligible,
                is_rate_limited,
                is_reauth_required: is_reauth,
                is_forbidden,
            });
        }

        let avg_remaining = if eligible_count == 0 {
            Some(0.0)
        } else if has_unknown_eligible {
            None
        } else {
            Some((total_known_remaining / summaries.len() as f64).clamp(0.0, 100.0))
        };

        Some(QuotaPoolSummary {
            remaining_percent: avg_remaining,
            total_accounts: summaries.len(),
            eligible_accounts: eligible_count,
            is_stale: pool_has_stale,
            accounts,
        })
    }

    fn switch_account(&self, account_id: SharedString, cx: &mut App) -> Task<Result<()>> {
        self.state
            .update(cx, |state, cx| state.switch_account(account_id, cx))
    }

    fn add_account(&self, cx: &mut App) -> Task<Result<()>> {
        self.state.update(cx, |state, cx| {
            state.sign_in(cx);
            Task::ready(Ok(()))
        })
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
                load_task
                    .await
                    .map_err(|error| AuthenticateError::Other(anyhow!("{error:#}")))?;
                let is_auth = weak_state
                    .read_with(&*cx, |state, _| state.is_authenticated())
                    .map_err(AuthenticateError::Other)?;
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
            Some("Configure SuperGrok".into())
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
        "Your SuperGrok session is invalid or has expired. \
        Sign in again via Settings > AI > LLM Providers to continue."
            .into()
    }

    fn missing_credentials_error_message(&self) -> SharedString {
        "You are not signed in to SuperGrok. \
        Sign in via Settings > AI > LLM Providers to continue."
            .into()
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
        let is_signing_in = state.is_signing_in();
        let can_cancel_sign_in = state.can_cancel_sign_in();

        if !accounts.is_empty() {
            let state_entity = self.state.clone();
            let add_account_label = if can_cancel_sign_in {
                "Cancel Sign In"
            } else {
                "Add Account"
            };
            return v_flex()
                .gap_2()
                .children(accounts.into_iter().map(|account| {
                    let state_entity = state_entity.clone();
                    let account_id = account.session_id.clone();
                    let mut label = account.email.unwrap_or_else(|| "SuperGrok account".into());
                    if let Some(plan) = account.plan_type.as_ref() {
                        label = format!("{label} · {plan}").into();
                    }
                    if let Some(quota) = account.quota.as_ref() {
                        label = format!(
                            "{label} · {:.0}% remaining",
                            remaining_percent(quota.used_percent)
                        )
                        .into();
                    }
                    if account.reauthentication_required {
                        label = format!("{label} · sign in required").into();
                    }
                    let (button_label, is_active) = if account.is_active {
                        ("Sign Out", true)
                    } else {
                        ("Switch", false)
                    };
                    ConfiguredApiCard::new(
                        format!("x-ai-subscribed-account-{}", account.session_id),
                        label,
                    )
                    .button_label(button_label)
                    .disabled(is_busy)
                    .on_click(move |_, _window, cx| {
                        if is_active {
                            state_entity
                                .update(cx, |state, cx| state.sign_out(cx))
                                .detach_and_log_err(cx);
                        } else {
                            state_entity
                                .update(cx, |state, cx| {
                                    state.switch_account(account_id.clone(), cx)
                                })
                                .detach_and_log_err(cx);
                        }
                    })
                }))
                .child(
                    h_flex()
                        .gap_2()
                        .when(!state.is_authenticated(), |this| {
                            let sign_in_state = state_entity.clone();
                            this.child(
                                Button::new("x-ai-subscribed-sign-in", "Sign In")
                                    .style(ButtonStyle::Outlined)
                                    .size(ButtonSize::Medium)
                                    .disabled(is_busy)
                                    .on_click(move |_, _window, cx| {
                                        sign_in_state.update(cx, |state, cx| state.sign_in(cx));
                                    }),
                            )
                        })
                        .child(
                            Button::new("x-ai-subscribed-add-account", add_account_label)
                                .style(ButtonStyle::Outlined)
                                .size(ButtonSize::Medium)
                                .disabled(is_busy && !can_cancel_sign_in)
                                .on_click(cx.listener(move |_this, _, _window, cx| {
                                    if can_cancel_sign_in {
                                        state_entity.update(cx, |state, cx| {
                                            state.cancel_sign_in(cx);
                                        });
                                    } else {
                                        state_entity.update(cx, |state, cx| {
                                            state.sign_in(cx);
                                        });
                                    }
                                })),
                        ),
                )
                .into_any_element();
        }

        let last_auth_error = state.last_auth_error();
        let provider_state = self.state.clone();
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
                            provider_state.update(cx, |state, cx| state.sign_in(cx));
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
    use gpui::AsyncApp;
    use http_client::FakeHttpClient;
    use parking_lot::Mutex;
    use std::future::Future;
    use std::pin::Pin;

    #[gpui::test]
    async fn test_authenticate_awaits_initial_load(cx: &mut gpui::TestAppContext) {
        let creds_json = serde_json::json!({
            "access_token": "fresh_access",
            "refresh_token": "fresh_refresh",
            "expires_at_ms": u64::MAX,
            "email": null,
        });
        let creds_provider = Arc::new(FakeCredentialsProvider::new());
        creds_provider.storage.lock().replace((
            "Bearer".to_string(),
            serde_json::to_vec(&creds_json).unwrap(),
        ));

        let http_client = FakeHttpClient::create(|_| async {
            Ok(http_client::Response::builder()
                .status(200)
                .body(http_client::AsyncBody::default())?)
        });

        let provider = cx.update(|cx| XAiSubscribedProvider::new(http_client, creds_provider, cx));
        let auth_task = cx.update(|cx| provider.authenticate(cx));
        cx.run_until_parked();
        auth_task
            .await
            .expect("authenticate should succeed after load completes with valid credentials");
        cx.update(|cx| {
            assert!(provider.is_authenticated(cx));
        });
    }

    struct FakeCredentialsProvider {
        storage: Mutex<Option<(String, Vec<u8>)>>,
    }

    impl FakeCredentialsProvider {
        fn new() -> Self {
            Self {
                storage: Mutex::new(None),
            }
        }
    }

    impl CredentialsProvider for FakeCredentialsProvider {
        fn read_credentials<'a>(
            &'a self,
            _url: &'a str,
            _cx: &'a AsyncApp,
        ) -> Pin<Box<dyn Future<Output = Result<Option<(String, Vec<u8>)>>> + 'a>> {
            Box::pin(async { Ok(self.storage.lock().clone()) })
        }

        fn write_credentials<'a>(
            &'a self,
            _url: &'a str,
            username: &'a str,
            password: &'a [u8],
            _cx: &'a AsyncApp,
        ) -> Pin<Box<dyn Future<Output = Result<()>> + 'a>> {
            self.storage
                .lock()
                .replace((username.to_string(), password.to_vec()));
            Box::pin(async { Ok(()) })
        }

        fn delete_credentials<'a>(
            &'a self,
            _url: &'a str,
            _cx: &'a AsyncApp,
        ) -> Pin<Box<dyn Future<Output = Result<()>> + 'a>> {
            *self.storage.lock() = None;
            Box::pin(async { Ok(()) })
        }
    }
}
