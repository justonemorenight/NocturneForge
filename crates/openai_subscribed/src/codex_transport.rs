use std::{
    collections::HashMap,
    sync::{Arc, OnceLock, Weak},
};

use futures::{FutureExt as _, future::BoxFuture};
use http_client::{
    AsyncBody, HttpClient, RedirectPolicy, Request, Response, Url,
    http::{HeaderValue, header},
};
use parking_lot::Mutex;
use reqwest::cookie::{CookieStore as _, Jar};

const MAX_TURN_STATE_BYTES: usize = 4 * 1024;

#[derive(Default)]
pub(super) struct CodexTransportState {
    turns: Mutex<HashMap<TurnScope, TurnStateEntry>>,
    cookies: Mutex<Option<OwnerCookies>>,
}

#[derive(PartialEq, Eq, Hash)]
struct TurnScope {
    session_id: String,
    auth_generation: u64,
    model_id: String,
    prompt_id: String,
}

struct TurnStateEntry {
    prompt_id: Weak<str>,
    value: Arc<OnceLock<Arc<str>>>,
}

struct OwnerCookies {
    session_id: String,
    auth_generation: u64,
    jar: Arc<Jar>,
}

#[derive(Clone)]
pub(super) struct TurnState {
    // Keep the registry entry alive until the request stream has been dropped.
    _prompt_id: Option<Arc<str>>,
    value: Arc<OnceLock<Arc<str>>>,
}

impl TurnState {
    pub(super) fn value(&self) -> Option<&str> {
        self.value.get().map(AsRef::as_ref)
    }

    pub(super) fn capture(&self, value: &str) -> bool {
        if value.is_empty() || value.len() > MAX_TURN_STATE_BYTES {
            return false;
        }
        self.value.set(Arc::from(value)).is_ok()
    }
}

impl CodexTransportState {
    pub(super) fn turn(
        &self,
        session_id: Option<&str>,
        auth_generation: u64,
        model_id: &str,
        prompt_id: Option<Arc<str>>,
    ) -> TurnState {
        let Some(session_id) = session_id else {
            return TurnState {
                _prompt_id: prompt_id,
                value: Arc::default(),
            };
        };
        let Some(prompt_id) = prompt_id else {
            return TurnState {
                _prompt_id: None,
                value: Arc::default(),
            };
        };

        let scope = TurnScope {
            session_id: session_id.to_owned(),
            auth_generation,
            model_id: model_id.to_owned(),
            prompt_id: prompt_id.to_string(),
        };
        let owner = Arc::downgrade(&prompt_id);
        let mut turns = self.turns.lock();
        turns.retain(|_, entry| entry.prompt_id.strong_count() > 0);
        let entry = turns.entry(scope).or_insert_with(|| TurnStateEntry {
            prompt_id: owner.clone(),
            value: Arc::default(),
        });
        // A restored request can reuse a serialized ID, but it does not own the
        // live turn that originally minted the token.
        let value = if entry.prompt_id.ptr_eq(&owner) {
            entry.value.clone()
        } else {
            Arc::default()
        };
        TurnState {
            _prompt_id: Some(prompt_id),
            value,
        }
    }

    pub(super) fn client(
        &self,
        inner: Arc<dyn HttpClient>,
        session_id: Option<&str>,
        auth_generation: u64,
    ) -> Arc<dyn HttpClient> {
        let jar = if let Some(session_id) = session_id {
            let mut cookies = self.cookies.lock();
            let owner = cookies.get_or_insert_with(|| OwnerCookies {
                session_id: session_id.to_owned(),
                auth_generation,
                jar: Arc::default(),
            });
            if owner.session_id != session_id || owner.auth_generation != auth_generation {
                *owner = OwnerCookies {
                    session_id: session_id.to_owned(),
                    auth_generation,
                    jar: Arc::default(),
                };
            }
            owner.jar.clone()
        } else {
            Arc::default()
        };
        Arc::new(ChatGptHttpClient { inner, jar })
    }
}

struct ChatGptHttpClient {
    inner: Arc<dyn HttpClient>,
    jar: Arc<Jar>,
}

impl HttpClient for ChatGptHttpClient {
    fn user_agent(&self) -> Option<&HeaderValue> {
        self.inner.user_agent()
    }

    fn proxy(&self) -> Option<&Url> {
        self.inner.proxy()
    }

    fn send(
        &self,
        mut request: Request<AsyncBody>,
    ) -> BoxFuture<'static, anyhow::Result<Response<AsyncBody>>> {
        let url = Url::parse(&request.uri().to_string())
            .ok()
            .filter(|url| url.scheme() == "https" && url.host_str() == Some("chatgpt.com"));
        if let Some(url) = &url {
            // The HTTP abstraction doesn't expose the final URL after a
            // redirect, so accepting one would lose the cookie's true origin.
            request.extensions_mut().insert(RedirectPolicy::NoFollow);
            if !request.headers().contains_key(header::COOKIE)
                && let Some(mut cookie) = self.jar.cookies(url)
            {
                cookie.set_sensitive(true);
                request.headers_mut().insert(header::COOKIE, cookie);
            }
        }
        let response = self.inner.send(request);
        let jar = self.jar.clone();
        async move {
            let response = response.await?;
            if let Some(url) = url {
                let mut cookies = response
                    .headers()
                    .get_all(header::SET_COOKIE)
                    .iter()
                    .filter(|value| {
                        value
                            .to_str()
                            .ok()
                            .and_then(|value| value.split_once('='))
                            .is_some_and(|(name, _)| is_infrastructure_cookie(name.trim()))
                    });
                jar.set_cookies(&mut cookies, &url);
            }
            Ok(response)
        }
        .boxed()
    }
}

fn is_infrastructure_cookie(name: &str) -> bool {
    matches!(
        name,
        "__cf_bm"
            | "__cflb"
            | "__cfruid"
            | "__cfseq"
            | "__cfwaitingroom"
            | "__oailb"
            | "_cfuvid"
            | "cf_clearance"
            | "cf_ob_info"
            | "cf_use_ob"
    ) || name.starts_with("cf_chl_")
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Context as _;
    use futures::executor::block_on;
    use http_client::{FakeHttpClient, http};

    #[test]
    fn invalid_state_does_not_consume_first_write() {
        let transport = CodexTransportState::default();
        let turn = transport.turn(Some("account"), 1, "model", Some(Arc::from("turn")));
        assert!(!turn.capture(""));
        assert!(!turn.capture(&"s".repeat(MAX_TURN_STATE_BYTES + 1)));
        assert!(turn.capture("opaque-state"));
        assert_eq!(turn.value(), Some("opaque-state"));
    }

    #[test]
    fn first_state_survives_continuations_and_length_changes() {
        for length in [292, 312, 332, 356, 780, 868] {
            let transport = CodexTransportState::default();
            let prompt: Arc<str> = "turn".into();
            let turn = transport.turn(Some("account"), 1, "model", Some(prompt.clone()));
            let state = "s".repeat(length);
            assert!(turn.capture(&state));
            let continuation = transport.turn(Some("account"), 1, "model", Some(prompt.clone()));
            assert!(!continuation.capture("different-state"));
            assert_eq!(continuation.value(), Some(state.as_str()));
            assert_eq!(turn.value(), continuation.value());
        }
    }

    #[test]
    fn concurrent_responses_share_first_write_without_overwriting_other_turns() {
        let transport = CodexTransportState::default();
        let prompt: Arc<str> = "turn".into();
        let first = transport.turn(Some("account"), 1, "model", Some(prompt.clone()));
        let second = transport.turn(Some("account"), 1, "model", Some(prompt));
        std::thread::scope(|scope| {
            scope.spawn(|| first.capture("first-response"));
            scope.spawn(|| second.capture("second-response"));
        });
        assert_eq!(first.value(), second.value());
        assert!(matches!(
            first.value(),
            Some("first-response" | "second-response")
        ));
        let next = transport.turn(Some("account"), 1, "model", Some(Arc::from("new-turn")));
        assert!(next.capture("new-state"));
        assert!(!first.capture("late-state"));
        assert_eq!(next.value(), Some("new-state"));
    }

    #[test]
    fn scope_components_cannot_collide_via_delimiters() {
        let transport = CodexTransportState::default();
        let first_prompt: Arc<str> = "turn".into();
        let second_prompt: Arc<str> = "nested:turn".into();
        let first = transport.turn(
            Some("account"),
            1,
            "model:nested",
            Some(first_prompt.clone()),
        );
        let second = transport.turn(Some("account"), 1, "model", Some(second_prompt.clone()));
        assert!(first.capture("first"));
        assert!(second.capture("second"));
        assert_eq!(
            transport
                .turn(Some("account"), 1, "model:nested", Some(first_prompt))
                .value(),
            Some("first")
        );
        assert_eq!(
            transport
                .turn(Some("account"), 1, "model", Some(second_prompt))
                .value(),
            Some("second")
        );
    }

    #[test]
    fn state_isolated_by_owner_model_and_live_turn() {
        let transport = CodexTransportState::default();
        let prompt: Arc<str> = "turn".into();
        let turn = transport.turn(Some("account"), 1, "model", Some(prompt.clone()));
        assert!(turn.capture("state"));
        for (account, generation, model, owner) in [
            ("other-account", 1, "model", prompt.clone()),
            ("account", 2, "model", prompt.clone()),
            ("account", 1, "other-model", prompt.clone()),
            ("account", 1, "model", Arc::from("new-turn")),
            ("account", 1, "model", Arc::from("turn")),
        ] {
            assert!(
                transport
                    .turn(Some(account), generation, model, Some(owner))
                    .value()
                    .is_none()
            );
        }
        assert!(
            transport
                .turn(None, 1, "model", Some(prompt))
                .value()
                .is_none()
        );
        assert!(
            transport
                .turn(Some("account"), 1, "model", None)
                .value()
                .is_none()
        );
    }

    #[test]
    fn retired_turns_are_collected_without_expiring_live_turns() {
        let transport = CodexTransportState::default();
        let prompt: Arc<str> = "turn".into();
        let turn = transport.turn(Some("account"), 1, "model", Some(prompt.clone()));
        assert!(turn.capture("state"));
        drop(turn);
        let continuation = transport.turn(Some("account"), 1, "model", Some(prompt.clone()));
        assert_eq!(continuation.value(), Some("state"));
        drop(prompt);
        drop(continuation);
        let new_turn = transport.turn(Some("account"), 1, "model", Some(Arc::from("new-turn")));
        assert!(new_turn.value().is_none());
        assert_eq!(transport.turns.lock().len(), 1);
    }

    #[test]
    fn routing_cookies_follow_scope_expiry_and_explicit_precedence() -> anyhow::Result<()> {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let captured_requests = requests.clone();
        let inner = FakeHttpClient::create(move |request| {
            if request.uri().scheme_str() == Some("https")
                && request.uri().host() == Some("chatgpt.com")
            {
                assert_eq!(
                    request.extensions().get::<RedirectPolicy>(),
                    Some(&RedirectPolicy::NoFollow)
                );
            }
            let mut requests = captured_requests.lock();
            let index = requests.len();
            requests.push((request.uri().to_string(), request.headers().clone()));
            async move {
                let mut response =
                    http::Response::builder().status(if index == 0 { 401 } else { 200 });
                if index == 0 {
                    response = response
                        .header(
                            header::SET_COOKIE,
                            "__oailb=route; Path=/backend-api; Secure",
                        )
                        .header(
                            header::SET_COOKIE,
                            "__cflb=balance; Path=/backend-api; Secure",
                        )
                        .header(header::SET_COOKIE, "chatgpt_session=secret; Path=/; Secure")
                        .header(
                            header::SET_COOKIE,
                            "__cf_bm=wrong-domain; Domain=evil.com; Path=/; Secure",
                        );
                }
                if index == 6 {
                    response = response.header(
                        header::SET_COOKIE,
                        "__oailb=; Max-Age=0; Path=/backend-api; Secure",
                    );
                }
                Ok(response.body(AsyncBody::default())?)
            }
        });
        let transport = CodexTransportState::default();
        let client = transport.client(inner, Some("account"), 1);
        for url in [
            "https://chatgpt.com/backend-api/codex/responses",
            "https://chatgpt.com/backend-api/codex/responses",
            "https://chatgpt.com/",
            "https://other.chatgpt.com/backend-api/codex/responses",
            "https://api.openai.com/backend-api/codex/responses",
            "http://chatgpt.com/backend-api/codex/responses",
            "https://chatgpt.com/backend-api/codex/responses",
            "https://chatgpt.com/backend-api/codex/responses",
        ] {
            block_on(client.send(Request::builder().uri(url).body(AsyncBody::default())?))?;
        }
        block_on(
            client.send(
                Request::builder()
                    .uri("https://chatgpt.com/backend-api/codex/responses")
                    .header(header::COOKIE, "explicit=value")
                    .body(AsyncBody::default())?,
            ),
        )?;
        let requests = requests.lock();
        let cookies = |index: usize| {
            requests
                .get(index)
                .and_then(|(_, headers)| headers.get(header::COOKIE))
                .and_then(|value| value.to_str().ok())
        };
        let replayed = cookies(1).context("missing routing cookies")?;
        assert!(
            requests
                .get(1)
                .and_then(|(_, headers)| headers.get(header::COOKIE))
                .context("missing cookie header")?
                .is_sensitive()
        );
        assert!(replayed.contains("__oailb=route"));
        assert!(replayed.contains("__cflb=balance"));
        assert!(!replayed.contains("secret"));
        assert!(!replayed.contains("wrong-domain"));
        for index in 2..=5 {
            assert_eq!(cookies(index), None);
        }
        assert_eq!(cookies(7), Some("__cflb=balance"));
        assert_eq!(cookies(8), Some("explicit=value"));
        Ok(())
    }

    #[test]
    fn cookie_owner_changes_do_not_affect_in_flight_clients() -> anyhow::Result<()> {
        let headers = Arc::new(Mutex::new(Vec::new()));
        let captured_headers = headers.clone();
        let inner = FakeHttpClient::create(move |request| {
            let mut headers = captured_headers.lock();
            let index = headers.len();
            headers.push(request.headers().clone());
            async move {
                let mut response = http::Response::builder().status(200);
                if index == 0 {
                    response =
                        response.header(header::SET_COOKIE, "__oailb=owner-a; Path=/; Secure");
                }
                Ok(response.body(AsyncBody::default())?)
            }
        });
        let transport = CodexTransportState::default();
        let first = transport.client(inner.clone(), Some("account-a"), 1);
        let send = |client: &Arc<dyn HttpClient>| {
            block_on(
                client.send(
                    Request::builder()
                        .uri("https://chatgpt.com/backend-api/codex/responses")
                        .body(AsyncBody::default())?,
                ),
            )
        };
        send(&first)?;
        send(&transport.client(inner.clone(), Some("account-a"), 1))?;
        send(&transport.client(inner.clone(), Some("account-a"), 2))?;
        send(&transport.client(inner, Some("account-b"), 3))?;
        send(&first)?;
        let headers = headers.lock();
        let cookie = |index: usize| {
            headers
                .get(index)
                .and_then(|headers| headers.get(header::COOKIE))
                .and_then(|value| value.to_str().ok())
        };
        assert_eq!(cookie(0), None);
        assert_eq!(cookie(1), Some("__oailb=owner-a"));
        assert_eq!(cookie(2), None);
        assert_eq!(cookie(3), None);
        assert_eq!(cookie(4), Some("__oailb=owner-a"));
        Ok(())
    }
}
