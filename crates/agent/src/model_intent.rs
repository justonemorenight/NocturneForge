//! Resolves native subagent roles onto models the user actually has.
//!
//! Roles are configured as either an explicit `provider`/`model` pin or a
//! capability `intent`. Pins win while they stay available; intent resolution is
//! the fallback so a role keeps working when the pinned model disappears or when
//! the user never pinned one. Every path fails safe to the parent model.

use agent_orchestration::{
    IntentModelCandidate, ModelIntent, classify_model_tier, resolve_model_intent,
};
use agent_settings::NativeSubagentRolesSettings;
use gpui::App;
use language_model::LanguageModelRegistry;
use settings::{LanguageModelProviderSetting, LanguageModelSelection, NativeSubagentModelIntent};

use crate::SubagentRole;

/// Builds the intent-resolution catalog from the models the user has available.
///
/// Only authenticated providers contribute, and only models that can actually
/// run an agent turn, so a resolved intent is always runnable.
pub(crate) fn intent_model_candidates(cx: &App) -> Vec<IntentModelCandidate> {
    let Some(registry) = LanguageModelRegistry::try_read_global(cx) else {
        return Vec::new();
    };
    let mut candidates = Vec::new();

    for provider in registry.visible_providers() {
        if !provider.is_authenticated(cx) {
            continue;
        }
        let provider_id = provider.id().0.to_string();
        let fast_model_id = provider.default_fast_model(cx).map(|model| model.id());
        let flagship_model_id = provider.default_model(cx).map(|model| model.id());

        for model in provider.provided_models(cx) {
            if model.is_disabled().is_some() || !model.supports_tools() {
                continue;
            }
            let model_id = model.id();
            candidates.push(IntentModelCandidate {
                provider_id: provider_id.clone(),
                tier: classify_model_tier(
                    fast_model_id.as_ref() == Some(&model_id),
                    flagship_model_id.as_ref() == Some(&model_id),
                ),
                model_id: model_id.0.to_string(),
                supports_thinking: model.supports_thinking(),
                effort_levels: model
                    .supported_effort_levels()
                    .into_iter()
                    .map(|level| level.value.to_string())
                    .collect(),
                default_effort: model
                    .default_effort_level()
                    .map(|level| level.value.to_string()),
            });
        }
    }

    candidates
}

fn to_model_intent(intent: NativeSubagentModelIntent) -> ModelIntent {
    match intent {
        NativeSubagentModelIntent::Fast => ModelIntent::Fast,
        NativeSubagentModelIntent::Balanced => ModelIntent::Balanced,
        NativeSubagentModelIntent::Strong => ModelIntent::Strong,
        NativeSubagentModelIntent::SameAsParent => ModelIntent::SameAsParent,
    }
}

fn intent_name(intent: NativeSubagentModelIntent) -> &'static str {
    match intent {
        NativeSubagentModelIntent::Fast => "fast",
        NativeSubagentModelIntent::Balanced => "balanced",
        NativeSubagentModelIntent::Strong => "strong",
        NativeSubagentModelIntent::SameAsParent => "same_as_parent",
    }
}

/// Resolves the model a role should run on, or `None` to inherit the parent model.
pub(crate) fn resolve_role_model_selection(
    role: SubagentRole,
    settings: &NativeSubagentRolesSettings,
    candidates: &[IntentModelCandidate],
    parent_provider_id: Option<&str>,
) -> Option<LanguageModelSelection> {
    if !settings.enabled {
        return None;
    }
    let role_settings = role.role_settings(settings);

    if let Some(pinned) = role.pinned_selection(settings) {
        if let Some(pinned) = normalize_model_selection(pinned.clone(), candidates) {
            return Some(pinned);
        }
        log::warn!(
            "native subagent role '{}' pins '{}/{}', which is not available; \
             resolving the '{}' intent instead",
            role.identifier(),
            pinned.provider.0,
            pinned.model,
            intent_name(role_settings.intent),
        );
    }

    let filtered_candidates: Vec<IntentModelCandidate>;
    let candidates = if let Some(allowed) = role_settings.allowed_models(settings) {
        filtered_candidates = candidates
            .iter()
            .filter(|candidate| is_model_allowed(candidate, allowed))
            .cloned()
            .collect();
        &filtered_candidates
    } else {
        candidates
    };

    resolve_intent_selection(
        to_model_intent(role_settings.intent),
        role_settings.effort.as_deref(),
        candidates,
        parent_provider_id,
    )
}

fn normalize_provider_token(provider_id: &str) -> String {
    let s = provider_id
        .trim()
        .replace([' ', '_', '-'], "")
        .to_lowercase();
    match s.as_str() {
        "supergrok" | "xaisubscribed" => "x_ai_subscribed".to_string(),
        "chatgpt" | "openaisubscribed" => "openai-subscribed".to_string(),
        "9router" | "9routerprovider" => "9router".to_string(),
        _ => s,
    }
}

fn normalize_model_token(s: &str) -> String {
    s.trim().replace([' ', '.'], "-").to_lowercase()
}

fn model_id_matches(candidate_model: &str, allowed_model: &str) -> bool {
    let cand = normalize_model_token(candidate_model);
    let allowed = normalize_model_token(allowed_model);
    if cand == allowed || cand.starts_with(&allowed) || cand.ends_with(&allowed) {
        return true;
    }
    let cand_no_slashes = cand.replace('/', "-");
    let allowed_no_slashes = allowed.replace('/', "-");
    if cand_no_slashes == allowed_no_slashes
        || cand_no_slashes.ends_with(&allowed_no_slashes)
        || cand_no_slashes.contains(&allowed_no_slashes)
    {
        return true;
    }
    false
}

fn is_model_allowed(candidate: &IntentModelCandidate, allowed_models: &[String]) -> bool {
    let candidate_full_id = format!("{}/{}", candidate.provider_id, candidate.model_id);
    let norm_candidate_full_id = normalize_model_token(&candidate_full_id);
    let norm_candidate_provider = normalize_provider_token(&candidate.provider_id);

    allowed_models.iter().any(|allowed| {
        let allowed = allowed.trim();
        let norm_allowed = normalize_model_token(allowed);

        if norm_allowed == norm_candidate_full_id {
            return true;
        }

        if model_id_matches(&candidate.model_id, allowed) {
            return true;
        }

        if let Some((provider, model)) = allowed.split_once('/') {
            let norm_provider = normalize_provider_token(provider);
            if norm_provider == norm_candidate_provider
                && model_id_matches(&candidate.model_id, model)
            {
                return true;
            }
        }
        false
    })
}

/// Resolves a bare model intent against the catalog, or `None` to inherit the
/// parent model. Shared by role routing and by callers that need a model for a
/// specific capability, such as an acceptance reviewer.
pub(crate) fn resolve_intent_selection(
    intent: ModelIntent,
    effort_override: Option<&str>,
    candidates: &[IntentModelCandidate],
    parent_provider_id: Option<&str>,
) -> Option<LanguageModelSelection> {
    let resolved = resolve_model_intent(intent, candidates, parent_provider_id, effort_override)?;
    log::debug!("native subagent model resolution: {}", resolved.reason);

    let supports_thinking = candidates
        .iter()
        .find(|candidate| {
            candidate.provider_id == resolved.provider_id && candidate.model_id == resolved.model_id
        })
        .map(|candidate| candidate.supports_thinking)
        .unwrap_or(true);

    Some(LanguageModelSelection {
        provider: LanguageModelProviderSetting(resolved.provider_id),
        model: resolved.model_id,
        enable_thinking: supports_thinking,
        effort: resolved.effort,
        speed: None,
    })
}

/// Builds a short text manifest describing each role's resolved model.
///
/// Injected into the `spawn_agent` tool guidance so the orchestrator knows
/// which models are available for delegation and can make informed decisions
/// about task assignment.
pub(crate) fn role_model_manifest(
    settings: &NativeSubagentRolesSettings,
    candidates: &[IntentModelCandidate],
    parent_provider_id: Option<&str>,
    parent_model_id: Option<&str>,
) -> Option<String> {
    if !settings.enabled || candidates.is_empty() {
        return None;
    }

    let roles = [
        (
            SubagentRole::Explorer,
            "explorer",
            "Read-only search, grep, file discovery",
        ),
        (
            SubagentRole::FlowReader,
            "flow_reader",
            "Read-only flow tracing, architecture analysis",
        ),
        (
            SubagentRole::CodingWorker,
            "coding_worker",
            "Implementation, edits, tests, debugging",
        ),
    ];

    let mut lines = Vec::new();
    for (role, name, purpose) in &roles {
        let role_settings = role.role_settings(settings);
        let resolved =
            resolve_role_model_selection(*role, settings, candidates, parent_provider_id);
        let model_desc = match &resolved {
            Some(selection) => format!("{}/{}", selection.provider.0, selection.model),
            None => match (parent_provider_id, parent_model_id) {
                (Some(provider), Some(model)) => format!("{provider}/{model} (inherits parent)"),
                _ => "(inherits parent model)".to_string(),
            },
        };
        let intent_label = intent_name(role_settings.intent);
        lines.push(format!(
            "- {name}: {model_desc} · {intent_label} intent · {purpose}"
        ));
    }

    Some(format!(
        "\n### Available worker models\n\nThe user has configured these subagent roles. Choose the role that best fits each delegated task:\n{}\n\nUse `agent_type` to select the role. The runtime resolves the model automatically.",
        lines.join("\n")
    ))
}

pub(crate) fn normalize_model_selection(
    mut selection: LanguageModelSelection,
    candidates: &[IntentModelCandidate],
) -> Option<LanguageModelSelection> {
    let candidate = candidates.iter().find(|candidate| {
        candidate.provider_id == selection.provider.0 && candidate.model_id == selection.model
    })?;
    selection.enable_thinking = candidate.supports_thinking;
    selection.effort = if candidate.supports_thinking {
        selection
            .effort
            .filter(|effort| candidate.effort_levels.iter().any(|level| level == effort))
            .or_else(|| candidate.default_effort.clone())
    } else {
        None
    };
    Some(selection)
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_orchestration::ModelTier;
    use agent_settings::NativeSubagentRoleSettings;
    use settings::{NativeSubagentRoleContent, NativeSubagentRolesContent};

    fn model_selection_id(selection: &LanguageModelSelection) -> String {
        format!("{}/{}", selection.provider.0, selection.model)
    }

    fn candidate(
        provider_id: &str,
        model_id: &str,
        tier: ModelTier,
        effort_levels: &[&str],
    ) -> IntentModelCandidate {
        IntentModelCandidate {
            provider_id: provider_id.to_string(),
            model_id: model_id.to_string(),
            tier,
            supports_thinking: !effort_levels.is_empty(),
            effort_levels: effort_levels
                .iter()
                .map(|level| level.to_string())
                .collect(),
            default_effort: effort_levels.last().map(|level| level.to_string()),
        }
    }

    fn settings_with_role(role: NativeSubagentRoleContent) -> NativeSubagentRolesSettings {
        NativeSubagentRolesSettings::from(NativeSubagentRolesContent {
            explorer: Some(role),
            ..Default::default()
        })
    }

    fn catalog() -> Vec<IntentModelCandidate> {
        vec![
            candidate("proxy", "flash", ModelTier::Balanced, &["low", "medium"]),
            candidate("proxy", "flagship", ModelTier::Strong, &["high", "xhigh"]),
        ]
    }

    #[test]
    fn unpinned_role_resolves_by_its_intent() {
        let settings = NativeSubagentRolesSettings::default();
        let resolved = resolve_role_model_selection(
            SubagentRole::CodingWorker,
            &settings,
            &catalog(),
            Some("proxy"),
        )
        .expect("coding worker resolves by intent");

        assert_eq!(resolved.model, "flash");
        assert_eq!(resolved.effort.as_deref(), Some("medium"));
        assert!(resolved.enable_thinking);
    }

    #[test]
    fn pinned_role_keeps_its_exact_model() {
        let settings = settings_with_role(NativeSubagentRoleContent {
            provider: Some(LanguageModelProviderSetting("proxy".to_string())),
            model: Some("flagship".to_string()),
            effort: Some("xhigh".to_string()),
            ..Default::default()
        });

        let resolved = resolve_role_model_selection(
            SubagentRole::Explorer,
            &settings,
            &catalog(),
            Some("proxy"),
        )
        .expect("pinned role resolves");

        assert_eq!(resolved.model, "flagship");
        assert_eq!(resolved.effort.as_deref(), Some("xhigh"));
    }

    #[test]
    fn pinned_role_normalizes_an_unsupported_effort() {
        let settings = settings_with_role(NativeSubagentRoleContent {
            provider: Some(LanguageModelProviderSetting("proxy".to_string())),
            model: Some("flash".to_string()),
            effort: Some("xhigh".to_string()),
            ..Default::default()
        });

        let resolved = resolve_role_model_selection(
            SubagentRole::Explorer,
            &settings,
            &catalog(),
            Some("proxy"),
        )
        .expect("pinned role resolves");

        assert_eq!(resolved.effort.as_deref(), Some("medium"));
    }

    #[test]
    fn unavailable_pin_falls_back_to_intent() {
        let settings = settings_with_role(NativeSubagentRoleContent {
            provider: Some(LanguageModelProviderSetting("retired-provider".to_string())),
            model: Some("retired-model".to_string()),
            ..Default::default()
        });

        let resolved = resolve_role_model_selection(
            SubagentRole::Explorer,
            &settings,
            &catalog(),
            Some("proxy"),
        )
        .expect("intent resolution recovers from a stale pin");

        // Explorer defaults to the fast intent, which has no fast candidate
        // here, so the nearest tier in the preferred order is chosen.
        assert_eq!(resolved.model, "flash");
    }

    #[test]
    fn same_as_parent_intent_inherits_the_parent_model() {
        let settings = settings_with_role(NativeSubagentRoleContent {
            intent: Some(NativeSubagentModelIntent::SameAsParent),
            ..Default::default()
        });

        assert!(
            resolve_role_model_selection(
                SubagentRole::Explorer,
                &settings,
                &catalog(),
                Some("proxy"),
            )
            .is_none()
        );
    }

    #[test]
    fn an_empty_catalog_inherits_the_parent_model() {
        let settings = NativeSubagentRolesSettings::default();
        assert!(
            resolve_role_model_selection(SubagentRole::Explorer, &settings, &[], Some("proxy"))
                .is_none()
        );
    }

    #[test]
    fn models_without_thinking_do_not_enable_thinking() {
        let candidates = vec![candidate(
            "supergrok",
            "grok-build-0.1",
            ModelTier::Fast,
            &[],
        )];

        let resolved =
            resolve_intent_selection(ModelIntent::Fast, None, &candidates, Some("supergrok"))
                .expect("fast intent resolves");

        assert!(!resolved.enable_thinking);
        assert_eq!(resolved.effort, None);
    }

    #[test]
    fn manifest_shows_resolved_models_for_each_role() {
        let settings = NativeSubagentRolesSettings::default();
        let manifest =
            role_model_manifest(&settings, &catalog(), Some("proxy"), Some("parent-model"))
                .expect("manifest should be built");

        assert!(
            manifest.contains("explorer"),
            "manifest should list explorer role"
        );
        assert!(
            manifest.contains("flow_reader"),
            "manifest should list flow_reader role"
        );
        assert!(
            manifest.contains("coding_worker"),
            "manifest should list coding_worker role"
        );
        assert!(
            manifest.contains("agent_type"),
            "manifest should mention agent_type"
        );
    }

    #[test]
    fn manifest_shows_pinned_model_when_configured() {
        let settings = settings_with_role(NativeSubagentRoleContent {
            provider: Some(LanguageModelProviderSetting("proxy".to_string())),
            model: Some("flagship".to_string()),
            effort: Some("xhigh".to_string()),
            ..Default::default()
        });
        let manifest =
            role_model_manifest(&settings, &catalog(), Some("proxy"), Some("parent-model"))
                .expect("manifest should be built");

        assert!(
            manifest.contains("proxy/flagship"),
            "manifest should show the pinned model"
        );
    }

    #[test]
    fn manifest_shows_parent_inheritance_for_same_as_parent_intent() {
        let mut settings = NativeSubagentRolesSettings::default();
        settings.explorer = NativeSubagentRoleSettings {
            provider: None,
            model: None,
            effort: None,
            intent: NativeSubagentModelIntent::SameAsParent,
            fallback: agent_settings::SubagentFallbackModelSettings::InheritFromParent,
            allowed_models: None,
        };
        let manifest =
            role_model_manifest(&settings, &catalog(), Some("proxy"), Some("parent-model"))
                .expect("manifest should be built");

        assert!(
            manifest.contains("inherits parent"),
            "same_as_parent intent should show inheritance: {manifest}"
        );
    }

    #[test]
    fn allowed_models_restricts_auto_intent_selection() {
        // Catalog has flash (Balanced) and flagship (Strong).
        // Restricting allowed_models to only "flagship" forces a Balanced intent to pick "flagship".
        let settings = NativeSubagentRolesSettings::from(NativeSubagentRolesContent {
            allowed_models: Some(vec!["proxy/flagship".to_string()]),
            ..Default::default()
        });

        let resolved = resolve_role_model_selection(
            SubagentRole::CodingWorker,
            &settings,
            &catalog(),
            Some("proxy"),
        )
        .expect("coding worker should resolve to allowed model");

        assert_eq!(resolved.model, "flagship");
    }

    #[test]
    fn allowed_models_matches_by_model_id_only() {
        let settings = NativeSubagentRolesSettings::from(NativeSubagentRolesContent {
            allowed_models: Some(vec!["flash".to_string()]),
            ..Default::default()
        });

        let resolved = resolve_role_model_selection(
            SubagentRole::CodingWorker,
            &settings,
            &catalog(),
            Some("proxy"),
        )
        .expect("coding worker should resolve to model matched by model_id");

        assert_eq!(resolved.model, "flash");
    }

    #[test]
    fn role_specific_allowed_models_overrides_global() {
        let settings = NativeSubagentRolesSettings::from(NativeSubagentRolesContent {
            allowed_models: Some(vec!["flash".to_string()]),
            coding_worker: Some(NativeSubagentRoleContent {
                allowed_models: Some(vec!["flagship".to_string()]),
                ..Default::default()
            }),
            ..Default::default()
        });

        // Explorer uses global list ("flash")
        let explorer = resolve_role_model_selection(
            SubagentRole::Explorer,
            &settings,
            &catalog(),
            Some("proxy"),
        )
        .expect("explorer should resolve to global allowed model");
        assert_eq!(explorer.model, "flash");

        // Coding worker uses role-specific override ("flagship")
        let worker = resolve_role_model_selection(
            SubagentRole::CodingWorker,
            &settings,
            &catalog(),
            Some("proxy"),
        )
        .expect("coding worker should resolve to role-specific allowed model");
        assert_eq!(worker.model, "flagship");
    }

    #[test]
    fn allowed_models_supports_provider_aliases_and_prefix_matching() {
        let candidates = vec![
            candidate("x_ai_subscribed", "grok-4.6", ModelTier::Balanced, &[]),
            candidate("anthropic", "claude-3-5-haiku-latest", ModelTier::Fast, &[]),
        ];

        let settings = NativeSubagentRolesSettings::from(NativeSubagentRolesContent {
            allowed_models: Some(vec![
                "supergrok/grok-4.6".to_string(),
                "anthropic/claude-3-5-haiku".to_string(),
            ]),
            ..Default::default()
        });

        // "supergrok" matches "x_ai_subscribed" provider
        let worker =
            resolve_role_model_selection(SubagentRole::CodingWorker, &settings, &candidates, None)
                .expect("coding worker should resolve to supergrok/grok-4.6");
        assert_eq!(worker.provider.0, "x_ai_subscribed");
        assert_eq!(worker.model, "grok-4.6");

        // "claude-3-5-haiku" matches prefix of "claude-3-5-haiku-latest"
        let explorer =
            resolve_role_model_selection(SubagentRole::Explorer, &settings, &candidates, None)
                .expect("explorer should resolve to claude-3-5-haiku-latest");
        assert_eq!(explorer.provider.0, "anthropic");
        assert_eq!(explorer.model, "claude-3-5-haiku-latest");
    }

    #[test]
    fn allowed_models_matches_spaced_names() {
        let candidates = vec![
            candidate("google", "gemini-3.8-flash", ModelTier::Fast, &[]),
            candidate("x_ai_subscribed", "grok-4.6", ModelTier::Balanced, &[]),
            candidate("deepseek", "deepseek-v4.1", ModelTier::Balanced, &[]),
            candidate("openai-subscribed", "gpt-5.6-luna", ModelTier::Strong, &[]),
        ];

        let settings = NativeSubagentRolesSettings::from(NativeSubagentRolesContent {
            allowed_models: Some(vec![
                "gemini 3.8 flash".to_string(),
                "grok 4.6".to_string(),
                "deepseek v4.1".to_string(),
                "gpt-5.6-luna".to_string(),
            ]),
            ..Default::default()
        });

        // Fast intent resolves to gemini-3.8-flash
        let fast =
            resolve_role_model_selection(SubagentRole::Explorer, &settings, &candidates, None)
                .expect("should resolve to gemini-3.8-flash");
        assert_eq!(fast.model, "gemini-3.8-flash");

        // Strong intent resolves to gpt-5.6-luna
        let strong = resolve_intent_selection(ModelIntent::Strong, None, &candidates, None)
            .expect("should resolve to gpt-5.6-luna");
        assert_eq!(strong.model, "gpt-5.6-luna");
    }

    #[test]
    fn allowed_models_matches_exact_user_settings_configuration() {
        // Exact candidates as configured in user's ~/.config/zed/settings.json
        let candidates = vec![
            candidate(
                "9Router Provider",
                "ag/gemini-3.8-flash-high",
                ModelTier::Fast,
                &[],
            ),
            candidate(
                "9Router Provider",
                "ag/claude-opus-4-6-thinking",
                ModelTier::Strong,
                &[],
            ),
            candidate(
                "9Router Provider",
                "cmc/deepseek/deepseek-v4.1-flash",
                ModelTier::Fast,
                &[],
            ),
            candidate("openai-subscribed", "gpt-5.6-luna", ModelTier::Fast, &[]),
            candidate("x_ai_subscribed", "grok-4.6", ModelTier::Balanced, &[]),
        ];

        // Test 1: Full identifier paths
        let settings_full = NativeSubagentRolesSettings::from(NativeSubagentRolesContent {
            allowed_models: Some(vec![
                "9Router Provider/ag/gemini-3.8-flash-high".to_string(),
                "9Router Provider/ag/claude-opus-4-6-thinking".to_string(),
                "9Router Provider/cmc/deepseek/deepseek-v4.1-flash".to_string(),
                "openai-subscribed/gpt-5.6-luna".to_string(),
                "x_ai_subscribed/grok-4.6".to_string(),
            ]),
            ..Default::default()
        });
        assert!(
            resolve_role_model_selection(
                SubagentRole::Explorer,
                &settings_full,
                &candidates,
                Some("9Router Provider"),
            )
            .is_some()
        );

        // Test 2: Natural names ("9router", shorthand model names)
        let settings_shorthand = NativeSubagentRolesSettings::from(NativeSubagentRolesContent {
            allowed_models: Some(vec![
                "9router/gemini 3.8 flash high".to_string(),
                "9router/opus 4.6".to_string(),
                "9router/deepseek v4.1 flash".to_string(),
                "chatgpt/gpt-5.6-luna".to_string(),
                "supergrok/grok-4.6".to_string(),
            ]),
            flow_reader: Some(NativeSubagentRoleContent {
                intent: Some(NativeSubagentModelIntent::Balanced),
                ..Default::default()
            }),
            coding_worker: Some(NativeSubagentRoleContent {
                intent: Some(NativeSubagentModelIntent::Strong),
                ..Default::default()
            }),
            ..Default::default()
        });

        // Strong intent resolves to ag/claude-opus-4-6-thinking
        let strong = resolve_role_model_selection(
            SubagentRole::CodingWorker,
            &settings_shorthand,
            &candidates,
            None,
        )
        .expect("should resolve opus 4.6");
        assert_eq!(strong.provider.0, "9Router Provider");
        assert_eq!(strong.model, "ag/claude-opus-4-6-thinking");

        // Balanced intent resolves to grok-4.6
        let balanced = resolve_role_model_selection(
            SubagentRole::FlowReader,
            &settings_shorthand,
            &candidates,
            None,
        )
        .expect("should resolve grok-4.6");
        assert_eq!(balanced.provider.0, "x_ai_subscribed");
        assert_eq!(balanced.model, "grok-4.6");
    }

    #[test]
    fn fallback_selection_does_not_duplicate_primary_when_inheriting_parent() {
        let parent_selection = LanguageModelSelection {
            provider: LanguageModelProviderSetting("proxy".to_string()),
            model: "flash".to_string(),
            enable_thinking: true,
            effort: None,
            speed: None,
        };
        let candidates = catalog();
        let settings = NativeSubagentRolesSettings::default(); // default has fallback: InheritFromParent

        let role = SubagentRole::Explorer;
        let fallback = role
            .fallback_model_selection(&settings, Some(&parent_selection))
            .and_then(|sel| normalize_model_selection(sel, &candidates))
            .expect("fallback selection exists");

        let parent_model_id = model_selection_id(&parent_selection);
        let fallback_id = model_selection_id(&fallback);

        // When primary inherits from parent (model_override is None),
        // effective_primary is parent_model_id.
        let effective_primary = parent_model_id.as_str();
        assert_eq!(
            effective_primary,
            fallback_id.as_str(),
            "fallback matches parent"
        );

        // Effective primary matching fallback must prevent setting fallback_model_override
        let should_set_fallback = effective_primary != fallback_id.as_str();
        assert!(
            !should_set_fallback,
            "fallback should NOT be set when it matches effective primary"
        );
    }

    #[test]
    fn disabled_settings_inherits_the_parent_model() {
        let mut settings = NativeSubagentRolesSettings::default();
        settings.enabled = false;
        assert!(
            resolve_role_model_selection(
                SubagentRole::Explorer,
                &settings,
                &catalog(),
                Some("proxy"),
            )
            .is_none()
        );
    }

    #[test]
    fn manifest_is_none_when_disabled_or_empty_catalog() {
        let settings = NativeSubagentRolesSettings::default();
        assert!(role_model_manifest(&settings, &[], Some("proxy"), Some("m")).is_none());

        let mut disabled = NativeSubagentRolesSettings::default();
        disabled.enabled = false;
        assert!(role_model_manifest(&disabled, &catalog(), Some("proxy"), Some("m")).is_none());
    }
}
