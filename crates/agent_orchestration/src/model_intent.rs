//! Catalog-driven resolution of [`ModelIntent`] onto concrete models.
//!
//! Intent is a capability request, so this module deliberately works on a
//! provider-agnostic candidate list instead of naming providers or models. The
//! caller builds the list from whatever the user has configured, which keeps
//! routing stable when providers are added, removed, or renamed.

use crate::ModelIntent;

/// Capability tier a candidate model sits in.
///
/// Tiers come from the provider's own declarations: a provider's fast model is
/// `Fast`, its flagship model is `Strong`, and every other offered model is
/// `Balanced`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelTier {
    Fast,
    Balanced,
    Strong,
}

/// A model the user actually has available, annotated with the capability
/// signals needed to satisfy an intent.
#[derive(Clone, Debug, PartialEq)]
pub struct IntentModelCandidate {
    pub provider_id: String,
    pub model_id: String,
    pub tier: ModelTier,
    pub supports_thinking: bool,
    /// Supported reasoning-effort values, in provider order.
    pub effort_levels: Vec<String>,
    pub default_effort: Option<String>,
}

/// The outcome of resolving an intent against the candidate catalog.
#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedIntentModel {
    pub provider_id: String,
    pub model_id: String,
    pub tier: ModelTier,
    /// `None` when the chosen model does not support reasoning effort, or when
    /// none of the preferred levels are supported.
    pub effort: Option<String>,
    pub reason: String,
}

impl ResolvedIntentModel {
    /// Selection ID in the `provider/model` form used by task overrides.
    pub fn selection_id(&self) -> String {
        format!("{}/{}", self.provider_id, self.model_id)
    }
}

/// Classifies a model by the provider's own fast/flagship declarations.
///
/// Single-model providers declare the same model for both roles; they land in
/// the middle tier so that every intent can still resolve to them.
pub fn classify_model_tier(is_fast: bool, is_flagship: bool) -> ModelTier {
    match (is_fast, is_flagship) {
        (true, false) => ModelTier::Fast,
        (false, true) => ModelTier::Strong,
        _ => ModelTier::Balanced,
    }
}

/// Tiers to try, in order, for an intent.
pub fn preferred_tiers(intent: ModelIntent) -> &'static [ModelTier] {
    const FAST: &[ModelTier] = &[ModelTier::Fast, ModelTier::Balanced, ModelTier::Strong];
    const BALANCED: &[ModelTier] = &[ModelTier::Balanced, ModelTier::Strong, ModelTier::Fast];
    const STRONG: &[ModelTier] = &[ModelTier::Strong, ModelTier::Balanced, ModelTier::Fast];

    match intent {
        ModelIntent::Fast => FAST,
        ModelIntent::Balanced => BALANCED,
        ModelIntent::Strong => STRONG,
        ModelIntent::SameAsParent => &[],
    }
}

/// Reasoning-effort values to try, in order, for an intent.
pub fn preferred_efforts(intent: ModelIntent) -> &'static [&'static str] {
    match intent {
        ModelIntent::Fast => &["minimal", "low"],
        ModelIntent::Balanced => &["medium", "low", "high"],
        ModelIntent::Strong => &["xhigh", "max", "high", "medium"],
        ModelIntent::SameAsParent => &[],
    }
}

/// Resolves `intent` to a model the user has available.
///
/// Candidates on the parent provider are considered first, so a role stays on
/// the same provider as its parent when that provider can satisfy the intent;
/// this preserves prompt-cache affinity and keeps reviewer continuity.
///
/// Returns `None` for [`ModelIntent::SameAsParent`], and for intents no
/// candidate can satisfy. Callers treat `None` as "keep the parent model",
/// which is the fail-safe behavior: an unresolvable intent must never break a
/// delegation.
pub fn resolve_model_intent(
    intent: ModelIntent,
    candidates: &[IntentModelCandidate],
    parent_provider_id: Option<&str>,
    effort_override: Option<&str>,
) -> Option<ResolvedIntentModel> {
    if intent == ModelIntent::SameAsParent {
        return None;
    }

    let (preferred, remaining): (Vec<&IntentModelCandidate>, Vec<&IntentModelCandidate>) =
        candidates
            .iter()
            .partition(|candidate| Some(candidate.provider_id.as_str()) == parent_provider_id);
    // Pass two falls back to the rest of the catalog, so a role still resolves
    // when the parent provider offers nothing for the requested tier.
    let ordered = preferred.into_iter().chain(remaining);

    for tier in preferred_tiers(intent) {
        let Some(candidate) = ordered.clone().find(|candidate| candidate.tier == *tier) else {
            continue;
        };
        let effort = resolve_effort(candidate, intent, effort_override);
        let reason = if Some(candidate.provider_id.as_str()) == parent_provider_id {
            format!(
                "intent {} resolved to the {} tier on the parent provider",
                intent_label(intent),
                tier_label(*tier)
            )
        } else {
            format!(
                "intent {} resolved to the {} tier from the catalog",
                intent_label(intent),
                tier_label(*tier)
            )
        };
        return Some(ResolvedIntentModel {
            provider_id: candidate.provider_id.clone(),
            model_id: candidate.model_id.clone(),
            tier: *tier,
            effort,
            reason,
        });
    }

    None
}

fn resolve_effort(
    candidate: &IntentModelCandidate,
    intent: ModelIntent,
    effort_override: Option<&str>,
) -> Option<String> {
    if !candidate.supports_thinking || candidate.effort_levels.is_empty() {
        return None;
    }

    if let Some(requested) = effort_override
        && candidate
            .effort_levels
            .iter()
            .any(|level| level.as_str() == requested)
    {
        return Some(requested.to_string());
    }

    for preferred in preferred_efforts(intent) {
        if candidate
            .effort_levels
            .iter()
            .any(|level| level.as_str() == *preferred)
        {
            return Some((*preferred).to_string());
        }
    }

    // An override that the model cannot honor still needs a usable effort, so
    // fall back to the provider's declared default before giving up.
    candidate.default_effort.clone()
}

pub(crate) fn intent_label(intent: ModelIntent) -> &'static str {
    match intent {
        ModelIntent::Fast => "fast",
        ModelIntent::Balanced => "balanced",
        ModelIntent::Strong => "strong",
        ModelIntent::SameAsParent => "same_as_parent",
    }
}

pub(crate) fn tier_label(tier: ModelTier) -> &'static str {
    match tier {
        ModelTier::Fast => "fast",
        ModelTier::Balanced => "balanced",
        ModelTier::Strong => "strong",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn catalog() -> Vec<IntentModelCandidate> {
        vec![
            candidate(
                "proxy",
                "flash",
                ModelTier::Balanced,
                &["low", "medium", "high"],
            ),
            candidate(
                "flagship",
                "flagship-model",
                ModelTier::Strong,
                &["high", "xhigh"],
            ),
            candidate(
                "flagship",
                "flagship-fast",
                ModelTier::Fast,
                &["minimal", "low"],
            ),
            candidate(
                "vendor",
                "vendor-flagship",
                ModelTier::Strong,
                &["medium", "high"],
            ),
        ]
    }

    #[test]
    fn single_model_providers_land_in_the_middle_tier() {
        assert_eq!(classify_model_tier(true, false), ModelTier::Fast);
        assert_eq!(classify_model_tier(false, true), ModelTier::Strong);
        assert_eq!(classify_model_tier(false, false), ModelTier::Balanced);
        // A provider that declares one model for both roles must remain
        // reachable for every intent.
        assert_eq!(classify_model_tier(true, true), ModelTier::Balanced);
    }

    #[test]
    fn same_as_parent_intent_keeps_the_parent_model() {
        assert_eq!(
            resolve_model_intent(ModelIntent::SameAsParent, &catalog(), Some("proxy"), None),
            None
        );
    }

    #[test]
    fn intent_prefers_its_own_tier_on_the_parent_provider() {
        let resolved =
            resolve_model_intent(ModelIntent::Strong, &catalog(), Some("flagship"), None)
                .expect("strong intent resolves");
        assert_eq!(resolved.provider_id, "flagship");
        assert_eq!(resolved.model_id, "flagship-model");
        assert_eq!(resolved.tier, ModelTier::Strong);
        assert!(resolved.reason.contains("parent provider"));
    }

    #[test]
    fn fast_intent_prefers_the_declared_fast_model() {
        let resolved = resolve_model_intent(ModelIntent::Fast, &catalog(), Some("flagship"), None)
            .expect("fast intent resolves");
        assert_eq!(resolved.model_id, "flagship-fast");
        assert_eq!(resolved.effort.as_deref(), Some("minimal"));
    }

    #[test]
    fn balanced_intent_falls_back_to_another_tier_when_needed() {
        let candidates = vec![candidate(
            "vendor",
            "only-flagship",
            ModelTier::Strong,
            &["high"],
        )];
        let resolved = resolve_model_intent(ModelIntent::Balanced, &candidates, None, None)
            .expect("balanced intent degrades to the strong tier");
        assert_eq!(resolved.model_id, "only-flagship");
        assert_eq!(resolved.tier, ModelTier::Strong);
        assert_eq!(resolved.effort.as_deref(), Some("high"));
    }

    #[test]
    fn intent_falls_back_to_the_catalog_when_the_parent_provider_cannot_satisfy_it() {
        let resolved = resolve_model_intent(ModelIntent::Fast, &catalog(), Some("proxy"), None)
            .expect("fast intent resolves from the rest of the catalog");
        // The parent provider offers no fast model, so the declared fast model
        // of any other provider is used instead.
        assert_eq!(resolved.model_id, "flagship-fast");
        assert!(resolved.reason.contains("from the catalog"));
    }

    #[test]
    fn effort_override_wins_only_when_the_model_supports_it() {
        let resolved = resolve_model_intent(
            ModelIntent::Strong,
            &catalog(),
            Some("flagship"),
            Some("high"),
        )
        .expect("strong intent resolves");
        assert_eq!(resolved.effort.as_deref(), Some("high"));

        let unsupported = resolve_model_intent(
            ModelIntent::Strong,
            &catalog(),
            Some("flagship"),
            Some("xhigh-not-declared"),
        )
        .expect("strong intent resolves");
        assert_eq!(unsupported.effort.as_deref(), Some("xhigh"));
    }

    #[test]
    fn models_without_reasoning_effort_resolve_without_effort() {
        let candidates = vec![IntentModelCandidate {
            provider_id: "supergrok".to_string(),
            model_id: "grok-build-0.1".to_string(),
            tier: ModelTier::Fast,
            supports_thinking: false,
            effort_levels: Vec::new(),
            default_effort: None,
        }];
        let resolved = resolve_model_intent(ModelIntent::Fast, &candidates, None, Some("high"))
            .expect("fast intent resolves");
        assert_eq!(resolved.effort, None);
        assert_eq!(resolved.selection_id(), "supergrok/grok-build-0.1");
    }

    #[test]
    fn an_empty_catalog_resolves_to_nothing() {
        assert_eq!(
            resolve_model_intent(ModelIntent::Fast, &[], None, None),
            None
        );
    }
}
