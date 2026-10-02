use chrono::{DateTime, Utc};
use globset::GlobSetBuilder;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::time::Duration;

use crate::plan_graph::OrchestrationTask;
use crate::truncate_text;

/// Classification of errors encountered during task execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ErrorClass {
    /// Network connection drops, timeouts, socket errors.
    TransientNetwork,
    /// Rate limit reached or quota exceeded temporarily.
    RateLimit,
    /// The requested provider/model is unavailable or not configured.
    ProviderUnavailable,
    /// The request cannot fit in the active model's context window.
    ContextOverflow,
    /// Tool returned an error that may be fixed with retry or different inputs.
    ToolExecutionError,
    /// Retained for deserializing historical runs that enforced token limits.
    TokenBudgetExceeded,
    /// Tool-call hard limit was exceeded.
    ToolCallBudgetExceeded,
    /// Verification checks or acceptance criteria failed.
    VerificationAssertionFailure,
    /// Critical unrecoverable error (e.g. malformed model output, internal bug).
    FatalError,
    /// Operation was explicitly cancelled.
    Cancelled,
}

/// The boundary at which a failed task attempt can be retried safely.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum FailureScope {
    /// Retry the same request without changing its model or credentials.
    Request,
    /// Retry with another account from the same provider pool.
    Account,
    /// Retry outside the current provider's failure domain.
    Provider,
    /// Retry with a different model, normally one with different capabilities.
    Model,
}

/// Structured failure information preserved across the task-executor boundary.
///
/// Executors use this error when an upstream service already classified a
/// failure. The scheduler can then make routing decisions without parsing a
/// provider's user-facing error text.
#[derive(Debug)]
pub struct TaskExecutionFailure {
    pub class: ErrorClass,
    pub scope: FailureScope,
    pub retry_after_ms: Option<u64>,
    pub message: String,
    source: Option<anyhow::Error>,
}

impl TaskExecutionFailure {
    pub fn new(class: ErrorClass, scope: FailureScope, message: impl Into<String>) -> Self {
        Self {
            class,
            scope,
            retry_after_ms: None,
            message: message.into(),
            source: None,
        }
    }

    pub fn with_retry_after(mut self, retry_after: Option<Duration>) -> Self {
        self.retry_after_ms =
            retry_after.map(|duration| duration.as_millis().min(u128::from(u64::MAX)) as u64);
        self
    }

    pub fn retry_after(&self) -> Option<Duration> {
        self.retry_after_ms.map(Duration::from_millis)
    }

    pub fn with_source(mut self, source: anyhow::Error) -> Self {
        self.source = Some(source);
        self
    }
}

impl fmt::Display for TaskExecutionFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for TaskExecutionFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source.as_ref().map(|source| source.as_ref())
    }
}

impl ErrorClass {
    pub fn label(self) -> &'static str {
        match self {
            Self::TransientNetwork => "transient network failure",
            Self::RateLimit => "rate limit",
            Self::ProviderUnavailable => "provider unavailable",
            Self::ContextOverflow => "context overflow",
            Self::ToolExecutionError => "tool execution error",
            Self::TokenBudgetExceeded => "token budget exceeded",
            Self::ToolCallBudgetExceeded => "tool call budget exceeded",
            Self::VerificationAssertionFailure => "verification failed",
            Self::FatalError => "fatal error",
            Self::Cancelled => "cancelled",
        }
    }

    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::TransientNetwork
                | Self::RateLimit
                | Self::ToolExecutionError
                | Self::VerificationAssertionFailure
        )
    }

    pub fn is_repairable(&self) -> bool {
        matches!(
            self,
            Self::VerificationAssertionFailure | Self::ToolExecutionError
        )
    }

    pub fn is_model_fallback_eligible(&self) -> bool {
        matches!(
            self,
            Self::TransientNetwork
                | Self::RateLimit
                | Self::ProviderUnavailable
                | Self::ContextOverflow
        )
    }
}

/// Reason explaining why a task is being retried.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RetryReason {
    TransientError(String),
    RateLimited { retry_after_secs: Option<u64> },
    VerificationFailed { feedback: String },
    ToolFailure(String),
    ManualUserRequest(String),
    Custom(String),
}

impl RetryReason {
    pub fn description(&self) -> &str {
        match self {
            Self::TransientError(msg) => msg,
            Self::RateLimited { .. } => "rate limit exceeded",
            Self::VerificationFailed { feedback } => feedback,
            Self::ToolFailure(msg) => msg,
            Self::ManualUserRequest(msg) => msg,
            Self::Custom(msg) => msg,
        }
    }
}

/// Overall verdict produced by verification checks on a task.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "snake_case")]
pub enum VerificationVerdict {
    #[default]
    Verified,
    Claimed,
    Partial,
    FailedVerification,
}

/// Result of evaluating a task's output against its acceptance criteria.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerificationResult {
    pub passed: bool,
    #[serde(default)]
    pub verdict: VerificationVerdict,
    #[serde(default)]
    pub criteria_verdicts: Vec<(String, bool)>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub criterion_results: Vec<CriterionClaim>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_output_satisfied: Option<bool>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub citations: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub citations_valid: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub feedback: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_class: Option<ErrorClass>,
    pub retryable: bool,
    pub repairable: bool,
    pub verified_at: DateTime<Utc>,
}

impl VerificationResult {
    pub fn pass() -> Self {
        Self {
            passed: true,
            verdict: VerificationVerdict::Verified,
            criteria_verdicts: Vec::new(),
            criterion_results: Vec::new(),
            expected_output_satisfied: None,
            citations: Vec::new(),
            citations_valid: None,
            feedback: None,
            error_class: None,
            retryable: false,
            repairable: false,
            verified_at: Utc::now(),
        }
    }

    pub fn fail(feedback: impl Into<String>, error_class: ErrorClass) -> Self {
        let retryable = error_class.is_retryable();
        let repairable = error_class.is_repairable();
        Self {
            passed: false,
            verdict: VerificationVerdict::FailedVerification,
            criteria_verdicts: Vec::new(),
            criterion_results: Vec::new(),
            expected_output_satisfied: None,
            citations: Vec::new(),
            citations_valid: None,
            feedback: Some(feedback.into()),
            error_class: Some(error_class),
            retryable,
            repairable,
            verified_at: Utc::now(),
        }
    }
}

/// Policy governing verification checks, retry bounds, and backoff.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct VerificationPolicy {
    /// Maximum number of automatic retries per task.
    #[serde(default = "default_max_retries")]
    pub max_retries: u8,
    /// Maximum number of repair cycles (code fix attempts) permitted before marking a semantic failure terminal.
    #[serde(default = "default_max_repair_cycles")]
    pub max_repair_cycles: u8,
    /// Initial backoff duration in milliseconds before first retry.
    #[serde(default = "default_backoff_initial_ms")]
    pub backoff_initial_ms: u64,
    /// Multiplier applied to backoff duration on consecutive retries.
    #[serde(default = "default_backoff_factor")]
    pub backoff_factor: f64,
    /// Maximum backoff duration cap in milliseconds.
    #[serde(default = "default_max_backoff_ms")]
    pub max_backoff_ms: u64,
    /// Whether to enforce acceptance criteria checks before marking task completed.
    #[serde(default = "default_true")]
    pub verify_outputs: bool,
    /// Whether to spawn repair tasks for recoverable verification failures.
    #[serde(default = "default_true")]
    pub allow_repair_tasks: bool,
}

fn default_max_retries() -> u8 {
    2
}

pub const DEFAULT_MAX_REPAIR_CYCLES: u8 = 1;

pub fn default_max_repair_cycles() -> u8 {
    DEFAULT_MAX_REPAIR_CYCLES
}

fn default_backoff_initial_ms() -> u64 {
    500
}

fn default_backoff_factor() -> f64 {
    2.0
}

fn default_max_backoff_ms() -> u64 {
    10_000
}

fn default_true() -> bool {
    true
}

impl Default for VerificationPolicy {
    fn default() -> Self {
        Self {
            max_retries: default_max_retries(),
            max_repair_cycles: default_max_repair_cycles(),
            backoff_initial_ms: default_backoff_initial_ms(),
            backoff_factor: default_backoff_factor(),
            max_backoff_ms: default_max_backoff_ms(),
            verify_outputs: true,
            allow_repair_tasks: true,
        }
    }
}

impl VerificationPolicy {
    /// Classifies an error message into a typed error category.
    pub fn classify_error(error_str: &str) -> ErrorClass {
        let lower = error_str.to_lowercase();
        if contains_any(
            &lower,
            &["rate limit", "429", "quota exceeded", "too many requests"],
        ) {
            ErrorClass::RateLimit
        } else if contains_any(
            &lower,
            &[
                "context window",
                "context overflow",
                "prompt too long",
                "maximum context",
            ],
        ) {
            ErrorClass::ContextOverflow
        } else if lower.contains("external_source_unreachable") {
            ErrorClass::FatalError
        } else if is_provider_unavailable_error(&lower) {
            ErrorClass::ProviderUnavailable
        } else if lower.contains("tool call budget") && lower.contains("exceeded") {
            ErrorClass::ToolCallBudgetExceeded
        } else if lower.contains("cancelled") || lower.contains("canceled") {
            ErrorClass::Cancelled
        } else if is_transient_network_error(&lower) {
            ErrorClass::TransientNetwork
        } else if contains_any(&lower, &["verification", "assertion", "criteria"]) {
            ErrorClass::VerificationAssertionFailure
        } else if lower.contains("tool") || lower.contains("command failed") {
            ErrorClass::ToolExecutionError
        } else {
            ErrorClass::FatalError
        }
    }

    /// Determines whether another retry should be attempted.
    pub fn should_retry(
        &self,
        attempt: u32,
        error_class: &ErrorClass,
        task_max_retries: Option<u8>,
    ) -> bool {
        let max_allowed = task_max_retries.unwrap_or(self.max_retries) as u32;
        // `attempt` is one-based in the scheduler, while max_retries is the
        // number of retries allowed after the initial attempt.
        attempt <= max_allowed && error_class.is_retryable()
    }

    /// Returns true when an error class represents a transient infrastructural issue
    /// (e.g. rate limit, network timeout, provider downtime) that warrants retrying
    /// a fresh attempt rather than code repair.
    pub fn is_transient_retryable(&self, error_class: &ErrorClass) -> bool {
        matches!(
            error_class,
            ErrorClass::TransientNetwork | ErrorClass::RateLimit | ErrorClass::ProviderUnavailable
        )
    }

    /// Calculates exponential backoff duration for a given attempt index.
    pub fn backoff_duration(&self, attempt: u32) -> Duration {
        if attempt == 0 {
            return Duration::from_millis(0);
        }
        let exp = (attempt - 1).min(10);
        let factor = self.backoff_factor.powi(exp as i32);
        let calculated_ms = (self.backoff_initial_ms as f64 * factor) as u64;
        let capped_ms = calculated_ms.min(self.max_backoff_ms);
        Duration::from_millis(capped_ms)
    }
}

fn contains_any(value: &str, patterns: &[&str]) -> bool {
    patterns.iter().any(|pattern| value.contains(pattern))
}

fn is_provider_unavailable_error(error: &str) -> bool {
    contains_any(
        error,
        &[
            "not configured or available",
            "model unavailable",
            "provider unavailable",
            "no api key",
            "unauthorized",
            "401",
            "forbidden",
            "403",
            "token_revoked",
            "refresh_token_invalidated",
            "session has ended",
            "reauthentication_required",
            "re-authenticate",
        ],
    ) || (error.contains("not configured") && contains_any(error, &["provider", "model"]))
}

fn is_transient_network_error(error: &str) -> bool {
    contains_any(
        error,
        &[
            "connection",
            "timeout",
            "timed out",
            "reset by peer",
            "broken pipe",
            "dns",
            "service unavailable",
            "503",
        ],
    )
}
/// Marker used by tasks to embed a structured verification claim in their output.
pub const VERIFICATION_START: &str = "<zed_orchestration_verification>";
/// Closing marker for a structured verification claim.
pub const VERIFICATION_END: &str = "</zed_orchestration_verification>";
const ESCAPED_VERIFICATION_START: &str = "<zed\\_orchestration\\_verification>";
const ESCAPED_VERIFICATION_END: &str = "</zed\\_orchestration\\_verification>";
const MAX_PERSISTED_VERIFICATION_CITATIONS: usize = 64;
const MAX_PERSISTED_VERIFICATION_EVIDENCE_BYTES: usize = 4 * 1024;
const MAX_PERSISTED_VERIFICATION_CITATION_BYTES: usize = 1024;

/// A claim from a task's output describing how each acceptance criterion was met.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerificationClaim {
    #[serde(default)]
    pub criteria: Vec<CriterionClaim>,
    pub expected_output_satisfied: Option<bool>,
    #[serde(default)]
    pub citations: Vec<String>,
}

/// A single acceptance criterion claim with supporting evidence.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CriterionClaim {
    pub criterion: String,
    pub passed: bool,
    pub evidence: String,
}

fn verification_envelope(output: &str) -> Option<(usize, usize, usize, usize)> {
    [
        (VERIFICATION_START, VERIFICATION_END),
        (ESCAPED_VERIFICATION_START, ESCAPED_VERIFICATION_END),
    ]
    .into_iter()
    .filter_map(|(start_marker, end_marker)| {
        let marker_start = output.rfind(start_marker)?;
        let content_start = marker_start + start_marker.len();
        let content_end = output[content_start..]
            .find(end_marker)
            .map(|index| content_start + index)?;
        Some((
            marker_start,
            content_start,
            content_end,
            content_end + end_marker.len(),
        ))
    })
    .max_by_key(|(marker_start, _, _, _)| *marker_start)
}

/// Parses the structured verification claim embedded at the end of a task output.
pub fn parse_verification_claim(output: &str) -> anyhow::Result<VerificationClaim> {
    let (_, start, end, _) = verification_envelope(output)
        .ok_or_else(|| anyhow::anyhow!("missing structured verification claim"))?;
    serde_json::from_str(output[start..end].trim())
        .map_err(|error| anyhow::anyhow!("invalid structured verification claim: {error}"))
}

/// Returns the valid verification envelope at the end of an output.
pub fn verification_claim_envelope(output: &str) -> Option<&str> {
    let (start, _, _, end) = verification_envelope(output)?;
    if !output[end..].trim().is_empty() || parse_verification_claim(output).is_err() {
        return None;
    }
    Some(&output[start..end])
}

/// Removes a valid trailing structured verification envelope from output shown
/// to the parent model or user. The unmodified output remains in runtime state
/// and artifacts for auditing and replay.
pub fn output_without_verification_claim(output: &str) -> &str {
    let Some(envelope) = verification_claim_envelope(output) else {
        return output;
    };
    let Some(start) = output.rfind(envelope) else {
        return output;
    };
    output[..start].trim_end()
}

/// Returns true when a citation looks like `path/to/file.rs:123` or
/// `path/to/file.rs:123-140`.
pub fn is_file_line_citation(citation: &str) -> bool {
    let Some((path, location)) = citation.rsplit_once(':') else {
        return false;
    };
    if path.trim().is_empty() || path.rsplit_once('.').is_none() {
        return false;
    }

    let parse_line = |line: &str| line.trim().parse::<u64>().ok().filter(|line| *line > 0);
    match location.split_once('-') {
        Some((start, end)) => match (parse_line(start), parse_line(end)) {
            (Some(start), Some(end)) => start <= end,
            _ => false,
        },
        None => parse_line(location).is_some(),
    }
}

pub fn web_citation_url(citation: &str) -> Option<&str> {
    let citation = citation.trim();
    let candidate = if citation.starts_with('[') && citation.ends_with(')') {
        citation.rsplit_once("](")?.1.strip_suffix(')')?
    } else {
        citation
    };
    let address = candidate
        .strip_prefix("https://")
        .or_else(|| candidate.strip_prefix("http://"));
    address
        .is_some_and(|address| {
            !address.is_empty()
                && address.contains('.')
                && !address.chars().any(char::is_whitespace)
        })
        .then_some(candidate)
}

pub fn is_web_citation(citation: &str) -> bool {
    web_citation_url(citation).is_some()
}

fn extract_web_citations(output: &str) -> Vec<String> {
    let visible_output = verification_envelope(output)
        .map(|(start, _, _, _)| &output[..start])
        .unwrap_or(output);
    let mut citations = Vec::new();
    let mut remainder = visible_output;

    while citations.len() < MAX_PERSISTED_VERIFICATION_CITATIONS {
        let https = remainder.find("https://");
        let http = remainder.find("http://");
        let Some(start) = https.into_iter().chain(http).min() else {
            break;
        };
        let candidate = &remainder[start..];
        let end = candidate
            .char_indices()
            .skip(1)
            .find_map(|(index, character)| {
                (character.is_whitespace()
                    || matches!(character, ')' | ']' | '}' | '>' | '"' | '\''))
                .then_some(index)
            })
            .unwrap_or(candidate.len());
        let citation = candidate[..end]
            .trim_end_matches([',', '.', ';', ':'])
            .to_string();
        if is_web_citation(&citation) && !citations.contains(&citation) {
            citations.push(citation);
        }
        remainder = &candidate[end..];
    }

    citations
}

/// Runs native verification checks against a task's output.
///
/// Checks are structural: every acceptance criterion must carry a passing claim
/// with concrete evidence, the expected output contract must be satisfied, and
/// citations (when evidence is required) must be file-and-line references that
/// fall within the task's declared scope.
#[derive(Debug, Default)]
pub struct VerificationRunner;

struct VerificationAssessment {
    criteria_verdicts: Vec<(String, bool)>,
    criterion_results: Vec<CriterionClaim>,
    expected_output_satisfied: Option<bool>,
    expected_output_valid: bool,
    citations: Vec<String>,
    citations_valid: Option<bool>,
    scope_valid: bool,
    scope_failure_details: Option<String>,
}

impl VerificationAssessment {
    fn failure_feedback(&self) -> Option<String> {
        if self.criteria_verdicts.iter().any(|(_, passed)| !passed) {
            Some(
                "One or more acceptance criteria lack a passing claim with concrete evidence"
                    .to_string(),
            )
        } else if !self.expected_output_valid {
            Some("The expected output contract was not satisfied".to_string())
        } else if self.citations_valid == Some(false) {
            Some(
                "Evidence required: provide valid file-and-line citations or source URLs"
                    .to_string(),
            )
        } else if !self.scope_valid {
            Some(match &self.scope_failure_details {
                Some(details) => format!("{SCOPE_FAILURE_FEEDBACK}. {details}"),
                None => SCOPE_FAILURE_FEEDBACK.to_string(),
            })
        } else {
            None
        }
    }

    fn into_result(self) -> VerificationResult {
        let feedback = self.failure_feedback();
        let passed = feedback.is_none();
        VerificationResult {
            passed,
            verdict: if passed {
                VerificationVerdict::Claimed
            } else {
                VerificationVerdict::FailedVerification
            },
            criteria_verdicts: self.criteria_verdicts,
            criterion_results: self.criterion_results,
            expected_output_satisfied: self.expected_output_satisfied,
            citations: self.citations,
            citations_valid: self.citations_valid,
            feedback,
            error_class: (!passed).then_some(ErrorClass::VerificationAssertionFailure),
            retryable: !passed,
            repairable: !passed,
            verified_at: Utc::now(),
        }
    }
}

impl VerificationRunner {
    pub fn verify(&self, task: &OrchestrationTask, output: &str) -> VerificationResult {
        if output.trim().is_empty() {
            return VerificationResult::fail("Task produced empty output", ErrorClass::FatalError);
        }

        let claim = match parse_verification_claim(output) {
            Ok(claim) => claim,
            Err(error) => {
                return VerificationResult::fail(
                    error.to_string(),
                    ErrorClass::VerificationAssertionFailure,
                );
            }
        };
        self.assess_claim(task, output, claim).into_result()
    }

    fn assess_claim(
        &self,
        task: &OrchestrationTask,
        output: &str,
        claim: VerificationClaim,
    ) -> VerificationAssessment {
        let criterion_results = task
            .acceptance_criteria
            .iter()
            .map(|criterion| {
                let criterion_claim = claim
                    .criteria
                    .iter()
                    .find(|result| result.criterion == *criterion);
                CriterionClaim {
                    criterion: criterion.clone(),
                    passed: criterion_claim
                        .is_some_and(|result| result.passed && !result.evidence.trim().is_empty()),
                    evidence: criterion_claim
                        .map(|result| {
                            truncate_text(
                                result.evidence.clone(),
                                MAX_PERSISTED_VERIFICATION_EVIDENCE_BYTES,
                            )
                        })
                        .unwrap_or_default(),
                }
            })
            .collect::<Vec<_>>();
        let criteria_verdicts = criterion_results
            .iter()
            .map(|result| (result.criterion.clone(), result.passed))
            .collect();
        let mut effective_citations = claim.citations.clone();
        if task.scope.is_none() {
            for citation in extract_web_citations(output) {
                if !effective_citations.contains(&citation) {
                    effective_citations.push(citation);
                }
            }
        }
        let persisted_citations = effective_citations
            .iter()
            .take(MAX_PERSISTED_VERIFICATION_CITATIONS)
            .map(|citation| web_citation_url(citation).unwrap_or(citation).to_string())
            .map(|citation| truncate_text(citation, MAX_PERSISTED_VERIFICATION_CITATION_BYTES))
            .collect::<Vec<_>>();
        let citations_valid = task.evidence_required.then(|| {
            !effective_citations.is_empty()
                && effective_citations
                    .iter()
                    .all(|citation| is_file_line_citation(citation) || is_web_citation(citation))
        });
        let scope_valid = self.scope_citations_valid(task, &effective_citations);
        let scope_failure_details = (!scope_valid)
            .then(|| scope_failure_details(task, &effective_citations))
            .flatten();
        let expected_output_satisfied = claim.expected_output_satisfied;
        let expected_output_valid =
            task.expected_output.is_none() || expected_output_satisfied == Some(true);

        VerificationAssessment {
            criteria_verdicts,
            criterion_results,
            expected_output_satisfied,
            expected_output_valid,
            citations: persisted_citations,
            citations_valid,
            scope_valid,
            scope_failure_details,
        }
    }

    /// Checks that the task has primary evidence within its declared scope.
    ///
    /// Directory scopes include their descendants. At least one citation must
    /// fall within the primary scope; additional citations may support the
    /// result from adjacent documentation or dependency metadata.
    fn scope_citations_valid(&self, task: &OrchestrationTask, citations: &[String]) -> bool {
        let Some(scope) = task.scope.as_deref() else {
            return true;
        };
        if citations.is_empty() {
            return true;
        }
        let Some(matcher) = ScopeMatcher::new(scope) else {
            return false;
        };
        citations
            .iter()
            .any(|citation| citation_path(citation).is_some_and(|path| matcher.contains(path)))
    }
}

const SCOPE_FAILURE_FEEDBACK: &str = "Citations fall outside the task's declared scope";
const MAX_SCOPE_FEEDBACK_PATHS: usize = 8;

/// Explains a scope failure so a repair can fix the claim instead of redoing
/// the task.
fn scope_failure_details(task: &OrchestrationTask, citations: &[String]) -> Option<String> {
    let scope = task.scope.as_deref()?;
    let declared = scope_entries(scope)
        .into_iter()
        .take(MAX_SCOPE_FEEDBACK_PATHS)
        .collect::<Vec<_>>();
    let cited = citations
        .iter()
        .take(MAX_SCOPE_FEEDBACK_PATHS)
        .map(String::as_str)
        .collect::<Vec<_>>();
    Some(format!(
        "Declared scope paths: {}. Cited: {}. If the work is already done inside the declared scope, do not redo it; re-emit the verification claim with citations written exactly as the declared scope paths.",
        if declared.is_empty() {
            "(none recognized)".to_string()
        } else {
            declared.join(", ")
        },
        if cited.is_empty() {
            "(none)".to_string()
        } else {
            cited.join(", ")
        },
    ))
}

/// Splits a task scope into normalized path or glob entries.
fn scope_entries(scope: &str) -> Vec<String> {
    scope
        // Task scopes are serialized by the orchestration UI as a
        // semicolon-separated list. Accept all supported separators so a
        // scope survives the round trip without becoming one invalid glob.
        .split([',', '\n', ';'])
        .flat_map(|entry| {
            let entry = entry.trim();
            if entry.split_whitespace().count() > 1 {
                entry
                    .split_whitespace()
                    .filter(|part| part.contains('/') || part.contains(['*', '?']))
                    .collect::<Vec<_>>()
            } else {
                vec![entry]
            }
        })
        .map(|entry| {
            entry.trim_matches(|character: char| {
                character.is_ascii_punctuation()
                    && !matches!(
                        character,
                        '/' | '*' | '?' | '[' | ']' | '(' | ')' | '.' | '-' | '_'
                    )
            })
        })
        .map(normalize_scope_path)
        .filter(|entry| !entry.is_empty())
        .collect()
}

fn normalize_scope_path(path: &str) -> String {
    let path = path.trim().replace('\\', "/");
    let is_absolute = path.starts_with('/');
    let components = path
        .split('/')
        .filter(|component| !component.is_empty() && *component != ".")
        .collect::<Vec<_>>();
    let joined = components.join("/");
    if is_absolute {
        format!("/{joined}")
    } else {
        joined
    }
}

fn is_absolute_scope_path(path: &str) -> bool {
    path.starts_with('/')
        || (path.as_bytes().first().is_some_and(u8::is_ascii_alphabetic)
            && path.as_bytes().get(1..3) == Some(b":/"))
}

fn is_glob_entry(entry: &str) -> bool {
    entry.contains(['*', '?'])
}

fn is_literal_component(component: &str) -> bool {
    !component.contains(['*', '?', '[', ']', '{', '}'])
}

/// Matches cited paths against a task's declared scope.
///
/// Workers cite paths relative to whatever they treat as the root: the
/// worktree (`Studio/labs/src/x.ts`), its contents (`labs/src/x.ts`), a nested
/// repository (`src/x.ts`), or the filesystem. A relative citation that is not
/// a direct match is accepted when it equals the tail of entries that all
/// identify one scope entry, so a tail present under different declared paths
/// stays ambiguous. Absolute citations must match an absolute scope entry;
/// the verifier has no project root with which to resolve relative scopes.
/// Tails keep at least two components and start with a literal component, so
/// `src/**/*.rs` never degrades into `**/*.rs`.
struct ScopeMatcher {
    entries: Vec<ScopeEntryMatcher>,
}

struct ScopeEntryMatcher {
    declared_path: String,
    full: globset::GlobSet,
    tails: Option<globset::GlobSet>,
}

impl ScopeMatcher {
    fn new(scope: &str) -> Option<Self> {
        let entries = scope_entries(scope)
            .iter()
            .filter_map(|entry| ScopeEntryMatcher::new(entry))
            .collect::<Vec<_>>();
        (!entries.is_empty()).then_some(Self { entries })
    }

    fn contains(&self, path: &str) -> bool {
        let path = normalize_scope_path(path);
        if path.is_empty() {
            return false;
        }
        let is_absolute = is_absolute_scope_path(&path);
        if self.entries.iter().any(|entry| {
            is_absolute_scope_path(&entry.declared_path) == is_absolute
                && entry.full.is_match(&path)
        }) {
            return true;
        }

        if is_absolute {
            return false;
        }

        let mut matched_paths = self
            .entries
            .iter()
            .filter(|entry| {
                entry
                    .tails
                    .as_ref()
                    .is_some_and(|tails| tails.is_match(&path))
            })
            .map(|entry| entry.declared_path.as_str());
        matched_paths
            .next()
            .is_some_and(|path| matched_paths.all(|other| other == path))
    }
}

impl ScopeEntryMatcher {
    fn new(entry: &str) -> Option<Self> {
        let is_glob = is_glob_entry(entry);
        let components = entry.trim_start_matches('/').split('/').collect::<Vec<_>>();

        let full = Self::glob_set(std::iter::once(entry), is_glob)?;
        let literal_prefix_len = if is_glob {
            components
                .iter()
                .take_while(|component| is_literal_component(component))
                .count()
        } else {
            components.len()
        };
        let tail_patterns = (1..components.len())
            .filter(|start| {
                *start <= literal_prefix_len
                    && components.len() - start >= 2
                    && (!is_glob || is_literal_component(components[*start]))
            })
            .map(|start| components[start..].join("/"))
            .collect::<Vec<_>>();
        let tails = Self::glob_set(tail_patterns.iter().map(String::as_str), is_glob);

        Some(Self {
            declared_path: entry.to_string(),
            full,
            tails,
        })
    }

    /// Literal entries are escaped so path characters like `[id]` or
    /// `(group)` in framework route directories are not read as glob syntax,
    /// and they also match their descendants.
    fn glob_set<'a>(
        patterns: impl Iterator<Item = &'a str>,
        is_glob: bool,
    ) -> Option<globset::GlobSet> {
        let mut builder = GlobSetBuilder::new();
        let mut pattern_count = 0;
        for pattern in patterns {
            let variants = if is_glob {
                vec![pattern.to_string()]
            } else {
                let escaped = globset::escape(pattern);
                vec![escaped.clone(), format!("{escaped}/**")]
            };
            for variant in variants {
                let Ok(glob) = globset::Glob::new(&variant) else {
                    continue;
                };
                builder.add(glob);
                pattern_count += 1;
            }
        }
        if pattern_count == 0 {
            return None;
        }
        builder.build().ok()
    }
}

/// Extracts the path portion of a `path:line` citation.
fn citation_path(citation: &str) -> Option<&str> {
    citation
        .rsplit_once(':')
        .map(|(path, _)| path)
        .filter(|path| !path.trim().is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan_graph::OrchestrationTask;

    fn task_with_criteria(criteria: &[&str]) -> OrchestrationTask {
        let mut task = OrchestrationTask::new("task", "Task", "Do work");
        task.acceptance_criteria = criteria.iter().map(|c| c.to_string()).collect();
        task
    }

    fn claim_json(criteria: &str, citations: &str) -> String {
        format!(
            "Done\n{VERIFICATION_START}{{\"criteria\":[{criteria}],\"expected_output_satisfied\":true,\"citations\":[{citations}]}}{VERIFICATION_END}"
        )
    }

    #[test]
    fn empty_output_fails_as_fatal() {
        let runner = VerificationRunner;
        let result = runner.verify(&task_with_criteria(&[]), "   \n  ");
        assert!(!result.passed);
        assert_eq!(result.error_class, Some(ErrorClass::FatalError));
        assert!(!result.retryable);
    }

    #[test]
    fn model_fallback_only_handles_infrastructure_failures() {
        for message in [
            "429 rate limit exceeded",
            "connection timed out",
            "503 service unavailable",
            "provider is not configured",
            "401 Unauthorized: token_revoked",
            "Your session has ended: refresh_token_invalidated",
            "maximum context length exceeded",
        ] {
            assert!(VerificationPolicy::classify_error(message).is_model_fallback_eligible());
        }

        for message in [
            "tool execution failed",
            "verification assertion failed",
            "operation cancelled by user",
            "tool call budget exceeded",
        ] {
            assert!(!VerificationPolicy::classify_error(message).is_model_fallback_eligible());
        }
    }

    #[test]
    fn missing_claim_is_verification_failure() {
        let runner = VerificationRunner;
        let result = runner.verify(&task_with_criteria(&[]), "Just done.");
        assert!(!result.passed);
        assert_eq!(
            result.error_class,
            Some(ErrorClass::VerificationAssertionFailure)
        );
        assert!(result.retryable);
    }

    #[test]
    fn criteria_require_passing_claim_with_evidence() {
        let runner = VerificationRunner;
        let mut task = task_with_criteria(&["tests pass", "diff reviewed"]);
        task.expected_output = Some("Summary with evidence".to_string());
        task.evidence_required = true;

        let incomplete = claim_json(
            r#"{"criterion":"tests pass","passed":true,"evidence":"cargo test passed"}"#,
            r#""src/main.rs:12""#,
        );
        assert!(!runner.verify(&task, &incomplete).passed);

        let complete = claim_json(
            r#"{"criterion":"tests pass","passed":true,"evidence":"cargo test passed"},{"criterion":"diff reviewed","passed":true,"evidence":"reviewed src/main.rs"}"#,
            r#""src/main.rs:12""#,
        );
        let result = runner.verify(&task, &complete);
        assert!(result.passed);
        assert_eq!(result.verdict, VerificationVerdict::Claimed);
        assert!(result.criteria_verdicts.iter().all(|(_, passed)| *passed));
        assert_eq!(result.criterion_results.len(), 2);
        assert_eq!(
            result
                .criterion_results
                .first()
                .map(|criterion| criterion.evidence.as_str()),
            Some("cargo test passed")
        );
        assert_eq!(result.expected_output_satisfied, Some(true));
        assert_eq!(result.citations, ["src/main.rs:12"]);
    }

    #[test]
    fn invalid_citation_fails_evidence_requirement() {
        let runner = VerificationRunner;
        let mut task = task_with_criteria(&[]);
        task.evidence_required = true;
        let output = claim_json("", r#""src/main.rs""#);
        let result = runner.verify(&task, &output);
        assert!(!result.passed);
        assert_eq!(result.citations_valid, Some(false));
    }

    #[test]
    fn file_line_ranges_are_valid_citations() {
        assert!(is_file_line_citation("src/main.rs:12-24"));
        assert!(is_file_line_citation("src/main.rs:12"));
        assert!(!is_file_line_citation("src/main.rs:24-12"));
        assert!(!is_file_line_citation("src/main.rs:12-"));
    }

    #[test]
    fn web_research_accepts_source_urls_from_visible_output() {
        let runner = VerificationRunner;
        let mut task = task_with_criteria(&["include a sourced finding"]);
        task.evidence_required = true;
        task.expected_output = Some("A sourced report".to_string());
        let output = format!(
            "Finding: [Source](https://example.com/research).\n{VERIFICATION_START}{{\"criteria\":[{{\"criterion\":\"include a sourced finding\",\"passed\":true,\"evidence\":\"The report links the primary source\"}}],\"expected_output_satisfied\":true,\"citations\":[]}}{VERIFICATION_END}"
        );

        let result = runner.verify(&task, &output);

        assert!(result.passed);
        assert_eq!(result.citations, ["https://example.com/research"]);
        assert_eq!(result.citations_valid, Some(true));
    }

    #[test]
    fn escaped_verification_metadata_is_not_visible_citation_evidence() {
        let runner = VerificationRunner;
        let mut task = task_with_criteria(&[]);
        task.evidence_required = true;
        let output = "Unsourced finding.\n<zed\\_orchestration\\_verification>{\"criteria\":[],\"expected_output_satisfied\":true,\"citations\":[],\"note\":\"https://example.com/metadata-only\"}</zed\\_orchestration\\_verification>";

        let result = runner.verify(&task, output);

        assert!(!result.passed);
        assert_eq!(result.citations_valid, Some(false));
    }

    #[test]
    fn markdown_link_citations_are_normalized_for_display() {
        let runner = VerificationRunner;
        let mut task = task_with_criteria(&[]);
        task.evidence_required = true;
        let output = claim_json(
            "",
            r#""[https://example.com/pull/1](https://example.com/pull/1)""#,
        );

        let result = runner.verify(&task, &output);

        assert!(result.passed);
        assert_eq!(result.citations, ["https://example.com/pull/1"]);
        assert_eq!(result.citations_valid, Some(true));
    }

    #[test]
    fn repository_scope_still_requires_an_in_scope_file_citation() {
        let runner = VerificationRunner;
        let mut task = task_with_criteria(&[]);
        task.evidence_required = true;
        task.scope = Some("src/**".to_string());
        let output = claim_json("", r#""https://example.com/research""#);

        let result = runner.verify(&task, &output);

        assert!(!result.passed);
        assert!(
            result
                .feedback
                .as_deref()
                .is_some_and(|feedback| feedback.starts_with(SCOPE_FAILURE_FEEDBACK)
                    && feedback.contains("Declared scope paths: src/**"))
        );
    }

    #[test]
    fn scope_accepts_citations_relative_to_nested_roots() {
        let runner = VerificationRunner;
        let mut task = task_with_criteria(&[]);
        task.evidence_required = true;
        task.scope = Some(
            "Studio/labs/apps/web/src/lib/voice/voice-client.ts,Studio/labs/apps/web/src/lib/voice/voice-state.ts"
                .to_string(),
        );

        for citation in [
            "Studio/labs/apps/web/src/lib/voice/voice-state.ts:17",
            "./Studio/labs/apps/web/src/lib/voice/voice-state.ts:17",
            "labs/apps/web/src/lib/voice/voice-state.ts:17",
            "apps/web/src/lib/voice/voice-client.ts:12",
        ] {
            let output = claim_json("", &format!("\"{citation}\""));
            assert!(runner.verify(&task, &output).passed, "{citation}");
        }

        let mut absolute_task = task.clone();
        absolute_task.scope = Some("/Users/someone/Desktop/Studio/labs/src".to_string());
        let output = claim_json("", r#""/Users/someone/Desktop/Studio/labs/src/main.ts:4""#);
        assert!(runner.verify(&absolute_task, &output).passed);

        for citation in [
            "voice-client.ts:12",
            "apps/web/src/lib/voice/other.ts:1",
            "/Users/someone/Desktop/Other/apps/web/src/lib/other.ts:3",
            "/Users/someone/Desktop/Studio/labs/apps/web/src/lib/voice/voice-client.ts:3",
        ] {
            let output = claim_json("", &format!("\"{citation}\""));
            assert!(!runner.verify(&task, &output).passed, "{citation}");
        }
    }

    #[test]
    fn scope_tail_shared_by_two_roots_is_ambiguous() {
        let runner = VerificationRunner;
        let mut task = task_with_criteria(&[]);
        task.evidence_required = true;
        task.scope = Some("first/apps/web/main.ts,second/apps/web/main.ts".to_string());

        let ambiguous = claim_json("", r#""apps/web/main.ts:1""#);
        assert!(!runner.verify(&task, &ambiguous).passed);

        let anchored = claim_json("", r#""second/apps/web/main.ts:1""#);
        assert!(runner.verify(&task, &anchored).passed);
    }

    #[test]
    fn scope_rejects_absolute_citations_from_other_checkouts() {
        let runner = VerificationRunner;
        let mut task = task_with_criteria(&[]);
        task.evidence_required = true;
        for scope in [
            "Studio/labs/src/main.rs",
            "**/*.rs",
            "/Users/someone/Desktop/Studio/labs/src/main.rs",
        ] {
            task.scope = Some(scope.to_string());
            let output = claim_json("", r#""/tmp/another-checkout/Studio/labs/src/main.rs:10""#);
            assert!(!runner.verify(&task, &output).passed, "{scope}");
        }

        task.scope = Some("Studio/labs/src/main.rs".to_string());
        let output = claim_json("", r#""C:/another-checkout/Studio/labs/src/main.rs:10""#);
        assert!(!runner.verify(&task, &output).passed);

        task.scope = Some("C:/allowed/Studio/labs/src/main.rs".to_string());
        let output = claim_json("", r#""C:/allowed/Studio/labs/src/main.rs:10""#);
        assert!(runner.verify(&task, &output).passed);
    }

    #[test]
    fn scope_tail_shared_by_nested_repositories_is_ambiguous() {
        let runner = VerificationRunner;
        let mut task = task_with_criteria(&[]);
        task.evidence_required = true;
        task.scope =
            Some("Studio/first/apps/web/main.ts;Studio/second/apps/web/main.ts".to_string());
        let ambiguous = claim_json("", r#""apps/web/main.ts:1""#);
        assert!(!runner.verify(&task, &ambiguous).passed);
        let anchored = claim_json("", r#""second/apps/web/main.ts:1""#);
        assert!(runner.verify(&task, &anchored).passed);
    }

    #[test]
    fn literal_scope_paths_keep_route_brackets_and_groups() {
        let runner = VerificationRunner;
        let mut task = task_with_criteria(&[]);
        task.evidence_required = true;
        task.scope =
            Some("Studio/labs/apps/web/src/app/(studio)/projects/[projectId]/page.tsx".to_string());

        let output = claim_json(
            "",
            r#""Studio/labs/apps/web/src/app/(studio)/projects/[projectId]/page.tsx:20""#,
        );
        assert!(runner.verify(&task, &output).passed);
    }

    #[test]
    fn strips_only_valid_trailing_verification_claims() {
        let output = format!(
            "Useful result\n\n{VERIFICATION_START}\n{}\n{VERIFICATION_END}\n",
            r#"{"criteria":[],"expected_output_satisfied":true,"citations":[]}"#
        );
        assert_eq!(output_without_verification_claim(&output), "Useful result");

        let escaped = "Useful result\n<zed\\_orchestration\\_verification>{\"criteria\":[],\"expected_output_satisfied\":true,\"citations\":[]}</zed\\_orchestration\\_verification>";
        assert_eq!(output_without_verification_claim(escaped), "Useful result");

        let malformed = format!("Useful result\n{VERIFICATION_START}not json{VERIFICATION_END}");
        assert_eq!(output_without_verification_claim(&malformed), malformed);
        let followed_by_text = format!(
            "Useful result\n{VERIFICATION_START}{{\"criteria\":[]}}{VERIFICATION_END}\nkeep me"
        );
        assert_eq!(
            output_without_verification_claim(&followed_by_text),
            followed_by_text
        );
    }

    #[test]
    fn legacy_verification_result_deserializes_with_empty_details() {
        let legacy = serde_json::json!({
            "passed": true,
            "verdict": "claimed",
            "criteria_verdicts": [["criterion", true]],
            "verified_at": "2026-09-05T00:00:00Z",
            "retryable": false,
            "repairable": false
        });
        let result: VerificationResult = serde_json::from_value(legacy).unwrap();

        assert!(result.criterion_results.is_empty());
        assert!(result.citations.is_empty());
        assert_eq!(result.expected_output_satisfied, None);
    }

    #[test]
    fn persisted_verification_details_are_bounded() {
        let runner = VerificationRunner;
        let mut task = task_with_criteria(&["bounded evidence"]);
        task.evidence_required = true;
        let claim = VerificationClaim {
            criteria: vec![CriterionClaim {
                criterion: "bounded evidence".to_string(),
                passed: true,
                evidence: "e".repeat(MAX_PERSISTED_VERIFICATION_EVIDENCE_BYTES * 2),
            }],
            expected_output_satisfied: Some(true),
            citations: (1..=MAX_PERSISTED_VERIFICATION_CITATIONS + 10)
                .map(|line| format!("src/main.rs:{line}"))
                .collect(),
        };
        let output = format!(
            "Done\n{VERIFICATION_START}{}{VERIFICATION_END}",
            serde_json::to_string(&claim).expect("serialize verification claim")
        );
        let result = runner.verify(&task, &output);

        assert!(result.passed);
        assert_eq!(result.citations.len(), MAX_PERSISTED_VERIFICATION_CITATIONS);
        assert!(result.criterion_results.first().is_some_and(
            |criterion| criterion.evidence.len() <= MAX_PERSISTED_VERIFICATION_EVIDENCE_BYTES
        ));
    }

    #[test]
    fn scope_mismatch_rejects_out_of_bounds_citations() {
        let runner = VerificationRunner;
        let mut task = task_with_criteria(&[]);
        task.evidence_required = true;
        task.scope = Some("src/**/*.rs".to_string());

        let in_scope = claim_json("", r#""src/main.rs:12""#);
        assert!(runner.verify(&task, &in_scope).passed);

        let out_of_scope = claim_json("", r#""tests/foo.rs:1""#);
        let result = runner.verify(&task, &out_of_scope);
        assert!(!result.passed);
        assert!(
            result
                .feedback
                .as_deref()
                .is_some_and(|f| f.contains("scope"))
        );
    }

    #[test]
    fn semicolon_separated_scopes_accept_each_declared_path() {
        let runner = VerificationRunner;
        let mut task = task_with_criteria(&[]);
        task.evidence_required = true;
        task.scope = Some("internal/services/host_runtime_service.go;internal/api".to_string());

        let output = claim_json("", r#""internal/services/host_runtime_service.go:42""#);
        assert!(runner.verify(&task, &output).passed);
    }

    #[test]
    fn scope_ignored_without_citations() {
        let runner = VerificationRunner;
        let mut task = task_with_criteria(&[]);
        task.scope = Some("src/**/*.rs".to_string());
        let output = claim_json("", "");
        assert!(runner.verify(&task, &output).passed);
    }

    #[test]
    fn directory_scope_accepts_descendants_and_supporting_citations() {
        let runner = VerificationRunner;
        let mut task = task_with_criteria(&[]);
        task.evidence_required = true;
        task.scope = Some(
            "vhmap-portal-web/src/modules/app-runtime and related dependency metadata".to_string(),
        );
        let output = claim_json(
            "",
            r#""vhmap-portal-web/src/modules/app-runtime/components/AppLoader.tsx:119-128","vhmap-portal-web/docs/architecture.md:220-249""#,
        );

        assert!(runner.verify(&task, &output).passed);
    }

    #[test]
    fn directory_scope_still_requires_primary_scope_evidence() {
        let runner = VerificationRunner;
        let mut task = task_with_criteria(&[]);
        task.evidence_required = true;
        task.scope = Some("vhmap-portal-web/src/modules/app-runtime".to_string());
        let output = claim_json("", r#""vhmap-portal-web/docs/architecture.md:220-249""#);

        let result = runner.verify(&task, &output);
        assert!(!result.passed);
        assert!(
            result
                .feedback
                .as_deref()
                .is_some_and(|feedback| feedback.contains("scope"))
        );
    }

    #[test]
    fn structured_failure_preserves_scope_and_retry_after() {
        let failure = TaskExecutionFailure::new(
            ErrorClass::RateLimit,
            FailureScope::Account,
            "account is temporarily rate limited",
        )
        .with_retry_after(Some(Duration::from_millis(2_500)));

        assert_eq!(failure.class, ErrorClass::RateLimit);
        assert_eq!(failure.scope, FailureScope::Account);
        assert_eq!(failure.retry_after(), Some(Duration::from_millis(2_500)));
        assert_eq!(failure.to_string(), "account is temporarily rate limited");
    }
}
