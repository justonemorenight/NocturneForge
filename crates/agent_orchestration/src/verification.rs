use chrono::{DateTime, Utc};
use globset::GlobSetBuilder;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
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
    /// Tool returned an error that may be fixed with retry or different inputs.
    ToolExecutionError,
    /// Token budget hard limit was exceeded.
    TokenBudgetExceeded,
    /// Verification checks or acceptance criteria failed.
    VerificationAssertionFailure,
    /// Critical unrecoverable error (e.g. malformed model output, internal bug).
    FatalError,
    /// Operation was explicitly cancelled.
    Cancelled,
}

impl ErrorClass {
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
        if lower.contains("rate limit")
            || lower.contains("429")
            || lower.contains("quota exceeded")
            || lower.contains("too many requests")
        {
            ErrorClass::RateLimit
        } else if (lower.contains("budget") && lower.contains("exceeded"))
            || lower.contains("token budget")
        {
            ErrorClass::TokenBudgetExceeded
        } else if lower.contains("cancelled") || lower.contains("canceled") {
            ErrorClass::Cancelled
        } else if lower.contains("connection")
            || lower.contains("timeout")
            || lower.contains("timed out")
            || lower.contains("reset by peer")
            || lower.contains("broken pipe")
            || lower.contains("dns")
        {
            ErrorClass::TransientNetwork
        } else if lower.contains("verification")
            || lower.contains("assertion")
            || lower.contains("criteria")
        {
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
/// Marker used by tasks to embed a structured verification claim in their output.
pub const VERIFICATION_START: &str = "<zed_orchestration_verification>";
/// Closing marker for a structured verification claim.
pub const VERIFICATION_END: &str = "</zed_orchestration_verification>";
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

/// Parses the structured verification claim embedded at the end of a task output.
pub fn parse_verification_claim(output: &str) -> anyhow::Result<VerificationClaim> {
    let start = output
        .rfind(VERIFICATION_START)
        .map(|index| index + VERIFICATION_START.len())
        .ok_or_else(|| anyhow::anyhow!("missing structured verification claim"))?;
    let end = output[start..]
        .find(VERIFICATION_END)
        .map(|index| start + index)
        .ok_or_else(|| anyhow::anyhow!("unterminated structured verification claim"))?;
    serde_json::from_str(output[start..end].trim())
        .map_err(|error| anyhow::anyhow!("invalid structured verification claim: {error}"))
}

/// Removes a valid trailing structured verification envelope from output shown
/// to the parent model or user. The unmodified output remains in runtime state
/// and artifacts for auditing and replay.
pub fn output_without_verification_claim(output: &str) -> &str {
    let Some(start) = output.rfind(VERIFICATION_START) else {
        return output;
    };
    let claim = &output[start..];
    let Some(end) = claim.find(VERIFICATION_END) else {
        return output;
    };
    let trailing = &claim[end + VERIFICATION_END.len()..];
    if !trailing.trim().is_empty() || parse_verification_claim(output).is_err() {
        return output;
    }
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

/// Runs native verification checks against a task's output.
///
/// Checks are structural: every acceptance criterion must carry a passing claim
/// with concrete evidence, the expected output contract must be satisfied, and
/// citations (when evidence is required) must be file-and-line references that
/// fall within the task's declared scope.
#[derive(Debug, Default)]
pub struct VerificationRunner;

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
        let criteria_verdicts = task
            .acceptance_criteria
            .iter()
            .map(|criterion| {
                let passed = claim.criteria.iter().any(|result| {
                    result.criterion == *criterion
                        && result.passed
                        && !result.evidence.trim().is_empty()
                });
                (criterion.clone(), passed)
            })
            .collect::<Vec<_>>();
        let criterion_results = task
            .acceptance_criteria
            .iter()
            .map(|criterion| {
                let claim = claim
                    .criteria
                    .iter()
                    .find(|result| result.criterion == *criterion);
                CriterionClaim {
                    criterion: criterion.clone(),
                    passed: claim
                        .is_some_and(|result| result.passed && !result.evidence.trim().is_empty()),
                    evidence: claim
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
        let persisted_citations = claim
            .citations
            .iter()
            .take(MAX_PERSISTED_VERIFICATION_CITATIONS)
            .cloned()
            .map(|citation| truncate_text(citation, MAX_PERSISTED_VERIFICATION_CITATION_BYTES))
            .collect::<Vec<_>>();
        let criteria_valid = criteria_verdicts.iter().all(|(_, passed)| *passed);
        let expected_output_valid =
            task.expected_output.is_none() || claim.expected_output_satisfied == Some(true);
        let citations_valid = task.evidence_required.then(|| {
            !claim.citations.is_empty()
                && claim
                    .citations
                    .iter()
                    .all(|citation| is_file_line_citation(citation))
        });
        let scope_valid = self.scope_citations_valid(task, &claim);

        if !criteria_valid
            || !expected_output_valid
            || citations_valid == Some(false)
            || !scope_valid
        {
            let feedback = if !criteria_valid {
                "One or more acceptance criteria lack a passing claim with concrete evidence"
            } else if !expected_output_valid {
                "The expected output contract was not satisfied"
            } else if citations_valid == Some(false) {
                "Evidence required: provide valid file-and-line citations such as path/to/file.rs:123"
            } else {
                "Citations fall outside the task's declared scope"
            };
            return VerificationResult {
                passed: false,
                verdict: VerificationVerdict::FailedVerification,
                criteria_verdicts,
                criterion_results,
                expected_output_satisfied: claim.expected_output_satisfied,
                citations: persisted_citations,
                citations_valid,
                feedback: Some(feedback.into()),
                error_class: Some(ErrorClass::VerificationAssertionFailure),
                retryable: true,
                repairable: true,
                verified_at: Utc::now(),
            };
        }

        VerificationResult {
            passed: true,
            verdict: VerificationVerdict::Claimed,
            criteria_verdicts,
            criterion_results,
            expected_output_satisfied: claim.expected_output_satisfied,
            citations: persisted_citations,
            citations_valid,
            feedback: None,
            error_class: None,
            retryable: false,
            repairable: false,
            verified_at: Utc::now(),
        }
    }

    /// Checks that the task has primary evidence within its declared scope.
    ///
    /// Directory scopes include their descendants. At least one citation must
    /// fall within the primary scope; additional citations may support the
    /// result from adjacent documentation or dependency metadata.
    fn scope_citations_valid(&self, task: &OrchestrationTask, claim: &VerificationClaim) -> bool {
        let Some(scope) = task.scope.as_deref() else {
            return true;
        };
        if claim.citations.is_empty() {
            return true;
        }
        let mut builder = GlobSetBuilder::new();
        let mut pattern_count = 0;
        for pattern in scope
            .split([',', '\n'])
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
            .map(|pattern| {
                pattern.trim_matches(|character: char| {
                    character.is_ascii_punctuation()
                        && !matches!(character, '/' | '*' | '?' | '[' | ']' | '.' | '-' | '_')
                })
            })
            .filter(|pattern| !pattern.is_empty())
        {
            let patterns = if pattern.contains(['*', '?', '[', ']']) {
                vec![pattern.to_string()]
            } else {
                let normalized = pattern.trim_end_matches('/');
                vec![normalized.to_string(), format!("{normalized}/**")]
            };
            for pattern in patterns {
                let Ok(glob) = globset::Glob::new(&pattern) else {
                    continue;
                };
                builder.add(glob);
                pattern_count += 1;
            }
        }
        if pattern_count == 0 {
            return false;
        }
        let Ok(glob_set) = builder.build() else {
            return false;
        };
        claim
            .citations
            .iter()
            .any(|citation| citation_path(citation).is_some_and(|path| glob_set.is_match(path)))
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
    fn strips_only_valid_trailing_verification_claims() {
        let output = format!(
            "Useful result\n\n{VERIFICATION_START}\n{}\n{VERIFICATION_END}\n",
            r#"{"criteria":[],"expected_output_satisfied":true,"citations":[]}"#
        );
        assert_eq!(output_without_verification_claim(&output), "Useful result");

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
}
