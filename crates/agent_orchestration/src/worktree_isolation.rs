use crate::ids::{RunId, TaskId};
use anyhow::{Context as _, Result, bail};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use smol::process::Command;
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use futures::io::AsyncReadExt;

struct SnapshotIndex(PathBuf);

impl Drop for SnapshotIndex {
    fn drop(&mut self) {
        match fs::remove_file(&self.0) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => log::warn!("failed to remove temporary worker index: {error}"),
        }
    }
}

/// Ownership marker stored inside each managed worktree to prevent accidental cleanup of user files.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorktreeOwnershipMarker {
    pub marker_version: u32,
    pub run_id: RunId,
    pub task_id: TaskId,
    pub attempt: u32,
    pub created_at: DateTime<Utc>,
    pub baseline_commit: String,
    pub parent_checkout: PathBuf,
    pub branch_name: String,
}

/// Snapshot captured immediately before applying changes into the parent checkout,
/// enabling safe rollback if post-apply verification fails.
#[derive(Clone, Debug)]
pub struct ParentPreApplySnapshot {
    pub baseline_commit: String,
    pub parent_repo: PathBuf,
    pub modified_files: Vec<String>,
    pub pre_existing_paths: HashSet<String>,
    index_path: PathBuf,
    index_contents: Option<Vec<u8>>,
}

impl ParentPreApplySnapshot {
    /// Reverts the parent checkout to its exact pre-apply state:
    /// - Files that existed before apply are restored from `baseline_commit`.
    /// - Files newly introduced by the patch are removed.
    /// - The parent git index is restored byte-for-byte, preserving staged work.
    pub async fn rollback(&self) -> Result<()> {
        if self.modified_files.is_empty() {
            return Ok(());
        }

        for file in &self.modified_files {
            let path = self.parent_repo.join(file);
            if self.pre_existing_paths.contains(file) {
                let output = Command::new("git")
                    .arg("-C")
                    .arg(&self.parent_repo)
                    .args(["checkout", &self.baseline_commit, "--", file])
                    .output()
                    .await
                    .context("failed to restore pre-existing file from snapshot")?;
                if !output.status.success() {
                    bail!(
                        "git checkout from snapshot failed for {}: {}",
                        file,
                        String::from_utf8_lossy(&output.stderr).trim()
                    );
                }
            } else if let Ok(metadata) = fs::symlink_metadata(&path) {
                if metadata.file_type().is_dir() {
                    std::fs::remove_dir_all(&path).with_context(|| {
                        format!("failed to remove newly-created directory {:?}", path)
                    })?;
                } else {
                    std::fs::remove_file(&path).with_context(|| {
                        format!("failed to remove newly-created file {:?}", path)
                    })?;
                }
            }
        }

        if let Some(index_contents) = &self.index_contents {
            let restore_path = self
                .index_path
                .with_extension(format!("restore-{}", uuid::Uuid::new_v4().simple()));
            fs::write(&restore_path, index_contents).with_context(|| {
                format!("failed to stage git index snapshot at {:?}", restore_path)
            })?;
            if let Err(error) = fs::rename(&restore_path, &self.index_path) {
                if let Err(cleanup_error) = fs::remove_file(&restore_path) {
                    log::warn!(
                        "failed to clean up temporary git index {:?}: {cleanup_error}",
                        restore_path
                    );
                }
                return Err(error).with_context(|| {
                    format!("failed to restore git index at {:?}", self.index_path)
                });
            }
        } else if fs::symlink_metadata(&self.index_path).is_ok() {
            fs::remove_file(&self.index_path)
                .with_context(|| format!("failed to remove git index {:?}", self.index_path))?;
        }

        Ok(())
    }
}

async fn git_index_path(repo: &Path) -> Result<PathBuf> {
    let output = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(["rev-parse", "--git-path", "index"])
        .output()
        .await
        .context("failed to resolve git index path")?;
    if !output.status.success() {
        bail!(
            "failed to resolve git index path for {:?}: {}",
            repo,
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    let index_path = PathBuf::from(
        String::from_utf8(output.stdout)
            .context("git index path is not valid UTF-8")?
            .trim(),
    );
    if index_path.is_absolute() {
        Ok(index_path)
    } else {
        Ok(repo.join(index_path))
    }
}

const VERIFICATION_TIMEOUT: Duration = Duration::from_secs(5 * 60);
const MAX_VERIFICATION_OUTPUT_BYTES: usize = 64 * 1024;

async fn read_bounded<R>(reader: Option<R>) -> std::io::Result<Vec<u8>>
where
    R: futures::io::AsyncRead + Unpin,
{
    let Some(mut reader) = reader else {
        return Ok(Vec::new());
    };
    let mut output = Vec::new();
    let mut buffer = [0_u8; 8 * 1024];
    loop {
        let read = reader.read(&mut buffer).await?;
        if read == 0 {
            break;
        }
        let remaining = MAX_VERIFICATION_OUTPUT_BYTES.saturating_sub(output.len());
        if remaining > 0 {
            output.extend_from_slice(&buffer[..read.min(remaining)]);
        }
    }
    Ok(output)
}

fn parse_verification_command(command: &str) -> Result<(String, Vec<String>)> {
    let trimmed = command.trim();
    if trimmed.is_empty()
        || trimmed.chars().any(|character| {
            matches!(
                character,
                '\n' | '\r' | ';' | '&' | '|' | '$' | '`' | '<' | '>'
            )
        })
    {
        bail!("verification command contains an empty value or shell control syntax");
    }
    let mut parts = trimmed.split_whitespace();
    let program = parts
        .next()
        .ok_or_else(|| anyhow::anyhow!("verification command is empty"))?;
    if program.contains('/') {
        bail!("verification command must use an allowlisted executable name");
    }
    let allowed_programs = [
        "bun", "cargo", "false", "git", "go", "just", "make", "npm", "pnpm", "pytest", "python",
        "python3", "test", "true", "yarn",
    ];
    if !allowed_programs.contains(&program) {
        bail!("verification executable '{program}' is not allowlisted");
    }
    Ok((
        program.to_string(),
        parts.map(ToString::to_string).collect(),
    ))
}

async fn run_verification_command(
    command: &str,
    current_dir: &Path,
) -> Result<(bool, Option<i32>, String, String)> {
    let (program, arguments) = parse_verification_command(command)?;
    let mut child = Command::new(program)
        .args(arguments)
        .current_dir(current_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .context("failed to spawn verification command")?;
    let status = child.status();
    let stdout = read_bounded(child.stdout.take());
    let stderr = read_bounded(child.stderr.take());
    let process = async { futures::future::join3(status, stdout, stderr).await };
    futures::pin_mut!(process);
    // This timeout guards an external process and has no GPUI executor to
    // attach to; the child is kill-on-drop when the timeout wins.
    #[allow(clippy::disallowed_methods)]
    let timeout = smol::Timer::after(VERIFICATION_TIMEOUT);
    futures::pin_mut!(timeout);
    let (status, stdout, stderr) = match futures::future::select(process, timeout).await {
        futures::future::Either::Left((result, _)) => result,
        futures::future::Either::Right(_) => {
            bail!("verification command exceeded {:?}", VERIFICATION_TIMEOUT)
        }
    };
    let status = status.context("verification process failed while waiting for exit")?;
    let exit_code = status.code();
    let stdout = String::from_utf8_lossy(&stdout?).into_owned();
    let stderr = String::from_utf8_lossy(&stderr?).into_owned();
    Ok((status.success(), exit_code, stdout, stderr))
}

impl WorktreeOwnershipMarker {
    pub const FILENAME: &'static str = ".zed_worktree_ownership.json";

    pub fn write_to(&self, worktree_dir: &Path) -> Result<()> {
        let marker_path = worktree_dir.join(Self::FILENAME);
        let json = serde_json::to_string_pretty(self)?;
        fs::write(&marker_path, json)
            .with_context(|| format!("failed to write ownership marker at {:?}", marker_path))?;
        Ok(())
    }

    pub fn read_from(worktree_dir: &Path) -> Result<Self> {
        let marker_path = worktree_dir.join(Self::FILENAME);
        if !marker_path.exists() {
            bail!("missing ownership marker at {:?}", marker_path);
        }
        let content = fs::read_to_string(&marker_path)
            .with_context(|| format!("failed to read ownership marker at {:?}", marker_path))?;
        let marker: Self = serde_json::from_str(&content)
            .with_context(|| format!("invalid ownership marker at {:?}", marker_path))?;
        if marker.marker_version != 2 {
            bail!("unsupported worktree ownership marker version");
        }
        Ok(marker)
    }
}

/// A git worktree dedicated to executing a single write task attempt in isolation.
#[derive(Clone, Debug)]
pub struct IsolatedWorktree {
    pub repo_path: PathBuf,
    pub worktree_path: PathBuf,
    pub branch_name: String,
    pub baseline_commit: String,
    pub run_id: RunId,
    pub task_id: TaskId,
    pub attempt: u32,
    pub created_at: DateTime<Utc>,
}

impl IsolatedWorktree {
    async fn include_untracked_files(&self) -> Result<()> {
        let output = Command::new("git")
            .arg("-C")
            .arg(&self.worktree_path)
            .args(["add", "--intent-to-add", "--all", "--", "."])
            .output()
            .await
            .context("failed to include untracked files in isolated worktree diff")?;
        if !output.status.success() {
            bail!(
                "git add --intent-to-add failed in {:?}: {}",
                self.worktree_path,
                String::from_utf8_lossy(&output.stderr).trim()
            );
        }
        Ok(())
    }

    pub async fn reopen_managed(
        worktree_path: PathBuf,
        run_id: &RunId,
        task_id: &TaskId,
        attempt: u32,
    ) -> Result<Self> {
        let marker = WorktreeOwnershipMarker::read_from(&worktree_path)?;
        if marker.run_id != *run_id || marker.task_id != *task_id || marker.attempt != attempt {
            bail!("managed worktree ownership does not match the requested task attempt");
        }
        let repo_path = fs::canonicalize(&marker.parent_checkout)
            .context("managed worktree parent checkout is unavailable")?;
        if Self::common_directory(&repo_path).await?
            != Self::common_directory(&worktree_path).await?
        {
            bail!("managed worktree and parent checkout belong to different repositories");
        }
        let branch_output = Command::new("git")
            .arg("-C")
            .arg(&worktree_path)
            .args(["branch", "--show-current"])
            .output()
            .await
            .context("failed to resolve managed worktree branch")?;
        if !branch_output.status.success() {
            bail!(
                "git branch failed for {:?}: {}",
                worktree_path,
                String::from_utf8_lossy(&branch_output.stderr).trim()
            );
        }
        let branch_name = String::from_utf8(branch_output.stdout)
            .context("invalid git branch output")?
            .trim()
            .to_string();
        if branch_name != marker.branch_name {
            bail!("managed worktree branch changed; refusing to apply or clean up");
        }
        Ok(Self {
            repo_path,
            worktree_path,
            branch_name,
            baseline_commit: marker.baseline_commit,
            run_id: marker.run_id,
            task_id: marker.task_id,
            attempt: marker.attempt,
            created_at: marker.created_at,
        })
    }

    /// Transfers an already verified worktree lease to the next retry attempt.
    /// This preserves partial worker changes across a process reconnect while
    /// making stale callbacks from the preceding attempt fail marker checks.
    pub async fn advance_attempt(self, next_attempt: u32) -> Result<Self> {
        anyhow::ensure!(
            next_attempt > self.attempt,
            "managed worktree attempt must advance"
        );
        let current = Self::reopen_managed(
            self.worktree_path.clone(),
            &self.run_id,
            &self.task_id,
            self.attempt,
        )
        .await?;
        let marker = WorktreeOwnershipMarker {
            marker_version: 2,
            run_id: current.run_id.clone(),
            task_id: current.task_id.clone(),
            attempt: next_attempt,
            created_at: current.created_at,
            baseline_commit: current.baseline_commit.clone(),
            parent_checkout: current.repo_path.clone(),
            branch_name: current.branch_name.clone(),
        };
        marker.write_to(&current.worktree_path)?;
        Ok(Self {
            attempt: next_attempt,
            ..current
        })
    }

    async fn common_directory(checkout: &Path) -> Result<PathBuf> {
        let output = Command::new("git")
            .arg("-C")
            .arg(checkout)
            .args(["rev-parse", "--path-format=absolute", "--git-common-dir"])
            .output()
            .await?;
        if !output.status.success() {
            bail!("failed to resolve repository identity for {:?}", checkout);
        }
        let directory = String::from_utf8(output.stdout)?;
        fs::canonicalize(directory.trim()).context("repository directory is unavailable")
    }

    /// Collects unified git diff of all changes made in this worktree against the baseline commit.
    pub async fn collect_diff(&self) -> Result<String> {
        if !self.worktree_path.exists() {
            bail!("worktree path does not exist: {:?}", self.worktree_path);
        }

        self.include_untracked_files().await?;

        let output = Command::new("git")
            .arg("-C")
            .arg(&self.worktree_path)
            .arg("diff")
            .arg("--binary")
            .arg("--no-ext-diff")
            .arg(&self.baseline_commit)
            .arg("--")
            .arg(".")
            .arg(format!(":(exclude){}", WorktreeOwnershipMarker::FILENAME))
            .output()
            .await
            .context("failed to execute git diff in isolated worktree")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            bail!("git diff failed in {:?}: {}", self.worktree_path, stderr);
        }

        let diff = String::from_utf8(output.stdout).context("git diff returned invalid UTF-8")?;
        Ok(diff)
    }

    /// Validates that changes in the worktree do not escape allowed subpaths
    /// and do not create symlinks traversing outside the worktree root.
    pub async fn validate_scope(&self, allowed_subpaths: Option<&[String]>) -> Result<Vec<String>> {
        let canonical_root = fs::canonicalize(&self.worktree_path)
            .with_context(|| format!("failed to canonicalize {:?}", self.worktree_path))?;

        self.include_untracked_files().await?;

        // Disable rename detection so both sides of a move are validated.
        let output = Command::new("git")
            .arg("-C")
            .arg(&self.worktree_path)
            .args(["diff", "--name-only", "--no-renames", "-z"])
            .arg(&self.baseline_commit)
            .args(["--", "."])
            .output()
            .await
            .context("failed to enumerate changes in isolated worktree")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            bail!("git diff failed in {:?}: {}", self.worktree_path, stderr);
        }

        let mut modified_files = Vec::new();
        for entry in output.stdout.split(|&b| b == 0) {
            if entry.is_empty() {
                continue;
            }
            let path_str =
                String::from_utf8(entry.to_vec()).context("changed path is not valid UTF-8")?;
            if path_str == WorktreeOwnershipMarker::FILENAME {
                continue;
            }

            let relative_path = Path::new(&path_str);
            if relative_path.is_absolute()
                || relative_path.components().any(|component| {
                    matches!(
                        component,
                        std::path::Component::ParentDir
                            | std::path::Component::RootDir
                            | std::path::Component::Prefix(_)
                    )
                })
            {
                bail!("security violation: invalid changed path '{}'", path_str);
            }
            let full_path = self.worktree_path.join(relative_path);

            // Check symlink path traversal
            if let Ok(symlink_target) = fs::read_link(&full_path) {
                let resolved_target = if symlink_target.is_absolute() {
                    symlink_target
                } else {
                    full_path
                        .parent()
                        .unwrap_or(&self.worktree_path)
                        .join(&symlink_target)
                };

                if let Ok(canonical_target) = fs::canonicalize(&resolved_target) {
                    if !canonical_target.starts_with(&canonical_root) {
                        bail!(
                            "security violation: symlink '{}' resolves outside worktree root to '{:?}'",
                            path_str,
                            canonical_target
                        );
                    }
                } else if !lexically_normalize(&resolved_target).starts_with(&canonical_root) {
                    bail!(
                        "security violation: symlink '{}' points outside worktree root to '{:?}'",
                        path_str,
                        resolved_target
                    );
                }
            }

            // Check allowed subpaths if specified
            if let Some(allowed) = allowed_subpaths {
                let matches_allowed = allowed.iter().any(|allowed_prefix| {
                    let normalized_prefix = Path::new(allowed_prefix.trim_start_matches('/'));
                    !normalized_prefix.as_os_str().is_empty()
                        && !normalized_prefix.components().any(|component| {
                            matches!(
                                component,
                                std::path::Component::ParentDir
                                    | std::path::Component::RootDir
                                    | std::path::Component::Prefix(_)
                            )
                        })
                        && relative_path.starts_with(normalized_prefix)
                });
                if !matches_allowed {
                    bail!(
                        "scope violation: modified file '{}' is outside allowed subpaths: {:?}",
                        path_str,
                        allowed
                    );
                }
            }

            modified_files.push(path_str);
        }

        Ok(modified_files)
    }

    /// Verifies if changes can be applied cleanly into the parent repository without conflicts.
    pub async fn can_apply_cleanly(&self, parent_repo: &Path) -> Result<bool> {
        let diff = self.collect_diff().await?;
        if diff.trim().is_empty() {
            return Ok(true);
        }

        let mut child = Command::new("git")
            .arg("-C")
            .arg(parent_repo)
            .arg("apply")
            .arg("--check")
            .stdin(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context("failed to spawn git apply --check")?;

        if let Some(mut stdin) = child.stdin.take() {
            use futures::io::AsyncWriteExt as _;
            stdin.write_all(diff.as_bytes()).await?;
        }

        let output = child.output().await?;
        Ok(output.status.success())
    }

    /// Safely applies the changes into the parent repository:
    /// - Verifies conflicts first using `git apply --check`.
    /// - Fails with conflict error without modifying parent files if not clean.
    /// - Applies the changes cleanly if verified.
    pub async fn apply_to_parent(
        &self,
        parent_repo: &Path,
        allowed_subpaths: Option<&[String]>,
    ) -> Result<Vec<String>> {
        if fs::canonicalize(parent_repo)? != fs::canonicalize(&self.repo_path)? {
            bail!("refusing to apply to a different parent checkout");
        }
        let modified_files = self.validate_scope(allowed_subpaths).await?;
        let diff = self.collect_diff().await?;
        if diff.trim().is_empty() {
            return Ok(Vec::new());
        }
        let diff = self
            .merge_patch_for_parent(parent_repo, &modified_files, &diff)
            .await?;

        let mut check_child = Command::new("git")
            .arg("-C")
            .arg(parent_repo)
            .arg("apply")
            .arg("--check")
            .stdin(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context("failed to spawn git apply --check")?;

        if let Some(mut stdin) = check_child.stdin.take() {
            use futures::io::AsyncWriteExt as _;
            stdin.write_all(diff.as_bytes()).await?;
        }

        let check_output = check_child.output().await?;
        if !check_output.status.success() {
            let stderr = String::from_utf8_lossy(&check_output.stderr);
            bail!(
                "cannot apply changes to parent checkout due to conflicts: {}",
                stderr.trim()
            );
        }

        let mut apply_child = Command::new("git")
            .arg("-C")
            .arg(parent_repo)
            .arg("apply")
            .stdin(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context("failed to spawn git apply")?;

        if let Some(mut stdin) = apply_child.stdin.take() {
            use futures::io::AsyncWriteExt as _;
            stdin.write_all(diff.as_bytes()).await?;
        }

        let apply_output = apply_child.output().await?;
        if !apply_output.status.success() {
            let stderr = String::from_utf8_lossy(&apply_output.stderr);
            bail!("git apply failed on parent checkout: {}", stderr.trim());
        }

        Ok(modified_files)
    }

    async fn merge_patch_for_parent(
        &self,
        parent: &Path,
        paths: &[String],
        diff: &str,
    ) -> Result<String> {
        let manager = WorktreeManager::new(parent, std::env::temp_dir());
        let head = manager.verify_local_git_repo().await?;
        let snapshot = manager.capture_baseline(&head, Some(paths)).await?;
        let index =
            std::env::temp_dir().join(format!("nocturne-merge-index-{}", uuid::Uuid::new_v4()));
        let _index_cleanup = SnapshotIndex(index.clone());
        manager
            .snapshot_git(&index, &["read-tree", &snapshot])
            .await?;
        let mut child = Command::new("git")
            .arg("-C")
            .arg(parent)
            .args(["apply", "--cached", "--3way"])
            .env("GIT_INDEX_FILE", &index)
            .stdin(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        if let Some(mut stdin) = child.stdin.take() {
            use futures::io::AsyncWriteExt as _;
            stdin.write_all(diff.as_bytes()).await?;
        }
        let output = child.output().await?;
        anyhow::ensure!(
            output.status.success(),
            "worker patch conflicts with parent changes; parent checkout was not modified: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let merged_tree = manager.snapshot_git(&index, &["write-tree"]).await?;
        manager
            .snapshot_git(
                &index,
                &[
                    "diff",
                    "--binary",
                    "--no-ext-diff",
                    &snapshot,
                    merged_tree.trim(),
                    "--",
                ],
            )
            .await
    }

    /// Cleans up this worktree, verifying the ownership marker first.
    pub async fn cleanup(&self, force: bool) -> Result<()> {
        if !self.worktree_path.exists() {
            return Ok(());
        }
        let reopened = Self::reopen_managed(
            self.worktree_path.clone(),
            &self.run_id,
            &self.task_id,
            self.attempt,
        )
        .await?;
        if reopened.branch_name != self.branch_name
            || reopened.repo_path != fs::canonicalize(&self.repo_path)?
        {
            bail!("managed worktree identity changed before cleanup");
        }

        // Verify ownership marker before deleting
        let marker = WorktreeOwnershipMarker::read_from(&self.worktree_path)
            .context("refusing to clean up worktree without valid ownership marker")?;

        if marker.run_id != self.run_id
            || marker.task_id != self.task_id
            || marker.attempt != self.attempt
        {
            bail!(
                "ownership marker mismatch: expected run_id='{}', task_id='{}', attempt={}; found run_id='{}', task_id='{}', attempt={}",
                self.run_id,
                self.task_id,
                self.attempt,
                marker.run_id,
                marker.task_id,
                marker.attempt
            );
        }

        // Remove worktree via git command
        let mut cmd = Command::new("git");
        cmd.arg("-C")
            .arg(&self.repo_path)
            .arg("worktree")
            .arg("remove");
        if force {
            cmd.arg("--force");
        }
        cmd.arg(&self.worktree_path);

        let output = cmd
            .output()
            .await
            .context("failed to remove git worktree")?;
        if !output.status.success() {
            bail!(
                "git worktree remove failed for {:?}: {}",
                self.worktree_path,
                String::from_utf8_lossy(&output.stderr).trim()
            );
        }

        let branch_output = Command::new("git")
            .arg("-C")
            .arg(&self.repo_path)
            .arg("branch")
            .arg("-D")
            .arg(&self.branch_name)
            .output()
            .await
            .context("failed to delete isolated worktree branch")?;
        if !branch_output.status.success() {
            log::warn!(
                "failed to delete isolated worktree branch '{}': {}",
                self.branch_name,
                String::from_utf8_lossy(&branch_output.stderr).trim()
            );
        }

        Ok(())
    }

    /// Captures a pre-apply snapshot of the parent repository for all files that will be modified.
    pub async fn prepare_parent_apply(
        &self,
        parent_repo: &Path,
        allowed_subpaths: Option<&[String]>,
    ) -> Result<ParentPreApplySnapshot> {
        if fs::canonicalize(parent_repo)? != fs::canonicalize(&self.repo_path)? {
            bail!("refusing to apply to a different parent checkout");
        }
        let modified_files = self.validate_scope(allowed_subpaths).await?;
        let canonical_parent = fs::canonicalize(parent_repo)?;

        let mut pre_existing_paths = HashSet::new();
        for file in &modified_files {
            let path = canonical_parent.join(file);
            if path.exists() || fs::symlink_metadata(&path).is_ok() {
                pre_existing_paths.insert(file.clone());
            }
        }

        let manager = WorktreeManager::new(parent_repo, std::env::temp_dir());
        let head = manager.verify_local_git_repo().await?;
        let baseline_commit = manager
            .capture_baseline(&head, Some(&modified_files))
            .await?;
        let index_path = git_index_path(&canonical_parent).await?;
        let index_contents = match fs::read(&index_path) {
            Ok(contents) => Some(contents),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("failed to read git index at {:?}", index_path));
            }
        };

        Ok(ParentPreApplySnapshot {
            baseline_commit,
            parent_repo: canonical_parent,
            modified_files,
            pre_existing_paths,
            index_path,
            index_contents,
        })
    }

    /// Verifies changes inside this isolated worktree before applying to parent.
    /// Runs scope validation and optional verification command.
    pub async fn verify_worktree(
        &self,
        command: Option<&str>,
        allowed_subpaths: Option<&[String]>,
    ) -> Result<()> {
        self.validate_scope(allowed_subpaths).await?;

        if let Some(cmd) = command {
            let (passed, exit_code, stdout, stderr) =
                run_verification_command(cmd, &self.worktree_path).await?;
            if !passed {
                bail!(
                    "worktree verification command '{cmd}' failed (exit code {:?}):\nstdout: {}\nstderr: {}",
                    exit_code,
                    stdout.trim(),
                    stderr.trim()
                );
            }
        }

        Ok(())
    }

    /// Verifies changes on the parent checkout after applying a patch.
    /// If verification fails, caller must invoke `ParentPreApplySnapshot::rollback`.
    pub async fn verify_parent(parent_repo: &Path, command: Option<&str>) -> Result<()> {
        if let Some(cmd) = command {
            let (passed, exit_code, stdout, stderr) =
                run_verification_command(cmd, parent_repo).await?;
            if !passed {
                bail!(
                    "parent post-apply verification command '{cmd}' failed (exit code {:?}):\nstdout: {}\nstderr: {}",
                    exit_code,
                    stdout.trim(),
                    stderr.trim()
                );
            }
        }

        Ok(())
    }

    /// Determines whether this worktree should be retained based on task wait reason.
    pub fn should_retain(wait_reason: Option<&crate::worker::StructuredWaitReason>) -> bool {
        matches!(
            wait_reason,
            Some(crate::worker::StructuredWaitReason::ApplyConflict { .. })
                | Some(crate::worker::StructuredWaitReason::WorktreeVerificationFailed { .. })
                | Some(crate::worker::StructuredWaitReason::PostApplyVerificationFailed { .. })
                | Some(crate::worker::StructuredWaitReason::AwaitingApply { .. })
        )
    }
}

/// Manages isolated git worktrees for write tasks.
#[derive(Clone, Debug)]
pub struct WorktreeManager {
    repo_path: PathBuf,
    worktrees_root: PathBuf,
}

impl WorktreeManager {
    pub fn new(repo_path: impl Into<PathBuf>, worktrees_root: impl Into<PathBuf>) -> Self {
        Self {
            repo_path: repo_path.into(),
            worktrees_root: worktrees_root.into(),
        }
    }

    /// Checks whether the repo path is a valid local Git repository, returning the current HEAD SHA.
    pub async fn verify_local_git_repo(&self) -> Result<String> {
        let output = Command::new("git")
            .arg("-C")
            .arg(&self.repo_path)
            .arg("rev-parse")
            .arg("--verify")
            .arg("HEAD")
            .output()
            .await
            .context("failed to check local git repository")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            bail!(
                "isolated worktrees require a valid local git repository at {:?}: {}",
                self.repo_path,
                stderr.trim()
            );
        }

        let sha = String::from_utf8(output.stdout)
            .context("invalid git rev-parse output")?
            .trim()
            .to_string();
        Ok(sha)
    }

    /// Creates a dedicated, isolated git worktree for a specific task execution attempt.
    pub async fn create_isolated_worktree(
        &self,
        run_id: &RunId,
        task_id: &TaskId,
        attempt: u32,
    ) -> Result<IsolatedWorktree> {
        self.create_isolated_worktree_with_scope(run_id, task_id, attempt, None)
            .await
    }

    pub async fn create_isolated_worktree_with_scope(
        &self,
        run_id: &RunId,
        task_id: &TaskId,
        attempt: u32,
        allowed_subpaths: Option<&[String]>,
    ) -> Result<IsolatedWorktree> {
        let head = self.verify_local_git_repo().await?;
        let baseline_commit = self.capture_baseline(&head, allowed_subpaths).await?;

        let unique_suffix = uuid::Uuid::new_v4().simple();
        let safe_task_id = safe_component(task_id.as_str());
        let safe_run_id = safe_component(run_id.as_str());
        let worktree_dir_name = format!(
            "wt_{}_{}_att{}_{}",
            safe_run_id, safe_task_id, attempt, unique_suffix
        );
        let worktree_path = self.worktrees_root.join(worktree_dir_name);
        let branch_name = format!(
            "zed-orch/{}/{}-att{}-{}",
            safe_run_id, safe_task_id, attempt, unique_suffix
        );

        if worktree_path.exists() {
            bail!(
                "refusing to reuse existing worktree path {:?}",
                worktree_path
            );
        }

        fs::create_dir_all(&self.worktrees_root).with_context(|| {
            format!(
                "failed to create worktrees root directory {:?}",
                self.worktrees_root
            )
        })?;

        // git worktree add -b <branch> <path> <baseline>
        let output = Command::new("git")
            .arg("-C")
            .arg(&self.repo_path)
            .arg("worktree")
            .arg("add")
            .arg("-b")
            .arg(&branch_name)
            .arg(&worktree_path)
            .arg(&baseline_commit)
            .output()
            .await
            .context("failed to execute git worktree add")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            bail!(
                "git worktree add failed for {:?}: {}",
                worktree_path,
                stderr.trim()
            );
        }

        let created_at = Utc::now();
        let marker = WorktreeOwnershipMarker {
            marker_version: 2,
            run_id: run_id.clone(),
            task_id: task_id.clone(),
            attempt,
            created_at,
            baseline_commit: baseline_commit.clone(),
            parent_checkout: fs::canonicalize(&self.repo_path)?,
            branch_name: branch_name.clone(),
        };
        marker.write_to(&worktree_path)?;

        Ok(IsolatedWorktree {
            repo_path: self.repo_path.clone(),
            worktree_path,
            branch_name,
            baseline_commit,
            run_id: run_id.clone(),
            task_id: task_id.clone(),
            attempt,
            created_at,
        })
    }

    async fn capture_baseline(
        &self,
        head: &str,
        allowed_subpaths: Option<&[String]>,
    ) -> Result<String> {
        let index =
            std::env::temp_dir().join(format!("nocturne-worker-index-{}", uuid::Uuid::new_v4()));
        let _index_cleanup = SnapshotIndex(index.clone());
        async {
            self.snapshot_git(&index, &["read-tree", head]).await?;
            let mut arguments = vec!["add", "--all", "--"];
            if let Some(paths) = allowed_subpaths {
                for path in paths {
                    let path_value = Path::new(path);
                    anyhow::ensure!(
                        !path_value.is_absolute()
                            && !path_value.components().any(|component| matches!(
                                component,
                                std::path::Component::ParentDir
                            )),
                        "invalid snapshot scope: {path}"
                    );
                    let exists = match fs::symlink_metadata(self.repo_path.join(path)) {
                        Ok(_) => true,
                        Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
                        Err(error) => return Err(error.into()),
                    };
                    if exists
                        || !self
                            .snapshot_git(&index, &["ls-files", "--", path])
                            .await?
                            .is_empty()
                    {
                        arguments.push(path);
                    }
                }
                anyhow::ensure!(!paths.is_empty(), "snapshot scope cannot be empty");
            } else {
                arguments.push(".");
            }
            if arguments.len() > 3 {
                self.snapshot_git(&index, &arguments).await?;
            }
            let tree = self.snapshot_git(&index, &["write-tree"]).await?;
            self.snapshot_git(
                &index,
                &[
                    "commit-tree",
                    tree.trim(),
                    "-p",
                    head,
                    "-m",
                    "Capture orchestration worker baseline",
                ],
            )
            .await
            .map(|commit| commit.trim().to_string())
        }
        .await
    }

    async fn snapshot_git(&self, index: &Path, arguments: &[&str]) -> Result<String> {
        let output = Command::new("git")
            .arg("-C")
            .arg(&self.repo_path)
            .args(arguments)
            .env("GIT_INDEX_FILE", index)
            .env("GIT_LITERAL_PATHSPECS", "1")
            .env("GIT_AUTHOR_NAME", "NocturneForge")
            .env("GIT_AUTHOR_EMAIL", "worker@nocturneforge.invalid")
            .env("GIT_COMMITTER_NAME", "NocturneForge")
            .env("GIT_COMMITTER_EMAIL", "worker@nocturneforge.invalid")
            .output()
            .await?;
        anyhow::ensure!(
            output.status.success(),
            "failed to snapshot parent checkout: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8(output.stdout).context("invalid Git snapshot response")
    }
}

fn safe_component(value: &str) -> String {
    let component = value
        .chars()
        .take(64)
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_') {
                character
            } else {
                '_'
            }
        })
        .collect::<String>();
    if component.is_empty() {
        "task".to_string()
    } else {
        component
    }
}

fn lexically_normalize(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                normalized.pop();
            }
            _ => normalized.push(component.as_os_str()),
        }
    }
    normalized
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn git(directory: &Path, arguments: &[&str]) -> Result<()> {
        let output = Command::new("git")
            .arg("-C")
            .arg(directory)
            .args(arguments)
            .env("GIT_CONFIG_NOSYSTEM", "1")
            .env("GIT_CONFIG_GLOBAL", "/dev/null")
            .output()
            .await?;
        anyhow::ensure!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        Ok(())
    }

    #[test]
    fn three_way_apply_preserves_parent_edits_and_rejects_conflicts() -> Result<()> {
        smol::block_on(async {
            let root =
                std::env::temp_dir().join(format!("nocturne-merge-test-{}", uuid::Uuid::new_v4()));
            let parent = root.join("parent");
            fs::create_dir_all(&parent)?;
            let result = async {
                git(&parent, &["init", "--quiet"]).await?;
                fs::write(parent.join("file.txt"), "first\nsecond\nthird\n")?;
                git(&parent, &["add", "."]).await?;
                git(
                    &parent,
                    &[
                        "-c",
                        "user.name=Test",
                        "-c",
                        "user.email=test@example.invalid",
                        "commit",
                        "-m",
                        "Initial",
                    ],
                )
                .await?;
                let manager = WorktreeManager::new(&parent, root.join("workers"));
                let worker = manager
                    .create_isolated_worktree(&RunId::new(), &TaskId::new("merge"), 1)
                    .await?;
                fs::write(
                    worker.worktree_path.join("file.txt"),
                    "first\nsecond\nworker\n",
                )?;
                fs::write(parent.join("file.txt"), "parent\nsecond\nthird\n")?;
                let index_before = fs::read(parent.join(".git/index"))?;
                worker.apply_to_parent(&parent, None).await?;
                assert_eq!(
                    fs::read_to_string(parent.join("file.txt"))?,
                    "parent\nsecond\nworker\n"
                );
                assert_eq!(fs::read(parent.join(".git/index"))?, index_before);
                fs::write(parent.join("file.txt"), "parent\nsecond\nconflicting\n")?;
                assert!(worker.apply_to_parent(&parent, None).await.is_err());
                assert_eq!(
                    fs::read_to_string(parent.join("file.txt"))?,
                    "parent\nsecond\nconflicting\n"
                );
                assert_eq!(fs::read(parent.join(".git/index"))?, index_before);
                assert!(worker.worktree_path.exists());
                worker.cleanup(true).await?;
                anyhow::Ok(())
            }
            .await;
            fs::remove_dir_all(&root)?;
            result
        })
    }

    #[test]
    fn dirty_snapshot_preserves_parent_index_and_excludes_unrelated_changes() -> Result<()> {
        smol::block_on(async {
            let root = std::env::temp_dir()
                .join(format!("nocturne-snapshot-test-{}", uuid::Uuid::new_v4()));
            let parent = root.join("parent");
            fs::create_dir_all(&parent)?;
            let result = async {
                git(&parent, &["init", "--quiet"]).await?;
                fs::write(parent.join("tracked.txt"), "committed\n")?;
                fs::write(parent.join("outside.txt"), "original\n")?;
                git(&parent, &["add", "."]).await?;
                git(
                    &parent,
                    &[
                        "-c",
                        "user.name=Test",
                        "-c",
                        "user.email=test@example.invalid",
                        "commit",
                        "-m",
                        "Initial",
                    ],
                )
                .await?;
                fs::write(parent.join("tracked.txt"), "staged\n")?;
                git(&parent, &["add", "tracked.txt"]).await?;
                fs::write(parent.join("tracked.txt"), "working\n")?;
                fs::write(parent.join("new.txt"), "untracked baseline\n")?;
                fs::write(parent.join("outside.txt"), "unrelated dirty\n")?;
                let index_before = fs::read(parent.join(".git/index"))?;
                let manager = WorktreeManager::new(&parent, root.join("workers"));
                let worker = manager
                    .create_isolated_worktree_with_scope(
                        &RunId::new(),
                        &TaskId::new("snapshot"),
                        1,
                        Some(&["tracked.txt".into(), "new.txt".into()]),
                    )
                    .await?;
                assert_eq!(fs::read(parent.join(".git/index"))?, index_before);
                assert_eq!(
                    fs::read_to_string(worker.worktree_path.join("tracked.txt"))?,
                    "working\n"
                );
                assert_eq!(
                    fs::read_to_string(worker.worktree_path.join("new.txt"))?,
                    "untracked baseline\n"
                );
                assert_eq!(
                    fs::read_to_string(worker.worktree_path.join("outside.txt"))?,
                    "original\n"
                );
                assert!(worker.collect_diff().await?.is_empty());
                fs::write(worker.worktree_path.join("new.txt"), "worker change\n")?;
                worker
                    .apply_to_parent(&parent, Some(&["new.txt".into()]))
                    .await?;
                assert_eq!(
                    fs::read_to_string(parent.join("new.txt"))?,
                    "worker change\n"
                );
                assert_eq!(fs::read_to_string(parent.join("tracked.txt"))?, "working\n");
                assert_eq!(
                    fs::read_to_string(parent.join("outside.txt"))?,
                    "unrelated dirty\n"
                );
                assert_eq!(fs::read(parent.join(".git/index"))?, index_before);
                worker.cleanup(true).await?;
                anyhow::Ok(())
            }
            .await;
            fs::remove_dir_all(&root)?;
            result
        })
    }

    #[test]
    fn linked_parent_is_preserved_and_switched_branch_is_not_removed() -> Result<()> {
        smol::block_on(async {
            let root = std::env::temp_dir()
                .join(format!("nocturne-worktree-test-{}", uuid::Uuid::new_v4()));
            fs::create_dir_all(&root)?;
            let result = async {
                git(&root, &["init", "--quiet"]).await?;
                git(
                    &root,
                    &[
                        "-c",
                        "user.name=Test",
                        "-c",
                        "user.email=test@example.invalid",
                        "commit",
                        "--allow-empty",
                        "-m",
                        "Initial",
                    ],
                )
                .await?;
                git(&root, &["worktree", "add", "-b", "parent", "linked-parent"]).await?;
                let parent = root.join("linked-parent");
                let manager = WorktreeManager::new(&parent, root.join("workers"));
                let run_id = RunId::new();
                let task_id = TaskId::new("worker");
                let worker = manager
                    .create_isolated_worktree(&run_id, &task_id, 1)
                    .await?;
                let restored = IsolatedWorktree::reopen_managed(
                    worker.worktree_path.clone(),
                    &run_id,
                    &task_id,
                    1,
                )
                .await?;
                assert_eq!(restored.repo_path, fs::canonicalize(&parent)?);
                fs::write(worker.worktree_path.join("result.txt"), "worker output\n")?;
                restored.apply_to_parent(&parent, None).await?;
                assert_eq!(
                    fs::read_to_string(parent.join("result.txt"))?,
                    "worker output\n"
                );
                assert!(!root.join("result.txt").exists());
                assert!(restored.apply_to_parent(&root, None).await.is_err());
                git(&worker.worktree_path, &["switch", "-c", "user-owned"]).await?;
                assert!(worker.cleanup(true).await.is_err());
                assert!(worker.worktree_path.exists());
                git(&worker.worktree_path, &["switch", &worker.branch_name]).await?;
                worker.cleanup(true).await?;
                git(&root, &["show-ref", "--verify", "refs/heads/user-owned"]).await?;
                anyhow::Ok(())
            }
            .await;
            fs::remove_dir_all(&root)?;
            result
        })
    }

    #[test]
    fn advancing_attempt_preserves_changes_and_invalidates_stale_ownership() -> Result<()> {
        smol::block_on(async {
            let root = std::env::temp_dir().join(format!(
                "nocturne-worktree-retry-test-{}",
                uuid::Uuid::new_v4()
            ));
            let parent = root.join("parent");
            fs::create_dir_all(&parent)?;
            let result = async {
                git(&parent, &["init", "--quiet"]).await?;
                fs::write(parent.join("file.txt"), "baseline\n")?;
                git(&parent, &["add", "."]).await?;
                git(
                    &parent,
                    &[
                        "-c",
                        "user.name=Test",
                        "-c",
                        "user.email=test@example.invalid",
                        "commit",
                        "-m",
                        "Initial",
                    ],
                )
                .await?;
                let run_id = RunId::new();
                let task_id = TaskId::new("retry");
                let worker = WorktreeManager::new(&parent, root.join("workers"))
                    .create_isolated_worktree(&run_id, &task_id, 1)
                    .await?;
                let stale_path = worker.worktree_path.clone();
                fs::write(worker.worktree_path.join("file.txt"), "partial change\n")?;

                let worker = worker.advance_attempt(2).await?;
                assert_eq!(worker.attempt, 2);
                assert_eq!(
                    fs::read_to_string(worker.worktree_path.join("file.txt"))?,
                    "partial change\n"
                );
                assert!(
                    IsolatedWorktree::reopen_managed(stale_path, &run_id, &task_id, 1)
                        .await
                        .is_err()
                );
                IsolatedWorktree::reopen_managed(
                    worker.worktree_path.clone(),
                    &run_id,
                    &task_id,
                    2,
                )
                .await?;
                worker.cleanup(true).await?;
                anyhow::Ok(())
            }
            .await;
            fs::remove_dir_all(&root)?;
            result
        })
    }

    #[test]
    fn test_ownership_marker_round_trip() {
        let temp_dir = std::env::temp_dir().join(format!("test_marker_{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&temp_dir).unwrap();

        let marker = WorktreeOwnershipMarker {
            marker_version: 2,
            run_id: RunId::new(),
            task_id: TaskId::new("t1"),
            attempt: 1,
            created_at: Utc::now(),
            baseline_commit: "abcd1234ef5678".to_string(),
            parent_checkout: temp_dir.clone(),
            branch_name: "test-branch".to_string(),
        };

        marker.write_to(&temp_dir).unwrap();
        let loaded = WorktreeOwnershipMarker::read_from(&temp_dir).unwrap();
        assert_eq!(loaded, marker);

        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn test_non_git_repository_rejected() {
        let non_git = std::env::temp_dir().join(format!("test_nongit_{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&non_git).unwrap();

        let manager = WorktreeManager::new(&non_git, non_git.join("worktrees"));
        let err = smol::block_on(manager.verify_local_git_repo()).unwrap_err();
        assert!(
            err.to_string()
                .contains("isolated worktrees require a valid local git repository")
        );

        let _ = fs::remove_dir_all(non_git);
    }

    #[test]
    fn test_parallel_worktrees_have_distinct_paths_and_branches() {
        let run_id = RunId::new();
        let task_1 = TaskId::new("task-1");
        let task_2 = TaskId::new("task-2");

        let wt1_name = format!("wt_{}_{}_att{}", run_id.as_str(), task_1.as_str(), 1);
        let wt2_name = format!("wt_{}_{}_att{}", run_id.as_str(), task_2.as_str(), 1);
        assert_ne!(wt1_name, wt2_name);

        let branch1 = format!("zed-orch/{}/{}-att{}", run_id.as_str(), task_1.as_str(), 1);
        let branch2 = format!("zed-orch/{}/{}-att{}", run_id.as_str(), task_2.as_str(), 1);
        assert_ne!(branch1, branch2);
    }

    #[test]
    fn test_cleanup_refuses_without_ownership_marker() {
        let temp_dir = std::env::temp_dir().join(format!("test_cleanup_{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&temp_dir).unwrap();

        let worktree = IsolatedWorktree {
            repo_path: temp_dir.clone(),
            worktree_path: temp_dir.clone(),
            branch_name: "test-branch".to_string(),
            baseline_commit: "abc".to_string(),
            run_id: RunId::new(),
            task_id: TaskId::new("t1"),
            attempt: 1,
            created_at: Utc::now(),
        };

        // Cleanup must fail because no ownership marker exists
        let result = smol::block_on(worktree.cleanup(true));
        assert!(result.is_err());
        let err_str = format!("{:#}", result.unwrap_err());
        assert!(err_str.contains("ownership marker"));

        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn test_parent_pre_apply_snapshot_and_rollback() -> Result<()> {
        smol::block_on(async {
            let root = std::env::temp_dir()
                .join(format!("test-pre-apply-rollback-{}", uuid::Uuid::new_v4()));
            let parent = root.join("parent");
            fs::create_dir_all(&parent)?;
            let result = async {
                git(&parent, &["init", "--quiet"]).await?;
                fs::write(parent.join("existing.txt"), "original parent content\n")?;
                git(&parent, &["add", "."]).await?;
                git(
                    &parent,
                    &[
                        "-c",
                        "user.name=Test",
                        "-c",
                        "user.email=test@example.invalid",
                        "commit",
                        "-m",
                        "Initial",
                    ],
                )
                .await?;

                let manager = WorktreeManager::new(&parent, root.join("workers"));
                let worker = manager
                    .create_isolated_worktree(&RunId::new(), &TaskId::new("rollback-task"), 1)
                    .await?;

                // Worker modifies existing file and adds a new file
                fs::write(
                    worker.worktree_path.join("existing.txt"),
                    "modified by worker\n",
                )?;
                fs::write(
                    worker.worktree_path.join("brand_new.txt"),
                    "created by worker\n",
                )?;

                // Capture snapshot
                let pre_apply = worker.prepare_parent_apply(&parent, None).await?;
                assert!(pre_apply.pre_existing_paths.contains("existing.txt"));
                assert!(!pre_apply.pre_existing_paths.contains("brand_new.txt"));

                // Apply to parent
                worker.apply_to_parent(&parent, None).await?;
                assert_eq!(
                    fs::read_to_string(parent.join("existing.txt"))?,
                    "modified by worker\n"
                );
                assert_eq!(
                    fs::read_to_string(parent.join("brand_new.txt"))?,
                    "created by worker\n"
                );

                // Rollback parent checkout
                pre_apply.rollback().await?;
                assert_eq!(
                    fs::read_to_string(parent.join("existing.txt"))?,
                    "original parent content\n"
                );
                assert!(!parent.join("brand_new.txt").exists());

                worker.cleanup(true).await?;
                anyhow::Ok(())
            }
            .await;
            let _ = fs::remove_dir_all(&root);
            result
        })
    }

    #[test]
    fn test_worktree_and_parent_verification_commands() -> Result<()> {
        smol::block_on(async {
            let root = std::env::temp_dir().join(format!(
                "test-verification-commands-{}",
                uuid::Uuid::new_v4()
            ));
            let parent = root.join("parent");
            fs::create_dir_all(&parent)?;
            let result = async {
                git(&parent, &["init", "--quiet"]).await?;
                fs::write(parent.join("file.txt"), "hello\n")?;
                git(&parent, &["add", "."]).await?;
                git(
                    &parent,
                    &[
                        "-c",
                        "user.name=Test",
                        "-c",
                        "user.email=test@example.invalid",
                        "commit",
                        "-m",
                        "Initial",
                    ],
                )
                .await?;

                let manager = WorktreeManager::new(&parent, root.join("workers"));
                let worker = manager
                    .create_isolated_worktree(&RunId::new(), &TaskId::new("verify-task"), 1)
                    .await?;

                // Worktree verification
                assert!(worker.verify_worktree(Some("true"), None).await.is_ok());
                let worktree_error = worker
                    .verify_worktree(Some("false"), None)
                    .await
                    .expect_err("false should fail verification");
                assert!(worktree_error.to_string().contains("exit code Some(1)"));
                assert!(worker.verify_worktree(None, None).await.is_ok());

                // Parent verification
                assert!(
                    IsolatedWorktree::verify_parent(&parent, Some("true"))
                        .await
                        .is_ok()
                );
                let parent_error = IsolatedWorktree::verify_parent(&parent, Some("false"))
                    .await
                    .expect_err("false should fail parent verification");
                assert!(parent_error.to_string().contains("exit code Some(1)"));
                assert!(IsolatedWorktree::verify_parent(&parent, None).await.is_ok());

                worker.cleanup(true).await?;
                anyhow::Ok(())
            }
            .await;
            let _ = fs::remove_dir_all(&root);
            result
        })
    }

    #[test]
    fn test_should_retain() {
        use crate::worker::StructuredWaitReason;
        assert!(IsolatedWorktree::should_retain(Some(
            &StructuredWaitReason::ApplyConflict {
                error: "conflict".to_string(),
                worktree_path: "/tmp/wt".to_string(),
            }
        )));
        assert!(IsolatedWorktree::should_retain(Some(
            &StructuredWaitReason::PostApplyVerificationFailed {
                error: "tests failed".to_string(),
                worktree_path: "/tmp/wt".to_string(),
                rollback_error: None,
            }
        )));
        assert!(IsolatedWorktree::should_retain(Some(
            &StructuredWaitReason::AwaitingApply {
                patch_id: None,
                worktree_path: Some("/tmp/wt".to_string()),
            }
        )));
        assert!(!IsolatedWorktree::should_retain(Some(
            &StructuredWaitReason::AwaitingApproval
        )));
        assert!(!IsolatedWorktree::should_retain(None));
    }
}
