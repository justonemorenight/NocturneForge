---
name: commit-message
description: Generate accurate Git commit messages that follow the repository's established style and explain intent without inventing changes.
disable-model-invocation: true
---

Write the commit message from the supplied staged or working-tree snapshot.

- Describe the actual behavior change, not the mechanics of editing files.
- Learn naming, prefixes, scopes, casing, and tone from recent repository and current-author subjects. Reuse established conventions and terminology when they accurately fit the new changes.
- Prefer the repository's established convention when the history is consistent. Otherwise use a concise imperative subject.
- Preserve an explicit user draft when it is accurate, improving it instead of changing direction.
- Follow the supplied commit template and explicit commit-message instructions. Explicit instructions have higher priority than this skill.
- Mention the most important affected subsystem or scope when that makes the subject more specific.
- Add a body only when it explains motivation, important tradeoffs, migrations, or behavior that the subject cannot capture.
- Do not mention files, tests, tools, or implementation details unless they are the meaningful user-facing change.
- Do not claim tests ran, issues were fixed, or behavior changed unless the provided context supports that claim.
- Return only the final commit message.
