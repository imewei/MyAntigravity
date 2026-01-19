---
name: speckit-taskstoissues
description: Convert tasks.md into GitHub issues.
version: 2.0.0
agents:
  primary: project-manager
skills:
- issue-management
- git-integration
- task-sync
allowed-tools: [Read, Write, Task, Bash, GitHub]
---

# Tasks to Issues Workflow

// turbo-all

# Tasks to Issues Workflow

Syncing plan to reality.

---

## User Input

```text
$ARGUMENTS
```

---

## Phase 1: Verification (Parallel)

// parallel

### Prerequisite Check
-   Run `.specify/scripts/bash/check-prerequisites.sh --json --require-tasks`.
-   Locate `tasks.md`.

### Remote Check
-   Run `git config --get remote.origin.url`.
-   **CAUTION**: Verify it is a valid GitHub URL.

// end-parallel

---

## Phase 2: Issue Creation

**Action**: Iterate through tasks.
-   **Input**: Task List from `tasks.md`.
-   **Output**: GitHub Issue for each task.

**Rule**: Do NOT create issues if remote mismatch.

---

## Completion

Report:
-   Number of issues created.
-   Link to repository issues page.
