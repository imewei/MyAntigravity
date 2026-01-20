---
name: speckit-taskstoissues
description: Convert tasks.md into GitHub issues.
version: 2.2.1
agents:
  primary: project-manager
skills:
- issue-management
- git-integration
- task-sync
allowed-tools: [Read, Write, Task, Bash, GitHub]
triggers:
- keyword:speckit
- keyword:issues
- file:tasks.md
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

---

## Tool Patterns

### Issue Creation via MCP

```python
# Create issue via GitHub MCP
mcp_github_mcp_server_issue_write(
    owner="owner",
    repo="repo",
    title="Task Title",
    body="Task Description",
    method="create"
)
```
