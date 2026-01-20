---
name: git-pr-patterns
description: Branching strategies, Conventional Commits, and PR Lifecycle management.
version: 2.2.2
agents:
  primary: code-reviewer
skills:
- branching-strategy
- pr-management
- commit-convention
- code-review
allowed-tools: [Read, Write, Task, Bash]
triggers:
- keyword:pr
- keyword:pull-request
- keyword:review

# Git & PR Patterns

// turbo-all

# Git & PR Patterns

Standards for collaboration, version control, and change management.



## Strategy & Standards (Parallel)

// parallel

### Branching Models

| Model | Flow | Use Case |
|-------|------|----------|
| **GitHub Flow** | `main` <- `feature/*` | CD, Web Apps (Deploy on merge). |
| **GitFlow** | `main`, `develop`, `release/*` | Versioned Software, Mobile Apps. |
| **Trunk-Based** | `main` (Short-lived branches) | High-Velocity Teams. |

### Semantic Commits

-   `feat`: New feature.
-   `fix`: Bug fix.
-   `docs`: Documentation only.
-   `style`: Formatting (lint).
-   `refactor`: No logic change.
-   `test`: Add/Edit tests.
-   `chore`: Build/Tooling.

// end-parallel

---

## Decision Framework

### PR Lifecycle

1.  **Draft**: Work in progress (WIP).
2.  **Open**: Ready for review (CI running).
3.  **Review**: Code review feedback loop.
4.  **Approved**: All checks pass + Approval.
5.  **Merge**: Squash & Merge (Cleaner history) or Rebase.

---

## Core Knowledge (Parallel)

// parallel

### Constitutional AI Principles

1.  **Atomicity (Target: 100%)**: One logical change per PR.
2.  **Context (Target: 100%)**: PR description explains "Why", not just "What".
3.  **Quality (Target: 100%)**: CI must pass before merge.

### Quick Reference Templates

-   **PR Title**: `feat(auth): enable sso login`
-   **PR Body**:
    ```markdown
    ## Summary
    Adds SSO via SAML.
    ## Testing
    - [x] Unit tests
    - [x] Tested with Okta
    ```

// end-parallel

---

## Quality Assurance

### Common Bad Habits

| Habit | Fix |
|-------|-----|
| "Huge PR" (1000+ lines) | Split into stacked PRs. |
| "WIP" title forever | Convert to Draft PR. |
| Ignoring CI | Fix build *before* asking for review. |
| Force Push Reviews | Communicate before overwriting history during review. |

### PR Checklist

- [ ] Title follows semantic convention
- [ ] Description filled out
- [ ] Related Issues linked (`Fixes #123`)
- [ ] CI passing
- [ ] Self-review completed
- [ ] No merge conflicts
