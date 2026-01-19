---
description: Enhance PRs with automated review and insights
triggers:
- /pr-enhance
- enhance pr
version: 2.0.0
allowed-tools: [Read, Task, Bash]
agents:
  primary: code-reviewer
skills:
- code-review-best-practices
argument-hint: '[--mode=basic|enhanced]'
---

# PR Enhancer (v2.0)

// turbo-all

## Phase 1: Analysis (Parallel)

// parallel

1.  **Diff Scan**
    - Action: `git diff --stat`. Identify scope.

2.  **Risk Assessment**
    - Action: Calculate risk score based on complexity.

3.  **Compliance Check**
    - Action: Check for tests, docs, secrets.

// end-parallel

## Phase 2: Content Generation (Sequential)

4.  **Description Draft**
    - Action: Generate PR description with Summary, Impact, Plan.

5.  **Review Comments**
    - Action: Generate automated review suggestions.

## Phase 3: Output

6.  **Final Polish**
    - Action: Format as Markdown for PR body.
