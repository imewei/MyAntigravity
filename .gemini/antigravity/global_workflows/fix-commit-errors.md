---
description: Analyze and fix CI/CD failures using multi-agent error analysis
triggers:
- /fix-commit-errors
- workflow for fix commit errors
version: 2.2.2
allowed-tools: "[Bash(gh:*), Bash(git:*), Bash(npm:*), Bash(pip:*), Read]"
agents:
  primary: devops-troubleshooter
  conditional:
  - agent: python-developer
    trigger: error "ModuleNotFoundError"
  - agent: frontend-developer
    trigger: error "npm ERR"
skills:
- ci-cd-debugging
- error-pattern-matching
argument-hint: '[run-id] [--auto-fix]'
---

# CI/CD Forensics (v2.2.2)

// turbo-all

## Phase 1: Forensics (Parallel)

// parallel

1.  **Log Extraction**
    - Action: `gh run view --log-failed`.
    - Goal: Extract error stack trace.

2.  **Context Diff**
    - Action: `git show` (what changed in this commit?).
    - Goal: Correlate changes to errors.

// end-parallel

## Phase 2: Diagnosis (Sequential)

3.  **Pattern Matching**
    - Match error against known patterns (missing deps, env vars, syntax errors).

4.  **Solution Hypothesis**
    - Generate High/Medium/Low confidence solution.

## Phase 3: Remediation (Sequential)

5.  **Apply Fix** (if auto-fix or High confidence)
    - Action: Apply code change or config update.

6.  **Local Verification**
    - Action: Run affected test/build locally.

7.  **Push & Watch**
    - Action: `git push`, `gh run watch`.
