---
description: Multi-agent code review workflow
triggers:
- /full-review
- workflow for full review
version: 2.0.0
allowed-tools: [Read, Task, Bash]
agents:
  primary: code-reviewer
skills:
- code-review-best-practices
- secure-coding
argument-hint: '[--mode=quick|standard|deep]'
---

# Multi-Perspective Review (v2.0)

// turbo-all

## Phase 1: Automated Scan (Parallel)

// parallel

1.  **Quality Metrics**
    - Action: Check complexity, duplication.

2.  **Security Scan**
    - Action: Check for secrets, CVEs.

3.  **Architecture Check**
    - Action: Check dependency boundaries.

// end-parallel

## Phase 2: Intelligent Review (Sequential)

4.  **Readability & Standard**
    - Review naming, comments, project structure.

5.  **Logic & Correctness**
    - Review algorithm correctness, edge cases.

6.  **Performance Check**
    - Identify N+1 queries, loops, memory issues.

## Phase 3: Feedback

7.  **Prioritized Report**
    - Critical (Must Fix) to Minor (Nice to have).
