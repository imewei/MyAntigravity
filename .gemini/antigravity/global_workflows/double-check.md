---
description: Multi-dimensional code validation (Lint, Test, Security, Perf)
triggers:
- /double-check
- workflow for double check
version: 2.2.0
allowed-tools: [Read, Task, Bash]
agents:
  primary: qa-engineer
  conditional:
  - agent: security-auditor
    trigger: flag "--security"
skills:
- quality-assurance
- security-auditing
argument-hint: '[--deep] [--security]'
---

# Comprehensive Validator (v2.0)

// turbo-all

## Phase 1: Automated Checks (Parallel)

// parallel

1.  **Lint & Form**
    - Action: `ruff check`, `eslint`, `cargo clippy`.

2.  **Type Safety**
    - Action: `mypy`, `tsc --noEmit`.

3.  **Test Suite**
    - Action: `pytest`, `npm test`, `cargo test`.

4.  **Basic Security**
    - Action: `npm audit`, `pip-audit`.

// end-parallel

## Phase 2: Deep Analysis (Sequential)

5.  **Manual Logic Review**
    - Check edge cases, error handling, readability.

6.  **Advanced Scans** (if --deep)
    - Action: Coverage report, Complexity analysis.

## Phase 3: Report

7.  **Summary**
    - Pass/Fail status per dimension.
    - Recommendations.
