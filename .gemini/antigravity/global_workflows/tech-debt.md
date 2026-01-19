---
description: Analyze and prioritize technical debt
triggers:
- /tech-debt
- analyze technical debt
version: 2.0.0
allowed-tools: [Bash, Read, Task]
agents:
  primary: technical-lead
skills:
- code-quality
- prioritization
argument-hint: '[--mode=quick|standard]'
---

# Debt Collector (v2.0)

// turbo-all

## Phase 1: Inventory (Parallel)

// parallel

1.  **Code Analysis**
    - Action: Locate TODOs, complexity (Cyclomatic).

2.  **Test Analysis**
    - Action: Check coverage gaps, flaky tests.

3.  **Dependency Analysis**
    - Action: Check outdated packages.

// end-parallel

## Phase 2: Scoring (Sequential)

4.  **Prioritization**
    - Action: Calculate Debt Score (Impact * Effort).

## Phase 3: Roadmap

5.  **Plan**
    - Action: Assign Quick Wins vs Long Term.
