---
description: Orchestrate systematic code migration with test-first discipline
triggers:
- /code-migrate
- orchestrate systematic code migration
version: 2.0.0
allowed-tools: [Read, Task, Bash]
agents:
  primary: legacy-modernizer
skills:
- refactoring-patterns
- test-engineering
argument-hint: '<source-path> [--target=framework]'
---

# Code Migration Orchestrator (v2.0)

// turbo-all

## Phase 1: Strategy (Sequential)

1.  **Assessment**
    - Analyze source vs target gap.
    - Select Strategy: Strangler Fig vs Big Bang.

## Phase 2: Baselining (Parallel)

// parallel

2.  **Snapshot**
    - Action: Capture current behavior (Characterization Tests).

3.  **Performance Baseline**
    - Action: Record current metrics (Time/Memory).

4.  **Contract Definition**
    - Action: Define API schemas/interfaces to preserve.

// end-parallel

## Phase 3: Execution (Iterative)

5.  **Codemod / Rewrite**
    - Action: Apply automated transforms or manual rewrite.

6.  **Validation Loop**
    - Run tests -> Fix -> Repeat.

## Phase 4: Final Validation (Parallel)

// parallel

7.  **Regression Test**
    - Verify all tests pass.

8.  **Performance Compare**
    - Compare against Phase 2 baseline.

// end-parallel
