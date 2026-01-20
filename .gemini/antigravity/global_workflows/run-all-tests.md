---
description: Iteratively run and fix tests until zero failures
triggers:
- /run-all-tests
- workflow for run all tests
version: 2.2.2
allowed-tools: [Bash, Read, Edit, Task]
agents:
  primary: test-engineer
skills:
- test-debugging
argument-hint: '[--fix] [--max-iterations=10]'
---

# Iterative Test Fixer (v2.2.2)

// turbo-all

## Phase 1: Execution (Sequential)

1.  **Run Suite**
    - Action: Run all tests (pytest/npm test).
    - Capture: Failure logs.

## Phase 2: Analysis (Parallel)

// parallel

2.  **Categorize Failures**
    - Group by type: Assertion, Runtime, Import, Timeout.

3.  **RCA Generation**
    - Generate hypothesis for each failure group.

// end-parallel

## Phase 3: Fix Loop (Iterative)

4.  **Apply Fixes**
    - Action: Fix highest priority failures.

5.  **Verify**
    - Action: Re-run specific tests.

6.  **Loop**
    - Repeat until 0 failures or Max Iterations.

## Phase 4: Final Validity

7.  **Final Sweep**
    - Action: One last full run to ensure no regressions.
