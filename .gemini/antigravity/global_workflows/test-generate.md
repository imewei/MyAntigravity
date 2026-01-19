---
description: Generate comprehensive test suites
triggers:
- /test-generate
- generate tests
version: 2.0.0
allowed-tools: [Read, Task, Bash]
agents:
  primary: qa-engineer
skills:
- test-automation
- scientific-testing
argument-hint: '<source-file>'
---

# Test Generator (v2.0)

// turbo-all

## Phase 1: Analysis (Sequential)

1.  **Code Scan**
    - Action: Parse AST for functions and branches.

## Phase 2: Generation (Parallel)

// parallel

2.  **Unit Tests**
    - Action: Generate happy path & edge cases.

3.  **Property Tests**
    - Action: Generate Hypothesis/QuickCheck properties.

4.  **Scientific Tests**
    - Action: Generate numerical tolerance checks (if applicable).

// end-parallel

## Phase 3: Validation

5.  **Execution**
    - Action: Run generated tests. Confirm Pass.
