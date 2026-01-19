---
description: Intelligent debugging with automated RCA
triggers:
- /smart-debug
- intelligent debugging
version: 2.0.0
allowed-tools: [Read, Task, Bash]
agents:
  primary: debugger
skills:
- debugging-mastery
- root-cause-analysis
argument-hint: '<error-description>'
---

# Smart Debug Engine (v2.0)

// turbo-all

## Phase 1: Triage (Sequential)

1.  **Error Parse**
    - Action: Classify error type and severity.

## Phase 2: Investigation (Parallel)

// parallel

2.  **Log Analysis**
    - Action: Check logs for stack traces, timestamps.

3.  **Environment Check**
    - Action: Verify config, resource usage.

4.  **Hypothesis Generation**
    - Action: Generate top 3 likely causes (5 Whys).

// end-parallel

## Phase 3: Reproduce & Fix (Sequential)

5.  **Reproduction**
    - Action: Create localized reproduction case.

6.  **Implementation**
    - Action: Apply fix.

## Phase 4: Validation (Parallel)

// parallel

7.  **Test Verification**
    - Action: Run tests.

8.  **Regression Check**
    - Action: Ensure no side effects.

// end-parallel
