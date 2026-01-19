---
description: Advanced reflection engine for AI reasoning
triggers:
- /reflection
- workflow for reflection
version: 2.0.0
allowed-tools: [Read, Grep, Task]
agents:
  primary: facilitator
skills:
- meta-cognition
argument-hint: '[--mode=quick|standard]'
---

# Deep Reflection (v2.0)

// turbo-all

## Phase 1: Data Gathering (Parallel)

// parallel

1.  **Session Log**
    - Action: Scan recent interactions.

2.  **Code Changes**
    - Action: `git diff --stat`.

// end-parallel

## Phase 2: Multi-Dimensional Analysis (Parallel)

// parallel

3.  **Reasoning Check**
    - Analyze logic coherence.

4.  **Technical Check**
    - Analyze code quality/debt.

5.  **Strategic Check**
    - Analyze goal alignment.

// end-parallel

## Phase 3: Synthesis

6.  **Report**
    - Action: Generate Scores and Recommendations.
