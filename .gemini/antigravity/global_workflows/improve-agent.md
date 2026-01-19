---
description: Systematic agent improvement through analysis
triggers:
- /improve-agent
- improve agent performance
version: 2.0.0
allowed-tools: [Read, Task, Bash]
agents:
  primary: agent-optimizer
skills:
- prompt-engineering
- data-analysis
argument-hint: '<agent-name> [--mode=check|optimize]'
---

# Agent Optimizer (v2.0)

// turbo-all

## Phase 1: Health Check (Parallel)

// parallel

1.  **Metric Analysis**
    - Action: Check success rates, latency.

2.  **Failure Analysis**
    - Action: Scan for common error patterns.

// end-parallel

## Phase 2: Diagnosis (Sequential)

3.  **Report Generation**
    - Action: Synthesize health report.

## Phase 3: Optimization (Iterative)

4.  **Prompt Refinement**
    - Action: Apply improvements.
    - Validate: A/B Test.
