---
description: Coordinate specialized agents for optimization
triggers:
- /multi-agent-optimize
- coordinate optimization
version: 2.0.0
allowed-tools: [Read, Task, Bash]
agents:
  primary: orchestrator
  orchestrated: true
skills:
- software-architecture
- performance-optimization
argument-hint: '<target-path>'
---

# Swarm Optimizer (v2.0)

// turbo-all

## Phase 1: Scan (Parallel Agents)

// parallel

1.  **Systems Analysis**
    - Agent: systems-architect
    - Action: Analyze Architecture.

2.  **Numerical Analysis**
    - Agent: hpc-coordinator
    - Action: Analyze numerical efficiency.

3.  **GPU Analysis**
    - Agent: jax-pro
    - Action: Check GPU utilization.

// end-parallel

## Phase 2: Synthesis (Sequential)

4.  **Conflict Resolution**
    - Action: De-duplicate and rank optimizations.

## Phase 3: Application (Iterative)

5.  **Apply Safe Patches**
    - Action: Apply and verify.
