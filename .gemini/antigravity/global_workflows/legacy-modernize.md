---
description: Comprehensive legacy system modernization using Strangler Fig pattern
triggers:
- /legacy-modernize
- comprehensive legacy system modernization
version: 2.0.0
allowed-tools: [Bash, Read, Write, Edit, Task]
agents:
  primary: system-architect
skills:
- legacy-refactoring
- system-design
- api-design
argument-hint: '<legacy-path> [--strategy=strangler|big-bang]'
---

# Legacy Modernization Orchestrator (v2.0)

// turbo-all

## Phase 1: Assessment (Parallel)

// parallel

1.  **Static Analysis**
    - Action: Count LOC, dependencies, and complexity metrics.
    - Goal: Identify "Hotspots".

2.  **Coupling Map**
    - Action: Map database dependencies and external API calls.
    - Goal: Define "Seams" for extraction.

3.  **Risk Audit**
    - Action: Identify security vulnerabilities and deprecated runtimes.

// end-parallel

## Phase 2: Strategy (Sequential)

4.  **Prioritization**
    - Method: (Business Value / Complexity) score.

5.  **Architecture Definition**
    - Define Target Architecture (Microservices, Modular Monolith).
    - Define Interop Layer (Facades/Adapters).

## Phase 3: Execution (Iterative Strangler Fig)

6.  **Extract Component**
    - Isolate logic, write tests (Characterization Tests).

7.  **Modernize**
    - Rewrite in new stack/pattern.

8.  **Switch Traffic**
    - Feature flag rollout (5% -> 100%).

## Phase 4: Decommission

9.  **Remove Legacy**
    - Delete old code paths once traffic is 0%.
