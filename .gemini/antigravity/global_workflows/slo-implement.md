---
description: Implement SLO/SLA monitoring
triggers:
- /slo-implement
- implement slo
version: 2.0.0
allowed-tools: [Read, Task, Bash]
agents:
  primary: sre-engineer
skills:
- site-reliability-engineering
argument-hint: '[service-name]'
---

# SLO Architect (v2.0)

// turbo-all

## Phase 1: Definition (Parallel)

// parallel

1.  **SLI Selection**
    - Action: Define Availability/Latency metrics.

2.  **SLO Target**
    - Action: Define target % (e.g., 99.9%).

3.  **Error Budget**
    - Action: Calculate budget minutes.

// end-parallel

## Phase 2: Implementation (Sequential)

4.  **Burn Rate Alerts**
    - Action: Configure Fast/Slow burn alerts.

5.  **Reporting**
    - Action: Setup monthly SLO report.
