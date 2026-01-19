---
description: Set up observability stack (Prometheus/Grafana)
triggers:
- /monitor-setup
- setup monitoring
version: 2.0.0
allowed-tools: [Bash, Write, Read, Task]
agents:
  primary: sre-engineer
skills:
- observability-monitoring
- infrastructure-as-code
argument-hint: '[--mode=quick|standard|enterprise]'
---

# Observability Architect (v2.0)

// turbo-all

## Phase 1: Infrastructure (Parallel)

// parallel

1.  **Prometheus Setup**
    - Action: Deploy Prometheus/VictoriaMetrics.

2.  **Grafana Setup**
    - Action: Deploy Grafana + Datasource.

3.  **Trace Collector**
    - Action: Deploy Jaeger/Tempo.

// end-parallel

## Phase 2: Configuration (Parallel)

// parallel

4.  **Dashboards**
    - Action: Import RED/USE dashboards.

5.  **Alert Rules**
    - Action: Configure Critical/Warning alerts.

// end-parallel

## Phase 3: Validation

6.  **Verify Signals**
    - Action: Check for metric ingestion.
