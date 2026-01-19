---
name: performance-engineer-observability-monitoring
description: Core Web Vitals, APM, Distributed Tracing, and RUM for full-stack optimization.
version: 2.0.0
agents:
  primary: performance-engineer
skills:
- open-telemetry
- distributed-tracing
- rum-analytics
- bottleneck-analysis
allowed-tools: [Read, Write, Task, Bash]
triggers:
- keyword:performance-engineer-observability-monitoring
---

# Observability & Monitoring

// turbo-all

# Observability & Monitoring

Data-driven optimization using Tracing, Metrics, and Logs (TML).

---

## Strategy & Telemetry (Parallel)

// parallel

### Telemetry Pillars

| Pillar | Tooling | Purpose |
|--------|---------|---------|
| **Metrics** | Prometheus, Datadog | Aggregates (Count, Gauge, Histogram). "Is it slow?" |
| **Logs** | ELK, Loki, Splunk | Events. "Why is it slow?" |
| **Traces** | Jaeger, Honeycomb | Request lifecycle. "Where is it slow?" |
| **RUM** | Google Analytics, Sentry | User perception. "Does the user care?" |

### Baseline KPIs

-   **Latency**: P50 < 100ms, P95 < 300ms, P99 < 1s.
-   **Vital**: LCP < 2.5s, CLS < 0.1, FID < 100ms.
-   **Error Rate**: < 0.1%.

// end-parallel

---

## Decision Framework

### Bottleneck Detection

1.  **Symptom**: High Latency alert fires.
2.  **Metric**: Check P95 breakdown (DB vs App vs Network).
3.  **Trace**: Find span duration > 100ms.
4.  **Profile**: Flame graph CPU/Memory for that span.
5.  **Fix**: Optimize query, cache result, or code check.

---

## Core Knowledge (Parallel)

// parallel

### Constitutional AI Principles

1.  **Data-First (Target: 100%)**: No optimization without a baseline measurement.
2.  **User-Centric (Target: 100%)**: RUM trumps synthetic checks.
3.  **Budgeted (Target: 100%)**: Enforce performance budgets in CI.

### Quick Reference Commands

-   `node --prof app.js`
-   `curl -w "@curl-format.txt" https://site.com`
-   `EXPLAIN ANALYZE SELECT ...`
-   `k6 run load-test.js`

// end-parallel

---

## Quality Assurance

### Common Anti-Patterns

| Anti-Pattern | Fix |
|--------------|-----|
| "Averages" | Use Percentiles (P95/P99). Averages hide outliers. |
| Log Spam | Sample logs or use dynamic levels. |
| Blind Optimization | Optimization without profiling is guessing. |
| Ignoring Database | 90% of issues are N+1 or missing indexes. |

### Obs Checklist

- [ ] P95/P99 Dashboards active
- [ ] Distributed Tracing connected (all microservices)
- [ ] Alerting thresholds tuned (no fatigue)
- [ ] Core Web Vitals tracked
- [ ] Slow Query Log enabled
- [ ] N+1 detection in CI
