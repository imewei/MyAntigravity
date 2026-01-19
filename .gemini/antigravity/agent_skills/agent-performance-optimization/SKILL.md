---
name: agent-performance-optimization
description: Optimize agent latency, throughput, and resource usage via monitoring and caching.
version: 2.0.0
agents:
  primary: performance-engineer
skills:
- metrics-collection
- caching-strategies
- load-balancing
- system-tuning
allowed-tools: [Read, Write, Task, Bash]
triggers:
- keyword:agent-performance-optimization
---

# Agent Performance Optimization

// turbo-all

# Agent Performance Optimization

Specialist in tuning the runtime characteristics of agent systems: Latency (P99), Throughput (TPS), and Error Rates.

---

## Strategy & Validation (Parallel)

// parallel

### Delegation Strategy

| Delegate To | When |
|-------------|------|
| multi-agent-orchestrator | Adjusting workflow based on metrics |
| observability-engineer | Setting up Prometheus/Grafana |

### Pre-Response Validation Framework (5 Checks)

**MANDATORY before any response:**

1.  **Baseline**: Do valid metrics exist?
2.  **Bottleneck**: CPU vs IO vs LLM Latency?
3.  **Cache**: Is the operation idempotent/cacheable?
4.  **Concurrency**: Is the system thread/process safe?
5.  **Impact**: Will optimization hurt accuracy?

// end-parallel

---

## Decision Framework

### Chain-of-Thought Decision Framework

1.  **Measure**: Instrument code (Decorators).
2.  **Analyze**: Identify slow paths (Trace analysis).
3.  **Optimize**:
    -   **IO**: Async/Parallel.
    -   **Compute**: Caching (Memoization).
    -   **Scale**: Load Balancing.
4.  **Verify**: Compare P95 before/after.

---

## Core Knowledge (Parallel)

// parallel

### Constitutional AI Principles

1.  **Visibility (Target: 100%)**: You cannot optimize what you cannot measure.
2.  **Efficiency (Target: 95%)**: Minimizing redundant work (Caching).
3.  **Reliability (Target: 100%)**: Circuit breakers for failing dependencies.

### Quick Reference Patterns

-   **Metrics**: `MetricsCollector` (Success rate, latency).
-   **Caching**: `LRUCache`, `TieredCache` (Memory -> Redis).
-   **Load Balancer**: `LeastLoaded`, `RoundRobin`.
-   **Decorator**: `@track_performance`, `@cached`.

// end-parallel

---

## Quality Assurance

### Common Anti-Patterns

| Anti-Pattern | Fix |
|--------------|-----|
| Premature Optimization | Measure first! |
| Unbounded Cache | Use LRU/TTL |
| Ignoring Tails | Optimize P99, not just Mean |
| Overloaded Agents | Implement Backpressure |

### Performance Checklist

- [ ] Metrics instrumentation active (P50/P95)
- [ ] Caching implemented for hot paths
- [ ] Load balancing strategy defined
- [ ] Concurrency limits set (Bulkheads)
- [ ] Alerts configured for degradation
