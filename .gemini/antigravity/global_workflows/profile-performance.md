---
description: Comprehensive performance profiling
triggers:
- /profile-performance
- profile performance
version: 2.0.0
allowed-tools: [Read, Task, Bash]
agents:
  primary: performance-engineer
skills:
- systems-profiling
argument-hint: '<binary-or-script>'
---

# Performance Profiler (v2.0)

// turbo-all

## Phase 1: Build (Sequential)

1.  **Compilation**
    - Action: Build with debug symbols (`-g`).

## Phase 2: Profiling (Parallel)

// parallel

2.  **CPU Profile**
    - Action: `perf record` -> Flamegraph.

3.  **Memory Profile**
    - Action: `valgrind --tool=massif`.

4.  **Cache Profile**
    - Action: `perf stat -e cache-misses`.

// end-parallel

## Phase 3: Analysis (Sequential)

5.  **Bottleneck Identification**
    - Action: Identify hotspots (>5% CPU).

6.  **Report**
    - Action: Synthesize findings and recommendations.
