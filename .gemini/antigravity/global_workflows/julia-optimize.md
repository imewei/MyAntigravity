---
description: Profile and optimize Julia code
triggers:
- /julia-optimize
- profile julia code
version: 2.0.0
allowed-tools: [Read, Task, Bash]
agents:
  primary: julia-scientific-programmer
skills:
- julia-performance
argument-hint: '<file-path>'
---

# Julia Optimizer (v2.0)

// turbo-all

## Phase 1: Baseline (Sequential)

1.  **Benchmark**
    - Action: Run `@benchmark` to capture current state.

## Phase 2: Analysis (Parallel)

// parallel

2.  **Type Stability**
    - Action: Run `@code_warntype`. Identify red flags.

3.  **Allocation Profile**
    - Action: Run allocation profiler.

// end-parallel

## Phase 3: Optimization (Iterative)

4.  **Apply Fixes**
    - Priority: Type Stability -> Allocations -> Algorithmic.

## Phase 4: Verification

5.  **Compare**
    - Action: Re-run benchmark. Calculate speedup.
