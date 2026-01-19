---
description: Scaffolding for Scientific Machine Learning
triggers:
- /sciml-setup
- setup sciml project
version: 2.0.0
allowed-tools: [Read, Task, Bash]
agents:
  primary: sciml-specialist
skills:
- julia-development
- machine-learning-ops
argument-hint: '<problem-description>'
---

# SciML Scaffolder (v2.0)

// turbo-all

## Phase 1: Analysis (Sequential)

1.  **Problem Detection**
    - Action: Identify ODE/PDE/SDE/Optimization from input.

## Phase 2: Implementation (Parallel)

// parallel

2.  **Solver Setup**
    - Action: Select and configure solver (DifferentialEquations.jl).

3.  **Visualization**
    - Action: Generate plotting code (Plots.jl).

4.  **Optimzation Loop**
    - Action: Generate optimization boilerplate (DiffEqFlux.jl).

// end-parallel

## Phase 3: Verification

5.  **Template Validation**
    - Action: Ensure code is runnable.
