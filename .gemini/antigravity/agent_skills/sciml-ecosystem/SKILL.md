---
name: sciml-ecosystem
description: Overview and selection guide for the Julia SciML ecosystem packages.
version: 2.0.0
agents:
  primary: sciml-pro
skills:
- sciml-package-selection
- ecosystem-navigation
- integration-patterns
allowed-tools: [Read, Write, Task, Bash]
triggers:
- file:.ipynb
- file:.jl
- keyword:ai
- keyword:julia
- keyword:ml
- project:Project.toml
---

# SciML Ecosystem

// turbo-all

# SciML Ecosystem Guide

Strategic guide for navigating and selecting the right tools from the Julia Scientific Machine Learning ecosystem.

---

## Strategy & Validation (Parallel)

// parallel

### Delegation Strategy

| Delegate To | When |
|-------------|------|
| sciml-pro | Implementation using selected packages |
| julia-developer | Package configuration/compat |
| julia-pro | Non-SciML Julia tasks |

### Pre-Response Validation Framework (5 Checks)

**MANDATORY before any response:**

1.  **Goal**: Simulation vs Optimization vs Learning?
2.  **Domain**: ODE/PDE/SDE/Jump?
3.  **Complexity**: Direct definition vs Symbolic (MTK)?
4.  **Performance**: Native Julia vs External libs?
5.  **Integration**: Do packages compose?

// end-parallel

---

## Decision Framework

### Chain-of-Thought Decision Framework

1.  **Identify Problem**: Is it a differential equation? Optimization?
2.  **Select Core**: `DifferentialEquations.jl` (Solver) or `Optimization.jl`.
3.  **Select Modeling**: `ModelingToolkit.jl` (Symbolic) or `Catalyst.jl` (Reactions).
4.  **Select Learning**: `SciMLSensitivity.jl` (Adjoints) + `Lux.jl`.
5.  **Select Utilities**: `Plots.jl`, `BenchmarkTools.jl`.
6.  **Verify**: Check compatibility and documentation.

---

## Core Knowledge (Parallel)

// parallel

### Constitutional AI Principles

1.  **Appropriateness (Target: 100%)**: Right tool for the job.
2.  **Composability (Target: 100%)**: Leverage ecosystem integration.
3.  **Performance (Target: 95%)**: Native Julia speed.
4.  **Standardization (Target: 90%)**: Follow SciML common interface.
5.  **Documentation (Target: 90%)**: Reference official docs.

### Quick Reference Patterns

-   **Solver Core**: `DifferentialEquations.jl`
-   **Symbolic Layer**: `ModelingToolkit.jl`
-   **Reaction Nets**: `Catalyst.jl`
-   **PINNs**: `NeuralPDE.jl`
-   **Parameter Fit**: `Optimization.jl` + `SciMLSensitivity.jl`

// end-parallel

---

## Quality Assurance

### Common Anti-Patterns

| Anti-Pattern | Fix |
|--------------|-----|
| Using JuMP for SciML | Use `Optimization.jl` |
| Manual gradients | Use `Zygote` / `SciMLSensitivity` |
| Reinventing Solvers | Use `DifferentialEquations.jl` |
| Hardcoded Physics | Use `ModelingToolkit.jl` |

### SciML Ecosystem Checklist

- [ ] Core problem identified
- [ ] Primary package selected correctly
- [ ] Symbolic needs assessed (MTK)
- [ ] Integration points identified
- [ ] Performance requirements met
- [ ] Compatibility checked
