---
name: nlsq-core-mastery
description: Detailed reference for NLSQ v0.6.4+ workflows, global optimization, and streaming.
version: 2.0.0
agents:
  primary: nlsq-pro
skills:
- nlsq-workflow-mastery
- cmaes-global-optimization
- massive-scale-fitting
- diagnostics-advanced
allowed-tools: [Read, Write, Task, Bash]
triggers:
- file:.ipynb
- keyword:ai
- keyword:ml
---

# NLSQ Core Mastery (v0.6.4+)

// turbo-all

# NLSQ Core Mastery

Deep dive into NLSQ v0.6.4+ features for complex fitting scenarios, optimization strategies, and massive dataset handling.

---

## Strategy & Validation (Parallel)

// parallel

### Delegation Strategy

| Delegate To | When |
|-------------|------|
| nlsq-pro | General fitting tasks |
| scientific-computing | Custom JAX kernel implementation |
| hpc-numerical-coordinator | Multi-node distributed fitting |

### Pre-Response Validation Framework (5 Checks)

**MANDATORY before any response:**

1.  **Complexity**: Is `auto_global` actually needed (>1000x scale diff)?
2.  **Bounds**: Are constraints physical? (e.g., sigma > 0).
3.  **Streaming**: Dataset > 16GB RAM?
4.  **Diagnostics**: Plan to check Covariance/Hessian?
5.  **JIT**: Model uses `lax.cond` or `jnp.where`?

// end-parallel

---

## Decision Framework

### Chain-of-Thought Decision Framework

1.  **Landscape Analysis**: Convex (Local) vs Rugged (Global).
2.  **Strategy Selection**: Levenberg-Marquardt (Standard) vs CMA-ES (Global).
3.  **Scale Handling**: In-memory vs Streaming (Generator).
4.  **Precision**: Float32 (Speed) vs Float64 (Accuracy).
5.  **Diagnostics**: Parameter Sloppiness (Eigenvalues).
6.  **Verification**: Residual Analysis.

---

## Core Knowledge (Parallel)

// parallel

### Constitutional AI Principles

1.  **Robustness (Target: 100%)**: Handle rank-deficient Hessians safely.
2.  **Scalability (Target: 100%)**: Zero OOM errors on large data.
3.  **Accuracy (Target: 95%)**: High precision result.
4.  **Global Search (Target: 90%)**: Avoid local minima traps.
5.  **Transparency (Target: 100%)**: Report convergence failure reasons.

### Quick Reference Patterns

-   **Workflow `auto`**: Default. Detects memory, picks chunking.
-   **Workflow `auto_global`**: Hybrid Multi-Start + CMA-ES. Needs bounds.
-   **Workflow `hpc`**: Checkpointing + Fault Tolerance.
-   **Streaming**: `strategy="streaming"` for out-of-core data.

// end-parallel

---

## Quality Assurance

### Common Anti-Patterns

| Anti-Pattern | Fix |
|--------------|-----|
| No Bounds in Global | Define `bounds=([min], [max])` |
| Python Control Flow | Use `jnp.where` |
| Ignoring Warnings | Check `stability="auto"` |
| Manual Chunking | Use `workflow="auto"` |
| Float Precision Mix | Ensure `x` and `y` match dtype |

### NLSQ Core Checklist

- [ ] Workflow aligned with problem scale/complexity
- [ ] Bounds enforced for global optimization
- [ ] JAX compatibility verified (No side effects)
- [ ] Memory limits respected (Streaming used if >RAM)
- [ ] Diagnostics enabled (Health Report)
- [ ] Loss function robust to outliers
- [ ] Initial guess `p0` provided (even for global)
