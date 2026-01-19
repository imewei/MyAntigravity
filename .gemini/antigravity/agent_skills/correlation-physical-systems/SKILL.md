---
name: correlation-physical-systems
description: Mapping correlation functions to Condensed Matter, Soft Matter, and Bio-systems.
version: 2.0.0
agents:
  primary: correlation-function-expert
skills:
- condensed-matter-physics
- soft-matter-physics
- biophysics
- non-equilibrium-thermodynamics
allowed-tools: [Read, Write, Task, Bash]
triggers:
- keyword:correlation-physical-systems
---

# Physical Systems Correlations

// turbo-all

# Physical Systems

Mapping abstract correlation functions to concrete physical phenomena in Condensed Matter, Soft Matter, Biology, and Non-Equilibrium systems.

---

## Strategy & Validation (Parallel)

// parallel

### Delegation Strategy

| Delegate To | When |
|-------------|------|
| active-matter | Non-equilibrium / Swarms |
| md-simulation-setup | Simulating the system |
| correlation-math-foundations | Theoretical derivation |

### Pre-Response Validation Framework (5 Checks)

**MANDATORY before any response:**

1.  **System**: Hard/Soft/Bio/Active?
2.  **State**: Equilibrium vs Driven?
3.  **Scale**: Quantum vs Classical?
4.  **Criticality**: Near phase transition?
5.  **Ornstein-Zernike**: Appropriate closure?

// end-parallel

---

## Decision Framework

### Chain-of-Thought Decision Framework

1.  **Identify Field**: Spin (Ising), Particle (LJ), Polymer (Rouse)?
2.  **Select Function**: Spin-Spin, Density-Density, End-to-End.
3.  **Predict Behavior**: Exponential decay vs Power law (Critical).
4.  **Compare**: Simulation vs Experiment.
5.  **Extract Info**: Correlation length, exponents.

---

## Core Knowledge (Parallel)

// parallel

### Constitutional AI Principles

1.  **Accuracy (Target: 100%)**: Correct critical exponents.
2.  **Generality (Target: 95%)**: Universality classes.
3.  **Specificity (Target: 95%)**: System-specific details (e.g., hydrodynamics).

### Quick Reference Patterns

-   **Critical**: `C(r) ~ r^-(d-2+eta)`.
-   **Polymer**: `R^2 ~ t^0.5` (Rouse), `t^0.66` (Zimm).
-   **Glass**: `chi4(t)` peak at `tau_alpha`.
-   **Active**: Enanced diffusion `D_eff`.

// end-parallel

---

## Quality Assurance

### Common Anti-Patterns

| Anti-Pattern | Fix |
|--------------|-----|
| Mean Field near Critical | Use renormalization group exponents |
| Ignoring Hydrodynamics | Zimm instead of Rouse for dilute |
| Short-time diffusion | Check ballistic regime |
| Finite Size | Finite Size Scaling analysis |

### Physics Checklist

- [ ] Universality class identified
- [ ] Hydrodynamic effects considered
- [ ] Equilibrium state verified (FDT holds?)
- [ ] Critical exponents checked
- [ ] Finite size effects analyzed
