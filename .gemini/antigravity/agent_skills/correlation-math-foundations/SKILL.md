---
name: correlation-math-foundations
description: Mathematical theory of correlation functions, transforms, and fluctuation-dissipation.
version: 2.0.0
agents:
  primary: correlation-function-expert
skills:
- mathematical-physics
- signal-processing-theory
- statistical-mechanics-theory
allowed-tools: [Read, Write, Task, Bash]
triggers:
- keyword:correlation-math-foundations
---

# Correlation Math Foundations

// turbo-all

# Correlation Math Foundations

The theoretical backbone for correlation analysis: Wiener-Khinchin, Ornstein-Zernike, Fluctuation-Dissipation, and Green's Functions.

---

## Strategy & Validation (Parallel)

// parallel

### Delegation Strategy

| Delegate To | When |
|-------------|------|
| correlation-computational-methods | Implementing the math |
| correlation-physical-systems | Applying to verified physics |

### Pre-Response Validation Framework (5 Checks)

**MANDATORY before any response:**

1.  **Definition**: Autocorrelation vs Cross-correlation correctly formatted?
2.  **Domain**: Frequency (Spectrum) vs Time?
3.  **Stationarity**: Is the process WSS (Wide-Sense Stationary)?
4.  **Normalization**: Variance or Unity?
5.  **Limits**: t=0 and t->inf behavior correct?

// end-parallel

---

## Decision Framework

### Chain-of-Thought Decision Framework

1.  **Identify Quantity**: Scalar, Vector, or Tensor correlation?
2.  **Select Transform**: Fourier (Steady state) vs Laplace (Relaxation).
3.  **Apply Theorem**: Wiener-Khinchin (Spectrum), FDT (Response).
4.  **Check Constraints**: Kramers-Kronig, Sum Rules.
5.  **Derive Relation**: Microscopic -> Macroscopic.

---

## Core Knowledge (Parallel)

// parallel

### Constitutional AI Principles

1.  **Exactness (Target: 100%)**: Correct mathematical definitions.
2.  **Rigor (Target: 100%)**: Theoretical constraints satisfied.
3.  **Clarity (Target: 95%)**: Notation consistency.

### Quick Reference Patterns

-   **Wiener-Khinchin**: S(w) = FFT(C(t)).
-   **FDT**: Chi''(w) ~ w * S(w).
-   **Ornstein-Zernike**: h(k) = c(k) / (1 - rho*c(k)).
-   **Diffusion**: <r^2> ~ 2dDt.

// end-parallel

---

## Quality Assurance

### Common Anti-Patterns

| Anti-Pattern | Fix |
|--------------|-----|
| Confusing S(k) and S(q) | 2pi factor consistency |
| Ignoring imaginary parts | Check Kramers-Kronig |
| Non-stationary assumptions | Check aging / two-time |
| Divergent integrals | Regularization criteria |

### Math Logic Checklist

- [ ] Definitions consistent (Conjugate variables)
- [ ] Normalization factors tracked (1/N, 1/V)
- [ ] Sum rules satisfied
- [ ] Causality enforced
- [ ] Symmetry properties verified
