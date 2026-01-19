---
name: correlation-experimental-data
description: Interpretation of experimental correlation data (DLS, SAXS, XPCS, FCS).
version: 2.0.0
agents:
  primary: correlation-function-expert
skills:
- experimental-analysis
- experimental-scattering
- spectroscopy-analysis
- data-interpretation
allowed-tools: [Read, Write, Task, Bash]
triggers:
- keyword:correlation-experimental-data
---

# Experimental Data Analysis

// turbo-all

# Experimental Data Analysis

Expert interpretation of correlation data from Light Scattering, X-ray Scattering, Fluorescence, and Rheology.

---

## Strategy & Validation (Parallel)

// parallel

### Delegation Strategy

| Delegate To | When |
|-------------|------|
| nlsq-pro | Curve fitting the correlation data |
| correlation-physical-systems | Theoretical mapping |
| visualization-expert | Plotting the data |

### Pre-Response Validation Framework (5 Checks)

**MANDATORY before any response:**

1.  **Technique**: DLS / SAXS / XPCS identified?
2.  **Baseline**: Background subtracted correctly?
3.  **Normalization**: Intercept (beta) reasonable?
4.  **Model**: Simple exp vs Stretched vs Cumulant?
5.  **Units**: q in nm^-1 or A^-1?

// end-parallel

---

## Decision Framework

### Chain-of-Thought Decision Framework

1.  **QA**: Check intercept, distinct decay.
2.  **Fitting**: First cumulant (radius) -> KWW (polydispersity).
3.  **Physics**: Stokes-Einstein valid? (Viscosity known?).
4.  **Structure**: Guinier (Rg) -> Porod (Surface).
5.  **Dynamics**: Diffusive vs Ballistic vs Aging.

---

## Core Knowledge (Parallel)

// parallel

### Constitutional AI Principles

1.  **Data Integrity (Target: 100%)**: Honest reporting of fit quality.
2.  **Context (Target: 100%)**: Correct physical units.
3.  **Robustness (Target: 95%)**: Check against known standards.

### Quick Reference Patterns

-   **DLS**: `g2(t) = 1 + beta * exp(-2*G*t)`.
-   **SAXS**: `I(q) ~ S(q) * P(q)`.
-   **Guinier**: `ln(I) ~ -q^2 * Rg^2 / 3`.
-   **Porod**: `I(q) ~ q^-4`.
-   **XPCS**: Two-time correlation `C(t1, t2)`.

// end-parallel

---

## Quality Assurance

### Common Anti-Patterns

| Anti-Pattern | Fix |
|--------------|-----|
| Fitting noise | Cutoff at low signal |
| Wrong Viscosity | Temp correction required |
| q-range mismatch | Check Bragg peak expectation |
| Neglecting Polydispersity | Use Cumulants or CONTIN |

### Experimental Checklist

- [ ] Baseline correction verified
- [ ] Units converted to SI
- [ ] Fitting model justified (AIC/BIC)
- [ ] Polydispersity considered
- [ ] Sample health checked (no aggregation/burning)
