---
name: numpyro-core-mastery
description: Detailed reference for NumPyro inference methods, guides, and diagnostics.
version: 2.0.0
agents:
  primary: numpyro-pro
skills:
- advanced-mcmc
- variational-inference-guides
- convergence-diagnostics
- arviz-integration
allowed-tools: [Read, Write, Task, Bash]
triggers:
- file:.ipynb
- keyword:ai
- keyword:ml
---

# NumPyro Core Mastery

// turbo-all

# NumPyro Core Mastery

Deep dive into NumPyro's inference engine, selecting correct samplers, designing variational guides, and interpreting diagnostics.

---

## Strategy & Validation (Parallel)

// parallel

### Delegation Strategy

| Delegate To | When |
|-------------|------|
| numpyro-pro | Standard inference tasks |
| nlsq-pro | MAP estimation (optimization only) |
| hpc-numerical-coordinator | Distributed Consensus MC |

### Pre-Response Validation Framework (5 Checks)

**MANDATORY before any response:**

1.  **Sampler**: NUTS (Default) vs HMC vs SVI (Large Data)?
2.  **Guide**: AutoNormal (Mean/Field) vs Multivariate (Correlated)?
3.  **Tuning**: `target_accept_prob` adjusted for divergences?
4.  **Diagnostics**: Plan MCMC summary check?
5.  **Data**: Vectorized inputs?

// end-parallel

---

## Decision Framework

### Chain-of-Thought Decision Framework

1.  **Data Scale**: <100k (MCMC) vs >100k (SVI/Subsampling).
2.  **Posterior Shape**: Simple (AutoNormal) vs Complex (NUTS).
3.  **Geometry**: Centered (Small data) vs Non-Centered (Hierarchical).
4.  **Convergence**: R-hat check. If fail -> Reparameterize.
5.  **Predictive**: Validation against observed data.
6.  **Production**: Saving `params` vs full traces.

---

## Core Knowledge (Parallel)

// parallel

### Constitutional AI Principles

1.  **Validity (Target: 100%)**: Correct statistical representation.
2.  **Efficiency (Target: 95%)**: Appropriate inference method for scale.
3.  **Diagnostics (Target: 100%)**: No ignored divergences.
4.  **Flexibility (Target: 90%)**: Custom guides where needed.
5.  **Reproducibility (Target: 100%)**: Deterministic PRNG usage.

### Quick Reference Patterns

-   **NUTS**: `MCMC(NUTS(model, target_accept_prob=0.8), ...)`
-   **SVI**: `SVI(model, AutoNormal(model), Adam(1e-3), Trace_ELBO())`
-   **Reparam**: `LocScaleReparam` for funnel issues.
-   **ArviZ**: `az.from_numpyro(mcmc)` for standard plots.

// end-parallel

---

## Quality Assurance

### Common Anti-Patterns

| Anti-Pattern | Fix |
|--------------|-----|
| MCMC on Big Data | Use SVI or HMCECS |
| Ignoring Divergences | Increase adapt / Reparam |
| Wrong Guide | Use Multivariate for correlations |
| Manual Gradients | Rely on JAX Autodiff |
| Looping Data | `numpyro.plate` |

### NumPyro Core Checklist

- [ ] Inference method matches data scale
- [ ] Guide selected appropriate for posterior complexity
- [ ] Convergence diagnostics planned (R-hat, ESS)
- [ ] Divergences addressed (Reparameterization)
- [ ] GPU acceleration enabled if available
- [ ] Posterior Predictive Checks implemented
- [ ] ArviZ Integration for visualization
