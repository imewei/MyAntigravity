---
name: nlsq-core-mastery
version: "1.1.1"
description: Master NLSQ (v0.6.4+) for GPU-accelerated curve fitting. Covers the new 3-tier workflow system (auto, auto_global, hpc), global optimization (CMA-ES/Multi-Start), and streaming for massive datasets.
---

# NLSQ Core Mastery (v0.6.4+)

**Links**: [Source](https://github.com/imewei/NLSQ) | [PyPI](https://pypi.org/project/nlsq/) | [Documentation](https://nlsq.readthedocs.io/)

## Workflow Selection (New 3-Tier System)

Replace manual memory config with smart workflows:

| Workflow | Use Case | Bounds | Scaling |
|----------|----------|--------|---------|
| `auto` | Default. Local optimization, auto-memory management | Optional | Single-scale |
| `auto_global` | Multi-modal, unknown initial guess, multi-scale | Required | Multi-scale |
| `hpc` | Long-running jobs with checkpointing | Required | Multi-scale |

## Basic Fitting (`workflow="auto"`)

```python
from nlsq import fit
import jax.numpy as jnp

def model(x, params):
    A, b, c = params
    return A * jnp.exp(-b * x) + c

# "auto" detects memory limits and selects standard/chunked/streaming strategy
result = fit(model, x, y, p0=[5.0, 0.5, 1.0], workflow="auto")
```

## Global Optimization (`workflow="auto_global"`)

Auto-switches between Multi-Start and CMA-ES based on parameter scale ratio:

```python
# Required: bounds for global search
bounds = ([0.0, 0.0, 0.0], [10.0, 5.0, 10.0])

# Automatically selects:
# - Multi-Start: if params have similar scales (<1000x ratio)
# - CMA-ES: if params are multi-scale (>1000x ratio)
result = fit(model, x, y, p0=p0, bounds=bounds, workflow="auto_global")
```

## Manual Global Control

### CMA-ES (Complex Landscapes)
For difficult multi-modal problems or severe scaling issues:

```python
from nlsq.global_optimization import CMAESOptimizer, CMAESConfig

config = CMAESConfig(population_size=32, sigma0=0.5)
opt = CMAESOptimizer(config=config)
result = opt.fit(model, x, y, bounds=bounds)
```

### Multi-Start (Simple Basins)
For finding the best local minimum in well-behaved landscapes:

```python
result = fit(model, x, y, bounds=bounds, multistart=True, n_starts=20, sampler="sobol")
```

## Large Datasets (Streaming)

The `auto` workflow handles this automatically. For explicit control over 100M+ points:

```python
# Memory-safe fitting for datasets larger than RAM
result = fit(model, x_massive, y_massive, p0=p0, 
             workflow="auto", 
             strategy="streaming",  # Force streaming
             memory_limit_gb=16.0)
```

## Pitfalls & Best Practices

1.  **JCP (JAX-Compatible Code)**: Model must be pure JAX. Use `jnp.where` instead of Python `if`.
2.  **Bounds**: Optional for `auto`, but **MANDATORY** for `auto_global` and `hpc`.
3.  **Outliers**: Use `loss='huber'` (mild) or `loss='cauchy'` (severe) for robust fitting.
4.  **Type Safety**: Use `params` tuple/array in model signature, not `*args`.

## Diagnostics

```python
from nlsq.diagnostics import create_health_report

# Post-fit analysis
report = create_health_report(result)
print(report.summary())
# Checks: Condition number, parameter sensitivity (sloppiness), gradient health
```
