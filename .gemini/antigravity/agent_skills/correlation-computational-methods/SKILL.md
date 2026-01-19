---
name: correlation-computational-methods
description: Implementation algorithms for correlation analysis (FFT, Multi-tau, Block averaging).
version: 2.0.0
agents:
  primary: correlation-function-expert
skills:
- algorithm-design
- numerical-analysis
- jax-optimization
- statistical-sampling
allowed-tools: [Read, Write, Task, Bash]
triggers:
- keyword:correlation-computational-methods
---

# Computational Methods (Correlation)

// turbo-all

# Computational Methods

Algorithms for efficient correlation calculation, ranging from FFT O(N log N) speedups to logarithmic multi-tau correlators and Bootstrap uncertainty estimation.

---

## Strategy & Validation (Parallel)

// parallel

### Delegation Strategy

| Delegate To | When |
|-------------|------|
| scientific-computing | JAX/GPU implementation specifics |
| correlation-function-expert | Physical interpretation of results |

### Pre-Response Validation Framework (5 Checks)

**MANDATORY before any response:**

1.  **Complexity**: Is O(N^2) avoided?
2.  **Memory**: Is block averaging used for N > 10^6?
3.  **Dynamic Range**: Linear (FFT) vs Log (Multi-tau)?
4.  **Hardware**: CPU vs GPU suitable?
5.  **Statistics**: Bootstrapping enabled?

// end-parallel

---

## Decision Framework

### Chain-of-Thought Decision Framework

1.  **Data Size**: < 1000 (Direct), > 1000 (FFT), > 1TB (Streaming).
2.  **Range**: Uniform sampling (FFT) vs Decades (Multi-tau).
3.  **Dimensionality**: 1D (Time) vs 3D (Spatial).
4.  **Platform**: NumPy (Standard) vs JAX (Accelerator).
5.  **Verification**: Convergence tests.

---

## Core Knowledge (Parallel)

// parallel

### Constitutional AI Principles

1.  **Efficiency (Target: 100%)**: Optimal algorithm complexity.
2.  **Accuracy (Target: 100%)**: Numerical precision maintained.
3.  **Robustness (Target: 95%)**: Outlier rejection.

### Quick Reference Patterns

-   **FFT Auto**: `ifft(abs(fft(x))**2)`.
-   **Multi-Tau**: Block averaging + cascaded buffers.
-   **Bootstrap**: Resample blocks with replacement.
-   **KD-Tree**: Spatial neighbor search < r_cut.

// end-parallel

---

## Quality Assurance

### Common Anti-Patterns

| Anti-Pattern | Fix |
|--------------|-----|
| Zero Padding Missing | Circular vs Linear convolution error |
| Memory Blowup | Chunking / Streaming methods |
| Bias in variance | Bessel correction (N-1) |
| Leaking Spectral Leakage | Windowing (e.g., Hamming) needed? |

### Computational Checklist

- [ ] FFT padding correct (2*N)
- [ ] Normalization factors (N-k) applied
- [ ] Memory limits respected
- [ ] Uncertainty quantification included
- [ ] Convergence with sample size verified
