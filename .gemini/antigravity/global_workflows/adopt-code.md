---
description: Analyze and modernize scientific computing codebases for adoption
triggers:
- /adopt-code
- analyze and modernize scientific
version: 2.0.0
allowed-tools: [Read, Task, Bash, Grep]
agents:
  primary: scientific-developer
skills:
- scientific-computing
- code-analysis
argument-hint: '<path-to-code> [target-framework]'
---

# Scientific Code Adoption (v2.0)

// turbo-all

## Phase 1: Discovery (Parallel)

// parallel

1.  **Inventory Scan**
    - Action: `find . -type f` to count LOC by extension.
    - Goal: Identify primary languages (Fortran, C++, Python).

2.  **Dependency Map**
    - Action: Grep for libraries (BLAS, LAPACK, MPI, CUDA).

3.  **Build Analysis**
    - Action: Check for Makefiles, CMakeLists, setup.py.

// end-parallel

## Phase 2: Architecture Analysis (Sequential)

4.  **Kernel Identification**
    - Identify computational hotspots (O(N^2) loops, solvers).

5.  **Precision Check**
    - Determine float32 vs float64 requirements.

## Phase 3: Modernization Strategy

6.  **Target Selection**
    - Recommend JAX/Julia/Modern C++ based on use case.

7.  **Plan Generation**
    - Output migration roadmap (Wrapper vs Rewrite).
