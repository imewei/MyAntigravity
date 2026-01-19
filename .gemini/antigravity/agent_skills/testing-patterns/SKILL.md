---
name: testing-patterns
description: Julia-specific testing strategies using Test.jl, Aqua.jl, and JET.jl.
version: 2.0.0
agents:
  primary: julia-pro
skills:
- julia-testing
- static-analysis
- quality-assurance
- package-validation
allowed-tools: [Read, Write, Task, Bash]
triggers:
- file:.jl
- keyword:julia
- keyword:qa
- keyword:testing
- project:Project.toml
---

# Julia Testing Patterns

// turbo-all

# Julia Testing Patterns

Ensuring correctness and stability in the Julia ecosystem.

---

## Strategy & Suites (Parallel)

// parallel

### Test Organization

| Component | Tool | Purpose |
|-----------|------|---------|
| **Unit** | `Test.jl` | Correctness of functions. |
| **Quality** | `Aqua.jl` | Ambiguities, unbound args, stale deps. |
| **Static** | `JET.jl` | Type stability and optimization passes. |
| **Doctest** | `Documenter.jl` | Verify examples in docstrings. |

### Suite Structure

-   `test/runtests.jl`: Entry point.
-   `@testset "Feature"`: Logical grouping.
-   `@test_throws`: Error handling checks.

// end-parallel

---

## Decision Framework

### Testing Hierarchy

1.  **Fast**: Unit tests in `test/`. Run on every save.
2.  **Strict**: `JET.report_package()` to catch type instability.
3.  **Clean**: `Aqua.test_all(MyPackage)` to catch rot.
4.  **Slow**: Integration/Simulation tests (CI only).

---

## Core Knowledge (Parallel)

// parallel

### Constitutional AI Principles

1.  **Stability (Target: 100%)**: No "MethodError" at runtime. (Use JET).
2.  **Privacy (Target: 100%)**: Test data must be synthetic.
3.  **Coverage (Target: 90%)**: Critical math functions tested.

### Quick Reference

-   `@test x ≈ y` (Approx equality for floats).
-   `@testset "Name" begin ... end`.
-   `Pkg.test()`.

// end-parallel

---

## Quality Assurance

### Common Pitfalls

| Pitfall | Fix |
|---------|-----|
| Running tests global | Use `@testset` to isolate scope. |
| Ignoring Type Instability | Code will be slow. Fix red warnings in JET. |
| Flaky Tests | Avoid `rand()` without `Random.seed!`. |
| Missing Docs | `Documenter.doctest` ensures examples work. |

### Checklist

- [ ] `runtests.jl` includes all sub-files
- [ ] Aqua.jl checks enabled (Ambiguity/Stale/Compat)
- [ ] JET.jl static analysis passing (opt/call)
- [ ] Doctests passing
- [ ] CI Matrix for Julia Versions (LTS/Stable)
