---
description: Bootstrap new Julia package with proper structure, testing, and CI/CD
triggers:
- /julia-scaffold
- bootstrap new julia package
allowed-tools: [Read, Task, Bash]
version: 2.0.0
agents:
  primary: julia-scientific-programmer
skills:
- julia-package-development
argument-hint: '<PackageName>'
---

# Julia Package Bootstrap (v2.0)

// turbo-all

## Phase 1: PkgTemplates Execution (Sequential)

1. **Template Generation**
   - Action: Execute `PkgTemplates` script to scaffold base structure.
   - Goal: `src/`, `test/`, `Project.toml`, `.github/`.

## Phase 2: Customization (Parallel)

// parallel

2. **Module Logic**
   - Action: Populate `src/PackageName.jl` with exports and includes.
   - Action: Create `src/types.jl`, `src/functions.jl`.

3. **Test Infrastructure**
   - Action: Setup `test/runtests.jl` with `Test` and `SafeTestsets`.

4. **Documentation**
   - Action: Configure `Documenter.jl` in `docs/make.jl`.

// end-parallel

## Phase 3: Verification (Sequential)

5. **Pkg Dev Setup**
   - Action: `Pkg.activate(".")`, `Pkg.instantiate()`, `Pkg.precompile()`.
   - Action: `Pkg.test()` verification.
