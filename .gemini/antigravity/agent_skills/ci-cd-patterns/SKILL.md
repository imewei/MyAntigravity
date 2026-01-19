---
name: ci-cd-patterns
description: Julia CI/CD patterns for testing, documentation, and release automation.
version: 2.0.0
agents:
  primary: julia-pro
skills:
- julia-ci
- github-actions
- documentation-deploy
- coverage-tracking
allowed-tools: [Read, Write, Task, Bash]
---

# Julia CI/CD Patterns

// turbo-all

# Julia CI/CD Patterns

Standardized GitHub Actions workflows for the Julia ecosystem.

---

## Strategy & Workflows (Parallel)

// parallel

### Core Workflows

| Workflow | Trigger | Steps |
|----------|---------|-------|
| **Test** | Push/PR | `setup-julia`, `buildpkg`, `runtest`, `processcoverage`. |
| **Docs** | Push (Main) | `instantiate`, `make.jl`, `deploydocs`. |
| **Compat** | Schedule | `CompatHelper.jl` (Update Project.toml). |
| **Release** | Tag (v*) | `TagBot.jl` (Create GitHub Release). |

### Matrix Strategy

-   **OS**: `ubuntu-latest` (Primary), `windows-latest`, `macos-latest`.
-   **Versions**: `1` (Current Stable), `1.6` (LTS), `nightly` (Bleeding Edge).
-   **Arch**: `x64`, `x86` (Optional).

// end-parallel

---

## Decision Framework

### Pipeline logic

1.  **Build**: Is the package structure valid?
2.  **Test**: Do unit tests pass on all matrix os/versions?
3.  **Cover**: Is code coverage > 90%? (Upload to Codecov).
4.  **Docs**: If main branch, build and push to `gh-pages`.

---

## Core Knowledge (Parallel)

// parallel

### Constitutional AI Principles

1.  **Reproducibility (Target: 100%)**: Use `Manifest.toml` for docs/tests where possible.
2.  **Compatibility (Target: 100%)**: Test against LTS and Latest.
3.  **Visibility (Target: 100%)**: Badges for Status/Coverage in README.

### Quick Reference Actions

-   `julia-actions/setup-julia@v2`
-   `julia-actions/julia-buildpkg@v1`
-   `julia-actions/julia-runtest@v1`
-   `codecov/codecov-action@v4`

// end-parallel

---

## Quality Assurance

### Common Pitfalls

| Pitfall | Fix |
|---------|-----|
| Missing Tokens | Set `GITHUB_TOKEN` and `DOCUMENTER_KEY`. |
| Flaky Network | Use `retries` or caching. |
| Slow Tests | Use `[skip ci]` in commit msg for docs-only changes. |
| Registry Lag | `Pkg.Registry.update()` before install. |

### CI Checklist

- [ ] Test matrix covers LTS/Stable/Nightly
- [ ] Codecov integration active
- [ ] Documentation auto-deployment configured
- [ ] CompatHelper running weekly
- [ ] TagBot configured for releases
