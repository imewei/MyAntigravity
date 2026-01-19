---
description: Generate GitHub Actions CI/CD for Julia packages
triggers:
- /julia-package-ci
- generate julia ci/cd
version: 2.0.0
allowed-tools: [Read, Write, Bash]
agents:
  primary: devops-engineer
skills:
- julia-package-development
argument-hint: ''
---

# Julia CI Generator (v2.0)

// turbo-all

## Phase 1: Generation (Parallel)

// parallel

1.  **CI Workflow**
    - Action: Generate `.github/workflows/CI.yml` (Test, Coverage).

2.  **CompatHelper**
    - Action: Generate `.github/workflows/CompatHelper.yml`.

3.  **TagBot**
    - Action: Generate `.github/workflows/TagBot.yml`.

// end-parallel

## Phase 2: Deployment (Sequential)

4.  **Commit & Push**
    - Action: `git add .github && git commit -m "add ci" && git push`.
