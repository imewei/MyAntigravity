---
description: Safe dependency upgrade orchestration with rollback protection
triggers:
- /deps-upgrade
- workflow for deps upgrade
version: 2.0.0
allowed-tools: [Bash, Edit, Read, Task]
agents:
  primary: devops-troubleshooter
skills:
- dependency-management
- regression-testing
argument-hint: '[--security-only] [--mode=quick|standard|deep]'
---

# Dependency Upgrade System (v2.0)

// turbo-all

## Phase 1: Analysis (Parallel)

// parallel

1.  **Audit State**
    - Action: Identify outdated packages: `npm outdated`, `pip list --outdated`.

2.  **Impact Analysis**
    - Action: Check for breaking changes (Major versions).

// end-parallel

## Phase 2: Strategy (Sequential)

3.  **Selection**
    - Security-only: Only patch vulnerabilities.
    - Standard: Patch + Minor updates.
    - Deep: Major updates (incremental).

4.  **Backup**
    - Action: Git checkpoint (`git commit -m "pre-upgrade"` or tag).

## Phase 3: Execution (Iterative)

**Constraint**: Upgrade one major version at a time. Run tests after EACH step.

5.  **Upgrade Command**
    - Action: `npm install lib@new`, `pip install --upgrade lib`.

6.  **Verification**
    - Action: `npm test`, build check.
    - Constraint: Rollback immediately on failure.

## Phase 4: Finalization

7.  **Lockfile Commit**
    - Action: Commit `package-lock.json` / `uv.lock`.
