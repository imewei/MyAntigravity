---
description: Merge all local branches into main and clean up
triggers:
- /merge-all
- workflow for merge all
version: 2.0.0
allowed-tools: [Bash(git:*), Read]
agents:
  primary: git-operator
argument-hint: '[--force]'
---

# Merge & Consolidate (v2.0)

// turbo-all

## Phase 1: Assessment (Sequential)

1.  **Status Check**
    - Action: Ensure on main/master, check other branches exist.

## Phase 2: Execution (Sequential/Iterative)

2.  **Merge Loop**
    - Action: Iterate through branches, `git merge --no-ff`.
    - Constraint: Stop on conflict.

## Phase 3: Cleanup (Turbo)

3.  **Delete Merged**
    - Action: `git branch -d <branch>`.
    - Note: Only delete if fully merged.

4.  **Report**
    - Summary of merged/deleted branches.
