---
description: Systematically fix broken imports across the codebase
triggers:
- /fix-imports
- workflow for fix imports
version: 2.0.0
allowed-tools: [Bash, Read, Write, Edit, Grep]
agents:
  primary: code-quality
skills:
- refactoring-patterns
- typescript-pro
- python-pro
argument-hint: '[path] [--quick|standard]'
---

# Import Resolution Engine (v2.0)

// turbo-all

## Phase 1: Detection (Parallel)

// parallel

1.  **Compiler Checks**
    - Action: `tsc --noEmit` (TS), `mypy` (Python), `cargo check` (Rust).
    - Goal: Precise error locations.

2.  **Grep Scan**
    - Action: Scan for suspicious patterns (e.g., relative paths `../../` that broke).

// end-parallel

## Phase 2: Planning (Sequential)

3.  **Resolution Strategy**
    - Match broken path to current file existence.
    - Logic: Exact match > Fuzzy Name > Export Symbol.

## Phase 3: Application (Iterative)

4.  **Apply Fixes**
    - Constraint: Fix one file or cluster at a time.

5.  **Verify**
    - Action: Re-run compiler check after batch.

## Phase 4: Final Validation

6.  **Circular Dependency Check**
    - Action: `madge` (JS/TS) or equivalent.
