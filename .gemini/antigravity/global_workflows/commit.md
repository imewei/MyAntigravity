---
description: Intelligent git commit with automated analysis, quality validation, and atomic commit enforcement
triggers:
- /commit
- workflow for commit
version: 2.2.1
allowed-tools: "[Bash(git:*), Read, Grep]"
agents:
  primary: code-reviewer
  conditional:
  - agent: security-auditor
    trigger: files "*.env|secrets|credentials|keys"
argument-hint: '[commit-message] [--quick] [--split] [--amend] [--no-verify]'
color: green
---

# Smart Commit Ecosystem (v2.0)

// turbo-all

## Phase 1: Context & Validation (Parallel)

// parallel

1. **Git Context Extraction**
   - Action: `git status --porcelain=v2`, diff checks, log history.
   - Goal: snapshot current state.

2. **Security Scan**
   - Agent: `security-auditor`
   - Action: Scan staged files for secrets/keys.
   - Constraint: Fast fail on critical findings.

3. **Type/Scope Inference**
   - Action: Infer `type` (feat/fix/etc.) from file paths.
   - Action: Infer `scope` from directory names.

// end-parallel

## Phase 2: Analysis (Sequential)

4. **Atomicity Check**
   - Constraint: Recommend splitting if >300 lines or >10 files (unless docs/lockfiles).

5. **Message Generation**
   - Agent: `code-reviewer`
   - Action: Generate conventional commit message.
   - Constraint: No AI attribution. Technical tone.

## Phase 3: Execution (Sequential)

6. **Final Commit**
   - Action: Execute `git commit`.
   - Option: Use `--amend` if requested and safe.
