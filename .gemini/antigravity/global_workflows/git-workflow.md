---
description: Complete Git workflow orchestration from code review through PR creation with parallel validation
triggers:
- /git-workflow
- /github
- workflow for git workflow
version: 2.2.1
allowed-tools: "[Bash(git:*), Read, Grep, Task]"
agents:
  primary: git-orchestrator
  conditional:
  - agent: quality-analyst
    trigger: phase-1
  - agent: test-automator
    trigger: phase-1
skills:
- git-advanced-workflows
- conventional-commits
- code-review-best-practices
argument-hint: '[target-branch] [--skip-tests] [--draft-pr] [--no-push] [--squash]'
---

# Git Workflow Orchestration (v2.0)

// turbo-all

## Phase 1: Review & Validation (Parallel Execution)

// parallel

1. **Static Analysis & style Check**
   - Agent: `quality-analyst`
   - Action: Review uncommitted changes for style and best practices.
   - Constraint: Fail if critical issues found.

2. **Test Execution**
   - Agent: `test-automator`
   - Action: Run relevant unit/integration tests.
   - Constraint: Skip if `--skip-tests` flag present.

3. **Breaking Change Detection**
   - Agent: `git-orchestrator`
   - Action: Analyze for API/schema compatibility issues.

// end-parallel

## Phase 2: Commit Preparation (Sequential)

4. **Conventional Commit Generation**
   - Skill: `conventional-commits`
   - Action: Generate `type(scope): subject` message.
   - Constraints: No AI attribution, strictly factual.

## Phase 3: Pre-Push (Sequential)

5. **Branch Validation**
   - Action: Ensure branch name follows `type/ticket-desc` pattern.
   - Action: Check for target branch conflicts.

## Phase 4: PR Creation (Sequential)

6.  **PR Metadata Generation**
    - Action: Generate title, body, labels, and reviewers.
    - Constraint: Use Draft mode if `--draft-pr`.

## Success Criteria

- Clean static analysis
- Tests passing (or explicitly skipped)
- Conventional Commit message
- Branch protection compliant
