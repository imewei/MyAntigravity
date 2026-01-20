---
description: Analyze and refactor code to improve quality using SOLID principles and design patterns
triggers:
- /refactor-clean
- workflow for refactor clean
allowed-tools: [Read, Task, Bash]
version: 2.2.2
agents:
  primary: code-architect
  conditional:
  - agent: static-analyst
    trigger: phase-1
skills:
- refactoring-patterns
- solid-principles
- clean-code-metrics
argument-hint: '[target-file-or-dir] [--quick|standard|comprehensive]'
---

# Refactor and Clean Code (v2.2.2)

// turbo-all

## Phase 1: Deep Analysis (Sequential)

1. **Metric Collection**
   - Agent: `static-analyst`
   - Action: Calculate Cyclomatic Complexity, method length, and class coupling.
   - Output: `analysis_report.json`

2. **Smell Detection**
   - Agent: `code-architect`
   - Action: Identify code smells (Long Method, God Class, Feature Envy).

## Phase 2: Refactoring Strategy (Parallel Planning)

// parallel

3. **Structure Optimization**
   - Goal: Reduce complexity and improve modularity.
   - Strategy: Extract Method, Extract Class.

4. **SOLID Alignment**
   - Goal: Enforce SRP, OCP, LSP, ISP, DIP.
   - Strategy: Dependency Injection, Interface Segregation.

5. **Naming & Cleanup**
   - Goal: Improve readability.
   - Strategy: Rename variables, remove dead code, simplify conditionals.

// end-parallel

## Phase 3: Execution (Sequential & Iterative)

**Constraint**: Apply one refactoring at a time. Run tests after EVERY change.

6. **Quick Fixes** (Quick Mode)
   - Action: Rename, Dead Code Removal, Constant Extraction.

7. **Architectural Changes** (Standard/Comprehensive)
   - Action: Extract Classes, Apply Design Patterns (Factory, Strategy).

## Phase 4: Verification (Parallel)

// parallel

8. **Test Suite**
   - Action: Validate no functional regressions.

9. **Metric Verification**
   - Action: Confirm complexity reduction and coverage maintenance.

// end-parallel
