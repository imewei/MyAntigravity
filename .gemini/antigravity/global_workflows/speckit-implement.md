---
name: speckit-implement
description: Execute the implementation plan by processing tasks.md.
version: 2.0.0
agents:
  primary: fullstack-developer
skills:
- code-implementation
- tdd-cycle
- dependency-management
- task-execution
allowed-tools: [Read, Write, Task, Bash]
---

# Feature Implementation Workflow

// turbo-all

# Feature Implementation Workflow

Building it right, step by step.

---

## User Input

```text
$ARGUMENTS
```

---

## Phase 1: Pre-Flight Checks (Parallel)

// parallel

### Environment Scan
-   Run `.specify/scripts/bash/check-prerequisites.sh --json --require-tasks`.
-   Parse `FEATURE_DIR`.

### Checklist Validation
-   Scan `FEATURE_DIR/checklists/`.
-   **Gate**: All items MUST be complete. If incomplete -> HALT (Ask user).

// end-parallel

---

## Phase 2: Context Loading

**Load Strategy**:
-   `tasks.md`: Execution plan (REQUIRED).
-   `plan.md`: Architecture (REQUIRED).
-   `data-model.md`: Entities (IF EXISTS).
-   `contracts/`: APIs (IF EXISTS).

---

## Phase 3: Project Setup (Parallel)

// parallel

### Ignore Files
-   Verify `.gitignore`, `.dockerignore`, `.eslintignore`, etc.
-   Create based on tech stack (Node, Python, Rust, etc.).

### Task Parsing
-   Extract Phases: Setup, Tests, Core, Integration.
-   Identify Dependencies & Parallel `[P]` markers.

// end-parallel

---

## Phase 4: Execution Loop

**Rules**:
1.  **Phase-by-Phase**: Complete Setup -> Tests -> Core.
2.  **TDD**: Tests first.
3.  **Coordination**: Sequential vs Parallel ([P]).

### Implementation Strategy
-   **Setup**: Init structure, dependencies.
-   **Tests**: Write contract tests.
-   **Core**: Models, CLI, Endpoints.
-   **Polish**: Optimization, Docs.

### Error Handling
-   HALT on non-parallel failure.
-   Continue parallel success, report failure.

---

## Phase 5: Completion

1.  **Validation**: Verify all tasks `[X]`.
2.  **Match**: Impl matches Spec.
3.  **Report**: Final status summary.
