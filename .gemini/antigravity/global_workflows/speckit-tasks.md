---
name: speckit-tasks
description: Generate an actionable, dependency-ordered tasks.md for the feature.
version: 2.2.2
agents:
  primary: tech-lead
skills:
- task-decomposition
- project-planning
- dependency-management
allowed-tools: [Read, Write, Task, Bash]
triggers:
- keyword:speckit
- keyword:spec

handoffs: 
  - label: Analyze Consistency
    agent: speckit.analyze
    prompt: Analyze the generated tasks for consistency.
    send: true
  - label: Start Implementation
    agent: speckit.implement
    prompt: Start implementing phase 1.
    send: true
---

# Task Generation Workflow

// turbo-all

# Task Generation Workflow

Turning plans into action.

---

## User Input

```text
$ARGUMENTS
```

---

## Phase 1: Context Extraction (Parallel)

// parallel

### Prerequisite Check
-   Run `.specify/scripts/bash/check-prerequisites.sh --json`.
-   Parse `FEATURE_DIR`.

### Design Extraction
-   **Spec**: User Stories (Priorities), Requirements.
-   **Plan**: Tech Stack, Structure, Phases.
-   **Data Model**: Entity to Story mapping.
-   **Contracts**: Endpoint to Story mapping.

// end-parallel

---

## Phase 2: Task Generation Strategy

**Rules**:
1.  **Story-Centric**: Organize by User Story (Phase 3+).
2.  **Checklist Format**: `- [ ] [T###] [P?] [US#] Description path/to/file`.
3.  **No Hallucination**: Only use stacks/files from Plan.

### Phase Structure
-   **Phase 1 Setup**: Init, Shared Infra.
-   **Phase 2 Foundations**: Blocking prereqs.
-   **Phase 3+ Stories**: Priority order.
-   **Final**: Polish.

---

## Phase 3: File Generation

**Action**: Write to `tasks.md`.
-   Use template structure.
-   Fill Implementation Strategy.
-   Generate Dependency Graph.

---

## Completion

Report:
-   Task Count (Total & Per Story).
-   Parallel opportunities.
-   MVP Scope recommendation.
