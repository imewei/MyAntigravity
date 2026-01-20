---
name: speckit-plan
description: Execute implementation planning workflow to generate design artifacts.
version: 2.2.0
agents:
  primary: system-architect
skills:
- system-design
- technical-planning
- research-intelligence
allowed-tools: [Read, Write, Task, Bash]
triggers:
- keyword:speckit
- keyword:spec

handoffs: 
  - label: Create Tasks
    agent: speckit.tasks
    prompt: Break the plan into tasks
    send: true
  - label: Create Checklist
    agent: speckit.checklist
    prompt: Create a checklist for the following domain...
---

# Implementation Plan Workflow

// turbo-all

# Implementation Plan Workflow

From requirements to engineering blueprint.

---

## User Input

```text
$ARGUMENTS
```

---

## Phase 1: Context Loading & Setup (Parallel)

// parallel

### Environment Setup
-   Run `.specify/scripts/bash/setup-plan.sh --json`
-   Parse `FEATURE_SPEC`, `IMPL_PLAN`, `SPECS_DIR`, `BRANCH`.

### Knowledge Loading
-   Read `FEATURE_SPEC`.
-   Read `.specify/memory/constitution.md`.
-   Load `IMPL_PLAN` template.

// end-parallel

---

## Phase 2: Design & Research (Iterative)

### Step 1: Technical Analysis
-   Fill "Technical Context" in plan.
-   Identify Unknowns -> Mark [NEEDS CLARIFICATION].

### Step 2: Research Execution (Parallel)

// parallel

#### Task: Unknown Resolution
-   Research specific unknowns.
-   Update `research.md`.

#### Task: Best Practices
-   Identify domain-specific patterns.
-   Verify against Constitution.

// end-parallel

### Step 3: Design Artifacts
1.  **Data Model**: Extract entities -> `data-model.md`.
2.  **Contracts**: Generate API schemas -> `/contracts/`.
3.  **Quickstart**: Create/Update `quickstart.md`.

---

## Phase 3: Validation & Context Update

### Step 1: Constitution Check
-   Verify design against `constitution.md`.
-   **Gate**: ERROR if violations found.

### Step 2: Agent Context
-   Run `.specify/scripts/bash/update-agent-context.sh codex`
-   Update agent-specific memory files.

---

## Completion

Report:
-   Branch
-   `IMPL_PLAN` path
-   Generated artifacts list.
