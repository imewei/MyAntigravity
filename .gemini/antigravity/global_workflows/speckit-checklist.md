---
name: speckit-checklist
description: Generate a custom checklist for the current feature based on user requirements.
version: 2.2.2
agents:
  primary: qa-engineer
skills:
- checklist-generation
- requirement-validation
- static-analysis
allowed-tools: [Read, Write, Task, Bash]
triggers:
- keyword:speckit
- keyword:spec

---

# Checklist Generation Workflow

// turbo-all

# Checklist Generation Workflow

"Unit Testing for English Requirements"

---

## User Input

```text
$ARGUMENTS
```

---

## Phase 1: Context & Intent (Parallel)

// parallel

### Prerequisite Check
-   Run `.specify/scripts/bash/check-prerequisites.sh --json`
-   Parse `FEATURE_DIR`, `AVAILABLE_DOCS`.

### Intent Clarification (Dynamic)
-   Derive up to 3 context questions.
-   **Criteria**:
    -   Materially changes checklist content.
    -   Precision over breadth.
    -   *Example*: "Is this checking compliance or just UX?"

// end-parallel

**Interaction**: Ask questions if needed. Wait for user response.

---

## Phase 2: Context Loading

### Load Documents
-   `spec.md`: Scope & Requirements.
-   `plan.md`: Tech details.
-   `tasks.md`: Implementation steps.

**Strategy**: Progressive disclosure. Only load what is needed for the focus area.

---

## Phase 3: Checklist Generation

### Core Principle
**Test the Requirements, Not the Implementation.**

### Item Generation (Parallel)

// parallel

#### Dimension: Completeness
-   "Are all necessary requirements present?"

#### Dimension: Clarity
-   "Are requirements unambiguous and specific?"

#### Dimension: Coverage
-   "Are edge cases and failure modes defined?"

#### Dimension: Measurability
-   "Can this be objectively verified?"

// end-parallel

### Formatting
-   File: `FEATURE_DIR/checklists/[domain].md` (e.g., `ux.md`, `security.md`).
-   IDs: `CHK001` sequential.
-   Traceability: `[Spec §X.Y]` or `[Gap]`.

---

## Completion

Report:
-   Full path to created checklist.
-   Item count.
-   Summary of focus areas.
