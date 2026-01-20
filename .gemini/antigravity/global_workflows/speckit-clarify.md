---
name: speckit-clarify
description: Identify underspecified areas in the feature spec by asking targeted clarification questions.
version: 2.0.0
agents:
  primary: system-architect
skills:
- ambiguity-detection
- requirement-analysis
- interactive-clarification
allowed-tools: [Read, Write, Task, Bash]
triggers:
- keyword:speckit
- keyword:spec

handoffs: 
  - label: Build Technical Plan
    agent: speckit.plan
    prompt: Create a plan for the spec. I am building with...
---

# Specification Clarification Workflow

// turbo-all

# Specification Clarification Workflow

Resolved ambiguities before they become bugs.

---

## User Input

```text
$ARGUMENTS
```

---

## Phase 1: Scan & Detect (Parallel)

// parallel

### Prerequisite Check
-   Run `.specify/scripts/bash/check-prerequisites.sh --json --paths-only`
-   Parse `FEATURE_DIR`, `FEATURE_SPEC`.

### Ambiguity Scan
Load spec and scan detection categories:
-   **Functional Scope**: Success criteria, out-of-scope.
-   **Data Model**: Entities, lifecycle.
-   **Interaction**: Journeys, errors, a11y.
-   **Non-Func**: Performance, Security, Scalability.

// end-parallel

---

## Phase 2: Prioritizations

**Selection Logic**:
1.  Max 5 candidate questions.
2.  Filter by Material Impact (Architecture, Model, UX).
3.  Exclude trivial or answered questions.
4.  Rank by `Impact * Uncertainty`.

---

## Phase 3: Interactive Questioning

**Loop**:
1.  **Present**: ONE question at a time.
2.  **Recommend**: Best practice option.
3.  **Format**: Table (Options A/B/C) or Short Answer.
4.  **Wait**: User response.
5.  **Record**: In-memory.

**Constraints**:
-   Max 5 questions total.
-   No future questions revealed.

---

## Phase 4: Integration (Atomic)

**After Each Answer**:
1.  **Append**: `## Clarifications` -> `### Session YYYY-MM-DD`.
2.  **Apply**: Update specific section (Functional, Data, NFR).
3.  **Save**: Write to `FEATURE_SPEC`.

---

## Completion

Report:
-   Questions asked/answered.
-   Updated `FEATURE_SPEC` path.
-   Coverage Summary (Resolved/Deferred/Clear).
