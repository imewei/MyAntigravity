---
name: speckit-analyze
description: Perform a cross-artifact consistency analysis across spec.md, plan.md, and tasks.md.
version: 2.0.0
agents:
  primary: tech-lead
skills:
- static-analysis
- consistency-check
- constitution-compliance
allowed-tools: [Read, Write, Task, Bash]
---

# Specification & Plan Analysis Workflow

// turbo-all

# Specification & Plan Analysis Workflow

Ensuring coherence before code.

---

## User Input

```text
$ARGUMENTS
```

## Constraints
-   **READ-ONLY**: Do not modify files.
-   **Constitution**: Non-negotiable source of truth.

---

## Phase 1: Context Loading (Parallel)

// parallel

### Initialize
-   Run `.specify/scripts/bash/check-prerequisites.sh --json --require-tasks`.
-   Parse `FEATURE_DIR`.

### Load Artifacts
-   `spec.md`: Req inventory.
-   `plan.md`: Arch context.
-   `tasks.md`: Execution map.
-   `constitution.md`: Rules.

// end-parallel

---

## Phase 2: Analysis Passes (Parallel)

// parallel

### Duplication & Consistency
-   Detect near-duplicate requirements.
-   Flag terminology drift.
-   Check entity mismatches (Spec vs Plan).

### Ambiguity & Underspecification
-   Flag vague terms ("fast", "robust").
-   Flag placeholders (TODOs).
-   Detect verbs missing objects.

### Coverage & Alignment
-   **Gaps**: Req with 0 tasks.
-   **Constitution**: Rule violations (CRITICAL).
-   **Ordering**: Dependency contradictions.

// end-parallel

---

## Phase 3: Reporting

### Report Structure
-   **Findings Table**: ID, Category, Severity, Location, Summary.
-   **Coverage**: Req Key -> Task IDs.
-   **Metrics**: Total Reqs, Tasks, Ambiguity Count.

### Next Actions
-   **CRITICAL**: Must resolve before implementation.
-   **Remediation**: Offer concrete suggestions (optional).
