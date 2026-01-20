---
name: speckit-constitution
description: Create or update the project constitution from principle inputs.
version: 2.0.0
agents:
  primary: product-manager
skills:
- governance
- compliance
- principles-design
allowed-tools: [Read, Write, Task, Bash]
triggers:
- keyword:speckit
- keyword:spec

handoffs: 
  - label: Build Specification
    agent: speckit.specify
    prompt: Implement the feature specification based on the updated constitution.
---

# Constitution Management Workflow

// turbo-all

# Constitution Management Workflow

Codifying the rules of engagement.

---

## User Input

```text
$ARGUMENTS
```

---

## Phase 1: Analysis & Collection (Parallel)

// parallel

### Template Analysis
-   Load `.specify/memory/constitution.md` template.
-   Identify `[ALL_CAPS_IDENTIFIER]` placeholders.

### Value Collection
-   Input: `$ARGUMENTS`.
-   Context: README, Docs.
-   Governance: `RATIFICATION_DATE` (original or today), `CONSTITUTION_VERSION` (Semantic).

// end-parallel

---

## Phase 2: Drafting

**Action**: Replace placeholders with concrete test.
-   **Principles**: Succinct name, non-negotiable rules.
-   **Governance**: Amendment procedure, versioning.

---

## Phase 3: Consistency Propagation (Parallel)

// parallel

### Plan Alignment
-   Check `.specify/templates/plan-template.md`.

### Spec Alignment
-   Check `.specify/templates/spec-template.md`.

### Task Alignment
-   Check `.specify/templates/tasks-template.md`.

### Command Alignment
-   Check `.specify/templates/commands/*.md`.

// end-parallel

---

## Phase 4: Impact Report & Finalize

1.  **Generate Report**: Version change, modified principles, pending updates.
2.  **Validate**: No placeholders left.
3.  **Write**: Overwrite `.specify/memory/constitution.md`.
4.  **Summary**: New version, commit message suggestion.
