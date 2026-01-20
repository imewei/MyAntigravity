---
name: speckit-specify
description: Create or update the feature specification from a natural language feature description.
version: 2.2.0
agents:
  primary: product-manager
skills:
- spec-writing
- requirement-analysis
- checklist-validation
allowed-tools: [Read, Write, Task, Bash]
triggers:
- keyword:spec
- keyword:specify
- file:.specify/spec.md
handoffs: 
  - label: Build Technical Plan
    agent: speckit.plan
    prompt: Create a plan for the spec. I am building with...
  - label: Clarify Spec Requirements
    agent: speckit.clarify
    prompt: Clarify specification requirements
    send: true
---

# Feature Specification Workflow

// turbo-all

# Feature Specification Workflow

Turning ideas into structured requirements.

---

## User Input

```text
$ARGUMENTS
```

You **MUST** consider the user input before proceeding (if not empty).

---

## Phase 1: Branch Setup (Parallel)

// parallel

### Context Analysis
The text the user typed after `/speckit.specify` is the feature description.

### Branch Name Generation
1.  **Analyze**: Extract keywords.
2.  **Generate**: 2-4 word short name (action-noun, e.g., "add-user-auth").
3.  **Format**: Lowercase, dashed.

### Feasibility Check
-   Fetch remotes & prune: `git fetch --all --prune`
-   Check for existing branches with same short-name.
-   Determine next version number (N+1).

// end-parallel

---

## Phase 2: Feature Branch Creation

**Action**: Run creation script.
```bash
.specify/scripts/bash/create-new-feature.sh --json "$ARGUMENTS" --number <N> --short-name "<short-name>"
```

**IMPORTANT**:
-   Run ONCE.
-   Parse JSON output for `BRANCH_NAME` and `SPEC_FILE`.

---

## Phase 3: Specification Drafting

### Step 1: Parse & Analyze
1.  **Inputs**: User description, `$ARGUMENTS`.
2.  **Extract**: Actors, Actions, Data, Constraints.
3.  **Gap Analysis**: Identify up to 3 [NEEDS CLARIFICATION] items.

### Step 2: Write Spec (Parallel)
// parallel

#### Section: Functional Requirements
-   Must be testable.
-   Use reasonable defaults (Assumptions).

#### Section: Success Criteria
-   Metric-driven (Time, count, %).
-   Tech-agnostic.

#### Section: Scenarios
-   User flows covered.
-   Edge cases identified.

// end-parallel

**Action**: Write to `SPEC_FILE` using `.specify/templates/spec-template.md`.

---

## Phase 4: Quality Validation

### Step 1: Generate Checklist
Create `FEATURE_DIR/checklists/requirements.md` using standard template.

### Step 2: Validate Spec
1.  **Run Check**: Verify all items.
2.  **Refine**: Fix non-clarification issues immediately.
3.  **Clarify**: If [NEEDS CLARIFICATION] remains, prompt user (max 3 Qs).

### Step 3: Update Checklist
Mark pass/fail.

---

## Completion

Report completion with:
-   Branch Name
-   Spec File Path
-   Checklist Status
-   Ready for `/speckit.plan`.
