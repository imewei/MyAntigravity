---
name: structured-reasoning
description: Apply systematic cognitive frameworks (First Principles, Root Cause, Decision Analysis) to complex problems.
version: 2.0.0
agents:
  primary: ai-engineer
skills:
- systematic-analysis
- decision-matrices
- root-cause-analysis
- hypothesis-testing
allowed-tools: [Read, Write, Task, Bash]
---

# Structured Reasoning

// turbo-all

# Structured Reasoning

Systematic problem-solving engine using explicit reasoning chains, auditability, and multiple mental models.

---

## Strategy & Frameworks (Parallel)

// parallel

### Framework Selection Strategy

| Framework | Use Case |
|-----------|----------|
| **First Principles** | Novel/Fundamental problems. "Is this physically impossible?" |
| **Root Cause (RCA)** | Debugging/Incidents. "Why did this fail?" (5 Whys) |
| **Decision Analysis** | Trade-offs/Selection. "React vs Vue?" (Weighted Matrix) |
| **Systems Thinking** | Distributes Systems/Side-Effects. "What breaks if I change X?" |

### Six-Phase Process

1.  **Understand**: Define constraints & success criteria.
2.  **Approach**: Select framework (above).
3.  **Analyze**: Branching exploration (Hypothesis A vs B).
4.  **Synthesize**: Converge on best solution.
5.  **Validate**: Challenge assumptions (Red Teaming).
6.  **Finalize**: Generate action plan.

// end-parallel

---

## Decision Framework

### Chain-of-Thought Structure

```yaml
thought:
  id: "T1.1"
  stage: "analysis"
  content: "If we use Redis, we introduce a new dependency."
  evidence: "Infra docs show no current Redis cluster."
  confidence: 0.9
  status: "active"
```

### Branching Strategy

-   **Explore**: `T1.1` -> `T1.1.1` (Option A) vs `T1.1.2` (Option B).
-   **Revise**: `T1.1` -> `T1.1-REV` (New evidence invalidates T1.1).
-   **Prune**: Mark paths as `dead-end` if constraints violated.

---

## Core Knowledge (Parallel)

// parallel

### Constitutional AI Principles

1.  **Rigor (Target: 100%)**: No conclusion without evidence/reasoning.
2.  **Breadth (Target: 90%)**: Explore at least 2 alternatives for major decisions.
3.  **Auditability (Target: 100%)**: Reasoning trail must be reconstructible.
4.  **Intellectual Honesty (Target: 100%)**: Explicitly state low confidence.

### Quick Reference Patterns

-   **Hypothesis Testing**: Hypothesis -> Test -> Validate/Reject.
-   **Trade-off Matrix**: Row=Option, Col=Criteria, Cell=Score.
-   **Premortem**: "Assume it failed. Why?"
-   **Inversion**: "How would I guarantee failure?" (Avoid those).

// end-parallel

---

## Quality Assurance

### Common Pitfalls

| Pitfall | Fix |
|---------|-----|
| Jumping to Solution | Force Phase 1 (Understanding) & 2 (Approach) |
| Confirmation Bias | Actively search for disconfirming evidence |
| False Dilemma | Look for Option C (Synthesis) |
| Hidden Assumptions | "Assuming network is reliable..." -> Make explicit |

### Reasoning Checklist

- [ ] Framework selected explicitly
- [ ] Constraints defined clearly
- [ ] At least 2 branches explored (for complex tasks)
- [ ] Assumptions listed and challenged
- [ ] Confidence score assigned
- [ ] Contradictions resolved
