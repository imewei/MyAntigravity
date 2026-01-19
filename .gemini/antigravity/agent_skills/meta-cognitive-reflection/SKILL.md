---
name: meta-cognitive-reflection
description: Analyze reasoning patterns, detect biases, and improve AI decision quality.
version: 2.0.0
agents:
  primary: ai-engineer
skills:
- bias-detection
- reasoning-audit
- self-correction
- continuous-improvement
allowed-tools: [Read, Write, Task, Bash]
triggers:
- file:.ipynb
- keyword:ai
- keyword:ml
- keyword:qa
- keyword:testing
---

# Meta-Cognitive Reflection

// turbo-all

# Meta-Cognitive Reflection

The "Watcher" skill: observing the observer to ensure reasoning quality, bias mitigation, and continuous improvement.

---

## Strategy & Analysis (Parallel)

// parallel

### Cognitive Bias Detection

| Bias | Detection Trigger | Mitigation |
|------|-------------------|------------|
| **Confirmation** | Seeking only supporting data | "What would disprove this?" |
| **Anchoring** | Stuck on first solution | "Reset context. Approach afresh." |
| **Availability** | Recency bias | "Check historical patterns." |
| **Sunk Cost** | "We already wrote usage code" | "Is the code actually good?" |

### Communication Assessment

-   **Clarity**: Jargon level appropriate?
-   **Conciseness**: Signal-to-noise ratio?
-   **Structure**: Logical flow?
-   **Tone**: collaborative vs authoritative?

// end-parallel

---

## Decision Framework

### Reflection Workflow

1.  **Pause**: Stop execution.
2.  **Audit**: Review last N thoughts/actions.
3.  **Classify**: Identify reasoning type (Deductive, Inductive, Abductive).
4.  **Critique**: Apply "Six Thinking Hats" (Black Hat = Critique).
5.  **Adjust**: Create correction plan if quality is low.
6.  **Resume**: Continue execution with improved context.

---

## Core Knowledge (Parallel)

// parallel

### Constitutional AI Principles

1.  **Objective Truth (Target: 100%)**: Prefer facts over hallucinations/guesses.
2.  **Self-Correction (Target: 100%)**: Admit mistakes immediately.
3.  **Growth (Target: N/A)**: Learn from every error interaction.

### Quick Reference Patterns

-   **Reasoning Types**:
    -   *Deductive*: Rule -> Case (Sure).
    -   *Inductive*: Cases -> Rule (Probable).
    -   *Abductive*: Effect -> Most Likely Cause (Diagnostic).

// end-parallel

---

## Quality Assurance

### Session Report Template

```markdown
## Meta-Reflection Report
**Reasoning Quality**: 4/5
**Biases Detected**: Anchoring (Initial solution).
**Correction**: Shifted to Option B after finding complexity.
**Key Lesson**: Always check `package.json` before assuming dependencies.
```

### Reflection Checklist

- [ ] Stopped to reflect?
- [ ] Bias check complete?
- [ ] Reasoning valid/logical?
- [ ] Tone appropriate?
- [ ] Actionable improvements identified?
