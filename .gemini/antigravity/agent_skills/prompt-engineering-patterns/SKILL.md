---
name: prompt-engineering-patterns
description: Advanced techniques for Chain-of-Thought, Few-Shot, and Self-Consistency.
version: 2.2.1
agents:
  primary: prompt-engineer
skills:
- chain-of-thought
- few-shot-learning
- prompt-optimization
- output-structuring
allowed-tools: [Read, Write, Task, Bash]
triggers:
- file:.ipynb
- keyword:ai
- keyword:ml
---

# Prompt Engineering Patterns

// turbo-all

# Prompt Engineering Patterns

Design patterns for unlocking reasoning, consistency, and reliability in Large Language Models.

---

## Strategy & Techniques (Parallel)

// parallel

### Core Techniques

| Technique | Trigger | Benefit |
|-----------|---------|---------|
| **Zero-Shot CoT** | "Let's think step by step" | Improves math/logic. |
| **Few-Shot** | Provide 3-5 examples | Enforces format/style. |
| **Self-Consistency** | Generate 5, take majority vote | Reduces hallucination. |
| **Role Prompting** | "You are an expert X" | Sets semantic priors. |

### Structure Anatomy

1.  **Role**: "You are..."
2.  **Context**: "Here is the data..."
3.  **Task**: "Extract the..."
4.  **Constraints**: "Do not..."
5.  **Output Format**: "Return JSON..."

// end-parallel

---

## Decision Framework

### Optimization Workflow

1.  **Baseline**: Write simple instruction. Test.
2.  **Iterate**:
    *   *Wrong Format?* -> Add Few-Shot examples.
    *   *Bad Reasoning?* -> Add CoT ("Think step by step").
    *   *Hallucination?* -> Add "Answer only from context".
3.  **Refine**: Compress tokens, clarify ambiguity.
4.  **Finalize**: Version control the prompt.

---

## Core Knowledge (Parallel)

// parallel

### Constitutional AI Principles

1.  **Clarity (Target: 100%)**: No ambiguous instructions.
2.  **Safety (Target: 100%)**: Prompt injection defenses (delimiters).
3.  **Efficiency (Target: 90%)**: Minimal tokens for maximum performance.

### Quick Reference Patterns

-   **Delimiters**: Use `"""` or `###` to separate instruction from data.
-   **Negative Constraints**: "Do NOT..." (Use sparingly, often weaker than DO).
-   **Persona**: "Act as..." or "Simulate..."
-   **Meta-Prompting**: Ask LLM to improve the prompt.

// end-parallel

---

## Quality Assurance

### Common Pitfalls

| Pitfall | Fix |
|---------|-----|
| Context Window Overflow | Truncate input data smartly. |
| Vague Instructions | Be incredibly specific. "Summarize" -> "Summarize in 3 bullets". |
| Example Bias | Ensure few-shot examples are balanced/diverse. |
| Prompt Drift | Re-eval prompts when changing models. |

### Prompt Checklist

- [ ] Role defined
- [ ] Task instructions clear
- [ ] Input data clearly delimited
- [ ] Output format specified (JSON/XML)
- [ ] Few-shot examples (if needed)
- [ ] Negative constraints checks
- [ ] Tested against edge cases
