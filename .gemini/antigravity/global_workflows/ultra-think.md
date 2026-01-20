---
description: Advanced structured reasoning engine using sequential thinking for complex problem solving
triggers:
- /ultra-think
- advanced structured reasoning
allowed-tools: [SequentialThinking, Read, Task]
version: 2.2.1
agents:
  primary: structured-reasoning
skills:
- structured-reasoning
- research-intelligence
argument-hint: '[problem-statement] [--mode=quick|standard|deep] [--depth=int]'
---

# Ultra-Think Engine (v2.0)

// turbo-all

## 1. Parameter Interpretation

Parse the user arguments to configure the `sequential-thinking` session:

- **Mode Mapping**:
  - `quick` → `totalThoughts: 5`
  - `standard` (default) → `totalThoughts: 15`
  - `deep` → `totalThoughts: 30`
  - `ultradeep` → `totalThoughts: 50+`

## 2. Execution Strategy

Use the `mcp_sequential-thinking_sequentialthinking` from the `sequential-thinking` server.

DO NOT output raw thought traces to the user. Instead, execute the thinking process internally and present the final synthesized output.

### Thought Structure (Internal)

1.  **Decomposition**: Break the problem into atomic constituents.
2.  **Hypothesis Generation**: Formulate multiple competing hypotheses.
3.  **Analysis**:
    - **First Principles**: Validating base assumptions.
    - **Systems Thinking**: Mapping feedback loops.
    - **Adversarial Review**: Self-critiquing the hypothesis.
4.  **Synthesis**: Converging on the optimal solution.

## 3. Usage Pattern

```python
# Example Tool Call
sequential_thinking(
    thought="Analyzing root cause...",
    thoughtNumber=1,
    totalThoughts=15,  # Based on mode
    nextThoughtNeeded=True
)
```

## 4. Output Format

Present the final result in the requested format (Executive Summary, Detailed Report, or Code), but include a "Reasoning Summary" section that briefly encapsulates the key insights from the thought chain.
