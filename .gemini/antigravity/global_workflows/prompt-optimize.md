---
description: Optimize prompts for LLM performance
triggers:
- /prompt-optimize
- optimize prompt
version: 2.0.0
allowed-tools: [Read, Task, Bash]
agents:
  primary: prompt-engineer
skills:
- prompt-engineering
argument-hint: '<prompt>'
---

# Prompt Refiner (v2.0)

// turbo-all

## Phase 1: Analysis (Sequential)

1.  **Assessment**
    - Action: Check Clarity, Structure, Alignment.

## Phase 2: Enhancement (Parallel)

// parallel

2.  **Strategy: CoT**
    - Action: Add Chain of Thought instructions.

3.  **Strategy: Few-Shot**
    - Action: Generate examples.

4.  **Strategy: Structure**
    - Action: Apply xml/json formatting.

// end-parallel

## Phase 3: Validation (Parallel)

// parallel

5.  **Test Case A**
    - Action: Run baseline.

6.  **Test Case B**
    - Action: Run optimized.

// end-parallel

## Phase 4: Selection

7.  **Final Polish**
    - Action: Select best performing variation.
