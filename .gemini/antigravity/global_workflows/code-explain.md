---
description: Detailed code explanation with visual aids and domain expertise
triggers:
- /code-explain
- detailed code explanation
version: 2.0.0
allowed-tools: [Read, Glob, Grep, Task]
agents:
  primary: code-tutor
  conditional:
  - agent: scientific-tutor
    trigger: domain "scientific|math|physics"
skills:
- code-comprehension
- technical-writing
argument-hint: '<code-path>'
---

# Code Explanation Engine (v2.0)

// turbo-all

## Phase 1: Comprehension (Parallel)

// parallel

1.  **Structure Analysis**
    - Action: Identify classes, functions, imports.

2.  **Complexity Scan**
    - Action: detailed complexity metrics.

3.  **Concept Extraction**
    - Action: Tag key patterns (Decorator, Async, Singleton).

// end-parallel

## Phase 2: Explanation Generation (Sequential)

4.  **High-Level Overview**
    - Summary of purpose and architecture.

5.  **Detailed Breakdown**
    - Step-by-step logic flow.

6.  **Visuals**
    - Generate Mermaid diagrams (Flowchart/Sequence).

## Phase 3: Contextualization

7.  **Pitfalls & Best Practices**
    - Highlight potential issues (Performance, Safety).

8.  **Domain Context**
    - Explain specific logic (e.g., Scientific formulas).
