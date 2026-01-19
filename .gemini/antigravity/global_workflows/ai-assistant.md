---
description: Build production-ready AI assistants
triggers:
- /ai-assistant
- build ai assistant
version: 2.0.0
allowed-tools: [Read, Task, Bash]
agents:
  primary: ai-architect
skills:
- llm-application-dev
- conversation-design
argument-hint: '[assistant-description]'
---

# AI Assistant Builder (v2.0)

// turbo-all

## Phase 1: Architecture (Parallel)

// parallel

1.  **NLU Definition**
    - Action: Define Intents and Entities.

2.  **Flow Design**
    - Action: Define Dialog States and Transitions.

3.  **Persona Design**
    - Action: Define Tone, Voice, and Constraints.

// end-parallel

## Phase 2: Implementation (Sequential)

4.  **System Prompting**
    - Action: Draft primary system prompt.

5.  **Tool Registration**
    - Action: Define function schemas.

## Phase 3: Validation

6.  **Test Cases**
    - Action: Run conversation simulation.
