---
description: Create production-ready LangChain agents
triggers:
- /langchain-agent
- create langchain agent
version: 2.0.0
allowed-tools: [Read, Task, Bash]
agents:
  primary: ai-engineer
skills:
- langchain-development
argument-hint: '<agent-description>'
---

# LangChain Architect (v2.0)

// turbo-all

## Phase 1: Setup (Parallel)

// parallel

1.  **Model Config**
    - Action: Init Claude/Voyage.

2.  **Tool Definition**
    - Action: Create async tool definitions.

3.  **Memory Store**
    - Action: Setup Redis/Postgres checkpointer.

// end-parallel

## Phase 2: Graph Construction (Sequential)

4.  **State Graph**
    - Action: Define Nodes and Edges (LangGraph).

## Phase 3: Validation

5.  **Trace Check**
    - Action: Verify LangSmith traces.
