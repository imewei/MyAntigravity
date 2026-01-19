---
name: langchain-architecture
description: Patterns for Chains, Agents, Memory, and Tools using LangChain.
version: 2.0.0
agents:
  primary: ai-engineer
skills:
- chain-design
- agent-construction
- tool-integration
- memory-management
allowed-tools: [Read, Write, Task, Bash]
---

# LangChain Architecture

// turbo-all

# LangChain Architecture

Architecting robust LLM applications using the LangChain framework components.

---

## Strategy & Components (Parallel)

// parallel

### Core Components Selection

| Component | Use Case |
|-----------|----------|
| **Chains** | Deterministic sequences (A -> B -> C). |
| **Agents** | Non-deterministic, tool-using reasoning loops. |
| **Memory** | Persisting state across turns. |
| **Callbacks** | Observability, logging, streaming. |

### Memory Strategy

-   **Buffer**: Short interactions.
-   **Summary**: Long interactions (compress old turns).
-   **Vector**: Infinite recall (RAG-based memory).

// end-parallel

---

## Decision Framework

### Architecture Decision Tree

1.  **Is the workflow fixed?**
    *   Yes -> Use `SequentialChain` or LCEL Pipe `|`.
    *   No -> Use `Agent` (ReAct / OpenAI Functions).
2.  **Does it need external data?**
    *   Yes -> Add `Retriever` tool (RAG).
3.  **Does it need state?**
    *   Yes -> Add `RunnableWithMessageHistory`.

---

## Core Knowledge (Parallel)

// parallel

### Constitutional AI Principles

1.  **Modularity (Target: 100%)**: Separation of Prompts, Logic, and Tools.
2.  **Observability (Target: 100%)**: Tracing enabled (LangSmith/Callbacks).
3.  **Robustness (Target: 95%)**: Handle LLM parsing errors gracefully.

### Quick Reference Patterns

-   **RAG Chain**: `Retriever` | `Prompt` | `LLM`.
-   **Router**: Classify input -> Select specific Chain.
-   **Function Calling**: Bind tools -> LLM decides execution.
-   **Evaluation**: Use `LangSmith` to test chains.

// end-parallel

---

## Quality Assurance

### Common Pitfalls

| Pitfall | Fix |
|---------|-----|
| Global State | Use `RunnableConfig` for per-request state. |
| Hardcoded Prompts | Pull from hub or config files. |
| Unbounded Loops | Set `max_iterations` on Agents. |
| Silent Tool Failures | Raise exceptions or return error strings to LLM. |

### LangChain Checklist

- [ ] LCEL syntax used (modern standard)
- [ ] Tracing/Callbacks enabled
- [ ] Max iterations set for agents
- [ ] Tools have descriptions and schemas
- [ ] Memory bounded (Window/Summary)
- [ ] API keys managed securely
