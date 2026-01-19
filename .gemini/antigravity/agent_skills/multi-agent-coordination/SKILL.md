---
name: multi-agent-coordination
description: Implementation patterns for agent communication, task allocation, and synchronization.
version: 2.0.0
agents:
  primary: multi-agent-orchestrator
skills:
- inter-agent-messaging
- task-scheduling
- capability-matching
- dag-execution
allowed-tools: [Read, Write, Task, Bash]
---

# Multi-Agent Coordination

// turbo-all

# Multi-Agent Coordination

Implementation of the protocols, messaging, and scheduling logic that powers multi-agent systems.

---

## Strategy & Validation (Parallel)

// parallel

### Delegation Strategy

| Delegate To | When |
|-------------|------|
| multi-agent-orchestrator | High-level strategy decisions |
| agent-performance-optimization | Load balancing logic |

### Pre-Response Validation Framework (5 Checks)

**MANDATORY before any response:**

1.  **Protocol**: Async vs Sync communication?
2.  **Format**: JSON/Protobuf message strictness?
3.  **Queue**: Is a broker needed (RabbitMQ/Redis)?
4.  **Idempotency**: Can tasks be retried safely?
5.  **Timeout**: Are deadlines set?

// end-parallel

---

## Decision Framework

### Chain-of-Thought Decision Framework

1.  **Pattern**: Pub/Sub vs Request/Response.
2.  **Discovery**: How do agents find each other? (Registry).
3.  **Allocation**: Round-robin vs Capability-based vs Least-loaded.
4.  **Synchronization**: Wait for all (Barrier) vs First result?
5.  **State**: Shared DB vs Message Passing.

---

## Core Knowledge (Parallel)

// parallel

### Constitutional AI Principles

1.  **Decoupling (Target: 100%)**: Agents shouldn't know internal details of others.
2.  **Reliability (Target: 95%)**: Message delivery guarantees.
3.  **Scalability (Target: 90%)**: Add agents without code changes.

### Quick Reference Patterns

-   **Registry**: `find_capable_agents(required_caps)`.
-   **Broker**: `publish(topic, msg)`, `subscribe(topic)`.
-   **DAG Engine**: Topological sort -> Execute ready -> Mark complete.
-   **Exponential Backoff**: `sleep(2**attempt)`.

// end-parallel

---

## Quality Assurance

### Common Anti-Patterns

| Anti-Pattern | Fix |
|--------------|-----|
| Hardcoded Agents | Use Dynamic Registry |
| Blocking Calls | Use Async/Await everywhere |
| Infinite Retries | Max retry count + Dead Letter Queue |
| Global Mutable State | Pass state in messages |

### Coordination Checklist

- [ ] Agent registry active
- [ ] Message schema defined
- [ ] Async execution model used
- [ ] Retry logic implemented
- [ ] Dead Letter Queue for failures
- [ ] Task ID correlation enabled
