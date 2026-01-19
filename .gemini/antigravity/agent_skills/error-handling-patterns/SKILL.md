---
name: error-handling-patterns
description: Resilient error strategies: Circut breaking, Retries, and Graceful degradation.
version: 2.0.0
agents:
  primary: backend-architect
skills:
- fault-tolerance
- exception-handling
- resilience-patterns
- reliability-engineering
allowed-tools: [Read, Write, Task, Bash]
---

# Error Handling Patterns

// turbo-all

# Error Handling Patterns

Building systems that bend but don't break.

---

## Strategy & Recovery (Parallel)

// parallel

### Error Types

| Type | Action | Example |
|------|--------|---------|
| **Transient** | Retry (Backoff) | Network timeout, 503 Service Unavailable. |
| **Logic/Data** | Fail Fast | 400 Bad Request, NullPointer. |
| **System** | Circuit Break | Database down, Third-party outage. |

### Resilience Patterns

-   **Retry**: Exponential Backoff (`sleep(2^n)`). Jitter is mandatory.
-   **Circuit Breaker**: Stop calling failing service. Fail fast -> Self-heal.
-   **Bulkhead**: Isolate pools (e.g., Reports thread pool vs API pool).
-   **Fallback**: Return cached data or default value.

// end-parallel

---

## Decision Framework

### Handling Hierarchy

1.  **Local**: Can I fix it here? (Retry).
2.  **Caller**: Return meaningful error (Result/Either type).
3.  **Global**: Middleware catches unhandled -> Logs -> 500 Response.
4.  **User**: Clean message ("Something went wrong") + Trace ID.

---

## Core Knowledge (Parallel)

// parallel

### Constitutional AI Principles

1.  **Observability (Target: 100%)**: Log stack traces + correlation IDs.
2.  **Security (Target: 100%)**: Never leak internal details (DB schema/IPs) to user.
3.  **Reliability (Target: 99%)**: Failures should be bounded.

### Quick Reference

-   **Go**: `if err != nil { return fmt.Errorf("context: %w", err) }`
-   **Rust**: `Result<T, E>` + `?` operator.
-   **Exceptions**: Define Custom Hierarchy (`AppError`, `NetworkError`).

// end-parallel

---

## Quality Assurance

### Common Antipatterns

| Antipattern | Fix |
|-------------|-----|
| Empty Catch | `try { } catch (e) {}`. NEVER DO THIS. |
| Catch All | `except Exception:`. Too broad. Catch specific. |
| Log & Throw | Just throw. Let top-level log. (Avoid duplicates). |
| 500 for Everything | Map exceptions to Status Codes (404, 403, 400). |

### Error Checklist

- [ ] Custom Exception Classes defined
- [ ] Global Error Handler (Middleware)
- [ ] Retries with Jitter implemented
- [ ] Circuit Breaker for external deps
- [ ] Fallbacks defined for critical paths
- [ ] Logs contain Trace IDs
