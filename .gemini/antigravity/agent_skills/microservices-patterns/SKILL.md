---
name: microservices-patterns
description: Design microservices with proper boundaries, events, and resilience.
version: 2.0.0
agents:
  primary: backend-architect
skills:
- distributed-systems
- resilience-engineering
- event-driven-architecture
allowed-tools: [Read, Write, Task, Bash]
triggers:
- keyword:microservice
- keyword:saga
- keyword:cqrs
---

# Skill: Microservices Patterns (v2.0)

// turbo-all

# Microservices Patterns

A comprehensive guide to service boundaries, communication, data management, and resilience patterns in distributed systems.



## Core Patterns (Parallel)

// parallel

### Decomposition Strategies

| Strategy | Approach | Example |
|----------|----------|---------|
| **Business Capability** | Organize by what business does | OrderSvc, BillingSvc |
| **DDD Subdomain** | Bounded Contexts | Core, User, Inventory |
| **Strangler Fig** | Incremental Migration | Intercept -> Divert |

### Communication Patterns

| Type | Mechanism | Use Case |
|------|-----------|----------|
| **Synchronous** | REST, gRPC | Interactive, Read-heavy |
| **Asynchronous** | Kafka, SQS | Decoupled, Write-heavy |
| **Event-Driven** | Pub/Sub | Choreography, Notification |

### Data Management

| Pattern | Description |
|---------|-------------|
| **Database per Service** | Strict isolation, no sharing |
| **CQRS** | Separate Read/Write models |
| **Event Sourcing** | State as sequence of events |
| **Saga** | Distributed Transactions (Orchestration/Choreography) |

// end-parallel

---

## Resilience & Reliability (Parallel)

// parallel

### Circuit Breaker
Prevents cascading failures by stopping calls to failing services.
- **States**: Closed (Normal), Open (Failing), Half-Open (Testing).
- **Libraries**: Hystrix, Resilience4j, Tenacity (Python).

### Retry w/ Backoff
Transient failure handling.
- **Parameters**: Max attempts, Exponential backoff, Jitter.
- **Warning**: Can cause thundering herd if not jittered.

### Bulkhead
Isolating resources to prevent total system crash.
- **Method**: Separate thread pools/semaphores per dependency.

// end-parallel

---

## Implementation Reference

### Saga Pattern (Orchestration)
```python
class OrderSaga:
    def execute(self, order):
        try:
            inventory.reserve(order)
            payment.charge(order)
            shipping.schedule(order)
        except Exception:
            self.compensate(order)

    def compensate(self, order):
        # Undo logic (Refund, Release, Cancel)
        pass
```

### Event-Driven Bus
```python
# Publisher
event_bus.publish("OrderCreated", payload={"id": 123})

# Subscriber
@event_bus.subscribe("OrderCreated")
def on_order_created(event):
    email_service.send_confirmation(event.id)
```

---

## Anti-Patterns & Fixes

| Pattern | Fix |
|---------|-----|
| **Distributed Monolith** | Loosen coupling, interface contracts |
| **Shared Database** | Split schema, use APIs |
| **Chatty Services** | API Gateway, Aggregation, Batching |
| **Silent Failure** | Dead Letter Queues (DLQ), Alerting |

## Checklist

- [ ] Boundaries aligned with Domain
- [ ] Database per Service enforcement
- [ ] Async communication default
- [ ] Circuit Breakers on all external calls
- [ ] Idempotency keys used
- [ ] Distributed Tracing (OpenTelemetry)
- [ ] Health Checks (Liveness/Readiness)
- [ ] Secrets Management
