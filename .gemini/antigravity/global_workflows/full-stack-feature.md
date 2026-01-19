---
description: Orchestrate end-to-end full-stack feature development
triggers:
- /full-stack-feature
- orchestrate full stack feature
version: 2.0.0
allowed-tools: [Bash, Read, Write, Edit, Task]
agents:
  primary: fullstack-developer
  orchestrated: true
skills:
- web-development
- api-design
- db-design
argument-hint: '[feature-name] [--stack=...]'
---

# Full-Stack Feature Orchestrator (v2.0)

// turbo-all

## Phase 1: Design (Sequential)

1.  **Schema & Contract**
    - Define DB Schema (SQL/NoSQL).
    - Define API Contract (OpenAPI/GraphQL).

## Phase 2: Implementation (Parallel)

// parallel

2.  **Backend Implementation**
    - Agent: backend-developer
    - Action: Implement API endpoints, Business Logic.

3.  **Frontend Implementation**
    - Agent: frontend-developer
    - Action: Implement UI Components, State Management.

4.  **Database Migration**
    - Action: Create and run migration scripts.

// end-parallel

## Phase 3: Verification (Parallel)

// parallel

5.  **Backend Tests**
    - Action: Unit/Integration tests.

6.  **Frontend Tests**
    - Action: Component/E2E tests.

// end-parallel

## Phase 4: Integration

7.  **Final Check**
    - Verify End-to-End flow.
