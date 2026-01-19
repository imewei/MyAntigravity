---
description: Orchestrate end-to-end feature development with parallel architecture, security, and full-stack implementation streams
triggers:
- /feature-development
- workflow for feature development
allowed-tools: [Read, Task, Bash, Create]
version: 2.0.0
agents:
  primary: comprehensive-review
  conditional:
  - agent: security-auditor
    trigger: manual-dispatch
  - agent: backend-development
    trigger: parallel-stream
  - agent: frontend-mobile-development
    trigger: parallel-stream
skills:
- architect-review-framework-migration
- security-auditor-full-stack-orchestration
- backend-architect-multi-platform-apps
- e2e-testing-patterns
argument-hint: '[--complexity=simple|medium|complex] [--mode=quick|standard] [--parallel]'
---

# Feature Development Orchestrator (v2.0)

// turbo-all

## Phase 1: Foundation (Sequential)

1. **Architecture & Design Review**
   - Agent: `comprehensive-review`
   - Action: Analyze requirements and define the technical architecture.
   - Output: `architecture_decision_record.md`

2. **Security Pre-Assessment**
   - Agent: `security-auditor`
   - Action: Identify potential risks and compliance requirements.
   - Output: `security_risk_matrix.md`

## Phase 2: Implementation (Parallel Streams)

// parallel

3. **Backend Implementation**
   - Agent: `backend-development`
   - Skill: `backend-architect-multi-platform-apps`
   - Action: Implement core services, APIs, and business logic.
   - Constraint: Must adhere to API contracts defined in Phase 1.

4. **Frontend Implementation**
   - Agent: `frontend-mobile-development`
   - Skill: `ui-ux-designer`
   - Action: Implement UI components and state management.
   - Constraint: Mock APIs until backend is ready.

// end-parallel

## Phase 3: Validation & Polish (Parallel Streams)

// parallel

5. **Test Automation**
   - Agent: `test-automator`
   - Skill: `e2e-testing-patterns`
   - Action: Implement unit, integration, and E2E tests.

6. **Security Validation**
   - Agent: `security-auditor`
   - Action: Perform SAST/DAST and dependency scanning.

7. **Performance Tuning**
   - Agent: `agent-performance-optimization`
   - Action: Optimize queries, bundle sizes, and latency.

// end-parallel

## Phase 4: Delivery

8. **Deployment Pipeline**
   - Agent: `devops-troubleshooter`
   - Action: Configure CI/CD and deployment gates.

9. **Documentation & Handoff**
   - Agent: `tutorial-engineer`
   - Action: Finalize API docs and user guides.
