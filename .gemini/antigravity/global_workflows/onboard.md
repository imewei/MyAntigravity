---
description: Orchestrate team onboarding
triggers:
- /onboard
- orchestrate onboarding
version: 2.0.0
allowed-tools: [Read, Task, Bash]
agents:
  primary: engineering-manager
skills:
- team-management
argument-hint: '<role>'
---

# Onboarding Orchestrator (v2.0)

// turbo-all

## Phase 1: Logistics (Parallel)

// parallel

1.  **Access Provisioning**
    - Action: Checklist for Email/Slack/GitHub.

2.  **Hardware Setup**
    - Action: Laptop tracking/config.

3.  **Documentation Prep**
    - Action: Customize Welcome Packet.

// end-parallel

## Phase 2: Orientation (Sequential)

4.  **Day 1 Schedule**
    - Action: Generate agenda.

## Phase 3: Planning (Parallel)

// parallel

5.  **30-Day Goals**
    - Define quick wins.

6.  **60-Day Goals**
    - Define feature contributions.

7.  **90-Day Goals**
    - Define independence milestones.

// end-parallel
