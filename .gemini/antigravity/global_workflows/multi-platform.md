---
description: Build and deploy features across web, mobile, and desktop
triggers:
- /multi-platform
- build multi platform feature
version: 2.0.0
allowed-tools: [Read, Task, Bash]
agents:
  primary: product-manager
  orchestrated: true
skills:
- mobile-development
- web-development
argument-hint: '[feature-name] [--platforms=web,ios,android]'
---

# Multi-Platform Orchestrator (v2.0)

// turbo-all

## Phase 1: Architecture (Sequential)

1.  **Shared Contract**
    - Define API Spec & Design System tokens.

## Phase 2: Implementation (Parallel)

// parallel

2.  **Web Implementation**
    - Agent: web-developer
    - Action: Implement React/Next.js feature.

3.  **Mobile Implementation**
    - Agent: mobile-developer
    - Action: Implement iOS/Android (SwiftUI/Kotlin/RN).

4.  **Desktop Implementation**
    - Agent: desktop-developer
    - Action: Implement Electron/Tauri feature.

// end-parallel

## Phase 3: Validation (Parallel)

// parallel

5.  **Cross-Platform Testing**
    - Verify functional parity.

6.  **Performance Check**
    - Web Vitals, App Startup time.

// end-parallel
