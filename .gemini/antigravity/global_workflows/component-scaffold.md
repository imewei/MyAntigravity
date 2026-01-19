---
description: Orchestrate production-ready React/React Native component generation
triggers:
- /component-scaffold
- orchestrate production ready react/react native
allowed-tools: [Write, Read, Task]
version: 2.0.0
agents:
  primary: frontend-developer
  conditional:
  - agent: multi-platform-mobile
    trigger: platform=native|universal
skills:
- react-component-patterns
- css-architecture
argument-hint: '[Name] [--platform=web|native] [--styling=...] [--tests]'
---

# React Component Generator (v2.0)

// turbo-all

## Phase 1: Specification (Sequential)

1. **Spec Parsing**
   - Action: Determine props, state, platform, and styling strategy.

## Phase 2: File Generation (Parallel)

// parallel

2. **Logic & Markup**
   - Action: Generate `{Name}.tsx` (Functional, Hook-based, Accessible).

3. **Type Definitions**
   - Action: Generate `{Name}.types.ts` or inline interface.

4. **Styling**
   - Action: Generate styles (CSS Module / Styled Comp / StyleSheet).

5. **Tests**
   - Action: Generate `{Name}.test.tsx` (RTL / Jest).
   - Constraint: 80% coverage target.

6. **Documentation**
   - Action: Generate `{Name}.stories.tsx` (Storybook).

// end-parallel

## Phase 3: Verification (Sequential)

7. **Integrity Check**
   - Action: TSC Check (`tsc --noEmit`).
   - Action: Lint check.
