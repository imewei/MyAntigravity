---
name: javascript-testing-patterns
description: Vitest, Jest, and Testing Library patterns for modern JS/TS apps.
version: 2.0.0
agents:
  primary: test-automator
skills:
- frontend-testing
- component-testing
- mocking-patterns
- tdd-workflow
allowed-tools: [Read, Write, Task, Bash]
---

# JavaScript Testing Patterns

// turbo-all

# JavaScript Testing Patterns

Testing confidence for the frontend and Node.js ecosystem.

---

## Strategy & Frameworks (Parallel)

// parallel

### Tool Selection

| Tool | Use Case | Speed |
|------|----------|-------|
| **Vitest** | Modern/Vite Apps | ⚡️ Fast (ESM Native) |
| **Jest** | Legacy/CRA Apps | 🐢 Moderate |
| **RTL** | React Components | User-centric (Roles) |
| **MSW** | API Mocking | Network Level |

### Test Types

-   **Unit**: Pure functions (`utils.ts`).
-   **Component**: UI rendering (`Button.tsx`).
-   **Hook**: State logic (`useCounter.ts`).
-   **Integration**: API + Component.

// end-parallel

---

## Decision Framework

### Testing Pyramid (Frontend)

1.  **Static**: TSServer / ESLint (Catch typos).
2.  **Unit/Component**: 70% of tests. (Button clicks, State updates).
3.  **Integration**: 20% of tests. (Login flow).
4.  **E2E**: 10% of tests. (Playwright - Critical user journeys).

---

## Core Knowledge (Parallel)

// parallel

### Constitutional AI Principles

1.  **Accessibility (Target: 100%)**: Use `getByRole` over `getByTestId` where possible.
2.  **Isolation (Target: 100%)**: Mock network calls (MSW/vi.mock).
3.  **Speed (Target: <2m)**: Run relevant tests only (`--changedSince`).

### Quick Reference

-   `render(<Comp />)`
-   `screen.getByRole('button', { name: /save/i })`
-   `await userEvent.click(btn)`
-   `expect(fn).toHaveBeenCalledTimes(1)`

// end-parallel

---

## Quality Assurance

### Common Antipatterns

| Antipattern | Fix |
|-------------|-----|
| Testing Implementation | `state.count === 1` -> `getByText('Count: 1')`. Test Output. |
| Excessive Mocking | Don't mock child components unless heavy. |
| `act()` warnings | Wait for async updates (`findByRole`). |
| Snapshot overuse | Fragile. Use explicit assertions. |

### JS Test Checklist

- [ ] Vitest configured
- [ ] RTL "User Event" used (not fireEvent)
- [ ] Mocks cleared `beforeEach`
- [ ] Handlers defined for MSW
- [ ] A11y checks (`jest-axe`) included
