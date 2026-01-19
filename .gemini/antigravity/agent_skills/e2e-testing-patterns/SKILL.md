---
name: e2e-testing-patterns
description: Robust End-to-End testing strategies using Playwright and Cypress.
version: 2.0.0
agents:
  primary: test-automator
skills:
- browser-automation
- visual-regression
- accessibility-testing
- ci-integration
allowed-tools: [Read, Write, Task, Bash]
triggers:
- keyword:qa
- keyword:testing
---

# E2E Testing Patterns

// turbo-all

# E2E Testing Patterns

Verifying the full stack from the user's perspective.

---

## Strategy & Tooling (Parallel)

// parallel

### Framework Choice

| Feature | Playwright | Cypress |
|---------|------------|---------|
| **Speed** | ⚡️ Fast (Parallel) | 🚶 Moderate |
| **Browsers**| All (WebKit/Gecko) | Chrome/Fox |
| **Events** | Native (Trusted) | Synthetic |
| **Tabs** | Multiple supported | Single tab only |

### Core Patterns

-   **Page Object Model (POM)**: Class representing a page. Encapsulates selectors.
-   **Interception**: Mock API responses to test edge cases (500 errors).
-   **Visual Diff**: Pixel-perfect regression testing.

// end-parallel

---

## Decision Framework

### Test Design

1.  **Scenario**: "User logs in and buys item."
2.  **Selectors**: Use `getByRole` or `data-testid`. (Never CSS classes).
3.  **Action**: `click`, `fill`, `press`.
4.  **Assertion**: `expect(page).toHaveURL(...)`.

---

## Core Knowledge (Parallel)

// parallel

### Constitutional AI Principles

1.  **Resilience (Target: 100%)**: Use "Auto-waiting" locators. No manual `sleep()`.
2.  **Accessibility (Target: 100%)**: Run `axe-core` scan on every page load.
3.  **Traceability (Target: 100%)**: Record video/trace on failure.

### Quick Reference (Playwright)

-   `page.getByRole('button', { name: 'Save' })`.
-   `expect(page.getByText('Success')).toBeVisible()`.
-   `await page.route('**/api/data', route => route.fulfill(...))`.

// end-parallel

---

## Quality Assurance

### Common Pitfalls

| Pitfall | Fix |
|---------|-----|
| Flaky Tests | usually network race conditions. Use `await expect`. |
| Hard Coded Auth | Use global setup to save "Storage State" (Cookies) once. |
| Testing External Sites | Don't. Mock Stripe/Auth0. |
| Brittle Selectors | `div > span:nth-child(3)` -> `getByTestId('price')`. |

### E2E Checklist

- [ ] Playwright configured
- [ ] POM implemented
- [ ] HTML Reports enable
- [ ] Traces captured on failure
- [ ] Visual Regression (Snapshot)
- [ ] A11y Audit (Axe)
