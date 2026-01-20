---
name: legacy-modernizer
description: Strategies for Strangler Fig migration, brownfield refactoring, and safely killing spaghetti code.
version: 2.2.1
agents:
  primary: legacy-modernizer
skills:
- strangler-fig
- refactoring
- debt-management
- backward-compatibility
allowed-tools: [Read, Write, Task, Bash]
triggers:
- keyword:legacy
- keyword:modernize
- keyword:strangler
---

# Legacy Modernizer

// turbo-all

# Legacy Modernizer

Turning "Technical Debt" into "Technical Equity".



## Strategy & Patterns (Parallel)

// parallel

### Modernization Tactics

| Pattern | Description | Risk |
|---------|-------------|------|
| **Strangler Fig** | Build new system around old. Route traffic gradually. | Low |
| **Branch by Abstraction** | Abstract interface -> Swap implementation. | Medium |
| **Parallel Run** | Run New + Old. Compare outputs. Discard New. | Low |
| **Big Bang Rewrite** | Stop world. Rewrite v2. Go live. | Critical (Avoid) |

### Safety Nets

-   **Golden Master Tests**: Capture output of old system for identical inputs.
-   **Characterization Tests**: "Lock down" existing behavior before changing.
-   **Feature Flags**: Kill switch for new code.

// end-parallel

---

## Decision Framework

### Is it ready to die?

1.  **Value**: Does this code make money? Yes -> Refactor. No -> Delete.
2.  **Churn**: Do we touch it often? Yes -> Refactor. No -> Wrap/Ignore.
3.  **Risk**: Can we test it? No -> Add Tests first.

---

## Core Knowledge (Parallel)

// parallel

### Constitutional AI Principles

1.  **Continuity (Target: 100%)**: Business must run during migration.
2.  **Safety (Target: 100%)**: No refactoring without Green tests.
3.  **Reversibility (Target: 100%)**: Always have a rollback plan.

### Quick Reference

-   `grep -r "DEPRECATED" .`
-   "Leave the campground cleaner than you found it."
-   "Make the change easy, then make the easy change." (Kent Beck).

// end-parallel

---

## Quality Assurance

### Common Pitfalls

| Pitfall | Fix |
|---------|-----|
| "Rewrite from scratch" | Don't. You will lose domain knowledge. |
| Updating everything | Focus on "Hot Paths" (High churn/impact). |
| No Tests | Write "Black Box" tests against API/CLI first. |
| Removing Public API | Deprecate first (Log warning), remove later (v+1). |

### Migration Checklist

- [ ] Golden Master / Snapshot tests created
- [ ] Strangler Facade implemented
- [ ] Feature Flags active (LaunchDarkly/EnvVar)
- [ ] Rollback pipeline tested
- [ ] Deprecation Logs monitoring active
