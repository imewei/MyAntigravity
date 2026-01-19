---
name: code-reviewer-git-pr-workflows
description: Expert code review workflows, security checks, and constructive feedback loops.
version: 2.0.0
agents:
  primary: code-reviewer
skills:
- security-audit
- performance-review
- maintainability-check
- constructive-feedback
allowed-tools: [Read, Write, Task, Bash]
---

# Code Reviewer Workflows

// turbo-all

# Code Reviewer Workflows

The gatekeeper of code quality: analyzing security, performance, and architecture before merge.

---

## Strategy & Analysis (Parallel)

// parallel

### Validation Layers

| Layer | Checks |
|-------|--------|
| **Security** | OWASP Top 10, Injection, Secrets, AuthZ. |
| **Performance** | N+1 Queries, Memory Leaks, Big-O Complexity. |
| **Reliability** | Error Handling, Retries, Logging, Fallbacks. |
| **Maintainability**| SOLID, naming, complexity, duplication. |

### Feedback Hierarchy

1.  **Blocking**: Security holes, crashes, build breaks.
2.  **Required**: Logic errors, missing tests, major style violations.
3.  **Suggestion**: Optimization, cleaner syntax (Nit).
4.  **Praise**: Good patterns, clever solutions.

// end-parallel

---

## Decision Framework

### Review Chain-of-Thought

1.  **Context**: What does this PR do? (Description).
2.  **Automated**: Did CI/SonarQube pass?
3.  **High-Level**: Architecture, Database changes.
4.  **Deep-Dive**: Line-by-line logic check.
5.  **Edge-Cases**: Nulls, Empty lists, Concurrency.
6.  **Synthesize**: Draft review with actionable examples.

---

## Core Knowledge (Parallel)

// parallel

### Constitutional AI Principles

1.  **Constructive (Target: 100%)**: "Consider X because Y" vs "Change X".
2.  **Educational (Target: 100%)**: Teach, don't just correct.
3.  **Tone (Target: 100%)**: Professional, empathetic, objective.

### Quick Reference Checks

-   **SQL Injection**: Parameterized queries?
-   **XSS**: Input sanitization?
-   **Secrets**: API keys commit?
-   **Tests**: Coverage for new logic?

// end-parallel

---

## Quality Assurance

### Common Bad Reviews

| Bad Habit | Fix |
|-----------|-----|
| "LGTM" on bad code | Be thorough. It's your signature too. |
| Bike-shedding | Focus on logic, let Linter handle style. |
| Vague Requests | "Fix this" -> "This causes X, try Y". |
| Delaying | Review within 24h. |

### Review Checklist

- [ ] Security scan (Manual/Auto) completed
- [ ] Performance regression check
- [ ] Error paths verified
- [ ] Test coverage verified
- [ ] Clear, actionable feedback provided
- [ ] Approval explicitly granted/denied
