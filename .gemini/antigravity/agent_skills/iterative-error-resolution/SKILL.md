---
name: iterative-error-resolution
description: Automated CI/CD troubleshooting, dependency conflict resolution, and self-correcting loops.
version: 2.0.0
agents:
  primary: devops-troubleshooter
skills:
- log-analysis
- automated-fixing
- dependency-resolution
- test-stabilization
allowed-tools: [Read, Write, Task, Bash]
triggers:
- keyword:iterative-error-resolution
---

# Iterative Error Resolution

// turbo-all

# Iterative Error Resolution

Systematic engine for resolving persistent failures in CI/CD, builds, and tests through iterative diagnosis and repair.

---

## Strategy & Diagnosis (Parallel)

// parallel

### Error Categories & Strategies

| Category | Fix Strategy |
|----------|--------------|
| **Dependency** | Relax versions, `--legacy-peer-deps`, align lockfiles |
| **Build/Types** | TypeScript assertion, ignore (if safe), correct types |
| **Test** | Update snapshot, fix logic, mock external service |
| **Runtime** | Increase memory (OOM), timeout extension, retry logic |
| **Network** | Retries, fallback mirrors |

### Validation Framework

1.  **Isolate**: Can I reproduce this locally?
2.  **Fix**: Apply high-confidence patch.
3.  **Verify**: Run minimal test case.
4.  **Loop**: If specific error gone -> Next error. If same -> Escalate.

// end-parallel

---

## Decision Framework

### Iterative Engine Logic

1.  **Capture**: Grep error logs for patterns (e.g., `ERESOLVE`, `ETIMEDOUT`).
2.  **Match**: Lookup `KnowledgeBase` for known fix.
3.  **Apply**: Execute fix (sed, npm install, code edit).
4.  **Commit**: Tag as `fix(ci): attempt N`.
5.  **Watch**: Monitor next run.
6.  **Learn**: If success, boost confidence of fix pattern.

---

## Core Knowledge (Parallel)

// parallel

### Constitutional AI Principles

1.  **Safety (Target: 100%)**: Don't break prod to fix CI.
2.  **Convergence (Target: 100%)**: Error count must decrease.
3.  **Efficiency (Target: 90%)**: Max 5 iterations.

### Quick Fix Patterns

-   **NPM ERESOLVE**: `npm i --legacy-peer-deps` or `overrides` in package.json.
-   **Python Pip**: `pip install --no-cache-dir`.
-   **OOM**: `NODE_OPTIONS="--max-old-space-size=4096"`.
-   **Flaky Test**: Search `@flaky` -> Mark or Retry.

// end-parallel

---

## Quality Assurance

### Common Pitfalls

| Pitfall | Fix |
|---------|-----|
| Infinite Loop | Set max_iterations (e.g., 5) |
| Shotgun Debugging | Apply one fix at a time per error type |
| Regression | Rollback if new errors > old errors |
| Masking | Don't just `// @ts-ignore` without reason |

### Resolution Checklist

- [ ] Logs captured and analyzed
- [ ] Error type identified
- [ ] Fix strategy selected from Knowledge Base
- [ ] Local validation attempted
- [ ] Iteration limit enforced
- [ ] Rollback plan present
