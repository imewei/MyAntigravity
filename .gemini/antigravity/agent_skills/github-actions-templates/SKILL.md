---
name: github-actions-templates
description: Reusable workflows for CI/CD, Containerization, and Security Scanning on GitHub.
version: 2.0.0
agents:
  primary: devops-engineer
skills:
- workflow-automation
- container-building
- security-scanning
- deployment-pipelines
allowed-tools: [Read, Write, Task, Bash]
triggers:
- file:.github/workflows/*.yml
- file:.ipynb
- keyword:ai
- keyword:audit
- keyword:ci-cd
- keyword:ml
- keyword:security
---

# GitHub Actions Templates

// turbo-all

# GitHub Actions Templates

Production-grade workflow patterns for the GitHub ecosystem.

---

## Strategy & Patterns (Parallel)

// parallel

### Workflow Types

| Type | Triggers | Actions |
|------|----------|---------|
| **CI** | `push`, `pull_request` | Install, Lint, Test, Coverage. |
| **Release** | `push tags: v*` | Build, ECR/GHCR Push, Deploy. |
| **Security** | `schedule`, `pr` | Trivy, CodeQL, Dependency Scan. |
| **Ops** | `workflow_dispatch` | Manual scripts, DB migrations. |

### Security Best Practices

-   **OIDC**: Use `aws-actions/configure-aws-credentials` (No long-lived keys).
-   **Pinning**: Use SHA (`@abcdef`) or specific tag (`@v4.1.0`), not `@latest`.
-   **Permissions**: `permissions: read-all` by default.

// end-parallel

---

## Decision Framework

### Pipeline Composition

1.  **Checkout**: Get code (`actions/checkout`).
2.  **Setup**: Install tools (`setup-node`, `setup-python`).
3.  **Cache**: Restore dependencies (`cache: npm`).
4.  **Execute**: Run build/test scripts.
5.  **Artifact**: Save reports/binaries (`upload-artifact`).
6.  **Report**: Comment on PR or Update Check Run.

---

## Core Knowledge (Parallel)

// parallel

### Constitutional AI Principles

1.  **Speed (Target: <5m)**: Cache aggressively.
2.  **Isolation (Target: 100%)**: Jobs should not leak state.
3.  **Visibility (Target: 100%)**: Fail fast and log clearly.

### Quick Reference Actions

-   `docker/build-push-action@v5`
-   `aquasecurity/trivy-action@master`
-   `slackapi/slack-github-action@v1`
-   `softprops/action-gh-release@v1`

// end-parallel

---

## Quality Assurance

### Common Pitfalls

| Pitfall | Fix |
|---------|-----|
| Hardcoded Secrets | Use `${{ secrets.KEY }}`. |
| Unbounded Matrix | Exclude incompatible combinations. |
| Duplicate Workflows | Use Reusable Workflows (`workflow_call`). |
| Missing Concurrency | Cancel old runs on new commit (`concurrency: group: ...`). |

### Actions Checklist

- [ ] Permissions restricted (Least Privilege)
- [ ] Secrets masked/redacted
- [ ] Dependencies cached
- [ ] Third-party actions audited/pinned
- [ ] Timeout minutes set (prevent stuck jobs)
- [ ] Concurrency limits configured
