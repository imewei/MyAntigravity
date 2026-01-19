---
name: terraform-specialist
description: Infrastructure as Code (IaC) mastery using Terraform/OpenTofu.
version: 2.0.0
agents:
  primary: cloud-architect
skills:
- iac-design
- module-development
- state-management
- policy-as-code
allowed-tools: [Read, Write, Task, Bash]
---

# Terraform Specialist

// turbo-all

# Terraform Specialist

Defining the world as code: Reproducible, versioned, and auditable infrastructure.

---

## Strategy & Patterns (Parallel)

// parallel

### Module Hierarchy

| Layer | Purpose | Example |
|-------|---------|---------|
| **Root** | Environment composition. | `envs/prod/main.tf` |
| **Composition** | Group resources. | `modules/stacks/web-cluster` |
| **Resource** | Atomic wrapper. | `modules/aws/s3-bucket` |

### State Strategy

-   **Remote**: Always use S3/GCS + DynamoDB (Locking).
-   **Isolation**: Separate state files per env (`prod`, `dev`) and region.
-   **Encryption**: Enable bucket encryption + Versioning.

// end-parallel

---

## Decision Framework

### Workflow Lifecycle

1.  **Write**: HCL code in VS Code.
2.  **Lint**: `terraform fmt`, `tflint`.
3.  **Plan**: `terraform plan -out=tfplan` (Verify changes).
4.  **Scan**: `tfsec` / `checkov` (Security audit).
5.  **Apply**: `terraform apply tfplan` (CI/CD only).

---

## Core Knowledge (Parallel)

// parallel

### Constitutional AI Principles

1.  **Idempotency (Target: 100%)**: Running twice changes nothing.
2.  **Immutability (Target: 100%)**: Replace servers, don't patch them.
3.  **DRY (Target: 90%)**: Use `for_each` and modules.

### Quick Reference HCL

-   `locals`: Computed variables.
-   `data`: Read-only external info.
-   `moved`: Refactor state without destroy.
-   `lifecycle`: `prevent_destroy = true`.

// end-parallel

---

## Quality Assurance

### Common Pitfalls

| Pitfall | Fix |
|---------|-----|
| Monolithic State | "Blast radius" too big. Split into smaller stacks. |
| Hardcoded IPs | Use `data` sources or Outputs. |
| Manual Changes | Causes "Drift". Detect with nightly plans. |
| Secrets in Stete | Use remote backends (S3 Encrypted). |

### TF Checklist

- [ ] Remote backend configured with locking
- [ ] Version constraints pinned (`~> 5.0`)
- [ ] `tflint` and `tfsec` passing
- [ ] Resources tagged (Cost allocation)
- [ ] Outputs defined for consumption
- [ ] Sensitive outputs marked `sensitive = true`
