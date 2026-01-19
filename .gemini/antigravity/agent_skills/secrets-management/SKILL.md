---
name: secrets-management
description: Secure lifecycle for API keys, certificates, and credentials.
version: 2.0.0
agents:
  primary: security-auditor
skills:
- secret-rotation
- vault-operations
- pipeline-security
- leak-detection
allowed-tools: [Read, Write, Task, Bash]
triggers:
- keyword:secrets
- keyword:vault
- keyword:credentials

# Secrets Management

// turbo-all

# Secrets Management

"Zero Trust" credential handling: Store, Inject, Rotate, Audit.

---

## Strategy & Storage (Parallel)

// parallel

### Secret Stores

| Store | Best For | Pattern |
|-------|----------|---------|
| **Vault** | Enterprise, Dynamic Secrets | App authenticates -> Gets short-lived token. |
| **AWS Secrets** | AWS-Native, RDS Rotation | App IAM Role -> SDK call. |
| **K8s Secrets** | Cluster-Internal | Encrypted ETCD + RBAC. |
| **GitHub** | CI/CD Pipelines | Repository Secrets (Masked). |

### Injection patterns

-   **Env Vars**: Standard (`process.env.API_KEY`).
-   **Files**: Volume Mount (`/etc/secrets/key`).
-   **Sidecar**: Agent fetches and rotates (transparent).

// end-parallel

---

## Decision Framework

### Leak Prevention

1.  **Pre-Commit**: `trufflehog` / `gitleaks` scans local git.
2.  **CI Scan**: Pipeline fails if high-entropy string found.
3.  **Runtime**: Secrets in memory only (no swap/logs).
4.  **Rotation**: Automated cron to change keys monthly/daily.

---

## Core Knowledge (Parallel)

// parallel

### Constitutional AI Principles

1.  **Ephemeral (Target: 90%)**: Short-lived dynamic secrets preferred.
2.  **Masked (Target: 100%)**: Logs must never show plaintext values.
3.  **Encrypted (Target: 100%)**: At rest and in transit.

### Quick Reference

-   `vault kv put secret/app key=value`
-   `aws secretsmanager get-secret-value`
-   `external-secrets` (K8s Operator).
-   `::add-mask::${SECRET}` (GitHub Actions).

// end-parallel

---

## Quality Assurance

### Common Pitfalls

| Pitfall | Fix |
|---------|-----|
| Committing `.env` | Add to `.gitignore` strictly. |
| Hardcoded Defaults | Fail startup if secret missing, don't fallback to "password". |
| Logs | Audit log aggregators for leaks. |
| Broad Access | App A should not see App B's secrets. |

### Secrets Checklist

- [ ] Central store configured (Vault/AWS/Azure)
- [ ] Automated rotation enabled
- [ ] Pre-commit hooks active (Gitleaks)
- [ ] CI Logs checked for leakage
- [ ] Least Privilege Access Policies
- [ ] `.gitignore` audit
