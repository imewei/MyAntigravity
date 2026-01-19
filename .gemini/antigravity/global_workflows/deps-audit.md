---
description: Audit project dependencies for security vulnerabilities and license compliance
triggers:
- /deps-audit
- workflow for deps audit
version: 2.0.0
allowed-tools: [Read, Task, Bash]
agents:
  primary: security-auditor
skills:
- dependency-management
- security-auditing
argument-hint: '[--mode=quick|standard|comprehensive]'
---

# Dependency Audit (v2.0)

// turbo-all

## Phase 1: Detection (Parallel)

// parallel

1.  **Vulnerability Scan**
    - Action: Run `npm audit`, `pip-audit`, `cargo audit`, or `govulncheck`.
    - Goal: Identify CVEs with specific severity scores.

2.  **License Compliance**
    - Action: Scan for non-permissive licenses (GPL, AGPL) in production deps.
    - Constraint: Flag exact package pairs.

3.  **Outdated Check**
    - Action: Check for major version updates.

// end-parallel

## Phase 2: Prioritization (Sequential)

4.  **Risk Scoring**
    - Formula: CVSS Score + Exploitability Context.
    - Output: Priority list (P0 Critical to P3 Low).

5.  **Report Generation**
    - Action: tailored report with remediation commands.
