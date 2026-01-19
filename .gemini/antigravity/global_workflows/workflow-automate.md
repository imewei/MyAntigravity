---
description: Automate development workflows (CI/CD)
triggers:
- /workflow-automate
- automate workflow
version: 2.0.0
allowed-tools: [Read, Write, Bash]
agents:
  primary: devops-architect
skills:
- ci-cd-pipelines
- infrastructure-as-code
argument-hint: '[--platform=github|gitlab]'
---

# Automation Architect (v2.0)

// turbo-all

## Phase 1: Discovery (Parallel)

// parallel

1.  **Platform Check**
    - Action: Detect GitHub/GitLab.

2.  **Gap Analysis**
    - Action: Check for existing CI files.

// end-parallel

## Phase 2: Design (Sequential)

3.  **Pipeline Strategy**
    - Design stages: Quality -> Test -> Build -> Deploy.

## Phase 3: Implementation (Parallel)

// parallel

4.  **CI Configuration**
    - Action: Generate YAML files.

5.  **Security Scanning**
    - Action: Add Trivy/Snyk steps.

6.  **Release Automation**
    - Action: Add Semantic Release.

// end-parallel
