---
name: gitlab-ci-patterns
description: GitLab CI/CD pipelines including caching, Docker-in-Docker, and multi-cloud deployment.
version: 2.0.0
agents:
  primary: devops-engineer
skills:
- pipeline-architecture
- docker-services
- environment-management
- artifact-handling
allowed-tools: [Read, Write, Task, Bash]
triggers:
- file:.github/workflows/*.yml
- file:Dockerfile
- file:docker-compose.yml
- keyword:aws
- keyword:ci-cd
- keyword:cloud
- keyword:docker
---

# GitLab CI Patterns

// turbo-all

# GitLab CI Patterns

Enterprise-grade pipeline definitions for `.gitlab-ci.yml`.

---

## Strategy & Stages (Parallel)

// parallel

### Pipeline Structure

| Stage | Activities |
|-------|------------|
| **.pre** | Linting, Static Analysis. |
| **build** | Compile, Docker Build (`dind`), Kaniko. |
| **test** | Unit/Integration Tests. |
| **security** | Container Scanning, SAST. |
| **deploy** | Helm Upgrade, Terraform Apply. |
| **.post** | Notifications (Slack/Email). |

### Caching Strategy

-   **Workspace**: Pass artifacts between stages (temporary).
-   **Cache**: Persist dependencies between runs (long-term).
-   **Key**: `${CI_COMMIT_REF_SLUG}` (Branch-specific) or `global`.

// end-parallel

---

## Decision Framework

### Dynamic Pipelines

1.  **Generate**: Script creates child YAML based on changes.
2.  **Trigger**: Parent pipeline triggers child.
3.  **Execute**: Child runs subset of jobs (e.g., only backend tests).

---

## Core Knowledge (Parallel)

// parallel

### Constitutional AI Principles

1.  **Feedback (Target: <10m)**: Devs get feedback fast.
2.  **Security (Target: 100%)**: Protected variables for Prod secrets.
3.  **Traceability (Target: 100%)**: Deployments link to Commit SHA.

### Quick Reference Variables

-   `CI_COMMIT_SHA`: Current commit.
-   `CI_REGISTRY_IMAGE`: Project container registry.
-   `CI_ENVIRONMENT_NAME`: prod/staging.
-   `CI_PIPELINE_SOURCE`: trigger, push, schedule.

// end-parallel

---

## Quality Assurance

### Common Pitfalls

| Pitfall | Fix |
|---------|-----|
| Artifact Bloat | Set `expire_in` for artifacts. |
| Zombie Runners | Use timeouts. |
| Secrets in Variables | Check "Masked" and "Protected" boxes. |
| Docker Limit | Use internal registry mirror. |

### GitLab Checklist

- [ ] `stages` defined clearly
- [ ] `image` pinned to SHA/Tag
- [ ] `cache` keys distinct (no thrashing)
- [ ] `before_script` handles auth
- [ ] `rules` optimize execution (avoid redundant jobs)
- [ ] Security scanners (SAST) included
