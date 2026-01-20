---
name: kubernetes-architect
description: Design and manage scalable, secure, and observable Kubernetes clusters.
version: 2.2.1
agents:
  primary: cloud-architect
skills:
- cluster-design
- gitops-workflow
- pod-security
- observability-stack
allowed-tools: [Read, Write, Task, Bash]
triggers:
- file:helm/**/*.yaml
- file:k8s*.yaml
- keyword:kubernetes
---

# Kubernetes Architect

// turbo-all

# Kubernetes Architect

Architecting cloud-native platforms with K8s, GitOps, and Service Meshes.

---

## Strategy & Architecture (Parallel)

// parallel

### Architectural Decisions

| Layer | Choices | Recommendation |
|-------|---------|----------------|
| **Core** | EKS, GKE, AKS, K3s | Managed K8s (reduce toil). |
| **Ingress** | Nginx, ALB, Istio | Nginx (Simple) -> Istio (Complex). |
| **GitOps** | ArgoCD, Flux | ArgoCD (Visual) or Flux (Headless). |
| **Secrets** | SealedSecrets, ExternalSecrets | ExternalSecrets (AWS/HashiCorp integration). |

### Workload Types

-   **Stateless**: Web APIs (`Deployment` + `HPA`).
-   **Stateful**: DBs (`StatefulSet` + `PVC`).
-   **Jobs**: Batch processing (`CronJob`).
-   **Daemon**: Agents (`DaemonSet` e.g., Fluentd).

// end-parallel

---

## Decision Framework

### Cluster Design Process

1.  **Compute**: Select Node Groups (On-Demand for control, Spot for batch).
2.  **Network**: VPC CNI plugin, IP range planning.
3.  **Security**: OIDC provider, RBAC groups.
4.  **Add-ons**: Install "Platform Layer" (Argo, Prom, Cert-Manager).
5.  **Tenancy**: Namespace isolation vs Multi-cluster.

---

## Core Knowledge (Parallel)

// parallel

### Constitutional AI Principles

1.  **GitOps (Target: 100%)**: No `kubectl apply` by hand. Push to Git.
2.  **Least Privilege (Target: 100%)**: No `cluster-admin` for users.
3.  **Resilience (Target: 99.9%)**: Multi-AZ node groups.

### Quick Reference Manifests

-   **HPA**: `minReplicas: 2`, `maxReplicas: 10`.
-   **PDB**: `minAvailable: 1` (Prevent downtime during upgrades).
-   **Probes**: `liveness` (Reset), `readiness` (Traffic).
-   **Resources**: Always set `requests` and `limits`.

// end-parallel

---

## Quality Assurance

### Common Pitfalls

| Pitfall | Fix |
|---------|-----|
| No Limits | "Noisy Neighbor" kills node. Set RAM limits. |
| CPU Throttling | Set CPU requests = limits (for guaranteed QoS). |
| Image Pull Backoff | Check image name, tag, and Secret. |
| CrashLoopBackOff | Check logs (`kubectl logs -p`). |

### K8s Checklist

- [ ] GitOps agent installed (ArgoCD/Flux)
- [ ] Ingress Controller configured with TLS
- [ ] Network Policies (Default Deny)
- [ ] Resource Quotas per namespace
- [ ] Prometheus/Grafana stack active
- [ ] Cluster Autoscaler enabled
- [ ] PDBs configured for critical apps
