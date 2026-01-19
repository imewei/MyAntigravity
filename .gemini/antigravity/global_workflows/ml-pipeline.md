---
description: Design and implement production-ready ML pipelines
triggers:
- /ml-pipeline
- design ml pipeline
version: 2.0.0
allowed-tools: [Bash, Read, Write, Task]
agents:
  primary: ml-engineer
skills:
- machine-learning-ops
- data-engineering
argument-hint: '[project-name]'
---

# ML Pipeline Architect (v2.0)

// turbo-all

## Phase 1: Design (Parallel)

// parallel

1.  **Data Strategy**
    - Define Ingestion, Validation, Storage.

2.  **Model Strategy**
    - Select Algorithms, Frameworks (PyTorch/JAX).

// end-parallel

## Phase 2: Development (Parallel)

// parallel

3.  **Training Pipeline**
    - Agent: ml-engineer
    - Action: Implement training script, hyperparam tuning.

4.  **Feature Pipeline**
    - Agent: data-engineer
    - Action: Implement ETL, Feature Store.

// end-parallel

## Phase 3: Deployment (Sequential)

5.  **Serving Infrastructure**
    - Implement Model Serving (FastAPI/Triton).

6.  **Monitoring**
    - Setup Drift Detection, Performance Metrics.

## Phase 4: Integration

7.  **CI/CD for ML**
    - Automate Retraining and Deployment.
