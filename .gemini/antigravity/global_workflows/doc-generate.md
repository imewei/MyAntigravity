---
description: Generate comprehensive documentation with AI analysis
triggers:
- /doc-generate
- workflow for doc generate
version: 2.0.0
allowed-tools: [Read, Task, Bash]
agents:
  primary: technical-writer
skills:
- documentation-best-practices
argument-hint: '[--api] [--readme] [--full]'
---

# Documentation Generator (v2.0)

// turbo-all

## Phase 1: Analysis (Parallel)

// parallel

1.  **Code Scan**
    - Action: Identify API endpoints, classes, exports.

2.  **Dependency Scan**
    - Action: List key libraries for context.

// end-parallel

## Phase 2: Generation (Parallel)

// parallel

3.  **API Reference**
    - Action: Generate OpenAPI spec or Argument Reference.

4.  **README.md**
    - Action: Generate Overview, Install, Usage sections.

5.  **Architecture Diagrams**
    - Action: Generate Mermaid diagrams (System/Flow).

// end-parallel

## Phase 3: Finalization

6.  **Validation**
    - Check for broken links and consistency.
