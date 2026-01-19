---
name: ai-assisted-debugging
description: Leverage LLMs and automated analysis tools for Root Cause Analysis (RCA).
version: 2.0.0
agents:
  primary: debugger
skills:
- log-analysis
- automated-rca
- anomaly-detection
- stack-trace-explanation
allowed-tools: [Read, Write, Task, Bash]
triggers:
- file:.ipynb
- keyword:ai
- keyword:ml
---

# AI Assisted Debugging

// turbo-all

# AI Assisted Debugging

Augmenting human intuition with machine speed for finding needles in haystacks.

---

## Strategy & Tools (Parallel)

// parallel

### AI Techniques

| Technique | Application |
|-----------|-------------|
| **Explanation** | "Explain this stack trace and suggest a fix." |
| **Correlation** | "Correlate error spikes with recent commits." |
| **Anomaly** | "Find outliers in log patterns." (Isolation Forest). |
| **Generation** | "Write a reproduction script for this bug." |

### Context Injection

-   **Code**: Snippet around the error line.
-   **State**: Variable values at crash time.
-   **Environment**: OS, Library versions.
-   **History**: Recent changes/deployments.

// end-parallel

---

## Decision Framework

### RCA Pipeline

1.  **Detect**: Alert triggers (High Error Rate).
2.  **Enrich**: Gather Logs + Traces + Metrics.
3.  **Analyze**:
    *   *Simple*: LLM explain stack trace.
    *   *Complex*: ML correlate metric shift with deployment.
4.  **Hypothesize**: "Database connection pool exhaustion due to leak."
5.  **Verify**: Check connection metrics.

---

## Core Knowledge (Parallel)

// parallel

### Constitutional AI Principles

1.  **Validation (Target: 100%)**: AI suggestions must be verified by humans.
2.  **Privacy (Target: 100%)**: Scrub PII/Secrets before sending to LLM.
3.  **Safety (Target: 100%)**: Never auto-apply AI fixes to Prod.

### Quick Reference Prompts

-   "Analyze this stack trace given source code context..."
-   "Propose a unit test to reproduce this edge case..."
-   "What configuration changes could cause this timeout?"

// end-parallel

---

## Quality Assurance

### Common Pitfalls

| Pitfall | Fix |
|---------|-----|
| Hallucinated APIs | Verify library documentation. |
| Context Window Limit | Summarize logs, don't dump 1GB. |
| Blind Trust | Always review logic. AI guesses based on probability. |
| Missing Context | LLM needs version numbers to know specific bugs. |

### Debug Checklist

- [ ] PII redaction active
- [ ] Context gathering automated
- [ ] Prompts optimized for RCA
- [ ] Anomaly detection baselined
- [ ] Human-in-the-loop verification
