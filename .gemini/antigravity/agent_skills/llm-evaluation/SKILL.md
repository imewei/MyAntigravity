---
name: llm-evaluation
description: Quantitative and qualitative evaluation of LLM outputs (BLEU, ROUGE, BERTScore, LLM-as-Judge).
version: 2.2.1
agents:
  primary: ai-engineer
skills:
- metrics-calculation
- llm-as-judge
- ab-testing
- regression-detection
allowed-tools: [Read, Write, Task, Bash]
triggers:
- keyword:llm
- keyword:evaluation
- keyword:benchmark

# LLM Evaluation

// turbo-all

# LLM Evaluation

Move beyond "vibes" to rigorous, scientific evaluation of LLM performance.



## Strategy & Metrics (Parallel)

// parallel

### Evaluation Tiers

| Tier | Method | Speed | Cost |
|------|--------|-------|------|
| **1. Unit** | Exact Match / Regex | Fast | Low |
| **2. Similarity** | ROUGE / BLEU / BERTScore | Fast | Low |
| **3. Model** | LLM-as-Judge (GPT-4) | Medium | Med |
| **4. Human** | Manual Labeling | Slow | High |

### LLM-as-Judge Patterns

-   **Pairwise**: "Compare A and B. Which is better?" (Less bias).
-   **Pointwise**: "Rate this 1-5 on accuracy."
-   **Reference**: "Compare prediction against gold answer."

// end-parallel

---

## Decision Framework

### Logic Pipeline

1.  **Define Dataset**: Golden Q&A pairs (n=50+).
2.  **Run Inference**: Generate answers from Model A and Model B.
3.  **Compute Metrics**: accuracy, groundedness, relevance.
4.  **Analyze**: Look for regressions (Did we fix X but break Y?).
5.  **Refine**: Update prompt/RAG based on failure modes.

---

## Core Knowledge (Parallel)

// parallel

### Constitutional AI Principles

1.  **Objectivity (Target: 90%)**: Minimize subjective metrics where possible.
2.  **Coverage (Target: 100%)**: Test happy paths AND adversarial inputs.
3.  **Reproducibility (Target: 100%)**: Fixed seeds, versioned datasets.

### Quick Reference Patterns

-   **Groundedness**: "Does the answer exist in the context?" (NLI check).
-   **Relevance**: "Does the answer address the user query?"
-   **Conciseness**: Length penalty if verbose.
-   **Coherence**: Logical flow check.

// end-parallel

---

## Quality Assurance

### Common Pitfalls

| Pitfall | Fix |
|---------|-----|
| Training on Test Data | Keep evaluation set finding separate. |
| Using Weak Judges | Judge model should be > capability of suspect. |
| Length Bias | Normalization for verbosity. |
| Single Metric | Use a scorecard (multiple dimensions). |

### Eval Checklist

- [ ] Golden dataset created (50+ items)
- [ ] Baseline metrics established
- [ ] LLM-as-Judge prompt validated
- [ ] Regression testing in CI
- [ ] Human audit of sub-sample
- [ ] Cost per eval tracked
