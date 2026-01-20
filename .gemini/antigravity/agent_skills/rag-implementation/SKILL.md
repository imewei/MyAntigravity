---
name: rag-implementation
description: "Production RAG patterns: Chunking, Hybrid Search, Reranking, and Grounding."
version: 2.2.1
agents:
  primary: ai-engineer
skills:
- content-ingestion
- retrieval-optimization
- generation-grounding
- vector-ops
allowed-tools: [Read, Write, Task, Bash]
triggers:
- keyword:rag
- keyword:retrieval
- keyword:vector
- keyword:embedding
---

# RAG Implementation

// turbo-all

# RAG Implementation

Building the bridge between private knowledge and LLM reasoning.



## Strategy & Pipeline (Parallel)

// parallel

### Ingestion Pipeline

| Stage | Strategy |
|-------|----------|
| **Load** | Text, PDF, HTML loaders. |
| **Split** | RecursiveCharacter, Semantic, Markdown. |
| **Embed** | OpenAI (Ada-002), Cohere, Open Source (BGE). |
| **Store** | Pinecone, Chroma, pgvector. |

### Retrieval Strategy

-   **Dense**: Semantic match (Vectors).
-   **Sparse**: Keyword match (BM25/Splade).
-   **Hybrid**: Weighted combination (0.7 Dense + 0.3 Sparse).
-   **Rerank**: Cross-Encoder step for Top-N refinement.

// end-parallel

---

## Decision Framework

### RAG Flow Logic

1.  **Query Transform**: "Apple" -> "Apple (Technology)" vs "Apple (Fruit)".
2.  **Retrieve**: Get Top-50 candidates.
3.  **Rerank**: Score candidates by relevance (Cross-Encoder). Keep Top-5.
4.  **Augment**: Insert Top-5 into Context Window.
5.  **Generate**: LLM answers user query + citations.

---

## Core Knowledge (Parallel)

// parallel

### Constitutional AI Principles

1.  **Grounding (Target: 100%)**: Answers must stem from retrieval, not param memory.
2.  **Freshness (Target: 95%)**: Index updates near real-time.
3.  **Citation (Target: 100%)**: Every claim linked to a source chunk.

### Quick Reference Patterns

-   **Parent Document**: Retrieve small chunks -> Return full parent doc to LLM.
-   **Self-Query**: LLM generates metadata filters (e.g., `year > 2023`).
-   **Multi-Query**: LLM generates 3 variations of query -> Union results.
-   **HyDE**: Hypothetical Document Embeddings (Generate answer -> Embed -> Search).

// end-parallel

---

## Quality Assurance

### Common Pitfalls

| Pitfall | Fix |
|---------|-----|
| Bad Chunking | Cutting sentences in half. Use overlap. |
| Retrieval Miss | Use Hybrid Search + Query Expansion. |
| Hallucination | "Answer 'I don't know' if context missing." |
| Latency | Async formatting + Parallel retrieval. |

### RAG Checklist

- [ ] Chunking strategy valid (Semantic/Recursive)
- [ ] Hybrid search (Dense+Sparse) active
- [ ] Reranker integrated (e.g., Cohere/BGE)
- [ ] Citations embedded in output
- [ ] Metadata filtering enabled
- [ ] "I don't know" fallback verified
