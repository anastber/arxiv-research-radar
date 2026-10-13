# Roadmap

This document outlines the planned evolution of arxiv-research-radar, from a 20-paper
demo to a research assistant that stays accurate over a corpus of several thousand papers.

## Current state

- 20 papers / ~1,000 chunks, refreshed weekly by GitHub Actions
- Dense retrieval only (`all-MiniLM-L6-v2` + exact FAISS search)
- Section-aware chunking over `pypdf`-extracted text
- Grounded answers with per-paper citations, retrieval threshold for off-topic questions
- Index and corpus versioned in the repository
- No quality measurement yet

## Approach

- **Measurement first.** Retrieval and answer quality are tracked against a fixed
  benchmark before any change to the pipeline, so every improvement is backed by numbers.
- **Incremental changes.** Each pipeline change is evaluated in isolation and its impact
  recorded in the README.
- **Minimal dependencies.** The core pipeline stays framework-free to keep every stage
  explicit and easy to inspect.

---

## Milestone 1 — Testing and CI

- [x] Unit tests for chunking (heading detection, bibliography removal, section merging),
      citation extraction, source building, and retrieval against a fixture index
- [x] API tests: validation, error mapping, daily cap, per-IP rate limit
- [x] Linting with `ruff`
- [x] CI workflow running lint and tests on every push to `main` and every pull request

## Milestone 2 — Evaluation framework

Because the live corpus changes weekly, evaluation runs against a **frozen snapshot**
(corpus + index) so results stay comparable over time.

- [x] Frozen benchmark corpus of 300–500 papers (437 papers, 2026-08-01 → 2026-09-30)
- [ ] Golden question set (~100 questions), generated from source chunks and manually
      reviewed, including off-topic questions and multi-paper questions
- [ ] Retrieval metrics: Recall@k, MRR, nDCG, off-topic rejection accuracy
- [ ] Answer metrics: faithfulness, citation precision, answer relevance (LLM-judged,
      calibrated against a hand-graded sample)
- [ ] `python -m eval.run` entry point producing a results table and a JSON report

## Milestone 3 — Retrieval quality

Each item is benchmarked against the previous configuration; results are published as an
ablation table in the README.

- [ ] **Document parsing:** use arXiv HTML or LaTeX sources instead of PDF text extraction
      for cleaner structure and more reliable section boundaries
- [ ] **Embedding model:** compare stronger open models (`bge`, `e5`, `gte` families)
- [ ] **Hybrid search:** BM25 + dense retrieval combined with Reciprocal Rank Fusion, to
      better match exact terms such as method names and acronyms
- [ ] **Reranking:** cross-encoder reranking of a wider candidate set, with the relevance
      threshold moved to the reranker score
- [ ] **Chunking:** chunk size tuning, parent-child retrieval, contextual chunk enrichment

## Milestone 4 — Scaling the corpus

Target: 5,000–10,000 papers across several arXiv categories.

- [ ] Move the index out of the repository into a published artifact
- [ ] Replace `papers.json` with a database (SQLite or Postgres)
- [ ] Incremental indexing (additions and removals) instead of full rebuilds
- [ ] Vector store with metadata filtering (Qdrant or pgvector): date, category, authors
- [ ] Bulk, resumable ingestion respecting arXiv rate limits
- [ ] Benchmark approximate-search recall and latency against exact search

## Milestone 5 — Query understanding and answer generation

- [ ] Structured filters extracted from the question ("since September", "in cs.CR")
- [ ] Query expansion (multi-query, HyDE), kept only where the benchmark shows a gain
- [ ] Multi-turn conversations with follow-up questions
- [ ] Streaming responses
- [ ] Sentence-level citation verification before an answer is returned
- [ ] Multi-step retrieval for comparative and multi-hop questions
- [ ] Corpus-level questions (e.g. "main trends this month") via summarization over
      paper clusters

## Milestone 6 — Production readiness

- [ ] Request tracing, including retrieved chunk ids per answer
- [ ] User feedback on answers, feeding the evaluation set
- [ ] Retrieval benchmark as a regression check in the weekly refresh workflow
- [ ] Docker image and public deployment
- [ ] Response caching

## Future directions

- Weekly topic digests and alerts
- Related-paper recommendations
- Citation graph integration (Semantic Scholar API)
