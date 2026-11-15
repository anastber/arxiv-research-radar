# Roadmap

This document outlines the remaining work on arxiv-research-radar: improving retrieval quality
against the benchmark, then shipping a public deployment. Larger-scale features are listed under
Future directions.

## Current state

- 20 papers / ~1,000 chunks; the weekly refresh workflow exists but is currently disabled
- Dense retrieval only (`all-MiniLM-L6-v2` + exact FAISS search)
- Section-aware chunking over `pypdf`-extracted text
- Grounded answers with per-paper citations, retrieval threshold for off-topic questions
- Index and corpus versioned in the repository
- Benchmark on a frozen 437-paper corpus (122 questions): paper-level retrieval metrics and
  LLM-judged answer metrics, run with `python -m eval.run`

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
- [x] Golden question set (~100 questions), generated from source chunks and manually
      reviewed, including off-topic questions and multi-paper questions (122 questions:
      90 single-paper, 20 multi-paper, 12 off-topic; 37 hand-reviewed so far)
- [x] Retrieval metrics: Recall@k, MRR, nDCG, off-topic rejection accuracy
- [x] Answer metrics: faithfulness, citation precision, answer relevance (LLM-judged,
      calibrated against a hand-graded sample)
      (calibration sample of 20 answers graded by Claude Opus 5.5; human grading pending)
- [x] `python -m eval.run` entry point producing a results table and a JSON report

## Milestone 3 — Retrieval quality

The baseline shows the main weakness: the right paper is often retrieved but not the passage
that answers the question (14 of 15 declined in-scope answers). Each change is benchmarked
against the previous configuration with `python -m eval.run`, and the results are published
as an ablation table in the README.

- [ ] **Embedding model:** compare a stronger open model (`bge` / `e5` family) to `all-MiniLM-L6-v2`
- [ ] **Hybrid search:** BM25 + dense retrieval combined with Reciprocal Rank Fusion, to
      better match exact terms such as method names and acronyms
- [ ] **Reranking:** cross-encoder reranking of a wider candidate set
- [ ] **Off-topic threshold:** add near-domain off-topic questions, then retune `MIN_SCORE`
      (or move it to the reranker score)

## Milestone 4 — Deployment

- [ ] Docker image for the API and the Streamlit app
- [ ] Public deployment
- [ ] Move the index out of the repository (built at image build time or published as an
      artifact), to stop the weekly refresh from growing git history

## Future directions

Out of scope for now, kept as possible extensions:

- Cleaner document parsing from arXiv HTML or LaTeX sources instead of PDF text
- Chunking experiments (chunk size, parent-child retrieval, contextual enrichment)
- Scaling to thousands of papers: database instead of `papers.json`, incremental indexing,
  a vector store with metadata filtering
- Query understanding: structured filters ("since September"), query expansion
- Multi-turn conversations, streaming responses, sentence-level citation verification
- Request tracing, user feedback, retrieval regression check in CI, response caching
- Weekly topic digests, related-paper recommendations, citation graph integration
