"""Build the FAISS index from data/papers.json.

Usage:  python -m rag.index
Writes data/index/faiss.index (vectors) and data/index/chunks.json (chunk text +
metadata). Row i of the index corresponds to chunks[i].
"""
import json
import sys

import faiss
from sentence_transformers import SentenceTransformer

import config
from rag.chunker import chunk_paper


def embedding_text(chunk: dict) -> str:
    # Prepending title + section gives short chunks the context they'd otherwise lack.
    return f"{chunk['title']} | {chunk['section']}\n{chunk['text']}"


def main() -> int:
    if not config.PAPERS_PATH.exists():
        print("ERROR: data/papers.json not found. Run `python -m ingest.fetch` first.", file=sys.stderr)
        return 1
    papers = json.loads(config.PAPERS_PATH.read_text())
    chunks = [c for p in papers for c in chunk_paper(p)]
    if not chunks:
        print("ERROR: no chunks produced.", file=sys.stderr)
        return 1
    print(f"{len(papers)} papers -> {len(chunks)} chunks. Embedding with {config.EMBED_MODEL} ...")

    model = SentenceTransformer(config.EMBED_MODEL)
    vectors = model.encode(
        [embedding_text(c) for c in chunks],
        batch_size=64,
        normalize_embeddings=True,  # unit vectors: inner product == cosine similarity
        show_progress_bar=True,
    )

    index = faiss.IndexFlatIP(vectors.shape[1])  # exact search; a few thousand vectors needs no ANN
    index.add(vectors)

    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(config.FAISS_PATH))
    config.CHUNKS_PATH.write_text(json.dumps(chunks, ensure_ascii=False))
    print(f"Saved {index.ntotal} vectors (dim {vectors.shape[1]}) -> {config.INDEX_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
