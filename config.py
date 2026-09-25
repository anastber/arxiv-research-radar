"""Central config. Everything tunable comes from environment variables (see .env.example)."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
PAPERS_PATH = DATA_DIR / "papers.json"

ARXIV_QUERY = os.getenv("ARXIV_QUERY", 'all:"retrieval augmented generation"')
ARXIV_MAX_PAPERS = int(os.getenv("ARXIV_MAX_PAPERS", "20"))

# --- Chunking / embedding / index ---
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
INDEX_DIR = DATA_DIR / "index"
FAISS_PATH = INDEX_DIR / "faiss.index"
CHUNKS_PATH = INDEX_DIR / "chunks.json"
CHUNK_WORDS = 160      # target chunk size; MiniLM truncates input beyond ~256 tokens
CHUNK_OVERLAP = 1      # sentences repeated between adjacent chunks within a section
