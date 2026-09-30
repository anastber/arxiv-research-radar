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

# --- Retrieval ---
TOP_K = int(os.getenv("TOP_K", "6"))                    # chunks sent to the LLM
MAX_CHUNKS_PER_PAPER = 2                                # keeps context spread across papers
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.3"))        # cosine sim below this = "not covered"

# --- Answer generation ---
# API key comes from GEMINI_API_KEY in the environment, never from code.
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "gemini-3.5-flash-lite")
MAX_ANSWER_TOKENS = int(os.getenv("MAX_ANSWER_TOKENS", "400"))  # hard cap on answer length per query
# Gemini counts "thinking" tokens against its output cap, so it gets extra headroom on top of
# MAX_ANSWER_TOKENS, and thinking is turned down. Empty GEMINI_THINKING_LEVEL = don't send the setting.
GEMINI_THINKING_HEADROOM = int(os.getenv("GEMINI_THINKING_HEADROOM", "600"))
GEMINI_THINKING_LEVEL = os.getenv("GEMINI_THINKING_LEVEL", "low")

# --- API: rate limiting, daily cap, usage log ---
ASK_RATE_LIMIT = os.getenv("ASK_RATE_LIMIT", "5/hour")            # per IP, slowapi syntax
DAILY_CAP = int(os.getenv("DAILY_CAP", "200"))                    # total /ask calls per UTC day
MAX_QUESTION_CHARS = int(os.getenv("MAX_QUESTION_CHARS", "500"))
# How many reverse proxies sit in front of the app (Render/Fly/HF Spaces = 1, local = 0).
# Used to find the real client IP in X-Forwarded-For without trusting client-supplied entries.
TRUSTED_PROXY_HOPS = int(os.getenv("TRUSTED_PROXY_HOPS", "0"))
USAGE_DB_PATH = Path(os.getenv("USAGE_DB_PATH", str(DATA_DIR / "usage.db")))
# Rough $/million tokens for the cost estimate in `python -m api.usage`. Default: Gemini free
# tier = $0. Override if you're on a paid Gemini tier.
PRICE_INPUT_PER_MTOK = float(os.getenv("PRICE_INPUT_PER_MTOK", "0"))
PRICE_OUTPUT_PER_MTOK = float(os.getenv("PRICE_OUTPUT_PER_MTOK", "0"))

# --- Frontend ---
API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
