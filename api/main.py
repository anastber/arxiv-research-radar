"""FastAPI app.  Run:  uvicorn api.main:app --port 8000

Cost/abuse controls on /ask, in the order they apply:
  1. request validation  (question length is capped)
  2. per-visitor rate limit (slowapi, ASK_RATE_LIMIT, default 5/hour)
  3. global daily cap    (SQLite counter, DAILY_CAP, default 200/day)
  4. retrieval threshold (nothing relevant -> the LLM is never called)
  5. max output tokens on the LLM call
"""
import json
import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi.errors import RateLimitExceeded

import config
from api import refresh, usage
from api.ratelimit import client_ip, limiter, rate_limit_handler
from rag.answer import answer_question
from rag.llm import AnswerError
from rag.retriever import IndexNotBuiltError, Retriever

logger = logging.getLogger("api")

# What the client is told for each failure kind (details go to the server log only).
ANSWER_ERRORS = {
    "config": (500, "The demo is misconfigured on the server side. Please try again later."),
    "rate_limit": (503, "The AI provider is busy right now. Please try again in a minute."),
    "unavailable": (503, "The AI provider is temporarily unavailable. Please try again shortly."),
    "bad_request": (502, "The AI provider couldn't process that question. Try rephrasing it."),
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    usage._connect().close()  # create tables early
    # Serve the index on disk right away if there is one; the refresh thread builds or renews it.
    try:
        app.state.retriever = Retriever()
        app.state.papers = refresh.paper_list(json.loads(config.PAPERS_PATH.read_text()))
    except (IndexNotBuiltError, FileNotFoundError):
        app.state.retriever, app.state.papers = None, []
    stop = threading.Event()
    if config.REFRESH_HOURS > 0:
        threading.Thread(target=refresh.loop, args=(app, stop), daemon=True).start()
    yield
    stop.set()


app = FastAPI(title="arXiv Research Radar", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_handler)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=config.MAX_QUESTION_CHARS)


class CitedPaper(BaseModel):
    id: int
    arxiv_id: str
    title: str
    url: str


class AskResponse(BaseModel):
    answer: str
    cited_papers: list[CitedPaper]
    truncated: bool


@app.get("/health")
def health(request: Request):
    status = "ok" if request.app.state.retriever is not None else "building_index"
    return {"status": status, "papers": len(request.app.state.papers)}


@app.get("/papers")
def papers(request: Request):
    return request.app.state.papers


@app.post("/ask", response_model=AskResponse)
@limiter.limit(config.ASK_RATE_LIMIT)
def ask(request: Request, body: AskRequest):
    # Plain `def` on purpose: embedding + the LLM call block, so FastAPI runs this in a threadpool.
    ip, question = client_ip(request), body.question.strip()
    retriever = request.app.state.retriever
    if retriever is None:  # fresh container: the first index build takes a minute or two
        raise HTTPException(status_code=503,
                            detail="The paper index is being built. Please try again in a minute.")

    day = usage.reserve_daily_slot(config.DAILY_CAP)
    if day is None:
        usage.log_query(ip, question, "daily_cap")
        return JSONResponse(
            {"error": "daily_cap", "detail": "Demo limit reached, please try again tomorrow."},
            status_code=429,
        )

    try:
        result = answer_question(question, retriever)
    except AnswerError as e:
        usage.release_daily_slot(day)  # no answer delivered, so don't burn a slot
        usage.log_query(ip, question, f"error:{e.kind}")
        logger.error("answer failed (%s): %s", e.kind, e, exc_info=e.__cause__)
        status, message = ANSWER_ERRORS[e.kind]
        raise HTTPException(status_code=status, detail=message) from e

    called_llm = result["usage"]["input_tokens"] > 0
    status = "ok" if called_llm else "no_context"
    usage.log_query(ip, question, status, result["usage"], len(result["cited_papers"]))
    return result
