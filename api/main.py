"""FastAPI app.  Run:  uvicorn api.main:app --port 8000

Cost/abuse controls on /ask, in the order they apply:
  1. request validation  (question length is capped)
  2. per-IP rate limit   (slowapi, ASK_RATE_LIMIT, default 5/hour)
  3. global daily cap    (SQLite counter, DAILY_CAP, default 200/day)
  4. retrieval threshold (nothing relevant -> Claude is never called)
  5. max_tokens on the Claude call
"""
import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi.errors import RateLimitExceeded

import config
from api import usage
from api.ratelimit import client_ip, limiter, rate_limit_handler
from rag.answer import AnswerError, answer_question
from rag.retriever import Retriever

logger = logging.getLogger("api")

# What the client is told for each failure kind (details go to the server log only).
ANSWER_ERRORS = {
    "config": (500, "The demo is misconfigured on the server side. Please try again later."),
    "rate_limit": (503, "The AI provider is busy right now. Please try again in a minute."),
    "unavailable": (503, "The AI provider is temporarily unavailable. Please try again shortly."),
    "billing": (503, "The demo is temporarily unavailable. Please try again later."),
    "bad_request": (502, "The AI provider couldn't process that question. Try rephrasing it."),
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    usage._connect().close()                       # create tables early
    app.state.retriever = Retriever()              # fail fast if the index isn't built
    app.state.papers = [
        {**{k: p[k] for k in ("arxiv_id", "title", "authors", "published", "category", "abs_url", "pdf_url")},
         "has_full_text": bool(p.get("full_text"))}
        for p in json.loads(config.PAPERS_PATH.read_text())
    ]
    yield


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
    return {"status": "ok", "papers": len(request.app.state.papers)}


@app.get("/papers")
def papers(request: Request):
    return request.app.state.papers


@app.post("/ask", response_model=AskResponse)
@limiter.limit(config.ASK_RATE_LIMIT)
def ask(request: Request, body: AskRequest):
    # Plain `def` on purpose: embedding + the Claude call block, so FastAPI runs this in a threadpool.
    ip, question = client_ip(request), body.question.strip()

    day = usage.reserve_daily_slot(config.DAILY_CAP)
    if day is None:
        usage.log_query(ip, question, "daily_cap")
        return JSONResponse(
            {"error": "daily_cap", "detail": "Demo limit reached, please try again tomorrow."},
            status_code=429,
        )

    try:
        result = answer_question(question, request.app.state.retriever)
    except AnswerError as e:
        usage.release_daily_slot(day)  # no answer delivered, so don't burn a slot
        usage.log_query(ip, question, f"error:{e.kind}")
        logger.error("answer failed (%s): %s", e.kind, e, exc_info=e.__cause__)
        status, message = ANSWER_ERRORS[e.kind]
        raise HTTPException(status_code=status, detail=message)

    called_claude = result["usage"]["input_tokens"] > 0
    usage.log_query(ip, question, "ok" if called_claude else "no_context", result["usage"], len(result["cited_papers"]))
    return result
