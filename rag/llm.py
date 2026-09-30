"""One `complete()` call over the Gemini API.

Turns the SDK's errors into AnswerError with a `kind`, so the API layer never needs to
know the details of the provider behind it.
"""
import os
from dataclasses import dataclass

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

import config


class AnswerError(RuntimeError):
    """LLM call failed. `kind` lets the API layer pick a status code and message:
    config | billing | rate_limit | unavailable | bad_request"""

    def __init__(self, message: str, kind: str):
        super().__init__(message)
        self.kind = kind


@dataclass
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    truncated: bool  # stopped because of the output-token cap


_client: genai.Client | None = None  # lazily created (tests can pre-fill this)


def _gemini_client() -> genai.Client:
    global _client
    if _client is None:
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise AnswerError("GEMINI_API_KEY is not set on the server.", "config")
        _client = genai.Client(
            api_key=key,
            http_options=genai_types.HttpOptions(
                timeout=30_000,  # milliseconds
                retry_options=genai_types.HttpRetryOptions(attempts=2),
            ),
        )
    return _client


def complete(system: str, user: str) -> Completion:
    client = _gemini_client()
    thinking = (
        genai_types.ThinkingConfig(thinking_level=config.GEMINI_THINKING_LEVEL)
        if config.GEMINI_THINKING_LEVEL else None
    )
    try:
        resp = client.models.generate_content(
            model=config.ANSWER_MODEL,
            contents=user,
            config=genai_types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=config.MAX_ANSWER_TOKENS + config.GEMINI_THINKING_HEADROOM,
                thinking_config=thinking,
            ),
        )
    except genai_errors.ClientError as e:  # 4xx
        if e.code == 429:
            raise AnswerError("Gemini quota/rate limit reached.", "rate_limit") from e
        if e.code in (401, 403, 404) or "api key" in str(e).lower():
            # bad/missing key, no access to the model, or a wrong model name: all our misconfiguration
            raise AnswerError(f"Gemini rejected the server's key or model ({e.code}).", "config") from e
        raise AnswerError("Gemini rejected the request.", "bad_request") from e
    except (genai_errors.ServerError, httpx.HTTPError) as e:  # 5xx, network, timeout
        raise AnswerError("Gemini is temporarily unavailable.", "unavailable") from e

    candidate = resp.candidates[0] if resp.candidates else None
    if candidate is None:  # the prompt itself was blocked
        raise AnswerError("Gemini blocked the request.", "bad_request")
    usage = resp.usage_metadata
    finish = str(candidate.finish_reason or "")
    parts = candidate.content.parts or [] if candidate.content else []
    text = "".join(p.text or "" for p in parts if not p.thought)
    if not text and "STOP" not in finish and "MAX_TOKENS" not in finish:  # SAFETY, RECITATION, ...
        raise AnswerError(f"Gemini returned no text (finish reason {finish}).", "bad_request")
    return Completion(
        text=text.strip(),
        input_tokens=(usage.prompt_token_count or 0) if usage else 0,
        # thinking tokens are billed/quota'd as output, so count them
        output_tokens=((usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)) if usage else 0,
        truncated="MAX_TOKENS" in finish,
    )
