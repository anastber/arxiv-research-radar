"""One `complete()` call over the supported LLM providers (Gemini, Anthropic).

Each provider function turns the provider's SDK errors into AnswerError with a `kind`,
so the API layer never needs to know which provider is behind it.
"""
import os
from dataclasses import dataclass

import anthropic
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


_clients: dict = {}  # lazily created, one per provider (tests can pre-fill this)


def complete(system: str, user: str) -> Completion:
    if config.LLM_PROVIDER == "gemini":
        return _complete_gemini(system, user)
    return _complete_anthropic(system, user)


# ---------------------------------------------------------------- Gemini

def _gemini_client() -> genai.Client:
    if "gemini" not in _clients:
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise AnswerError("GEMINI_API_KEY is not set on the server.", "config")
        _clients["gemini"] = genai.Client(
            api_key=key,
            http_options=genai_types.HttpOptions(
                timeout=30_000,  # milliseconds
                retry_options=genai_types.HttpRetryOptions(attempts=2),
            ),
        )
    return _clients["gemini"]


def _complete_gemini(system: str, user: str) -> Completion:
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
    text = "".join(p.text or "" for p in (candidate.content.parts or []) if not p.thought) if candidate.content else ""
    if not text and "STOP" not in finish and "MAX_TOKENS" not in finish:  # SAFETY, RECITATION, ...
        raise AnswerError(f"Gemini returned no text (finish reason {finish}).", "bad_request")
    return Completion(
        text=text.strip(),
        input_tokens=(usage.prompt_token_count or 0) if usage else 0,
        # thinking tokens are billed/quota'd as output, so count them
        output_tokens=((usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)) if usage else 0,
        truncated="MAX_TOKENS" in finish,
    )


# ---------------------------------------------------------------- Anthropic

def _anthropic_client() -> anthropic.Anthropic:
    if "anthropic" not in _clients:
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise AnswerError("ANTHROPIC_API_KEY is not set on the server.", "config")
        _clients["anthropic"] = anthropic.Anthropic(timeout=30.0, max_retries=2)
    return _clients["anthropic"]


def _complete_anthropic(system: str, user: str) -> Completion:
    client = _anthropic_client()
    try:
        response = client.messages.create(
            model=config.ANSWER_MODEL,
            max_tokens=config.MAX_ANSWER_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
    except anthropic.AuthenticationError as e:
        raise AnswerError("The server's Anthropic API key was rejected.", "config") from e
    except anthropic.RateLimitError as e:
        raise AnswerError("The AI provider is rate limiting us right now.", "rate_limit") from e
    except anthropic.BadRequestError as e:
        # Anthropic reports an empty account balance as a 400; that's our problem, not the user's.
        if "credit balance" in e.message.lower():
            raise AnswerError("The Anthropic account is out of credits.", "billing") from e
        raise AnswerError("The AI provider rejected the request.", "bad_request") from e
    except (anthropic.APIConnectionError, anthropic.APIStatusError) as e:
        raise AnswerError("The AI provider is temporarily unavailable.", "unavailable") from e

    return Completion(
        text="".join(b.text for b in response.content if b.type == "text").strip(),
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        truncated=response.stop_reason == "max_tokens",
    )
