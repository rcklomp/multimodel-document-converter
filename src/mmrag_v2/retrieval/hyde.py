"""v2.12 Phase 3 — HyDE (Hypothetical Document Embeddings).

Generates a hypothetical answer to the user's query via Dashscope
`qwen-max`, then uses the answer's embedding for retrieval rather
than the question's. The intuition: answers and the chunks that
contain them share vocabulary + style; questions and chunks do not.

Trade-off: adds ~0.7-1.2s latency per query (qwen-max generation
call), plus ~$0.001 per query in Dashscope spend. Default-on or
opt-in is a per-deployment decision driven by latency tolerance.

Failure modes:
  - 5xx / timeout from Dashscope: fall back to literal-query embed.
  - Refusal / "I don't know" output: still returns the literal text;
    the retrieval pipeline embeds whatever string came back.
  - Invalid JSON / parse error: same fallback.

The single-shot design (one hypothetical per query, no paraphrase
fan-out) was chosen for cost + simplicity. Multi-query HyDE or
RAG-fusion-style query rewriting is a v2.13 candidate.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

DEFAULT_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"
DEFAULT_MODEL = "qwen-max"
DEFAULT_TEMPERATURE = 0.3  # some diversity, not full hallucination
DEFAULT_MAX_TOKENS = 250  # ~50-100 words of hypothetical answer
DEFAULT_TIMEOUT = 45
DEFAULT_RETRIES = 3


SYSTEM_PROMPT = (
    "You write hypothetical answers to user questions. Your answer will "
    "be used to retrieve relevant documents from a knowledge base; the "
    "answer itself does NOT need to be factually correct. Write the "
    "answer in the same language as the question. Be confident — write "
    "as if you knew the answer. 50-100 words, single paragraph, no "
    "preamble. No phrases like 'I don't know' or 'I'm not sure'."
)


USER_PROMPT_TEMPLATE = "Question: {query}\n\nAnswer:"


class HydeError(RuntimeError):
    """Raised when the HyDE generation call fails unrecoverably."""


def generate_hypothetical_answer(
    query: str,
    api_key: str | None = None,
    *,
    model: str = DEFAULT_MODEL,
    url: str = DEFAULT_URL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
) -> str:
    """Generate a single hypothetical answer to the query.

    Returns the generated string. On unrecoverable failure (network,
    parse, no choices in response), raises HydeError — callers should
    catch and fall back to the literal query for retrieval.
    """
    if not api_key:
        api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        if not api_key:
            raise HydeError(
                "generate_hypothetical_answer requires api_key arg or "
                "DASHSCOPE_API_KEY env var"
            )

    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(query=query)},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode("utf-8")

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=body, method="POST")
            req.add_header("Authorization", f"Bearer {api_key}")
            req.add_header("Content-Type", "application/json")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            try:
                detail = e.read().decode("utf-8", errors="replace")[:200]
            except Exception:
                detail = ""
            raise HydeError(f"HyDE HTTP {e.code}: {detail}") from e
        except (urllib.error.URLError, ConnectionError, OSError, TimeoutError) as e:
            last_err = e
            time.sleep(2 ** attempt)
            continue

        # Parse response.
        choices = payload.get("choices") or []
        if not choices:
            raise HydeError("HyDE response had no `choices`")
        message = (choices[0] or {}).get("message") or {}
        content = (message.get("content") or "").strip()
        if not content:
            raise HydeError("HyDE response had empty `content`")
        return content

    raise HydeError(
        f"HyDE failed after {retries} retries; last error: {last_err}"
    ) from last_err


def generate_with_fallback(query: str, api_key: str | None = None, **kwargs) -> str:
    """Like `generate_hypothetical_answer` but never raises — falls back
    to the literal query if generation fails.

    Production retrieval paths typically want this variant: HyDE adds
    quality lift, but a single failed call shouldn't fail the whole
    retrieval. The literal-query embed is a graceful degradation.
    """
    try:
        return generate_hypothetical_answer(query, api_key, **kwargs)
    except HydeError:
        return query
