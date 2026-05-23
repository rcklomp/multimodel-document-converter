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

# v2.14 Phase 4a: local-LLM HyDE provider. Endpoint history:
#   2026-05-22: Qwen2.5-14B-Instruct (retired)
#   2026-05-23: Qwen3.6-27B-FP8 + native MTP=3 (retired)
#   2026-05-23: Qwen3-30B-A3B-Instruct-2507-FP8 (ACTIVE — 30B total,
#               3B active per MoE pass; pure-instruct variant, no
#               <think> mode, but the chat template still accepts the
#               enable_thinking kwarg which we send defensively)
# Each model swap invalidates the prior Phase 0 calibration verdict
# per memory/feedback_gx10_deployment_guardrails.md (constraint #5).
# HyDE generation itself is leniency-robust — judge bias direction
# doesn't affect retrieval-target hypothetical answers.
VLLM_DEFAULT_URL = "http://10.0.10.239:8000/v1/chat/completions"
VLLM_DEFAULT_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"


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
    provider: str = "dashscope",
    model: str | None = None,
    url: str | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
) -> str:
    """Generate a single hypothetical answer to the query.

    Returns the generated string. On unrecoverable failure (network,
    parse, no choices in response), raises HydeError — callers should
    catch and fall back to the literal query for retrieval.

    `provider` selects the backend:
      - "dashscope" (default): cloud `qwen-max`. Requires DASHSCOPE_API_KEY.
      - "vllm": local OpenAI-compatible vLLM (defaults to the GX10 at
        http://10.0.10.239:8000 with Qwen/Qwen3.6-27B-FP8 + native
        MTP=3 as of 2026-05-23). No auth required by default; pass
        `api_key` if your endpoint gates on a bearer token.
        v2.14 Phase 4a addition.

    `model` and `url` override the provider's defaults. If omitted,
    each provider uses its own `DEFAULT_*` / `VLLM_DEFAULT_*` constants.
    """
    if url is None:
        url = VLLM_DEFAULT_URL if provider == "vllm" else DEFAULT_URL
    if model is None:
        model = VLLM_DEFAULT_MODEL if provider == "vllm" else DEFAULT_MODEL

    if provider == "dashscope" and not api_key:
        api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        if not api_key:
            raise HydeError(
                "generate_hypothetical_answer with provider='dashscope' "
                "requires api_key arg or DASHSCOPE_API_KEY env var"
            )
    elif provider == "vllm" and api_key is None:
        # vLLM defaults to no auth; allow empty bearer to be sent.
        api_key = os.environ.get("VLLM_API_KEY", "")

    request_payload: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(query=query)},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if provider == "vllm":
        # GX10 vLLM endpoint runs Qwen3 with `--reasoning-parser qwen3`,
        # which defaults to thinking mode. Thinking routes the answer into
        # `message.reasoning` and starves `message.content` of token budget,
        # so the parser at line ~150 sees empty content and the call raises
        # HydeError. Disable thinking so the answer lands in `content`.
        # Dashscope (cloud qwen-max) doesn't have this mode; don't send the
        # extension there.
        request_payload["chat_template_kwargs"] = {"enable_thinking": False}
    body = json.dumps(request_payload).encode("utf-8")

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=body, method="POST")
            if api_key:
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
