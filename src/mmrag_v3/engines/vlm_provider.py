"""VLM provider — OpenAI-compatible vision chat endpoint client.

Modular wrapper around the actual API call so the extraction adapter
stays agnostic of the concrete VLM backend. Works with any
OpenAI-compatible vision endpoint (local vLLM Qwen-VL, OpenAI,
Anthropic via OpenAI-compat proxy, omlx-server, etc.).

Responsibilities:
    * Base64-encode image payloads.
    * Build the multimodal chat-completions request.
    * Issue the HTTP call with bounded retries on transient errors.
    * Return the raw assistant text content. The adapter is responsible
      for JSON parsing — this layer does no schema-aware processing.
"""

from __future__ import annotations

import base64
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

import requests

logger = logging.getLogger(__name__)


DEFAULT_ENDPOINT_ENV = "VLM_NATIVE_ENDPOINT"
DEFAULT_MODEL_ENV = "VLM_NATIVE_MODEL"
DEFAULT_API_KEY_ENV = "VLM_NATIVE_API_KEY"

# Fallback defaults — used when the explicit VLM_NATIVE_* env vars are
# unset. Anchors Phase C on OpenRouter so the pipeline keeps working
# even when the local omlx-server is down or has no VLM loaded.
OPENROUTER_DEFAULT_ENDPOINT = "https://openrouter.ai/api/v1"
OPENROUTER_DEFAULT_MODEL = "qwen/qwen3-vl-8b-instruct"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"


class VlmProviderError(RuntimeError):
    """Raised when the VLM endpoint returns a non-recoverable error.

    This is the *semantic* base: malformed response shape, empty content,
    non-retryable 4xx, or exhausted retries on an application-level error.
    The page router may demote a single page that fails this way to the
    Docling CPU path without poisoning the rest of the run.
    """


class VlmInfraError(VlmProviderError):
    """Infrastructure / transport failure: the endpoint is unreachable.

    Raised when the terminal cause is a connect/read timeout, connection
    refused, or a gateway status (502/503/504/408). Distinct from the
    semantic base so the router can treat it as a CIRCUIT-BREAKER trip:
    propagate and halt the batch instead of silently falling back to
    Docling. A silent CPU fallback during a network outage fabricates
    pages that masquerade as VLM baselines and corrupts the whole run.
    """


@dataclass
class VlmProviderConfig:
    """Connection + behavior config for a VLM endpoint.

    All fields are explicit so call sites cannot accidentally depend on
    ambient env state. Use ``VlmProviderConfig.from_env()`` to populate
    from the documented env vars.
    """

    endpoint: str
    model: str
    api_key: Optional[str] = None
    timeout_seconds: float = 180.0
    max_retries: int = 3
    retry_backoff_seconds: float = 2.0
    temperature: float = 0.0
    # Bounded output budget — omlx/vLLM servers OOM and drop the
    # connection when asked to generate unbounded JSON for dense
    # academic pages. 4096 tokens covers a typical UIR page payload
    # while keeping server memory deterministic. Override via env or
    # constructor if a backend genuinely needs more.
    max_completion_tokens: int = 4096
    # Whether to send the OpenAI ``response_format={"type":"json_object"}``
    # hint. OpenAI / OpenRouter honor it; many self-hosted servers
    # (mlx-vlm, some vLLM builds) reject it with HTTP 400 instead of
    # ignoring it. The per-page prompt already mandates JSON, so omitting
    # the hint is safe. Defaulted endpoint-aware in ``from_env``.
    send_response_format: bool = True

    @classmethod
    def from_env(cls) -> "VlmProviderConfig":
        """Construct from env, defaulting to OpenRouter when nothing is set.

        Resolution order:
            * Endpoint:  ``VLM_NATIVE_ENDPOINT`` → OpenRouter default
            * Model:     ``VLM_NATIVE_MODEL``    → ``qwen/qwen3-vl-8b-instruct``
            * API key:   ``VLM_NATIVE_API_KEY`` → if endpoint is on
              openrouter.ai, fall back to ``OPENROUTER_API_KEY``

        Raises ``VlmProviderError`` only when the resolved endpoint
        needs an API key and none is available in the environment.
        """
        endpoint = (os.environ.get(DEFAULT_ENDPOINT_ENV) or "").strip()
        model = (os.environ.get(DEFAULT_MODEL_ENV) or "").strip()
        if not endpoint:
            endpoint = OPENROUTER_DEFAULT_ENDPOINT
        if not model:
            model = OPENROUTER_DEFAULT_MODEL

        api_key = os.environ.get(DEFAULT_API_KEY_ENV) or None
        if not api_key and "openrouter.ai" in endpoint:
            api_key = os.environ.get(OPENROUTER_API_KEY_ENV) or None
        if not api_key and "openrouter.ai" in endpoint:
            raise VlmProviderError(
                "OpenRouter endpoint selected but neither "
                f"{DEFAULT_API_KEY_ENV} nor {OPENROUTER_API_KEY_ENV} is set"
            )

        max_tokens_raw = (os.environ.get("VLM_NATIVE_MAX_TOKENS") or "").strip()
        max_tokens = cls.max_completion_tokens  # type: ignore[attr-defined]
        if max_tokens_raw:
            try:
                max_tokens = max(256, int(max_tokens_raw))
            except ValueError:
                pass

        # response_format hint: default on for OpenAI/OpenRouter (which
        # honor it), off for everything else (self-hosted mlx-vlm/vLLM
        # servers 400 on it). Explicit ``VLM_NATIVE_RESPONSE_FORMAT``
        # override wins: "json_object"/"1"/"on"/"true" force on,
        # "none"/"0"/"off"/"false" force off.
        send_rf = ("openrouter.ai" in endpoint) or ("openai.com" in endpoint)
        rf_raw = (os.environ.get("VLM_NATIVE_RESPONSE_FORMAT") or "").strip().lower()
        if rf_raw in ("json_object", "1", "on", "true", "yes"):
            send_rf = True
        elif rf_raw in ("none", "0", "off", "false", "no"):
            send_rf = False

        return cls(
            endpoint=endpoint.rstrip("/"),
            model=model,
            api_key=api_key,
            max_completion_tokens=max_tokens,
            send_response_format=send_rf,
        )


class VlmProvider:
    """Thin OpenAI-compatible vision chat client."""

    def __init__(self, config: VlmProviderConfig) -> None:
        self.config = config

    @staticmethod
    def encode_image_bytes(image_bytes: bytes, *, mime: str = "image/png") -> str:
        """Return a ``data:`` URI suitable for the OpenAI image_url field."""
        b64 = base64.b64encode(image_bytes).decode("ascii")
        return f"data:{mime};base64,{b64}"

    def describe(
        self,
        image_bytes: bytes,
        prompt: str,
        *,
        mime: str = "image/png",
        response_format_json: bool = True,
    ) -> str:
        """Send a single (prompt, image) pair to the VLM and return assistant text.

        Args:
            image_bytes: Raw page render bytes.
            prompt: System-style instruction the VLM should follow.
            mime: MIME type for the image (default PNG).
            response_format_json: When True, ask the endpoint to constrain
                output to a JSON object via the OpenAI ``response_format``
                hint. Endpoints that don't honor this still receive a
                strict instruction in the prompt itself.

        Returns:
            Raw assistant message content. The caller deserializes.

        Raises:
            VlmProviderError: On unrecoverable transport / API errors.
        """
        data_uri = self.encode_image_bytes(image_bytes, mime=mime)

        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                }
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_completion_tokens,
        }
        if response_format_json and self.config.send_response_format:
            payload["response_format"] = {"type": "json_object"}

        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        # OpenRouter best-practice attribution headers (optional but
        # documented). Harmless when the endpoint is not OpenRouter.
        if "openrouter.ai" in self.config.endpoint:
            headers.setdefault("HTTP-Referer", "https://github.com/mmrag-v3")
            headers.setdefault("X-Title", "mmrag-v3 Phase C VLM-native")

        base = self.config.endpoint.rstrip("/")
        suffix = "/chat/completions" if base.endswith("/v1") else "/v1/chat/completions"
        url = f"{base}{suffix}"

        last_error: Optional[Exception] = None
        # Tracks whether the *most recent* failure was an infrastructure
        # fault (node unreachable) vs a semantic one. Decides the class of
        # the terminal raise so the router can trip its circuit breaker.
        last_was_infra = False
        for attempt in range(1, self.config.max_retries + 1):
            try:
                response = requests.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=self.config.timeout_seconds,
                )
            except requests.RequestException as exc:
                last_error = exc
                # Connect/read timeout and connection refused mean the node
                # is unreachable -> infra fault. Other RequestExceptions
                # (bad URL, too many redirects) are config/semantic.
                last_was_infra = isinstance(
                    exc,
                    (requests.exceptions.Timeout, requests.exceptions.ConnectionError),
                )
                logger.warning(
                    "VLM transport failure (attempt %d/%d): %s",
                    attempt,
                    self.config.max_retries,
                    exc,
                )
            else:
                if response.status_code == 200:
                    try:
                        body = response.json()
                        content = body["choices"][0]["message"]["content"]
                    except (ValueError, KeyError, IndexError, TypeError) as exc:
                        raise VlmProviderError(
                            f"Malformed VLM response shape: {exc}"
                        ) from exc
                    if not content or not str(content).strip():
                        finish = body["choices"][0].get("finish_reason")
                        last_error = VlmProviderError(
                            f"Empty VLM content (finish_reason={finish!r})"
                        )
                        # 200 reached the model; empty output is semantic.
                        last_was_infra = False
                        logger.warning(
                            "VLM returned empty content (attempt %d/%d, "
                            "finish_reason=%s)",
                            attempt,
                            self.config.max_retries,
                            finish,
                        )
                    else:
                        return str(content)
                if response.status_code in (408, 429, 500, 502, 503, 504):
                    last_error = VlmProviderError(
                        f"Retryable VLM status {response.status_code}: "
                        f"{response.text[:200]}"
                    )
                    # Gateway/unavailable/request-timeout statuses mean the
                    # serving infra is down or saturated -> infra fault.
                    # 429 (rate limit) and 500 (app error) stay semantic.
                    last_was_infra = response.status_code in (408, 502, 503, 504)
                    logger.warning(
                        "VLM retryable status %d (attempt %d/%d)",
                        response.status_code,
                        attempt,
                        self.config.max_retries,
                    )
                else:
                    raise VlmProviderError(
                        f"VLM status {response.status_code}: "
                        f"{response.text[:500]}"
                    )

            if attempt < self.config.max_retries:
                time.sleep(self.config.retry_backoff_seconds * attempt)

        err_cls = VlmInfraError if last_was_infra else VlmProviderError
        raise err_cls(
            f"VLM call failed after {self.config.max_retries} attempts: {last_error}"
        )

    def probe_health(self, *, timeout_seconds: float = 10.0) -> bool:
        """Lightweight liveness probe: ``GET {base}/models`` -> True iff HTTP 200.

        Used by resilient batch harnesses to poll for endpoint recovery after a
        :class:`VlmInfraError` WITHOUT paying for a full inference call. Never
        raises: any transport error or non-200 status returns False ("still
        down"). ``/v1/models`` is the portable liveness route exposed by every
        OpenAI-compatible backend (vLLM, mlx-vlm, OpenAI, OpenRouter).

        NOTE: a 200 means the HTTP server is up and the model is listed; it does
        NOT prove inference works. The harness pairs this with a resume-attempt
        cap so an "HTTP-up-but-inference-dead" endpoint cannot loop forever.
        """
        base = self.config.endpoint.rstrip("/")
        suffix = "/models" if base.endswith("/v1") else "/v1/models"
        url = f"{base}{suffix}"
        headers = {}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        try:
            response = requests.get(url, headers=headers, timeout=timeout_seconds)
        except requests.RequestException as exc:
            logger.debug("VLM health probe failed (transport): %s", exc)
            return False
        if response.status_code == 200:
            return True
        logger.debug("VLM health probe non-200: %s", response.status_code)
        return False
