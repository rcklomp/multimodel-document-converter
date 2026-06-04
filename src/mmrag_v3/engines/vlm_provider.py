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

# Hard ceiling for the one-shot budget escalation that A1 triggers on a
# truncated (finish_reason=length) response. Bounded to respect the
# self-hosted-OOM constraint already noted on max_completion_tokens: a
# runaway budget on a dense page is what crashes mlx-vlm/vLLM. Past this,
# escalation stops and the partial body is surfaced (typed) for A4 repair.
TRUNCATION_ESCALATION_CAP = 16384

# A read timeout means the endpoint accepted the request and is generating, just
# too slowly to answer within timeout_seconds (measured on dense interior
# magazine pages 2026-06-03: median ~265s, several pages exceed any reasonable
# single-call budget). Unlike a connect failure, retrying a read timeout almost
# never helps - the same heavy page will blow the same timeout again, wasting a
# full timeout_seconds per retry. So read timeouts get their OWN small attempt
# cap, independent of max_retries (which still governs connect/connection
# faults, where a retry can legitimately recover a transient blip).
READ_TIMEOUT_MAX_ATTEMPTS = 1


# A3 (Charter Blocker A): the UIR page schema for guided/constrained JSON
# decoding. Mirrors the per-page prompt contract in vlm_native._build_schema_
# prompt (kept here because the provider owns wire-format concerns and
# importing vlm_native would be circular). Deliberately NOT strict/closed:
# `content` + `type` are the only required element fields and additional
# properties are allowed, so the constraint guarantees a well-formed JSON
# OBJECT with a typed `elements` array (eliminating the malformation class)
# without forcing the model to invent fields it would otherwise omit.
UIR_PAGE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "page_number": {"type": "integer"},
        "width": {"type": "integer"},
        "height": {"type": "integer"},
        "classification": {"type": "string", "enum": ["digital", "scanned", "hybrid"]},
        "elements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["text", "image", "table", "code", "form"],
                    },
                    "content": {"type": "string"},
                    "bbox": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                    "confidence": {"type": "number"},
                    "source_label": {"type": "string"},
                },
                "required": ["type", "content"],
            },
        },
    },
    "required": ["elements"],
}

# Structured-output modes (config.structured_output_mode). "json_schema" is the
# A3 constrained decode (mlx-vlm / vLLM OpenAI-compatible); "guided_json" is the
# vLLM extra-body form; "json_object" is the legacy weak hint; "off" sends no
# structured-output field (prompt-only contract).
STRUCTURED_JSON_SCHEMA = "json_schema"
STRUCTURED_GUIDED_JSON = "guided_json"
STRUCTURED_JSON_OBJECT = "json_object"
STRUCTURED_OFF = "off"
_STRUCTURED_MODES = (
    STRUCTURED_JSON_SCHEMA,
    STRUCTURED_GUIDED_JSON,
    STRUCTURED_JSON_OBJECT,
    STRUCTURED_OFF,
)


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


class VlmTruncationError(VlmProviderError):
    """The VLM hit its output-token ceiling (``finish_reason == "length"``).

    A *typed* signal for the dense-page failure mode (Charter Blocker A): a
    200 with non-empty but truncated content that downstream ``json.loads``
    silently rejects, mass-demoting dense pages to Docling. ``describe``
    escalates the token budget once and retries before raising this; the
    ``partial_content`` it carries is the longest truncated body seen, which
    the bounded JSON-repair stage (A4) can salvage to the last complete
    element instead of discarding the whole page. Semantic (not infra), so
    the router's circuit breaker does not trip on it.
    """

    def __init__(
        self,
        message: str,
        *,
        partial_content: str = "",
        finish_reason: str = "length",
    ) -> None:
        super().__init__(message)
        self.partial_content = partial_content
        self.finish_reason = finish_reason


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
    # Per-call read timeout. 600s, not the old 180s: the 2026-06-04 dense-doc
    # measurement (Combat Aircraft interior pages) showed dense pages generate
    # to the 8192-token budget cap, which at M5's ~33 tok/s is ~248s and CANNOT
    # finish inside 180s - 180s silently failed ~46% of interior pages while
    # 600s completed 13/13 (max observed 249s, with headroom for slower decode).
    # It is a ceiling, not a target: fast cloud endpoints still return in
    # seconds. Override via ``VLM_NATIVE_TIMEOUT``.
    timeout_seconds: float = 600.0
    max_retries: int = 3
    retry_backoff_seconds: float = 2.0
    temperature: float = 0.0
    # Bounded output budget — omlx/vLLM servers OOM and drop the
    # connection when asked to generate unbounded JSON for dense
    # academic pages. This is the FLOOR/default when a caller passes no
    # per-page override; the dense-page soak hit the old 4096 default and
    # truncated, so the floor is 8192 (the known-good overnight value).
    # The VLM path scales above this per page via describe(max_tokens=...)
    # (A2). Override via env or constructor if a backend needs more.
    max_completion_tokens: int = 8192
    # Sampling repetition penalty (mlx-vlm / mlx_lm native; >1.0 discourages
    # token repetition). The 2026-06-04 crucible found the VLM LOOPS on dense
    # pages - re-emitting the same paragraph until it hits the token cap (which
    # also inflates latency and under-extracts the real content). A mild penalty
    # curbs that at the source; the chunker's within-page dedup is the safety
    # net. Only sent when > 1.0; folded into the fail-open 400-strip so a backend
    # that rejects it degrades gracefully. Defaulted endpoint-aware in from_env.
    repetition_penalty: Optional[float] = None
    # Whether to send the OpenAI ``response_format={"type":"json_object"}``
    # hint. OpenAI / OpenRouter honor it; many self-hosted servers
    # (mlx-vlm, some vLLM builds) reject it with HTTP 400 instead of
    # ignoring it. The per-page prompt already mandates JSON, so omitting
    # the hint is safe. Defaulted endpoint-aware in ``from_env``. Only
    # consulted when ``structured_output_mode == "json_object"``.
    send_response_format: bool = True
    # A3 structured-output mode: "json_schema" (constrained decode, the
    # structural fix for Blocker A), "guided_json" (vLLM extra body),
    # "json_object" (legacy weak hint), or "off". A 400 that the endpoint
    # returns for an unsupported structured-output field is handled
    # fail-open in ``describe`` (strip the field and retry once), so a
    # backend that does not support the schema degrades to the prompt-only
    # contract rather than hard-failing. ``from_env`` resolves the default
    # endpoint-aware: OFF for self-hosted (mlx-vlm json_schema decode is too
    # slow on dense pages - 2026-06-03 live check), json_object for
    # OpenAI/OpenRouter. json_schema/guided_json are opt-in via env.
    structured_output_mode: str = STRUCTURED_JSON_OBJECT

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

        # ``VLM_NATIVE_TIMEOUT`` (seconds) - the per-call read timeout for the V3
        # extraction VLM. The default is now 600s (see the config field); this env
        # tunes it per-run without touching the CLI's separate (legacy-refiner)
        # --vlm-timeout. Raise it for slower-decode/text-dense workloads.
        timeout_raw = (os.environ.get("VLM_NATIVE_TIMEOUT") or "").strip()
        timeout_seconds = cls.timeout_seconds  # type: ignore[attr-defined]
        if timeout_raw:
            try:
                timeout_seconds = max(10.0, float(timeout_raw))
            except ValueError:
                pass

        # Repetition penalty: default a mild 1.1 for self-hosted (the mlx-vlm /
        # vLLM targets that loop on dense pages), off for cloud (OpenAI/OpenRouter
        # use frequency_penalty, a different param). ``VLM_NATIVE_REPETITION_
        # PENALTY`` overrides; a value <= 1.0 (or "off"/"none") disables it.
        is_hosted_cloud_rp = ("openrouter.ai" in endpoint) or ("openai.com" in endpoint)
        repetition_penalty: Optional[float] = None if is_hosted_cloud_rp else 1.1
        rp_raw = (os.environ.get("VLM_NATIVE_REPETITION_PENALTY") or "").strip().lower()
        if rp_raw in ("off", "none", "0"):
            repetition_penalty = None
        elif rp_raw:
            try:
                val = float(rp_raw)
                repetition_penalty = val if val > 1.0 else None
            except ValueError:
                pass

        # response_format hint: default on for OpenAI/OpenRouter (which
        # honor it), off for everything else (self-hosted mlx-vlm/vLLM
        # servers 400 on the bare json_object). Explicit
        # ``VLM_NATIVE_RESPONSE_FORMAT`` override wins: "json_object"/"1"/
        # "on"/"true" force on, "none"/"0"/"off"/"false" force off.
        send_rf = ("openrouter.ai" in endpoint) or ("openai.com" in endpoint)
        rf_raw = (os.environ.get("VLM_NATIVE_RESPONSE_FORMAT") or "").strip().lower()
        if rf_raw in ("json_object", "1", "on", "true", "yes"):
            send_rf = True
        elif rf_raw in ("none", "0", "off", "false", "no"):
            send_rf = False

        # A3 structured-output mode. Defaults are conservative based on the
        # 2026-06-03 live M5 bounded check: mlx-vlm ACCEPTS json_schema (no 400)
        # but its grammar-constrained decode is pathologically slow on dense
        # pages - one Combat Aircraft page exceeded the 180s read timeout and
        # tripped the breaker, whereas prompt-only returned the same pages
        # cleanly at ~88 s/page (0 truncation, 0 fallback). So self-hosted
        # defaults to OFF (prompt-only, the known-good pre-A3 behavior; the
        # per-page prompt still mandates JSON), and OpenAI/OpenRouter to the
        # json_object hint. json_schema / guided_json remain OPT-IN via
        # ``VLM_NATIVE_STRUCTURED_OUTPUT`` for backends that decode them
        # efficiently (e.g. vLLM + xgrammar on the GX10). The override wins.
        is_hosted_cloud = ("openrouter.ai" in endpoint) or ("openai.com" in endpoint)
        mode = STRUCTURED_JSON_OBJECT if is_hosted_cloud else STRUCTURED_OFF
        so_raw = (os.environ.get("VLM_NATIVE_STRUCTURED_OUTPUT") or "").strip().lower()
        if so_raw in _STRUCTURED_MODES:
            mode = so_raw

        return cls(
            endpoint=endpoint.rstrip("/"),
            model=model,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            max_completion_tokens=max_tokens,
            send_response_format=send_rf,
            structured_output_mode=mode,
            repetition_penalty=repetition_penalty,
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

    def _structured_output_fields(self) -> dict:
        """Payload fragment that constrains the decode per ``structured_output_mode``.

        Returns the request-body key(s) to merge into the chat-completions
        payload. ``{}`` when no structured output applies. A1/A4 still apply on
        top: a constrained decode can hit the token cap (valid-but-incomplete),
        so json_schema PAIRS with truncation handling, it does not replace it.
        """
        mode = self.config.structured_output_mode
        if mode == STRUCTURED_JSON_SCHEMA:
            return {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "uir_page", "schema": UIR_PAGE_JSON_SCHEMA},
                }
            }
        if mode == STRUCTURED_GUIDED_JSON:
            return {"guided_json": UIR_PAGE_JSON_SCHEMA}
        if mode == STRUCTURED_JSON_OBJECT and self.config.send_response_format:
            return {"response_format": {"type": "json_object"}}
        return {}

    def describe(
        self,
        image_bytes: bytes,
        prompt: str,
        *,
        mime: str = "image/png",
        response_format_json: bool = True,
        max_tokens: Optional[int] = None,
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
            max_tokens: Per-call output budget override (A2). When the caller
                has a cheap per-page density estimate it passes the scaled
                budget here; the value is floored at the config default and
                capped at ``TRUNCATION_ESCALATION_CAP`` so a single page can
                neither under-run the default nor OOM the server. ``None``
                uses the config default unchanged.

        Returns:
            Raw assistant message content. The caller deserializes.

        Raises:
            VlmProviderError: On unrecoverable transport / API errors.
        """
        data_uri = self.encode_image_bytes(image_bytes, mime=mime)

        budget = self.config.max_completion_tokens
        if max_tokens is not None:
            # Floor at the config default (never below the known-good value),
            # cap at the OOM ceiling shared with the truncation escalation.
            budget = max(budget, min(int(max_tokens), TRUNCATION_ESCALATION_CAP))

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
            "max_tokens": budget,
        }
        # A3: constrain the decode (json_schema/guided_json) when configured.
        # Plus the optional repetition penalty (#2). Both are "optional" params a
        # backend may reject with a 400; ``optional_params_active`` lets the loop
        # strip them and retry on a 400 (fail-open).
        structured_fields = self._structured_output_fields() if response_format_json else {}
        payload.update(structured_fields)
        if self.config.repetition_penalty and self.config.repetition_penalty > 1.0:
            payload["repetition_penalty"] = float(self.config.repetition_penalty)
        optional_params_active = bool(structured_fields) or ("repetition_penalty" in payload)

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
        # A1 (Charter Blocker A): longest truncated body seen + whether the
        # one-shot budget escalation has fired. A finish_reason=length 200 is
        # truncation, not success: keep the partial, escalate the budget once,
        # retry, then surface a typed VlmTruncationError for A4 repair.
        best_partial = ""
        budget_escalated = False
        read_timeout_attempts = 0
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
                # A read timeout (server is generating, just too slowly) gets a
                # small dedicated cap: retrying repeats the full timeout for a
                # page that is simply too heavy. ConnectTimeout is BOTH a
                # ReadTimeout-sibling and a ConnectionError, so exclude it here
                # (a connect failure can recover and keeps the full max_retries).
                if isinstance(exc, requests.exceptions.ReadTimeout) and not isinstance(
                    exc, requests.exceptions.ConnectTimeout
                ):
                    read_timeout_attempts += 1
                    if read_timeout_attempts >= READ_TIMEOUT_MAX_ATTEMPTS:
                        logger.warning(
                            "VLM read timeout after %d attempt(s) at %.0fs; not "
                            "retrying (a heavier retry would just repeat the wait)",
                            read_timeout_attempts,
                            self.config.timeout_seconds,
                        )
                        break
            else:
                if response.status_code == 200:
                    try:
                        body = response.json()
                        choice0 = body["choices"][0]
                        content = choice0["message"]["content"]
                    except (ValueError, KeyError, IndexError, TypeError) as exc:
                        raise VlmProviderError(f"Malformed VLM response shape: {exc}") from exc
                    finish = choice0.get("finish_reason")
                    if not content or not str(content).strip():
                        last_error = VlmProviderError(
                            f"Empty VLM content (finish_reason={finish!r})"
                        )
                        # 200 reached the model; empty output is semantic.
                        last_was_infra = False
                        logger.warning(
                            "VLM returned empty content (attempt %d/%d, " "finish_reason=%s)",
                            attempt,
                            self.config.max_retries,
                            finish,
                        )
                    elif finish == "length":
                        # A1: truncation. Non-empty but cut at the token
                        # ceiling -> json.loads would fail downstream. Retain
                        # the longest partial, escalate the budget ONCE, retry.
                        text = str(content)
                        if len(text) > len(best_partial):
                            best_partial = text
                        if not budget_escalated:
                            budget_escalated = True
                            escalated = min(payload["max_tokens"] * 2, TRUNCATION_ESCALATION_CAP)
                            if escalated > payload["max_tokens"]:
                                payload["max_tokens"] = escalated
                            last_error = VlmProviderError(
                                "VLM output truncated (finish_reason=length)"
                            )
                            last_was_infra = False
                            logger.warning(
                                "VLM TRUNCATED (finish_reason=length, attempt %d/%d); "
                                "escalating max_tokens to %d and retrying",
                                attempt,
                                self.config.max_retries,
                                payload["max_tokens"],
                            )
                        else:
                            # Escalation already spent and still truncated.
                            # Stop retrying truncation; surface typed below.
                            logger.warning(
                                "VLM still truncated after budget escalation "
                                "(attempt %d/%d); surfacing typed truncation for repair",
                                attempt,
                                self.config.max_retries,
                            )
                            break
                    else:
                        return str(content)
                elif response.status_code in (408, 429, 500, 502, 503, 504):
                    last_error = VlmProviderError(
                        f"Retryable VLM status {response.status_code}: " f"{response.text[:200]}"
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
                elif response.status_code == 400 and optional_params_active:
                    # A3 + #2 fail-open: the endpoint rejected an optional param
                    # (structured-output field or repetition_penalty). Strip them
                    # and retry with the plain prompt-only request rather than
                    # hard-failing the page.
                    optional_params_active = False
                    payload.pop("response_format", None)
                    payload.pop("guided_json", None)
                    payload.pop("repetition_penalty", None)
                    last_error = VlmProviderError(
                        f"optional params rejected (400): {response.text[:200]}"
                    )
                    last_was_infra = False
                    logger.warning(
                        "VLM endpoint rejected optional params (400, attempt "
                        "%d/%d); retrying without them (plain prompt-only request)",
                        attempt,
                        self.config.max_retries,
                    )
                else:
                    raise VlmProviderError(
                        f"VLM status {response.status_code}: " f"{response.text[:500]}"
                    )

            if attempt < self.config.max_retries:
                time.sleep(self.config.retry_backoff_seconds * attempt)

        # A1: a retained partial body means the terminal cause was truncation,
        # not an outage. Surface it typed (with the partial) so the caller can
        # repair to the last complete element instead of discarding the page.
        if best_partial:
            raise VlmTruncationError(
                f"VLM output truncated (finish_reason=length) after "
                f"{self.config.max_retries} attempts; {len(best_partial)} chars "
                "of partial content retained for repair",
                partial_content=best_partial,
                finish_reason="length",
            )

        err_cls = VlmInfraError if last_was_infra else VlmProviderError
        raise err_cls(f"VLM call failed after {self.config.max_retries} attempts: {last_error}")

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
