"""
Language-model access with a fallback chain.

Everything that needs prose — the coaching review and the in-game Q&A — goes
through `generate()` here rather than talking to a provider directly.

Why a chain. The Gemini free tier caps *generate_content* at 20 requests per
day, and the quota is per model: the observed 429 names
`GenerateRequestsPerDayPerProjectPerModel-FreeTier`, `limit: 20`. A handful of
questions plus a couple of reviews exhausts one model for the rest of the day.
Verified directly: with gemini-3.6-flash returning 429, gemini-3.5-flash-lite
answered normally. So the chain is several Gemini models followed by an
optional local model, and a model that reports its daily quota gone is skipped
until the cooldown expires instead of being retried.

Three failure kinds, treated differently:

  QuotaExhausted      daily allowance gone. Never retried; the model is put on
                      cooldown so later requests skip straight past it.
  RateLimited         too many requests too fast. Honours the server's
                      retryDelay if it gives one, otherwise short backoff.
  ProviderUnavailable 503 / overloaded / network. Bounded retries with backoff.

Local provider: set LOCAL_LLM_URL to any OpenAI-compatible endpoint (Ollama,
LM Studio, llama.cpp server). Leave it unset and the chain is Gemini-only —
nothing else changes. No paid service is contacted.

The result always records which provider and model actually answered, so the UI
can say so truthfully and never imply a fallback ran when it did not.
"""
import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# The SDK logs an AFC notice on every call regardless of how it is configured.
# We never use function calling, so the notice is pure noise in the timing logs.
logging.getLogger("google_genai.models").setLevel(logging.ERROR)

# ── Configuration (all optional; defaults keep current behaviour) ─────────────

def _model_chain() -> list[str]:
    raw = os.getenv("GEMINI_MODELS", "").strip()
    if raw:
        return [m.strip() for m in raw.split(",") if m.strip()]
    # Ordered fastest-first, because the whole chain is free-tier and the only
    # thing distinguishing these is latency and daily allowance. Each model
    # carries its own 20 requests/day, so the chain length is the headroom.
    #
    # Measured round-trip on a trivial prompt:
    #   3.5-flash-lite 0.4s | flash-lite-latest 1.0s | 3.1-flash-lite 5.2s
    #   flash-latest 2.4s | 3.7-flash 5.2s | 3.5-flash 15.1s
    #
    # The last three are "thinking" models: they spend ~90-110 tokens on
    # internal reasoning before emitting anything, which is why they sit at the
    # back and why MAX_OUTPUT_TOKENS carries THINKING_HEADROOM_TOKENS.
    # gemini-2.5-flash is deliberately absent: it 404s on this key.
    return [
        os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        "gemini-flash-lite-latest",
        "gemini-3.6-flash",
        "gemini-3.1-flash-lite",
        "gemini-flash-latest",
        "gemini-3.7-flash",
        "gemini-3.5-flash",
    ]


LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "").strip()
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "llama3.2").strip()
LOCAL_LLM_TIMEOUT_S = float(os.getenv("LOCAL_LLM_TIMEOUT_S", "60"))

REQUEST_TIMEOUT_S = float(os.getenv("LLM_REQUEST_TIMEOUT_S", "45"))
MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "400"))

# Thinking models consume output budget on internal reasoning before writing a
# single visible token — measured at 86-109 tokens on a trivial prompt. Without
# headroom they hit the cap mid-thought and return an empty response, which
# looks like a provider failure and silently drops them from the chain.
THINKING_HEADROOM_TOKENS = 250

TRANSIENT_RETRIES = 2
RETRY_BACKOFF_S = 1.5

# How long to skip a model whose daily quota is gone, when the server does not
# say. Free-tier daily quotas reset on a 24h boundary; an hour is long enough to
# stop hammering and short enough to recover without a restart.
DEFAULT_QUOTA_COOLDOWN_S = 3600.0
MAX_HONOURED_RETRY_DELAY_S = 120.0


# ── Errors ────────────────────────────────────────────────────────────────────

class LLMError(RuntimeError):
    """Base for anything the caller should surface."""


class QuotaExhausted(LLMError):
    """Daily allowance for this model is gone. Do not retry it."""
    def __init__(self, msg, retry_after: float | None = None):
        super().__init__(msg)
        self.retry_after = retry_after


class RateLimited(LLMError):
    def __init__(self, msg, retry_after: float | None = None):
        super().__init__(msg)
        self.retry_after = retry_after


class ProviderUnavailable(LLMError):
    """Transient: 503, overloaded, network blip. Worth retrying."""


class EmptyResponse(ProviderUnavailable):
    """
    Model answered but produced no text.

    Deterministic for a given model and prompt — a thinking model that spent
    its whole output budget reasoning will do it again. Retrying just burns
    more of that model's daily allowance, so move to the next one instead.
    """


class AllProvidersFailed(LLMError):
    def __init__(self, msg, attempts: list[dict] | None = None):
        super().__init__(msg)
        self.attempts = attempts or []


@dataclass
class LLMResult:
    text: str
    provider: str          # "gemini" | "local"
    model: str
    latency_s: float
    attempts: list[dict] = field(default_factory=list)

    @property
    def used_fallback(self) -> bool:
        """True only when something other than the first choice answered."""
        return len(self.attempts) > 1


# ── Cooldown bookkeeping ──────────────────────────────────────────────────────

_cooldowns: dict[str, float] = {}      # "provider:model" -> epoch seconds


def _key(provider: str, model: str) -> str:
    return f"{provider}:{model}"


def _cooling(provider: str, model: str) -> float:
    """Seconds remaining on this model's cooldown, 0 if it is usable."""
    until = _cooldowns.get(_key(provider, model), 0.0)
    return max(0.0, until - time.time())


def _set_cooldown(provider: str, model: str, seconds: float):
    _cooldowns[_key(provider, model)] = time.time() + seconds
    log.warning("[llm] %s on cooldown for %.0fs", _key(provider, model), seconds)


def reset_cooldowns():
    """Test hook."""
    _cooldowns.clear()


def cooldown_status() -> dict[str, float]:
    now = time.time()
    return {k: round(v - now, 1) for k, v in _cooldowns.items() if v > now}


# ── Error classification ──────────────────────────────────────────────────────

_RETRY_DELAY = re.compile(r"'retryDelay':\s*'(\d+(?:\.\d+)?)s'")
_RETRY_IN = re.compile(r"retry in (\d+(?:\.\d+)?)s", re.I)


def _retry_after(blob: str) -> float | None:
    for pattern in (_RETRY_DELAY, _RETRY_IN):
        m = pattern.search(blob)
        if m:
            return float(m.group(1))
    return None


def classify(e: Exception) -> LLMError:
    """
    Turn a provider exception into one of our three kinds.

    The distinction that matters: a per-DAY quota must not be retried, while a
    per-minute rate limit should be. Gemini reports both as 429, and the
    quotaId is what separates them.
    """
    blob = str(e)
    low = blob.lower()
    delay = _retry_after(blob)

    if "429" in blob or "resource_exhausted" in low:
        if "perday" in low.replace("_", "").replace("-", ""):
            return QuotaExhausted(
                "Daily free-tier quota for this model is used up.", delay
            )
        if "perminute" in low.replace("_", "").replace("-", "") or (delay and delay <= 60):
            return RateLimited("Sending requests too quickly.", delay)
        # Unlabelled 429: treat as quota so we stop rather than hammer.
        return QuotaExhausted("Quota for this model is used up.", delay)

    if "503" in blob or "unavailable" in low or "overloaded" in low:
        return ProviderUnavailable("Model is busy.")
    if "500" in blob or "internal" in low:
        return ProviderUnavailable("Provider internal error.")
    if isinstance(e, (TimeoutError, urllib.error.URLError, ConnectionError, OSError)):
        return ProviderUnavailable(f"Could not reach provider: {e}")
    return LLMError(f"{type(e).__name__}: {blob[:200]}")


# ── Providers ─────────────────────────────────────────────────────────────────

class GeminiProvider:
    name = "gemini"

    def __init__(self, model: str):
        self.model = model

    def available(self) -> bool:
        return bool(os.getenv("GEMINI_API_KEY"))

    def generate(self, prompt: str, max_output_tokens: int) -> str:
        from google import genai
        from google.genai import types as genai_types

        client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY"),
            http_options=genai_types.HttpOptions(timeout=int(REQUEST_TIMEOUT_S * 1000)),
        )
        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                max_output_tokens=max_output_tokens + THINKING_HEADROOM_TOKENS,
            ),
        )
        text = getattr(response, "text", None)
        if not text or not text.strip():
            # Usually a thinking model that spent its whole budget reasoning.
            # Not fatal: fall through to the next model in the chain.
            raise EmptyResponse(
                f"{self.model} returned no text (likely exhausted its output "
                f"budget on internal reasoning)."
            )
        return text.strip()


class LocalProvider:
    """
    Any OpenAI-compatible chat endpoint (Ollama, LM Studio, llama.cpp).

    Uses urllib rather than adding a dependency. Disabled unless LOCAL_LLM_URL
    is set, so the chain is unchanged when nothing is configured.
    """
    name = "local"

    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def available(self) -> bool:
        return bool(self.base_url)

    def generate(self, prompt: str, max_output_tokens: int) -> str:
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_output_tokens,
            "temperature": 0.7,
            "stream": False,
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=LOCAL_LLM_TIMEOUT_S) as resp:
            payload = json.loads(resp.read().decode())
        try:
            text = payload["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, AttributeError) as e:
            raise LLMError(f"Unexpected local LLM response shape: {e}") from e
        if not text:
            raise LLMError("Local model returned no text.")
        return text


def build_chain() -> list:
    chain = [GeminiProvider(m) for m in _model_chain()]
    if LOCAL_LLM_URL:
        chain.append(LocalProvider(LOCAL_LLM_URL, LOCAL_LLM_MODEL))
    return chain


# ── Entry point ───────────────────────────────────────────────────────────────

def generate(prompt: str, *, max_output_tokens: int = MAX_OUTPUT_TOKENS,
             label: str = "llm") -> LLMResult:
    """
    Try each provider in turn; return the first success.

    Raises AllProvidersFailed with a per-attempt breakdown if none answer.
    """
    attempts: list[dict] = []
    chain = build_chain()

    for provider in chain:
        model = provider.model
        if not provider.available():
            attempts.append({"provider": provider.name, "model": model,
                             "outcome": "not_configured"})
            continue

        cooling = _cooling(provider.name, model)
        if cooling:
            attempts.append({"provider": provider.name, "model": model,
                             "outcome": "cooldown", "seconds_left": round(cooling)})
            log.info("[llm] %s skipping %s:%s (%.0fs cooldown left)",
                     label, provider.name, model, cooling)
            continue

        for attempt in range(TRANSIENT_RETRIES + 1):
            t0 = time.perf_counter()
            try:
                text = provider.generate(prompt, max_output_tokens)
                dt = time.perf_counter() - t0
                attempts.append({"provider": provider.name, "model": model,
                                 "outcome": "ok", "latency_s": round(dt, 2)})
                log.info("[llm] %s answered by %s:%s in %.2fs (attempt %d of chain position %d)",
                         label, provider.name, model, dt, attempt + 1,
                         chain.index(provider) + 1)
                return LLMResult(text=text, provider=provider.name, model=model,
                                 latency_s=dt, attempts=attempts)
            except Exception as raw:
                err = classify(raw)
                dt = time.perf_counter() - t0

                if isinstance(err, QuotaExhausted):
                    cooldown = err.retry_after or DEFAULT_QUOTA_COOLDOWN_S
                    # A daily quota that claims a short retry is still a daily
                    # quota; don't come back in 47 seconds and burn the retry.
                    cooldown = max(cooldown, DEFAULT_QUOTA_COOLDOWN_S)
                    _set_cooldown(provider.name, model, cooldown)
                    attempts.append({"provider": provider.name, "model": model,
                                     "outcome": "quota_exhausted",
                                     "latency_s": round(dt, 2)})
                    log.warning("[llm] %s quota exhausted on %s:%s after %.2fs",
                                label, provider.name, model, dt)
                    break                       # next provider, no retry

                if isinstance(err, RateLimited):
                    wait = min(err.retry_after or RETRY_BACKOFF_S,
                               MAX_HONOURED_RETRY_DELAY_S)
                    if attempt < TRANSIENT_RETRIES:
                        log.info("[llm] %s rate limited on %s:%s, waiting %.1fs",
                                 label, provider.name, model, wait)
                        time.sleep(wait)
                        continue
                    attempts.append({"provider": provider.name, "model": model,
                                     "outcome": "rate_limited"})
                    break

                if isinstance(err, EmptyResponse):
                    attempts.append({"provider": provider.name, "model": model,
                                     "outcome": "empty_response",
                                     "latency_s": round(dt, 2)})
                    log.warning("[llm] %s %s:%s returned nothing — next model",
                                label, provider.name, model)
                    break                       # no retry: it will repeat

                if isinstance(err, ProviderUnavailable):
                    if attempt < TRANSIENT_RETRIES:
                        wait = RETRY_BACKOFF_S * (attempt + 1)
                        log.info("[llm] %s transient on %s:%s, retrying in %.1fs",
                                 label, provider.name, model, wait)
                        time.sleep(wait)
                        continue
                    attempts.append({"provider": provider.name, "model": model,
                                     "outcome": "unavailable"})
                    break

                attempts.append({"provider": provider.name, "model": model,
                                 "outcome": "error", "detail": str(err)[:160]})
                log.warning("[llm] %s error on %s:%s: %s",
                            label, provider.name, model, err)
                break

    raise AllProvidersFailed(_summarise(attempts), attempts)


def _summarise(attempts: list[dict]) -> str:
    """A message a person can act on, naming what actually happened."""
    outcomes = {a["outcome"] for a in attempts}
    if not attempts:
        return "No language model is configured."
    # Nothing was even tried: say that, rather than blaming a quota that was
    # never consumed.
    if outcomes == {"not_configured"}:
        return ("No language model is configured. Set GEMINI_API_KEY, "
                "or LOCAL_LLM_URL for a local model.")
    if outcomes <= {"quota_exhausted", "cooldown", "not_configured"}:
        base = ("Every configured model has used up its free daily quota. "
                "It resets on a 24-hour cycle.")
        if not LOCAL_LLM_URL:
            base += " Set LOCAL_LLM_URL to fall back to a local model."
        return base
    if "unavailable" in outcomes or "rate_limited" in outcomes:
        return "The models are busy right now. Try again in a moment."
    if outcomes <= {"empty_response", "quota_exhausted", "cooldown", "not_configured"}:
        return ("No model produced a usable answer. Try again, or raise "
                "LLM_MAX_OUTPUT_TOKENS if this keeps happening.")
    detail = next((a.get("detail") for a in attempts if a.get("detail")), None)
    return f"Could not generate a response. {detail or ''}".strip()
