"""
Provider chain: quota handling, fallback, retries, and honest attribution.

No network. Providers are stubbed, so these check the policy — which failures
are retried, which are not, and what the user is told — rather than any vendor.
"""
import time

import pytest

from analytics import llm


@pytest.fixture(autouse=True)
def _clean():
    llm.reset_cooldowns()
    yield
    llm.reset_cooldowns()


class Stub:
    """A provider that fails in a scripted way."""
    name = "gemini"

    def __init__(self, model, *, raises=None, text="ok", configured=True):
        self.model = model
        self.raises = raises
        self.text = text
        self.configured = configured
        self.calls = 0

    def available(self):
        return self.configured

    def generate(self, prompt, max_output_tokens):
        self.calls += 1
        if self.raises:
            raise self.raises
        return self.text


DAILY_429 = RuntimeError(
    "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your "
    "current quota', 'details': [{'violations': [{'quotaId': "
    "'GenerateRequestsPerDayPerProjectPerModel-FreeTier'}]}], "
    "'retryDelay': '47s'}}"
)
MINUTE_429 = RuntimeError(
    "429 RESOURCE_EXHAUSTED {'quotaId': "
    "'GenerateRequestsPerMinutePerProjectPerModel-FreeTier', 'retryDelay': '2s'}"
)
BUSY_503 = RuntimeError("503 UNAVAILABLE. The model is overloaded.")


# ─── Classification ───────────────────────────────────────────────────────────

def test_daily_quota_is_not_a_rate_limit():
    err = llm.classify(DAILY_429)
    assert isinstance(err, llm.QuotaExhausted)


def test_per_minute_limit_is_a_rate_limit():
    err = llm.classify(MINUTE_429)
    assert isinstance(err, llm.RateLimited)
    assert err.retry_after == 2.0


def test_503_is_transient():
    assert isinstance(llm.classify(BUSY_503), llm.ProviderUnavailable)


def test_retry_delay_is_read_from_the_error():
    assert llm.classify(DAILY_429).retry_after == 47.0


# ─── Quota policy ─────────────────────────────────────────────────────────────

def test_daily_quota_is_never_retried(monkeypatch):
    """The whole point: hammering an exhausted daily quota cannot help."""
    dead = Stub("m1", raises=DAILY_429)
    good = Stub("m2", text="answer")
    monkeypatch.setattr(llm, "build_chain", lambda: [dead, good])

    out = llm.generate("q")
    assert dead.calls == 1, f"exhausted model was called {dead.calls} times"
    assert out.text == "answer" and out.model == "m2"


def test_exhausted_model_is_skipped_on_later_calls(monkeypatch):
    dead = Stub("m1", raises=DAILY_429)
    good = Stub("m2", text="answer")
    monkeypatch.setattr(llm, "build_chain", lambda: [dead, good])

    llm.generate("first")
    llm.generate("second")
    assert dead.calls == 1, "cooldown did not prevent a second attempt"
    assert good.calls == 2


def test_cooldown_outlives_a_short_server_retry_hint(monkeypatch):
    """A daily quota claiming 'retry in 47s' is still a daily quota."""
    dead = Stub("m1", raises=DAILY_429)
    monkeypatch.setattr(llm, "build_chain", lambda: [dead, Stub("m2")])
    llm.generate("q")
    left = llm.cooldown_status()["gemini:m1"]
    assert left > 100, f"cooldown only {left}s; would retry far too soon"


def test_rate_limit_is_retried_then_succeeds(monkeypatch):
    calls = {"n": 0}

    class Flaky(Stub):
        def generate(self, prompt, max_output_tokens):
            calls["n"] += 1
            if calls["n"] == 1:
                raise MINUTE_429
            return "recovered"

    monkeypatch.setattr(llm, "build_chain", lambda: [Flaky("m1")])
    monkeypatch.setattr(llm, "RETRY_BACKOFF_S", 0.01)
    monkeypatch.setattr(llm, "MAX_HONOURED_RETRY_DELAY_S", 0.01)

    out = llm.generate("q")
    assert out.text == "recovered" and calls["n"] == 2


def test_transient_failure_retries_are_bounded(monkeypatch):
    always = Stub("m1", raises=BUSY_503)
    monkeypatch.setattr(llm, "build_chain", lambda: [always])
    monkeypatch.setattr(llm, "RETRY_BACKOFF_S", 0.01)

    with pytest.raises(llm.AllProvidersFailed):
        llm.generate("q")
    assert always.calls == llm.TRANSIENT_RETRIES + 1


# ─── Fallback and attribution ─────────────────────────────────────────────────

def test_result_names_the_model_that_actually_answered(monkeypatch):
    monkeypatch.setattr(llm, "build_chain",
                        lambda: [Stub("m1", raises=DAILY_429), Stub("m2", text="hi")])
    out = llm.generate("q")
    assert out.model == "m2" and out.provider == "gemini"
    assert out.used_fallback is True


def test_no_fallback_is_claimed_when_the_first_model_answers(monkeypatch):
    monkeypatch.setattr(llm, "build_chain", lambda: [Stub("m1", text="hi")])
    out = llm.generate("q")
    assert out.used_fallback is False
    assert [a["outcome"] for a in out.attempts] == ["ok"]


def test_local_provider_is_used_when_gemini_is_exhausted(monkeypatch):
    local = Stub("local-model", text="from local")
    local.name = "local"
    monkeypatch.setattr(llm, "build_chain",
                        lambda: [Stub("m1", raises=DAILY_429), local])
    out = llm.generate("q")
    assert out.provider == "local" and out.text == "from local"


def test_chain_is_gemini_only_when_no_local_is_configured(monkeypatch):
    monkeypatch.setattr(llm, "LOCAL_LLM_URL", "")
    assert all(p.name == "gemini" for p in llm.build_chain())


def test_local_provider_is_appended_when_configured(monkeypatch):
    monkeypatch.setattr(llm, "LOCAL_LLM_URL", "http://localhost:11434/v1")
    chain = llm.build_chain()
    assert chain[-1].name == "local"


# ─── Messages the user sees ───────────────────────────────────────────────────

def test_all_quota_gone_says_so_and_suggests_local(monkeypatch):
    monkeypatch.setattr(llm, "LOCAL_LLM_URL", "")
    monkeypatch.setattr(llm, "build_chain",
                        lambda: [Stub("m1", raises=DAILY_429), Stub("m2", raises=DAILY_429)])
    with pytest.raises(llm.AllProvidersFailed) as ei:
        llm.generate("q")
    msg = str(ei.value).lower()
    assert "quota" in msg and "local_llm_url" in msg


def test_missing_configuration_is_not_reported_as_quota(monkeypatch):
    """Claiming a quota was used up when no key was ever set is a lie."""
    monkeypatch.setattr(llm, "build_chain",
                        lambda: [Stub("m1", configured=False)])
    with pytest.raises(llm.AllProvidersFailed) as ei:
        llm.generate("q")
    msg = str(ei.value).lower()
    assert "not configured" in msg or "no language model is configured" in msg
    assert "quota" not in msg


def test_busy_models_report_busy_not_quota(monkeypatch):
    monkeypatch.setattr(llm, "build_chain", lambda: [Stub("m1", raises=BUSY_503)])
    monkeypatch.setattr(llm, "RETRY_BACKOFF_S", 0.01)
    with pytest.raises(llm.AllProvidersFailed) as ei:
        llm.generate("q")
    assert "busy" in str(ei.value).lower()


# ─── Thinking models ──────────────────────────────────────────────────────────

def test_empty_response_moves_on_without_retrying(monkeypatch):
    """A thinking model that spends its budget reasoning returns nothing.

    That is deterministic, so retrying only burns more of its daily 20.
    """
    empty = Stub("thinker", raises=llm.EmptyResponse("no text"))
    good = Stub("lite", text="answer")
    monkeypatch.setattr(llm, "build_chain", lambda: [empty, good])

    out = llm.generate("q")
    assert empty.calls == 1, f"empty model retried {empty.calls} times"
    assert out.model == "lite"


def test_empty_response_is_not_reported_as_busy(monkeypatch):
    monkeypatch.setattr(llm, "build_chain",
                        lambda: [Stub("m1", raises=llm.EmptyResponse("no text"))])
    with pytest.raises(llm.AllProvidersFailed) as ei:
        llm.generate("q")
    msg = str(ei.value).lower()
    assert "busy" not in msg and "quota" not in msg


def test_thinking_headroom_is_added_to_the_token_cap(monkeypatch):
    """Without headroom a thinking model hits the cap mid-thought."""
    seen = {}

    class Recorder(llm.GeminiProvider):
        def generate(self, prompt, max_output_tokens):
            seen["cap"] = max_output_tokens + llm.THINKING_HEADROOM_TOKENS
            return "ok"

    monkeypatch.setattr(llm, "build_chain", lambda: [Recorder("m1")])
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    llm.generate("q", max_output_tokens=100)
    assert seen["cap"] == 100 + llm.THINKING_HEADROOM_TOKENS
    assert llm.THINKING_HEADROOM_TOKENS >= 150, "measured overhead was 86-109 tokens"


def test_default_chain_leads_with_a_fast_model(monkeypatch):
    monkeypatch.delenv("GEMINI_MODELS", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    chain = llm._model_chain()
    assert chain[0] == "gemini-3.5-flash-lite"
    assert len(chain) >= 5, "chain length is the daily-quota headroom"
    assert "gemini-2.5-flash" not in chain, "that model 404s on this key"


def test_chain_is_configurable_by_env(monkeypatch):
    monkeypatch.setenv("GEMINI_MODELS", "alpha, beta ,gamma")
    assert llm._model_chain() == ["alpha", "beta", "gamma"]
