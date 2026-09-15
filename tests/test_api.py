"""
Tests for lichess/api.py

Covers:
- Bot API endpoint usage (not Board API)
- Token validation and error safety
- validate_bot_account() account guard rules
- SSE stream error propagation (no Python traceback)
- Windows CP1252 safety of all print/log strings in the game path
"""
import pytest
import respx
import httpx
import json

import lichess.api as api
from lichess.api import (
    make_move,
    stream_game,
    validate_bot_account,
    LICHESS_BOT_USERNAME,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _set_token(monkeypatch, value="test_bot_token_abc"):
    monkeypatch.setattr(api, "LICHESS_BOT_TOKEN", value)


def _set_bot_username(monkeypatch, value="ChessPersonaBot"):
    monkeypatch.setattr(api, "LICHESS_BOT_USERNAME", value)


def _set_senserobot_username(monkeypatch, value="Sainju"):
    monkeypatch.setattr(api, "SENSEROBOT_LICHESS_USERNAME", value)


# ── Test 1: make_move uses /api/bot/ not /api/board/ ─────────────────────────

@pytest.mark.asyncio
async def test_make_move_uses_bot_endpoint(monkeypatch):
    _set_token(monkeypatch)
    with respx.mock:
        route = respx.post(
            "https://lichess.org/api/bot/game/abc123/move/e2e4"
        ).mock(return_value=httpx.Response(200, json={"ok": True}))

        result = await make_move("abc123", "e2e4")

    assert result is True
    assert route.called
    # Ensure the /api/board/ URL was not used
    assert "/api/bot/game/" in str(route.calls[0].request.url)
    assert "/api/board/" not in str(route.calls[0].request.url)


# ── Test 2: stream_game uses /api/bot/game/stream/ ───────────────────────────

@pytest.mark.asyncio
async def test_stream_game_uses_bot_endpoint(monkeypatch):
    _set_token(monkeypatch)
    game_event = {"type": "gameFull", "white": {"id": "bot"}, "black": {"id": "human"}, "state": {"moves": "", "status": "started"}}
    stream_body = json.dumps(game_event) + "\n"

    with respx.mock:
        route = respx.get(
            "https://lichess.org/api/bot/game/stream/abc123"
        ).mock(return_value=httpx.Response(200, text=stream_body))

        events = []
        async for event in stream_game("abc123"):
            events.append(event)

    assert route.called
    assert "/api/bot/game/stream/" in str(route.calls[0].request.url)
    assert "/api/board/" not in str(route.calls[0].request.url)
    assert len(events) == 1
    assert events[0]["type"] == "gameFull"


# ── Test 3: missing bot token raises RuntimeError without token in message ────

@pytest.mark.asyncio
async def test_missing_bot_token_raises(monkeypatch):
    monkeypatch.setattr(api, "LICHESS_BOT_TOKEN", "")

    with pytest.raises(RuntimeError) as exc_info:
        await make_move("abc123", "e2e4")

    assert "LICHESS_BOT_TOKEN" in str(exc_info.value)
    # Token value must not appear in error message (it's empty here, but guard the pattern)
    assert "Bearer" not in str(exc_info.value)


# ── Test 4: non-BOT account is rejected ──────────────────────────────────────

@pytest.mark.asyncio
async def test_non_bot_account_rejected(monkeypatch):
    _set_token(monkeypatch)
    _set_bot_username(monkeypatch, "chesspersonabot")
    _set_senserobot_username(monkeypatch, "sainju")

    account_data = {"id": "chesspersonabot", "username": "ChessPersonaBot", "title": None}

    with respx.mock:
        respx.get("https://lichess.org/api/account").mock(
            return_value=httpx.Response(200, json=account_data)
        )
        with pytest.raises(ValueError) as exc_info:
            await validate_bot_account()

    assert "not a BOT account" in str(exc_info.value)
    assert "upgrade.py" in str(exc_info.value)


# ── Test 5: bot cannot challenge itself ──────────────────────────────────────

@pytest.mark.asyncio
async def test_bot_cannot_challenge_itself(monkeypatch):
    """play_game should yield an error if opponent == bot username."""
    _set_token(monkeypatch)
    _set_bot_username(monkeypatch, "chesspersonabot")
    _set_senserobot_username(monkeypatch, "sainju")

    from lichess.player import play_game
    import lichess.player as player_module

    # Patch validate_bot_account to pass
    async def fake_validate():
        return {"id": "chesspersonabot", "username": "ChessPersonaBot", "title": "BOT"}

    monkeypatch.setattr(player_module, "validate_bot_account", fake_validate)
    monkeypatch.setattr(player_module, "LICHESS_BOT_USERNAME", "chesspersonabot")

    events = []
    async for event in play_game("ChessPersonaBot", senserobot_mode=False):
        events.append(event)

    assert any(e["type"] == "error" for e in events)
    error_texts = [e["text"] for e in events if e["type"] == "error"]
    assert any("cannot challenge itself" in t for t in error_texts)


# ── Test 6: sainju cannot be bot ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_sainju_cannot_be_bot(monkeypatch):
    _set_token(monkeypatch, "some_token")
    _set_bot_username(monkeypatch, "sainju")
    _set_senserobot_username(monkeypatch, "sainju")

    account_data = {"id": "sainju", "username": "Sainju", "title": "BOT"}

    with respx.mock:
        respx.get("https://lichess.org/api/account").mock(
            return_value=httpx.Response(200, json=account_data)
        )
        with pytest.raises(ValueError) as exc_info:
            await validate_bot_account()

    # Should fail either on the senserobot check or the forbidden account check
    err = str(exc_info.value).lower()
    assert "sainju" in err or "senserobot" in err or "not allowed" in err


# ── Test 7: rohan_sainju cannot be bot ───────────────────────────────────────

@pytest.mark.asyncio
async def test_rohan_sainju_cannot_be_bot(monkeypatch):
    _set_token(monkeypatch, "some_token")
    _set_bot_username(monkeypatch, "rohan_sainju")
    _set_senserobot_username(monkeypatch, "sainju")

    account_data = {"id": "rohan_sainju", "username": "Rohan_Sainju", "title": "BOT"}

    with respx.mock:
        respx.get("https://lichess.org/api/account").mock(
            return_value=httpx.Response(200, json=account_data)
        )
        with pytest.raises(ValueError) as exc_info:
            await validate_bot_account()

    assert "rohan_sainju" in str(exc_info.value).lower()


# ── Test 14: secrets not in error text ───────────────────────────────────────

@pytest.mark.asyncio
async def test_secrets_not_in_error_text(monkeypatch):
    secret_token = "super_secret_token_xyz_12345"
    monkeypatch.setattr(api, "LICHESS_BOT_TOKEN", secret_token)
    _set_bot_username(monkeypatch, "chesspersonabot")
    _set_senserobot_username(monkeypatch, "sainju")

    account_data = {"id": "chesspersonabot", "username": "ChessPersonaBot", "title": None}

    with respx.mock:
        respx.get("https://lichess.org/api/account").mock(
            return_value=httpx.Response(200, json=account_data)
        )
        with pytest.raises(ValueError) as exc_info:
            await validate_bot_account()

    assert secret_token not in str(exc_info.value)


# ── Test 15: missing token yields SSE error event, not traceback ─────────────

@pytest.mark.asyncio
async def test_missing_token_yields_error_event(monkeypatch):
    """When LICHESS_BOT_TOKEN is missing, play_game should yield type=error."""
    import lichess.player as player_module

    # Force validate_bot_account to raise RuntimeError (token not set)
    async def fake_validate_raises():
        raise RuntimeError("LICHESS_BOT_TOKEN not set")

    monkeypatch.setattr(player_module, "validate_bot_account", fake_validate_raises)

    from lichess.player import play_game

    events = []
    async for event in play_game("someuser"):
        events.append(event)

    assert len(events) >= 1
    assert events[0]["type"] == "error"
    assert "Traceback" not in events[0].get("text", "")
    assert "LICHESS_BOT_TOKEN" in events[0].get("text", "")


# ── Test 16: no non-CP1252 characters in game-path print/log strings ──────────

def test_no_non_cp1252_in_debug_prints():
    """
    Regression for Windows UnicodeEncodeError (cp1252 codec).
    Scans all string literals passed to print() and log.*() in player.py
    and verifies they encode cleanly under cp1252.
    """
    import ast, pathlib

    source = pathlib.Path("lichess/player.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    bad = []
    for node in ast.walk(tree):
        # Match print(...) and log.info/debug/warn/error(...)
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_print = isinstance(func, ast.Name) and func.id == "print"
        is_log = (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "log"
        )
        if not (is_print or is_log):
            continue
        # Check every string constant argument
        for arg in node.args:
            for subnode in ast.walk(arg):
                if isinstance(subnode, ast.Constant) and isinstance(subnode.value, str):
                    try:
                        subnode.value.encode("cp1252")
                    except UnicodeEncodeError as e:
                        bad.append(
                            f"Line {node.lineno}: {subnode.value!r} -> {e}"
                        )

    assert bad == [], "Non-CP1252 characters found in print/log calls:\n" + "\n".join(bad)


# ── Tests 17-19: frontend SSE event structure + delay logic ──────────────────

def test_capture_quip_json_is_boolean_true():
    """
    Verify Python json.dumps serialises capture=True as JSON true (not 1 or "True").
    Simulates the SSE bytes the browser receives and parses.
    """
    import json

    # Exact structure player.py emits for an AI capture quip
    event = {"type": "quip", "text": "Ha! I'll take that.", "personality": "Cocky", "capture": True}
    sse_line = f"data: {json.dumps(event)}\n\n"

    # Simulate JS: line.slice(6) → JSON.parse(...)
    payload = sse_line.strip()
    assert payload.startswith("data: ")
    parsed = json.loads(payload[6:])

    assert parsed["capture"] is True,  "capture must be Python True"
    assert type(parsed["capture"]) is bool, "capture must be bool, not int or str"

    # Non-capture quip
    nc_event = {"type": "quip", "text": "Interesting.", "personality": "Cocky", "capture": False}
    nc_parsed = json.loads(f"data: {json.dumps(nc_event)}"[6:])
    assert nc_parsed["capture"] is False
    assert type(nc_parsed["capture"]) is bool


def test_frontend_delay_logic_using_exact_sse_json():
    """
    Simulate the exact JS delay calculation:
        const captureDelay = (isSenseRobotMode && e.capture === true) ? 5000 : 0;

    Uses the exact JSON payloads produced by json.dumps() as they would arrive
    over SSE. Covers all four combinations of SenseRobot mode × capture flag.
    """
    import json

    def js_capture_delay(sse_json_str: str, is_senserobot_mode: bool) -> int:
        """Python equivalent of the JS delay expression."""
        e = json.loads(sse_json_str)
        return 5000 if (is_senserobot_mode and e.get("capture") is True) else 0

    capture_payload    = json.dumps({"type": "quip", "text": "...", "personality": "Cocky", "capture": True})
    no_capture_payload = json.dumps({"type": "quip", "text": "...", "personality": "Cocky", "capture": False})

    # SenseRobot mode ON, capture move → 5000 ms
    assert js_capture_delay(capture_payload, True)  == 5000

    # SenseRobot mode ON, non-capture move → 0 ms (immediate)
    assert js_capture_delay(no_capture_payload, True)  == 0

    # SenseRobot mode OFF, capture move → 0 ms (immediate — no delay outside SenseRobot)
    assert js_capture_delay(capture_payload, False) == 0

    # SenseRobot mode OFF, non-capture move → 0 ms
    assert js_capture_delay(no_capture_payload, False) == 0


def test_senserobot_client_delays_capture_quip():
    """
    Verify senserobot_client.py delays capture quips with time.sleep(5).
    Checks the source code directly without executing it.
    """
    import ast, pathlib

    source = pathlib.Path("senserobot_client.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    # Find the quip branch: look for the time.sleep call near the quip handler
    sleep_calls = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "sleep"
        ):
            # Check that the argument is 5
            if node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == 5:
                sleep_calls.append(node.lineno)

    assert sleep_calls, (
        "senserobot_client.py must contain time.sleep(5) for the capture delay"
    )

    # Also verify the capture guard: event.get("capture") is True
    source_text = source
    assert 'event.get("capture") is True' in source_text, (
        'senserobot_client.py must check event.get("capture") is True before sleeping'
    )
