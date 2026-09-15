"""
Tests for lichess/player.py

Covers:
- ai_side determination from gameFull
- AI only moves on its own turn
- Double-processing guard
- _game_over_trigger results
- capture flag on quip events (for SenseRobot TTS delay)
"""
import chess
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import lichess.player as player_module
from lichess.player import _game_over_trigger, play_game


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_gamefull(white_id: str, black_id: str, moves: str = "") -> dict:
    return {
        "type": "gameFull",
        "id": "testgame1",
        "white": {"id": white_id.lower(), "name": white_id},
        "black": {"id": black_id.lower(), "name": black_id},
        "state": {
            "type": "gameState",
            "moves": moves,
            "status": "started",
        },
    }


def make_gamestate(moves: str, status: str = "started") -> dict:
    return {
        "type": "gameState",
        "moves": moves,
        "status": status,
    }


# ── Helper: run play_game and collect events up to N events or until done ─────

async def collect_events(gen, limit=50):
    events = []
    async for event in gen:
        events.append(event)
        if len(events) >= limit:
            break
        if event.get("type") in ("done", "error", "declined", "timeout"):
            break
    return events


# ── Test 8: ai_side determined from bot username in gameFull ─────────────────

@pytest.mark.asyncio
async def test_ai_side_from_bot_username(monkeypatch):
    """ai_side=WHITE when bot is white_id; ai_side=BLACK when bot is black_id."""
    BOT = "chesspersonabot"
    monkeypatch.setattr(player_module, "LICHESS_BOT_USERNAME", BOT)

    async def fake_validate():
        return {"id": BOT, "username": "ChessPersonaBot", "title": "BOT"}

    async def fake_challenge(username, color="black"):
        return {"challenge": {"id": "game1"}}

    async def fake_stream_events():
        yield {"type": "gameStart", "game": {"gameId": "game1", "color": color}}

    # Test 8a: bot is White
    for bot_color, expected_side in [("white", chess.WHITE), ("black", chess.BLACK)]:
        color = bot_color

        gamefull_white = "chesspersonabot"
        gamefull_black = "human"
        if bot_color == "black":
            gamefull_white = "human"
            gamefull_black = "chesspersonabot"

        async def fake_stream_game(game_id):
            gf = make_gamefull(gamefull_white, gamefull_black)
            yield gf

        monkeypatch.setattr(player_module, "validate_bot_account", fake_validate)
        monkeypatch.setattr(player_module, "challenge_user", fake_challenge)
        monkeypatch.setattr(player_module, "stream_account_events", fake_stream_events)
        monkeypatch.setattr(player_module, "stream_game", fake_stream_game)
        monkeypatch.setattr(player_module, "_open_engine", AsyncMock(return_value=None))

        events = await collect_events(
            play_game("human", color=bot_color, senserobot_mode=False)
        )

        started = next((e for e in events if e["type"] == "started"), None)
        assert started is not None, f"No 'started' event for bot_color={bot_color}"
        assert started["ai_side"] == bot_color, f"Expected ai_side={bot_color}"


# ── Test 9: AI only moves on its own turn ─────────────────────────────────────

@pytest.mark.asyncio
async def test_ai_only_moves_on_its_turn(monkeypatch):
    """make_move should not be called when board.turn != ai_side."""
    BOT = "chesspersonabot"
    monkeypatch.setattr(player_module, "LICHESS_BOT_USERNAME", BOT)

    async def fake_validate():
        return {"id": BOT, "username": "ChessPersonaBot", "title": "BOT"}

    async def fake_challenge(username, color="black"):
        return {"challenge": {"id": "game1"}}

    async def fake_stream_events():
        yield {"type": "gameStart", "game": {"gameId": "game1"}}

    # Bot is Black; gameFull shows no moves → it's White's turn → bot should not move
    async def fake_stream_game(game_id):
        yield make_gamefull("human", "chesspersonabot", moves="")

    make_move_mock = AsyncMock()

    monkeypatch.setattr(player_module, "validate_bot_account", fake_validate)
    monkeypatch.setattr(player_module, "challenge_user", fake_challenge)
    monkeypatch.setattr(player_module, "stream_account_events", fake_stream_events)
    monkeypatch.setattr(player_module, "stream_game", fake_stream_game)
    monkeypatch.setattr(player_module, "make_move", make_move_mock)
    monkeypatch.setattr(player_module, "_open_engine", AsyncMock(return_value=None))

    await collect_events(play_game("human", color="black"))

    make_move_mock.assert_not_called()


# ── Test 10: physical move triggers exactly one AI response ──────────────────

@pytest.mark.asyncio
async def test_physical_move_triggers_one_ai_response(monkeypatch):
    """After one human move, make_move should be called exactly once."""
    BOT = "chesspersonabot"
    monkeypatch.setattr(player_module, "LICHESS_BOT_USERNAME", BOT)

    async def fake_validate():
        return {"id": BOT, "username": "ChessPersonaBot", "title": "BOT"}

    async def fake_challenge(username, color="black"):
        return {"challenge": {"id": "game1"}}

    async def fake_stream_events():
        yield {"type": "gameStart", "game": {"gameId": "game1"}}

    # Bot is Black; one White move played (e2e4)
    async def fake_stream_game(game_id):
        yield make_gamefull("human", "chesspersonabot", moves="e2e4")
        # After this, bot's turn — stream ends (simulates waiting)

    make_move_mock = AsyncMock()

    async def fake_best_move(engine, board):
        return chess.Move.from_uci("e7e5")

    monkeypatch.setattr(player_module, "validate_bot_account", fake_validate)
    monkeypatch.setattr(player_module, "challenge_user", fake_challenge)
    monkeypatch.setattr(player_module, "stream_account_events", fake_stream_events)
    monkeypatch.setattr(player_module, "stream_game", fake_stream_game)
    monkeypatch.setattr(player_module, "make_move", make_move_mock)
    monkeypatch.setattr(player_module, "_open_engine", AsyncMock(return_value=None))
    monkeypatch.setattr(player_module, "_best_move", fake_best_move)

    await collect_events(play_game("human", color="black"))

    make_move_mock.assert_called_once_with("game1", "e7e5")


# ── Test 11: duplicate gameState does not trigger double AI move ─────────────

@pytest.mark.asyncio
async def test_bot_move_not_processed_twice(monkeypatch):
    """Duplicate gameState with same moves → make_move called only once."""
    BOT = "chesspersonabot"
    monkeypatch.setattr(player_module, "LICHESS_BOT_USERNAME", BOT)

    async def fake_validate():
        return {"id": BOT, "username": "ChessPersonaBot", "title": "BOT"}

    async def fake_challenge(username, color="black"):
        return {"challenge": {"id": "game1"}}

    async def fake_stream_events():
        yield {"type": "gameStart", "game": {"gameId": "game1"}}

    # Duplicate gameState — same moves sent twice
    async def fake_stream_game(game_id):
        yield make_gamefull("human", "chesspersonabot", moves="e2e4")
        yield make_gamestate("e2e4")   # duplicate — same move count

    make_move_mock = AsyncMock()

    async def fake_best_move(engine, board):
        return chess.Move.from_uci("e7e5")

    monkeypatch.setattr(player_module, "validate_bot_account", fake_validate)
    monkeypatch.setattr(player_module, "challenge_user", fake_challenge)
    monkeypatch.setattr(player_module, "stream_account_events", fake_stream_events)
    monkeypatch.setattr(player_module, "stream_game", fake_stream_game)
    monkeypatch.setattr(player_module, "make_move", make_move_mock)
    monkeypatch.setattr(player_module, "_open_engine", AsyncMock(return_value=None))
    monkeypatch.setattr(player_module, "_best_move", fake_best_move)

    await collect_events(play_game("human", color="black"))

    # make_move should only be called once despite duplicate event
    assert make_move_mock.call_count == 1


# ── Test 12: SenseRobot mode — no /move prompt in SSE events ─────────────────

@pytest.mark.asyncio
async def test_senserobot_mode_no_human_move_prompt(monkeypatch):
    """In SenseRobot mode, no SSE event should instruct calling /move."""
    BOT = "chesspersonabot"
    monkeypatch.setattr(player_module, "LICHESS_BOT_USERNAME", BOT)

    async def fake_validate():
        return {"id": BOT, "username": "ChessPersonaBot", "title": "BOT"}

    async def fake_challenge(username, color="black"):
        return {"challenge": {"id": "game1"}}

    async def fake_stream_events():
        yield {"type": "gameStart", "game": {"gameId": "game1"}}

    async def fake_stream_game(game_id):
        yield make_gamefull("human", "chesspersonabot", moves="")

    monkeypatch.setattr(player_module, "validate_bot_account", fake_validate)
    monkeypatch.setattr(player_module, "challenge_user", fake_challenge)
    monkeypatch.setattr(player_module, "stream_account_events", fake_stream_events)
    monkeypatch.setattr(player_module, "stream_game", fake_stream_game)
    monkeypatch.setattr(player_module, "_open_engine", AsyncMock(return_value=None))

    events = await collect_events(play_game("human", color="black", senserobot_mode=True))

    # None of the events should have a type that implies calling /move
    for event in events:
        assert event.get("type") not in ("submit_move", "human_move_needed")


# ── Tests 13a-13e: _game_over_trigger results ─────────────────────────────────

def test_result_ai_white_wins():
    board = chess.Board()
    board.set_fen("8/8/8/8/8/8/8/R6K w - - 0 1")  # arbitrary; we'll mock result
    # Use a board in a checkmate position: Fool's mate
    board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    # Black queen gives checkmate — result is 0-1
    assert board.is_checkmate()
    trigger = _game_over_trigger(board, chess.WHITE)
    assert trigger == "game_over_human_wins"


def test_result_ai_black_wins():
    # Fool's mate: 0-1 result, ai_side = BLACK → robot wins
    board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    assert board.is_checkmate()
    trigger = _game_over_trigger(board, chess.BLACK)
    assert trigger == "game_over_robot_wins"


def test_result_human_wins_vs_ai_white():
    # 0-1 result: White (ai) loses → human wins
    board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    trigger = _game_over_trigger(board, chess.WHITE)
    assert trigger == "game_over_human_wins"


def test_result_human_wins_vs_ai_black():
    # Scholar's mate: 1-0, ai_side = BLACK → human wins
    board = chess.Board("r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4")
    assert board.is_checkmate()
    trigger = _game_over_trigger(board, chess.BLACK)
    assert trigger == "game_over_human_wins"


def test_result_draw():
    # Stalemate position — result is 1/2-1/2
    board = chess.Board("k7/8/1Q6/8/8/8/8/7K b - - 0 1")
    assert board.is_stalemate()
    trigger = _game_over_trigger(board, chess.WHITE)
    assert trigger == "draw"
    trigger2 = _game_over_trigger(board, chess.BLACK)
    assert trigger2 == "draw"


# ── Tests 16-18: capture flag on quip events ─────────────────────────────────

def _common_capture_test_patches(monkeypatch, bot="chesspersonadbot"):
    monkeypatch.setattr(player_module, "LICHESS_BOT_USERNAME", bot)

    async def fake_validate():
        return {"id": bot, "username": bot, "title": "BOT"}

    async def fake_challenge(username, color="black"):
        return {"challenge": {"id": "game1"}}

    async def fake_stream_events():
        yield {"type": "gameStart", "game": {"gameId": "game1"}}

    monkeypatch.setattr(player_module, "validate_bot_account", fake_validate)
    monkeypatch.setattr(player_module, "challenge_user", fake_challenge)
    monkeypatch.setattr(player_module, "stream_account_events", fake_stream_events)
    monkeypatch.setattr(player_module, "_open_engine", AsyncMock(return_value=None))
    monkeypatch.setattr(player_module, "make_move", AsyncMock())
    # Always emit a quip so capture flag is observable
    monkeypatch.setattr(player_module, "get_quip", lambda personality, trigger: "test quip")


@pytest.mark.asyncio
async def test_capture_quip_has_capture_true(monkeypatch):
    """AI capture move quip event must have capture=True.

    Position after e2e4 e7e5 d2d4: Black (bot) can play e5xd4 — a capture.
    """
    _common_capture_test_patches(monkeypatch)

    async def fake_stream_game(game_id):
        # 3 half-moves played; it's Black's turn
        yield make_gamefull("human", "chesspersonadbot", moves="e2e4 e7e5 d2d4")

    async def fake_best_move(engine, board):
        return chess.Move.from_uci("e5d4")   # pawn captures d4

    monkeypatch.setattr(player_module, "stream_game", fake_stream_game)
    monkeypatch.setattr(player_module, "_best_move", fake_best_move)

    events = await collect_events(play_game("human", color="black"))

    quip_events = [e for e in events if e["type"] == "quip"]
    # The quip emitted after the AI's own move (e5d4 capture) must have capture=True
    ai_move_quips = [e for e in quip_events if e.get("capture") is True]
    assert len(ai_move_quips) >= 1, (
        f"Expected at least one quip with capture=True, got: {quip_events}"
    )


@pytest.mark.asyncio
async def test_non_capture_quip_has_capture_false(monkeypatch):
    """AI non-capture move quip event must have capture=False.

    Position after e2e4: Black (bot) plays e7e5 — no capture.
    """
    _common_capture_test_patches(monkeypatch)

    async def fake_stream_game(game_id):
        yield make_gamefull("human", "chesspersonadbot", moves="e2e4")

    async def fake_best_move(engine, board):
        return chess.Move.from_uci("e7e5")   # pawn advance, no capture

    monkeypatch.setattr(player_module, "stream_game", fake_stream_game)
    monkeypatch.setattr(player_module, "_best_move", fake_best_move)

    events = await collect_events(play_game("human", color="black"))

    quip_events = [e for e in events if e["type"] == "quip"]
    for e in quip_events:
        assert e.get("capture") is not True, (
            f"Non-capture move quip must not have capture=True, got: {e}"
        )


@pytest.mark.asyncio
async def test_ai_check_quip_has_delay_flag(monkeypatch):
    """AI check move (non-capture) quip must have capture=True so browser delays 5s.

    After 1.e4 g5 2.d4 f5, bot (White) plays Qd1-h5+ — check through the
    now-open f7 diagonal, no piece captured.
    """
    _common_capture_test_patches(monkeypatch)

    async def fake_stream_game(game_id):
        # 4 half-moves: e2e4 g7g5 d2d4 f7f5 — White to move
        yield make_gamefull("chesspersonadbot", "human", moves="e2e4 g7g5 d2d4 f7f5")

    async def fake_best_move(engine, board):
        return chess.Move.from_uci("d1h5")   # Qh5+ — check, no capture

    monkeypatch.setattr(player_module, "stream_game", fake_stream_game)
    monkeypatch.setattr(player_module, "_best_move", fake_best_move)

    events = await collect_events(play_game("human", color="white"))

    quip_events = [e for e in events if e["type"] == "quip"]
    delay_quips = [e for e in quip_events if e.get("capture") is True]
    assert len(delay_quips) >= 1, (
        f"Expected capture=True on AI check (non-capture) quip, got: {quip_events}"
    )


@pytest.mark.asyncio
async def test_capture_and_check_uses_capture_flag(monkeypatch):
    """When an AI move is both a capture and gives check, quip has capture=True.

    After e2e4 d7d5 e4d5 (bot White captures d5 pawn) e7e6 d5e6 (bot captures e6,
    giving discovered check is complex to arrange simply, so we verify the flag
    via the new_moves replay path: stream a gameFull where the bot's move is a
    capture that happens to give check, using a constructed position.

    Simple approach: replay moves where Black (human) blundered a piece, bot
    captures it. The capture=True flag must be set regardless of which trigger
    fires (robot_check, robot_captures, etc.).
    """
    _common_capture_test_patches(monkeypatch)

    # After e2e4 e7e5 d2d4 e5d4 (already replayed in gameFull moves),
    # the bot (White this time) can recapture with c2c3... that's complex.
    # Simpler: put a completed capture+check into the moves string so it
    # is replayed through the new_moves loop and verify capture flag.
    #
    # Scholar's mate sequence ends with Qf7# (queen captures f7 and gives check).
    # 1.e4 e5 2.Bc4 Nc6 3.Qh5 Nf6?? 4.Qxf7#
    # moves: e2e4 e7e5 f1c4 b8c6 d1h5 g8f6 h5f7
    # White (bot) plays h5f7 — captures f7 pawn and gives checkmate.
    scholar_moves = "e2e4 e7e5 f1c4 b8c6 d1h5 g8f6"

    async def fake_stream_game(game_id):
        # Bot is White; 6 moves played, it's White's turn
        yield make_gamefull("chesspersonadbot", "human", moves=scholar_moves)

    async def fake_best_move(engine, board):
        return chess.Move.from_uci("h5f7")   # Qxf7# — capture + checkmate

    monkeypatch.setattr(player_module, "stream_game", fake_stream_game)
    monkeypatch.setattr(player_module, "_best_move", fake_best_move)

    events = await collect_events(play_game("human", color="white"))

    quip_events = [e for e in events if e["type"] == "quip"]
    # The AI's Qxf7# quip must have capture=True
    capture_quips = [e for e in quip_events if e.get("capture") is True]
    assert len(capture_quips) >= 1, (
        f"Expected capture=True on capture+checkmate quip, got: {quip_events}"
    )
