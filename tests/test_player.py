"""
Tests for lichess/player.py

Covers:
- ai_side determination from gameFull
- AI only moves on its own turn
- Double-processing guard
- _game_over_trigger results
- capture flag on quip events (for SenseRobot TTS delay)
"""
from pathlib import Path
import time
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

    async def fake_best_move(engine, board, move_time=0.5):
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

    async def fake_best_move(engine, board, move_time=0.5):
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

    async def fake_best_move(engine, board, move_time=0.5):
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

    async def fake_best_move(engine, board, move_time=0.5):
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

    async def fake_best_move(engine, board, move_time=0.5):
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

    async def fake_best_move(engine, board, move_time=0.5):
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


# ─── Difficulty ───────────────────────────────────────────────────────────────

def test_resolve_difficulty_known_keys():
    for key, cfg in player_module.DIFFICULTIES.items():
        resolved, resolved_cfg = player_module.resolve_difficulty(key)
        assert resolved == key
        assert resolved_cfg is cfg


def test_resolve_difficulty_is_case_and_space_insensitive():
    assert player_module.resolve_difficulty("  BEGINNER ")[0] == "beginner"


def test_resolve_difficulty_falls_back_on_unknown():
    for bad in ("nonsense", "", None, "grandmaster"):
        assert player_module.resolve_difficulty(bad)[0] == player_module.DEFAULT_DIFFICULTY


def test_every_difficulty_is_well_formed():
    for key, cfg in player_module.DIFFICULTIES.items():
        assert set(cfg) == {"label", "elo", "move_time", "blurb"}
        assert cfg["move_time"] > 0
        # elo None means "full strength"; otherwise within Stockfish's range
        assert cfg["elo"] is None or 1320 <= cfg["elo"] <= 3190


def test_configure_strength_clamps_to_engine_range():
    """An Elo below the engine's floor is clamped, not passed through."""
    class FakeOption:
        def __init__(self, lo, hi): self.min, self.max = lo, hi

    class FakeEngine:
        options = {"UCI_Elo": FakeOption(1320, 3190), "UCI_LimitStrength": FakeOption(None, None)}
        def __init__(self): self.configured = None
        def configure(self, opts): self.configured = opts

    eng = FakeEngine()
    player_module._configure_strength(eng, 800)
    assert eng.configured == {"UCI_LimitStrength": True, "UCI_Elo": 1320}

    eng2 = FakeEngine()
    player_module._configure_strength(eng2, 9999)
    assert eng2.configured == {"UCI_LimitStrength": True, "UCI_Elo": 3190}


def test_configure_strength_none_leaves_engine_untouched():
    class FakeEngine:
        options = {}
        def __init__(self): self.configured = None
        def configure(self, opts): self.configured = opts

    eng = FakeEngine()
    player_module._configure_strength(eng, None)
    assert eng.configured is None


def test_configure_strength_survives_engine_without_elo_support():
    """An engine lacking UCI_Elo must not raise — it just plays full strength."""
    class FakeEngine:
        options = {}
        def __init__(self): self.configured = None
        def configure(self, opts): self.configured = opts

    eng = FakeEngine()
    player_module._configure_strength(eng, 1600)   # must not raise
    assert eng.configured is None


# ─── Quip delay is scheduled by consumers, not slept in the game loop ─────────

@pytest.mark.asyncio
async def test_move_event_precedes_quip_for_ai_move(monkeypatch):
    """The `move` event must be emitted before the quip that follows it.

    `move` describes something that already happened. If it trails a delayed
    quip, it overwrites the fresher "your turn" status the `fen` event set.
    """
    _common_capture_test_patches(monkeypatch)

    async def fake_stream_game(game_id):
        yield make_gamefull("chesspersonadbot", "human", moves="e2e4 g7g5 d2d4 f7f5")

    async def fake_best_move(engine, board, move_time=0.5):
        return chess.Move.from_uci("d1h5")   # Qh5+ — guarantees a quip trigger

    monkeypatch.setattr(player_module, "stream_game", fake_stream_game)
    monkeypatch.setattr(player_module, "_best_move", fake_best_move)

    events = await collect_events(play_game("human", color="white"))
    types = [e["type"] for e in events]

    assert "move" in types, f"no move event emitted: {types}"
    move_idx = types.index("move")

    # `fen` must follow `move`: it sets whose turn it is, so it has to have the
    # last word on status.
    fens_after = [i for i, t in enumerate(types) if t == "fen" and i > move_idx]
    assert fens_after, f"expected a fen event after the move event, got: {types}"

    # The quip belonging to this move comes last, so a delayed quip can never
    # overwrite the fresher "your turn" status.
    later_quips = [i for i, t in enumerate(types) if t == "quip" and i > move_idx]
    assert later_quips, f"expected a quip after the move event, got: {types}"
    assert later_quips[0] > fens_after[0], (
        f"quip must follow the fen that sets the turn, got: {types}"
    )


@pytest.mark.asyncio
async def test_ai_move_quip_carries_delay_ms(monkeypatch):
    """The arm delay travels as `delay_ms` on the quip, not as a server sleep."""
    _common_capture_test_patches(monkeypatch)

    async def fake_stream_game(game_id):
        yield make_gamefull("chesspersonadbot", "human", moves="e2e4 g7g5 d2d4 f7f5")

    async def fake_best_move(engine, board, move_time=0.5):
        return chess.Move.from_uci("d1h5")

    monkeypatch.setattr(player_module, "stream_game", fake_stream_game)
    monkeypatch.setattr(player_module, "_best_move", fake_best_move)
    monkeypatch.setattr(player_module, "ROBOT_MOVE_DELAY", 7.0)

    events = await collect_events(play_game("human", color="white"))

    quips = [e for e in events if e["type"] == "quip"]
    assert quips, "expected at least one quip"
    # Every quip carries the field; the AI-move one carries the arm delay.
    assert all("delay_ms" in q for q in quips), f"missing delay_ms: {quips}"
    assert any(q["delay_ms"] == 7000 for q in quips), (
        f"expected an AI-move quip with delay_ms=7000, got: "
        f"{[q['delay_ms'] for q in quips]}"
    )


@pytest.mark.asyncio
async def test_game_loop_does_not_sleep_for_the_arm_delay(monkeypatch):
    """A long ROBOT_MOVE_DELAY must not slow the game loop down.

    This is the whole point of the change: the loop has to stay free to read
    the Lichess stream while the quip is waiting to be spoken.
    """
    _common_capture_test_patches(monkeypatch)

    async def fake_stream_game(game_id):
        yield make_gamefull("chesspersonadbot", "human", moves="e2e4 g7g5 d2d4 f7f5")

    async def fake_best_move(engine, board, move_time=0.5):
        return chess.Move.from_uci("d1h5")

    monkeypatch.setattr(player_module, "stream_game", fake_stream_game)
    monkeypatch.setattr(player_module, "_best_move", fake_best_move)
    monkeypatch.setattr(player_module, "ROBOT_MOVE_DELAY", 30.0)   # absurd on purpose

    started = time.monotonic()
    events = await collect_events(play_game("human", color="white"))
    elapsed = time.monotonic() - started

    assert elapsed < 5.0, f"game loop slept for the arm delay ({elapsed:.1f}s)"
    assert any(e["type"] == "quip" and e.get("delay_ms") == 30000 for e in events), (
        "the delay should be handed to the consumer, not dropped"
    )


# ─── Database isolation ───────────────────────────────────────────────────────

def test_tests_do_not_use_the_production_database():
    """Guard the guard: a pytest run must never write to chess_analytics.db.

    Without CHESS_DB_PATH (set in conftest.py before analytics.db is imported)
    every run inserted a fake game into the real analytics DB, which then showed
    up in the game count and the recent-games list.
    """
    from pathlib import Path
    from analytics.db import DB_PATH

    production = (Path(__file__).parent.parent / "chess_analytics.db").resolve()
    assert Path(DB_PATH).resolve() != production
    assert "test" in Path(DB_PATH).name.lower()


def test_recorded_game_lands_in_the_test_database():
    """Writes go somewhere, and that somewhere is the temp DB."""
    from analytics.db import DB_PATH, record_game_start, get_game

    record_game_start("isolation-probe", "someone", "Cocky", "white", "casual")
    row = get_game("isolation-probe")
    assert row is not None and row["difficulty"] == "casual"
    assert "test" in Path(DB_PATH).name.lower()


# ─── SQLite connection handling ───────────────────────────────────────────────

def test_conn_closes_its_connection():
    """_conn must close, not just commit.

    sqlite3.Connection's own context manager is transaction-scoped: it commits
    or rolls back but leaves the handle open. Every call site uses
    `with _conn() as con:`, so a non-closing _conn leaked one connection per
    database operation — and kept the file locked on Windows.
    """
    import sqlite3
    from analytics.db import _conn

    with _conn() as con:
        con.execute("SELECT 1").fetchone()

    # Using a closed connection raises; that is what we want to see.
    with pytest.raises(sqlite3.ProgrammingError):
        con.execute("SELECT 1")


def test_conn_does_not_accumulate_connections():
    import gc
    import sqlite3
    from analytics.db import _conn, get_insights

    def alive():
        gc.collect()
        return sum(1 for o in gc.get_objects() if isinstance(o, sqlite3.Connection))

    before = alive()
    for _ in range(25):
        get_insights()
    assert alive() <= before + 1, "database connections are accumulating"


def test_conn_rolls_back_on_error():
    """Wrapping `with con` must preserve the original rollback semantics."""
    from analytics.db import _conn, record_game_start, get_game

    record_game_start("rollback-probe", "someone", "Cocky", "white", "casual")
    try:
        with _conn() as con:
            con.execute("UPDATE games SET opponent = 'changed' WHERE game_id = ?",
                        ("rollback-probe",))
            raise ValueError("boom")
    except ValueError:
        pass

    assert get_game("rollback-probe")["opponent"] == "someone", "write was not rolled back"


def test_background_tasks_are_referenced():
    """analyze_game is fire-and-forget; asyncio only weak-refs running tasks."""
    import lichess.player as pm

    assert hasattr(pm, "_background_tasks") and hasattr(pm, "_spawn")
