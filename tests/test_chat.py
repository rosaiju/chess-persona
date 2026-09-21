"""
Tests for the in-game Q&A layer.

Gemini is mocked throughout: these check the chess reasoning and the guardrails,
not the model. The one thing that cannot be automated is whether an answer reads
well, which is noted in the summary as manual testing.
"""
import chess
import pytest

import analytics.chat as chat
from analytics.chat import (
    ChatError, _fallback_answer, _format_facts, _verify_moves,
    answer_question_sync, build_context,
)
from analytics.db import record_game_start, record_move


def _seed(game_id: str, ucis: list[str], ai_color: str = "black"):
    """Record a game the way lichess/player.py does, so chat reads it the same."""
    record_game_start(game_id, "human", "Cocky", ai_color, "beginner")
    board = chess.Board()
    ai_side = chess.BLACK if ai_color == "black" else chess.WHITE
    for i, uci in enumerate(ucis, start=1):
        mv = chess.Move.from_uci(uci)
        is_ai = (board.turn == ai_side)
        san = board.san(mv)
        board.push(mv)
        record_move(game_id, i, uci, san, board.fen(), None, None, None,
                    None, is_ai, False, board.is_check())
    return board


# Scholar's mate setup: after 1.e4 e5 2.Bc4 Nc6 3.Qh5 Nf6, White mates with Qxf7#.
SCHOLARS = ["e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6"]


# ─── Position understanding ───────────────────────────────────────────────────

def test_context_tracks_the_live_position():
    board = _seed("t-pos", SCHOLARS)
    ctx = build_context("t-pos")
    assert ctx["fen"] == board.fen(), "chat is not looking at the current position"
    assert ctx["turn"] == "White"
    assert ctx["human_colour"] == "White" and ctx["robot_colour"] == "Black"
    assert ctx["human_to_move"] is True


def test_engine_finds_the_mate_in_the_current_position():
    _seed("t-mate", SCHOLARS)
    ctx = build_context("t-mate")
    best = ctx["current_lines"][0]
    assert best["san"] == "Qxf7#", f"expected the mate, got {best['san']}"
    assert best["mate"] == 1


def test_threat_probe_sees_what_the_opponent_wants():
    """'What is my opponent threatening?' needs the side-to-move flipped.

    Here it is White's move, so Black's threat (Nxh5, winning the queen) does
    not appear in White's own candidate lines at all.
    """
    _seed("t-threat", SCHOLARS)
    ctx = build_context("t-threat")
    assert ctx["threat"] is not None
    assert ctx["threat"]["san"] == "Nxh5", f"got {ctx['threat']['san']}"


def test_last_human_move_is_attributed_to_the_human():
    _seed("t-attrib", SCHOLARS)
    ctx = build_context("t-attrib")
    assert ctx["last_human"]["san"] == "Qh5"     # White = human
    assert ctx["last_ai"]["san"] == "Nf6"        # Black = robot


def test_colours_swap_when_the_robot_plays_white():
    _seed("t-white", SCHOLARS, ai_color="white")
    ctx = build_context("t-white")
    assert ctx["human_colour"] == "Black" and ctx["robot_colour"] == "White"
    assert ctx["last_human"]["san"] == "Nf6"
    assert ctx["last_ai"]["san"] == "Qh5"


def test_unknown_game_is_rejected():
    with pytest.raises(ChatError):
        build_context("no-such-game")


# ─── Guardrails ───────────────────────────────────────────────────────────────

def test_illegal_moves_are_rejected():
    board = chess.Board()
    for uci in SCHOLARS:
        board.push(chess.Move.from_uci(uci))
    for bad in ["Ke5", "O-O", "Rd8", "Nxa8"]:
        assert _verify_moves(f"You should play {bad}.", [board]), f"{bad} slipped through"


def test_legal_moves_are_accepted():
    board = chess.Board()
    for uci in SCHOLARS:
        board.push(chess.Move.from_uci(uci))
    for good in ["Qxf7#", "Qd1", "Nf3", "Bxf7+"]:
        assert not _verify_moves(f"Consider {good}.", [board]), f"{good} wrongly rejected"


def test_square_names_in_prose_are_not_treated_as_moves():
    """Regression: 'the f7 square' used to be flagged as an illegal move,
    which made almost every well-formed answer fall back."""
    board = chess.Board()
    for uci in SCHOLARS:
        board.push(chess.Move.from_uci(uci))
    prose = ("Your knight on f6 defends h5 and the f7 square is weak, "
             "while you still control e4 and d4.")
    assert _verify_moves(prose, [board]) == []


def test_fallback_answer_is_built_only_from_engine_facts():
    _seed("t-fb", SCHOLARS)
    ctx = build_context("t-fb")
    answer = _fallback_answer(ctx)
    assert "Qxf7#" in answer
    assert answer and not answer.lower().startswith("i don't have")


def test_facts_never_offer_an_illegal_move():
    """Everything handed to the model must be legal in the position it describes."""
    _seed("t-facts", SCHOLARS)
    ctx = build_context("t-facts")
    board = ctx["board"]
    legal = {board.san(m) for m in board.legal_moves}
    for line in ctx["current_lines"]:
        assert line["san"] in legal, f"{line['san']} is not legal in the live position"


# ─── End-to-end with the model mocked ─────────────────────────────────────────

def test_answer_uses_the_model_reply_when_it_checks_out(monkeypatch):
    _seed("t-ok", SCHOLARS)
    monkeypatch.setattr(chat, "_ask_gemini",
                        lambda prompt: "Play Qxf7# and it is over.")
    out = answer_question_sync("t-ok", "What should I play?", "Cocky")
    assert out["verified"] is True and out["fell_back"] is False
    assert "Qxf7#" in out["answer"]


def test_answer_falls_back_when_the_model_invents_a_move(monkeypatch):
    """An illegal suggestion must never reach the player."""
    _seed("t-bad", SCHOLARS)
    monkeypatch.setattr(chat, "_ask_gemini", lambda prompt: "Just play Ke5, easy.")
    out = answer_question_sync("t-bad", "What should I play?", "Cocky")
    assert out["fell_back"] is True
    assert "Ke5" not in out["answer"]
    assert "Qxf7#" in out["answer"], "fallback should carry the real engine line"


def test_one_regeneration_is_allowed_before_falling_back(monkeypatch):
    _seed("t-retry", SCHOLARS)
    calls = []

    def flaky(prompt):
        calls.append(prompt)
        return "Play Ke5." if len(calls) == 1 else "Play Qxf7#."

    monkeypatch.setattr(chat, "_ask_gemini", flaky)
    out = answer_question_sync("t-retry", "What now?", "Cocky")
    assert len(calls) == 2, "should retry once with corrective feedback"
    assert out["verified"] is True and "Qxf7#" in out["answer"]


def test_empty_question_is_rejected():
    _seed("t-empty", SCHOLARS)
    with pytest.raises(ChatError):
        answer_question_sync("t-empty", "   ", "Cocky")


def test_personality_defaults_to_the_one_the_game_was_played_with(monkeypatch):
    _seed("t-persona", SCHOLARS)
    monkeypatch.setattr(chat, "_ask_gemini", lambda prompt: "Qxf7# wins.")
    out = answer_question_sync("t-persona", "What now?", None)
    assert out["personality"] == "Cocky"


def test_question_is_passed_to_the_model(monkeypatch):
    _seed("t-q", SCHOLARS)
    seen = {}
    def capture(prompt):
        seen["prompt"] = prompt
        return "Qxf7# wins."
    monkeypatch.setattr(chat, "_ask_gemini", capture)
    answer_question_sync("t-q", "Why was my last move bad?", "Cocky")
    assert "Why was my last move bad?" in seen["prompt"]
    assert "STRONGEST MOVE" in seen["prompt"], "engine facts missing from the prompt"
