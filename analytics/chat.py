"""
In-game question answering — "why was that bad?", "what is he threatening?"

Same division of labour as analytics/coaching.py: Stockfish is the only source
of chess truth, and Gemini only puts that truth into words. The difference is
that this runs mid-game, against the live position, and has to answer a specific
question rather than summarise a finished game.

Three things keep the model honest:

1. It is handed an explicit list of legal moves with engine evaluations, and
   told to use only those. It is never asked to find a move itself.
2. Every move-like token in the reply is checked for legality against the
   position actually being discussed (`_verify_moves`). One regeneration is
   allowed; if the reply is still wrong, a deterministic answer built straight
   from the engine facts is returned instead.
3. No evaluation number is invented, because the prompt carries the numbers and
   the model is told not to produce new ones.

The live position comes from the `moves` table, which `lichess/player.py` writes
after every ply. Nothing here touches the running game loop or the engines it
owns.
"""
import logging
import os
import re
import time

import chess
import chess.engine

from analytics import llm
from analytics.analysis import STOCKFISH_PATH
from analytics.db import _conn
from analytics.timing import Timer

log = logging.getLogger(__name__)

# Per-position search budget. Higher than the in-game eval because the player is
# waiting on this answer and a wrong "best move" is worse than a slow one.
CHAT_ANALYSIS_TIME = 0.3

# How many candidate moves to offer the model.
CANDIDATE_MOVES = 3

# Plies of principal variation to show, in SAN.
PV_LENGTH = 6

MAX_QUESTION_CHARS = 300

# Answers are 2-4 sentences by instruction; this bounds a runaway generation.
CHAT_MAX_TOKENS = 220



class ChatError(RuntimeError):
    """Something went wrong that the user should see."""


# ─── Position context ─────────────────────────────────────────────────────────

def _load_game(game_id: str):
    with _conn() as con:
        game = con.execute(
            "SELECT * FROM games WHERE game_id = ?", (game_id,)
        ).fetchone()
        moves = con.execute(
            "SELECT * FROM moves WHERE game_id = ? ORDER BY ply", (game_id,)
        ).fetchall()
    return game, moves


def _replay(moves) -> tuple[chess.Board, list[dict]]:
    """
    Rebuild the game from recorded UCI, capturing the board before each move.

    Replaying beats trusting the stored FEN: it guarantees the position we
    analyse is reachable by the moves we describe, so a legality check against
    it means something.
    """
    board = chess.Board()
    history = []
    for row in moves:
        try:
            move = chess.Move.from_uci(row["uci"])
        except ValueError:
            log.warning("chat: unparseable uci %r in %s", row["uci"], row["game_id"])
            break
        if move not in board.legal_moves:
            log.warning("chat: illegal recorded move %r in %s", row["uci"], row["game_id"])
            break
        history.append({
            "ply": row["ply"],
            "san": row["san"] or board.san(move),
            "uci": row["uci"],
            "is_ai_move": bool(row["is_ai_move"]),
            "fen_before": board.fen(),
            "cp_loss": row["cp_loss"],
        })
        board.push(move)
    return board, history


def _analyse(engine, board: chess.Board, multipv: int = CANDIDATE_MOVES) -> list[dict]:
    """Top lines for the side to move, scored from that side's perspective."""
    if board.is_game_over():
        return []
    infos = engine.analyse(
        board, chess.engine.Limit(time=CHAT_ANALYSIS_TIME), multipv=multipv
    )
    if isinstance(infos, dict):
        infos = [infos]

    lines = []
    for info in infos:
        pv = info.get("pv") or []
        if not pv:
            continue
        score = info["score"].pov(board.turn)
        mate = score.mate()
        lines.append({
            "san": board.san(pv[0]),
            "uci": pv[0].uci(),
            "cp": None if mate is not None else score.score(),
            "mate": mate,
            "pv_san": board.variation_san(pv[:PV_LENGTH]),
        })
    return lines


def _describe(line: dict) -> str:
    if line["mate"] is not None:
        return f"mate in {abs(line['mate'])}" + ("" if line["mate"] > 0 else " against")
    cp = line["cp"]
    if cp is None:
        return "unclear"
    return f"{cp/100:+.2f}"


def build_context(game_id: str) -> dict:
    """
    Everything the model is allowed to know, all of it from Stockfish or the
    recorded game. Also returns the boards, so answers can be legality-checked.
    """
    game, move_rows = _load_game(game_id)
    if not game:
        raise ChatError("That game is not in the database yet.")

    board, history = _replay(move_rows)

    ai_color = game["ai_color"]                    # colour the robot plays
    human_is_white = (ai_color == "black")
    human_colour = chess.WHITE if human_is_white else chess.BLACK

    try:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    except Exception as e:
        raise ChatError("Stockfish is not available, so I can't analyse the position.") from e

    try:
        current_lines = _analyse(engine, board)

        # The player's own most recent move, re-analysed: what they played,
        # what the engine preferred, and the gap between them.
        last_human = None
        for entry in reversed(history):
            if not entry["is_ai_move"]:
                before = chess.Board(entry["fen_before"])
                alts = _analyse(engine, before)
                played = chess.Move.from_uci(entry["uci"])
                after = before.copy()
                after.push(played)
                played_score = None
                if not after.is_game_over():
                    info = engine.analyse(after, chess.engine.Limit(time=CHAT_ANALYSIS_TIME))
                    played_score = info["score"].pov(before.turn).score()
                last_human = {
                    "san": entry["san"],
                    "uci": entry["uci"],
                    "fen_before": entry["fen_before"],
                    "alternatives": alts,
                    "played_cp": played_score,
                    "move_number": (entry["ply"] + 1) // 2,
                }
                break

        # What the opponent threatens: hand them a free move and see what they
        # would play. This null-move probe is the only way to answer "what is he
        # threatening?" when it is NOT their turn — the lines above only cover
        # the side to move. Illegal while in check, so skip it there.
        # Done by flipping the side to move in the FEN rather than pushing a
        # null move: python-chess refuses to send null-move history to the
        # engine and warns about it. is_valid() rejects the flip when it would
        # leave a king capturable.
        threat = None
        threat_fen = None
        if not board.is_game_over() and not board.is_check():
            parts = board.fen().split()
            parts[1] = "b" if parts[1] == "w" else "w"
            parts[3] = "-"                      # en passant no longer applies
            probe = chess.Board(" ".join(parts))
            if probe.is_valid():
                threat_lines = _analyse(engine, probe, multipv=1)
                if threat_lines:
                    threat = threat_lines[0]
                    threat_fen = probe.fen()

        last_ai = next(
            ({"san": e["san"], "move_number": (e["ply"] + 1) // 2}
             for e in reversed(history) if e["is_ai_move"]),
            None,
        )
    finally:
        engine.quit()

    return {
        "game_id": game_id,
        "board": board,
        "fen": board.fen(),
        "finished": bool(game["result"]),
        "result": game["result"],
        "personality": game["personality"],
        "difficulty": game["difficulty"],
        "human_colour": "White" if human_is_white else "Black",
        "robot_colour": "Black" if human_is_white else "White",
        "turn": "White" if board.turn == chess.WHITE else "Black",
        "human_to_move": board.turn == human_colour,
        "move_number": board.fullmove_number,
        "in_check": board.is_check(),
        "current_lines": current_lines,
        "threat": threat,
        "threat_fen": threat_fen,
        "last_human": last_human,
        "last_ai": last_ai,
        "recent_san": [e["san"] for e in history[-8:]],
    }


# ─── Prompt ───────────────────────────────────────────────────────────────────

_SYSTEM = """\
You are the voice of a physical chess robot that the player is currently playing \
against. Answer the player's question about the game in character: {persona_note}

Rules you must follow:
- Use ONLY the engine facts listed below. Do not invent evaluations, move \
numbers, or variations.
- If you name a move, it must be one that appears in the facts below, copied \
exactly as written.
- If the facts do not answer the question, say so plainly in one sentence.
- Speak directly to the player as "you". The robot is "I".
- 2 to 4 sentences, under 70 words. This will be read aloud, so write plain \
prose: no markdown, no bullet points, no tables, no notation the ear cannot follow.

--- VERIFIED ENGINE FACTS ---
{facts}
--- PLAYER'S QUESTION ---
{question}
"""

_PERSONA_NOTES = {
    "Cocky":      "smug and self-satisfied, but still give a genuinely useful answer.",
    "Aggressive": "blunt and intense, short punchy sentences, still accurate.",
    "Nervous":    "anxious and apologetic, hedging a little, but still correct.",
    "Friendly":   "warm and encouraging, like a patient coach.",
}


def _format_facts(ctx: dict) -> str:
    out = [
        f"Player is {ctx['human_colour']}; I am {ctx['robot_colour']}.",
        f"Move {ctx['move_number']}, {ctx['turn']} to move"
        + (" (the player)" if ctx["human_to_move"] else " (me)") + ".",
        f"Position (FEN): {ctx['fen']}",
    ]
    if ctx["in_check"]:
        out.append(f"{ctx['turn']} is in check.")
    if ctx["finished"]:
        out.append(f"The game is over. Result: {ctx['result']}.")
    if ctx["recent_san"]:
        out.append("Recent moves: " + " ".join(ctx["recent_san"]))

    if ctx["current_lines"]:
        best_now = ctx["current_lines"][0]
        out.append(
            f"\nSTRONGEST MOVE for {ctx['turn']} right now: {best_now['san']} "
            f"(evaluation {_describe(best_now)}), continuing {best_now['pv_san']}"
        )
        rest = ctx["current_lines"][1:]
        if rest:
            out.append("Playable but weaker here: " + ", ".join(
                f"{l['san']} ({_describe(l)})" for l in rest
            ))

    if ctx.get("threat"):
        t = ctx["threat"]
        who = "I am" if ctx["human_to_move"] else "you are"
        out.append(
            f"\nTHREAT: if given a free move, {who} threatening {t['san']} "
            f"(evaluation {_describe(t)} for the threatening side)."
        )

    lh = ctx["last_human"]
    if lh:
        out.append(f"\nPlayer's last move: {lh['san']} (move {lh['move_number']}).")
        if lh["played_cp"] is not None:
            out.append(f"  Evaluation after it, from the player's side: {lh['played_cp']/100:+.2f}")
        if lh["alternatives"]:
            best = lh["alternatives"][0]
            out.append(
                f"  THE ONE BEST ALTERNATIVE there was {best['san']} "
                f"(evaluation {_describe(best)}). If the player asks what they "
                f"should have played, the answer is {best['san']} and nothing else."
            )
            if lh["played_cp"] is not None and best["cp"] is not None:
                loss = best["cp"] - lh["played_cp"]
                verdict = ("about as good as the best move" if loss < 50
                           else "an inaccuracy" if loss < 120
                           else "a mistake" if loss < 250 else "a blunder")
                out.append(f"  That makes the played move {verdict} (lost {max(loss,0)} centipawns).")
            other = ", ".join(
                a["san"] for a in lh["alternatives"][1:] if a["san"] != lh["san"]
            )
            if other:
                out.append(
                    f"  Weaker but playable there (do NOT call these best): {other}")

    if ctx["last_ai"]:
        out.append(f"\nMy last move: {ctx['last_ai']['san']} (move {ctx['last_ai']['move_number']}).")

    return "\n".join(out)


# ─── Move legality verification ───────────────────────────────────────────────

# Only unambiguous move notation: piece moves (Nf3, Rad1, Qxh7+), pawn captures
# (exd5), castling, and promotions (e8=Q).
#
# Bare squares like "e4" or "f7" are deliberately NOT matched. In prose they are
# far more often square references — "the f7 square", "your knight on f6",
# "defends h5" — than move suggestions, and treating them as moves made the
# verifier reject almost every well-formed answer. The cost is that an invented
# bare pawn push would slip through unverified; the alternative was a checker
# that fired constantly on correct text.
_MOVE_TOKEN = re.compile(
    r"\b(?:O-O-O|O-O"
    r"|[KQRBN][a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?"
    r"|[a-h]x[a-h][1-8](?:=[QRBN])?"
    r"|[a-h][1-8]=[QRBN])(?:[+#])?"
)


def _legal_san(board: chess.Board) -> set[str]:
    out = set()
    for mv in board.legal_moves:
        san = board.san(mv)
        out.add(san)
        out.add(san.rstrip("+#"))
    return out


def _verify_moves(answer: str, boards: list[chess.Board]) -> list[str]:
    """
    Return move tokens in `answer` that are not legal in ANY of the positions
    under discussion (current position, and the one before the player's last
    move). Anything unrecognised as a move is ignored.
    """
    allowed = set()
    for b in boards:
        allowed |= _legal_san(b)

    bad = []
    for token in _MOVE_TOKEN.findall(answer):
        if token in allowed or token.rstrip("+#") in allowed:
            continue
        bad.append(token)
    return bad


def _fallback_answer(ctx: dict) -> str:
    """Deterministic answer from engine facts, used when the model misbehaves."""
    parts = []
    lh = ctx["last_human"]
    if lh and lh["alternatives"]:
        best = lh["alternatives"][0]
        if lh["played_cp"] is not None and best["cp"] is not None:
            loss = max(best["cp"] - lh["played_cp"], 0)
            parts.append(
                f"You played {lh['san']}. The engine preferred {best['san']}, "
                f"worth about {loss} centipawns more."
            )
        else:
            parts.append(f"You played {lh['san']}; the engine preferred {best['san']}.")
    if ctx["current_lines"]:
        top = ctx["current_lines"][0]
        parts.append(
            f"In this position the strongest move for {ctx['turn']} is "
            f"{top['san']}, evaluating to {_describe(top)}."
        )
    if not parts:
        parts.append("I don't have enough analysis of this position to answer that.")
    return " ".join(parts)


# ─── Entry point ──────────────────────────────────────────────────────────────

def _ask_model(prompt: str) -> llm.LLMResult:
    """
    Ask the provider chain. Model selection, quota cooldowns and bounded
    retries all live in analytics/llm.py, shared with the coaching review.
    """
    return llm.generate(prompt, max_output_tokens=CHAT_MAX_TOKENS, label="chat")


def answer_question_sync(game_id: str, question: str, personality: str | None = None) -> dict:
    """
    Answer one question about a game. Blocking; call via asyncio.to_thread.

    Returns {answer, personality, verified, fell_back}.
    """
    question = (question or "").strip()
    if not question:
        raise ChatError("Ask me something about the game.")
    if len(question) > MAX_QUESTION_CHARS:
        question = question[:MAX_QUESTION_CHARS]

    timer = Timer(f"chat {game_id}", log_on_exit=False)
    timer.__enter__()
    with timer.phase("stockfish"):
        ctx = build_context(game_id)
    persona = personality or ctx["personality"] or "Cocky"
    facts = _format_facts(ctx)

    # Positions an answer is allowed to talk about.
    boards = [ctx["board"]]
    if ctx["last_human"]:
        boards.append(chess.Board(ctx["last_human"]["fen_before"]))
    if ctx.get("threat_fen"):
        # The opponent's threat is legal only with the side to move flipped.
        # Answers are meant to mention it, so it has to count as legal here.
        boards.append(chess.Board(ctx["threat_fen"]))

    prompt = _SYSTEM.format(
        persona_note=_PERSONA_NOTES.get(persona, _PERSONA_NOTES["Cocky"]),
        facts=facts,
        question=question,
    )

    fell_back = False
    result = None
    try:
        with timer.phase("llm"):
            result = _ask_model(prompt)
        answer = result.text
        bad = _verify_moves(answer, boards)
        if bad:
            log.warning("chat %s: illegal moves %s — regenerating", game_id, bad)
            with timer.phase("llm_retry"):
                result = _ask_model(
                    prompt
                    + "\n\nYour previous answer referred to "
                    + ", ".join(bad)
                    + ", which is not legal here. Rewrite it using only the moves listed above."
                )
            answer = result.text
            bad = _verify_moves(answer, boards)
            if bad:
                log.error("chat %s: still illegal %s — using engine fallback", game_id, bad)
                answer = _fallback_answer(ctx)
                fell_back = True
    except llm.AllProvidersFailed as e:
        # Every model is out of quota or unreachable. Rather than show nothing,
        # answer from Stockfish alone and say plainly no model was involved.
        log.warning("chat %s: no model available (%s)", game_id, e)
        timer.__exit__(None, None, None)
        return {
            "answer": _fallback_answer(ctx),
            "personality": persona,
            "verified": False,
            "fell_back": True,
            "source": "stockfish_only",
            "provider": None,
            "model": None,
            "note": str(e),
            "timing": timer.as_dict(),
        }
    except ChatError:
        raise
    except Exception as e:
        log.exception("chat %s: generation failed", game_id)
        raise ChatError(str(llm.classify(e))) from e

    timer.__exit__(None, None, None)
    log.info("[timing] %s", timer.summary())
    return {
        "answer": answer,
        "personality": persona,
        "verified": not fell_back,
        "fell_back": fell_back,
        "source": "stockfish_only" if fell_back else "model",
        "provider": result.provider if result else None,
        "model": result.model if result else None,
        "timing": timer.as_dict(),
    }
