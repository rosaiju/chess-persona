"""
AI coaching review — calls Gemini to interpret Stockfish analysis data.

Stockfish is the sole source of truth for evaluations and best moves.
Gemini's job is to explain what the numbers mean in plain coaching language.
The prompt passes explicit facts; the model must not invent evaluations or moves
not present in the data.
"""
import asyncio
import logging
import os
import time

from google import genai
from google.genai import types as genai_types
import chess
import chess.pgn

from analytics.db import _conn, save_coaching_review

log = logging.getLogger(__name__)

MODEL = "gemini-3.6-flash"

# cp_loss at or above this came from a mate score, not a real centipawn count.
MATE_SCORE_THRESHOLD = 10_000

# Tracks active generation calls: game_id → wall-clock start time.
# Used only for diagnostic logging — to detect concurrent duplicate calls.
_active_reviews: dict[str, float] = {}

_PROMPT_TEMPLATE = """\
You are a chess coach writing a post-game review for an amateur player.

Use ONLY the engine analysis data provided below. Do not invent evaluations, \
centipawn values, or better moves that are not listed. Engine analysis is the \
source of truth — your role is to explain what it means in plain, encouraging language.

Write exactly these 5 sections with these exact markdown headers, in this order:

## Game Summary
## What You Did Well
## Key Mistakes
## Patterns I Noticed
## Focus for Next Game

Section rules:
- "Game Summary": 2-3 sentences covering the result, overall accuracy, and general impression.
- "What You Did Well": cite specific move numbers from the strong-moves list. If the list is empty say so briefly.
- "Key Mistakes": for each critical error in the list, name the move number, what was played, \
and the better alternative. Explain briefly WHY the played move was bad (concretely — loose piece, \
king safety, tactical motif). Use only moves from the list.
- "Patterns I Noticed": identify 1-2 recurring themes across the errors (e.g. "you consistently \
underestimated open lines", "opening inaccuracies set up problems later"). Be specific.
- "Focus for Next Game": 1-2 actionable, concrete things to practise or watch for.

Keep each section to 2-4 sentences. Total review under 450 words. Address the player directly ("you").

--- ENGINE ANALYSIS DATA ---
{game_data}
"""


def _build_prompt(game_id: str) -> str | None:
    with _conn() as con:
        game = con.execute("SELECT * FROM games WHERE game_id = ?", (game_id,)).fetchone()
        if not game:
            return None
        moves = con.execute(
            "SELECT * FROM moves WHERE game_id = ? ORDER BY ply", (game_id,)
        ).fetchall()

    if not moves:
        return None

    human_is_white = (game["ai_color"] == "black")
    human_moves = [m for m in moves if not m["is_ai_move"]]

    # Replay game to convert UCI best-move into SAN.
    # We need the board state BEFORE each move to compute SAN.
    board = chess.Board()
    best_san_by_id: dict[int, str] = {}
    for m in moves:
        try:
            move = chess.Move.from_uci(m["uci"])
        except Exception:
            board.push(chess.Move.null())
            continue
        if m["best_uci"] and m["best_uci"] != m["uci"]:
            try:
                best_move = chess.Move.from_uci(m["best_uci"])
                best_san_by_id[m["id"]] = board.san(best_move)
            except Exception:
                best_san_by_id[m["id"]] = m["best_uci"]  # fallback to UCI
        board.push(move)

    # Critical errors: cp_loss > 75
    #
    # cp_loss is stored raw, so a move that walks into or throws away a forced
    # mate carries a mate score (~100000). Handing "cp loss: 98166" to the model
    # invites nonsense, so describe those in words instead of as a number.
    critical_lines = []
    for m in human_moves:
        if m["cp_loss"] is None or m["cp_loss"] <= 75:
            continue
        move_num = (m["ply"] + 1) // 2
        side = "White" if human_is_white else "Black"
        played = m["san"] or m["uci"]
        best = best_san_by_id.get(m["id"]) or m["best_uci"] or "unknown"
        if m["cp_loss"] >= MATE_SCORE_THRESHOLD:
            detail = "lost a forced mate or allowed one — decisive"
        else:
            category = "Blunder" if m["cp_loss"] > 200 else "Mistake"
            detail = f"cp loss: {m['cp_loss']}, {category}"
        critical_lines.append(
            f"  Move {move_num} ({side}, {m['phase']}): "
            f"played {played} — better was {best} "
            f"[{detail}]"
        )

    # Strong moves: cp_loss == 0 (matched engine best exactly)
    strong_lines = []
    for m in human_moves:
        if m["cp_loss"] == 0 and m["san"]:
            move_num = (m["ply"] + 1) // 2
            strong_lines.append(f"  Move {move_num}: {m['san']}")

    # Phase breakdown
    phase_lines = []
    for phase in ("opening", "middlegame", "endgame"):
        ph = [m for m in human_moves if m["phase"] == phase]
        if not ph:
            continue
        analyzed = [m for m in ph if m["cp_loss"] is not None]
        blunders = sum(1 for m in analyzed if m["cp_loss"] > 200)
        mistakes = sum(1 for m in analyzed if 75 < m["cp_loss"] <= 200)
        phase_lines.append(
            f"  {phase.capitalize()}: {blunders} blunder(s), {mistakes} mistake(s) "
            f"in {len(ph)} human moves"
        )

    game_data = f"""
Result: {game['result']}  (human played {'White' if human_is_white else 'Black'})
Human accuracy: {game['accuracy_human']}%   (engine: {game['accuracy_ai']}%)
Total plies: {game['total_plies']}

Critical errors — DO use these in "Key Mistakes" ({len(critical_lines)} total):
{chr(10).join(critical_lines) if critical_lines else '  None'}

Strong moves — DO cite these in "What You Did Well" (matched engine best, {len(strong_lines)} total):
{chr(10).join(strong_lines[:10]) if strong_lines else '  None recorded'}

Errors by game phase:
{chr(10).join(phase_lines) if phase_lines else '  No phase data available'}

PGN (for context only — do not derive new evaluations from this):
{game['pgn'] or '(not available)'}
"""
    return _PROMPT_TEMPLATE.format(game_data=game_data)


def _generate_sync(game_id: str):
    t0 = time.monotonic()

    # Detect duplicate concurrent calls for the same game.
    if game_id in _active_reviews:
        prior_age = t0 - _active_reviews[game_id]
        log.warning(
            "[coaching] %s: DUPLICATE _generate_sync — another call has been "
            "active for %.1fs. Two Gemini requests will now run concurrently.",
            game_id, prior_age,
        )
    _active_reviews[game_id] = t0
    log.info("[coaching] %s: generation started (active calls: %d)", game_id, len(_active_reviews))

    try:
        prompt = _build_prompt(game_id)
        if not prompt:
            log.warning("[coaching] %s: cannot build prompt — no data", game_id)
            return
        log.info("[coaching] %s: prompt built in %.3fs (%d chars)", game_id, time.monotonic() - t0, len(prompt))

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            log.error("[coaching] %s: GEMINI_API_KEY not set", game_id)
            return

        t_api = time.monotonic()
        log.info("[coaching] %s: sending request to Gemini (model=%s)", game_id, MODEL)

        # 120-second timeout prevents a hung free-tier request from waiting forever.
        # The SDK's HttpOptions.timeout is in milliseconds.
        client = genai.Client(
            api_key=api_key,
            http_options=genai_types.HttpOptions(timeout=120_000),
        )
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )
        review_text = response.text.strip()
        t_api_done = time.monotonic()
        log.info(
            "[coaching] %s: Gemini responded in %.2fs (%d chars)",
            game_id, t_api_done - t_api, len(review_text),
        )

        t_db = time.monotonic()
        save_coaching_review(game_id, review_text)
        log.info(
            "[coaching] %s: saved to DB in %.3fs — total elapsed %.2fs",
            game_id, time.monotonic() - t_db, time.monotonic() - t0,
        )
    finally:
        _active_reviews.pop(game_id, None)


async def generate_review(game_id: str):
    """Async wrapper — runs in a thread so it doesn't block the event loop."""
    try:
        await asyncio.to_thread(_generate_sync, game_id)
    except Exception:
        log.exception("coaching review failed for %s", game_id)
