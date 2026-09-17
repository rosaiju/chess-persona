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

from google import genai
import chess
import chess.pgn

from analytics.db import _conn, save_coaching_review

log = logging.getLogger(__name__)

MODEL = "gemini-2.0-flash"

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
    critical_lines = []
    for m in human_moves:
        if m["cp_loss"] is None or m["cp_loss"] <= 75:
            continue
        move_num = (m["ply"] + 1) // 2
        side = "White" if human_is_white else "Black"
        played = m["san"] or m["uci"]
        best = best_san_by_id.get(m["id"]) or m["best_uci"] or "unknown"
        category = "Blunder" if m["cp_loss"] > 200 else "Mistake"
        critical_lines.append(
            f"  Move {move_num} ({side}, {m['phase']}): "
            f"played {played} — better was {best} "
            f"[cp loss: {m['cp_loss']}, {category}]"
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
    prompt = _build_prompt(game_id)
    if not prompt:
        log.warning("coaching: cannot build prompt for %s — no data", game_id)
        return

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        log.error("coaching: GEMINI_API_KEY not set")
        return

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=MODEL, contents=prompt)
    review_text = response.text.strip()
    save_coaching_review(game_id, review_text)
    log.info("coaching review saved for game %s", game_id)


async def generate_review(game_id: str):
    """Async wrapper — runs in a thread so it doesn't block the event loop."""
    try:
        await asyncio.to_thread(_generate_sync, game_id)
    except Exception:
        log.exception("coaching review failed for %s", game_id)
