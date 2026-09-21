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

import chess
import chess.pgn

from analytics import llm
from analytics.db import (
    _conn, get_coaching_review, save_coaching_review, save_coaching_error,
)
from analytics.timing import Timer

log = logging.getLogger(__name__)

MODEL = "gemini-3.6-flash"

# cp_loss at or above this came from a mate score, not a real centipawn count.
MATE_SCORE_THRESHOLD = 10_000

# Hard ceiling on one Gemini request. The SDK timeout is in milliseconds.
# Free-tier requests can hang; without this the review page span forever.
# Kept for chat.py, which imports it. Per-request timeouts now live in llm.py.
REQUEST_TIMEOUT_S = 90

# Output cap: reviews are ~450 words by instruction, so this bounds a runaway
# generation without truncating a normal one.
REVIEW_MAX_TOKENS = 700

# Prompt trimming. The review is capped at ~450 words, so it can only discuss a
# handful of moves; sending more is tokens spent for nothing. The PGN was
# dropped entirely — the move list already carries everything the review cites.
MAX_CRITICAL_LINES = 12
MAX_STRONG_LINES = 8

# Whole-review deadline. The chain can try several models, so allow for that,
# but never leave the page waiting indefinitely.
REVIEW_DEADLINE_S = float(os.getenv("REVIEW_DEADLINE_S", "150"))

# Tracks active generation calls: game_id → wall-clock start time.
# The route consults this to deduplicate, so a second request for the same
# review does not start a second generation and burn more daily quota.
_active_reviews: dict[str, float] = {}

# Generation counter per game. asyncio.wait_for abandons the *await*, but the
# worker thread keeps running and would otherwise overwrite the timeout message
# with whatever it eventually produced — showing the user a cause that is not
# the real one. A run only writes if it is still the current generation.
_generation: dict[str, int] = {}


def _claim(game_id: str) -> int:
    _generation[game_id] = _generation.get(game_id, 0) + 1
    return _generation[game_id]


def _still_current(game_id: str, token: int) -> bool:
    return _generation.get(game_id) == token

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


def _friendly_error(e: Exception) -> str:
    """
    Turn an SDK exception into something a person can act on.

    The raw text is a JSON error blob; the page needs to say what to do next.
    """
    raw = str(e)
    low = raw.lower()
    if "api_key_invalid" in low or "api key not valid" in low:
        return "Gemini rejected the API key. Check GEMINI_API_KEY in .env."
    if "resource_exhausted" in low or "429" in raw or "quota" in low:
        return ("Gemini free-tier quota is exhausted. It resets daily — "
                "try again later.")
    if "unavailable" in low or "503" in raw or "overloaded" in low:
        return "Gemini is busy right now. Try again in a moment."
    if "permission_denied" in low or "403" in raw:
        return "Gemini denied the request. The API key may lack access to this model."
    if "not found" in low and "model" in low:
        return f"Gemini model '{MODEL}' was not found. It may have been renamed."
    if "timeout" in low or "timed out" in low or "deadline" in low:
        return "Gemini did not respond in time. Try again."
    if isinstance(e, (ConnectionError, OSError)):
        return "Could not reach Gemini. Check your internet connection."
    # Unrecognised: keep the real text, trimmed, so nothing is hidden.
    return f"Review failed — {type(e).__name__}: {raw[:200]}"


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

    # Only the worst errors are worth the tokens. A 142-ply game produced a
    # 5.4k-character prompt, most of it a list the review could never cite in
    # 450 words. Sorted by severity so trimming drops the least important.
    critical_lines.sort(key=lambda line: -int(
        line.split("cp loss: ")[1].split(",")[0] if "cp loss: " in line else 10**6
    ))
    critical_lines = critical_lines[:MAX_CRITICAL_LINES]

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
{chr(10).join(strong_lines[:MAX_STRONG_LINES]) if strong_lines else '  None recorded'}

Errors by game phase:
{chr(10).join(phase_lines) if phase_lines else '  No phase data available'}
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

    # Cache check lives here, not only in the route, so the guard holds for
    # every caller. Regenerating a finished review spends daily quota to
    # produce something already stored.
    existing = get_coaching_review(game_id)
    if existing["done"] and existing["review"]:
        log.info("[coaching] %s: already generated — serving cached review", game_id)
        return

    token = _claim(game_id)
    timer = Timer(f"coaching {game_id}")
    try:
        with timer:
            with timer.phase("build_prompt"):
                prompt = _build_prompt(game_id)
            if not prompt:
                log.warning("[coaching] %s: cannot build prompt — no data", game_id)
                save_coaching_error(game_id, "No analysis data for this game yet.")
                return
            log.info("[coaching] %s: prompt %d chars (~%d tokens)",
                     game_id, len(prompt), len(prompt) // 4)

            # The provider chain handles model selection, quota cooldowns and
            # bounded retries. It raises AllProvidersFailed with a per-attempt
            # breakdown if nothing answers.
            with timer.phase("llm"):
                result = llm.generate(
                    prompt, max_output_tokens=REVIEW_MAX_TOKENS,
                    label=f"coaching {game_id}",
                )

            if not _still_current(game_id, token):
                log.info("[coaching] %s: superseded or timed out — discarding result",
                         game_id)
                return
            with timer.phase("db_write"):
                save_coaching_review(
                    game_id, result.text,
                    provider=result.provider, model=result.model,
                )
            log.info("[coaching] %s: %s:%s answered in %.2fs, %d chars",
                     game_id, result.provider, result.model,
                     result.latency_s, len(result.text))
    except Exception as e:
        # Every failure has to leave a durable marker. Without one the status
        # endpoint keeps answering {done: false} and the page never stops
        # spinning.
        log.exception("[coaching] %s: generation failed", game_id)
        if _still_current(game_id, token):
            msg = str(e) if isinstance(e, llm.LLMError) else _friendly_error(e)
            save_coaching_error(game_id, msg)
        else:
            log.info("[coaching] %s: superseded — not overwriting recorded state",
                     game_id)
        raise
    finally:
        _active_reviews.pop(game_id, None)


async def generate_review(game_id: str):
    """
    Async wrapper — runs in a thread so it doesn't block the event loop.

    Bounded by a wall-clock timeout as well as the SDK's own, so a request that
    hangs below the HTTP layer still resolves into a visible error rather than
    an endless spinner.
    """
    try:
        await asyncio.wait_for(
            asyncio.to_thread(_generate_sync, game_id),
            timeout=REVIEW_DEADLINE_S,
        )
    except asyncio.TimeoutError:
        log.error("[coaching] %s: timed out after %ss", game_id, REVIEW_DEADLINE_S)
        # Retire the in-flight generation first: the worker thread is still
        # running and must not overwrite this message when it finishes.
        _claim(game_id)
        _active_reviews.pop(game_id, None)
        await asyncio.to_thread(
            save_coaching_error, game_id,
            f"Timed out after {REVIEW_DEADLINE_S:.0f}s with no response.",
        )
    except Exception:
        # _generate_sync already recorded the specific reason.
        log.exception("coaching review failed for %s", game_id)
