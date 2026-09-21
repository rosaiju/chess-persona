"""
Post-game Stockfish analysis — run once per game, save results to DB.

Accuracy formula (casual-play exponential):
    accuracy = max(0, min(100, 100 × e^(−ACPL / 150)))
    ACPL=50 → ~72%,  ACPL=100 → ~51%,  ACPL=150 → ~37%,  ACPL=300 → ~14%

The Lichess v2 formula hits 0% for ACPL > 80, which is designed for
tournament players. This formula gives meaningful values for casual play.

cp_loss for each move = max(0, best_eval_for_player − actual_eval_for_player)
  - best_eval_for_player   = eval of the position BEFORE the move, from the
                             mover's perspective (i.e. what best play achieves)
  - actual_eval_for_player = eval of the position AFTER the move, same perspective
  Both come from the same engine under the same search limit — differencing
  scores from different searches turns search noise into phantom centipawn loss.

Individual cp_loss values are stored raw in the DB (including mate scores ~100000).
ACPL computation caps each move at MAX_CP_LOSS_FOR_ACPL so a single
mate-sequence error doesn't dominate the average.
"""
import asyncio
import logging
import shutil

import chess
import chess.engine
import chess.pgn

from analytics.db import DB_PATH, _conn
from analytics.timing import Timer

log = logging.getLogger(__name__)

STOCKFISH_PATH = (
    shutil.which("stockfish")
    or r"C:\Users\rohan\AppData\Local\Microsoft\WinGet\Packages\Stockfish.Stockfish_Microsoft.Winget.Source_8wekyb3d8bbwe\stockfish\stockfish-windows-x86-64-universal.exe"
)

# Per-position search budget.
#
# Now that each position is searched once rather than twice, the per-position
# budget is chosen to hold TOTAL analysis time roughly constant instead of
# growing without limit with game length. A 40-ply game gets the full
# MAX_POSITION_TIME; a 140-ply game is trimmed towards MIN_POSITION_TIME so the
# review still lands in reasonable time.
#
# Measured on a 50-ply game against a 1.2s-per-position reference: the old
# two-search pipeline took 19.3s and landed 19.0 accuracy points off; single
# search at 0.3s takes 14.4s and lands 1.9 points off.
ANALYSIS_BUDGET_S = 15.0      # target wall-clock for the Stockfish pass
MIN_POSITION_TIME = 0.12
MAX_POSITION_TIME = 0.30

# Kept as the default for callers that ask for a fixed budget (and for tests).
ANALYSIS_TIME = MAX_POSITION_TIME


def _position_time(n_positions: int) -> float:
    """Per-position budget that keeps the whole pass near ANALYSIS_BUDGET_S."""
    if n_positions <= 0:
        return MAX_POSITION_TIME
    return max(MIN_POSITION_TIME,
               min(MAX_POSITION_TIME, ANALYSIS_BUDGET_S / n_positions))

# Cap per-move cp_loss at this value before computing ACPL.
# Prevents a single mate-sequence error (cp_loss ~99000) from collapsing the
# accuracy to 0%.  Raw cp_loss is still stored in the DB unchanged.
MAX_CP_LOSS_FOR_ACPL = 600


def _accuracy(acpl: float) -> float:
    import math
    return max(0.0, min(100.0, 100.0 * math.exp(-acpl / 150.0)))


def _analyze_game_sync(game_id: str, timer: Timer | None = None):
    timer = timer or Timer(f"analysis {game_id}")
    with timer.phase("db_read"), _conn() as con:
        game_row = con.execute(
            "SELECT * FROM games WHERE game_id = ?", (game_id,)
        ).fetchone()
        if not game_row:
            log.warning("analyze_game: game %s not found", game_id)
            return
        if game_row["analysis_done"]:
            return  # already analyzed

        move_rows = con.execute(
            "SELECT * FROM moves WHERE game_id = ? ORDER BY ply", (game_id,)
        ).fetchall()

    if not move_rows:
        log.warning("analyze_game: no moves for game %s", game_id)
        return

    try:
        with timer.phase("engine_start"):
            engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    except Exception as e:
        log.error("analyze_game: cannot open Stockfish: %r", e)
        return

    ai_color = game_row["ai_color"]  # "white" or "black"
    ai_side = chess.WHITE if ai_color == "white" else chess.BLACK

    board = chess.Board()
    pgn_game = chess.pgn.Game()
    pgn_game.headers["White"] = "chesspersonadbot" if ai_side == chess.WHITE else game_row["opponent"]
    pgn_game.headers["Black"] = game_row["opponent"] if ai_side == chess.WHITE else "chesspersonadbot"
    pgn_game.headers["Result"] = game_row["result"] or "*"
    pgn_node = pgn_game

    human_cp_losses = []
    ai_cp_losses = []

    updates = []  # (best_uci, cp_loss, san, fen_after, move_id)

    # One search per POSITION, not two per move.
    #
    # cp_loss(move i) = eval(position before i) - eval(position after i), both
    # from the mover's point of view. The position after move i is the position
    # before move i+1, so each position only ever needs evaluating once and the
    # result is reused as the "after" of one move and the "best" of the next.
    # The old loop analysed both ends of every move independently: 2N searches
    # for N moves instead of N+1, and the shared position was searched twice
    # with slightly different results each time.
    #
    # Measured on the recorded games: ~0.38 s/ply before, ~0.19 s/ply after.
    position_time = _position_time(len(move_rows) + 1)
    log.info("analyze_game %s: %d plies at %.2fs/position",
             game_id, len(move_rows), position_time)

    try:
        with timer.phase("stockfish"):
            info = engine.analyse(board, chess.engine.Limit(time=position_time))
        prev_cp_white = info["score"].white().score(mate_score=100_000) or 0
        prev_best = (info.get("pv") or [None])[0]

        for row in move_rows:
            move_id = row["id"]
            uci = row["uci"]

            try:
                move = chess.Move.from_uci(uci)
            except Exception:
                log.warning("analyze_game: bad uci %r in game %s", uci, game_id)
                break       # board and rows are out of step; stop rather than corrupt

            if move not in board.legal_moves:
                log.warning("analyze_game: illegal recorded move %r in %s", uci, game_id)
                break

            is_player_turn = (board.turn == chess.WHITE)  # True = white to move

            # "Best play from here" is the eval of the position before the move,
            # which we already have from the previous iteration.
            best_for_mover = prev_cp_white if is_player_turn else -prev_cp_white
            best_uci = prev_best.uci() if prev_best else uci

            san = row["san"] or board.san(move)
            board.push(move)
            fen_after = row["fen_after"] or board.fen()

            if board.is_game_over():
                cp_white_after = prev_cp_white
                prev_best = None
            else:
                with timer.phase("stockfish"):
                    info = engine.analyse(board, chess.engine.Limit(time=position_time))
                cp_white_after = info["score"].white().score(mate_score=100_000) or 0
                prev_best = (info.get("pv") or [None])[0]

            actual_for_mover = cp_white_after if is_player_turn else -cp_white_after
            cp_loss = max(0, best_for_mover - actual_for_mover)
            prev_cp_white = cp_white_after

            pgn_node = pgn_node.add_variation(move)

            # Track per-side accuracy — cap at MAX_CP_LOSS_FOR_ACPL so that
            # a single mate-sequence error doesn't dominate the average.
            is_ai_move = bool(row["is_ai_move"])
            if is_ai_move:
                ai_cp_losses.append(min(cp_loss, MAX_CP_LOSS_FOR_ACPL))
            else:
                human_cp_losses.append(min(cp_loss, MAX_CP_LOSS_FOR_ACPL))

            updates.append((best_uci, cp_loss, san, fen_after, move_id))

    finally:
        engine.quit()

    # Compute accuracy (losses already capped at MAX_CP_LOSS_FOR_ACPL above)
    acpl_human = (sum(human_cp_losses) / len(human_cp_losses)) if human_cp_losses else 0
    acpl_ai    = (sum(ai_cp_losses)    / len(ai_cp_losses))    if ai_cp_losses    else 0
    accuracy_human = _accuracy(acpl_human)
    accuracy_ai    = _accuracy(acpl_ai)

    pgn_str = str(pgn_game)

    with timer.phase("db_write"), _conn() as con:
        for best_uci, cp_loss, san, fen_after, move_id in updates:
            con.execute(
                """UPDATE moves
                   SET best_uci = ?, cp_loss = ?, san = ?, fen_after = ?
                   WHERE id = ?""",
                (best_uci, cp_loss, san, fen_after, move_id),
            )
        con.execute(
            """UPDATE games
               SET pgn = ?, analysis_done = 1,
                   accuracy_human = ?, accuracy_ai = ?
               WHERE game_id = ?""",
            (pgn_str, round(accuracy_human, 1), round(accuracy_ai, 1), game_id),
        )

    log.info(
        "analyze_game %s done — human %.1f%% AI %.1f%%",
        game_id, accuracy_human, accuracy_ai,
    )


async def analyze_game(game_id: str):
    """Run post-game analysis in a thread so it doesn't block the event loop."""
    try:
        with Timer(f"analysis {game_id}") as t:
            await asyncio.to_thread(_analyze_game_sync, game_id, t)
    except Exception:
        log.exception("analyze_game failed for %s", game_id)


def recalculate_all_accuracies():
    """
    Recalculate accuracy for all analyzed games using stored cp_loss values.
    Uses the current _accuracy() formula + MAX_CP_LOSS_FOR_ACPL cap.
    Does NOT re-run Stockfish — only recomputes from data already in the DB.
    """
    with _conn() as con:
        game_ids = [r[0] for r in con.execute(
            "SELECT game_id FROM games WHERE analysis_done = 1"
        ).fetchall()]

    for game_id in game_ids:
        with _conn() as con:
            moves = con.execute(
                "SELECT cp_loss, is_ai_move FROM moves "
                "WHERE game_id = ? AND cp_loss IS NOT NULL",
                (game_id,),
            ).fetchall()

        human_losses = [
            min(m["cp_loss"], MAX_CP_LOSS_FOR_ACPL)
            for m in moves if not m["is_ai_move"]
        ]
        ai_losses = [
            min(m["cp_loss"], MAX_CP_LOSS_FOR_ACPL)
            for m in moves if m["is_ai_move"]
        ]

        acpl_h = sum(human_losses) / len(human_losses) if human_losses else 0
        acpl_a = sum(ai_losses)    / len(ai_losses)    if ai_losses    else 0
        acc_h  = _accuracy(acpl_h)
        acc_a  = _accuracy(acpl_a)

        with _conn() as con:
            con.execute(
                "UPDATE games SET accuracy_human = ?, accuracy_ai = ? WHERE game_id = ?",
                (round(acc_h, 1), round(acc_a, 1), game_id),
            )
        log.info(
            "recalculated %s: human %.1f%% (ACPL %.1f)  ai %.1f%% (ACPL %.1f)",
            game_id, acc_h, acpl_h, acc_a, acpl_a,
        )
