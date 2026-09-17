"""
Post-game Stockfish analysis — run once per game, save results to DB.

Accuracy formula (Lichess v2):
    accuracy = max(0, min(100, 103.1668 × e^(−0.04354 × ACPL) − 3.1668))
where ACPL = average centipawn loss per move for the player.

cp_loss for each move = max(0, best_eval_for_player − actual_eval_for_player)
  - best_eval_for_player = centipawn score Stockfish would get with best move
  - actual_eval_for_player = centipawn score after the move actually played
  Both expressed from the moving player's perspective (positive = player is winning).
"""
import asyncio
import logging
import shutil

import chess
import chess.engine
import chess.pgn

from analytics.db import DB_PATH, _conn

log = logging.getLogger(__name__)

STOCKFISH_PATH = (
    shutil.which("stockfish")
    or r"C:\Users\rohan\AppData\Local\Microsoft\WinGet\Packages\Stockfish.Stockfish_Microsoft.Winget.Source_8wekyb3d8bbwe\stockfish\stockfish-windows-x86-64-universal.exe"
)

ANALYSIS_TIME = 0.05  # seconds per position


def _lichess_accuracy(acpl: float) -> float:
    import math
    return max(0.0, min(100.0, 103.1668 * math.exp(-0.04354 * acpl) - 3.1668))


def _analyze_game_sync(game_id: str):
    with _conn() as con:
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

    try:
        for row in move_rows:
            move_id = row["id"]
            uci = row["uci"]

            try:
                move = chess.Move.from_uci(uci)
            except Exception:
                log.warning("analyze_game: bad uci %r in game %s", uci, game_id)
                board.push(chess.Move.null())
                continue

            is_player_turn = (board.turn == chess.WHITE)  # True = white to move

            # SAN and FEN for this move
            san = row["san"] or board.san(move)
            board.push(move)
            fen_after = row["fen_after"] or board.fen()

            # Eval AFTER move — from the perspective of the player who just moved
            # cp_white is already stored live; use it if available, else analyse
            if row["cp_white"] is not None:
                cp_white_after = row["cp_white"]
            else:
                info = engine.analyse(board, chess.engine.Limit(time=ANALYSIS_TIME))
                cp_white_after = info["score"].white().score(mate_score=100_000) or 0

            # For the player who just moved: positive = they are winning
            if is_player_turn:  # white just moved
                actual_for_mover = cp_white_after
            else:               # black just moved
                actual_for_mover = -cp_white_after

            # Best move eval: rewind, find best, replay best, eval
            board.pop()
            best_info = engine.analyse(board, chess.engine.Limit(time=ANALYSIS_TIME))
            best_move_obj = best_info.get("pv", [None])[0]
            best_uci = best_move_obj.uci() if best_move_obj else uci

            if best_move_obj and best_move_obj != move:
                board.push(best_move_obj)
                best_info2 = engine.analyse(board, chess.engine.Limit(time=ANALYSIS_TIME))
                cp_white_best = best_info2["score"].white().score(mate_score=100_000) or 0
                board.pop()
                best_for_mover = cp_white_best if is_player_turn else -cp_white_best
            else:
                best_for_mover = actual_for_mover

            cp_loss = max(0, best_for_mover - actual_for_mover)

            # Re-push the actual move to continue the game
            board.push(move)

            # PGN node
            pgn_node = pgn_node.add_variation(move)

            # Track per-side accuracy
            is_ai_move = bool(row["is_ai_move"])
            if is_ai_move:
                ai_cp_losses.append(cp_loss)
            else:
                human_cp_losses.append(cp_loss)

            updates.append((best_uci, cp_loss, san, fen_after, move_id))

    finally:
        engine.quit()

    # Compute accuracy
    acpl_human = (sum(human_cp_losses) / len(human_cp_losses)) if human_cp_losses else 0
    acpl_ai    = (sum(ai_cp_losses)    / len(ai_cp_losses))    if ai_cp_losses    else 0
    accuracy_human = _lichess_accuracy(acpl_human)
    accuracy_ai    = _lichess_accuracy(acpl_ai)

    pgn_str = str(pgn_game)

    with _conn() as con:
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
        await asyncio.to_thread(_analyze_game_sync, game_id)
    except Exception:
        log.exception("analyze_game failed for %s", game_id)
