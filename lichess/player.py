import asyncio
import random
import shutil
import chess
import chess.engine

from lichess.api import (
    make_move,
    stream_game,
    challenge_user,
    stream_account_events,
    resign_game,
)
from persona.personality import get_quip

STOCKFISH_PATH = (
    shutil.which("stockfish")
    or r"C:\Users\rohan\AppData\Local\Microsoft\WinGet\Packages\Stockfish.Stockfish_Microsoft.Winget.Source_8wekyb3d8bbwe\stockfish\stockfish-windows-x86-64-universal.exe"
)


# ─── Engine helpers ───────────────────────────────────────────────────────────

def _open_engine_sync():
    try:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        print("[INFO] Stockfish loaded")
        return engine
    except Exception as e:
        print(f"[WARN] Stockfish not found: {e!r}")
        return None


async def _open_engine():
    return await asyncio.to_thread(_open_engine_sync)


def _eval_sync(engine, board_fen: str) -> int | None:
    """Centipawn eval from White's perspective. None if unavailable."""
    try:
        board = chess.Board(board_fen)
        if board.is_game_over():
            return None
        info = engine.analyse(board, chess.engine.Limit(time=0.05))
        return info["score"].white().score(mate_score=100_000)
    except Exception:
        return None


async def _eval_position(engine, board: chess.Board) -> int | None:
    if not engine:
        return None
    return await asyncio.to_thread(_eval_sync, engine, board.fen())


def _best_move_sync(engine, board_fen: str) -> str | None:
    try:
        board = chess.Board(board_fen)
        result = engine.play(board, chess.engine.Limit(time=0.5))
        return result.move.uci() if result.move else None
    except Exception:
        return None


async def _best_move(engine, board: chess.Board) -> chess.Move | None:
    if not engine:
        # Fall back to first legal move
        moves = list(board.legal_moves)
        return moves[0] if moves else None
    uci = await asyncio.to_thread(_best_move_sync, engine, board.fen())
    return chess.Move.from_uci(uci) if uci else None


# ─── Trigger classification ───────────────────────────────────────────────────

def _classify_trigger(
    board_before: chess.Board,
    move: chess.Move,
    board_after: chess.Board,
    eval_before: int | None,
    eval_after: int | None,
    is_robot_move: bool,
    total_moves: int,
    last_positional_trigger_at: int,
) -> str | None:
    is_capture = board_before.is_capture(move)
    gives_check = board_after.is_check()

    if is_robot_move:
        # Priority: check > capture > positional/default
        if gives_check:
            return "robot_check"
        if is_capture:
            return "robot_captures"
        # Emit positional commentary at most once every 6 moves
        if total_moves - last_positional_trigger_at >= 6:
            if eval_after is not None and eval_after > 300:
                return "robot_winning"
            if total_moves >= 30:
                return "endgame"
        # Only quip on ~45% of plain moves (skip opening, reduce chatter)
        if total_moves > 6 and random.random() < 0.45:
            return "robot_move"
        return None
    else:
        # Human's move
        if gives_check:
            return "human_check"
        if eval_before is not None and eval_after is not None:
            delta = eval_after - eval_before  # positive = White improved = human blundered
            if delta > 200:
                return "human_blunders"
            if delta > 80:   # raised from 50 — only react to real mistakes
                return "human_mistake"
            if delta < -100:
                return "human_good_move"
        # Positional commentary for human advantage
        if total_moves - last_positional_trigger_at >= 6:
            if eval_after is not None and eval_after < -300:
                return "human_winning"
        return None


def _game_over_trigger(board: chess.Board) -> str:
    result = board.result()
    if result == "1-0":
        return "game_over_robot_wins"
    if result == "0-1":
        return "game_over_human_wins"
    return "draw"


# ─── Main entry point ─────────────────────────────────────────────────────────

async def play_game(opponent_username: str, personality: str = "Cocky", color: str = "black"):
    """
    Async generator that yields SSE-ready event dicts.
    Robot plays as the requested color (best Stockfish moves).
    Quip events are emitted after every move.
    """
    robot_side = chess.WHITE if color == "white" else chess.BLACK
    engine = await _open_engine()

    try:
        # Challenge the opponent
        yield {"type": "challenging", "opponent": opponent_username}
        try:
            result = await challenge_user(opponent_username, color)
        except Exception as e:
            yield {"type": "error", "text": str(e)}
            return
        game_id = (result.get("challenge") or result)["id"]
        yield {
            "type": "waiting",
            "gameId": game_id,
            "opponent": opponent_username,
            "url": f"https://lichess.org/{game_id}",
        }

        # Wait for acceptance (2 minutes)
        accepted = False
        try:
            async with asyncio.timeout(120):
                async for event in stream_account_events():
                    if (
                        event.get("type") == "gameStart"
                        and event.get("game", {}).get("gameId") == game_id
                    ):
                        accepted = True
                        break
                    if event.get("type") == "challengeDeclined":
                        break
        except TimeoutError:
            yield {"type": "timeout"}
            return

        if not accepted:
            yield {"type": "declined", "opponent": opponent_username}
            return

        yield {"type": "started", "gameId": game_id, "url": f"https://lichess.org/{game_id}", "opponent": opponent_username, "color": color}

        # Game start quip
        quip = get_quip(personality, "game_start")
        if quip:
            yield {"type": "quip", "text": quip, "personality": personality}

        board = chess.Board()
        prev_eval: int | None = None
        total_moves = 0          # half-moves (plies)
        last_positional_at = -99
        seen_move_count = 0       # server move count we've already processed

        async for event in stream_game(game_id):
            if event["type"] not in ("gameFull", "gameState"):
                continue

            # On gameFull, lock in robot_side from what Lichess actually assigned
            if event["type"] == "gameFull":
                white_id = event.get("white", {}).get("id", "").lower()
                robot_side = chess.BLACK if white_id == opponent_username.lower() else chess.WHITE
                actual_color = "white" if robot_side == chess.WHITE else "black"
                if actual_color != color:
                    print(f"[WARN] Requested {color} but Lichess assigned {actual_color} to the robot")
                else:
                    print(f"[INFO] Robot is {actual_color}")

            state = event.get("state", event) if event["type"] == "gameFull" else event
            server_moves = [m for m in state.get("moves", "").split() if m]

            # Check game-over status first
            status = state.get("status", "")
            if status and status not in ("created", "started"):
                # Rebuild board to get result
                final_board = chess.Board()
                for uci in server_moves:
                    final_board.push(chess.Move.from_uci(uci))
                trigger = _game_over_trigger(final_board)
                quip = get_quip(personality, trigger)
                if quip:
                    yield {"type": "quip", "text": quip, "personality": personality}
                yield {
                    "type": "done",
                    "status": status,
                    "gameId": game_id,
                    "url": f"https://lichess.org/{game_id}",
                    "result": final_board.result(),
                }
                break

            # Process any NEW moves from the server
            new_moves = server_moves[seen_move_count:]
            for uci in new_moves:
                move = chess.Move.from_uci(uci)
                board_before = board.copy()
                is_robot_move = (board.turn == robot_side)

                eval_before = prev_eval
                board.push(move)
                total_moves += 1
                seen_move_count += 1

                raw = await _eval_position(engine, board)
                eval_after = raw if robot_side == chess.WHITE else (-raw if raw is not None else None)
                yield {"type": "fen", "fen": board.fen()}

                trigger = _classify_trigger(
                    board_before, move, board,
                    eval_before, eval_after,
                    is_robot_move, total_moves, last_positional_at,
                )
                if trigger and trigger in (
                    "robot_winning", "human_winning", "endgame"
                ):
                    last_positional_at = total_moves

                if trigger:
                    quip = get_quip(personality, trigger)
                    if quip:
                        yield {"type": "quip", "text": quip, "personality": personality}

                prev_eval = eval_after

            # If it's now the robot's turn and game is still going, play best move
            if board.turn == robot_side and not board.is_game_over():
                yield {"type": "thinking"}
                move = await _best_move(engine, board)
                if not move:
                    break

                is_capture = board.is_capture(move)
                board_before = board.copy()
                eval_before = prev_eval

                await make_move(game_id, move.uci())

                board.push(move)
                total_moves += 1
                seen_move_count += 1

                raw = await _eval_position(engine, board)
                eval_after = raw if robot_side == chess.WHITE else (-raw if raw is not None else None)
                yield {"type": "fen", "fen": board.fen(), "move": board.san(move) if False else move.uci()}

                trigger = _classify_trigger(
                    board_before, move, board,
                    eval_before, eval_after,
                    True, total_moves, last_positional_at,
                )
                if trigger and trigger in ("robot_winning", "endgame"):
                    last_positional_at = total_moves

                if trigger:
                    quip = get_quip(personality, trigger)
                    if quip:
                        yield {"type": "quip", "text": quip, "personality": personality}

                prev_eval = eval_after

                yield {
                    "type": "move",
                    "uci": move.uci(),
                    "fen": board.fen(),
                    "moveNum": (total_moves + 1) // 2,
                    "gameId": game_id,
                }

                if board.is_game_over():
                    trigger = _game_over_trigger(board)
                    quip = get_quip(personality, trigger)
                    if quip:
                        yield {"type": "quip", "text": quip, "personality": personality}
                    yield {
                        "type": "done",
                        "status": "mate",
                        "gameId": game_id,
                        "url": f"https://lichess.org/{game_id}",
                        "result": board.result(),
                    }
                    break

    finally:
        if engine:
            try:
                await asyncio.to_thread(engine.quit)
            except Exception:
                pass
