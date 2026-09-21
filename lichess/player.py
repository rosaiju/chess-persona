import asyncio
import logging
import os
import random
import shutil
import time
import chess
import chess.engine

# Seconds to wait after submitting an AI move before speaking the quip.
# Gives the SenseRobot arm time to physically complete the move.
# Override via ROBOT_MOVE_DELAY env var (set to 0 to disable).
ROBOT_MOVE_DELAY = float(os.getenv("ROBOT_MOVE_DELAY", "7"))

log = logging.getLogger(__name__)

from lichess.api import (
    make_move,
    stream_game,
    challenge_user,
    stream_account_events,
    resign_game,
    validate_bot_account,
    LICHESS_BOT_USERNAME,
)
from persona.personality import get_quip
from analytics.db import record_game_start, record_move, record_game_end
from analytics.analysis import analyze_game

STOCKFISH_PATH = (
    shutil.which("stockfish")
    or r"C:\Users\rohan\AppData\Local\Microsoft\WinGet\Packages\Stockfish.Stockfish_Microsoft.Winget.Source_8wekyb3d8bbwe\stockfish\stockfish-windows-x86-64-universal.exe"
)


# ─── Difficulty levels ────────────────────────────────────────
# Stockfish's UCI_Elo range is 1320–3190. "Max" disables the limiter entirely
# and lets the engine play at full strength.
#
# move_time is the per-move search budget in seconds. It stays low at the weaker
# levels so the robot answers quickly — at a capped Elo, extra thinking time
# buys almost nothing.

DIFFICULTIES = {
    "beginner": {"label": "Beginner", "elo": 1320, "move_time": 0.10,
                 "blurb": "Hangs pieces. A fair fight for a first game."},
    "casual":   {"label": "Casual",   "elo": 1600, "move_time": 0.20,
                 "blurb": "Solid basics, still misses tactics."},
    "club":     {"label": "Club",     "elo": 1900, "move_time": 0.30,
                 "blurb": "Punishes real mistakes. You'll need a plan."},
    "strong":   {"label": "Strong",   "elo": 2200, "move_time": 0.50,
                 "blurb": "Rarely errs. Expect to be ground down."},
    "max":      {"label": "Max",      "elo": None, "move_time": 0.50,
                 "blurb": "Unrestricted Stockfish. Good luck."},
}

DEFAULT_DIFFICULTY = "casual"


def resolve_difficulty(name: str | None) -> tuple[str, dict]:
    """Normalize a difficulty key, falling back to the default if unknown."""
    key = (name or "").strip().lower()
    if key not in DIFFICULTIES:
        key = DEFAULT_DIFFICULTY
    return key, DIFFICULTIES[key]


# ─── Engine helpers ───────────────────────────────────────────────────────────

def _configure_strength(engine, elo: int | None):
    """
    Cap the engine's playing strength via UCI_Elo. `elo=None` leaves the engine
    at full strength. Clamped to whatever range this Stockfish build reports.
    Failures are logged and ignored — a wrongly-configured engine still plays.
    """
    if elo is None:
        return
    try:
        opt = engine.options.get("UCI_Elo")
        if opt is None or engine.options.get("UCI_LimitStrength") is None:
            log.warning("engine has no UCI_Elo support — playing at full strength")
            return
        lo = opt.min if opt.min is not None else elo
        hi = opt.max if opt.max is not None else elo
        clamped = max(lo, min(hi, elo))
        engine.configure({"UCI_LimitStrength": True, "UCI_Elo": clamped})
        if clamped != elo:
            log.info("UCI_Elo %d clamped to %d (engine range %s–%s)", elo, clamped, lo, hi)
        print(f"[INFO] Engine strength capped at Elo {clamped}")
    except Exception as e:
        log.warning("could not set engine strength: %r", e)


def _open_engine_sync(elo: int | None = None):
    try:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        print("[INFO] Stockfish loaded")
    except Exception as e:
        print(f"[WARN] Stockfish not found: {e!r}")
        return None
    _configure_strength(engine, elo)
    return engine


async def _open_engine(elo: int | None = None):
    return await asyncio.to_thread(_open_engine_sync, elo)


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


def _best_move_sync(engine, board_fen: str, move_time: float) -> str | None:
    try:
        board = chess.Board(board_fen)
        result = engine.play(board, chess.engine.Limit(time=move_time))
        return result.move.uci() if result.move else None
    except Exception:
        return None


async def _best_move(
    engine, board: chess.Board, move_time: float = 0.5
) -> chess.Move | None:
    if not engine:
        # Fall back to first legal move
        moves = list(board.legal_moves)
        return moves[0] if moves else None
    uci = await asyncio.to_thread(_best_move_sync, engine, board.fen(), move_time)
    return chess.Move.from_uci(uci) if uci else None


# ─── Trigger classification ───────────────────────────────────────────────────

def _classify_trigger(
    board_before: chess.Board,
    move: chess.Move,
    board_after: chess.Board,
    eval_before: int | None,
    eval_after: int | None,
    is_ai_move: bool,
    total_moves: int,
    last_positional_trigger_at: int,
) -> str | None:
    is_capture = board_before.is_capture(move)
    gives_check = board_after.is_check()

    if is_ai_move:
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


def _game_over_trigger(board: chess.Board, ai_side: int) -> str:
    result = board.result()
    if result == "1/2-1/2":
        return "draw"
    if result == "1-0":
        return "game_over_robot_wins" if ai_side == chess.WHITE else "game_over_human_wins"
    if result == "0-1":
        return "game_over_robot_wins" if ai_side == chess.BLACK else "game_over_human_wins"
    return "draw"


# ─── Main entry point ─────────────────────────────────────────────────────────

async def play_game(
    opponent_username: str,
    personality: str = "Cocky",
    color: str = "black",
    senserobot_mode: bool = False,
    difficulty: str = DEFAULT_DIFFICULTY,
):
    """
    Async generator that yields SSE-ready event dicts.
    AI plays as the requested color via the Bot API, at the strength set by
    `difficulty` (see DIFFICULTIES). Quip events are emitted after every move.
    """
    # Validate bot account before issuing any challenge
    try:
        await validate_bot_account()
    except (ValueError, RuntimeError) as e:
        yield {"type": "error", "text": str(e)}
        return

    # Prevent bot from challenging itself
    if opponent_username.lower() == LICHESS_BOT_USERNAME.lower():
        yield {"type": "error", "text": "The bot cannot challenge itself."}
        return

    difficulty_key, diff_cfg = resolve_difficulty(difficulty)
    move_time = diff_cfg["move_time"]

    ai_side = chess.WHITE if color == "white" else chess.BLACK
    physical_player_side = chess.BLACK if ai_side == chess.WHITE else chess.WHITE
    engine = await _open_engine(diff_cfg["elo"])

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
                    if (
                        event.get("type") == "challengeDeclined"
                        and event.get("challenge", {}).get("id") == game_id
                    ):
                        break
        except TimeoutError:
            yield {"type": "timeout"}
            return

        if not accepted:
            yield {"type": "declined", "opponent": opponent_username}
            return

        start_fen = chess.Board().fen()

        yield {
            "type": "started",
            "gameId": game_id,
            "url": f"https://lichess.org/{game_id}",
            "opponent": opponent_username,
            "color": color,
            "fen": start_fen,
            "ai_side": "white" if ai_side == chess.WHITE else "black",
            "physical_player_side": "black" if ai_side == chess.WHITE else "white",
            "difficulty": difficulty_key,
            "difficulty_label": diff_cfg["label"],
            "difficulty_elo": diff_cfg["elo"],
        }

        record_game_start(game_id, opponent_username, personality, color, difficulty_key)

        # Game start quip
        quip = get_quip(personality, "game_start")
        if quip:
            yield {"type": "quip", "text": quip, "personality": personality, "capture": False}

        board = chess.Board()
        prev_eval: int | None = None
        total_moves = 0          # half-moves (plies)
        last_positional_at = -99
        seen_move_count = 0       # server move count we've already processed

        async for event in stream_game(game_id):
            if event["type"] not in ("gameFull", "gameState"):
                continue

            # Determine ai_side from gameFull — authoritative source
            if event["type"] == "gameFull":
                white_id = event.get("white", {}).get("id", "").lower()
                black_id = event.get("black", {}).get("id", "").lower()
                bot_id = LICHESS_BOT_USERNAME.lower()
                if white_id == bot_id:
                    ai_side = chess.WHITE
                elif black_id == bot_id:
                    ai_side = chess.BLACK
                else:
                    yield {"type": "error", "text": "Bot account not found in this game"}
                    break
                physical_player_side = chess.BLACK if ai_side == chess.WHITE else chess.WHITE
                print(f"[{time.strftime('%H:%M:%S')}][DEBUG] gameFull: white={white_id} black={black_id} -> ai is {'white' if ai_side == chess.WHITE else 'black'}")
                log.info("AI is %s", "white" if ai_side == chess.WHITE else "black")

            state = event.get("state", event) if event["type"] == "gameFull" else event
            server_moves = [m for m in state.get("moves", "").split() if m]

            # Check game-over status first
            status = state.get("status", "")
            if status and status not in ("created", "started"):
                # Rebuild board to get result
                final_board = chess.Board()
                for uci in server_moves:
                    final_board.push(chess.Move.from_uci(uci))
                trigger = _game_over_trigger(final_board, ai_side)
                quip = get_quip(personality, trigger)
                if quip:
                    yield {"type": "quip", "text": quip, "personality": personality, "capture": False}
                record_game_end(game_id, final_board.result(), len(server_moves))
                asyncio.create_task(analyze_game(game_id))
                yield {
                    "type": "done",
                    "status": status,
                    "gameId": game_id,
                    "url": f"https://lichess.org/{game_id}",
                    "result": final_board.result(),
                }
                break

            # Process any NEW moves from the server.
            # seen_move_count prevents reprocessing moves already handled.
            new_moves = server_moves[seen_move_count:]
            print(f"[{time.strftime('%H:%M:%S')}][DEBUG] {event['type']}: server_moves={server_moves} new={new_moves} ai_side={'BLACK' if ai_side == chess.BLACK else 'WHITE'} board_turn={'b' if board.turn == chess.BLACK else 'w'}")
            for uci in new_moves:
                move = chess.Move.from_uci(uci)
                board_before = board.copy()
                is_ai_move = (board.turn == ai_side)
                is_capture = board_before.is_capture(move)
                san = board_before.san(move)

                eval_before = prev_eval
                board.push(move)
                total_moves += 1
                seen_move_count += 1

                raw = await _eval_position(engine, board)
                eval_after = raw if ai_side == chess.WHITE else (-raw if raw is not None else None)
                yield {"type": "fen", "fen": board.fen()}

                trigger = _classify_trigger(
                    board_before, move, board,
                    eval_before, eval_after,
                    is_ai_move, total_moves, last_positional_at,
                )
                if trigger and trigger in (
                    "robot_winning", "human_winning", "endgame"
                ):
                    last_positional_at = total_moves

                record_move(game_id, total_moves, uci, san, board.fen(),
                            eval_before, eval_after, raw,
                            trigger, is_ai_move, is_capture, board.is_check())

                if trigger:
                    quip = get_quip(personality, trigger)
                    if quip:
                        _cap = is_ai_move and (is_capture or board.is_check())
                        print(f"[{time.strftime('%H:%M:%S')}][QUIP] trigger={trigger} capture={_cap} text={quip[:40]!r}")
                        yield {"type": "quip", "text": quip, "personality": personality, "capture": _cap}

                prev_eval = eval_after

            # If it's now the AI's turn and game is still going, play best move
            if board.turn == ai_side and not board.is_game_over():
                yield {"type": "thinking"}
                move = await _best_move(engine, board, move_time)
                print(f"[{time.strftime('%H:%M:%S')}][DEBUG] AI move chosen: {move}")
                if not move:
                    print(f"[{time.strftime('%H:%M:%S')}][DEBUG] No move found - breaking")
                    break

                is_capture = board.is_capture(move)
                board_before = board.copy()
                eval_before = prev_eval
                san = board_before.san(move)

                print(f"[{time.strftime('%H:%M:%S')}][DEBUG] Calling make_move({game_id}, {move.uci()})")
                try:
                    await make_move(game_id, move.uci())
                    print(f"[{time.strftime('%H:%M:%S')}][DEBUG] make_move succeeded")
                except Exception as e:
                    print(f"[{time.strftime('%H:%M:%S')}][DEBUG] make_move FAILED: {e!r}")
                    yield {"type": "error", "text": f"Move failed: {e}"}
                    break

                board.push(move)
                total_moves += 1
                seen_move_count += 1

                raw = await _eval_position(engine, board)
                eval_after = raw if ai_side == chess.WHITE else (-raw if raw is not None else None)
                yield {"type": "fen", "fen": board.fen(), "move": move.uci()}

                trigger = _classify_trigger(
                    board_before, move, board,
                    eval_before, eval_after,
                    True, total_moves, last_positional_at,
                )
                if trigger and trigger in ("robot_winning", "endgame"):
                    last_positional_at = total_moves

                record_move(game_id, total_moves, move.uci(), san, board.fen(),
                            eval_before, eval_after, raw,
                            trigger, True, is_capture, board.is_check())

                if trigger:
                    quip = get_quip(personality, trigger)
                    if quip:
                        _cap = is_capture or board.is_check()
                        if ROBOT_MOVE_DELAY > 0:
                            await asyncio.sleep(ROBOT_MOVE_DELAY)
                        print(f"[{time.strftime('%H:%M:%S')}][QUIP] trigger={trigger} capture={_cap} text={quip[:40]!r}")
                        yield {"type": "quip", "text": quip, "personality": personality, "capture": _cap}

                prev_eval = eval_after

                yield {
                    "type": "move",
                    "uci": move.uci(),
                    "fen": board.fen(),
                    "moveNum": (total_moves + 1) // 2,
                    "gameId": game_id,
                }

                if board.is_game_over():
                    trigger = _game_over_trigger(board, ai_side)
                    quip = get_quip(personality, trigger)
                    if quip:
                        yield {"type": "quip", "text": quip, "personality": personality, "capture": False}
                    record_game_end(game_id, board.result(), total_moves)
                    asyncio.create_task(analyze_game(game_id))
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
