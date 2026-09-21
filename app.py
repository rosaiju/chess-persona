import asyncio
import json
import logging
import os
import time
from pathlib import Path
from asyncio import Queue
from typing import Set

_log = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()

import edge_tts
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse, Response
from pydantic import BaseModel

from lichess.player import play_game, DIFFICULTIES, DEFAULT_DIFFICULTY
from analytics.db import (
    get_insights, get_game, get_game_moves, get_coaching_review, clear_coaching_error,
)
from analytics.coaching import generate_review as generate_coaching_review
from lichess.api import (
    make_human_board_move,
    stop_game,
    validate_bot_account,
    LICHESS_BOT_USERNAME,
    SENSEROBOT_LICHESS_USERNAME,
)

app = FastAPI()

# Resolve templates relative to this file, not the working directory, so the
# app serves correctly no matter where uvicorn is launched from.
TEMPLATES = Path(__file__).parent / "templates"

# ── External event broadcast (for SenseRobot client) ─────────────────────────
_subscribers: Set[Queue] = set()

def _broadcast(event: dict):
    dead = set()
    for q in _subscribers:
        try:
            q.put_nowait(event)
        except Exception:
            dead.add(q)
    _subscribers.difference_update(dead)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    content = (TEMPLATES / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(
        content=content,
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


@app.get("/events")
async def events():
    """SSE stream for external subscribers (e.g. SenseRobot client)."""
    q: Queue = Queue()
    _subscribers.add(q)

    async def stream():
        try:
            while True:
                event = await q.get()
                yield f"data: {json.dumps(event)}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            _subscribers.discard(q)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class PlayRequest(BaseModel):
    opponent: str
    personality: str = "Cocky"
    color: str = "black"   # color the AI plays as; physical board player plays the opposite
    senserobot_mode: bool = False
    difficulty: str = DEFAULT_DIFFICULTY


class MoveRequest(BaseModel):
    game_id: str
    uci: str


VOICE_CONFIG = {
    "Cocky":      {"voice": "en-US-AriaNeural",  "rate": "+20%", "pitch": "+0Hz"},
    "Aggressive": {"voice": "en-US-AvaNeural",   "rate": "+30%", "pitch": "+12Hz"},
    "Nervous":    {"voice": "en-US-JennyNeural", "rate": "-20%", "pitch": "-5Hz"},
    "Friendly":   {"voice": "en-US-EmmaNeural",  "rate": "+8%",  "pitch": "+5Hz"},
}


@app.get("/tts")
async def tts(text: str, personality: str = "Cocky"):
    cfg = VOICE_CONFIG.get(personality, VOICE_CONFIG["Cocky"])
    communicate = edge_tts.Communicate(
        text, voice=cfg["voice"], rate=cfg["rate"], pitch=cfg["pitch"]
    )
    audio = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio += chunk["data"]
    return Response(content=audio, media_type="audio/mpeg")


@app.post("/play")
async def play(req: PlayRequest):
    async def event_stream():
        # First event: mode announcement
        yield f"data: {json.dumps({'type': 'mode', 'senserobot_mode': req.senserobot_mode})}\n\n"

        async for event in play_game(
            req.opponent.strip(),
            req.personality,
            req.color,
            senserobot_mode=req.senserobot_mode,
            difficulty=req.difficulty,
        ):
            yield f"data: {json.dumps(event)}\n\n"
            # Broadcast selected events to external subscribers (SenseRobot)
            if event.get("type") in ("quip", "started", "fen", "move", "done"):
                _broadcast(event)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/move")
async def human_move(req: MoveRequest):
    """
    Browser-only endpoint. Not called in SenseRobot mode — the physical board
    handles human moves directly via the Lichess Board API.
    """
    try:
        await make_human_board_move(req.game_id, req.uci)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True}


@app.post("/game/{game_id}/resign")
async def resign(game_id: str):
    """
    End the current game from the UI.

    Resigns an in-progress game, aborts one still on its first move, or cancels
    a challenge the opponent has not accepted — whichever Lichess allows. The
    /play stream ends on its own once Lichess reports the game over.
    """
    try:
        action = await stop_game(game_id)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    _log.info("[resign] %s: %s", game_id, action)
    return {"ok": True, "action": action}


@app.get("/review/{game_id}", response_class=HTMLResponse)
async def review(game_id: str):
    content = (TEMPLATES / "review.html").read_text(encoding="utf-8")
    return HTMLResponse(content=content, headers={"Cache-Control": "no-store, no-cache"})


@app.post("/review/{game_id}/coaching")
async def trigger_coaching(game_id: str):
    t_req = time.monotonic()
    _log.info("[coaching] POST /review/%s/coaching received", game_id)
    game = await asyncio.to_thread(get_game, game_id)
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    if not game.get("analysis_done"):
        raise HTTPException(status_code=400, detail="Stockfish analysis not complete yet")
    existing = await asyncio.to_thread(get_coaching_review, game_id)
    if existing["done"]:
        _log.info("[coaching] %s: already done — returning cached (%.3fs)", game_id, time.monotonic() - t_req)
        return {"status": "already_done", "review": existing["review"]}
    # A POST is a retry: drop the previous failure so the page stops showing it.
    if existing.get("error"):
        _log.info("[coaching] %s: retrying after previous failure", game_id)
        await asyncio.to_thread(clear_coaching_error, game_id)
    # Guard against duplicate concurrent tasks.
    from analytics.coaching import _active_reviews as _cr
    if game_id in _cr:
        _log.info(
            "[coaching] %s: generation already active (%.1fs) — returning 'started' without a new task",
            game_id, time.monotonic() - _cr[game_id],
        )
        return {"status": "started"}
    _log.info("[coaching] %s: no active generation found — starting new task", game_id)
    asyncio.create_task(generate_coaching_review(game_id))
    return {"status": "started"}


@app.get("/review/{game_id}/coaching")
async def coaching_status(game_id: str):
    return await asyncio.to_thread(get_coaching_review, game_id)


@app.get("/game/{game_id}")
async def game_data(game_id: str):
    game = await asyncio.to_thread(get_game, game_id)
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    moves = await asyncio.to_thread(get_game_moves, game_id)
    return {"game": game, "moves": moves}


@app.get("/insights")
async def insights_all():
    return await asyncio.to_thread(get_insights)


@app.get("/insights/{opponent}")
async def insights_opponent(opponent: str):
    return await asyncio.to_thread(get_insights, opponent)


@app.get("/difficulties")
async def difficulties():
    """Engine strength levels, for the play-screen selector."""
    return {
        "default": DEFAULT_DIFFICULTY,
        "levels": [
            {"key": key, "label": cfg["label"], "elo": cfg["elo"], "blurb": cfg["blurb"]}
            for key, cfg in DIFFICULTIES.items()
        ],
    }


@app.get("/account")
async def account():
    try:
        info = await validate_bot_account()
        return {
            "bot_username": info["username"],
            "senserobot_username": SENSEROBOT_LICHESS_USERNAME,
            "status": "ok",
        }
    except ValueError as e:
        raise HTTPException(400, detail=str(e))
    except RuntimeError:
        raise HTTPException(500, detail="Server configuration error — token not set")
