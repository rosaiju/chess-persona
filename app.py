import asyncio
import json
import os
from pathlib import Path
from asyncio import Queue
from typing import Set

from dotenv import load_dotenv
load_dotenv()

import edge_tts
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse, Response
from pydantic import BaseModel

from lichess.player import play_game
from analytics.db import get_insights, get_game, get_game_moves
from lichess.api import (
    make_human_board_move,
    validate_bot_account,
    LICHESS_BOT_USERNAME,
    SENSEROBOT_LICHESS_USERNAME,
)

app = FastAPI()

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
    content = Path("templates/index.html").read_text(encoding="utf-8")
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
