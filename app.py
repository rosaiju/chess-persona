import json
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import edge_tts
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse, Response
from pydantic import BaseModel

from lichess.player import play_game
from lichess.api import make_human_move

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
async def index():
    return Path("templates/index.html").read_text(encoding="utf-8")


class PlayRequest(BaseModel):
    opponent: str
    personality: str = "Cocky"


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
        async for event in play_game(req.opponent.strip(), req.personality):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/move")
async def human_move(req: MoveRequest):
    token = os.environ.get("LICHESS_HUMAN_TOKEN", "")
    if not token:
        raise HTTPException(status_code=400, detail="LICHESS_HUMAN_TOKEN not set in .env")
    try:
        await make_human_move(req.game_id, req.uci, token)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True}
