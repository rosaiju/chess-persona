import json
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel

from lichess.player import play_game

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
async def index():
    return Path("templates/index.html").read_text(encoding="utf-8")


class PlayRequest(BaseModel):
    opponent: str
    personality: str = "Cocky"


@app.post("/play")
async def play(req: PlayRequest):
    async def event_stream():
        async for event in play_game(req.opponent.strip(), req.personality):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
