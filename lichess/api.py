import json
import os
import re

import httpx

BASE = "https://lichess.org"
TOKEN = os.environ.get("LICHESS_TOKEN", "")


def _auth() -> dict:
    return {"Authorization": f"Bearer {TOKEN}"}


async def make_move(game_id: str, move: str) -> bool:
    async with httpx.AsyncClient() as client:
        res = await client.post(
            f"{BASE}/api/board/game/{game_id}/move/{move}",
            headers=_auth(),
        )
        if not res.is_success:
            raise RuntimeError(f"Move failed ({move}): {res.status_code} {res.text}")
        return True


async def stream_game(game_id: str):
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "GET", f"{BASE}/api/board/game/stream/{game_id}", headers=_auth()
        ) as res:
            res.raise_for_status()
            async for line in res.aiter_lines():
                if line.strip():
                    yield json.loads(line)


async def challenge_user(username: str, color: str = "white") -> dict:
    async with httpx.AsyncClient() as client:
        res = await client.post(
            f"{BASE}/api/challenge/{username}",
            headers={**_auth(), "Content-Type": "application/x-www-form-urlencoded"},
            data={
                "rated": "false",
                "clock.limit": 900,
                "clock.increment": 10,
                "color": color,
            },
        )
        if not res.is_success:
            raise RuntimeError(f"Lichess {res.status_code}: {res.text}")
        return res.json()


async def stream_account_events():
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "GET", f"{BASE}/api/stream/event", headers=_auth()
        ) as res:
            res.raise_for_status()
            async for line in res.aiter_lines():
                if line.strip():
                    yield json.loads(line)


async def make_human_move(game_id: str, move: str, token: str) -> bool:
    async with httpx.AsyncClient() as client:
        res = await client.post(
            f"{BASE}/api/board/game/{game_id}/move/{move}",
            headers={"Authorization": f"Bearer {token}"},
        )
        if not res.is_success:
            raise RuntimeError(f"Lichess {res.status_code}: {res.text}")
        return True


async def resign_game(game_id: str) -> bool:
    async with httpx.AsyncClient() as client:
        res = await client.post(
            f"{BASE}/api/board/game/{game_id}/resign", headers=_auth()
        )
        return res.is_success
