import json
import os
import re
import urllib.parse

import httpx
from dotenv import load_dotenv

load_dotenv()

BASE = "https://lichess.org"

LICHESS_BOT_TOKEN = os.environ.get("LICHESS_BOT_TOKEN", "")
LICHESS_HUMAN_TOKEN = os.environ.get("LICHESS_HUMAN_TOKEN", "")
LICHESS_BOT_USERNAME = os.environ.get("LICHESS_BOT_USERNAME", "")
SENSEROBOT_LICHESS_USERNAME = os.environ.get("SENSEROBOT_LICHESS_USERNAME", "")


def _bot_headers() -> dict:
    if not LICHESS_BOT_TOKEN:
        raise RuntimeError("LICHESS_BOT_TOKEN not set")
    return {"Authorization": f"Bearer {LICHESS_BOT_TOKEN}"}


def _human_headers() -> dict:
    if not LICHESS_HUMAN_TOKEN:
        raise RuntimeError("LICHESS_HUMAN_TOKEN not set")
    return {"Authorization": f"Bearer {LICHESS_HUMAN_TOKEN}"}


async def make_move(game_id: str, move: str) -> bool:
    async with httpx.AsyncClient() as client:
        res = await client.post(
            f"{BASE}/api/bot/game/{game_id}/move/{move}",
            headers=_bot_headers(),
        )
        if not res.is_success:
            raise RuntimeError(f"Move failed ({move}): {res.status_code} {res.text}")
        return True


async def stream_game(game_id: str):
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "GET", f"{BASE}/api/bot/game/stream/{game_id}", headers=_bot_headers()
        ) as res:
            res.raise_for_status()
            async for line in res.aiter_lines():
                if line.strip():
                    yield json.loads(line)


async def challenge_user(username: str, color: str = "black") -> dict:
    body = urllib.parse.urlencode({
        "rated": "false",
        "clock.limit": "900",
        "clock.increment": "10",
        "color": color,
    })
    async with httpx.AsyncClient() as client:
        res = await client.post(
            f"{BASE}/api/challenge/{username}",
            headers={**_bot_headers(), "Content-Type": "application/x-www-form-urlencoded"},
            content=body,
        )
        if not res.is_success:
            raise RuntimeError(f"Lichess {res.status_code}: {res.text}")
        return res.json()


async def stream_account_events():
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "GET", f"{BASE}/api/stream/event", headers=_bot_headers()
        ) as res:
            res.raise_for_status()
            async for line in res.aiter_lines():
                if line.strip():
                    yield json.loads(line)


async def make_human_board_move(game_id: str, uci: str) -> bool:
    async with httpx.AsyncClient() as client:
        res = await client.post(
            f"{BASE}/api/board/game/{game_id}/move/{uci}",
            headers=_human_headers(),
        )
        if not res.is_success:
            raise RuntimeError(f"Lichess {res.status_code}: {res.text}")
        return True


async def resign_game(game_id: str) -> bool:
    async with httpx.AsyncClient() as client:
        res = await client.post(
            f"{BASE}/api/bot/game/{game_id}/resign", headers=_bot_headers()
        )
        return res.is_success


async def validate_bot_account() -> dict:
    """
    Validate that LICHESS_BOT_TOKEN belongs to a proper BOT account
    that is not the SenseRobot or any forbidden human account.

    Returns the account dict on success.
    Raises ValueError with a descriptive message (no token values) on failure.
    Raises RuntimeError if the token is not set.
    """
    async with httpx.AsyncClient() as client:
        res = await client.get(
            f"{BASE}/api/account",
            headers=_bot_headers(),
        )
        res.raise_for_status()
        account = res.json()

    account_id = account.get("id", "").lower()
    title = account.get("title", "")

    if title != "BOT":
        raise ValueError(
            f"Account '{account_id}' is not a BOT account. Run lichess/upgrade.py first."
        )

    if account_id != LICHESS_BOT_USERNAME.lower():
        raise ValueError(
            "LICHESS_BOT_TOKEN does not belong to LICHESS_BOT_USERNAME. Check .env."
        )

    if SENSEROBOT_LICHESS_USERNAME and account_id == SENSEROBOT_LICHESS_USERNAME.lower():
        raise ValueError(
            "Bot account cannot be the same as the SenseRobot account."
        )

    if account_id in ("sainju", "rohan_sainju"):
        raise ValueError(f"Account '{account_id}' is not allowed as the bot.")

    return account
