"""
Standalone CLI script to upgrade a Lichess account to BOT status.

WARNING: This action is IRREVERSIBLE. A BOT account can never play as a human again.
Never import this module from app.py or player.py.

Usage:
    python lichess/upgrade.py
"""
import os
import sys

import httpx
from dotenv import load_dotenv

load_dotenv()


def main():
    token = os.environ.get("LICHESS_BOT_TOKEN", "")
    if not token:
        print("Error: LICHESS_BOT_TOKEN not set in your .env file.")
        sys.exit(1)

    res = httpx.post(
        "https://lichess.org/api/bot/account/upgrade",
        headers={"Authorization": f"Bearer {token}"},
    )

    if res.status_code == 200:
        print("Account successfully upgraded to BOT status.")
    elif res.status_code == 400:
        try:
            body = res.json()
            reason = body.get("error", res.text)
        except Exception:
            reason = res.text
        print(f"Upgrade failed: {reason}")
        sys.exit(1)
    elif res.status_code == 401:
        print("Invalid token — check LICHESS_BOT_TOKEN in your .env file.")
        sys.exit(1)
    else:
        print(f"Unexpected response ({res.status_code}): {res.text}")
        sys.exit(1)


if __name__ == "__main__":
    main()
