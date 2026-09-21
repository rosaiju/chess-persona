# Chess Persona

A chess-playing bot with personality. Challenges a Lichess user, plays using Stockfish, and trash-talks via TTS.

---

## Architecture Overview

Three separate Lichess accounts are required:

| Account | Role | Token |
|---|---|---|
| `ChessPersonaBot` | BOT account — runs Stockfish, submits moves via `/api/bot/game/...` | `LICHESS_BOT_TOKEN` |
| `Sainju` | Physical SenseRobot board — submits human moves via `/api/board/game/...` | `LICHESS_HUMAN_TOKEN` (browser `/move` only) |
| `Rohan_Sainju` | Normal human — never automated | — |

**Why they must be separate:** The SenseRobot hardware mirrors moves made by the *opposing* account. If the bot and the physical board share an account, the board will not physically execute the bot's moves (it would see them as its own). The bot must be a dedicated account upgraded to BOT status.

### 8-Step Game Flow

1. User clicks Play in the browser → `POST /play`
2. `ChessPersonaBot` challenges `Sainju` via Lichess challenge API
3. `Sainju` accepts (manually or via SenseRobot board)
4. SenseRobot streams the game; `Sainju` submits White's physical moves
5. `ChessPersonaBot` receives the move via Bot API stream
6. Stockfish calculates the best reply
7. `ChessPersonaBot` submits the reply via `POST /api/bot/game/{id}/move/{uci}`
8. SenseRobot mirrors the Black piece physically

---

## Initial Setup

### 1. Create the bot account

Create a new Lichess account for the bot (e.g. `ChessPersonaBot`). Do **not** use `Sainju` or `Rohan_Sainju`.

### 2. Generate a BOT API token

In `ChessPersonaBot`'s Lichess settings → API access tokens → generate with scopes:
- `bot:play`
- `challenge:write`
- `challenge:read`

### 3. Configure `.env`

```
LICHESS_BOT_TOKEN=your_chesspersonabot_token
LICHESS_BOT_USERNAME=ChessPersonaBot
LICHESS_HUMAN_TOKEN=your_sainju_token
SENSEROBOT_LICHESS_USERNAME=Sainju
```

### 4. Upgrade the bot account (one-time, irreversible)

```bash
python lichess/upgrade.py
```

This calls `POST https://lichess.org/api/bot/account/upgrade`. **This cannot be undone** — the account can never play as a human again. Never run this for `Sainju` or `Rohan_Sainju`.

---

## Running the App

```bash
pip install -r requirements.txt
uvicorn app:app --port 8001 --reload
```

Open `http://localhost:8001`. Verify the setup with:

```
GET http://localhost:8001/account
```

Expected response:
```json
{"bot_username": "ChessPersonaBot", "senserobot_username": "Sainju", "status": "ok"}
```

---

## Difficulty

Pick a strength level on the play screen before challenging. The level caps Stockfish via `UCI_Elo`, so the robot is beatable:

| Level | Approx. Elo | Feel |
|---|---|---|
| Beginner | 1320 | Hangs pieces. A fair fight for a first game. |
| **Casual** (default) | 1600 | Solid basics, still misses tactics. |
| Club | 1900 | Punishes real mistakes. You'll need a plan. |
| Strong | 2200 | Rarely errs. Expect to be ground down. |
| Max | unlimited | Unrestricted Stockfish. Good luck. |

The selector is locked while a game is in progress, and your last choice is remembered between sessions. The level played is recorded with each game.

Post-game accuracy and the AI coaching review always run against **full-strength** Stockfish, so your numbers stay comparable across levels.

---

## SenseRobot Mode vs Browser Mode

| Mode | How moves are submitted |
|---|---|
| **Browser mode** (default) | Click pieces on the board → `POST /move` → Lichess Board API as `Sainju` |
| **SenseRobot mode** | Physical board submits moves directly to Lichess as `Sainju`; the browser board is view-only |

Check "SenseRobot mode (physical board)" before clicking Play to enable view-only mode. The board will display moves as they happen but will not respond to clicks.

---

## Running Tests

No live Lichess accounts needed — all API calls are mocked.

```bash
pip install -r requirements.txt
pytest tests/ -v
```

All 37 tests should pass.

---

## Account Constraints

- **BOT upgrade is irreversible.** Only upgrade `ChessPersonaBot`. Never upgrade `Sainju` or `Rohan_Sainju`.
- `LICHESS_BOT_TOKEN` must belong to a BOT-titled account. The app validates this on every `/play` request via `validate_bot_account()`.
- The bot account cannot be the same as `SENSEROBOT_LICHESS_USERNAME`.
- `sainju` and `rohan_sainju` are hard-blocked from being used as the bot account.
