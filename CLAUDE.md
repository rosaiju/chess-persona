# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the app

```bash
python -m uvicorn app:app --port 8001 --reload
```

The app serves everything from a single FastAPI server — no separate frontend build step. The UI is a vanilla HTML/JS file served directly by FastAPI at `GET /`.

**Required:** A `.env` file with a Lichess API token (see `.env.example`):
```
LICHESS_TOKEN=your_lichess_api_token_here
```

**Optional:** `LICHESS_HUMAN_TOKEN` in `.env` — needed only if the `/move` endpoint is used (human plays moves programmatically instead of on Lichess directly).

**SenseRobot client** (separate process, run on the machine connected to the robot):
```bash
python senserobot_client.py http://<server-ip>:8001
```

## Architecture

The app is a chess-playing robot with personality. The robot challenges a Lichess user, plays as White using Stockfish, and emits trash-talk quips via TTS based on what's happening in the game.

### Request flow

1. **`POST /play`** — browser sends `{opponent, personality, difficulty, senserobot_mode}` → returns an SSE stream
2. `lichess/player.py::play_game()` is the core async generator — it challenges the opponent, waits for acceptance, then loops on `stream_game()` events
3. After each move (robot's or human's), `_classify_trigger()` picks a quip trigger based on Stockfish eval delta, check, capture, or game phase
4. `persona/personality.py::get_quip()` returns a random quip for `(personality, trigger)`, avoiding back-to-back repeats
5. Quip events are both streamed to the browser (SSE on `/play`) and broadcast to any `/events` subscribers (for the SenseRobot client)

### SSE event types

| type | meaning |
|---|---|
| `challenging` | sent challenge to opponent |
| `waiting` | challenge sent, waiting for accept |
| `started` | game accepted, play begins (includes `difficulty`, `difficulty_label`, `difficulty_elo`) |
| `thinking` | robot is computing its move |
| `fen` | board position updated |
| `move` | robot played a move (includes `uci`, `fen`, `moveNum`, `gameId`) |
| `quip` | personality line to speak/display (includes `text`, `personality`) |
| `done` | game over (includes `status`, `result`, `gameId`, `url`) |
| `declined` / `timeout` / `error` | challenge not accepted |

### TTS

`GET /tts?text=...&personality=...` — generates MP3 audio via `edge-tts`. The browser fetches this and plays it inline. The SenseRobot client also fetches from this endpoint and plays the audio locally.

Voice-per-personality mapping is in `app.py::VOICE_CONFIG`.

### Dual-audience broadcast

`_subscribers` in `app.py` is an in-memory set of asyncio Queues. `POST /play` writes to these queues via `_broadcast()` for any connected `/events` SSE clients (the SenseRobot client). The browser and the robot receive the same quip events independently.

### Stockfish

Loaded once per game via `chess.engine.SimpleEngine` in a thread (to avoid blocking the event loop). Path falls back to a hardcoded WinGet install location if `stockfish` isn't on PATH. If Stockfish is unavailable, the robot falls back to the first legal move and skips eval-based quip triggers.

### Difficulty

`lichess/player.py::DIFFICULTIES` defines the playable strength levels. Each entry sets a `UCI_Elo` cap (applied by `_configure_strength()` when the engine opens) and a per-move search budget:

| key | label | UCI_Elo | move_time |
|---|---|---|---|
| `beginner` | Beginner | 1320 | 0.10s |
| `casual` | Casual (default) | 1600 | 0.20s |
| `club` | Club | 1900 | 0.30s |
| `strong` | Strong | 2200 | 0.50s |
| `max` | Max | *unlimited* | 0.50s |

`elo: None` means the strength limiter is left off entirely. Values are clamped to whatever range the installed Stockfish reports for `UCI_Elo` (1320–3190 on Stockfish 19); an engine build with no `UCI_Elo` support logs a warning and plays at full strength rather than failing.

`GET /difficulties` serves this table to the frontend, which renders the selector from it — the server is the single source of truth for labels and Elo caps. The chosen level is stored per-game in `games.difficulty` and echoed back on the `started` event as `difficulty` / `difficulty_label` / `difficulty_elo`. The browser remembers the last pick in `localStorage` under `cp-difficulty`.

**Post-game analysis is unaffected** — `analytics/analysis.py` opens its own unrestricted engine, so accuracy and coaching are always measured against full-strength Stockfish regardless of the level played.

## Key files

- `app.py` — FastAPI routes, TTS endpoint, SSE broadcast infrastructure
- `lichess/player.py` — game loop, Stockfish integration, trigger classification
- `lichess/api.py` — Lichess Board API calls (challenge, move, stream game/events, resign)
- `persona/personality.py` — all quips for all 4 personalities × all triggers
- `templates/index.html` — entire frontend (vanilla JS, chess.js for board rendering, no build step)
- `senserobot_client.py` — standalone script that listens on `/events` and plays TTS audio locally

## Quip triggers

Defined in `lichess/player.py::_classify_trigger()`. Triggers are strings like `"robot_captures"`, `"human_blunders"`, `"robot_winning"`, etc. Positional triggers (`robot_winning`, `human_winning`, `endgame`) are rate-limited to fire at most once every 6 half-moves. Plain `robot_move` fires on ~45% of moves after move 6.
