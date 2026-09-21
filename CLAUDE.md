# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the app

```bash
python dev.py           # port 8001, auto-reload
python dev.py 8080      # any other port
```

**Do not use `uvicorn --reload` here.** On this setup uvicorn's reloader logs
"WatchFiles detected changes ... Reloading" but the worker process never
actually restarts — verified by watching PIDs across an edit, the worker kept
its PID and kept serving stale code for 50s+. Edits appear to apply while the
old code is still running, which is worse than no reload at all.

`dev.py` therefore runs uvicorn as a plain subprocess with no reloader and does
the watching itself, restarting that subprocess on any `*.py` change (~2s).
`tests/` is excluded so running pytest does not bounce the server.
`templates/*.html` are not watched because `app.py` re-reads them per request —
HTML and JS edits apply on a browser refresh with no restart.

For production, or when reload is not wanted:

```bash
python -m uvicorn app:app --port 8001
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
| `quip` | personality line to speak/display (includes `text`, `personality`, `capture`, `delay_ms`) |
| `done` | game over (includes `status`, `result`, `gameId`, `url`) |
| `declined` / `timeout` / `error` | challenge not accepted |

### TTS

`GET /tts?text=...&personality=...` — generates MP3 audio via `edge-tts`. The browser fetches this and plays it inline. The SenseRobot client also fetches from this endpoint and plays the audio locally.

Voice-per-personality mapping is in `app.py::VOICE_CONFIG`.

### Quip timing

After the robot moves, its quip should not be spoken until the SenseRobot arm
has physically finished the move. That wait is **not** performed in the game
loop — `play_game()` is the only consumer of `stream_game()`, so sleeping there
would leave the human's next move unread until the delay elapsed.

Instead the quip event carries `delay_ms`, and each consumer schedules it:
`templates/index.html` via `speak(text, delayMs)`, and `senserobot_client.py`
when it logs. `ROBOT_MOVE_DELAY` (env var, default 7s) sets the value; quips not
tied to an AI move carry `delay_ms: 0`. The browser adds its own extra 5s pause
for capture/check quips in SenseRobot mode on top of `delay_ms`.

Event order after an AI move is `move` → `fen` → `quip`. `move` announces what
just happened, `fen` sets whose turn it is and so has the last word on the
status line, and the quip only ever adds a chat bubble. Emitting `move` last
would let it overwrite the fresher "your turn" status.

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

### Evaluation vs play strength

`play_game()` opens **two** engines. `engine` plays and is capped to the chosen
difficulty; `eval_engine` only scores positions and is always full strength.
`UCI_LimitStrength` distorts `analyse()` as well as `play()` — a quiet position
reads about -5cp uncapped but -25 to -35cp at 1600, with much wider spread.
Those scores drive the quip triggers and are stored as `moves.cp_white`, so
scoring with the capped engine would make accuracy depend on the difficulty the
human picked. When difficulty is `max` there is no cap and one engine serves both.

Post-game analysis (`analytics/analysis.py`) re-evaluates every position with
its own full-strength engine at `ANALYSIS_TIME` (0.2s, ~depth 20) rather than
reusing the live `cp_white`. Both halves of `cp_loss` must come from the same
search: differencing two independent searches turns search noise into phantom
centipawn loss. On a real 50-ply game mean cp_loss was 1991 under the old scheme
and 35 once the evals were consistent. cp_loss is
`eval(position before, mover's POV) − eval(position after, mover's POV)`, which
needs two searches per move rather than three.

### Ending a game early

`POST /game/{game_id}/resign` stops whatever is in flight. Lichess accepts only
one of resign / abort / cancel depending on state (under way / first move /
challenge still pending), so `lichess/api.py::stop_game()` tries each in turn
and reports which took. The `/play` stream ends on its own once Lichess reports
the game over — the endpoint does not touch it.

The UI shows a Resign button from `waiting` onward and hides it on every
terminal event.

### Coaching failures

Every failure path in `analytics/coaching.py` writes `games.ai_review_error`.
Previously a failure just left `ai_review_done` at 0, so `GET
/review/{id}/coaching` kept answering `{done: false}` with HTTP 200 and the page
polled every 2s forever with a spinner.

`_friendly_error()` maps SDK exceptions to something actionable (invalid key,
exhausted free-tier quota, permission denied, timeout) rather than surfacing a
raw JSON error blob. The request is bounded twice: the SDK gets
`REQUEST_TIMEOUT_S` (90s) and `generate_review` wraps it in an
`asyncio.wait_for` 30s wider, so a hang below the HTTP layer still resolves into
a visible error. The page also caps polling at `COACHING_MAX_POLLS` (~3 min).

A `POST` is treated as a retry and clears any recorded error first; a successful
review clears it too.

## Testing

`analytics/db.py` reads `CHESS_DB_PATH` at import time, falling back to
`chess_analytics.db` at the repo root. `tests/conftest.py` sets that env var to
a temp file **before** any test module imports `analytics.db` — that ordering is
the whole point, since DB_PATH is resolved once at import.

Without it, every pytest run inserted a fake `game1` into the live analytics DB
(37 move rows per run), which then showed up in `/insights` as an extra game and
sat at the top of the recent-games list. A session fixture asserts the test DB is
not the production one, and `test_tests_do_not_use_the_production_database`
checks the same thing.

## Key files

- `app.py` — FastAPI routes, TTS endpoint, SSE broadcast infrastructure
- `lichess/player.py` — game loop, Stockfish integration, trigger classification
- `lichess/api.py` — Lichess Board API calls (challenge, move, stream game/events, resign)
- `persona/personality.py` — all quips for all 4 personalities × all triggers
- `templates/index.html` — entire frontend (vanilla JS, chess.js for board rendering, no build step)
- `senserobot_client.py` — standalone script that listens on `/events` and plays TTS audio locally

## Quip triggers

Defined in `lichess/player.py::_classify_trigger()`. Triggers are strings like `"robot_captures"`, `"human_blunders"`, `"robot_winning"`, etc. Positional triggers (`robot_winning`, `human_winning`, `endgame`) are rate-limited to fire at most once every 6 half-moves. Plain `robot_move` fires on ~45% of moves after move 6.
