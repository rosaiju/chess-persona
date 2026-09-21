import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

# The live database lives at the repo root. CHESS_DB_PATH overrides it so the
# test suite can point at a throwaway file — without it, running pytest writes
# fake games into the real analytics DB and corrupts the insights.
# It is read at import time, so it must be set before analytics.db is imported
# (tests/conftest.py does this).
DB_PATH = Path(
    os.getenv("CHESS_DB_PATH") or (Path(__file__).parent.parent / "chess_analytics.db")
)


# Set once per process, on the first connection.
_pragmas_applied = False


@contextmanager
def _conn():
    """
    A connection scoped to the `with` block, committed and then closed.

    sqlite3.Connection's own context manager only wraps the *transaction* — it
    commits or rolls back but never closes. Every caller here used
    `with _conn() as con:`, so each call leaked a connection until the garbage
    collector got to it; on Windows that also kept the file locked. Wrapping
    `with con` preserves the exact commit/rollback semantics callers already
    rely on, and the finally closes the handle.

    timeout lets a writer wait for a competing write instead of failing
    immediately — post-game analysis writes from a worker thread while requests
    are reading.
    """
    global _pragmas_applied
    con = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30.0)
    con.row_factory = sqlite3.Row
    try:
        if not _pragmas_applied:
            # WAL lets reads proceed during a write, which is the common shape
            # here (insights read while analysis writes). It is a durable
            # property of the file, so it only needs setting once.
            con.execute("PRAGMA journal_mode=WAL")
            con.execute("PRAGMA synchronous=NORMAL")
            _pragmas_applied = True
        with con:
            yield con
    finally:
        con.close()


def init_db():
    with _conn() as con:
        con.executescript("""
            CREATE TABLE IF NOT EXISTS games (
                game_id      TEXT PRIMARY KEY,
                opponent     TEXT NOT NULL,
                personality  TEXT NOT NULL,
                ai_color     TEXT NOT NULL,
                result       TEXT,
                human_result TEXT,
                total_plies  INTEGER,
                played_at    TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS moves (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id      TEXT NOT NULL,
                ply          INTEGER NOT NULL,
                uci          TEXT NOT NULL,
                eval_before  INTEGER,
                eval_after   INTEGER,
                trigger      TEXT,
                is_ai_move   INTEGER NOT NULL,
                is_capture   INTEGER NOT NULL,
                gives_check  INTEGER NOT NULL,
                phase        TEXT NOT NULL,
                FOREIGN KEY (game_id) REFERENCES games(game_id)
            );
            CREATE INDEX IF NOT EXISTS idx_moves_game    ON moves(game_id);
            CREATE INDEX IF NOT EXISTS idx_moves_trigger ON moves(trigger);
            CREATE INDEX IF NOT EXISTS idx_games_opp     ON games(opponent);
        """)


def _add_col(con, table: str, column: str, col_type: str):
    existing = {row[1] for row in con.execute(f"PRAGMA table_info({table})")}
    if column not in existing:
        con.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")


def migrate_db():
    """Add new columns to existing tables. Safe to run on any existing DB."""
    with _conn() as con:
        for col, typ in [
            ("san",       "TEXT"),
            ("fen_after", "TEXT"),
            ("cp_white",  "INTEGER"),   # centipawns from White's perspective
            ("best_uci",  "TEXT"),      # Stockfish best move (filled by post-game analysis)
            ("cp_loss",   "INTEGER"),   # centipawn loss for moving player (filled by analysis)
        ]:
            _add_col(con, "moves", col, typ)

        for col, typ in [
            ("pgn",            "TEXT"),
            ("analysis_done",  "INTEGER DEFAULT 0"),
            ("accuracy_human", "REAL"),
            ("accuracy_ai",    "REAL"),
            ("ai_review",      "TEXT"),
            ("ai_review_done", "INTEGER DEFAULT 0"),
            ("difficulty",     "TEXT"),   # engine strength level the AI played at
            ("ai_review_error", "TEXT"),  # why the last coaching attempt failed
        ]:
            _add_col(con, "games", col, typ)


init_db()
migrate_db()


def _phase(ply: int) -> str:
    if ply <= 20:
        return "opening"
    if ply <= 60:
        return "middlegame"
    return "endgame"


def record_game_start(
    game_id: str,
    opponent: str,
    personality: str,
    ai_color: str,
    difficulty: str | None = None,
):
    with _conn() as con:
        con.execute(
            """INSERT OR IGNORE INTO games
               (game_id, opponent, personality, ai_color, difficulty, played_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (game_id, opponent, personality, ai_color, difficulty,
             datetime.now(timezone.utc).isoformat()),
        )


def record_move(
    game_id: str,
    ply: int,
    uci: str,
    san: "str | None",
    fen_after: "str | None",
    eval_before,
    eval_after,
    cp_white,           # raw centipawns from White's perspective (eval_after before negation)
    trigger,
    is_ai_move: bool,
    is_capture: bool,
    gives_check: bool,
):
    with _conn() as con:
        con.execute(
            """INSERT INTO moves
               (game_id, ply, uci, san, fen_after, eval_before, eval_after, cp_white,
                trigger, is_ai_move, is_capture, gives_check, phase)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (game_id, ply, uci, san, fen_after, eval_before, eval_after, cp_white,
             trigger, int(is_ai_move), int(is_capture), int(gives_check), _phase(ply)),
        )


def record_game_end(game_id: str, result: str, total_plies: int):
    with _conn() as con:
        row = con.execute(
            "SELECT ai_color FROM games WHERE game_id = ?", (game_id,)
        ).fetchone()
        if not row:
            return
        ai_color = row["ai_color"]
        if result == "1/2-1/2":
            human_result = "draw"
        elif result == "1-0":
            human_result = "loss" if ai_color == "white" else "win"
        elif result == "0-1":
            human_result = "win" if ai_color == "white" else "loss"
        else:
            human_result = None
        con.execute(
            """UPDATE games
               SET result = ?, human_result = ?, total_plies = ?
               WHERE game_id = ?""",
            (result, human_result, total_plies, game_id),
        )


def get_game(game_id: str) -> dict | None:
    with _conn() as con:
        row = con.execute("SELECT * FROM games WHERE game_id = ?", (game_id,)).fetchone()
        if not row:
            return None
        return dict(row)


def get_game_moves(game_id: str) -> list[dict]:
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM moves WHERE game_id = ? ORDER BY ply", (game_id,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_insights(opponent: str | None = None) -> dict:
    with _conn() as con:
        if opponent:
            games_rows = con.execute(
                "SELECT * FROM games WHERE opponent = ? ORDER BY played_at DESC",
                (opponent,),
            ).fetchall()
        else:
            games_rows = con.execute(
                "SELECT * FROM games ORDER BY played_at DESC"
            ).fetchall()

        if not games_rows:
            return {"opponent": opponent or "all", "games_played": 0}

        game_ids = [g["game_id"] for g in games_rows]
        placeholders = ",".join("?" * len(game_ids))
        moves_rows = con.execute(
            f"SELECT * FROM moves WHERE game_id IN ({placeholders})", game_ids
        ).fetchall()

    n = len(games_rows)
    n_analyzed = sum(1 for g in games_rows if g["analysis_done"])
    wins   = sum(1 for g in games_rows if g["human_result"] == "win")
    losses = sum(1 for g in games_rows if g["human_result"] == "loss")
    draws  = sum(1 for g in games_rows if g["human_result"] == "draw")

    human_moves = [m for m in moves_rows if not m["is_ai_move"]]

    # Use post-game Stockfish cp_loss for all quality classifications.
    # Thresholds match the inline game review panel exactly.
    blunders = [m for m in human_moves if m["cp_loss"] is not None and m["cp_loss"] > 200]
    mistakes = [m for m in human_moves if m["cp_loss"] is not None and 75 < m["cp_loss"] <= 200]
    # Good = best move (cp_loss=0) + excellent (cp_loss 1-25)
    good     = [m for m in human_moves if m["cp_loss"] is not None and m["cp_loss"] <= 25]

    phase_breakdown = {}
    for phase in ("opening", "middlegame", "endgame"):
        ph = [m for m in human_moves if m["phase"] == phase]
        phase_breakdown[phase] = {
            "blunders":   sum(1 for m in ph if m["cp_loss"] is not None and m["cp_loss"] > 200),
            "mistakes":   sum(1 for m in ph if m["cp_loss"] is not None and 75 < m["cp_loss"] <= 200),
            "good_moves": sum(1 for m in ph if m["cp_loss"] is not None and m["cp_loss"] <= 25),
        }

    sq_counts: dict[str, int] = {}
    for m in blunders:
        sq = m["uci"][2:4]
        sq_counts[sq] = sq_counts.get(sq, 0) + 1
    top_squares = sorted(sq_counts.items(), key=lambda x: -x[1])[:5]

    recent = []
    for g in games_rows[:5]:
        gid = g["game_id"]
        gm = [m for m in moves_rows if m["game_id"] == gid and not m["is_ai_move"]]
        recent.append({
            "game_id":        gid,
            "opponent":       g["opponent"],
            "result":         g["human_result"],
            "difficulty":     g["difficulty"],
            "blunders":       sum(1 for m in gm if m["cp_loss"] is not None and m["cp_loss"] > 200),
            "mistakes":       sum(1 for m in gm if m["cp_loss"] is not None and 75 < m["cp_loss"] <= 200),
            "good_moves":     sum(1 for m in gm if m["cp_loss"] is not None and m["cp_loss"] <= 25),
            "played_at":      g["played_at"],
            "url":            f"https://lichess.org/{gid}",
            "accuracy_human": g["accuracy_human"],
            "accuracy_ai":    g["accuracy_ai"],
            "analysis_done":  g["analysis_done"],
        })

    return {
        "opponent":        opponent or "all",
        "games_played":    n,
        "wins":            wins,
        "losses":          losses,
        "draws":           draws,
        "win_rate":        round(wins / n, 2) if n else 0,
        "per_game": {
            "blunders":    round(len(blunders) / n_analyzed, 2) if n_analyzed else 0,
            "mistakes":    round(len(mistakes) / n_analyzed, 2) if n_analyzed else 0,
            "good_moves":  round(len(good)     / n_analyzed, 2) if n_analyzed else 0,
        },
        "phase_breakdown": phase_breakdown,
        "blunder_squares": [{"square": sq, "count": cnt} for sq, cnt in top_squares],
        "recent_games":    recent,
    }


def get_coaching_review(game_id: str) -> dict:
    with _conn() as con:
        row = con.execute(
            "SELECT ai_review, ai_review_done, ai_review_error FROM games WHERE game_id = ?",
            (game_id,),
        ).fetchone()
        if not row:
            return {"done": False, "review": None, "error": None}
        return {
            "done": bool(row["ai_review_done"]),
            "review": row["ai_review"],
            "error": row["ai_review_error"],
        }


def save_coaching_review(game_id: str, review: str):
    with _conn() as con:
        con.execute(
            "UPDATE games SET ai_review = ?, ai_review_done = 1, ai_review_error = NULL "
            "WHERE game_id = ?",
            (review, game_id),
        )


def save_coaching_error(game_id: str, message: str):
    """
    Record why a coaching attempt failed.

    Without this a failure left ai_review_done at 0 forever, so the status
    endpoint kept answering {done: false} with HTTP 200 and the review page
    polled indefinitely with a spinner.
    """
    with _conn() as con:
        con.execute(
            "UPDATE games SET ai_review_error = ? WHERE game_id = ?",
            (message[:500], game_id),
        )


def clear_coaching_error(game_id: str):
    """Called when a retry starts, so the UI stops showing the previous failure."""
    with _conn() as con:
        con.execute(
            "UPDATE games SET ai_review_error = NULL WHERE game_id = ?", (game_id,)
        )
