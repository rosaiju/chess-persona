"""
Development server with working auto-reload.

    python dev.py            # port 8001
    python dev.py 8080       # any other port

Why this does the watching itself instead of passing --reload to uvicorn:

On this setup uvicorn's own reloader detects the change and logs
"WatchFiles detected changes ... Reloading", but the worker process never
actually restarts -- verified by watching PIDs across an edit, the worker kept
its PID and kept serving the old code for 50s+. Edits therefore appeared to
apply while the server silently ran stale code, which is worse than no reload
at all.

So uvicorn is started as a plain subprocess with no reloader, and this script
watches the source tree and restarts that subprocess itself. Killing a process
we spawned is something we can actually guarantee.

templates/*.html are not watched on purpose: app.py re-reads them per request,
so HTML and JS edits show up on a browser refresh with no restart at all.
"""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from watchfiles import watch

ROOT = Path(__file__).parent

# Directories holding server code. tests/ is excluded so running pytest
# alongside the server does not bounce it.
WATCH_DIRS = [ROOT, ROOT / "lichess", ROOT / "analytics", ROOT / "persona"]

IGNORE_PARTS = {"__pycache__", ".pytest_cache", ".git", "tests", "templates"}


def _is_source_change(_change, path: str) -> bool:
    p = Path(path)
    if p.suffix != ".py":
        return False
    return not (IGNORE_PARTS & set(p.parts))


def _spawn(port: int) -> subprocess.Popen:
    cmd = [sys.executable, "-m", "uvicorn", "app:app",
           "--host", "127.0.0.1", "--port", str(port)]
    kwargs = {"cwd": str(ROOT)}
    if os.name == "nt":
        # Own process group, so we can signal the whole tree (uvicorn may have
        # Stockfish children mid-game).
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    return subprocess.Popen(cmd, **kwargs)


def _stop(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    if os.name == "nt":
        # terminate() alone can leave the tree alive on Windows.
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       capture_output=True)
    else:
        proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def main() -> int:
    port = 8001
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Not a port number: {sys.argv[1]!r}")
            return 2

    print(f"Chess Persona dev server  ->  http://127.0.0.1:{port}")
    print("Watching *.py for changes. HTML/JS edits apply on browser refresh.")
    print("Ctrl+C to stop.\n")

    proc = _spawn(port)
    try:
        for changes in watch(*WATCH_DIRS, watch_filter=_is_source_change,
                             debounce=400, step=100):
            names = sorted({Path(p).name for _, p in changes})
            print(f"\n[dev] changed: {', '.join(names)} -- restarting\n")
            _stop(proc)
            time.sleep(0.3)          # let the port free up
            proc = _spawn(port)
    except KeyboardInterrupt:
        print("\n[dev] stopping")
    finally:
        _stop(proc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
