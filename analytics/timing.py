"""
Phase timing for the analysis and review pipeline.

Added because "the review takes 2-3 minutes" was not attributable without
measurement, and the assumption that the language model was responsible turned
out to be wrong: Stockfish dominated. Every slow path now reports where its
time went, so the next regression does not need guesswork either.

Usage:

    with Timer("analysis") as t:
        with t.phase("stockfish"):
            ...
        with t.phase("db_write"):
            ...
    # logs: [timing] analysis 19.31s (stockfish 19.18s, db_write 0.09s)

Timings are also returned as a dict so endpoints can hand them to the UI.
"""
import logging
import time
from contextlib import contextmanager

log = logging.getLogger(__name__)


class Timer:
    def __init__(self, label: str, log_on_exit: bool = True):
        self.label = label
        self.log_on_exit = log_on_exit
        self.phases: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self.total = 0.0
        self._start = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.total = time.perf_counter() - self._start
        if self.log_on_exit:
            log.info("[timing] %s", self.summary())
        return False

    @contextmanager
    def phase(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self.phases[name] = self.phases.get(name, 0.0) + dt
            self.counts[name] = self.counts.get(name, 0) + 1

    def add(self, name: str, seconds: float):
        """Record a phase measured elsewhere."""
        self.phases[name] = self.phases.get(name, 0.0) + seconds
        self.counts[name] = self.counts.get(name, 0) + 1

    def summary(self) -> str:
        elapsed = self.total or (time.perf_counter() - self._start)
        parts = []
        for name, secs in sorted(self.phases.items(), key=lambda kv: -kv[1]):
            n = self.counts.get(name, 1)
            parts.append(f"{name} {secs:.2f}s" + (f" x{n}" if n > 1 else ""))
        detail = ", ".join(parts)
        return f"{self.label} {elapsed:.2f}s" + (f" ({detail})" if detail else "")

    def as_dict(self) -> dict:
        return {
            "total_s": round(self.total, 3),
            "phases": {k: round(v, 3) for k, v in self.phases.items()},
            "counts": dict(self.counts),
        }
