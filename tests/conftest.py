"""
Test-wide fixtures.

The important job here is isolating the database. `analytics.db` resolves
DB_PATH at import time, so CHESS_DB_PATH has to be set before any test module
(or anything they import) touches it. conftest.py is imported before test
collection, which makes this the only reliable place to do it.

Without this, every pytest run inserted a fake "game1" into the real
chess_analytics.db — polluting the recent-games list and the game count.
"""
import os
import tempfile
from pathlib import Path

import pytest

# Must happen at import time, before `import analytics.db` anywhere.
_TMP_DB = Path(tempfile.gettempdir()) / "chess_persona_test.db"
_TMP_DB.unlink(missing_ok=True)
os.environ["CHESS_DB_PATH"] = str(_TMP_DB)


@pytest.fixture(scope="session", autouse=True)
def _guard_production_db():
    """Fail loudly if a test is ever pointed at the real database."""
    from analytics.db import DB_PATH

    real = (Path(__file__).parent.parent / "chess_analytics.db").resolve()
    assert Path(DB_PATH).resolve() != real, (
        f"tests are writing to the production database at {real} — "
        "CHESS_DB_PATH was not applied before analytics.db was imported"
    )
    yield
    # analytics.db closes every connection it opens, so nothing holds the file
    # here. If this ever starts raising on Windows, a connection is leaking
    # again — that is worth failing on rather than swallowing.
    _TMP_DB.unlink(missing_ok=True)
