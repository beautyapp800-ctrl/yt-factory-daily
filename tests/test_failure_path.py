"""Failure-path tests. Run: python tests/test_failure_path.py (no pytest needed)."""
import logging
from contextlib import closing
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import run
from core import db
from core.logger import get_logger
from core.retry import retry


class ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok: {msg}")


def test_stage_failure(tmp):
    print("test_stage_failure")
    db.DB_PATH = Path(tmp) / "test.db"  # never touch the real DB
    db.init_db()

    class Boom:
        @staticmethod
        def run(video_id, cfg):
            raise RuntimeError("boom in tts")

    original = run.STAGES
    run.STAGES = [(n, Boom if n == "tts" else m, st) for n, m, st in original]
    try:
        vid = db.create_video("failure test topic")
        result = run.process_video(vid, {})  # must not raise
    finally:
        run.STAGES = original

    check(result is False, "process_video returned False without raising")
    with closing(sqlite3.connect(db.DB_PATH)) as c:
        status, error = c.execute("SELECT status, error FROM videos WHERE id=?", (vid,)).fetchone()
        events = c.execute("SELECT stage, level, message FROM events WHERE video_id=?", (vid,)).fetchall()
    check(status == "failed", f"status is failed (got {status})")
    check("boom in tts" in error, f"error saved in videos.error ({error!r})")
    check(("tts", "error", "boom in tts") in events, "error saved in events table")
    check(not any(s in ("images", "render", "upload") for s, _, _ in events),
          "later stages were not run")


def test_retry(tmp):
    print("test_retry")
    handler = ListHandler()
    logging.getLogger("factory").addHandler(handler)
    real_sleep, time.sleep = time.sleep, lambda s: None
    try:
        calls = {"n": 0}

        @retry(attempts=3, base_delay=2)
        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise ValueError(f"fail {calls['n']}")
            return "done"

        check(flaky() == "done" and calls["n"] == 3, "succeeds on 3rd attempt, returns result")
        warnings = [r for r in handler.records if r.levelno == logging.WARNING]
        check(len(warnings) == 2, f"2 failed attempts logged ({len(warnings)})")

        handler.records.clear()
        calls["n"] = 0

        @retry(attempts=3, base_delay=2)
        def always():
            calls["n"] += 1
            raise KeyError("nope")

        try:
            always()
            check(False, "should have raised")
        except KeyError:
            check(calls["n"] == 3, "always-failing raised after exactly 3 attempts")
        warnings = [r for r in handler.records if r.levelno == logging.WARNING]
        check(len(warnings) == 3, f"3 failed attempts logged ({len(warnings)})")
    finally:
        time.sleep = real_sleep
        logging.getLogger("factory").removeHandler(handler)


def test_main_survives(tmp):
    print("test_main_survives")
    db.DB_PATH = Path(tmp) / "main.db"

    class Boom:
        @staticmethod
        def run(video_id, cfg):
            raise RuntimeError("boom")

    original = run.STAGES
    run.STAGES = [("topic", Boom, "pending")] + original[1:]
    sys.argv = ["run.py"]
    try:
        code = run.main()
    finally:
        run.STAGES = original
    check(code == 1, "main() returned exit code 1 instead of crashing")


if __name__ == "__main__":
    get_logger("test")  # ensure logging is set up
    with tempfile.TemporaryDirectory() as tmp:
        test_stage_failure(tmp)
        test_retry(tmp)
        test_main_survives(tmp)
    print("ALL TESTS PASSED")
