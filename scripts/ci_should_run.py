"""Whether this run has anything to do, so a second cron can exist as a safety net.

    python scripts/ci_should_run.py        # prints  run=true|false  and  reason=...

GitHub's free runners start scheduled jobs when they have room, not when the cron says: every
scheduled run of this repository so far began between 2h55 and 5h44 late, and GitHub's own
documentation warns that a scheduled run may be dropped altogether when the load is high
enough. One cron a day is therefore one chance a day, and a dropped chance is a dark day on
the channel three days later.

So the workflow has two crons, and this decides what the later one does:

- a video is half-built and still has attempts left  -> run, and carry it on. Resuming is
  cheap: the pictures and the voice already paid for are restored from the cache.
- nothing was finished yet today                     -> run, and start one. This is the case
  that rescues a dropped or failed first cron.
- a video was already finished today                 -> do nothing. Two videos in one day
  would cost more Neurons than the daily allowance holds anyway.

Lines are KEY=value so the output can be appended straight to $GITHUB_OUTPUT.
"""
import sqlite3
import sys
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import db


def finished_today(now=None):
    """The id of a video whose upload stage finished today (UTC), or None."""
    day = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%d")
    with closing(sqlite3.connect(db.DB_PATH)) as conn:
        row = conn.execute(
            "SELECT video_id FROM events WHERE stage = 'upload' AND level = 'info' "
            "AND message = 'stage finished' AND created_at >= ? ORDER BY id DESC LIMIT 1",
            (day,)).fetchone()
    return row[0] if row else None


def decide(now=None):
    """(run, reason)."""
    unfinished = db.oldest_unfinished(db.MAX_ATTEMPTS)
    if unfinished:
        attempts = (unfinished.get("attempts") or 0) + 1
        return True, (f"video {unfinished['id']} is unfinished (attempt {attempts} of "
                      f"{db.MAX_ATTEMPTS}), carrying it on")
    done = finished_today(now)
    if done is None:
        return True, "nothing has been finished today, starting a video"
    return False, (f"video {done} was already finished today; a second one would not fit "
                   f"the daily Neuron allowance")


def main():
    db.init_db()
    run, reason = decide()
    print(f"run={'true' if run else 'false'}")
    print(f"reason={reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
