"""Which video a CI run is about to work on, so its files can be restored before it starts.

    python scripts/ci_state.py            # prints  video_id=7  and  resuming=true|false

A GitHub runner is thrown away after every run, and output/ is not in git, so the files a
run produced - above all the pictures, which cost Neurons - only survive if something carries
them over. That something is the Actions cache, keyed by video id, and the id has to be known
before the run starts, because the cache is restored first. This answers it from the database
that was checked out: the unfinished video the run will carry on with (run.py picks the same
one, using the same function), or the id the next new video will get.

Lines are written as KEY=value so the output can be appended straight to $GITHUB_OUTPUT.
"""
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import db

MAX_ATTEMPTS = 3        # must match run.py


def next_video_id():
    """The id a video created now would get. AUTOINCREMENT never reuses one, so it is the
    sequence counter plus one, not the highest id still present."""
    with closing(sqlite3.connect(db.DB_PATH)) as conn:
        row = conn.execute("SELECT seq FROM sqlite_sequence WHERE name = 'videos'").fetchone()
        return (row[0] if row else 0) + 1


def main():
    db.init_db()
    existing = db.oldest_unfinished(db.MAX_ATTEMPTS)
    if existing:
        print(f"video_id={existing['id']}")
        print("resuming=true")
    else:
        print(f"video_id={next_video_id()}")
        print("resuming=false")
    return 0


if __name__ == "__main__":
    sys.exit(main())
