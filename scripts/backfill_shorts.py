"""Make and upload the Short for videos that went out without one.

    python scripts/backfill_shorts.py 43 44

For a video whose daily run lost its Short. Uses the same code as the daily run, so a Short
made here is the Short the daily run would have made. Exits 1 if any of them failed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import core.logger  # noqa: F401,E402
from core import db  # noqa: E402
from core.config import load_config  # noqa: E402
from pipeline import short  # noqa: E402


def main(ids):
    cfg = load_config()
    db.init_db()
    failed = []
    for vid in ids:
        try:
            short.run(int(vid), cfg)
            print(f"video {vid}: Short {db.get_video(int(vid)).get('short_youtube_id')}")
        except Exception as e:                                   # noqa: BLE001
            print(f"video {vid}: FAILED {type(e).__name__}: {e}")
            failed.append(vid)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
