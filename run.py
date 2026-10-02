import argparse
import sys
from datetime import datetime

from core import db
from core.config import ConfigError, load_config
from core.logger import get_logger
from pipeline import images, render, script, seo, thumbnail, topic, tts

log = get_logger("run")

# (stage name, module, video status while the stage runs)
STAGES = [
    ("topic", topic, "pending"),
    ("script", script, "scripting"),
    ("tts", tts, "tts"),
    ("images", images, "images"),
    ("render", render, "rendering"),
    # seo before thumbnail: the thumbnail's words are taken from the final YouTube title,
    # which is the seo stage's output, not the script's working title.
    ("seo", seo, "seo"),
    ("thumbnail", thumbnail, "thumbnail"),
    # No upload here on purpose. Production ends at status `ready`; publication is a separate
    # process on its own schedule (publish.py), so a slow or failed render can never make
    # a publishing day late. See README, "виробництво окремо від публікації".
]


def process_video(video_id, cfg):
    """Run all stages for one video. Never raises: failures are recorded as status=failed."""
    for name, module, status in STAGES:
        try:
            db.update_video(video_id, status=status)
            db.log_event(video_id, name, "info", "stage started")
            if not module.run(video_id, cfg):
                raise RuntimeError(f"stage {name} returned False")
            db.log_event(video_id, name, "info", "stage finished")
        except Exception as e:
            log.error("video %s failed at stage %s: %s", video_id, name, e)
            try:
                db.log_event(video_id, name, "error", str(e))
                db.update_video(video_id, status="failed", error=f"{name}: {e}")
            except Exception as db_err:
                log.error("could not record failure in DB: %s", db_err)
            return False
    db.update_video(video_id, status="ready")
    log.info("video %s finished: ready", video_id)
    return True


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="YouTube Content Factory")
    parser.add_argument("--dry-run", action="store_true",
                        help="show the stages without touching the database")
    args = parser.parse_args()

    try:
        cfg = load_config()
    except ConfigError as e:
        log.error("config error: %s", e)
        return 2

    if args.dry_run:
        log.info("dry-run: stages would run in order: %s", " -> ".join(s[0] for s in STAGES))
        return 0

    db.init_db()
    placeholder = f"Placeholder topic {datetime.now():%Y-%m-%d %H:%M:%S}"
    video_id = db.create_video(placeholder)
    log.info("created video id=%s", video_id)
    return 0 if process_video(video_id, cfg) else 1


if __name__ == "__main__":
    sys.exit(main())
