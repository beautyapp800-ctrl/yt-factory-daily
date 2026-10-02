"""Publication, as its own process: upload the oldest finished video to YouTube.

    python publish.py            # the oldest video with status `ready`
    python publish.py --video 7  # a specific video (it must not be on YouTube already)

Run by the scheduler at the publishing hour (config.publish_time). It does no rendering and
takes minutes, so the day's publication does not depend on how long production took or
whether it succeeded: whatever is `ready` in the queue goes out, and production keeps the
queue stocked (README, "виробництво окремо від публікації").

Gates and outcomes:
  - config.youtube_publish must be true, otherwise nothing is uploaded and the exit code
    is 0. It is the one switch between "the pipeline works" and "the pipeline posts".
  - an empty queue publishes nothing, warns, and exits 3. A skipped day is better than a
    random video.
  - a failed upload puts the video back to `ready` (it is a good video; the upload is what
    failed) and exits 1. A resumable upload is resumed by the next run, not restarted.
  - exit codes: 0 done or disabled, 1 failed, 2 bad config, 3 empty queue.
"""
import argparse
import sys

from core import db
from core.config import ConfigError, load_config
from core.logger import get_logger
from pipeline import upload

log = get_logger("publish")

LOW_QUEUE = 3                   # fewer ready videos than this is a warning (README)


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Upload a finished video to YouTube")
    parser.add_argument("--video", type=int, help="upload this video instead of the oldest ready one")
    args = parser.parse_args(argv)

    try:
        cfg = load_config()
    except ConfigError as e:
        log.error("config error: %s", e)
        return 2

    if not cfg.get("youtube_publish"):
        log.warning("youtube_publish is false in config.json: nothing was uploaded")
        return 0

    db.init_db()
    video = db.get_video(args.video) if args.video else db.oldest_ready()
    if args.video and not video:
        log.error("there is no video %s", args.video)
        return 1
    if not video:
        log.warning("the publish queue is empty: no video has status ready, so nothing was "
                    "published today")
        return 3

    video_id = video["id"]
    queued = db.count_ready()
    if queued - 1 < LOW_QUEUE:
        log.warning("only %d ready video(s) will be left in the queue after this one",
                    max(queued - 1, 0))

    try:
        db.update_video(video_id, status="uploading")
        db.log_event(video_id, "upload", "info", "stage started")
        upload.run(video_id, cfg)
    except Exception as e:
        log.error("video %s was not uploaded: %s", video_id, e)
        db.log_event(video_id, "upload", "error", str(e))
        db.update_video(video_id, status="ready")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
