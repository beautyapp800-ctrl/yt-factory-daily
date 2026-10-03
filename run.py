import argparse
import sys
from datetime import datetime

from core import db
from core.config import ConfigError, load_config
from core.logger import get_logger
from pipeline import images, render, script, seo, thumbnail, topic, tts, upload

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
    # Upload belongs here now. The queue of finished videos lives on YouTube rather than on
    # disk: each video goes up scheduled for config.publish_days_ahead days ahead, so the
    # separation the README asks for is still there - a run that fails today changes nothing
    # a viewer sees for three days - without carrying 300 MB files between disposable
    # runners. publish.py remains for uploading a video by hand.
    ("upload", upload, "uploading"),
]

MAX_ATTEMPTS = db.MAX_ATTEMPTS


LENGTH_STAGES = ("script", "tts")      # what is thrown away and redone when the length is wrong
LENGTH_DEFAULTS = {"min_minutes": 27, "max_minutes": 33, "attempts": 3}


def length_problem(video_id, cfg):
    """None if the narration is a publishable length, else a sentence saying how it is not.

    Measured on the finished voice track, not predicted from the word count: the pace of the
    same voice varies about +-2.5% between scripts, and the prediction is only as good as the
    pace it is given. The window is wider than the 28-32 minutes the script stage aims for
    (config.length_gate), so this is the net under that aim, not a second copy of it.
    """
    gate = {**LENGTH_DEFAULTS, **(cfg.get("length_gate") or {})}
    minutes = (db.get_video(video_id).get("duration_s") or 0) / 60
    if gate["min_minutes"] <= minutes <= gate["max_minutes"]:
        return None
    side = "short" if minutes < gate["min_minutes"] else "long"
    return (f"the narration is {minutes:.1f} minutes, too {side}: the window is "
            f"{gate['min_minutes']}-{gate['max_minutes']}")


def _discard_narration(video_id):
    """Delete the files of a narration that is being redone, so nothing downstream can
    mistake it for the new one."""
    from core.config import output_dir
    out = output_dir(video_id)
    for name in ("voice.mp3", "timings.json", "narration.wav", "script.txt", "outline.json"):
        (out / name).unlink(missing_ok=True)


def process_video(video_id, cfg):
    """Run all stages for one video. Never raises: failures are recorded as status=failed.

    After the narration is voiced its length is checked; a script outside the window is
    discarded and written again, up to config.length_gate.attempts times, within this run.
    """
    gate = {**LENGTH_DEFAULTS, **(cfg.get("length_gate") or {})}
    redone = 0
    done = db.finished_stages(video_id)
    if done:
        log.info("resuming video %s: %s already finished", video_id,
                 ", ".join(n for n, _, _ in STAGES if n in done))
    index = 0
    while index < len(STAGES):
        name, module, status = STAGES[index]
        if name in done:
            index += 1
            continue
        try:
            db.update_video(video_id, status=status)
            db.log_event(video_id, name, "info", "stage started")
            if not module.run(video_id, cfg):
                raise RuntimeError(f"stage {name} returned False")
            db.log_event(video_id, name, "info", "stage finished")
            done.add(name)

            if name == "tts":
                problem = length_problem(video_id, cfg)
                if problem:
                    redone += 1
                    if redone >= gate["attempts"]:
                        raise RuntimeError(f"{problem}; gave up after {redone} scripts")
                    log.warning("video %s: %s; writing the script again (attempt %d of %d)",
                                video_id, problem, redone + 1, gate["attempts"])
                    db.invalidate_stages(video_id, LENGTH_STAGES, problem)
                    _discard_narration(video_id)
                    done -= set(LENGTH_STAGES)
                    index = next(i for i, s in enumerate(STAGES) if s[0] == LENGTH_STAGES[0])
                    continue
        except Exception as e:
            log.error("video %s failed at stage %s: %s", video_id, name, e)
            try:
                db.log_event(video_id, name, "error", str(e))
                db.update_video(video_id, status="failed", error=f"{name}: {e}")
            except Exception as db_err:
                log.error("could not record failure in DB: %s", db_err)
            return False
        index += 1
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
    existing = db.oldest_unfinished(MAX_ATTEMPTS)
    if existing:
        video_id = existing["id"]
        attempts = (existing.get("attempts") or 0) + 1
        log.info("carrying on with video %s (attempt %d of %d), status %s", video_id,
                 attempts, MAX_ATTEMPTS, existing["status"])
    else:
        placeholder = f"Placeholder topic {datetime.now():%Y-%m-%d %H:%M:%S}"
        video_id = db.create_video(placeholder)
        attempts = 1
        log.info("created video id=%s", video_id)
    db.update_video(video_id, attempts=attempts)

    ok = process_video(video_id, cfg)
    if not ok and attempts >= MAX_ATTEMPTS:
        log.error("video %s has now failed %d times and will not be retried; the next run "
                  "will start a new one", video_id, attempts)
        db.log_event(video_id, "run", "error",
                     f"abandoned after {attempts} attempts")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
