import argparse
import sys
from datetime import datetime, timezone

import time
from zoneinfo import ZoneInfo
from core import db
from core.concede import concede
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


# Stages whose failure costs quality, not the video. The channel publishes
# every day at 21:00 and that is the only hard requirement, so nothing here
# is allowed to end a run: topic, script, tts, images and render are what a
# video physically needs, and upload is what puts it on the channel.
OPTIONAL_STAGES = ("seo", "thumbnail")

# Days of scheduled video the pipeline tries to hold. Five means four
# consecutive bad days before a viewer notices anything.
DEFAULT_BUFFER_DAYS = 5
# How long a single run may spend building buffer. The GitHub job is capped at
# 300 minutes; this leaves room for the run to finish tidying up.
DEFAULT_TIME_BUDGET_MIN = 240

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
                        # Out of rewrites. The narration exists, so a video can
                        # be assembled; it will simply be longer or shorter than
                        # the window wants, which is a note in the email rather
                        # than a day with nothing on the channel.
                        concede(video_id, "tts", problem,
                                using=f"the narration as recorded after "
                                      f"{redone} rewrites")
                        index += 1
                        continue
                    log.warning("video %s: %s; writing the script again (attempt %d of %d)",
                                video_id, problem, redone + 1, gate["attempts"])
                    db.invalidate_stages(video_id, LENGTH_STAGES, problem)
                    _discard_narration(video_id)
                    done -= set(LENGTH_STAGES)
                    index = next(i for i, s in enumerate(STAGES) if s[0] == LENGTH_STAGES[0])
                    continue
        except Exception as e:
            if name in OPTIONAL_STAGES:
                # A video with a default thumbnail or a plain description is
                # worth publishing. A day with no video is not recoverable.
                concede(video_id, name, f"stage failed: {e}",
                        using="whatever the earlier stages produced")
                done.add(name)
                index += 1
                continue
            log.error("video %s failed at stage %s: %s", video_id, name, e)
            try:
                db.log_event(video_id, name, "error", str(e))
                db.update_video(video_id, status="failed", error=f"{name}: {e}")
            except Exception as db_err:
                log.error("could not record failure in DB: %s", db_err)
            return False
        index += 1
    # "ready" means produced and waiting; a video that reached YouTube is past that. The
    # upload stage sets published, and overwriting it here would have made every finished
    # run look unpublished - caught on the first real run.
    on_youtube = (db.get_video(video_id) or {}).get("youtube_id")
    db.update_video(video_id, status="published" if on_youtube else "ready")
    log.info("video %s finished: %s", video_id, "published" if on_youtube else "ready")
    return True



def queue_depth_days(cfg, now=None):
    """How many days of videos are already scheduled on YouTube, beyond today.

    Zero means the next slot is today's: nothing is in hand. The buffer, not a
    clean run, is what actually protects the 21:00 slot - a day lost to a
    failure then costs buffer instead of leaving a hole on the channel.
    """
    last = db.last_scheduled_publish()
    if not last:
        return 0
    zone = ZoneInfo(cfg.get("timezone") or "Europe/Kiev")
    now = (now or datetime.now(timezone.utc)).astimezone(zone)
    return max(0, (last.astimezone(zone).date() - now.date()).days)


def build_until_buffered(cfg):
    """Keep making videos until the buffer is deep enough, or something stops us.

    Returns (videos_made, stopped_because). One video per run was enough when
    every run worked; it is not enough when one does not, because the next run
    is a day later and the slot has already passed. So a run with room keeps
    going, and the only things that stop it are a full buffer, a spent time
    budget, or a video that could not be assembled at all.
    """
    target = int(cfg.get("buffer_days", DEFAULT_BUFFER_DAYS))
    budget_s = int(cfg.get("run_time_budget_min", DEFAULT_TIME_BUDGET_MIN)) * 60
    started = time.monotonic()
    made = 0

    while True:
        depth = queue_depth_days(cfg)
        log.info("buffer: %d day(s) scheduled ahead, target %d", depth, target)
        if depth >= target:
            return made, f"the buffer is {depth} days deep"
        spent = time.monotonic() - started
        if made and spent > budget_s:
            return made, (f"the {budget_s // 60} minute run budget is spent after "
                          f"{made} video(s); the buffer is {depth} of {target} days")

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
        if ok:
            made += 1
            continue
        if attempts >= MAX_ATTEMPTS:
            log.error("video %s has now failed %d times and will not be retried; the next "
                      "run will start a new one", video_id, attempts)
            db.log_event(video_id, "run", "error", f"abandoned after {attempts} attempts")
        return made, (f"video {video_id} could not be assembled; the buffer is "
                      f"{queue_depth_days(cfg)} of {target} days")


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
    made, because = build_until_buffered(cfg)
    log.info("made %d video(s) this run: %s", made, because)
    # A run that made nothing new but is sitting on a full buffer has done its
    # job. A run that made nothing and has no buffer has not.
    return 0 if (made or queue_depth_days(cfg) > 0) else 1


if __name__ == "__main__":
    sys.exit(main())
