"""Give videos already on YouTube the bust thumbnail (pipeline/bust_thumbnail.py).

    python scripts/refresh_thumbnails.py               # what the workflow runs
    python scripts/refresh_thumbnails.py --dry-run     # build, do not send to YouTube
    python scripts/refresh_thumbnails.py --only 24     # one video
    python scripts/refresh_thumbnails.py --restore 22=path/to/thumbnail.jpg   # put one back

Order, and it is the Neuron priority too: the narration frames of the next video come first
(a reserve is held back for them until they are drawn), then the cover of a video that has no
bust yet - the daily run could not afford one - earliest publish time first, then a redo of a
bust made to an older layout, then videos already public, at most --old-per-day (2) a day. A
video whose thumbnail the pipeline already made in the bust style is left alone, and so is
one this script has done at the current layout.

When the daily run could not draw a bust, this is what puts it right: a cron entry runs it
just after the allowance resets (see core/neurons.py), well before the video's publish time.

Budgets. Neurons: whatever core.neurons says the UTC day has left, after the reserve - the
workflow only runs once a daily run has succeeded, so the narration frames have had theirs.
YouTube: thumbnails.set is about 50 of 10,000 units; a quota error stops the run and the rest
waits for tomorrow.

Only a bust Gemini has looked at is sent. The pipeline may use one picked by measurement,
because a new video must not wait; a video that already has a thumbnail can. That rule
exists because the first run, with Gemini answering 503, sent two pictures with another
service's logo on them.

State is data/thumb_refresh.json, not the database, so this job never races the daily run
over data/factory.db. It records what was done, what was spent (core.neurons reads it), and
copies of every thumbnail made go to output/_refresh/ for the workflow's artifact.
"""
import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import db, neurons
from core.config import OUTPUT_DIR, load_config, output_dir
from core.logger import get_logger
from pipeline import bust_thumbnail

log = get_logger("refresh")

SAMPLES = OUTPUT_DIR / "_refresh"
# Bumped when the look changes so that thumbnails made to an older one are done again. 2: the
# bottom-right corner is kept clear of YouTube's duration stamp.
LAYOUT = 2


def _when(value):
    try:
        moment = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def made_by_pipeline(video_id):
    with db._connect() as conn:
        row = conn.execute("SELECT 1 FROM events WHERE video_id = ? AND stage = 'thumbnail' "
                           "AND message LIKE 'bust:%' LIMIT 1", (video_id,)).fetchone()
    return row is not None


def plan(ledger, now, old_per_day, only=None):
    """(urgent, old): video rows to do now, in the order they should be done.

    urgent = no bust at all yet, then bust at an older layout, each by publish time; both are
    still waiting for their publish time or at least recent. old = public ones, newest first.
    """
    with db._connect() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM videos WHERE youtube_id IS NOT NULL AND youtube_id != '' "
            "ORDER BY id DESC")]
    done = ledger["done"]
    todo = []
    for r in rows:
        entry = done.get(str(r["id"]))
        if entry is not None and entry.get("layout", 1) >= LAYOUT:
            continue
        if entry is None and made_by_pipeline(r["id"]):
            continue
        r["_redo"] = entry is not None
        todo.append(r)
    if only is not None:
        todo = [r for r in todo if r["id"] == only]
    scheduled = [r for r in todo if (_when(r["published_at"]) or now) > now]
    old = [r for r in todo if r not in scheduled]
    today = neurons.window_start(now)
    old_today = sum(1 for e in ledger["set_log"]
                    if e.get("kind") == "old" and (_when(e.get("at")) or now) >= today)
    room = max(0, old_per_day - old_today) if only is None else len(old)
    urgent = sorted(scheduled, key=lambda r: (r["_redo"], _when(r["published_at"])))
    return urgent, old[:room]


def save(ledger, path=neurons.LEDGER_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ledger, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def restore(pairs):
    from core.youtube import build_service, load_credentials
    from pipeline.upload import set_thumbnail
    service = build_service(load_credentials())
    ledger = neurons.load_ledger()
    for pair in pairs:
        vid, path = pair.split("=", 1)
        video = db.get_video(int(vid))
        set_thumbnail(service, video["youtube_id"], path)
        ledger["done"].pop(str(vid), None)
        ledger["set_log"] = [e for e in ledger["set_log"] if str(e.get("video_id")) != str(vid)]
        save(ledger)
        print(f"- {vid}: thumbnail restored from {path}")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", type=int)
    parser.add_argument("--old-per-day", type=int, default=2)
    parser.add_argument("--restore", action="append", default=[],
                        help="VIDEO_ID=PATH: set this file as the video's thumbnail and "
                             "forget that it was refreshed, so it is done again later")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if args.restore:
        return restore(args.restore)

    cfg = load_config()
    ledger = neurons.load_ledger()
    now = datetime.now(timezone.utc)
    scheduled, old = plan(ledger, now, args.old_per_day, args.only)
    log.info("to do now: %d scheduled (%s), %d already public (%s)", len(scheduled),
             [r["id"] for r in scheduled], len(old), [r["id"] for r in old])
    if not scheduled and not old:
        print("nothing to refresh")
        return 0

    service = None
    if not args.dry_run:
        from core.youtube import build_service, load_credentials
        service = build_service(load_credentials())
    from pipeline.upload import UploadError, set_thumbnail

    SAMPLES.mkdir(parents=True, exist_ok=True)
    conf = bust_thumbnail.settings(cfg)
    report = []
    queue = [("scheduled", r) for r in scheduled] + [("old", r) for r in old]
    for n, (kind, row) in enumerate(queue):
        vid = row["id"]
        free = neurons.available(conf["reserve_neurons"],
                                 frames_reserve=conf["frames_reserve_neurons"])
        # Each video gets an equal share of what there is, but at least one candidate if one
        # fits: one picture each beats three for the first and none for the rest.
        one = bust_thumbnail.neurons_per_image(conf["steps"])
        left = len(queue) - n
        budget = max(free / left, one) if free >= one else free
        budget = min(budget, free)
        log.info("video %s: %.0f Neurons free now, %.0f for this one", vid, free, budget)
        title = row.get("seo_title") or row.get("title") or row["topic"]
        try:
            details = bust_thumbnail.make(vid, title, row["topic"],
                                          bust_thumbnail.video_text(vid), output_dir(vid),
                                          cfg, budget=budget)
        except Exception as e:                                       # noqa: BLE001
            log.warning("video %s: no bust thumbnail (%s); stopping for today", vid, e)
            report.append(f"- {vid}: not made ({e})")
            break
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if details["neurons"]:
            ledger["spend"].append({"at": stamp, "video_id": vid,
                                    "neurons": details["neurons"]})
        copy = SAMPLES / f"{len(report) + 1:02d}_video{vid}_{row['youtube_id']}.jpg"
        shutil.copyfile(details["path"], copy)
        line = bust_thumbnail.summary(details)
        if details["chosen"]["judged_by"] != "gemini" or not details["passed"]:
            # A thumbnail already on YouTube is replaced only by one that is both judged by
            # Gemini and passes every check; otherwise it waits for the next window.
            report.append(f"- {vid}: built but NOT sent ({'not judged by Gemini' if details['chosen']['judged_by'] != 'gemini' else 'failed: ' + '; '.join(details['chosen']['problems'] or [details['chosen']['vision']])}): {line}")
            save(ledger)
            continue
        if args.dry_run:
            report.append(f"- {vid} ({kind}, dry run): {line}")
            save(ledger)
            continue
        try:
            set_thumbnail(service, row["youtube_id"], details["path"])
        except UploadError as e:
            log.warning("video %s: thumbnails.set failed: %s", vid, e)
            report.append(f"- {vid}: built but not set ({e})")
            save(ledger)
            if "quota" in str(e).lower():
                break
            continue
        ledger["done"][str(vid)] = {"at": stamp, "youtube_id": row["youtube_id"],
                                    "kind": kind, "layout": LAYOUT,
                                    "headline": details["headline"],
                                    "attribution": details["attribution"],
                                    "passed": details["passed"],
                                    "neurons": details["neurons"]}
        ledger["set_log"].append({"at": stamp, "video_id": vid, "kind": kind})
        save(ledger)
        report.append(f"- {vid} ({kind}, https://youtu.be/{row['youtube_id']}): {line}")
        log.info("video %s: thumbnail set", vid)

    print("\n".join(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
