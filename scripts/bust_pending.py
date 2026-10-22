"""Set the bust on videos that went up without one, before they are published.

    python scripts/bust_pending.py              # what the workflow runs, every half hour
    python scripts/bust_pending.py --dry-run    # draw, do not send to YouTube

The daily run asks for the bust first, but Cloudflare's allowance does not free up on any
schedule this project has managed to pin down, and when it refuses the video is still made and
uploaded - private, scheduled - with no thumbnail of its own. This keeps asking until the publish
time. A refusal costs nothing, so asking often is free; a drawn bust costs about 330 Neurons.

A video is pending when it is on YouTube, its publish time is still ahead, and it has no bust at
the current layout: either the thumbnail stage logged "bust pending" (or something other than a
bust, as video 28 did), or the upload stage could not set it. It is settled once this script has
set a bust, which data/bust_pending.json records.

A drawn bust is sent only if every check passed and the picture was found acceptable (Gemini's
verdict, or OCR's when Gemini is down). A rejected one is drawn again on the next run, at most
MAX_PAID_TRIES times per video. Every bust that is sent is also kept in assets/busts/
(pipeline/bust_library.py).

Two hours before publication, if no bust of its own has come, the video's phrase is set on a
kept bust instead - the one longest out of use - and sent the same way. Never a frame from the
video, never YouTube's own pick. State is data/bust_pending.json, never the database, so this job
cannot race the daily run over data/factory.db.
"""
import argparse
import json
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import db
from core.config import OUTPUT_DIR, load_config, output_dir
from core.logger import get_logger
from pipeline import bust_library, bust_thumbnail

log = get_logger("bust_pending")

LEDGER = Path(__file__).resolve().parent.parent / "data" / "bust_pending.json"
SAMPLES = OUTPUT_DIR / "_bust_pending"
MAX_PAID_TRIES = 3
WARN_BEFORE = timedelta(hours=3)
# From this close to publication a kept bust is used if no new one has come (owner's choice).
LIBRARY_BEFORE = timedelta(hours=2)


def _when(value):
    try:
        moment = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def load_ledger(path=None):
    path = path or LEDGER
    if not Path(path).exists():
        return {"done": {}, "tries": {}, "log": [], "library_uses": []}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for key, empty in (("done", {}), ("tries", {}), ("log", []), ("library_uses", [])):
        data.setdefault(key, empty)
    return data


def save_ledger(ledger, path=None):
    path = path or LEDGER
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(ledger, indent=2, ensure_ascii=False) + "\n",
                          encoding="utf-8")


def has_bust(video_id):
    """True if the daily run gave this video a bust at the current layout and set it."""
    with db._connect() as conn:
        made = conn.execute(
            "SELECT 1 FROM events WHERE video_id = ? AND stage = 'thumbnail' "
            "AND message LIKE 'bust:%' AND message LIKE ? LIMIT 1",
            (video_id, f"%[layout {bust_thumbnail.LAYOUT}]%")).fetchone()
        unset = conn.execute(
            "SELECT 1 FROM events WHERE video_id = ? AND stage = 'upload' "
            "AND message LIKE 'thumbnail not set%' LIMIT 1", (video_id,)).fetchone()
    return made is not None and unset is None


def pending(ledger, now):
    """Videos still to do, the soonest publish time first."""
    with db._connect() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM videos WHERE youtube_id IS NOT NULL AND youtube_id != ''")]
    out = []
    for r in rows:
        when = _when(r.get("published_at"))
        if when is None or when <= now:
            continue
        if str(r["id"]) in ledger["done"] or has_bust(r["id"]):
            continue
        out.append(r)
    return sorted(out, key=lambda r: _when(r["published_at"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    cfg = load_config()
    ledger = load_ledger()
    now = datetime.now(timezone.utc)
    todo = pending(ledger, now)
    if not todo:
        print("no video is waiting for its bust")
        return 0
    log.info("waiting for a bust: %s", [(r["id"], r["published_at"]) for r in todo])

    one = bust_thumbnail.neurons_per_image(bust_thumbnail.settings(cfg)["steps"]) + 1
    service = None
    report = []
    cloudflare_open = True
    SAMPLES.mkdir(parents=True, exist_ok=True)
    for row in todo:
        vid, when = row["id"], _when(row["published_at"])
        title = row.get("seo_title") or row.get("title") or row["topic"]
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        tries = int(ledger["tries"].get(str(vid), 0))
        details, source = None, "drawn"

        if cloudflare_open and tries < MAX_PAID_TRIES:
            try:
                drawn = bust_thumbnail.make(vid, title, row["topic"],
                                            bust_thumbnail.video_text(vid), output_dir(vid),
                                            cfg, budget=one)
            except RuntimeError as e:
                # Cloudflare said no; it will say the same for the next video this minute.
                report.append(f"- {vid}: not drawn this time ({e})")
                cloudflare_open = False
            else:
                tries += 1
                ledger["tries"][str(vid)] = tries
                chosen = drawn["chosen"]
                shutil.copyfile(drawn["path"],
                                SAMPLES / f"video{vid}_{row['youtube_id']}_try{tries}.jpg")
                good = drawn["passed"] and chosen["vision_ok"]
                ledger["log"].append({"at": stamp, "video_id": vid, "neurons": drawn["neurons"],
                                      "good": good, "summary": bust_thumbnail.summary(drawn)})
                if good:
                    details = drawn
                    bust_library.add(output_dir(vid) / "bust" / chosen["source"], vid,
                                     drawn["philosopher"], chosen["judged_by"], chosen["vision"])
                else:
                    report.append(f"- {vid}: drawn but not sent ({chosen['vision']}; "
                                  f"{'; '.join(chosen['problems']) or 'checks passed'})")
        elif tries >= MAX_PAID_TRIES:
            report.append(f"- {vid}: {tries} busts drawn and none good enough; not drawing more")

        if details is None and when - now <= LIBRARY_BEFORE:
            kept = bust_library.pick(ledger["library_uses"])
            if kept is None:
                report.append(f"- {vid}: no bust of its own and none kept to fall back on")
            else:
                details, source = bust_thumbnail.from_library(
                    vid, title, row["topic"], output_dir(vid), kept), "library"
                shutil.copyfile(details["path"],
                                SAMPLES / f"video{vid}_{row['youtube_id']}_library.jpg")

        if details is None:
            if when - now <= WARN_BEFORE:
                print(f"::warning::video {vid} goes public at {row['published_at']} and still "
                      f"has no bust")
            save_ledger(ledger)
            continue
        line = bust_thumbnail.summary(details)
        if args.dry_run:
            report.append(f"- {vid} (dry run, {source}): {line}")
            save_ledger(ledger)
            continue
        if service is None:
            from core.youtube import build_service, load_credentials
            service = build_service(load_credentials())
        from pipeline.upload import UploadError, set_thumbnail
        try:
            set_thumbnail(service, row["youtube_id"], details["path"])
        except UploadError as e:
            report.append(f"- {vid}: {source} bust ready but thumbnails.set failed ({e})")
            save_ledger(ledger)
            continue
        ledger["done"][str(vid)] = {"at": stamp, "youtube_id": row["youtube_id"],
                                    "source": source, "headline": details["headline"],
                                    "layout": details["layout"],
                                    "publish_at": row["published_at"]}
        if source == "library":
            ledger["library_uses"].append({"file": details["library"], "video_id": vid,
                                           "at": stamp})
            print(f"::notice::video {vid}: no bust could be drawn in time; set a kept one "
                  f"({details['library']})")
        save_ledger(ledger)
        report.append(f"- {vid} (https://youtu.be/{row['youtube_id']}): {source} bust set, {line}")

    save_ledger(ledger)
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
