"""Did the videos YouTube was told to publish actually go public?

    python scripts/check_published.py [--grace-minutes 30] [--owner NAME] [--run-url URL]

Every video is uploaded private with status.publishAt set, and YouTube is trusted to flip it
to public at that moment. Trusted, but not checked - until this. Two things could quietly
leave a video private for ever: an API project that has not passed audit is not allowed to
publish, and a schedule that is silently dropped looks exactly like one that worked.

How it checks, and why this way. The upload token carries one scope, youtube.upload, which
cannot read a video's status back - videos.list answers 403 insufficientPermissions. Rather
than widen the scope of a key that can post to the channel, this asks the public web the same
question a viewer would: YouTube's oEmbed endpoint answers 200 with the title for a video
anyone can watch, and 403 for one that is private. No credentials, no API key, no quota.
Measured against both cases before this was written.

Exit code 2 if any video is past its moment and still not public, which turns the workflow
red as well as sending the mail. 0 otherwise.
"""
import argparse
import json
import sqlite3
import sys
import urllib.error
import urllib.request
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import db

OEMBED = "https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={id}&format=json"
CONFIRMED = "publish confirmed"
# A video nobody intends to publish - a test upload, or one withdrawn by hand - would
# otherwise be reported as overdue every evening for ever, and an alert that cries wolf
# nightly is worse than no alert.
IGNORED = "publish not expected"
TIMEOUT_S = 30


def is_public(youtube_id, opener=None):
    """(public?, detail). None when the question could not be answered at all, so a network
    failure is never reported as a video that failed to publish."""
    request = urllib.request.Request(OEMBED.format(id=youtube_id),
                                     headers={"User-Agent": "yt-factory/1.0"})
    try:
        with (opener or urllib.request.urlopen)(request, timeout=TIMEOUT_S) as response:
            title = json.loads(response.read()).get("title", "")
        return True, f"public, titled {title!r}"
    except urllib.error.HTTPError as e:
        if e.code in (401, 403, 404):
            return False, f"not public (oEmbed answered HTTP {e.code})"
        return None, f"could not check: HTTP {e.code}"
    except Exception as e:                                        # noqa: BLE001
        return None, f"could not check: {e}"


def due_videos(now, grace_minutes):
    """Videos whose publish moment has passed by more than the grace period and which have
    not already been confirmed public."""
    cutoff = (now - timedelta(minutes=grace_minutes)).isoformat(timespec="seconds")
    with closing(sqlite3.connect(db.DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row
        settled = {r[0] for r in conn.execute(
            "SELECT video_id FROM events WHERE message IN (?, ?)", (CONFIRMED, IGNORED))}
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM videos WHERE youtube_id IS NOT NULL AND youtube_id != '' "
            "AND published_at IS NOT NULL AND published_at <= ? ORDER BY published_at",
            (cutoff.replace("+00:00", "Z"),))]
    return [r for r in rows if r["id"] not in settled]


def build_report(results, now, owner="", run_url=""):
    late = [r for r in results if r["public"] is False]
    unknown = [r for r in results if r["public"] is None]
    stamp = now.strftime("%d.%m.%Y %H:%M UTC")
    if not late and not unknown:
        return None                                   # nothing to say, so nothing is sent
    mention = f"@{owner} " if owner else ""
    lines = [f"{mention}**A video that should be public is not** ({stamp})", ""]
    for r in late:
        lines.append(f"- **{r['title'][:70]}** - https://youtu.be/{r['youtube_id']}")
        lines.append(f"  - YouTube was asked to publish it at {r['published_at']}")
        lines.append(f"  - {r['detail']}")
    if late:
        lines += ["",
                  "The usual cause is the API project not having passed audit: videos uploaded "
                  "by an unaudited project are held private whatever is asked, and a schedule "
                  "on one is dropped without an error. Until the audit comes through, make it "
                  "public by hand in YouTube Studio: Content, open the video, Visibility.",
                  "",
                  "Until it is public or the row is removed, this check will say so again "
                  "every evening."]
    for r in unknown:
        lines.append(f"- {r['title'][:70]}: {r['detail']} (not treated as a failure)")
    if run_url:
        lines += ["", f"Run: {run_url}"]
    return "\n".join(lines) + "\n"


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--grace-minutes", type=int, default=30)
    parser.add_argument("--owner", default="")
    parser.add_argument("--run-url", default="")
    parser.add_argument("--ignore", type=int, nargs="*", default=[],
                        help="video ids that are not meant to go public, ever")
    args = parser.parse_args(argv)

    now = datetime.now(timezone.utc)
    db.init_db()
    for video_id in args.ignore:
        db.log_event(video_id, "publish", "info", IGNORED)
        print(f"video {video_id}: marked as not expected to go public")
    results = []
    for video in due_videos(now, args.grace_minutes):
        public, detail = is_public(video["youtube_id"])
        results.append({"id": video["id"], "youtube_id": video["youtube_id"],
                        "title": video.get("seo_title") or video.get("title") or "untitled",
                        "published_at": video["published_at"], "public": public, "detail": detail})
        state = {True: "PUBLIC", False: "STILL PRIVATE", None: "UNKNOWN"}[public]
        print(f"video {video['id']} {video['youtube_id']}: {state} - {detail}")
        if public:
            db.log_event(video["id"], "publish", "info", CONFIRMED)
        elif public is False:
            db.log_event(video["id"], "publish", "error",
                         f"still not public {args.grace_minutes} min after {video['published_at']}")

    if not results:
        print("no video is past its publish moment yet")
    report = build_report(results, now, args.owner, args.run_url)
    if report:
        Path("publish_alert.md").write_text(report, encoding="utf-8")
        print("\n" + report)
    return 2 if any(r["public"] is False for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
