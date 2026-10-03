"""The message that goes out after a daily run: what happened, and what to do about it.

    python scripts/notify.py --outcome success --run-url https://github.com/.../runs/123

Prints Markdown. The workflow posts it as a comment on a standing issue that mentions the
repository owner, which is what turns it into an email: GitHub emails a person about a
comment that @-mentions them, with no mail server, no password and no extra secret. That was
the simplest free route that needs nothing beyond the token every workflow already has. Its
limit is that the email comes from GitHub and arrives only if the account's notification
settings allow email, which they do by default.

Success says which video, where it is, and when YouTube will publish it. Failure says which
stage failed and the error text, how many attempts the video has had, and what the next run
will do about it. Both end with the run's own link.
"""
import argparse
import json
import re
import sqlite3
import sys
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import db

MAX_ATTEMPTS = db.MAX_ATTEMPTS


def latest_video():
    """The video the run that just finished was working on: the one most recently touched."""
    with closing(sqlite3.connect(db.DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT v.* FROM videos v LEFT JOIN events e ON e.video_id = v.id "
            "GROUP BY v.id ORDER BY MAX(COALESCE(e.created_at, v.created_at)) DESC, v.id DESC "
            "LIMIT 1").fetchone()
        return dict(row) if row else None


def run_events(video_id):
    with closing(sqlite3.connect(db.DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute(
            "SELECT * FROM events WHERE video_id = ? ORDER BY id", (video_id,))]


def neurons_in(events):
    """Total Neurons the images stage reports for this video, summed across runs."""
    total = 0
    for e in events:
        m = re.search(r"(\d+) neurons over (\d+) Cloudflare calls", e.get("message") or "")
        if m:
            total += int(m.group(1))
    return total


def split_error(error):
    """('render', 'ffmpeg ...') from the 'stage: message' form run.py stores."""
    stage, _, message = (error or "").partition(": ")
    return (stage, message) if message else ("unknown", error or "no error recorded")


def format_when(iso):
    if not iso:
        return "not scheduled"
    try:
        when = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return iso
    return when.astimezone(timezone.utc).strftime("%d.%m.%Y %H:%M UTC")


def build_message(outcome, video, events, run_url, owner="", now=None):
    mention = f"@{owner} " if owner else ""
    stamp = (now or datetime.now(timezone.utc)).strftime("%d.%m.%Y %H:%M UTC")

    if outcome == "not-started":
        # The preflight failed, so no stage ran and the newest video in the database is an
        # old one. Describing it would be a lie; say what actually happened.
        return (f"{mention}**Run did not start** ({stamp})\n\n"
                f"The preflight check failed before any stage ran: a secret is missing, a tool "
                f"is not installed, edge-tts cannot speak from the runner, or the YouTube token "
                f"no longer works. Nothing was spent. The log of the first failing step says "
                f"which: {run_url}\n")

    if video is None:
        return (f"{mention}**Run finished without a video** ({outcome}), {stamp}.\n\n"
                f"Nothing was recorded in the database. Run: {run_url}\n")

    title = video.get("seo_title") or video.get("title") or video.get("topic") or f"video {video['id']}"
    attempts = video.get("attempts") or 1
    neurons = neurons_in(events)
    spent = f"\n- Cloudflare Neurons spent on this video so far: **{neurons}** of 10 000 a day" if neurons else ""

    if outcome == "success" and video.get("youtube_id"):
        url = f"https://youtu.be/{video['youtube_id']}"
        return (f"{mention}**Video published to the queue** ({stamp})\n\n"
                f"**{title}**\n\n"
                f"- Link: {url}\n"
                f"- YouTube makes it public: **{format_when(video.get('published_at'))}**\n"
                f"- Length: {round((video.get('duration_s') or 0) / 60, 1)} min\n"
                f"- Attempts: {attempts}{spent}\n\n"
                f"Run: {run_url}\n")

    stage, message = split_error(video.get("error"))
    errors = [e for e in events if e["level"] == "error"][-3:]
    detail = "\n".join(f"  - `{e['stage']}`: {e['message'][:300]}" for e in errors)
    left = max(MAX_ATTEMPTS - attempts, 0)
    if left:
        plan = (f"The next run carries on with this video from the stage that failed "
                f"({left} attempt(s) left); finished stages and the pictures already paid "
                f"for are not repeated.")
    else:
        plan = ("This video has used all its attempts and will not be retried; the next "
                "run starts a new one.")
    return (f"{mention}**Run FAILED** ({stamp})\n\n"
            f"**{title}** (video {video['id']})\n\n"
            f"- Failed at stage: **{stage}**\n"
            f"- Error: `{message[:500]}`\n"
            f"- Attempt {attempts} of {MAX_ATTEMPTS}{spent}\n\n"
            f"{('Recent errors:' + chr(10) + detail + chr(10) + chr(10)) if detail else ''}"
            f"{plan}\n\nRun: {run_url}\n")


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--outcome", choices=("success", "failure", "not-started"), required=True)
    parser.add_argument("--run-url", default="")
    parser.add_argument("--owner", default="")
    args = parser.parse_args(argv)
    video = latest_video()
    events = run_events(video["id"]) if video else []
    print(build_message(args.outcome, video, events, args.run_url, args.owner))
    return 0


if __name__ == "__main__":
    sys.exit(main())
