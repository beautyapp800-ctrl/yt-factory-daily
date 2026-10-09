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

from core import db, failures

MAX_ATTEMPTS = db.MAX_ATTEMPTS
PREFLIGHT_PROBLEMS = Path(__file__).resolve().parent.parent / "preflight-problems.txt"


def preflight_problems(path=None):
    """What the preflight step found, if it left anything. Empty when it passed or never ran."""
    path = Path(path or PREFLIGHT_PROBLEMS)
    try:
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    except OSError:
        return []


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


def relaxed_line(video):
    """Names the rules a plan was let through without, when it was let through at all.

    The outline stage may accept a plan on its last attempt rather than lose the day's video
    over a blemish. That is a safety valve, and a safety valve nobody can see is the same
    thing as the rule not being there: every use of it is named here, and the weekly report
    counts how often it happens.
    """
    relaxed = (video or {}).get("relaxed")
    if not relaxed:
        return ""
    return (f"- **Went out with a relaxed plan**, after three attempts could not satisfy: "
            f"`{relaxed[:400]}`. Nothing here is a banned word or a copied title - those "
            f"still stop a video. Worth watching: if this starts arriving often, the rule "
            f"and the prompt disagree somewhere and the valve is covering for it.\n")


def thumbnail_line(events):
    """Says so when the thumbnail was made the weak way.

    The thumbnail stage has two quiet fallbacks - cutting the title when no phrase could be
    written, and keeping the largest layout when none passed the feed-size check - and the
    whole point of the new phrase and the new picture is that the first is not what happens.
    A fallback that is only in the database is a fallback nobody sees until the CTR has been
    low for two weeks, so it goes in the run email, in words.
    """
    warnings = [e for e in events if e.get("stage") == "thumbnail" and e.get("level") == "warning"]
    if not warnings:
        return ""
    return (f"- **Thumbnail was made the weak way**: {warnings[-1]['message'][:300]}. "
            f"Worth a look at the tile before it goes public.\n")


def queue_line(now=None):
    """How much the channel has left to show. The one number worth reading in a hurry:
    a failed run matters very differently with six days of queue than with none."""
    now = now or datetime.now(timezone.utc)
    try:
        waiting = db.scheduled_after(now)
    except Exception:                                            # noqa: BLE001
        return ""
    if not waiting:
        return ("- Queue: **empty - the channel has nothing scheduled**, so the next day "
                "without a finished video is a day with no video\n\n")
    last = waiting[-1]["when"]
    days = (last - now).days
    return (f"- Queue: **{len(waiting)} video(s)** still to go out, the last on "
            f"{last.astimezone(timezone.utc):%d.%m.%Y %H:%M UTC} "
            f"(~{days} day(s) of cover)\n\n")


def build_message(outcome, video, events, run_url, owner="", now=None):
    mention = f"@{owner} " if owner else ""
    stamp = (now or datetime.now(timezone.utc)).strftime("%d.%m.%Y %H:%M UTC")

    if outcome == "not-started":
        # The preflight failed, so no stage ran and the newest video in the database is an
        # old one. Describing it would be a lie; say what actually happened.
        problems = preflight_problems()
        if problems:
            listed = "\n".join(f"- {p}" for p in problems)
            actions = "\n".join(f"- {a}" for a in sorted(
                {failures.action_for("preflight", p) for p in problems}))
            return (f"{mention}**Run did not start** ({stamp})\n\n"
                    f"The preflight check stopped it before any stage ran, so nothing was "
                    f"spent. What failed:\n\n{listed}\n\n**What to do**\n\n{actions}\n\n"
                    f"Run: {run_url}\n")
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
                f"- Attempts: {attempts}{spent}\n"
                f"{relaxed_line(video)}{thumbnail_line(events)}\n"
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
    kind, heals, action = failures.classify(stage, message)
    heading = ("**What to do: nothing** - this kind heals by itself."
               if heals else "**What to do**")
    return (f"{mention}**Run FAILED** ({stamp})\n\n"
            f"**{title}** (video {video['id']})\n\n"
            f"- Failed at stage: **{stage}** ({kind})\n"
            f"- Error: `{message[:500]}`\n"
            f"- Attempt {attempts} of {MAX_ATTEMPTS}{spent}\n"
            f"{queue_line()}\n"
            f"{('Recent errors:' + chr(10) + detail + chr(10) + chr(10)) if detail else ''}"
            f"{heading}\n\n{action}\n\n{plan}\n\nRun: {run_url}\n")


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
