"""The daily self-check: is the factory still feeding the channel?

    python scripts/heartbeat.py            # prints a report; exit 1 means "email this"

The daily workflow reports on itself, which covers every way a run can go wrong - and none of
the ways a run can fail to happen at all. A cron that GitHub drops, a workflow disabled for
repository inactivity, a schedule that quietly stops after an account change: all of those are
silent, and the first sign would be the channel going quiet days later, once the queue ran out.

This runs on its own schedule and asks the only question that matters from outside: did a new
video join the queue today? If not it says so the same day, with what the last failure was and
what to do about it - not when the queue finally empties.

Checks, in the order they appear in the message:
  - a video was finished in the last `--window` hours;
  - the pipeline is running at all (any event at all in that window);
  - how many videos the channel still has scheduled, and the date it would go quiet;
  - the YouTube token still refreshes, which is the one failure nothing else catches when
    the daily workflow itself is not running.

Exit code 1 means at least one check failed and the workflow should send the message.
"""
import argparse
import sqlite3
import sys
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import db, failures

DEFAULT_WINDOW_H = 24
# Below this many days of scheduled video, the queue is worth mentioning even when today's
# run was fine: it is the warning that one more bad day starts to show.
THIN_QUEUE_DAYS = 2


def finished_since(since):
    with closing(sqlite3.connect(db.DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute(
            "SELECT v.id, v.seo_title, v.title, v.youtube_id, v.published_at "
            "FROM events e JOIN videos v ON v.id = e.video_id "
            "WHERE e.stage = 'upload' AND e.level = 'info' AND e.message = 'stage finished' "
            "AND e.created_at >= ? GROUP BY v.id", (since.strftime("%Y-%m-%dT%H:%M:%S"),))]


def events_since(since):
    with closing(sqlite3.connect(db.DB_PATH)) as conn:
        return conn.execute("SELECT COUNT(*) FROM events WHERE created_at >= ?",
                            (since.strftime("%Y-%m-%dT%H:%M:%S"),)).fetchone()[0]


def last_failure():
    """The most recent recorded failure, as (video_id, stage, message), or None."""
    with closing(sqlite3.connect(db.DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT video_id, stage, message FROM events WHERE level = 'error' "
            "ORDER BY id DESC LIMIT 1").fetchone()
    return (row["video_id"], row["stage"], row["message"]) if row else None


def token_problem():
    """None if the YouTube token still refreshes, else why not. Costs no API quota."""
    try:
        from core import youtube
        youtube.load_credentials()
        return None
    except Exception as e:                                       # noqa: BLE001
        return str(e)


def build_report(now=None, window_h=DEFAULT_WINDOW_H, owner="", run_url="", check_token=True):
    """(message, alert). alert=True means something needs the owner."""
    now = now or datetime.now(timezone.utc)
    since = now - timedelta(hours=window_h)
    lines, alerts = [], []

    produced = finished_since(since)
    if produced:
        for v in produced:
            name = v.get("seo_title") or v.get("title") or f"video {v['id']}"
            lines.append(f"- Added to the queue today: **{name}** "
                         f"(https://youtu.be/{v['youtube_id']})" if v.get("youtube_id")
                         else f"- Added to the queue today: **{name}**")
    else:
        alerts.append("no new video reached the queue")
        if events_since(since) == 0:
            lines.append(f"- **Nothing ran in the last {window_h} hours.** The daily workflow "
                         f"did not start, or stopped before it could write anything down. "
                         f"That is the schedule itself failing, not a stage.")
            alerts.append("the daily workflow did not run")
        else:
            lines.append(f"- **No new video reached the queue in the last {window_h} hours.** "
                         f"The run did start - it failed or is still going.")

    waiting = db.scheduled_after(now)
    if waiting:
        last = waiting[-1]["when"]
        days = (last - now).days
        lines.append(f"- Queue: **{len(waiting)} video(s)** scheduled, the last on "
                     f"{last:%d.%m.%Y %H:%M UTC}. The channel goes quiet after that "
                     f"unless a run finishes first (~{days} day(s) of cover).")
        if days <= THIN_QUEUE_DAYS:
            alerts.append(f"only {days} day(s) of queue left")
    else:
        lines.append("- Queue: **empty**. Nothing is scheduled: the next day without a "
                     "finished video is a day with no video at all.")
        alerts.append("the queue is empty")

    if check_token:
        problem = token_problem()
        if problem:
            lines.append(f"- YouTube token: **does not work** - `{problem[:200]}`")
            alerts.append("the YouTube token no longer works")
        else:
            lines.append("- YouTube token: refreshes normally.")

    actions = []
    recent = last_failure()
    if recent and not produced:
        video_id, stage, message = recent
        kind, heals, action = failures.classify(stage, message)
        lines.append(f"- Last recorded failure: video {video_id}, stage **{stage}** "
                     f"({kind}) - `{(message or '')[:300]}`")
        # The action sentences for self-healing kinds already begin with "Nothing", so a
        # prefix here would say it twice.
        actions.append(action if (heals and action.startswith("Nothing"))
                       else (("Nothing to do - this heals by itself. " if heals else "") + action))
    if check_token and alerts and "the YouTube token no longer works" in alerts:
        actions.append(failures.action_for("preflight", "the YouTube token does not work"))

    mention = f"@{owner} " if owner else ""
    stamp = now.strftime("%d.%m.%Y %H:%M UTC")
    if not alerts:
        head = f"{mention}**Daily check: the factory is feeding the channel** ({stamp})\n\n"
    else:
        head = (f"{mention}**Daily check: {'; '.join(alerts)}** ({stamp})\n\n")
    body = "\n".join(lines)
    tail = ""
    if actions:
        tail = "\n\n**What to do**\n\n" + "\n".join(f"- {a}" for a in dict.fromkeys(actions))
    if run_url:
        tail += f"\n\nThis check: {run_url}\n"
    return head + body + tail + "\n", bool(alerts)


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-hours", type=int, default=DEFAULT_WINDOW_H)
    parser.add_argument("--owner", default="")
    parser.add_argument("--run-url", default="")
    parser.add_argument("--no-token-check", action="store_true",
                        help="skip the YouTube token check (for offline tests)")
    args = parser.parse_args(argv)
    db.init_db()
    message, alert = build_report(window_h=args.window_hours, owner=args.owner,
                                  run_url=args.run_url, check_token=not args.no_token_check)
    print(message)
    return 1 if alert else 0


if __name__ == "__main__":
    sys.exit(main())
