"""Fail when the channel is about to run quiet.

A channel with nothing scheduled looks exactly like a channel that is working,
right up until the day it posts nothing. Everything upstream can be green: the
runs succeed, the token refreshes, the uploads return 200 - and the queue is
still empty, because the failures that drained it were days ago and nobody was
told.

So this is the one check that is allowed to fail loudly on a healthy-looking
repository: it looks at what YouTube has queued, not at whether the last run
passed.
"""
import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import db  # noqa: E402

ALERT_FILE = Path(__file__).resolve().parent.parent / "publish_alert.md"


def upcoming(now=None):
    now = now or datetime.now(timezone.utc)
    rows = db.scheduled_after(now)
    out = []
    for r in rows:
        when = r.get("published_at") if isinstance(r, dict) else r[-1]
        if isinstance(when, str):
            when = datetime.fromisoformat(when.replace("Z", "+00:00"))
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        title = (r.get("seo_title") or r.get("title")) if isinstance(r, dict) else ""
        vid = r.get("youtube_id") if isinstance(r, dict) else ""
        out.append((when, title or "", vid or ""))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-days", type=float, default=2.0,
                    help="fail when the queue runs out sooner than this")
    ap.add_argument("--run-url", default="")
    args = ap.parse_args()

    now = datetime.now(timezone.utc)
    rows = upcoming(now)

    print("## Scheduled on YouTube\n")
    if not rows:
        print("Nothing is scheduled.\n")
    for when, title, vid in rows:
        days = (when - now).total_seconds() / 86400
        print(f"- {when:%Y-%m-%d %H:%M} UTC  (+{days:.1f}d)  {title[:60]}  `{vid}`")

    last = rows[-1][0] if rows else now
    runway = (last - now).total_seconds() / 86400
    print(f"\n**Videos queued: {len(rows)}. Runway: {runway:.1f} days.**")

    if runway >= args.min_days:
        print(f"\nQueue is healthy (at least {args.min_days:g} days ahead).")
        return 0

    goes_quiet = (last + timedelta(days=1)).date() if rows else now.date()
    message = (
        f"# The channel is about to go quiet\n\n"
        f"Only **{len(rows)}** video(s) are scheduled, the last on "
        f"**{last:%Y-%m-%d}**. That is **{runway:.1f} days** of runway, below the "
        f"{args.min_days:g}-day floor.\n\n"
        f"**First day with nothing to post: {goes_quiet}.**\n\n"
        f"Run the `daily` workflow manually once per missing day, or wait for the "
        f"scheduled runs if there is still time. Each run adds one day to the end "
        f"of the queue.\n"
    )
    if args.run_url:
        message += f"\n[This check]({args.run_url})\n"
    ALERT_FILE.write_text(message, encoding="utf-8")
    print("\n" + message)
    return 3


if __name__ == "__main__":
    sys.exit(main())
