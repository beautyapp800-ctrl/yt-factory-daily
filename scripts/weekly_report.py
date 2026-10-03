"""The weekly health report: how many videos are queued ahead, how many runs failed, and what
is left of the Cloudflare allowance.

    python scripts/weekly_report.py [--run-url URL] [--owner NAME]

Prints Markdown, posted by the workflow the same way as the daily message. The exit code is
2 when no video is queued ahead - the alarm level in the README - so the run turns red as
well as sending the email, 0 otherwise.

What each number is, and what it is not:

Queued ahead is every video with a YouTube id whose publishAt is still in the future. It is
read from the database, which is what this system scheduled; it does not ask YouTube, so a
video someone deleted in Studio is still counted. That is the honest limit of reading our
own records, and it is stated in the report.

Failures are runs recorded as an error in the events table in the last 7 days.

Cloudflare has no endpoint that returns the remaining allowance to the token this project
uses (that needs Account Analytics read permission, which the token deliberately does not
have). So the report shows two things instead: the Neurons our own runs logged in the last
24 hours, which is a lower bound on what is used, and a live probe - one real image call -
saying whether the allowance is open right now. The probe costs 96 Neurons.
"""
import argparse
import re
import sqlite3
import sys
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import db

QUEUE_WARN, QUEUE_ALARM = 3, 1
DAILY_FREE_NEURONS = 10000


def _rows(sql, params=()):
    with closing(sqlite3.connect(db.DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute(sql, params)]


def _parse(iso):
    try:
        return datetime.fromisoformat((iso or "").replace("Z", "+00:00"))
    except ValueError:
        return None


def queued_ahead(now):
    """[(publish moment, title)] for videos on YouTube that are not public yet, soonest first."""
    out = []
    for v in _rows("SELECT * FROM videos WHERE youtube_id IS NOT NULL AND youtube_id != ''"):
        when = _parse(v.get("published_at"))
        if when and when > now:
            out.append((when, v.get("seo_title") or v.get("title") or f"video {v['id']}"))
    return sorted(out)


def failures_since(since):
    """Failed runs in the window, one per video and stage, with the error text."""
    cutoff = since.isoformat(timespec="seconds")
    seen, out = set(), []
    for e in _rows("SELECT * FROM events WHERE level = 'error' AND created_at >= ? ORDER BY id",
                   (cutoff,)):
        key = (e["video_id"], e["stage"])
        if key not in seen:
            seen.add(key)
            out.append(e)
    return out


def neurons_logged_since(since):
    cutoff = since.isoformat(timespec="seconds")
    total = 0
    for e in _rows("SELECT message FROM events WHERE stage = 'images' AND created_at >= ?",
                   (cutoff,)):
        m = re.search(r"(\d+) neurons over \d+ Cloudflare calls", e["message"] or "")
        if m:
            total += int(m.group(1))
    return total


def cloudflare_probe():
    """(open?, one-line description). Never raises: a monitor that crashes tells nothing."""
    try:
        from core.llm import get_key
        if not (get_key("CLOUDFLARE_ACCOUNT_ID") and get_key("CLOUDFLARE_API_TOKEN")):
            return None, "probe skipped: Cloudflare credentials are not set in this workflow"
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import cf_window_probe
        record = cf_window_probe.probe()
    except Exception as e:                                    # noqa: BLE001
        return None, f"probe failed to run: {e}"
    if record["success"]:
        return True, f"open (HTTP {record['http']}, {record['neurons']} Neurons for the probe)"
    return False, (f"BLOCKED (HTTP {record['http']}, code {record['error_codes']}): "
                   f"{(record['error_message'] or '')[:160]}")


def build_report(now, owner="", run_url="", probe=(None, "not run")):
    queue = queued_ahead(now)
    week_ago = now - timedelta(days=7)
    published = [v for v in _rows("SELECT * FROM videos WHERE youtube_id IS NOT NULL "
                                  "AND youtube_id != '' AND created_at >= ?",
                                  (week_ago.isoformat(timespec="seconds"),))]
    failed = failures_since(week_ago)
    spent = neurons_logged_since(now - timedelta(days=1))

    if len(queue) < QUEUE_ALARM:
        level = "ALARM: no video is queued ahead"
    elif len(queue) < QUEUE_WARN:
        level = f"warning: only {len(queue)} video(s) queued ahead"
    else:
        level = "healthy"

    lines = [f"{'@' + owner + ' ' if owner else ''}**Weekly report: {level}** "
             f"({now.strftime('%d.%m.%Y %H:%M UTC')})", ""]
    lines.append(f"**Queued ahead on YouTube: {len(queue)}**"
                 f" (warning below {QUEUE_WARN}, alarm below {QUEUE_ALARM})")
    for when, title in queue[:10]:
        lines.append(f"- {when.strftime('%d.%m %H:%M UTC')} - {title[:70]}")
    if not queue:
        lines.append("- nothing is scheduled")
    lines.append("- counted from this system's own records, not asked of YouTube: a video "
                 "deleted in Studio would still be counted here")
    lines += ["", f"**Uploaded in the last 7 days: {len(published)}**   "
              f"**Failed runs in the last 7 days: {len(failed)}**"]
    for e in failed[:8]:
        lines.append(f"- video {e['video_id']}, stage `{e['stage']}`: {(e['message'] or '')[:200]}")

    lines += ["", "**Cloudflare**",
              f"- Neurons our own runs logged in the last 24 h: **{spent}** of "
              f"{DAILY_FREE_NEURONS} (a lower bound; probes and retries that failed are not in it)",
              f"- Live probe: {probe[1]}",
              "- the exact remaining allowance is not available to this token; reading it "
              "would need Account Analytics permission"]
    if run_url:
        lines += ["", f"Run: {run_url}"]
    return "\n".join(lines) + "\n", len(queue)


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-url", default="")
    parser.add_argument("--owner", default="")
    parser.add_argument("--no-probe", action="store_true")
    args = parser.parse_args(argv)
    now = datetime.now(timezone.utc)
    probe = (None, "not run") if args.no_probe else cloudflare_probe()
    report, queued = build_report(now, args.owner, args.run_url, probe)
    print(report)
    return 2 if queued < QUEUE_ALARM else 0


if __name__ == "__main__":
    sys.exit(main())
