"""How much of Cloudflare's daily 10,000 Neurons is still free to spend.

Cloudflare does not tell this token what is left, so it is worked out from what this
project has recorded spending: the images stage and the thumbnail stage each log a line
with "<n> neurons" in it, and the thumbnail refresh job keeps its own ledger in
data/thumb_refresh.json (it runs outside the video pipeline and does not write the
database).

NOTHING IN THE DAILY RUN DECIDES ON THIS FIGURE ANY MORE. It is an estimate of a rule nobody
has pinned down, and twice now the rule assumed here has been wrong in a way that cost the
channel: first a window turning over at 07:00 UTC, then (20 October) a rolling 24 hours, which
the probe of 19 October seemed to confirm. On 22 October that rolling window said 12,902 had
been spent in the last day - and Cloudflare served the 6,912 Neurons of that morning's frames
without a single refusal, while this estimate said 0 was free and kept the bust from even being
asked for. Video 28 went out with a substitute thumbnail because of it.

So the bust stage asks Cloudflare directly (a refusal costs nothing), the frames are held to a
fixed per-video cap (images.max_neurons), and this module is left for the retired thumbnail
refresh job and for reporting. The window below is still the rolling 24 hours; treat what it
says as a rough guide, never as a reason not to ask.
"""
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from core import db

WINDOW = timedelta(hours=24)
DAILY_FREE_NEURONS = 10000
DEFAULT_RESERVE = 300
LEDGER_PATH = Path(__file__).resolve().parent.parent / "data" / "thumb_refresh.json"

_SPENT = re.compile(r"(\d+(?:\.\d+)?) neurons\b")


def _parse(when):
    try:
        moment = datetime.fromisoformat(str(when).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def load_ledger(path=LEDGER_PATH):
    path = Path(path)
    if not path.exists():
        return {"done": {}, "spend": [], "set_log": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    for key, empty in (("done", {}), ("spend", []), ("set_log", [])):
        data.setdefault(key, empty)
    return data


def window_start(now=None):
    """The moment the current allowance window began - 24 hours ago - as aware UTC."""
    return (now or datetime.now(timezone.utc)).astimezone(timezone.utc) - WINDOW


def window_events(stage_like, now=None):
    """[(created_at, message)] of this window's events whose message has "<n> neurons"."""
    since = window_start(now)
    try:
        with db._connect() as conn:
            rows = conn.execute(
                "SELECT message, created_at FROM events WHERE stage IN ('images', 'bust', 'thumbnail') "
                "AND message LIKE '%neurons%'").fetchall()
    except Exception:                                               # noqa: BLE001
        return []
    return [(_parse(r["created_at"]), r["message"] or "") for r in rows
            if _parse(r["created_at"]) and _parse(r["created_at"]) >= since
            and stage_like in (r["message"] or "")]


def frames_done(now=None):
    """True once this window's narration frames have been drawn: their summary line is the
    only one that says "redrawn for lettering"."""
    return bool(window_events("redrawn for lettering", now))


def spent_today(now=None, ledger_path=LEDGER_PATH):
    """Neurons recorded in the current allowance window (see the module docstring)."""
    since = window_start(now)
    total = 0.0
    try:
        with db._connect() as conn:
            rows = conn.execute(
                "SELECT message, created_at FROM events "
                "WHERE stage IN ('images', 'bust', 'thumbnail') AND message LIKE '%neurons%'").fetchall()
    except Exception:                                               # noqa: BLE001
        rows = []
    for row in rows:
        moment = _parse(row["created_at"])
        if moment is None or moment < since:
            continue
        total += sum(float(n) for n in _SPENT.findall(row["message"] or ""))
    for entry in load_ledger(ledger_path).get("spend", []):
        moment = _parse(entry.get("at"))
        if moment is not None and moment >= since:
            total += float(entry.get("neurons") or 0)
    return total


def available(reserve=DEFAULT_RESERVE, now=None, ledger_path=LEDGER_PATH, frames_reserve=0):
    """Neurons free to spend now. `frames_reserve` is held back for the window's narration
    frames when they have not been drawn yet - they come first, then the new video's cover,
    then everything else."""
    held = 0 if frames_done(now) else frames_reserve
    return max(0.0, DAILY_FREE_NEURONS - reserve - held - spent_today(now, ledger_path))
