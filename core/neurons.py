"""How much of Cloudflare's daily 10,000 Neurons is still free to spend.

Cloudflare does not tell this token what is left, so it is worked out from what this
project has recorded spending: the images stage and the thumbnail stage each log a line
with "<n> neurons" in it, and the thumbnail refresh job keeps its own ledger in
data/thumb_refresh.json (it runs outside the video pipeline and does not write the
database).

The window is the UTC day, Cloudflare's documented reset. The README records one probe that
seemed to contradict it, but the runs since do not: on 17 October all 63 narration frames
were drawn at 05:47 UTC although 8,256 Neurons had been spent at 06:06 the day before, which
a trailing 24-hour window would have refused.

The narration frames come first. The thumbnail stage runs after them, so whatever it sees
here is what they left; the refresh job runs only after a successful daily run and keeps
the same reserve.
"""
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from core import db

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


def spent_today(now=None, ledger_path=LEDGER_PATH):
    """Neurons recorded since 00:00 UTC."""
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    since = now.replace(hour=0, minute=0, second=0, microsecond=0)
    total = 0.0
    try:
        with db._connect() as conn:
            rows = conn.execute(
                "SELECT message, created_at FROM events "
                "WHERE stage IN ('images', 'thumbnail') AND message LIKE '%neurons%'").fetchall()
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


def available(reserve=DEFAULT_RESERVE, now=None, ledger_path=LEDGER_PATH):
    return max(0.0, DAILY_FREE_NEURONS - reserve - spent_today(now, ledger_path))
