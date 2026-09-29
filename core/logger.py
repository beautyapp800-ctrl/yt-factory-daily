import logging
import sys
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_PREFIX = "factory."
_configured = False


class _StageFilter(logging.Filter):
    def filter(self, record):
        record.stage = record.name[len(_PREFIX):] if record.name.startswith(_PREFIX) else record.name
        return True


def _setup():
    global _configured
    if _configured:
        return
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger("factory")
    root.setLevel(logging.INFO)
    root.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(stage)-10s | %(message)s")
    for handler in (logging.StreamHandler(sys.stdout),
                    logging.FileHandler(LOG_DIR / "factory.log", encoding="utf-8")):
        handler.setFormatter(fmt)
        handler.addFilter(_StageFilter())
        root.addHandler(handler)
    _configured = True


def get_logger(stage="main"):
    _setup()
    return logging.getLogger(_PREFIX + stage)
