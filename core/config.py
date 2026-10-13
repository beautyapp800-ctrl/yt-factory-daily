import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.json"
OUTPUT_DIR = ROOT / "output"
ASSETS_DIR = ROOT / "assets"


def output_dir(video_id):
    """Per-video output folder, created on demand."""
    path = OUTPUT_DIR / str(video_id)
    path.mkdir(parents=True, exist_ok=True)
    return path

REQUIRED_FIELDS = [
    "channel_topic", "language", "target_duration_min", "videos_per_week",
    "publish_time", "timezone", "youtube_publish", "privacy_status",
    "aspect", "voice", "words_per_minute", "image_style", "tts", "images", "render",
]


class ConfigError(Exception):
    pass


def load_config(path=CONFIG_PATH):
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ConfigError(f"Invalid JSON in {path}: {e}") from e
    missing = [k for k in REQUIRED_FIELDS if k not in cfg]
    if missing:
        raise ConfigError(f"Missing required config fields: {', '.join(missing)}")
    cfg["words_per_minute"] = speaking_wpm(cfg)
    return cfg


def speaking_wpm(cfg):
    """The speaking rate of the voice that is actually going to read the script.

    The word budget is derived from this number, so the two have to describe
    the same voice. They did not: 166 wpm was measured on edge-tts, the
    provider was switched to piper, and piper at length_scale 1.08 speaks
    217 wpm. A 4783-word script - a normal one by the budget - came out at
    23.7 minutes against a 27-33 minute window, the pipeline called the script
    the fault and rewrote it shorter, and that rewrite then failed the budget
    gate. Two whole days of publication were lost to a number that belonged to
    a voice no longer in use.

    So the rate lives in the provider's own block and is read from the active
    one. A provider with no measured rate is an error rather than a silent
    borrow from whichever voice was measured last.
    """
    tts_cfg = cfg.get("tts") or {}
    provider = tts_cfg.get("provider")
    block = tts_cfg.get(provider) or {}
    measured = block.get("measured_wpm")
    if measured:
        return int(measured)
    raise ConfigError(
        f"tts.{provider}.measured_wpm is not set, so the word budget would be "
        f"derived from a rate measured on a different voice. Measure it with "
        f"scripts/calibrate_tempo.py and write it into config.json.")
