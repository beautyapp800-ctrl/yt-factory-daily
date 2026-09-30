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
    "aspect", "voice", "words_per_minute", "image_style", "tts", "images",
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
    return cfg
