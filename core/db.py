import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "factory.db"

STATUSES = ("pending", "scripting", "tts", "images", "rendering",
            "seo", "thumbnail", "upload", "ready", "uploading", "published", "failed")

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    topic TEXT NOT NULL,
    title TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    duration_s REAL,
    video_path TEXT,
    thumbnail_path TEXT,
    youtube_id TEXT,
    published_at TEXT,
    error TEXT,
    seed TEXT
);
CREATE TABLE IF NOT EXISTS scenes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id INTEGER NOT NULL REFERENCES videos(id),
    idx INTEGER NOT NULL,
    text TEXT,
    image_prompt TEXT,
    image_path TEXT,
    audio_path TEXT,
    duration_s REAL
);
CREATE TABLE IF NOT EXISTS scene_images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id INTEGER NOT NULL REFERENCES videos(id),
    scene_id INTEGER NOT NULL REFERENCES scenes(id),
    idx INTEGER NOT NULL,
    prompt TEXT,
    seed INTEGER,
    provider TEXT,
    path TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id INTEGER NOT NULL REFERENCES videos(id),
    kind TEXT NOT NULL,
    value TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id INTEGER,
    stage TEXT,
    level TEXT,
    message TEXT,
    created_at TEXT NOT NULL
);
"""

_VIDEO_FIELDS = {"topic", "title", "status", "duration_s", "video_path",
                 "thumbnail_path", "youtube_id", "published_at", "error", "seed",
                 "seo_title", "description", "tags"}

# Columns added after the first release, applied to existing databases by _migrate().
# `title` is the script's working title; `seo_title` is the one published on YouTube
# (the seo stage writes it, upload prefers it). `tags` is a JSON list.
_MIGRATIONS = [("videos", "seed", "TEXT"), ("videos", "seo_title", "TEXT"),
               ("videos", "description", "TEXT"), ("videos", "tags", "TEXT")]


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def _connect():
    """Commit on success, roll back on error, and always close (sqlite3's own
    context manager doesn't close, which locks the file on Windows)."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def init_db():
    with _connect() as conn:
        conn.executescript(SCHEMA)
        _migrate(conn)


def _migrate(conn):
    """Add columns that older databases predate. Safe to run on every start."""
    for table, column, decl in _MIGRATIONS:
        existing = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")


def create_video(topic):
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO videos (created_at, topic, status) VALUES (?, ?, 'pending')",
            (_now(), topic))
        return cur.lastrowid


def get_video(video_id):
    with _connect() as conn:
        row = conn.execute("SELECT * FROM videos WHERE id = ?", (video_id,)).fetchone()
        return dict(row) if row else None


def update_video(video_id, **fields):
    if not fields:
        return
    bad = set(fields) - _VIDEO_FIELDS
    if bad:
        raise ValueError(f"Unknown video fields: {', '.join(sorted(bad))}")
    if "status" in fields and fields["status"] not in STATUSES:
        raise ValueError(f"Unknown status: {fields['status']}")
    sets = ", ".join(f"{k} = ?" for k in fields)
    with _connect() as conn:
        conn.execute(f"UPDATE videos SET {sets} WHERE id = ?", (*fields.values(), video_id))


def add_scene(video_id, idx, text=None, image_prompt=None,
              image_path=None, audio_path=None, duration_s=None):
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO scenes (video_id, idx, text, image_prompt, image_path, audio_path, duration_s)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (video_id, idx, text, image_prompt, image_path, audio_path, duration_s))
        return cur.lastrowid


def clear_scenes(video_id):
    """Drop a video's scenes so a regenerated script does not pile up on the old one."""
    with _connect() as conn:
        conn.execute("DELETE FROM scenes WHERE video_id = ?", (video_id,))


_SCENE_FIELDS = {"text", "image_prompt", "image_path", "audio_path", "duration_s"}


def update_scene(scene_id, **fields):
    if not fields:
        return
    bad = set(fields) - _SCENE_FIELDS
    if bad:
        raise ValueError(f"Unknown scene fields: {', '.join(sorted(bad))}")
    sets = ", ".join(f"{k} = ?" for k in fields)
    with _connect() as conn:
        conn.execute(f"UPDATE scenes SET {sets} WHERE id = ?", (*fields.values(), scene_id))


def get_scenes(video_id):
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM scenes WHERE video_id = ? ORDER BY idx", (video_id,))
        return [dict(r) for r in rows]


def clear_scene_images(video_id):
    """Drop a video's images so a regenerated set does not pile up on the old one."""
    with _connect() as conn:
        conn.execute("DELETE FROM scene_images WHERE video_id = ?", (video_id,))


def add_scene_image(video_id, scene_id, idx, prompt=None, seed=None, provider=None, path=None):
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO scene_images (video_id, scene_id, idx, prompt, seed, provider, "
            "path, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (video_id, scene_id, idx, prompt, seed, provider, path, _now()))
        return cur.lastrowid


def get_scene_images(scene_id):
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM scene_images WHERE scene_id = ? ORDER BY idx", (scene_id,))
        return [dict(r) for r in rows]


def get_video_images(video_id):
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM scene_images WHERE video_id = ? ORDER BY scene_id, idx", (video_id,))
        return [dict(r) for r in rows]


def oldest_ready():
    """The oldest finished video that has not been uploaded yet, or None. This is the
    publish queue: production leaves videos at 'ready', publish.py takes them from here."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM videos WHERE status = 'ready' "
            "AND (youtube_id IS NULL OR youtube_id = '') ORDER BY created_at, id LIMIT 1"
        ).fetchone()
        return dict(row) if row else None


def count_ready():
    with _connect() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM videos WHERE status = 'ready' "
            "AND (youtube_id IS NULL OR youtube_id = '')").fetchone()[0]


def log_event(video_id, stage, level, msg):
    with _connect() as conn:
        conn.execute(
            "INSERT INTO events (video_id, stage, level, message, created_at) VALUES (?, ?, ?, ?, ?)",
            (video_id, stage, level, msg, _now()))


def topic_exists(topic):
    with _connect() as conn:
        row = conn.execute("SELECT 1 FROM videos WHERE lower(topic) = lower(?) LIMIT 1",
                           (topic.strip(),)).fetchone()
        return row is not None


def recent_topics(limit=200):
    with _connect() as conn:
        rows = conn.execute("SELECT topic FROM videos ORDER BY id DESC LIMIT ?", (limit,))
        return [r["topic"] for r in rows if r["topic"]]


def seed_counts():
    """How many videos used each seed theme, for picking the least-used one."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT seed, COUNT(*) AS n FROM videos WHERE seed IS NOT NULL GROUP BY seed")
        return {r["seed"]: r["n"] for r in rows}


def add_tics(video_id, pairs):
    """Record the proper names and colour words one video used, for later videos to avoid."""
    rows = [(video_id, kind, value, _now()) for kind, value in dict.fromkeys(pairs)]
    if not rows:
        return
    with _connect() as conn:
        conn.execute("DELETE FROM tics WHERE video_id = ?", (video_id,))
        conn.executemany(
            "INSERT INTO tics (video_id, kind, value, created_at) VALUES (?, ?, ?, ?)", rows)


def recent_tics(videos=10):
    """Names and colours used by the last `videos` videos, as {kind: [value, ...]}."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT DISTINCT kind, value FROM tics WHERE video_id IN "
            "(SELECT id FROM videos ORDER BY id DESC LIMIT ?) ORDER BY kind, value",
            (videos,)).fetchall()
    out = {}
    for row in rows:
        out.setdefault(row["kind"], []).append(row["value"])
    return out
