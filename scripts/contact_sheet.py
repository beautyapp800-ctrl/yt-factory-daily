"""Build one contact-sheet image from a video's generated scene images, so the whole
set can be judged for consistency at a glance instead of opened one at a time.

    python scripts/contact_sheet.py          # newest video
    python scripts/contact_sheet.py 7        # a specific video

Writes output/<video_id>/contact_sheet.jpg. Each tile is labelled with its scene and
image index in the corner, small enough to stay out of the way.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw, ImageFont

from core import db
from core.config import OUTPUT_DIR
from core.logger import get_logger

log = get_logger("contact_sheet")

TILE_SIZE = 220          # each thumbnail is a TILE_SIZE x TILE_SIZE square
COLUMNS = 10
PADDING = 4
LABEL_HEIGHT = 16


def latest_video_id():
    import sqlite3
    from contextlib import closing
    with closing(sqlite3.connect(db.DB_PATH)) as conn:
        row = conn.execute("SELECT id FROM videos ORDER BY id DESC LIMIT 1").fetchone()
        return row[0] if row else None


def build(video_id, out_path=None, columns=COLUMNS, tile_size=TILE_SIZE):
    rows_data = db.get_video_images(video_id)
    if not rows_data:
        raise SystemExit(f"No images recorded for video {video_id}.")

    rows = (len(rows_data) + columns - 1) // columns
    cell_w = tile_size + PADDING
    cell_h = tile_size + PADDING + LABEL_HEIGHT
    sheet = Image.new("RGB", (columns * cell_w + PADDING, rows * cell_h + PADDING),
                      (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 11)
    except OSError:
        font = ImageFont.load_default()

    scene_of = {r["id"]: r for r in db.get_scenes(video_id)}
    missing = 0
    for i, row in enumerate(rows_data):
        col, line = i % columns, i // columns
        x = PADDING + col * cell_w
        y = PADDING + line * cell_h

        path = Path(row["path"]) if row["path"] else None
        if path and path.exists():
            try:
                thumb = Image.open(path).convert("RGB")
                thumb.thumbnail((tile_size, tile_size))
                offset = ((tile_size - thumb.width) // 2, (tile_size - thumb.height) // 2)
                tile_bg = Image.new("RGB", (tile_size, tile_size), (40, 40, 40))
                tile_bg.paste(thumb, offset)
                sheet.paste(tile_bg, (x, y))
            except Exception as e:
                missing += 1
                log.warning("could not read %s: %s", path, e)
                draw.rectangle([x, y, x + tile_size, y + tile_size], fill=(60, 20, 20))
        else:
            missing += 1
            draw.rectangle([x, y, x + tile_size, y + tile_size], fill=(60, 20, 20))

        scene = scene_of.get(row["scene_id"], {})
        label = f"s{scene.get('idx', '?')}.{row['idx']} {row['provider'] or '?'}"
        draw.text((x + 2, y + tile_size + 1), label, fill=(200, 200, 200), font=font)

    out_path = Path(out_path) if out_path else OUTPUT_DIR / str(video_id) / "contact_sheet.jpg"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=90)
    log.info("%d images (%d missing/unreadable) -> %s (%dx%d)", len(rows_data), missing,
             out_path, sheet.width, sheet.height)
    return out_path


def main():
    video_id = int(sys.argv[1]) if len(sys.argv) > 1 else latest_video_id()
    if video_id is None:
        print("No videos in the database yet.")
        return 1
    out_path = build(video_id)
    print(f"contact sheet: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
