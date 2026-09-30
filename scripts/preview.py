"""Show the shape of a generated video at a glance, without reading 5000 words.

    python scripts/preview.py          # the newest video
    python scripts/preview.py 3        # a specific video id

Prints the title, every lesson title, and the first two sentences of each part, so
repeated openings and a drift into productivity advice are both obvious on sight.
"""
import json
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import db, text as txt
from core.config import OUTPUT_DIR

WIDTH = 96


def latest_video_id():
    with closing(sqlite3.connect(db.DB_PATH)) as conn:
        row = conn.execute("SELECT id FROM videos ORDER BY id DESC LIMIT 1").fetchone()
        return row[0] if row else None


def two_sentences(text):
    return " ".join(txt.split_sentences(text)[:2])


def wrap(text, indent=6):
    """Fold to WIDTH so the console output stays readable."""
    words, lines, line = text.split(), [], ""
    for word in words:
        if len(line) + len(word) + 1 > WIDTH - indent:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        lines.append(line)
    return ("\n" + " " * indent).join(lines)


def parts_from_script(path, lesson_count):
    """Split script.txt back into hook, lessons and outro on its blank lines."""
    body = path.read_text(encoding="utf-8").split("\n", 1)[1].strip()
    blocks = [b.strip() for b in body.split("\n\n") if b.strip()]
    if len(blocks) != lesson_count + 2:
        return None
    return blocks


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    video_id = int(sys.argv[1]) if len(sys.argv) > 1 else latest_video_id()
    if video_id is None:
        print("No videos in the database yet.")
        return 1
    video = db.get_video(video_id)
    if not video:
        print(f"No video with id {video_id}.")
        return 1

    out = OUTPUT_DIR / str(video_id)
    outline_path, script_path = out / "outline.json", out / "script.txt"
    outline = json.loads(outline_path.read_text(encoding="utf-8")) if outline_path.exists() else {}
    lessons = outline.get("lessons", [])

    print("=" * WIDTH)
    print(f"VIDEO {video_id}   status: {video['status']}   seed: {video['seed'] or '-'}")
    print(f"TITLE   {video['title'] or '(none yet)'}")
    print(f"TOPIC   {video['topic']}")
    if outline:
        print(f"SIZE    {outline.get('total_words', '?')} words, "
              f"{outline.get('scene_count', '?')} scenes, "
              f"~{round(outline.get('estimated_duration_s', 0) / 60, 1)} min")
    print("=" * WIDTH)

    if lessons:
        print("\nLESSON TITLES")
        for i, lesson in enumerate(lessons, 1):
            print(f"  {i:2d}. {lesson['title']}")

    if not script_path.exists():
        print("\nNo script.txt yet.")
        return 0

    blocks = parts_from_script(script_path, len(lessons))
    if blocks is None:
        print("\nscript.txt does not split into the expected number of parts.")
        return 0

    names = ["HOOK"] + [f"LESSON {i}" for i in range(1, len(lessons) + 1)] + ["OUTRO"]
    settings = outline.get("settings", {})
    print("\nFIRST TWO SENTENCES OF EACH PART")
    openers = []
    for name, block in zip(names, blocks):
        if name == "HOOK":
            key = "hook"
        elif name.startswith("LESSON"):
            key = str(int(name.split()[1]) - 1)
        else:
            key = None
        where = settings.get(key) if key else None
        suffix = f", set in {where}" if where else ""
        print("")
        print(f"  {name}  ({txt.word_count(block)} words{suffix})")
        print(f"      {wrap(two_sentences(block))}")
        openers.append(txt.opening_words(block, 4))

    clashes = {o for o in openers if openers.count(o) > 1 and o}
    print("\n" + "-" * WIDTH)
    print(f"distinct openings: {len(set(openers))}/{len(openers)}"
          + (f"   CLASHING: {sorted(clashes)}" if clashes else "   no two parts open alike"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
