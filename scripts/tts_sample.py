"""Render the same real excerpt through every candidate voice, to compare by ear.

    python scripts/tts_sample.py          # excerpt from the newest video's hook
    python scripts/tts_sample.py 7        # excerpt from a specific video

Writes one mp3 per voice to output/_voice_samples/, named after the provider and
voice, so they can be listened to side by side before picking config.tts.

mp3, not wav: piper's native output is wav, but mp3 is far smaller for something
meant to be attached and played back, and every player handles it.
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import db, tts
from core.config import OUTPUT_DIR
from core.logger import get_logger

log = get_logger("tts_sample")

SAMPLE_DIR = OUTPUT_DIR / "_voice_samples"
# Long enough to judge pacing and tone, short enough to render and listen to quickly.
MAX_WORDS = 70


def latest_video_id():
    import sqlite3
    from contextlib import closing
    with closing(sqlite3.connect(db.DB_PATH)) as conn:
        row = conn.execute("SELECT id FROM videos ORDER BY id DESC LIMIT 1").fetchone()
        return row[0] if row else None


def excerpt_for(video_id):
    """The hook, trimmed to MAX_WORDS, from a real generated script."""
    script_path = OUTPUT_DIR / str(video_id) / "script.txt"
    if not script_path.exists():
        raise SystemExit(f"No script.txt for video {video_id}; generate one first.")
    body = script_path.read_text(encoding="utf-8").split("\n", 1)[1].strip()
    hook = body.split("\n\n", 1)[0].strip()
    words = hook.split()
    return " ".join(words[:MAX_WORDS])


def to_mp3(wav_path, mp3_path):
    result = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_path), str(mp3_path)],
        capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed converting {wav_path}: {result.stderr[-300:]}")


def main():
    video_id = int(sys.argv[1]) if len(sys.argv) > 1 else latest_video_id()
    if video_id is None:
        print("No videos in the database yet.")
        return 1

    text = excerpt_for(video_id)
    log.info("using the hook from video %d (%d words)", video_id, len(text.split()))
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    made = []

    def render(label, synth_fn, voice):
        wav_path = SAMPLE_DIR / f"{label}_{voice}.wav"
        try:
            synth_fn(text, wav_path, voice)
        except tts.TTSError as e:
            log.warning("skipping %s voice %s: %s", label, voice, e)
            return
        duration = tts.wav_duration(wav_path)
        mp3_path = wav_path.with_suffix(".mp3")
        to_mp3(wav_path, mp3_path)
        wav_path.unlink(missing_ok=True)
        log.info("%s/%s: %.1fs -> %s", label, voice, duration, mp3_path.name)
        made.append(mp3_path)

    for voice in tts.PIPER_VOICES:
        render("piper", tts.synthesize_piper, voice)
    for voice in tts.EDGE_VOICES:
        render("edge", tts.synthesize_edge, voice)

    print(f"\n{len(made)} samples written to {SAMPLE_DIR}:")
    for p in made:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
