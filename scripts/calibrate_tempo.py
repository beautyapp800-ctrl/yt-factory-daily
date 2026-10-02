"""Measure edge-tts's actual speaking pace at a few rate settings, on real text.

    python scripts/calibrate_tempo.py          # 300 words from the newest video
    python scripts/calibrate_tempo.py 7        # 300 words from a specific video

config.words_per_minute was a guess, not a measurement (145, picked before any audio
existed). The voice is judged by ear; the pace it actually speaks at is not, so this
renders one continuous 300-word excerpt at rate 0%, -5%, -10%, -15% and reports the
real words-per-minute for each, with no pauses inserted - the pure speaking rate.

The voice is whatever config.tts.edge.voice says. The number that goes into
config.words_per_minute is best taken from the real tts stage's own log line on a whole
video (words / seconds spoken), which this excerpt only approximates.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import text as txt, tts
from core.config import OUTPUT_DIR, load_config
from core.logger import get_logger

log = get_logger("calibrate")

VOICE = tts.provider_settings(load_config())[1]["voice"]
# edge-tts requires an explicit sign on the rate string (its own validator rejects a
# bare "0%"), so the neutral rate is written "+0%".
RATES = ["+0%", "-5%", "-10%", "-15%"]
TARGET_WORDS = 300
OUT_DIR = OUTPUT_DIR / "_calibration"


def latest_video_id():
    import sqlite3
    from contextlib import closing
    from core import db
    with closing(sqlite3.connect(db.DB_PATH)) as conn:
        row = conn.execute("SELECT id FROM videos ORDER BY id DESC LIMIT 1").fetchone()
        return row[0] if row else None


def excerpt_for(video_id, target_words=TARGET_WORDS):
    """~target_words of real narration, whole sentences only, from the script's body."""
    script_path = OUTPUT_DIR / str(video_id) / "script.txt"
    if not script_path.exists():
        raise SystemExit(f"No script.txt for video {video_id}; generate one first.")
    body = script_path.read_text(encoding="utf-8").split("\n", 1)[1].strip()
    sentences = txt.split_sentences(body)
    picked, count = [], 0
    for sentence in sentences:
        if picked and count >= target_words:
            break
        picked.append(sentence)
        count += txt.word_count(sentence)
    return " ".join(picked), count


def main():
    video_id = int(sys.argv[1]) if len(sys.argv) > 1 else latest_video_id()
    if video_id is None:
        print("No videos in the database yet.")
        return 1

    text, word_count = excerpt_for(video_id)
    log.info("using %d words from video %d, voice %s", word_count, video_id, VOICE)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for rate in RATES:
        out_path = OUT_DIR / f"rate_{rate.replace('%', 'pct').replace('-', 'neg')}.wav"
        tts.synthesize_edge(text, out_path, VOICE, rate)
        duration = tts.wav_duration(out_path)
        wpm = word_count / duration * 60
        log.info("rate %-5s -> %.1fs for %d words -> %.1f words/min", rate, duration,
                 word_count, wpm)
        rows.append((rate, duration, wpm))

    print(f"\n{word_count} words, voice {VOICE}, no pauses (pure speaking rate)\n")
    print(f"{'rate':>6}  {'duration':>9}  {'words/min':>10}")
    for rate, duration, wpm in rows:
        print(f"{rate:>6}  {duration:>8.1f}s  {wpm:>10.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
