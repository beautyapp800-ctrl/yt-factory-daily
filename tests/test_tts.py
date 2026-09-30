"""Offline tests for the tts stage. Run: python tests/test_tts.py

No audio is synthesized: core.tts.synthesize and concatenate are stubbed, so this
checks the pipeline's own logic (pace measurement, db writes, provider selection)
without needing piper's models or a network call to edge-tts.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Keep test output out of the working logs/factory.log.
import core.logger
core.logger.LOG_DIR = Path(tempfile.mkdtemp(prefix="yt-factory-test-"))

from core import config, db, tts
import pipeline.tts as stage


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok: {msg}")


def test_provider_settings():
    print("test_provider_settings")
    provider, settings = tts.provider_settings({})
    check(provider == "piper", "provider defaults to piper when tts block is absent")
    check(settings["voice"] == tts.DEFAULT_PIPER_VOICE, "default piper voice is used")

    provider, settings = tts.provider_settings(
        {"tts": {"provider": "edge", "edge": {"voice": "en-GB-RyanNeural"}}})
    check(provider == "edge", "provider switches to edge")
    check(settings["voice"] == "en-GB-RyanNeural", "edge voice override is read")
    check(settings["rate"] == tts.DEFAULT_EDGE_RATE, "edge rate falls back to the default")

    try:
        tts.provider_settings({"tts": {"provider": "nonsense"}})
        check(False, "an unknown provider should raise")
    except tts.TTSError:
        check(True, "an unknown provider raises TTSError")


def test_pace_warning(tmp):
    print("test_pace_warning")
    db.DB_PATH = tmp / "pace.db"
    config.OUTPUT_DIR = tmp / "output"
    db.init_db()
    vid = db.create_video("pace test")

    # Ten scenes of 100 words each, forced to render at a fixed, too-fast pace.
    for i in range(1, 11):
        db.add_scene(vid, i, text=" ".join(["word"] * 100), image_prompt="x")

    original_synth, original_concat = tts.synthesize, tts.concatenate
    tts.synthesize = lambda text, out_path, cfg: 20.0       # 100 words / 20s = 300 wpm
    tts.concatenate = lambda paths, out: 200.0              # 10 * 20s
    stage.tts.synthesize, stage.tts.concatenate = tts.synthesize, tts.concatenate
    try:
        cfg = {"words_per_minute": 145, "tts": {"provider": "piper"}}
        check(stage.run(vid, cfg) is True, "the stage completes")
    finally:
        tts.synthesize, tts.concatenate = original_synth, original_concat

    video = db.get_video(vid)
    check(video["duration_s"] == 200.0, "the video's duration_s is the concatenated length")
    events = [e for e in _events(vid) if e["stage"] == "tts"]
    check(any(e["level"] == "warning" for e in events),
          "a 300 wpm actual pace against a 145 wpm target logs a warning")
    scenes = db.get_scenes(vid)
    check(all(s["audio_path"] and s["duration_s"] == 20.0 for s in scenes),
          "every scene got its own audio_path and duration_s")


def test_pace_within_tolerance(tmp):
    print("test_pace_within_tolerance")
    db.DB_PATH = tmp / "pace2.db"
    db.init_db()
    vid = db.create_video("pace test 2")
    for i in range(1, 4):
        db.add_scene(vid, i, text=" ".join(["word"] * 145), image_prompt="x")

    original_synth, original_concat = tts.synthesize, tts.concatenate
    tts.synthesize = lambda text, out_path, cfg: 60.0        # 145 words / 60s = 145 wpm
    tts.concatenate = lambda paths, out: 180.0
    stage.tts.synthesize, stage.tts.concatenate = tts.synthesize, tts.concatenate
    try:
        cfg = {"words_per_minute": 145, "tts": {"provider": "piper"}}
        stage.run(vid, cfg)
    finally:
        tts.synthesize, tts.concatenate = original_synth, original_concat

    events = [e for e in _events(vid) if e["stage"] == "tts"]
    check(not any(e["level"] == "warning" for e in events),
          "a pace matching the target logs no warning")


def _events(video_id):
    import sqlite3
    from contextlib import closing
    with closing(sqlite3.connect(db.DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute(
            "SELECT * FROM events WHERE video_id = ?", (video_id,))]


def test_403_message():
    print("test_403_message")
    check(tts._is_403(Exception("403, message='Forbidden'")), "a 403 string is recognised")
    check(tts._is_403(Exception("NoAudioReceived: no audio was received")),
          "edge-tts's own empty-stream error is treated the same as a 403")
    check(not tts._is_403(Exception("connection reset")),
          "an unrelated network error is not mistaken for a 403")


def test_no_scenes_raises(tmp):
    print("test_no_scenes_raises")
    db.DB_PATH = tmp / "empty.db"
    db.init_db()
    vid = db.create_video("no scenes")
    try:
        stage.run(vid, {"words_per_minute": 145, "tts": {"provider": "piper"}})
        check(False, "running tts with zero scenes should raise")
    except RuntimeError as e:
        check("no scenes" in str(e), "the error names the actual problem")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        test_provider_settings()
        test_pace_warning(tmp)
        test_pace_within_tolerance(tmp)
        test_403_message()
        test_no_scenes_raises(tmp)
    print("ALL TTS TESTS PASSED")
