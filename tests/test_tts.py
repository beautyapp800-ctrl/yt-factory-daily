"""Offline tests for the tts stage. Run: python tests/test_tts.py

No audio is synthesized and ffmpeg is never called: core.tts's synthesize, silence,
concatenate, master, measure_volume and extract_sample are stubbed, so this checks
the pipeline's own logic (sentence-level pause placement, lesson-boundary detection,
pace measurement, db writes) without needing piper's models, a network call, or
ffmpeg on PATH.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Keep test output out of the working logs/factory.log.
import core.logger
core.logger.LOG_DIR = Path(tempfile.mkdtemp(prefix="yt-factory-test-"))

from core import config, db, text as txt, tts
import pipeline.tts as stage


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok: {msg}")


# --- stubs -------------------------------------------------------------------

WORDS_PER_SECOND = 145 / 60          # matches the 145 wpm config target exactly


class FakeAudio:
    """Replaces every ffmpeg/piper call in core.tts with deterministic arithmetic, so
    pause placement and pace can be checked exactly instead of approximately."""

    def __init__(self):
        self.silences = []
        self.concats = []

    def synthesize(self, text, out_path, cfg):
        Path(out_path).write_bytes(b"\x00")
        return max(0.05, txt.word_count(text) / WORDS_PER_SECOND)

    def silence(self, duration_s, out_path):
        Path(out_path).write_bytes(b"\x00")
        self.silences.append(duration_s)
        return out_path

    def concatenate(self, paths, out_path):
        Path(out_path).write_bytes(b"\x00")
        # Each stub file "is" a known duration via a side table keyed by path identity
        # is not available post-hoc, so duration is reconstructed by the caller summing
        # what it already knows; concatenate here only needs to report SOMETHING usable
        # as the final duration when called on the full track, which the test derives
        # independently and compares against.
        self.concats.append((tuple(str(p) for p in paths), str(out_path)))
        return sum(self._path_durations.get(str(p), 0.0) for p in paths)

    _path_durations = {}

    def master(self, in_wav, out_mp3, cfg):
        Path(out_mp3).write_bytes(b"\x00" * 1000)
        return Path(out_mp3)

    def measure_volume(self, path):
        return -20.0, -1.0                    # comfortably above the silence threshold

    def extract_sample(self, in_audio, out_path, seconds):
        Path(out_path).write_bytes(b"\x00" * 100)
        return Path(out_path)


def _install_fake(monkey):
    for name in ("synthesize", "silence", "concatenate", "master", "measure_volume",
                 "extract_sample"):
        setattr(tts, name, getattr(monkey, name))
        setattr(stage.tts, name, getattr(monkey, name))


def _restore(original):
    for name, fn in original.items():
        setattr(tts, name, fn)
        setattr(stage.tts, name, fn)


def _setup_video(tmp, lessons=3, sentences_per_lesson=2):
    """A minimal video with a real script.txt/outline.json/scenes triple, small enough
    to reason about by hand: a 1-sentence hook, `lessons` lessons of
    `sentences_per_lesson` sentences each, and a 1-sentence outro. Each scene is built
    to hold exactly one sentence, so scene boundaries and part boundaries coincide
    except where the test deliberately misaligns them.
    """
    db.DB_PATH = tmp / f"{lessons}.db"
    config.OUTPUT_DIR = tmp / f"out_{lessons}"
    db.init_db()
    vid = db.create_video("tts test")

    hook = "This is the hook sentence here."
    lesson_blocks = []
    for i in range(lessons):
        sentences = [f"Lesson {i} sentence {j} goes here now." for j in range(sentences_per_lesson)]
        lesson_blocks.append(" ".join(sentences))
    outro = "This is the outro sentence here."

    parts = [hook] + lesson_blocks + [outro]
    full_text = "\n\n".join(parts)
    out = config.output_dir(vid)
    (out / "script.txt").write_text(f"Title\n\n{full_text}\n", encoding="utf-8")
    (out / "outline.json").write_text(
        json.dumps({"lessons": [{"title": f"L{i}"} for i in range(lessons)]}), encoding="utf-8")

    idx = 1
    for block in parts:
        for sentence in txt.split_sentences(block):
            db.add_scene(vid, idx, text=sentence, image_prompt="x")
            idx += 1
    return vid


def test_lesson_boundaries(tmp):
    print("test_lesson_boundaries")
    vid = _setup_video(tmp, lessons=3, sentences_per_lesson=1)
    scenes = db.get_scenes(vid)
    total = sum(len(txt.split_sentences(s["text"])) for s in scenes)
    boundaries = stage._lesson_boundary_indices(vid, total)
    # hook(1) + lesson0(1) + lesson1(1) + lesson2(1) + outro(1) = 5 sentences, global
    # indices 0..4. Boundaries: start of lesson0=1, lesson1=2, lesson2=3, outro=4.
    check(boundaries == {1, 2, 3, 4}, f"boundaries at every part start except the hook: {boundaries}")


def test_boundary_mismatch_falls_back(tmp):
    print("test_boundary_mismatch_falls_back")
    vid = _setup_video(tmp, lessons=2, sentences_per_lesson=1)
    # Corrupt outline.json so its lesson count no longer matches script.txt's parts.
    out = config.output_dir(vid)
    (out / "outline.json").write_text(json.dumps({"lessons": [{"title": "only one"}]}),
                                      encoding="utf-8")
    boundaries = stage._lesson_boundary_indices(vid, 4)
    check(boundaries == set(), "a part-count mismatch returns no boundaries instead of guessing")


def test_pause_placement_and_timings(tmp):
    print("test_pause_placement_and_timings")
    vid = _setup_video(tmp, lessons=2, sentences_per_lesson=2)
    original = {name: getattr(tts, name) for name in
               ("synthesize", "silence", "concatenate", "master", "measure_volume",
                "extract_sample")}
    fake = FakeAudio()
    _install_fake(fake)
    try:
        cfg = {"words_per_minute": 145,
              "tts": {"provider": "piper",
                      "pauses": {"sentence_ms": 220, "scene_ms": 450, "lesson_ms": 900}}}
        check(stage.run(vid, cfg) is True, "the stage completes with stubbed audio")
    finally:
        _restore(original)

    out = config.output_dir(vid)
    timings = json.loads((out / "timings.json").read_text(encoding="utf-8"))
    scenes = db.get_scenes(vid)
    # Parts: hook(1) lesson0(2) lesson1(2) outro(1) = 6 sentences total.
    check(len(timings) == 6, f"one timings entry per sentence ({len(timings)})")
    check(timings[0]["start_s"] == 0.0, "the very first sentence starts at t=0, no lead-in pause")

    word_seconds = 6 / WORDS_PER_SECOND        # "This is the hook sentence here." = 6 words
    check(abs(timings[0]["end_s"] - word_seconds) < 0.01, "first sentence duration matches its word count")

    # lesson0's first sentence (index 1) is a lesson boundary -> 900ms gap before it.
    gap = timings[1]["start_s"] - timings[0]["end_s"]
    check(abs(gap - 0.9) < 0.01, f"lesson-boundary gap is 0.9s ({gap:.3f})")

    # lesson0's second sentence (index 2) is mid-scene... actually each sentence is its
    # own scene here (one sentence per scene, see _setup_video), so this is a new SCENE
    # but not a lesson boundary -> 450ms gap.
    gap2 = timings[2]["start_s"] - timings[1]["end_s"]
    check(abs(gap2 - 0.45) < 0.01, f"ordinary scene gap is 0.45s ({gap2:.3f})")

    check(all(s["audio_path"] for s in scenes), "every scene got an audio_path")
    check(all(s["duration_s"] is not None for s in scenes), "every scene got a duration_s")

    video = db.get_video(vid)
    check(video["duration_s"] is not None, "videos.duration_s was set")


def test_mid_scene_lesson_boundary(tmp):
    print("test_mid_scene_lesson_boundary")
    # Two sentences from DIFFERENT lessons folded into one scene on purpose, so the
    # lesson-boundary pause must land INSIDE a scene, not only between scenes.
    db.DB_PATH = tmp / "mid.db"
    config.OUTPUT_DIR = tmp / "out_mid"
    db.init_db()
    vid = db.create_video("mid boundary test")
    hook = "This is the hook sentence here."
    lesson0 = "Lesson zero sentence one goes here."
    lesson1 = "Lesson one sentence one goes here."
    outro = "This is the outro sentence here."
    full_text = "\n\n".join([hook, lesson0, lesson1, outro])
    out = config.output_dir(vid)
    (out / "script.txt").write_text(f"Title\n\n{full_text}\n", encoding="utf-8")
    (out / "outline.json").write_text(
        json.dumps({"lessons": [{"title": "L0"}, {"title": "L1"}]}), encoding="utf-8")
    # One scene holds the hook alone; the second holds BOTH lesson sentences together,
    # which is where the lesson-boundary pause must be detected mid-scene; the third
    # holds the outro, keeping the part count consistent with outline.json's 2 lessons.
    db.add_scene(vid, 1, text=hook, image_prompt="x")
    db.add_scene(vid, 2, text=f"{lesson0} {lesson1}", image_prompt="x")
    db.add_scene(vid, 3, text=outro, image_prompt="x")

    original = {name: getattr(tts, name) for name in
               ("synthesize", "silence", "concatenate", "master", "measure_volume",
                "extract_sample")}
    fake = FakeAudio()
    _install_fake(fake)
    try:
        cfg = {"words_per_minute": 145, "tts": {"provider": "piper"}}
        stage.run(vid, cfg)
    finally:
        _restore(original)

    timings = json.loads((out / "timings.json").read_text(encoding="utf-8"))
    check(len(timings) == 4, "four sentences across three scenes")
    # timings[1] is lesson0's sentence (scene 2, local index 0); timings[2] is
    # lesson1's sentence (scene 2, local index 1) - same scene, but a lesson boundary.
    gap = timings[2]["start_s"] - timings[1]["end_s"]
    check(abs(gap - 0.9) < 0.01,
          f"the lesson-boundary pause fires mid-scene, not just between scenes ({gap:.3f})")


def test_scene_duration_warning(tmp):
    print("test_scene_duration_warning")
    db.DB_PATH = tmp / "dur.db"
    config.OUTPUT_DIR = tmp / "out_dur"
    db.init_db()
    vid = db.create_video("duration test")
    # A single short scene: well under the 15s floor at the stub's fixed word rate.
    db.add_scene(vid, 1, text="Five short words right here.", image_prompt="x")

    original = {name: getattr(tts, name) for name in
               ("synthesize", "silence", "concatenate", "master", "measure_volume",
                "extract_sample")}
    fake = FakeAudio()
    _install_fake(fake)
    try:
        cfg = {"words_per_minute": 145, "tts": {"provider": "piper"}}
        stage.run(vid, cfg)
    finally:
        _restore(original)

    import sqlite3
    from contextlib import closing
    with closing(sqlite3.connect(db.DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row
        events = [dict(r) for r in conn.execute(
            "SELECT * FROM events WHERE video_id = ? AND stage = 'tts'", (vid,))]
    check(any("outside" in e["message"] and e["level"] == "warning" for e in events),
          "a too-short scene logs a warning naming the range")


def test_silence_deliverable_raises(tmp):
    print("test_silence_deliverable_raises")
    db.DB_PATH = tmp / "sil.db"
    config.OUTPUT_DIR = tmp / "out_sil"
    db.init_db()
    vid = db.create_video("silence test")
    db.add_scene(vid, 1, text="A single scene of narration text.", image_prompt="x")

    original = {name: getattr(tts, name) for name in
               ("synthesize", "silence", "concatenate", "master", "measure_volume",
                "extract_sample")}
    fake = FakeAudio()
    fake.measure_volume = lambda path: (-70.0, -60.0)   # below SILENCE_THRESHOLD_DBFS
    _install_fake(fake)
    try:
        cfg = {"words_per_minute": 145, "tts": {"provider": "piper"}}
        try:
            stage.run(vid, cfg)
            check(False, "a silent deliverable should raise")
        except RuntimeError as e:
            check("silence" in str(e), f"the error names the actual problem ({e})")
    finally:
        _restore(original)


def test_no_scenes_raises(tmp):
    print("test_no_scenes_raises")
    db.DB_PATH = tmp / "empty.db"
    config.OUTPUT_DIR = tmp / "out_empty"
    db.init_db()
    vid = db.create_video("no scenes")
    try:
        stage.run(vid, {"words_per_minute": 145, "tts": {"provider": "piper"}})
        check(False, "running tts with zero scenes should raise")
    except RuntimeError as e:
        check("no scenes" in str(e), "the error names the actual problem")


def test_provider_settings():
    print("test_provider_settings")
    provider, settings = tts.provider_settings({})
    check(provider == "piper", "provider defaults to piper when tts block is absent")
    provider, settings = tts.provider_settings(
        {"tts": {"provider": "edge", "edge": {"voice": "en-GB-RyanNeural"}}})
    check(provider == "edge" and settings["voice"] == "en-GB-RyanNeural",
          "edge provider and voice override are read")
    try:
        tts.provider_settings({"tts": {"provider": "nonsense"}})
        check(False, "an unknown provider should raise")
    except tts.TTSError:
        check(True, "an unknown provider raises TTSError")


def test_pause_and_mastering_settings():
    print("test_pause_and_mastering_settings")
    p = tts.pause_settings({})
    check((p["sentence_ms"], p["scene_ms"], p["lesson_ms"]) == (220, 450, 900),
          "default pauses match the spec (220/450/900ms)")
    m = tts.mastering_settings({})
    check(m["target_lufs"] == -14 and m["highpass_hz"] == 80 and m["sample_rate"] == 48000
          and m["bitrate_kbps"] == 192, "default mastering settings match the spec")
    p2 = tts.pause_settings({"tts": {"pauses": {"lesson_ms": 2000}}})
    check(p2["lesson_ms"] == 2000 and p2["sentence_ms"] == 220,
          "a partial override keeps the other defaults")


def test_403_message():
    print("test_403_message")
    check(tts._is_403(Exception("403, message='Forbidden'")), "a 403 string is recognised")
    check(tts._is_403(Exception("NoAudioReceived: no audio was received")),
          "edge-tts's own empty-stream error is treated the same as a 403")
    check(not tts._is_403(Exception("connection reset")),
          "an unrelated network error is not mistaken for a 403")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        test_provider_settings()
        test_pause_and_mastering_settings()
        test_403_message()
        test_lesson_boundaries(tmp)
        test_boundary_mismatch_falls_back(tmp)
        test_pause_placement_and_timings(tmp)
        test_mid_scene_lesson_boundary(tmp)
        test_scene_duration_warning(tmp)
        test_silence_deliverable_raises(tmp)
        test_no_scenes_raises(tmp)
    print("ALL TTS TESTS PASSED")
