"""Offline tests for the render stage. Run: python tests/test_render.py

No ffmpeg calls: core.render's kenburns_clip, concat_clips, xfade_chain, mux_audio,
extract_preview, probe_duration and frame_brightness are all stubbed, so this checks
the pipeline's own logic (per-scene resumability, movement cycling, the sanity
checks) without rendering real video.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Keep test output out of the working logs/factory.log.
import core.logger
core.logger.LOG_DIR = Path(tempfile.mkdtemp(prefix="yt-factory-test-"))

from core import config, db, render
import pipeline.render as stage


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok: {msg}")


RENDER_CFG = {
    "width": 1920, "height": 1080, "fps": 24, "crf": 21, "preset": "medium",
    "audio_bitrate_kbps": 192, "crossfade_s": 0.6, "upscale_factor": 4,
    "zoom_max": 1.15, "exposures_per_image": 2, "preview_seconds": 60,
    "movements": [{"zoom": "in", "focus": "left"}, {"zoom": "in", "focus": "right"},
                 {"zoom": "out", "focus": "left"}, {"zoom": "out", "focus": "right"}],
}


class FakeFFmpeg:
    """Deterministic stand-ins for every core.render ffmpeg call: writes a tiny
    placeholder file and tracks what was asked for, instead of invoking ffmpeg."""

    def __init__(self):
        self.kenburns_calls = []
        self.concat_calls = []
        self.xfade_calls = []
        self.durations = {}   # path (str) -> duration, for probe_duration to report
        self.brightness = 150.0

    def kenburns_clip(self, image_path, out_path, duration_s, movement, cfg_render, cache_dir):
        self.kenburns_calls.append((str(image_path), duration_s, dict(movement)))
        Path(out_path).write_bytes(b"\x00")
        self.durations[str(out_path)] = duration_s
        return out_path

    def concat_clips(self, clip_paths, out_path):
        self.concat_calls.append([str(p) for p in clip_paths])
        Path(out_path).write_bytes(b"\x00")
        self.durations[str(out_path)] = sum(self.durations.get(str(p), 0) for p in clip_paths)
        return out_path

    def xfade_chain(self, clip_paths, out_path, crossfade_s, cfg_render):
        self.xfade_calls.append([str(p) for p in clip_paths])
        Path(out_path).write_bytes(b"\x00")
        total = sum(self.durations.get(str(p), 0) for p in clip_paths)
        total -= crossfade_s * max(0, len(clip_paths) - 1)
        self.durations[str(out_path)] = total
        return total

    def mux_audio(self, video_path, audio_path, out_path, cfg_render):
        Path(out_path).write_bytes(b"\x00" * 2_000_000)   # ~2MB placeholder
        self.durations[str(out_path)] = self.durations.get(str(video_path), 0)
        return out_path

    def extract_preview(self, video_path, out_path, seconds, cfg_render):
        Path(out_path).write_bytes(b"\x00" * 100_000)
        return out_path

    def probe_duration(self, path):
        return self.durations.get(str(path), 60.0)

    def frame_brightness(self, video_path, at_seconds):
        return self.brightness


def _install_fake(fake):
    original = {name: getattr(render, name) for name in
               ("kenburns_clip", "concat_clips", "xfade_chain", "mux_audio",
                "extract_preview", "probe_duration", "frame_brightness", "require_ffmpeg")}
    render.kenburns_clip = fake.kenburns_clip
    render.concat_clips = fake.concat_clips
    render.xfade_chain = fake.xfade_chain
    render.mux_audio = fake.mux_audio
    render.extract_preview = fake.extract_preview
    render.probe_duration = fake.probe_duration
    render.frame_brightness = fake.frame_brightness
    render.require_ffmpeg = lambda: None
    stage.render = render
    return original


def _restore(original):
    for name, fn in original.items():
        setattr(render, name, fn)


def _setup_video(tmp, name, scene_durations, images_per_scene=1):
    db.DB_PATH = tmp / f"{name}.db"
    config.OUTPUT_DIR = tmp / f"out_{name}"
    db.init_db()
    vid = db.create_video(f"{name} test")
    out = config.output_dir(vid)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    for i, dur in enumerate(scene_durations, 1):
        db.add_scene(vid, i, text=f"Scene {i}.", image_prompt="x", duration_s=dur)
        scene = db.get_scenes(vid)[-1]
        for j in range(images_per_scene):
            img_path = img_dir / f"scene_{i:03d}_{j:02d}.jpg"
            img_path.write_bytes(b"\xff\xd8\xff")
            db.add_scene_image(vid, scene["id"], j, prompt="x", seed=i * 1000 + j,
                               provider="cloudflare", path=str(img_path))
    (out / "voice.mp3").write_bytes(b"\x00" * 500_000)
    return vid


def test_movement_cycles_globally():
    print("test_movement_cycles_globally")
    movements = RENDER_CFG["movements"]
    seq = [stage._movement_for(i, movements) for i in range(6)]
    check(seq[0] == movements[0] and seq[3] == movements[3],
          "the cycle follows the configured order")
    check(seq[4] == movements[0], "the cycle wraps around after the list length")


def test_full_run_builds_expected_clips(tmp):
    print("test_full_run_builds_expected_clips")
    vid = _setup_video(tmp, "basic", [20, 30], images_per_scene=2)
    fake = FakeFFmpeg()
    fake.durations[str(config.output_dir(vid) / "voice.mp3")] = 20 + 30 - 0.6
    original = _install_fake(fake)
    try:
        check(stage.run(vid, {"render": RENDER_CFG}) is True, "the stage completes")
    finally:
        _restore(original)

    # 2 scenes x 2 images x 2 exposures = 8 Ken Burns sub-shots.
    check(len(fake.kenburns_calls) == 8, f"8 sub-shots rendered ({len(fake.kenburns_calls)})")
    check(len(fake.concat_calls) == 2, "one concat per scene (2 scenes)")
    check(len(fake.xfade_calls) == 1, "one cross-fade pass across both scene clips")
    check(len(fake.xfade_calls[0]) == 2, "the cross-fade saw exactly the 2 scene clips")

    video = db.get_video(vid)
    check(video["video_path"] is not None, "videos.video_path was written")
    check(video["duration_s"] is not None, "videos.duration_s was written")

    out = config.output_dir(vid)
    check((out / "final.mp4").exists(), "final.mp4 exists")
    check((out / "preview_60s.mp4").exists(), "preview_60s.mp4 exists")
    check(not (out / "_video_silent.mp4").exists(),
          "the silent intermediate file is cleaned up")


def test_per_image_duration_split_evenly(tmp):
    print("test_per_image_duration_split_evenly")
    vid = _setup_video(tmp, "split", [40], images_per_scene=2)   # 2 images, 40s scene
    fake = FakeFFmpeg()
    fake.durations[str(config.output_dir(vid) / "voice.mp3")] = 40
    original = _install_fake(fake)
    try:
        stage.run(vid, {"render": RENDER_CFG})
    finally:
        _restore(original)

    # 40s / 2 images / 2 exposures = 10s per sub-shot.
    sub_durations = {round(d, 3) for _, d, _ in fake.kenburns_calls}
    check(sub_durations == {10.0}, f"each sub-shot got 10s ({sub_durations})")


def test_movements_alternate_within_and_across_images(tmp):
    print("test_movements_alternate_within_and_across_images")
    vid = _setup_video(tmp, "altern", [24], images_per_scene=2)
    fake = FakeFFmpeg()
    fake.durations[str(config.output_dir(vid) / "voice.mp3")] = 24
    original = _install_fake(fake)
    try:
        stage.run(vid, {"render": RENDER_CFG})
    finally:
        _restore(original)

    used = [m for _, _, m in fake.kenburns_calls]
    check(used[0]["focus"] == "left" and used[1]["focus"] == "right",
          "the two exposures of one image use different focus points")
    check(used == RENDER_CFG["movements"][:4],
          f"movements cycle through the configured list in order ({used})")


def test_existing_scene_clip_is_skipped(tmp):
    print("test_existing_scene_clip_is_skipped")
    vid = _setup_video(tmp, "resume", [20, 20], images_per_scene=1)
    out = config.output_dir(vid)
    clips_dir = out / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    (clips_dir / "scene_001.mp4").write_bytes(b"\x00" * 1000)   # pre-exists

    fake = FakeFFmpeg()
    fake.durations[str(clips_dir / "scene_001.mp4")] = 20.0
    fake.durations[str(out / "voice.mp3")] = 40 - 0.6
    original = _install_fake(fake)
    try:
        stage.run(vid, {"render": RENDER_CFG})
    finally:
        _restore(original)

    rendered_scenes = {Path(p).stem for calls in [fake.concat_calls] for group in calls
                       for p in group}
    # Only scene 2's sub-shots should have been built; scene 1 was reused untouched.
    check(all("scene_002" in str(c[0]) for c in fake.kenburns_calls) or not fake.kenburns_calls,
          "kenburns_clip was only called for the scene without an existing clip")
    check(len(fake.concat_calls) == 1, "only the missing scene got concatenated")


def test_movement_cycle_accounts_for_skipped_scenes(tmp):
    print("test_movement_cycle_accounts_for_skipped_scenes")
    vid = _setup_video(tmp, "cycleresume", [20, 20, 20], images_per_scene=1)
    out = config.output_dir(vid)
    clips_dir = out / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    (clips_dir / "scene_001.mp4").write_bytes(b"\x00" * 1000)

    fake = FakeFFmpeg()
    fake.durations[str(clips_dir / "scene_001.mp4")] = 20.0
    fake.durations[str(out / "voice.mp3")] = 60 - 1.2
    original = _install_fake(fake)
    try:
        stage.run(vid, {"render": RENDER_CFG})
    finally:
        _restore(original)

    # Scene 1 (1 image x 2 exposures) is skipped but must still advance the global
    # counter, so scene 2's first exposure continues the cycle at index 2, not 0.
    first_used = fake.kenburns_calls[0][2]
    check(first_used == RENDER_CFG["movements"][2],
          f"the cycle picks up where a skipped scene left off ({first_used})")


def test_black_first_frame_raises(tmp):
    print("test_black_first_frame_raises")
    vid = _setup_video(tmp, "black", [20], images_per_scene=1)
    fake = FakeFFmpeg()
    fake.durations[str(config.output_dir(vid) / "voice.mp3")] = 20
    fake.brightness = 2.0   # looks black
    original = _install_fake(fake)
    try:
        try:
            stage.run(vid, {"render": RENDER_CFG})
            check(False, "a black first/last frame should raise")
        except RuntimeError as e:
            check("black" in str(e), f"the error names the actual problem ({e})")
    finally:
        _restore(original)


def test_duration_mismatch_raises(tmp):
    print("test_duration_mismatch_raises")
    vid = _setup_video(tmp, "mismatch", [20], images_per_scene=1)
    fake = FakeFFmpeg()
    original = _install_fake(fake)
    try:
        # Force a bogus gap between the final video and the audio track.
        real_probe = fake.probe_duration
        def bad_probe(path):
            if "voice.mp3" in str(path):
                return 9999.0
            return real_probe(path)
        fake.probe_duration = bad_probe
        try:
            stage.run(vid, {"render": RENDER_CFG})
            check(False, "a large video/audio duration mismatch should raise")
        except RuntimeError as e:
            check("apart" in str(e) or "s but" in str(e), f"the error is specific ({e})")
    finally:
        _restore(original)


def test_no_scenes_raises(tmp):
    print("test_no_scenes_raises")
    db.DB_PATH = tmp / "empty.db"
    config.OUTPUT_DIR = tmp / "out_empty"
    db.init_db()
    vid = db.create_video("no scenes")
    try:
        stage.run(vid, {"render": RENDER_CFG})
        check(False, "running render with zero scenes should raise")
    except RuntimeError as e:
        check("no scenes" in str(e), "the error names the actual problem")


def test_missing_voice_raises(tmp):
    print("test_missing_voice_raises")
    vid = _setup_video(tmp, "novoice", [20], images_per_scene=1)
    (config.output_dir(vid) / "voice.mp3").unlink()
    try:
        stage.run(vid, {"render": RENDER_CFG})
        check(False, "a missing voice.mp3 should raise")
    except RuntimeError as e:
        check("voice.mp3" in str(e), "the error names the actual problem")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        test_movement_cycles_globally()
        test_full_run_builds_expected_clips(tmp)
        test_per_image_duration_split_evenly(tmp)
        test_movements_alternate_within_and_across_images(tmp)
        test_existing_scene_clip_is_skipped(tmp)
        test_movement_cycle_accounts_for_skipped_scenes(tmp)
        test_black_first_frame_raises(tmp)
        test_duration_mismatch_raises(tmp)
        test_no_scenes_raises(tmp)
        test_missing_voice_raises(tmp)
    print("ALL RENDER TESTS PASSED")
