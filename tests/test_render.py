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
    "shuffle_min_block_images": 3, "shuffle_gap_shots": [2, 4],
    "movements": [{"zoom": "in", "focus": "left"}, {"zoom": "in", "focus": "right"},
                 {"zoom": "out", "focus": "left"}, {"zoom": "out", "focus": "right"}],
}


class FakeFFmpeg:
    """Deterministic stand-ins for every core.render ffmpeg call: writes a tiny
    placeholder file and tracks what was asked for, instead of invoking ffmpeg."""

    def __init__(self):
        self.kenburns_calls = []
        self.static_calls = []
        self.concat_calls = []
        self.xfade_calls = []
        self.durations = {}   # path (str) -> duration, for probe_duration to report
        self.brightness = 150.0
        self.kenburns_always_fails = False

    def kenburns_clip(self, image_path, out_path, duration_s, movement, cfg_render, cache_dir):
        self.kenburns_calls.append((str(image_path), duration_s, dict(movement)))
        if self.kenburns_always_fails:
            raise render.RenderError("Ken Burns timed out every attempt (simulated)")
        Path(out_path).write_bytes(b"\x00")
        self.durations[str(out_path)] = duration_s
        return out_path

    def static_clip(self, image_path, out_path, duration_s, cfg_render, cache_dir):
        self.static_calls.append((str(image_path), duration_s))
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
               ("kenburns_clip", "static_clip", "concat_clips", "xfade_chain", "mux_audio",
                "extract_preview", "probe_duration", "frame_brightness", "require_ffmpeg")}
    render.kenburns_clip = fake.kenburns_clip
    render.static_clip = fake.static_clip
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


def _fake_scene_items(images_per_scene, duration_s=30.0):
    """Scene rows as plan_shots sees them: (scene, images). Plain dicts are enough -
    plan_shots only reads idx, duration_s and hands the image rows straight through."""
    return [({"idx": i, "duration_s": duration_s},
             [{"id": i * 100 + j, "path": f"scene_{i:03d}_{j:02d}.jpg"} for j in range(n)])
            for i, n in enumerate(images_per_scene, 1)]


def _shot_gaps(shots):
    """How many other shots sit between each image's two showings."""
    seen, gaps = {}, []
    for k, shot in enumerate(shots):
        image_id = shot["image"]["id"]
        if image_id in seen:
            gaps.append(k - seen[image_id] - 1)
        seen[image_id] = k
    return gaps


def test_repeats_are_never_shown_back_to_back():
    print("test_repeats_are_never_shown_back_to_back")
    # The shape that broke the first cut of video 7: mostly one-image scenes, where
    # a scene's own two exposures are necessarily the same picture.
    items = _fake_scene_items([1, 2, 1, 1, 2, 1, 1, 1])
    shots = stage.plan_shots(items, RENDER_CFG, seed=7)

    check(len(shots) == 20, f"every image got both exposures ({len(shots)} shots)")
    adjacent = [k for k in range(1, len(shots))
                if shots[k]["image"]["id"] == shots[k - 1]["image"]["id"]]
    check(not adjacent, f"no image is ever shown twice in a row ({adjacent})")
    gaps = _shot_gaps(shots)
    check(min(gaps) >= 1, f"every repeat has at least one other shot before it ({min(gaps)})")
    spread = sum(1 for g in gaps if 2 <= g <= 4) / len(gaps)
    check(spread >= 0.8, f"{spread:.0%} of repeats land in the asked-for 2-4 shot gap")


def test_shot_lengths_still_fill_each_scene_exactly():
    print("test_shot_lengths_still_fill_each_scene_exactly")
    # The one thing shuffling must never do: move a scene boundary. The voice track
    # does not shuffle with it, so any drift here desyncs the rest of the video.
    items = _fake_scene_items([1, 2, 1, 1, 2], duration_s=33.0)
    items[2][0]["duration_s"] = 19.0        # an odd one out, to catch an averaged split
    shots = stage.plan_shots(items, RENDER_CFG, seed=3)

    per_scene = {}
    for shot in shots:
        idx = shot["scene"]["idx"]
        per_scene[idx] = per_scene.get(idx, 0) + shot["duration_s"]
    worst = max(abs(per_scene[scene["idx"]] - scene["duration_s"]) for scene, _ in items)
    check(worst < 1e-6, f"each scene's shots still add up to its own duration ({worst:.2e}s)")


def test_a_picture_stays_near_its_own_text():
    print("test_a_picture_stays_near_its_own_text")
    items = _fake_scene_items([1, 2, 1, 1, 2, 1, 1, 1])
    shots = stage.plan_shots(items, RENDER_CFG, seed=7)
    distances = [abs(s["scene"]["idx"] - s["home_scene_idx"]) for s in shots]
    check(max(distances) <= 1,
          f"no picture plays more than one scene from the words it was drawn for "
          f"({max(distances)})")
    at_home = sum(1 for d in distances if d == 0) / len(distances)
    check(at_home >= 0.4, f"{at_home:.0%} of shots still play inside their own scene")


def test_two_showings_differ_in_framing_and_zoom():
    print("test_two_showings_differ_in_framing_and_zoom")
    items = _fake_scene_items([1, 2, 1, 1])
    shots = stage.plan_shots(items, RENDER_CFG, seed=11)
    by_image = {}
    for shot in shots:
        by_image.setdefault(shot["image"]["id"], []).append(shot["movement"])

    same_focus = [i for i, ms in by_image.items() if ms[0]["focus"] == ms[1]["focus"]]
    same_zoom = [i for i, ms in by_image.items() if ms[0]["zoom"] == ms[1]["zoom"]]
    check(not same_focus, f"the two showings never share a framing ({same_focus})")
    check(not same_zoom, f"the two showings never share a zoom direction ({same_zoom})")


def test_plan_is_deterministic_in_the_video_id():
    print("test_plan_is_deterministic_in_the_video_id")
    items = _fake_scene_items([1, 2, 1, 1, 2, 1])
    first = stage.plan_shots(items, RENDER_CFG, seed=7)
    again = stage.plan_shots(items, RENDER_CFG, seed=7)
    other = stage.plan_shots(items, RENDER_CFG, seed=8)

    def signature(shots):
        return [(s["image"]["id"], s["movement"]["focus"], s["movement"]["zoom"])
                for s in shots]

    check(signature(first) == signature(again),
          "the same video id always plans the same video, so a resumed run matches")
    check(signature(first) != signature(other),
          "a different video id plans a different order")


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


def test_contrasting_pairs_rejects_a_lookalike_repeat():
    print("test_contrasting_pairs_rejects_a_lookalike_repeat")
    pairs = render.contrasting_pairs(RENDER_CFG["movements"])
    check(pairs, "the configured movements yield usable pairs")
    check(all(a["focus"] != b["focus"] and a["zoom"] != b["zoom"] for a, b in pairs),
          "every pair differs in both framing and zoom direction")

    # A list that cannot satisfy the strict rule must degrade, not crash: a repeat
    # still has to be rendered somehow.
    weak = render.contrasting_pairs([{"zoom": "in", "focus": "left"},
                                     {"zoom": "in", "focus": "right"}])
    check(weak and all(a["focus"] != b["focus"] for a, b in weak),
          "with only one zoom direction configured it falls back to differing framings")
    single = render.contrasting_pairs([{"zoom": "in", "focus": "left"}])
    check(len(single) == 1, "a single configured movement still yields a usable pair")


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

    # Only scene 2's two slots should have been built; scene 1 was reused untouched.
    check(len(fake.kenburns_calls) == 2,
          f"only the missing scene's shots were rendered ({len(fake.kenburns_calls)})")
    check(len(fake.concat_calls) == 1, "only the missing scene got concatenated")


def test_resumed_run_renders_the_same_shots_it_would_have(tmp):
    print("test_resumed_run_renders_the_same_shots_it_would_have")
    # Shots are planned for the whole video up front from the video id, so a run that
    # reuses earlier scene clips must still render the later scenes exactly as the
    # interrupted run would have. Rendered once whole, once resumed, compared.
    vid = _setup_video(tmp, "resumeplan", [20, 20, 20], images_per_scene=1)
    out = config.output_dir(vid)
    clips_dir = out / "clips"

    fake = FakeFFmpeg()
    fake.durations[str(out / "voice.mp3")] = 60 - 1.2
    original = _install_fake(fake)
    try:
        stage.run(vid, {"render": RENDER_CFG})
        whole = [(Path(img).name, dict(m)) for img, _, m in fake.kenburns_calls]

        # Throw away everything after scene 1 and run again.
        for leftover in list(clips_dir.glob("scene_002.mp4")) + \
                        list(clips_dir.glob("scene_003.mp4")) + \
                        list((clips_dir / "_parts").glob("scene_00[23]_*.mp4")):
            leftover.unlink()
        fake.kenburns_calls.clear()
        stage.run(vid, {"render": RENDER_CFG})
        resumed = [(Path(img).name, dict(m)) for img, _, m in fake.kenburns_calls]
    finally:
        _restore(original)

    check(resumed == whole[-len(resumed):] and len(resumed) == 4,
          f"the resumed run rendered scenes 2-3 exactly as the first run did ({resumed})")


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


def test_static_fallback_keeps_timing_when_kenburns_fails(tmp):
    print("test_static_fallback_keeps_timing_when_kenburns_fails")
    vid = _setup_video(tmp, "staticfb", [24], images_per_scene=1)
    fake = FakeFFmpeg()
    fake.durations[str(config.output_dir(vid) / "voice.mp3")] = 24
    fake.kenburns_always_fails = True
    original = _install_fake(fake)
    try:
        check(stage.run(vid, {"render": RENDER_CFG}) is True,
              "the stage completes even when Ken Burns fails for every sub-shot")
    finally:
        _restore(original)

    check(len(fake.static_calls) == 2,
          f"both exposures fell back to a motionless shot ({len(fake.static_calls)})")
    durations = {round(d, 3) for _, d in fake.static_calls}
    check(durations == {12.0},
          f"the fallback holds each sub-shot's exact duration, so the voice track "
          f"stays in sync ({durations})")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        test_repeats_are_never_shown_back_to_back()
        test_shot_lengths_still_fill_each_scene_exactly()
        test_a_picture_stays_near_its_own_text()
        test_two_showings_differ_in_framing_and_zoom()
        test_plan_is_deterministic_in_the_video_id()
        test_full_run_builds_expected_clips(tmp)
        test_per_image_duration_split_evenly(tmp)
        test_contrasting_pairs_rejects_a_lookalike_repeat()
        test_existing_scene_clip_is_skipped(tmp)
        test_resumed_run_renders_the_same_shots_it_would_have(tmp)
        test_black_first_frame_raises(tmp)
        test_duration_mismatch_raises(tmp)
        test_no_scenes_raises(tmp)
        test_missing_voice_raises(tmp)
        test_static_fallback_keeps_timing_when_kenburns_fails(tmp)
    print("ALL RENDER TESTS PASSED")
