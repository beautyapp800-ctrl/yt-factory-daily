"""Offline tests for the render stage. Run: python tests/test_render.py

No ffmpeg calls: core.render's kenburns_clip, concat_clips, join_clips, mux_audio,
extract_preview, probe_duration and frame_brightness are all stubbed, so this checks
the pipeline's own logic (per-scene resumability, movement cycling, the sanity
checks) without rendering real video.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Keep test output out of the working logs/factory.log.
import core.logger
core.logger.LOG_DIR = Path(tempfile.mkdtemp(prefix="yt-factory-test-"))

from core import config, db, music, render
import pipeline.render as stage


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok: {msg}")


RENDER_CFG = {
    "width": 1920, "height": 1080, "fps": 24, "crf": 21, "preset": "medium",
    "audio_bitrate_kbps": 192, "scene_fade_s": 0.5, "upscale_factor": 4,
    "zoom_min": 1.08, "zoom_max": 1.26, "preview_seconds": 60,
    "double_exposure": False,
    "pan_movements": [{"zoom": "in", "pan": "right"}, {"zoom": "out", "pan": "left"},
                     {"zoom": "out", "pan": "right"}, {"zoom": "in", "pan": "left"}],
    # Only read when double_exposure is on; kept here so the parked scheme stays tested.
    "exposures_per_image": 2,
    "shuffle_min_block_images": 3, "shuffle_gap_shots": [2, 4],
    "movements": [{"zoom": "in", "focus": "left"}, {"zoom": "in", "focus": "right"},
                 {"zoom": "out", "focus": "left"}, {"zoom": "out", "focus": "right"}],
}

# The parked double-exposure scheme. Switched off in config.json because 44% of its
# shots played a scene away from their own text; still tested, so turning the flag
# back on cannot quietly be broken in the meantime.
DOUBLE_CFG = dict(RENDER_CFG, double_exposure=True)


class FakeFFmpeg:
    """Deterministic stand-ins for every core.render ffmpeg call: writes a tiny
    placeholder file and tracks what was asked for, instead of invoking ffmpeg."""

    def __init__(self):
        self.kenburns_calls = []
        self.static_calls = []
        self.fades = []
        self.concat_calls = []
        self.music_paths = []
        self.bed_calls = []
        self.join_calls = []
        self.durations = {}   # path (str) -> duration, for probe_duration to report
        self.brightness = 150.0
        self.kenburns_always_fails = False

    def kenburns_clip(self, image_path, out_path, duration_s, movement, cfg_render, cache_dir,
                      fade_in=False, fade_out=False):
        self.kenburns_calls.append((str(image_path), duration_s, dict(movement)))
        self.fades.append((Path(out_path).name, fade_in, fade_out))
        if self.kenburns_always_fails:
            raise render.RenderError("Ken Burns timed out every attempt (simulated)")
        Path(out_path).write_bytes(b"\x00")
        self.durations[str(out_path)] = duration_s
        return out_path

    def static_clip(self, image_path, out_path, duration_s, cfg_render, cache_dir,
                    fade_in=False, fade_out=False):
        self.static_calls.append((str(image_path), duration_s))
        self.fades.append((Path(out_path).name, fade_in, fade_out))
        Path(out_path).write_bytes(b"\x00")
        self.durations[str(out_path)] = duration_s
        return out_path

    def concat_clips(self, clip_paths, out_path):
        self.concat_calls.append([str(p) for p in clip_paths])
        Path(out_path).write_bytes(b"\x00")
        self.durations[str(out_path)] = sum(self.durations.get(str(p), 0) for p in clip_paths)
        return out_path

    def build_bed(self, track, seconds, out_path, cfg, voice_lufs):
        self.bed_calls.append((Path(track).name, round(seconds, 2), round(voice_lufs, 1)))
        Path(out_path).write_bytes(b"0")
        return Path(out_path), {"track": Path(track).name, "voice_lufs": voice_lufs}

    def join_clips(self, clip_paths, out_path):
        self.join_calls.append([str(p) for p in clip_paths])
        Path(out_path).write_bytes(b"\x00")
        total = sum(self.durations.get(str(p), 0) for p in clip_paths)
        self.durations[str(out_path)] = total
        return total

    def mux_audio(self, video_path, audio_path, out_path, cfg_render, music_path=None):
        self.music_paths.append(str(music_path) if music_path else None)
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
               ("kenburns_clip", "static_clip", "concat_clips", "join_clips", "mux_audio",
                "extract_preview", "probe_duration", "frame_brightness", "require_ffmpeg")}
    render.kenburns_clip = fake.kenburns_clip
    render.static_clip = fake.static_clip
    render.concat_clips = fake.concat_clips
    render.join_clips = fake.join_clips
    render.mux_audio = fake.mux_audio
    render.extract_preview = fake.extract_preview
    render.probe_duration = fake.probe_duration
    render.frame_brightness = fake.frame_brightness
    render.require_ffmpeg = lambda: None
    stage.render = render
    # The music bed is a separate concern with its own tests; here it only has to be built
    # and handed to the mux, so the ffmpeg behind it is replaced the same way.
    original.update({f"music.{n}": getattr(music, n) for n in
                     ("available_tracks", "loudness", "build_bed")})
    music.available_tracks = lambda: [Path("assets/music/fake-track.ogg")]
    music.loudness = lambda path: -15.8 if "voice" in str(path) else -41.8
    music.build_bed = fake.build_bed
    return original


def _restore(original):
    for name, fn in original.items():
        module, _, attr = name.partition(".")
        setattr(music if module == "music" else render, attr or module, fn)


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


def test_scene_lengths_cover_the_silence_between_scenes(tmp):
    print("test_scene_lengths_cover_the_silence_between_scenes")
    # The tts stage puts 450ms (or 900ms at a lesson) between scenes. That silence is in
    # voice.mp3 but in no scene's spoken duration, so a video built from spoken durations
    # drifts: 20.7s over 43 scenes, measured on video 7.
    db.DB_PATH = tmp / "gaps.db"
    config.OUTPUT_DIR = tmp / "out_gaps"
    db.init_db()
    vid = db.create_video("gap test")
    for i, spoken in enumerate([10.0, 20.0, 15.0], 1):
        db.add_scene(vid, i, text=f"Scene {i}.", image_prompt="x", duration_s=spoken)
    # sentence starts: scene 1 at 0, scene 2 at 10.45, scene 3 at 30.9 (450ms gaps); the
    # track runs 46.35s, which is 45 spoken plus two gaps.
    (config.output_dir(vid) / "timings.json").write_text(json.dumps([
        {"scene_idx": 1, "sentence_idx": 0, "start_s": 0.0, "end_s": 10.0},
        {"scene_idx": 2, "sentence_idx": 0, "start_s": 10.45, "end_s": 30.45},
        {"scene_idx": 3, "sentence_idx": 0, "start_s": 30.9, "end_s": 45.9},
    ]), encoding="utf-8")

    on_screen = stage.display_durations(vid, db.get_scenes(vid), 46.35)
    check(on_screen == {1: 10.45, 2: 20.45, 3: 15.45},
          f"each scene holds the screen until the next one speaks ({on_screen})")
    check(abs(sum(on_screen.values()) - 46.35) < 1e-6,
          "the scenes add up to the soundtrack exactly, so nothing can drift")
    check(stage.display_durations(vid, db.get_scenes(vid) + [{"idx": 9}], 46.35) is None,
          "a scene timings.json does not cover falls back rather than guessing")


def test_every_image_plays_once_in_its_own_scene():
    print("test_every_image_plays_once_in_its_own_scene")
    items = _fake_scene_items([1, 2, 1, 1, 2, 1, 1, 1], duration_s=33.0)
    shots = stage.plan_shots(items, RENDER_CFG, seed=7)

    images = [img["id"] for _, imgs in items for img in imgs]
    check(len(shots) == len(images), f"one shot per image, no more ({len(shots)})")
    check(sorted(s["image"]["id"] for s in shots) == sorted(images),
          "every generated image is used exactly once")
    strays = [s for s in shots if s["scene"]["idx"] != s["home_scene_idx"]]
    check(not strays, f"no picture is borrowed by another scene ({len(strays)})")

    lengths = {round(s["duration_s"], 3) for s in shots}
    check(lengths == {33.0, 16.5},
          f"a one-image scene holds its picture for the whole scene, a two-image "
          f"scene splits it evenly ({sorted(lengths)})")


def test_the_drift_reverses_on_every_shot():
    print("test_the_drift_reverses_on_every_shot")
    # A 22 second shot has to carry itself on the camera move alone, so neighbouring
    # shots must not slide the same way: that is what reads as monotony.
    items = _fake_scene_items([1] * 9)
    pans = [s["movement"]["pan"] for s in stage.plan_shots(items, RENDER_CFG, seed=7)]
    repeats = [k for k in range(1, len(pans)) if pans[k] == pans[k - 1]]
    check(not repeats, f"the sideways drift reverses on every shot ({pans})")

    moves = [(s["movement"]["zoom"], s["movement"]["pan"])
             for s in stage.plan_shots(_fake_scene_items([1] * 8), RENDER_CFG, seed=7)]
    check(len(set(moves)) == 4, f"all four push/pull/left/right moves get used ({set(moves)})")

    # Deterministic, and not identical from one video to the next.
    again = [s["movement"] for s in stage.plan_shots(items, RENDER_CFG, seed=7)]
    other = [s["movement"] for s in stage.plan_shots(items, RENDER_CFG, seed=8)]
    check(again == [s["movement"] for s in stage.plan_shots(items, RENDER_CFG, seed=7)],
          "the same video id always plans the same camera")
    check(again != other, "a different video id does not open with the same move")


def test_parked_double_exposure_never_repeats_back_to_back():
    print("test_parked_double_exposure_never_repeats_back_to_back")
    # The shape that broke the first cut of video 7: mostly one-image scenes, where
    # a scene's own two exposures are necessarily the same picture.
    items = _fake_scene_items([1, 2, 1, 1, 2, 1, 1, 1])
    shots = stage.plan_shots(items, DOUBLE_CFG, seed=7)

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
    shots = stage.plan_shots(items, DOUBLE_CFG, seed=3)

    per_scene = {}
    for shot in shots:
        idx = shot["scene"]["idx"]
        per_scene[idx] = per_scene.get(idx, 0) + shot["duration_s"]
    worst = max(abs(per_scene[scene["idx"]] - scene["duration_s"]) for scene, _ in items)
    check(worst < 1e-6, f"each scene's shots still add up to its own duration ({worst:.2e}s)")


def test_a_picture_stays_near_its_own_text():
    print("test_a_picture_stays_near_its_own_text")
    items = _fake_scene_items([1, 2, 1, 1, 2, 1, 1, 1])
    shots = stage.plan_shots(items, DOUBLE_CFG, seed=7)
    distances = [abs(s["scene"]["idx"] - s["home_scene_idx"]) for s in shots]
    check(max(distances) <= 1,
          f"no picture plays more than one scene from the words it was drawn for "
          f"({max(distances)})")
    at_home = sum(1 for d in distances if d == 0) / len(distances)
    check(at_home >= 0.4, f"{at_home:.0%} of shots still play inside their own scene")


def test_two_showings_differ_in_framing_and_zoom():
    print("test_two_showings_differ_in_framing_and_zoom")
    items = _fake_scene_items([1, 2, 1, 1])
    shots = stage.plan_shots(items, DOUBLE_CFG, seed=11)
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
    first = stage.plan_shots(items, DOUBLE_CFG, seed=7)
    again = stage.plan_shots(items, DOUBLE_CFG, seed=7)
    other = stage.plan_shots(items, DOUBLE_CFG, seed=8)

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
    fake.durations[str(config.output_dir(vid) / "voice.mp3")] = 20 + 30
    original = _install_fake(fake)
    try:
        check(stage.run(vid, {"render": RENDER_CFG}) is True, "the stage completes")
    finally:
        _restore(original)

    # 2 scenes x 2 images, shown once each = 4 shots.
    check(len(fake.kenburns_calls) == 4, f"4 shots rendered ({len(fake.kenburns_calls)})")
    check(len(fake.concat_calls) == 2, "one concat per scene (2 scenes)")
    check(len(fake.join_calls) == 1, "one join pass across both scene clips")
    check(len(fake.join_calls[0]) == 2, "the join saw exactly the 2 scene clips")

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

    # 40s scene, 2 images, one showing each = 20s per shot.
    sub_durations = {round(d, 3) for _, d, _ in fake.kenburns_calls}
    check(sub_durations == {20.0}, f"each shot got 20s ({sub_durations})")


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


def test_each_scene_fades_in_and_out_once(tmp):
    print("test_each_scene_fades_in_and_out_once")
    # Scene boundaries are made by fading each scene's first and last shot, because an
    # xfade chain over the finished clips was measured at over 1h50m for 43 clips.
    vid = _setup_video(tmp, "fades", [30, 30], images_per_scene=2)
    fake = FakeFFmpeg()
    fake.durations[str(config.output_dir(vid) / "voice.mp3")] = 60
    original = _install_fake(fake)
    try:
        stage.run(vid, {"render": RENDER_CFG})
    finally:
        _restore(original)

    by_scene = {}
    for name, fade_in, fade_out in fake.fades:
        by_scene.setdefault(name[:9], []).append((fade_in, fade_out))
    check(len(by_scene) == 2, f"two scenes were rendered ({sorted(by_scene)})")
    for scene, shots in by_scene.items():
        check(shots[0][0] and not shots[0][1], f"{scene}: the first shot fades in, not out")
        check(shots[-1][1] and not shots[-1][0], f"{scene}: the last shot fades out, not in")
        middle = shots[1:-1]
        check(not any(a or b for a, b in middle),
              f"{scene}: shots inside the scene are hard cuts ({middle})")


def test_the_music_bed_is_built_to_the_voice_and_handed_to_the_mux(tmp):
    print("test_the_music_bed_is_built_to_the_voice_and_handed_to_the_mux")
    vid = _setup_video(tmp, "music", [30], images_per_scene=1)
    fake = FakeFFmpeg()
    fake.durations[str(config.output_dir(vid) / "voice.mp3")] = 30
    original = _install_fake(fake)
    try:
        stage.run(vid, {"render": RENDER_CFG})
    finally:
        _restore(original)
    check(fake.bed_calls == [("fake-track.ogg", 30.0, -15.8)],
          f"the bed is built to the voice's own length and loudness ({fake.bed_calls})")
    check(fake.music_paths and fake.music_paths[0] and "music_bed" in fake.music_paths[0],
          f"and the mux was given it ({fake.music_paths})")

    other = _setup_video(tmp, "nomusic", [30], images_per_scene=1)
    off = FakeFFmpeg()
    off.durations[str(config.output_dir(other) / "voice.mp3")] = 30
    original = _install_fake(off)
    try:
        stage.run(other, {"render": RENDER_CFG, "music": {"enabled": False}})
    finally:
        _restore(original)
    check(off.bed_calls == [] and off.music_paths == [None],
          "switched off in config, no bed is built and the mux gets the voice alone")


def test_existing_scene_clip_is_skipped(tmp):
    print("test_existing_scene_clip_is_skipped")
    vid = _setup_video(tmp, "resume", [20, 20], images_per_scene=1)
    out = config.output_dir(vid)
    clips_dir = out / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    (clips_dir / "scene_001.mp4").write_bytes(b"\x00" * 1000)   # pre-exists

    fake = FakeFFmpeg()
    fake.durations[str(clips_dir / "scene_001.mp4")] = 20.0
    fake.durations[str(out / "voice.mp3")] = 40
    original = _install_fake(fake)
    try:
        stage.run(vid, {"render": RENDER_CFG})
    finally:
        _restore(original)

    # Only scene 2's two slots should have been built; scene 1 was reused untouched.
    check(len(fake.kenburns_calls) == 1,
          f"only the missing scene's shot was rendered ({len(fake.kenburns_calls)})")
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
    fake.durations[str(out / "voice.mp3")] = 60
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

    check(resumed == whole[-len(resumed):] and len(resumed) == 2,
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

    check(len(fake.static_calls) == 1,
          f"the shot fell back to a motionless still ({len(fake.static_calls)})")
    durations = {round(d, 3) for _, d in fake.static_calls}
    check(durations == {24.0},
          f"the fallback holds each sub-shot's exact duration, so the voice track "
          f"stays in sync ({durations})")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        test_scene_lengths_cover_the_silence_between_scenes(tmp)
        test_every_image_plays_once_in_its_own_scene()
        test_the_drift_reverses_on_every_shot()
        test_parked_double_exposure_never_repeats_back_to_back()
        test_shot_lengths_still_fill_each_scene_exactly()
        test_a_picture_stays_near_its_own_text()
        test_two_showings_differ_in_framing_and_zoom()
        test_plan_is_deterministic_in_the_video_id()
        test_full_run_builds_expected_clips(tmp)
        test_per_image_duration_split_evenly(tmp)
        test_contrasting_pairs_rejects_a_lookalike_repeat()
        test_each_scene_fades_in_and_out_once(tmp)
        test_the_music_bed_is_built_to_the_voice_and_handed_to_the_mux(tmp)
        test_existing_scene_clip_is_skipped(tmp)
        test_resumed_run_renders_the_same_shots_it_would_have(tmp)
        test_black_first_frame_raises(tmp)
        test_duration_mismatch_raises(tmp)
        test_no_scenes_raises(tmp)
        test_missing_voice_raises(tmp)
        test_static_fallback_keeps_timing_when_kenburns_fails(tmp)
    print("ALL RENDER TESTS PASSED")
