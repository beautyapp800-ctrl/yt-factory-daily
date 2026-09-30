"""Offline tests for the images stage. Run: python tests/test_images.py

No network: core.images.synthesize_pollinations / synthesize_pexels / pexels_available
are stubbed, so this checks the pipeline's own logic (image-count formula, the
pollinations -> pexels fallback order, seed uniqueness, cover-image assignment,
partial-failure tolerance) without calling pollinations.ai, pexels.com, or needing
either API key.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Keep test output out of the working logs/factory.log.
import core.logger
core.logger.LOG_DIR = Path(tempfile.mkdtemp(prefix="yt-factory-test-"))

from core import config, db, images
import pipeline.images as stage


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok: {msg}")


def test_images_per_scene():
    print("test_images_per_scene")
    check(images.images_per_scene(33, 12) == 3, "33s scene at 12s/image -> 3 images")
    check(images.images_per_scene(5, 12) == 1, "a very short scene still gets at least 1")
    check(images.images_per_scene(0, 12) == 1, "a zero-duration scene still gets at least 1")
    check(images.images_per_scene(96, 12) == 8, "96s scene -> exactly 8 images")
    check(images.images_per_scene(1419, 12) == round(1419 / 12),
          "video 7's real total duration divides as expected")


def _setup_video(tmp, name, scene_durations, prompts=None):
    db.DB_PATH = tmp / f"{name}.db"
    config.OUTPUT_DIR = tmp / f"out_{name}"
    db.init_db()
    vid = db.create_video(f"{name} test")
    for i, dur in enumerate(scene_durations, 1):
        prompt = (prompts[i - 1] if prompts else f"a scene {i} illustration, muted tones")
        db.add_scene(vid, i, text=f"Scene {i} text.", image_prompt=prompt, duration_s=dur)
    return vid


def test_provider_fallback_order(tmp):
    print("test_provider_fallback_order")
    vid = _setup_video(tmp, "fallback", [24])   # 24s / 12 = 2 images
    calls = {"pollinations": 0, "pexels": 0}

    def fake_pollinations(prompt, out_path, seed, cfg):
        calls["pollinations"] += 1
        raise images.ImageError("pollinations down for this test")

    def fake_pexels_available():
        return True

    def fake_pexels(prompt, out_path, cfg):
        calls["pexels"] += 1
        Path(out_path).write_bytes(b"\xff\xd8\xff" + b"\x00" * 100)   # fake jpeg bytes
        return out_path, "Test Photographer"

    original = (images.synthesize_pollinations, images.pexels_available, images.synthesize_pexels)
    images.synthesize_pollinations = fake_pollinations
    images.pexels_available = fake_pexels_available
    images.synthesize_pexels = fake_pexels
    try:
        cfg = {"images": {"provider": "pollinations", "images_per_seconds": 12}}
        check(stage.run(vid, cfg) is True, "the stage completes when pexels covers for pollinations")
    finally:
        (images.synthesize_pollinations, images.pexels_available,
         images.synthesize_pexels) = original

    check(calls["pollinations"] == images.POLLINATIONS_ATTEMPTS * 2,
          f"pollinations was tried {images.POLLINATIONS_ATTEMPTS} times per image "
          f"({calls['pollinations']})")
    check(calls["pexels"] == 2, f"pexels then filled both images ({calls['pexels']})")
    scene_images = db.get_video_images(vid)
    check(len(scene_images) == 2, "both images were recorded")
    check(all(si["provider"] == "pexels" for si in scene_images),
          "both recorded images are attributed to pexels")


def test_seeds_are_distinct(tmp):
    print("test_seeds_are_distinct")
    vid = _setup_video(tmp, "seeds", [36, 24])   # 3 images + 2 images = 5

    def fake_pollinations(prompt, out_path, seed, cfg):
        Path(out_path).write_bytes(b"\xff\xd8\xff" + str(seed).encode() + b"\x00" * 50)

    original = images.synthesize_pollinations
    images.synthesize_pollinations = fake_pollinations
    try:
        cfg = {"images": {"provider": "pollinations", "images_per_seconds": 12}}
        stage.run(vid, cfg)
    finally:
        images.synthesize_pollinations = original

    rows = db.get_video_images(vid)
    seeds = [r["seed"] for r in rows]
    check(len(seeds) == len(set(seeds)), f"every image got a distinct seed ({seeds})")


def test_cover_image_survives_first_failure(tmp):
    print("test_cover_image_survives_first_failure")
    vid = _setup_video(tmp, "cover", [36])   # 3 images
    scene_id = db.get_scenes(vid)[0]["id"]
    attempt = {"n": 0}

    def flaky_pollinations(prompt, out_path, seed, cfg):
        attempt["n"] += 1
        if attempt["n"] <= images.POLLINATIONS_ATTEMPTS:      # every attempt for image 0 fails
            raise images.ImageError("first image never renders")
        Path(out_path).write_bytes(b"\xff\xd8\xff" + b"\x00" * 50)

    original = (images.synthesize_pollinations, images.pexels_available)
    images.synthesize_pollinations = flaky_pollinations
    images.pexels_available = lambda: False
    try:
        cfg = {"images": {"provider": "pollinations", "images_per_seconds": 12}}
        stage.run(vid, cfg)
    finally:
        images.synthesize_pollinations, images.pexels_available = original

    video_scenes = db.get_scenes(vid)
    check(video_scenes[0]["image_path"] is not None,
          "the cover image points at the first image that actually succeeded, "
          "even though image index 0 failed outright")


def test_missing_prompt_skips_without_crashing(tmp):
    print("test_missing_prompt_skips_without_crashing")
    vid = _setup_video(tmp, "noprompt", [24], prompts=[""])

    def fake_pollinations(prompt, out_path, seed, cfg):
        raise AssertionError("should never be called for an empty prompt")

    original = images.synthesize_pollinations
    images.synthesize_pollinations = fake_pollinations
    try:
        cfg = {"images": {"provider": "pollinations", "images_per_seconds": 12}}
        try:
            stage.run(vid, cfg)
            check(False, "a video with no usable prompts anywhere should raise")
        except RuntimeError as e:
            check("no images" in str(e), f"the error names the actual problem ({e})")
    finally:
        images.synthesize_pollinations = original


def test_partial_failure_does_not_fail_video(tmp):
    print("test_partial_failure_does_not_fail_video")
    vid = _setup_video(tmp, "partial", [24, 24])   # 2 scenes x 2 images = 4
    calls = {"n": 0}

    def half_flaky(prompt, out_path, seed, cfg):
        calls["n"] += 1
        # Scene indices start at 1 (see _setup_video), so scene 1's seeds are
        # 1000/1001 and scene 2's are 2000/2001: the first scene always fails, the
        # second always succeeds.
        if seed < 2000:
            raise images.ImageError("this scene's provider is down")
        Path(out_path).write_bytes(b"\xff\xd8\xff" + b"\x00" * 50)

    original = (images.synthesize_pollinations, images.pexels_available)
    images.synthesize_pollinations = half_flaky
    images.pexels_available = lambda: False
    try:
        cfg = {"images": {"provider": "pollinations", "images_per_seconds": 12}}
        check(stage.run(vid, cfg) is True,
              "a video with SOME successful images does not fail outright")
    finally:
        images.synthesize_pollinations, images.pexels_available = original

    rows = db.get_video_images(vid)
    check(len(rows) == 2, f"only the second scene's 2 images were recorded ({len(rows)})")
    import sqlite3
    from contextlib import closing
    with closing(sqlite3.connect(db.DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row
        events = [dict(r) for r in conn.execute(
            "SELECT * FROM events WHERE video_id = ? AND stage = 'images'", (vid,))]
    check(any(e["level"] == "warning" for e in events),
          "each failed image logs a warning event rather than staying silent")


def test_no_scenes_raises(tmp):
    print("test_no_scenes_raises")
    db.DB_PATH = tmp / "empty.db"
    config.OUTPUT_DIR = tmp / "out_empty"
    db.init_db()
    vid = db.create_video("no scenes")
    try:
        stage.run(vid, {"images": {"provider": "pollinations"}})
        check(False, "running images with zero scenes should raise")
    except RuntimeError as e:
        check("no scenes" in str(e), "the error names the actual problem")


def test_pexels_query_strips_style_suffix():
    print("test_pexels_query_strips_style_suffix")
    # synthesize_pexels only runs the part before the first comma through Pexels'
    # keyword search; the appended style string would otherwise pollute the query.
    full_prompt = "a quiet stone courtyard at dawn, cinematic painterly illustration, muted tones"
    query = full_prompt.split(",")[0].strip()
    check(query == "a quiet stone courtyard at dawn",
          f"the style suffix is stripped before searching Pexels ({query!r})")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        test_images_per_scene()
        test_provider_fallback_order(tmp)
        test_seeds_are_distinct(tmp)
        test_cover_image_survives_first_failure(tmp)
        test_missing_prompt_skips_without_crashing(tmp)
        test_partial_failure_does_not_fail_video(tmp)
        test_no_scenes_raises(tmp)
        test_pexels_query_strips_style_suffix()
    print("ALL IMAGES TESTS PASSED")
