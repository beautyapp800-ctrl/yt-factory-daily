"""Stage 5: assemble scene images and the mastered voice track into one video.

Scope, deliberately minimal: no parallax, no subtitles, no music, no intro/outro -
that is stage 6. The goal here is a watchable MP4 that proves the pipeline reaches
the end, not a finished product.

Double exposure (required, see README): each generated image is shown twice, with a
different framing and a different zoom direction each time. This is not a visual
flourish; it is how a 30 minute video's image budget fits Cloudflare's free daily
Neuron allocation at all (see README's "Обов'язкова вимога" section) - half as many
generated images, the same number of visible cuts.

Spreading the repeats: the two showings used to sit back to back, which read as the
same picture held for half a minute. They are now planned across a block of
neighbouring scenes, so the order runs A B C A D B - every repeat has 2 to 4 other
shots in front of it. The plan is seeded from the video id, so it is reproducible and
a resumed run continues the same video rather than inventing a new one.

The cost, stated plainly: a scene carries one image per 22 seconds of narration, so
most scenes have one or two. There is no way to put other shots between an image's
two showings using only that scene's own images - a one-image scene has nothing else
to show. Spreading them therefore lets a picture drift up to a scene away from the
words it was drawn for. plan_shots keeps blocks as short as the gap allows and logs
how far pictures actually moved.

Resumable per scene: output/<id>/clips/scene_XXX.mp4 is skipped if it already
exists, so a failure partway through does not lose earlier scenes and a re-run
picks up where it stopped. Because the whole shot plan is computed up front from the
video id, a resumed run reproduces exactly the plan of the run it resumes.
"""
import random
import time
from pathlib import Path

from core import db, render
from core.config import output_dir
from core.logger import get_logger

log = get_logger("render")

MIN_BRIGHTNESS = 8          # mean luminance (0-255) below this reads as a black frame
DURATION_TOLERANCE_S = 1.0
MIN_BYTES_PER_SECOND = 20_000       # ~160 kbps floor - catches a near-empty/broken file
MAX_BYTES_PER_SECOND = 3_000_000    # ~24 Mbps ceiling - catches a runaway encode

DEFAULT_MIN_BLOCK_IMAGES = 4
DEFAULT_GAP_SHOTS = [2, 4]          # other shots between an image's two showings


def _blocks(scene_items, min_images):
    """Consecutive scenes grouped until a group holds enough images to shuffle.

    Each item is (scene_row, image_rows). A group closes as soon as it holds
    min_images pictures, so groups stay as short as the requested gap allows -
    typically two or three scenes - and a picture never travels far from its own
    text. A short tail joins the previous group rather than forming a group too
    small to interleave at all.
    """
    blocks, current, count = [], [], 0
    for item in scene_items:
        current.append(item)
        count += len(item[1])
        if count >= min_images:
            blocks.append(current)
            current, count = [], 0
    if current:
        if blocks:
            blocks[-1].extend(current)
        else:
            blocks.append(current)
    return blocks


def _interleave(n_images, exposures, gaps):
    """The order images appear in across one block's shot slots.

    Returns a list of image indexes, length n_images * exposures. Every image's first
    showing comes in scene order; each repeat is held back until gaps[image] other
    shots have gone by. The result reads A B C A D B rather than A A B B C C.

    An image is never placed directly after itself, including at the tail of a block
    where the queue has to drain faster than the gaps asked for.
    """
    total = n_images * exposures
    order, pending, next_new = [], [], 0
    while len(order) < total:
        ready = next((k for k, p in enumerate(pending) if p[0] <= len(order)), None)
        if ready is not None:
            _, img, left = pending.pop(ready)
        elif next_new < n_images:
            img, left, next_new = next_new, exposures, next_new + 1
        else:
            # Nothing is due yet and there are no first showings left: the tail of
            # the block. Take whichever repeat has waited longest, skipping the
            # picture just shown so draining cannot undo the point of this function.
            pending.sort()
            k = next((k for k, p in enumerate(pending) if p[1] != order[-1]), 0)
            _, img, left = pending.pop(k)
        order.append(img)
        left -= 1
        if left > 0:
            pending.append([len(order) - 1 + gaps[img] + 1, img, left])
    return _unstick(order)


def _unstick(order):
    """Swap apart any image that still ended up next to itself.

    The queue in _interleave avoids this while it has a choice, but it can run out of
    one at the very last slot of a block: if the only repeat still owing is the
    picture just shown, it has nowhere else to go. Swapping it with some earlier slot
    costs that slot nothing - the gaps are already approximate - and keeps the one
    rule that matters absolutely, which is that a picture never follows itself.

    A swap is kept only if it creates no new adjacency of its own. A block holding a
    single picture has no valid swap at all; there the duplicate stays, because two
    showings of one image is all there is to work with.
    """
    def collides(k):
        return ((k > 0 and order[k] == order[k - 1]) or
                (k + 1 < len(order) and order[k] == order[k + 1]))

    for k in range(1, len(order)):
        if order[k] != order[k - 1]:
            continue
        for j in range(len(order)):
            if j == k or order[j] == order[k]:
                continue
            order[j], order[k] = order[k], order[j]
            if not collides(j) and not collides(k):
                break
            order[j], order[k] = order[k], order[j]
    return order


def plan_shots(scene_items, cfg_render, seed):
    """Decide, for the whole video, which image fills every shot slot, how long it is
    held and which way the camera moves. Returns shot dicts in playback order.

    Slot lengths come from the scene the slot belongs to: a scene keeps exactly its
    own number of slots, each an equal share of its own narration. Changing which
    picture fills a slot therefore cannot move a scene boundary and cannot drift the
    video out of sync with the voice track.

    Deterministic in `seed` (the video id): the same video always plans the same way,
    so a resumed run continues the plan it is resuming instead of mixing two.
    """
    exposures = cfg_render["exposures_per_image"]
    min_images = cfg_render.get("shuffle_min_block_images", DEFAULT_MIN_BLOCK_IMAGES)
    lo, hi = cfg_render.get("shuffle_gap_shots", DEFAULT_GAP_SHOTS)
    lo, hi = max(1, lo), max(1, hi)
    pairs = render.contrasting_pairs(cfg_render["movements"])

    shots = []
    for block_i, block in enumerate(_blocks(scene_items, min_images)):
        rng = random.Random(f"{seed}:{block_i}")
        images = [(scene, img) for scene, imgs in block for img in imgs]
        slots = []
        for scene, imgs in block:
            count = len(imgs) * exposures
            per_slot = (scene["duration_s"] or 0) / count
            slots.extend((scene, per_slot) for _ in range(count))

        gaps = [rng.randint(lo, hi) for _ in images]
        moves = [pairs[rng.randrange(len(pairs))] for _ in images]
        shown = {}
        for slot_i, img_i in enumerate(_interleave(len(images), exposures, gaps)):
            scene, per_slot = slots[slot_i]
            exposure = shown.get(img_i, 0)
            shown[img_i] = exposure + 1
            home_scene, image_row = images[img_i]
            shots.append({
                "scene": scene,
                "image": image_row,
                "home_scene_idx": home_scene["idx"],
                "movement": moves[img_i][exposure % len(moves[img_i])],
                "duration_s": per_slot,
                "exposure": exposure,
            })
    return shots


def _build_scene_clip(scene, shots, cfg_render, parts_dir, cache_dir):
    """Render one scene's shots as sub-clips and return their paths, in order."""
    sub_clips = []
    for slot, shot in enumerate(shots):
        image_path = Path(shot["image"]["path"])
        sub_path = parts_dir / f"scene_{scene['idx']:03d}_slot{slot:02d}.mp4"
        if not sub_path.exists():
            try:
                render.kenburns_clip(image_path, sub_path, shot["duration_s"],
                                     shot["movement"], cfg_render, cache_dir)
            except render.RenderError as e_render:
                # Ken Burns timed out every attempt. Hold the frame still for the
                # same duration rather than dropping the shot: a gap here would
                # desync everything after it from the voice track.
                log.warning("scene %d shot %d: %s; falling back to a motionless shot",
                            scene["idx"], slot, e_render)
                render.static_clip(image_path, sub_path, shot["duration_s"], cfg_render,
                                   cache_dir)
        sub_clips.append(sub_path)
    return sub_clips


def run(video_id, cfg):
    scenes = db.get_scenes(video_id)
    if not scenes:
        raise RuntimeError("no scenes to render; did the script stage run?")

    cfg_render = cfg["render"]
    out_dir = output_dir(video_id)
    voice_path = out_dir / "voice.mp3"
    if not voice_path.exists():
        raise RuntimeError(f"{voice_path} is missing; did the tts stage run?")

    render.require_ffmpeg()
    t0 = time.time()

    clips_dir = out_dir / "clips"
    parts_dir = clips_dir / "_parts"
    cache_dir = clips_dir / "_cache"
    parts_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    scene_items = []
    for scene in scenes:
        images_rows = [row for row in db.get_scene_images(scene["id"])
                       if row["path"] and Path(row["path"]).exists()]
        if not images_rows:
            log.warning("scene %d has no usable image file, skipping it entirely",
                        scene["idx"])
            continue
        scene_items.append((scene, images_rows))
    if not scene_items:
        raise RuntimeError("no scene has a usable image; did the images stage run?")

    shots = plan_shots(scene_items, cfg_render, video_id)
    by_scene = {}
    for shot in shots:
        by_scene.setdefault(shot["scene"]["idx"], []).append(shot)

    moved = [abs(s["scene"]["idx"] - s["home_scene_idx"]) for s in shots]
    log.info("%d shots planned over %d scenes: %.0f%% play inside their own scene, "
             "the rest at most %d scene(s) away",
             len(shots), len(scene_items),
             100 * sum(1 for d in moved if d == 0) / len(moved), max(moved))

    scene_clip_paths = []
    built, skipped = 0, 0

    for scene, _ in scene_items:
        scene_clip = clips_dir / f"scene_{scene['idx']:03d}.mp4"
        if scene_clip.exists():
            skipped += 1
            scene_clip_paths.append(scene_clip)
            continue

        scene_t0 = time.time()
        scene_shots = by_scene[scene["idx"]]
        sub_clips = _build_scene_clip(scene, scene_shots, cfg_render, parts_dir, cache_dir)
        render.concat_clips(sub_clips, scene_clip)
        scene_clip_paths.append(scene_clip)
        built += 1
        log.info("scene %d/%d clip built in %.1fs (%d shots)", scene["idx"], len(scenes),
                 time.time() - scene_t0, len(scene_shots))

    log.info("%d scene clips built, %d already on disk and reused", built, skipped)

    log.info("cross-fading %d scene clips (%.1fs each)", len(scene_clip_paths),
             cfg_render["crossfade_s"])
    silent_path = out_dir / "_video_silent.mp4"
    silent_duration = render.xfade_chain(scene_clip_paths, silent_path,
                                         cfg_render["crossfade_s"], cfg_render)
    log.info("silent video: %.1f min", silent_duration / 60)

    final_path = out_dir / "final.mp4"
    render.mux_audio(silent_path, voice_path, final_path, cfg_render)
    silent_path.unlink(missing_ok=True)

    # --- checks -------------------------------------------------------------
    final_duration = render.probe_duration(final_path)
    audio_duration = render.probe_duration(voice_path)
    if abs(final_duration - audio_duration) > DURATION_TOLERANCE_S:
        raise RuntimeError(f"final.mp4 is {final_duration:.1f}s but voice.mp3 is "
                           f"{audio_duration:.1f}s, more than {DURATION_TOLERANCE_S}s apart")
    log.info("duration check: video %.1fs vs audio %.1fs (within %.0fs)",
             final_duration, audio_duration, DURATION_TOLERANCE_S)

    first_brightness = render.frame_brightness(final_path, 0.1)
    last_brightness = render.frame_brightness(final_path, max(0.1, final_duration - 0.5))
    if first_brightness < MIN_BRIGHTNESS:
        raise RuntimeError(f"final.mp4's first frame is black (mean brightness "
                           f"{first_brightness:.1f})")
    if last_brightness < MIN_BRIGHTNESS:
        raise RuntimeError(f"final.mp4's last frame is black (mean brightness "
                           f"{last_brightness:.1f})")
    log.info("brightness check: first frame %.1f, last frame %.1f (floor %d)",
             first_brightness, last_brightness, MIN_BRIGHTNESS)

    size_bytes = final_path.stat().st_size
    bytes_per_second = size_bytes / final_duration if final_duration else 0
    if not MIN_BYTES_PER_SECOND <= bytes_per_second <= MAX_BYTES_PER_SECOND:
        log.warning("final.mp4 is %.1f MB over %.1f min (%.0f bytes/s), outside the "
                    "expected %d-%d bytes/s range - worth a look, not failing the video",
                    size_bytes / 1e6, final_duration / 60, bytes_per_second,
                    MIN_BYTES_PER_SECOND, MAX_BYTES_PER_SECOND)
        db.log_event(video_id, "render", "warning",
                     f"file size {size_bytes / 1e6:.1f}MB looks unusual for "
                     f"{final_duration / 60:.1f} min of video")

    preview_path = out_dir / f"preview_{cfg_render['preview_seconds']}s.mp4"
    render.extract_preview(final_path, preview_path, cfg_render["preview_seconds"], cfg_render)

    db.update_video(video_id, video_path=str(final_path), duration_s=round(final_duration, 1))

    elapsed_min = (time.time() - t0) / 60
    log.info("done: %.1f min render time, %.1f MB, %.1f min final video -> %s",
             elapsed_min, size_bytes / 1e6, final_duration / 60, final_path)
    db.log_event(video_id, "render", "info",
                 f"{elapsed_min:.1f} min render, {size_bytes / 1e6:.1f} MB, "
                 f"{final_duration / 60:.1f} min video, {built} scenes rendered fresh, "
                 f"{skipped} reused")
    return True
