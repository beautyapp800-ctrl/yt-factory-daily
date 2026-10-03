"""Stage 5: assemble scene images and the mastered voice track into one video.

Scope, deliberately minimal: no parallax, no subtitles, no music, no intro/outro -
that is stage 6. The goal here is a watchable MP4 that proves the pipeline reaches
the end, not a finished product.

One showing per image, inside its own scene (config.render.double_exposure=false).
Every picture plays under the words it was drawn for; nothing is borrowed from a
neighbouring scene and no picture is ever repeated.

This replaced a double-exposure scheme, where each image was shown twice and the two
showings were spread across a block of neighbouring scenes to keep them apart. That
doubled the rate of visible cuts for free, but 44% of shots ended up playing a scene
away from their own text, which is too high a price for a channel whose pictures are
supposed to illustrate the script. The code is still here behind the flag.

What it costs, stated plainly: an image is generated for every 22 seconds of
narration, so a shot now lasts around 22 seconds instead of 11. The camera move
therefore has to carry a long take - a wider zoom range plus a sideways drift, see
core.render.kenburns_clip. The image budget is unchanged either way: the images stage
plans round(duration / images_per_seconds) pictures per scene and never knew how many
times each would be shown.

Resumable per scene: output/<id>/clips/scene_XXX.mp4 is skipped if it already
exists, so a failure partway through does not lose earlier scenes and a re-run
picks up where it stopped. Because the whole shot plan is computed up front from the
video id, a resumed run reproduces exactly the plan of the run it resumes.
"""
import json
import random
import time
from pathlib import Path

from core import db, music, render
from core.config import output_dir
from core.logger import get_logger

log = get_logger("render")

MIN_BRIGHTNESS = 8          # mean luminance (0-255) below this reads as a black frame
DURATION_TOLERANCE_S = 1.0
MIN_BYTES_PER_SECOND = 20_000       # ~160 kbps floor - catches a near-empty/broken file
MAX_BYTES_PER_SECOND = 3_000_000    # ~24 Mbps ceiling - catches a runaway encode

DEFAULT_MIN_BLOCK_IMAGES = 4
DEFAULT_GAP_SHOTS = [2, 4]          # other shots between an image's two showings

# Camera moves for the single-showing scheme: a push or a pull that drifts sideways
# as it goes. Cycled in order, and the order matters - the drift reverses on every
# single step, so no two shots in a row slide the same way, while the four entries
# still cover both zoom directions crossed with both drifts.
DEFAULT_PAN_MOVEMENTS = [
    {"zoom": "in", "pan": "right"},
    {"zoom": "out", "pan": "left"},
    {"zoom": "out", "pan": "right"},
    {"zoom": "in", "pan": "left"},
]


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
    """Decide, for the whole video, which image fills every shot, how long it is held
    and which way the camera moves. Returns shot dicts in playback order.

    Two schemes, picked by config.render.double_exposure:

    False (the default): one showing per image, held for its share of its own scene.
    Every picture plays under the words it was drawn for, nothing is borrowed from a
    neighbouring scene, and a repeat cannot happen because there are none. A shot
    then lasts around 22 seconds, so the camera move has to carry it - see
    config.render.pan_movements and core.render.kenburns_clip.

    True: each image is shown twice, spread across neighbouring scenes. Kept because
    it doubles the rate of visible cuts at no extra generation cost, and may be worth
    revisiting; switched off because the price is too high - 44% of shots played a
    scene away from their own text, which undoes the work of making the pictures
    match the script in the first place.
    """
    if cfg_render.get("double_exposure", False):
        return _plan_double_exposure(scene_items, cfg_render, seed)
    return _plan_single(scene_items, cfg_render, seed)


def _plan_single(scene_items, cfg_render, seed):
    """One showing per image, always inside its own scene, in scene order.

    Camera moves cycle through config.render.pan_movements, offset by the video id so
    consecutive videos do not all open with the same move. Deterministic either way:
    the same video always plans the same camera.
    """
    movements = cfg_render.get("pan_movements") or DEFAULT_PAN_MOVEMENTS
    shots = []
    for scene, images_rows in scene_items:
        per_shot = (scene["duration_s"] or 0) / len(images_rows)
        for image_row in images_rows:
            shots.append({
                "scene": scene,
                "image": image_row,
                "home_scene_idx": scene["idx"],
                "movement": movements[(len(shots) + seed) % len(movements)],
                "duration_s": per_shot,
                "exposure": 0,
            })
    return shots


def _plan_double_exposure(scene_items, cfg_render, seed):
    """Two showings per image, spread 2-4 shots apart across a block of scenes.

    Slot lengths come from the scene the slot belongs to: a scene keeps exactly its
    own number of slots, each an equal share of its own narration. Changing which
    picture fills a slot therefore cannot move a scene boundary and cannot drift the
    video out of sync with the voice track.

    Deterministic in `seed` (the video id): the same video always plans the same way,
    so a resumed run continues the plan it is resuming instead of mixing two.

    Off by default - see plan_shots for why it is kept rather than deleted.
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
        # The scene fades in on its first shot and out on its last, which is what makes the
        # boundary between scenes; the joins inside a scene are hard cuts.
        fades = {"fade_in": slot == 0, "fade_out": slot == len(shots) - 1}
        if not sub_path.exists():
            try:
                render.kenburns_clip(image_path, sub_path, shot["duration_s"],
                                     shot["movement"], cfg_render, cache_dir, **fades)
            except render.RenderError as e_render:
                # Ken Burns timed out every attempt. Hold the frame still for the
                # same duration rather than dropping the shot: a gap here would
                # desync everything after it from the voice track.
                log.warning("scene %d shot %d: %s; falling back to a motionless shot",
                            scene["idx"], slot, e_render)
                render.static_clip(image_path, sub_path, shot["duration_s"], cfg_render,
                                   cache_dir, **fades)
        sub_clips.append(sub_path)
    return sub_clips


def display_durations(video_id, scenes, voice_seconds):
    """How long each scene is on screen, which is NOT how long it is spoken.

    The tts stage puts silence between scenes - 450ms, or 900ms where a lesson begins - and
    that silence is in voice.mp3 but belongs to no scene's own duration. A video built from
    the spoken durations is therefore shorter than its soundtrack, and every scene after the
    first drifts further from the words: measured on video 7, 20.7s of drift over 43 scenes.

    So a scene holds the screen from its own first sentence until the next scene's first
    sentence, and the last one until the track ends. The picture then changes exactly when
    the narration moves on, and the parts add up to the soundtrack by construction.

    Returns {scene idx: seconds}, or None if timings.json cannot answer, in which case the
    caller falls back to the spoken durations and the stage's duration check catches it.
    """
    path = output_dir(video_id) / "timings.json"
    if not path.exists():
        log.warning("no timings.json: scene lengths fall back to spoken duration, which "
                    "leaves the gaps between scenes out of the video")
        return None
    starts = {}
    for entry in json.loads(path.read_text(encoding="utf-8")):
        starts.setdefault(entry["scene_idx"], entry["start_s"])
    if not all(scene["idx"] in starts for scene in scenes):
        log.warning("timings.json does not cover every scene: scene lengths fall back to "
                    "spoken duration")
        return None
    ordered = sorted(starts)
    out = {}
    for i, idx in enumerate(ordered):
        end = starts[ordered[i + 1]] if i + 1 < len(ordered) else voice_seconds
        out[idx] = round(end - starts[idx], 3)
    return out


def _music_bed(video_id, cfg, voice_path, out_dir):
    """The looped, levelled music bed for this video, or (None, reason) when there is none.

    Never fatal: a video without music is a video; a failed render is not. Every way this
    can come to nothing is logged with what to do about it.
    """
    settings = music.settings(cfg)
    if not settings["enabled"]:
        return None, "music is switched off in config.music.enabled"
    tracks = music.available_tracks()
    if not tracks:
        log.warning("no music in %s, rendering without a bed", music.MUSIC_DIR)
        return None, "no tracks in assets/music"
    track = music.pick_track(video_id, tracks)
    try:
        voice_lufs = music.loudness(voice_path)
        bed, details = music.build_bed(track, render.probe_duration(voice_path),
                                       out_dir / "music_bed.m4a", cfg, voice_lufs)
    except music.MusicError as e:
        log.warning("could not build a music bed (%s), rendering without one", e)
        db.log_event(video_id, "render", "warning", f"no music bed: {e}")
        return None, str(e)
    details["voice_lufs"] = round(voice_lufs, 1)
    return bed, details


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

    on_screen = display_durations(video_id, scenes, render.probe_duration(voice_path))
    if on_screen:
        spoken = sum(s["duration_s"] or 0 for s, _ in scene_items)
        scene_items = [({**dict(scene), "duration_s": on_screen[scene["idx"]]}, images)
                       for scene, images in scene_items]
        log.info("scene lengths taken from timings.json: %.1fs on screen against %.1fs "
                 "spoken, the difference being the silence between scenes",
                 sum(s["duration_s"] for s, _ in scene_items), spoken)

    shots = plan_shots(scene_items, cfg_render, video_id)
    by_scene = {}
    for shot in shots:
        by_scene.setdefault(shot["scene"]["idx"], []).append(shot)

    moved = [abs(s["scene"]["idx"] - s["home_scene_idx"]) for s in shots]
    lengths = [s["duration_s"] for s in shots]
    if max(moved) == 0:
        log.info("%d shots planned over %d scenes, every picture inside its own "
                 "scene, %.0f-%.0fs each", len(shots), len(scene_items),
                 min(lengths), max(lengths))
    else:
        log.info("%d shots planned over %d scenes: %.0f%% play inside their own "
                 "scene, the rest at most %d scene(s) away",
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

    log.info("joining %d scene clips", len(scene_clip_paths))
    silent_path = out_dir / "_video_silent.mp4"
    silent_duration = render.join_clips(scene_clip_paths, silent_path)
    log.info("silent video: %.1f min", silent_duration / 60)

    bed_path, bed_details = _music_bed(video_id, cfg, voice_path, out_dir)
    final_path = out_dir / "final.mp4"
    render.mux_audio(silent_path, voice_path, final_path, cfg_render, bed_path)
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

    if bed_path:
        measured = music.loudness(final_path)
        gap = bed_details["voice_lufs"] - music.loudness(bed_path)
        log.info("music check: bed at %.1f LUFS is %.1f dB under the voice (asked for %d); "
                 "the finished mix measures %.1f LUFS against the voice's %.1f",
                 music.loudness(bed_path), gap, music.settings(cfg)["gain_below_voice_db"],
                 measured, bed_details["voice_lufs"])
        if measured > bed_details["voice_lufs"] + 1.5:
            log.warning("the mix is %.1f dB louder than the voice alone: the music is not "
                        "sitting under it", measured - bed_details["voice_lufs"])
        (out_dir / "music.json").write_text(json.dumps(bed_details, indent=2), encoding="utf-8")

    db.update_video(video_id, video_path=str(final_path), duration_s=round(final_duration, 1))

    elapsed_min = (time.time() - t0) / 60
    log.info("done: %.1f min render time, %.1f MB, %.1f min final video -> %s",
             elapsed_min, size_bytes / 1e6, final_duration / 60, final_path)
    db.log_event(video_id, "render", "info",
                 f"{elapsed_min:.1f} min render, {size_bytes / 1e6:.1f} MB, "
                 f"{final_duration / 60:.1f} min video, {built} scenes rendered fresh, "
                 f"{skipped} reused")
    return True
