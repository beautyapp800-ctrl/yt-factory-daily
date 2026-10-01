"""Stage 5: assemble scene images and the mastered voice track into one video.

Scope, deliberately minimal: no parallax, no subtitles, no music, no intro/outro -
that is stage 6. The goal here is a watchable MP4 that proves the pipeline reaches
the end, not a finished product.

Double exposure (required, see README): each generated image is shown twice, back
to back, with a different Ken Burns camera movement each time - a push toward the
left third, then the right, alternating push-in and pull-out across the video so it
does not read as repetitive. This is not a visual flourish; it is how a 30 minute
video's image budget fits Cloudflare's free daily Neuron allocation at all (see
README's "Обов'язкова вимога" section) - half as many generated images, the same
number of visible cuts.

Resumable per scene: output/<id>/clips/scene_XXX.mp4 is skipped if it already
exists, so a failure partway through does not lose earlier scenes and a re-run
picks up where it stopped. Crossfades (xfade, 0.6s by default) happen only at
scene boundaries, in one chained ffmpeg pass across however many scene clips exist -
not between the two exposures of the same image, which hard-cut.
"""
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


def _movement_for(global_index, movements):
    return movements[global_index % len(movements)]


def _build_scene_clip(scene, images_rows, cfg_render, parts_dir, cache_dir, exposure_counter):
    """One scene's clip: every image in the scene shown exposures_per_image times
    with alternating camera movement, hard-cut concatenated. Returns the clip path.
    """
    exposures = cfg_render["exposures_per_image"]
    movements = cfg_render["movements"]
    scene_duration = scene["duration_s"] or 0
    per_image = scene_duration / max(1, len(images_rows))
    per_exposure = per_image / exposures

    sub_clips = []
    for img_idx, image_row in enumerate(images_rows):
        image_path = Path(image_row["path"])
        if not image_path.exists():
            log.warning("scene %d image %d: %s is missing, skipping that image",
                        scene["idx"], img_idx, image_path)
            continue
        for e in range(exposures):
            movement = _movement_for(exposure_counter["n"], movements)
            exposure_counter["n"] += 1
            sub_path = parts_dir / f"scene_{scene['idx']:03d}_img{img_idx:02d}_exp{e}.mp4"
            if not sub_path.exists():
                try:
                    render.kenburns_clip(image_path, sub_path, per_exposure, movement,
                                         cfg_render, cache_dir)
                except render.RenderError as e_render:
                    # Ken Burns timed out every attempt. Hold the frame still for the
                    # same duration rather than dropping the shot: a gap here would
                    # desync everything after it from the voice track.
                    log.warning("scene %d image %d exposure %d: %s; falling back to a "
                                "motionless shot", scene["idx"], img_idx, e, e_render)
                    render.static_clip(image_path, sub_path, per_exposure, cfg_render,
                                       cache_dir)
            sub_clips.append(sub_path)

    if not sub_clips:
        raise RuntimeError(f"scene {scene['idx']} has no usable images to render")
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

    scene_clip_paths = []
    exposure_counter = {"n": 0}
    built, skipped = 0, 0

    for scene in scenes:
        scene_clip = clips_dir / f"scene_{scene['idx']:03d}.mp4"
        if scene_clip.exists():
            skipped += 1
            scene_clip_paths.append(scene_clip)
            # Keep the global movement cycle in step with what a fresh run would
            # have produced, so a resumed run's untouched scenes do not throw off
            # the alternation for the scenes still to come.
            images_rows = db.get_scene_images(scene["id"])
            exposure_counter["n"] += len(images_rows) * cfg_render["exposures_per_image"]
            continue

        images_rows = db.get_scene_images(scene["id"])
        if not images_rows:
            log.warning("scene %d has no images at all, skipping it entirely", scene["idx"])
            continue

        scene_t0 = time.time()
        sub_clips = _build_scene_clip(scene, images_rows, cfg_render, parts_dir,
                                      cache_dir, exposure_counter)
        render.concat_clips(sub_clips, scene_clip)
        scene_clip_paths.append(scene_clip)
        built += 1
        log.info("scene %d/%d clip built in %.1fs (%d images)", scene["idx"], len(scenes),
                 time.time() - scene_t0, len(images_rows))

    if not scene_clip_paths:
        raise RuntimeError("no scene clips were produced for this video")
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
