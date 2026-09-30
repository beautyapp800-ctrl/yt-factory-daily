"""Stage 4: illustrate the script.

One still image per scene is too long on screen: video 7 averages 33s a scene, and
channels in this niche change the image every 8-15 seconds. So each scene gets
max(1, round(scene.duration_s / images_per_seconds)) images instead of one, all
sharing the scene's own image_prompt but rendered with a different seed each, so the
same description still produces visibly different frames.

pollinations.ai is tried first per image (core.images.synthesize handles its own
pollinations -> pexels fallback); a single image that still fails after both is
logged and skipped rather than failing the whole video, since a video with a few
scenes short on images is still useful and a total block on one flaky provider is not
worth losing 30+ minutes of already-paid script and voice work over.
"""
import random

from core import db, images
from core.config import output_dir
from core.logger import get_logger

log = get_logger("images")

DEFAULT_IMAGES_PER_SECONDS = 12


def run(video_id, cfg):
    scenes = db.get_scenes(video_id)
    if not scenes:
        raise RuntimeError("no scenes to illustrate; did the script stage run?")

    images_per_seconds = (cfg.get("images") or {}).get(
        "images_per_seconds", DEFAULT_IMAGES_PER_SECONDS)
    provider = (cfg.get("images") or {}).get("provider", "pollinations")
    pexels_note = "pexels available as fallback" if images.pexels_available() \
        else "pexels unavailable (no PEXELS_API_KEY)"
    log.info("provider %s, %s, %.0fs per image", provider, pexels_note, images_per_seconds)

    plan = []
    for scene in scenes:
        duration = scene["duration_s"] or 0
        count = images.images_per_scene(duration, images_per_seconds)
        plan.append((scene, count))
    total_planned = sum(c for _, c in plan)
    log.info("%d scenes -> %d images planned", len(scenes), total_planned)

    image_dir = output_dir(video_id) / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    db.clear_scene_images(video_id)
    made, failed, by_provider = 0, 0, {}

    for scene, count in plan:
        prompt = scene["image_prompt"] or ""
        if not prompt.strip():
            log.warning("scene %d has no image_prompt, skipping its %d image(s)",
                        scene["idx"], count)
            failed += count
            continue

        cover_set = False
        for i in range(count):
            # A fixed seed per (scene, image) so a re-run reproduces the same frame
            # rather than drawing a new one every time.
            seed = scene["idx"] * 1000 + i
            out_path = image_dir / f"scene_{scene['idx']:03d}_{i:02d}.jpg"
            try:
                used = images.synthesize(prompt, out_path, seed, cfg)
            except images.ImageError as e:
                log.warning("scene %d image %d/%d failed: %s", scene["idx"], i + 1, count, e)
                db.log_event(video_id, "images", "warning",
                            f"scene {scene['idx']} image {i + 1}/{count} failed: {e}")
                failed += 1
                continue

            db.add_scene_image(video_id, scene["id"], i, prompt=prompt, seed=seed,
                               provider=used, path=str(out_path))
            by_provider[used] = by_provider.get(used, 0) + 1
            made += 1
            if not cover_set:
                # The scene's own single-image field becomes its cover/first frame,
                # for anything downstream that only wants one representative image.
                # Whichever image succeeds first, not necessarily index 0.
                db.update_scene(scene["id"], image_path=str(out_path))
                cover_set = True

        if made % 10 < count:  # rough progress line, not exact, cheap to compute
            log.info("progress: %d/%d images made so far", made, total_planned)

    log.info("done: %d/%d images made (%s), %d failed", made, total_planned,
             ", ".join(f"{k}={v}" for k, v in by_provider.items()) or "none", failed)
    db.log_event(video_id, "images", "info",
                 f"{made}/{total_planned} images made ({by_provider}), {failed} failed")

    if made == 0:
        raise RuntimeError(f"no images were produced for any of {len(scenes)} scenes")
    return True
