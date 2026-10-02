"""Write fresh image prompts for a video that already has its scenes.

    python scripts/regen_image_prompts.py 7

The prompts were first written by the script stage under the old IMAGE_SYSTEM, which
banned every text-bearing noun and produced interchangeable scenery. The rules have
since changed (core/prompts.py: name one action or object from the scene's own text,
show it without its writing), so a video generated before that needs its prompts asked
for again. Only the image prompts are redone - the script, scenes and audio are not
touched.

The old prompts are saved to output/<id>/image_prompts_before.json first, so this can be
undone, and the run ends by printing each scene's text beside its new prompt: the point of
the change is that a prompt names something the scene actually says, which is a thing to
read, not a thing to assume.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import db
from core.config import load_config, output_dir
from core.logger import get_logger
from core.prompts import find_banned_image_words
from pipeline import script

log = get_logger("regen_prompts")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    video_id = int(sys.argv[1])
    cfg = load_config()

    scenes = db.get_scenes(video_id)
    if not scenes:
        raise SystemExit(f"video {video_id} has no scenes")
    out = output_dir(video_id)
    outline = json.loads((out / "outline.json").read_text(encoding="utf-8"))

    backup = out / "image_prompts_before.json"
    if not backup.exists():
        backup.write_text(json.dumps(
            [{"idx": s["idx"], "image_prompt": s["image_prompt"]} for s in scenes],
            indent=2, ensure_ascii=False), encoding="utf-8")
        log.info("old prompts saved to %s", backup)

    log.info("asking for %d new image prompts (%d per batch)", len(scenes), script.IMAGE_BATCH)
    new_prompts = script.add_image_prompts([s["text"] for s in scenes], cfg, outline)

    flagged = 0
    for scene, prompt in zip(scenes, new_prompts):
        if find_banned_image_words(prompt):
            flagged += 1        # the images stage rewrites these before generating
        db.update_scene(scene["id"], image_prompt=prompt)

    log.info("%d prompts written; %d still name a banned word and will be rewritten by "
             "the images stage", len(new_prompts), flagged)
    for scene, prompt in zip(scenes, new_prompts):
        print(f"\nSCENE {scene['idx']}: {scene['text'][:230]}{'...' if len(scene['text']) > 230 else ''}")
        print(f"  -> {prompt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
