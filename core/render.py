"""Video rendering: ffmpeg wrappers for the Ken Burns effect, scene assembly and
the final mux. No re-encoding of audio content: voice.mp3 is already mastered by
the tts stage and gets attached as-is, never touched.

The zoompan jitter trap: ffmpeg's zoompan filter steps the crop window in whole
source-pixels, so a gentle zoom on a 1920x1080 source forces each frame's crop to
jump by uneven whole-pixel amounts - visible as stutter, not a smooth pan. Fixed by
upscaling the source before zoompan, so each step has more source pixels to round
to - see prescale_image() for the two-step form this actually has to take.

The first version of this fix upscaled correctly but then had zoompan itself
output at that same huge size for every frame of the sub-shot's whole duration,
re-encoded at 4x the final resolution - 16x the pixels of the delivered frame,
for every single frame. Live on video 7, sub-shots that should take 20-40s took
over an hour; one was still running after 67 minutes. The upscale only needs to
happen ONCE per image, as a cached still (prescale_image); zoompan then crops
directly down to the final width x height, every frame, which is both the
correct fix and the fast one.

Measured frame-to-frame pixel-difference variance (same image, same zoom path,
lower is smoother): 36.8% with no upscale at all, 30.7% at upscale_factor=2,
19.7% at upscale_factor=4 (the old, slow architecture), 16.9% at
upscale_factor=3 with the cached-still architecture - which is both the
smoothest measured and, because the upscale is a one-off per image rather than
carried through every frame, by far the fastest.
"""
import subprocess
import time
from pathlib import Path

from core.logger import get_logger

log = get_logger("render")

USER_AGENT = "yt-factory/1.0"   # unused by ffmpeg itself; kept for log message parity

# Where each framing takes its slice from, as a fraction of the source width: the
# region's width, and the left edge of that region. "left" and "right" overlap in the
# middle by 30%, so the two exposures of one image share some content but are plainly
# different shots.
#
# This replaces an earlier attempt to get the same effect by biasing zoompan's x
# expression toward a third of the frame. That could not work: at zoom_max 1.15 the
# crop window is 87% of the image, so it can only slide 6.5% either side of centre
# before hitting the edge, where ffmpeg clamps it. Every "left third" shot was in
# practice a plain centred zoom, which is why both exposures of an image looked
# identical. Cutting the region out of the image itself has no such ceiling.
# (fraction of the axis the region covers, fraction where the region starts)
FRAMINGS_HORIZONTAL = {
    "left":   (0.65, 0.00),
    "right":  (0.65, 0.35),
    "center": (0.80, 0.10),
}
FRAMINGS_VERTICAL = {
    "top":    (0.65, 0.00),
    "bottom": (0.65, 0.35),
    "middle": (0.80, 0.10),
}


class RenderError(Exception):
    pass


_ffmpeg_checked = False


def require_ffmpeg():
    global _ffmpeg_checked
    if _ffmpeg_checked:
        return
    import shutil
    if shutil.which("ffmpeg") is None:
        raise RenderError(
            "ffmpeg was not found on PATH. On Windows: winget install Gyan.FFmpeg, "
            "then open a new terminal so PATH picks it up.")
    _ffmpeg_checked = True


def _frames_reported(stderr):
    """The frame count from ffmpeg's own -stats line, or None if it printed none.

    Worth having permanently: if a sub-shot ever renders far slower than its
    neighbours, this says whether ffmpeg actually processed more frames than the
    clip needs (a d/duration miscalculation generating frames that -t then throws
    away) or exactly the right number (so the time went somewhere else).
    """
    import re
    matches = re.findall(r"frame=\s*(\d+)", stderr or "")
    return int(matches[-1]) if matches else None


def _run_ffmpeg(args, error_context, timeout=None):
    require_ffmpeg()
    result = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-stats"] + args,
                            capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RenderError(f"{error_context}: {result.stderr.strip()[-500:]}")
    return result


def probe_duration(path):
    require_ffmpeg()
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True)
    if result.returncode != 0 or not result.stdout.strip():
        raise RenderError(f"ffprobe could not read duration of {path}: "
                          f"{result.stderr.strip()[-300:]}")
    return float(result.stdout.strip())


def prescale_image(image_path, cache_dir, width, height, upscale_factor):
    """Upscale a still image once to upscale_factor x the final resolution, cached by
    source filename so every exposure of the same image (2 per image, by default)
    reuses this one file instead of re-upscaling per sub-shot.

    This is the fix for a real mistake the first version of this module made:
    scaling up and having zoompan itself output at that same huge size, every frame,
    for the whole sub-shot's duration - 16x the pixels of the final frame, encoded
    repeatedly. Live on video 7, single sub-shots that should have taken 20-40s took
    over an hour. The upscale only needs to happen ONCE per image, as a cached still;
    zoompan should read that and crop straight down to the final delivery size.

    Crops to the target aspect first, and pins SAR to 1. Cloudflare's flux-1-schnell
    takes no width/height at all - its schema is prompt and steps - and returns square
    1024x1024 images whatever the config asks for. Scaling a square straight to
    5760x3240 stretches it, and ffmpeg then quietly sets a compensating 9:16 sample
    aspect ratio, so the finished mp4 was 1920x1080 pixels tagged as 1:1 display
    aspect: players drew it as a square with black bars down both sides. A centred
    crop to 16:9 before scaling gives a true widescreen frame with no bars and no
    distortion; setsar=1 makes sure nothing re-introduces a non-square pixel.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    big_w, big_h = width * upscale_factor, height * upscale_factor
    cache_path = cache_dir / f"{Path(image_path).stem}_{big_w}x{big_h}.png"
    if cache_path.exists():
        return cache_path
    # min(...) on both axes handles a source that is too wide as well as too tall.
    crop = f"crop='min(iw,ih*{width}/{height})':'min(ih,iw*{height}/{width})'"
    _run_ffmpeg(["-i", str(image_path),
                "-vf", f"{crop},scale={big_w}:{big_h}:flags=lanczos,setsar=1",
                "-frames:v", "1", str(cache_path)],
               f"ffmpeg could not pre-scale {image_path}")
    return cache_path


def framing_crop(focus, width, height):
    """The ffmpeg crop that cuts one framing's region out of the pre-scaled still.

    The region is a real slice of the picture - the left 65% of its width, the right
    65%, the top or bottom 65% of its height - re-scaled to fill the frame. Two
    exposures of one image therefore show genuinely different parts of it, which a
    zoompan offset could never achieve (see the FRAMINGS comment).
    """
    if focus in FRAMINGS_VERTICAL:
        span, start = FRAMINGS_VERTICAL[focus]
        return f"crop=iw:ih*{span}:0:ih*{start}"
    span, start = FRAMINGS_HORIZONTAL.get(focus, FRAMINGS_HORIZONTAL["center"])
    return f"crop=iw*{span}:ih:iw*{start}:0"


def contrasting_pairs(movements):
    """Pairs of configured movements that differ in BOTH framing and zoom direction.

    An image is shown twice (see pipeline/render.py); the pair picked for it decides
    how different the two showings look. Differing in framing alone leaves two shots
    that drift the same way across different halves of the picture; differing in zoom
    alone leaves the identical crop running backwards. Both differing is what makes
    the repeat read as another shot rather than the same one again.

    Falls back to a weaker rule rather than failing if the configured list cannot
    supply a fully contrasting pair: the repeat still has to come from somewhere.
    """
    def pairs_where(ok):
        return [(a, b) for a in movements for b in movements if ok(a, b)]

    both = pairs_where(lambda a, b: a.get("focus") != b.get("focus")
                       and a.get("zoom") != b.get("zoom"))
    if both:
        return both
    framing = pairs_where(lambda a, b: a.get("focus") != b.get("focus"))
    if framing:
        return framing
    distinct = pairs_where(lambda a, b: a != b)
    return distinct or [(movements[0], movements[0])]


def _fade_filters(duration_s, cfg_render, fade_in, fade_out):
    """fade=in / fade=out filters for a clip that begins or ends a scene.

    This is how scene boundaries are made, and it replaces an xfade chain across the
    finished scene clips. xfade was measured on this video: 43 clips did not finish in
    1 hour 50 minutes and had to be killed, while the concat demuxer joined the same
    clips in 3.9 seconds. Fading each clip costs nothing, because the clip is being
    encoded anyway, and the join afterwards is a stream copy.
    """
    seconds = cfg_render.get("scene_fade_s", 0.5)
    if seconds <= 0 or duration_s <= 2 * seconds:
        return []
    out = []
    if fade_in:
        out.append(f"fade=t=in:st=0:d={seconds}")
    if fade_out:
        out.append(f"fade=t=out:st={duration_s - seconds:.3f}:d={seconds}")
    return out


def kenburns_clip(image_path, out_path, duration_s, movement, cfg_render, cache_dir,
                  fade_in=False, fade_out=False):
    """One Ken Burns sub-shot: a slice of the image, re-framed to fill the screen,
    with a slow zoom in or out across it. Video only, no audio.

    Reads from the cached upscaled still (prescale_image), crops the framing's region,
    scales that back up to the delivery size, then zoompan moves within it and outputs
    straight at the final width x height.
    """
    width, height = cfg_render["width"], cfg_render["height"]
    fps = cfg_render["fps"]
    factor = cfg_render["upscale_factor"]
    zoom_min = cfg_render.get("zoom_min", 1.0)
    zoom_max = cfg_render["zoom_max"]
    frames = max(1, round(duration_s * fps))

    prescaled = prescale_image(image_path, cache_dir, width, height, factor)

    progress = f"on/{frames}"
    span = zoom_max - zoom_min
    if movement.get("zoom") == "out":
        zoom_expr = f"{zoom_max}-{span}*{progress}"
    else:
        zoom_expr = f"{zoom_min}+{span}*{progress}"

    # Horizontal drift. zoompan's crop window is iw/zoom wide, so the room it has to
    # travel is (iw - iw/zoom) - the whole of which this uses, which is why the pan
    # is worth having only when the zoom actually opens some room: at zoom 1.0 there
    # is none at all, and the drift starts from nothing and widens with the push.
    pan = movement.get("pan")
    if pan == "right":
        x_expr = f"(iw-iw/zoom)*{progress}"
    elif pan == "left":
        x_expr = f"(iw-iw/zoom)*(1-{progress})"
    else:
        x_expr = "(iw/2)-(iw/zoom/2)"
    y_expr = "(ih/2)-(ih/zoom/2)"

    big_w, big_h = width * factor, height * factor
    chain = []
    focus = movement.get("focus")
    if focus:
        # Double-exposure framing: cut a real slice of the picture and blow it back
        # up to the working size, so zoompan still has plenty of source pixels to
        # round to and the move stays smooth. Only used when an image is shown more
        # than once and the second showing has to look like a different shot;
        # otherwise the whole frame is the shot.
        chain.append(framing_crop(focus, width, height))
        chain.append(f"scale={big_w}:{big_h}:flags=lanczos")
    chain.append(f"zoompan=z='{zoom_expr}':x='{x_expr}':y='{y_expr}':d={frames}:"
                 f"s={width}x{height}:fps={fps}")
    chain.append("setsar=1")
    chain += _fade_filters(duration_s, cfg_render, fade_in, fade_out)
    vf = ",".join(chain)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    args = ["-loop", "1", "-i", str(prescaled), "-vf", vf, "-t", f"{duration_s:.3f}",
            "-r", str(fps), "-pix_fmt", "yuv420p", "-an",
            "-c:v", "libx264", "-crf", str(cfg_render["crf"]),
            "-preset", cfg_render["preset"], str(out_path)]

    attempts = cfg_render.get("subshot_attempts", 3)
    timeout = cfg_render.get("subshot_timeout_s", 300)
    multiplier = cfg_render.get("subshot_timeout_multiplier", 2)

    for attempt in range(1, attempts + 1):
        started = time.time()
        try:
            result = _run_ffmpeg(
                args, f"ffmpeg could not render a Ken Burns clip from {image_path}",
                timeout=timeout)
        except subprocess.TimeoutExpired:
            # subprocess.run has already killed ffmpeg by this point. The output file
            # is a truncated fragment, so it has to go before the retry, or a resumed
            # run would later mistake it for a finished clip.
            out_path.unlink(missing_ok=True)
            log.warning("sub-shot %s hit the %ds timeout on attempt %d/%d; killed it, "
                        "deleted the partial file, retrying", out_path.name, timeout,
                        attempt, attempts)
            timeout = int(timeout * multiplier)
            continue

        elapsed = time.time() - started
        reported = _frames_reported(result.stderr)
        log.info("sub-shot %s: %.2fs wanted, d=%d, ffmpeg wrote %s frames, took %.1fs",
                 out_path.name, duration_s, frames,
                 reported if reported is not None else "?", elapsed)
        if reported is not None and reported > frames * 1.5:
            log.warning("sub-shot %s processed %d frames for a %d frame clip - ffmpeg "
                        "is generating frames that -t then discards",
                        out_path.name, reported, frames)
        return out_path

    raise RenderError(f"Ken Burns for {out_path.name} timed out {attempts} times")


def static_clip(image_path, out_path, duration_s, cfg_render, cache_dir,
                fade_in=False, fade_out=False):
    """A still frame held for duration_s, with no camera movement at all.

    The last resort when Ken Burns has timed out repeatedly. Deliberately not a
    plain "skip": dropping the sub-shot would shorten the video against a voice
    track that does not shorten with it, so everything after that point drifts out
    of sync and the whole render then fails its own duration check. A motionless
    shot costs one scene its movement; a missing shot costs the video. There is no
    zoompan here, so this encodes in a fraction of the time.
    """
    width, height = cfg_render["width"], cfg_render["height"]
    fps = cfg_render["fps"]
    prescaled = prescale_image(image_path, cache_dir, width, height,
                               cfg_render["upscale_factor"])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(
        ["-loop", "1", "-i", str(prescaled),
         "-vf", ",".join([f"scale={width}:{height}:flags=lanczos", "setsar=1"]
                         + _fade_filters(duration_s, cfg_render, fade_in, fade_out)),
         "-t", f"{duration_s:.3f}", "-r", str(fps), "-pix_fmt", "yuv420p", "-an",
         "-c:v", "libx264", "-crf", str(cfg_render["crf"]),
         "-preset", cfg_render["preset"], str(out_path)],
        f"ffmpeg could not render even a static clip from {image_path}",
        timeout=cfg_render.get("subshot_timeout_s", 300))
    log.warning("sub-shot %s rendered as a motionless still, with no camera movement",
                out_path.name)
    return out_path


def concat_clips(clip_paths, out_path):
    """Hard-cut concatenation (hence -c copy, no re-encode) of same-format clips,
    via the concat demuxer. Used for the sub-shots inside one scene."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    list_path = out_path.with_suffix(".concat.txt")
    list_path.write_text(
        "\n".join(f"file '{Path(p).resolve().as_posix()}'" for p in clip_paths),
        encoding="utf-8")
    try:
        _run_ffmpeg(["-f", "concat", "-safe", "0", "-i", str(list_path), "-c", "copy",
                    str(out_path)],
                   f"ffmpeg could not concatenate {len(clip_paths)} clips")
    finally:
        list_path.unlink(missing_ok=True)
    return out_path


def join_clips(clip_paths, out_path):
    """Join the finished scene clips into one video and return its duration.

    A stream copy through the concat demuxer, which is why the scene boundaries are faded
    into and out of black while each clip is being encoded (_fade_filters) instead of being
    cross-faded here. Measured on video 7: an xfade chain over these 43 clips had not
    finished after 1 hour 50 minutes and was killed; this takes about 4 seconds.
    """
    concat_clips(clip_paths, out_path)
    return probe_duration(out_path)


def mux_audio(video_path, audio_path, out_path, cfg_render, music_path=None):
    """Attach the finished audio track to the finished (silent) video. The video
    stream is copied as-is - it was already encoded at final settings by
    join_clips - only the audio gets transcoded, to AAC."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    args = ["-i", str(video_path), "-i", str(audio_path)]
    if music_path:
        # normalize=0 matters: amix's default divides every input by the number of inputs,
        # which would drop the narration 6 dB to make room for a bed that is already 26 dB
        # down. The bed was built to the voice's length, so duration=first cannot truncate.
        args += ["-i", str(music_path),
                 "-filter_complex", "[1:a][2:a]amix=inputs=2:duration=first:normalize=0[a]",
                 "-map", "0:v:0", "-map", "[a]"]
    else:
        args += ["-map", "0:v:0", "-map", "1:a:0"]
    _run_ffmpeg(
        args + ["-c:v", "copy", "-c:a", "aac", "-b:a", f"{cfg_render['audio_bitrate_kbps']}k",
                "-movflags", "+faststart", "-shortest", str(out_path)],
        "ffmpeg could not mux audio into the final video")
    return out_path


def extract_preview(video_path, out_path, seconds, cfg_render):
    """First `seconds` of the final video, re-encoded (not a stream copy) so the cut
    lands exactly on the second rather than the nearest keyframe."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(
        ["-i", str(video_path), "-t", str(seconds),
         "-c:v", "libx264", "-crf", str(cfg_render["crf"]), "-preset", cfg_render["preset"],
         "-c:a", "aac", "-b:a", f"{cfg_render['audio_bitrate_kbps']}k",
         "-movflags", "+faststart", str(out_path)],
        f"ffmpeg could not cut a {seconds}s preview")
    return out_path


def frame_brightness(video_path, at_seconds):
    """Mean luminance (0-255) of the frame at at_seconds. Used to catch a black
    first/last frame, which a file that merely "opens fine" does not rule out."""
    require_ffmpeg()
    from PIL import Image
    import io
    result = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(at_seconds), "-i", str(video_path),
         "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "pipe:1"],
        capture_output=True, timeout=60)
    if result.returncode != 0 or not result.stdout:
        raise RenderError(f"could not extract a frame at {at_seconds}s from {video_path}: "
                          f"{result.stderr.decode('utf-8', 'replace')[-300:]}")
    image = Image.open(io.BytesIO(result.stdout)).convert("L")
    import numpy as np
    return float(np.asarray(image, dtype=np.float64).mean())
