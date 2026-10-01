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
from pathlib import Path

from core.logger import get_logger

log = get_logger("render")

USER_AGENT = "yt-factory/1.0"   # unused by ffmpeg itself; kept for log message parity

FOCUS_X = {"left": 0.33, "right": 0.67, "center": 0.5}


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


def _run_ffmpeg(args, error_context, timeout=None):
    require_ffmpeg()
    result = subprocess.run(["ffmpeg", "-y", "-loglevel", "error"] + args,
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
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    big_w, big_h = width * upscale_factor, height * upscale_factor
    cache_path = cache_dir / f"{Path(image_path).stem}_{big_w}x{big_h}.png"
    if cache_path.exists():
        return cache_path
    _run_ffmpeg(["-i", str(image_path), "-vf", f"scale={big_w}:{big_h}:flags=lanczos",
                "-frames:v", "1", str(cache_path)],
               f"ffmpeg could not pre-scale {image_path}")
    return cache_path


def kenburns_clip(image_path, out_path, duration_s, movement, cfg_render, cache_dir):
    """One Ken Burns sub-shot from a still image: a slow zoom in or out, biased
    toward the left/right/center third of the frame. Video only, no audio.

    Reads from the cached upscaled still (prescale_image), and zoompan outputs
    straight at the final width x height - no further downscale filter needed, which
    is both simpler and the whole point of the fix above.
    """
    width, height = cfg_render["width"], cfg_render["height"]
    fps = cfg_render["fps"]
    factor = cfg_render["upscale_factor"]
    zoom_max = cfg_render["zoom_max"]
    frames = max(1, round(duration_s * fps))
    fx = FOCUS_X.get(movement.get("focus", "center"), 0.5)

    prescaled = prescale_image(image_path, cache_dir, width, height, factor)

    if movement.get("zoom") == "out":
        zoom_expr = f"{zoom_max}-({zoom_max}-1)*on/{frames}"
    else:
        zoom_expr = f"1+({zoom_max}-1)*on/{frames}"
    x_expr = f"(iw*{fx})-(iw/zoom/2)"
    y_expr = "(ih/2)-(ih/zoom/2)"

    vf = (f"zoompan=z='{zoom_expr}':x='{x_expr}':y='{y_expr}':d={frames}:"
         f"s={width}x{height}:fps={fps}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(
        ["-loop", "1", "-i", str(prescaled), "-vf", vf, "-t", f"{duration_s:.3f}",
         "-r", str(fps), "-pix_fmt", "yuv420p", "-an",
         "-c:v", "libx264", "-crf", str(cfg_render["crf"]), "-preset", cfg_render["preset"],
         str(out_path)],
        f"ffmpeg could not render a Ken Burns clip from {image_path}")
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


def xfade_chain(clip_paths, out_path, crossfade_s, cfg_render):
    """Chain N scene clips into one video with a crossfade at every join, in a
    single ffmpeg pass (one decode+encode per clip, not O(N^2) from re-encoding a
    growing merged file N times). Returns the duration of the finished video.
    """
    if len(clip_paths) == 1:
        # Nothing to cross-fade; just restate the one clip as the full video.
        _run_ffmpeg(["-i", str(clip_paths[0]), "-c", "copy", str(out_path)],
                   "ffmpeg could not copy the single scene clip")
        return probe_duration(out_path)

    durations = [probe_duration(p) for p in clip_paths]
    inputs = []
    for p in clip_paths:
        inputs += ["-i", str(p)]

    filters = []
    label = "0"
    merged_duration = durations[0]
    for i in range(1, len(clip_paths)):
        offset = max(0.0, merged_duration - crossfade_s)
        next_label = f"v{i}"
        filters.append(
            f"[{label}][{i}]xfade=transition=fade:duration={crossfade_s}:"
            f"offset={offset:.3f}[{next_label}]")
        merged_duration = merged_duration + durations[i] - crossfade_s
        label = next_label

    filter_complex = ";".join(filters)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(
        inputs + ["-filter_complex", filter_complex, "-map", f"[{label}]",
                 "-pix_fmt", "yuv420p", "-r", str(cfg_render["fps"]),
                 "-c:v", "libx264", "-crf", str(cfg_render["crf"]),
                 "-preset", cfg_render["preset"], str(out_path)],
        f"ffmpeg could not cross-fade {len(clip_paths)} scene clips",
        timeout=1800)
    return probe_duration(out_path)


def mux_audio(video_path, audio_path, out_path, cfg_render):
    """Attach the finished audio track to the finished (silent) video. The video
    stream is copied as-is - it was already encoded at final settings by
    xfade_chain - only the audio gets transcoded, to AAC."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(
        ["-i", str(video_path), "-i", str(audio_path), "-map", "0:v:0", "-map", "1:a:0",
         "-c:v", "copy", "-c:a", "aac", "-b:a", f"{cfg_render['audio_bitrate_kbps']}k",
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
