"""Text-to-speech: two interchangeable providers, picked by config['tts']['provider'].

piper runs fully offline from local voice models in assets/voices/ (not in git: see
README for the download command). edge-tts calls Microsoft's free cloud endpoint,
which some server/cloud IPs see answer with a 403; that case is caught and logged as
a plain explanation, not a stack trace, because the fix is switching provider, not
retrying harder.

Both providers converge on the same output: a 22050 Hz mono wav file, so the render
stage never has to know which one made it.
"""
import asyncio
import subprocess
import wave
from pathlib import Path

from core.logger import get_logger
from core.retry import retry

log = get_logger("tts")

VOICES_DIR = Path(__file__).resolve().parent.parent / "assets" / "voices"

DEFAULT_PIPER_VOICE = "en_US-ryan-high"
DEFAULT_PIPER_LENGTH_SCALE = 1.08
DEFAULT_EDGE_VOICE = "en-US-GuyNeural"
DEFAULT_EDGE_RATE = "-8%"

# Candidates downloaded / offered for the by-ear comparison. Not all of these need
# to be the one config.json picks; they are what scripts/tts_sample.py renders.
PIPER_VOICES = ["en_US-ryan-high", "en_GB-alan-medium", "en_US-joe-medium"]
EDGE_VOICES = ["en-US-GuyNeural", "en-GB-RyanNeural", "en-US-BrianNeural",
               "en-US-ChristopherNeural"]

SAMPLE_RATE = 22050


class TTSError(Exception):
    pass


# --- piper -------------------------------------------------------------------

_piper_cache = {}


def _load_piper_voice(name):
    if name not in _piper_cache:
        from piper import PiperVoice
        model_path = VOICES_DIR / f"{name}.onnx"
        if not model_path.exists():
            raise TTSError(
                f"piper voice '{name}' is not downloaded (looked in {model_path}). Get it "
                f"with:\n  python -m piper.download_voices --download-dir {VOICES_DIR} {name}")
        _piper_cache[name] = PiperVoice.load(str(model_path))
    return _piper_cache[name]


def synthesize_piper(text, out_path, voice=DEFAULT_PIPER_VOICE,
                     length_scale=DEFAULT_PIPER_LENGTH_SCALE):
    """Render text to a wav file with a local piper voice model."""
    from piper.config import SynthesisConfig
    pv = _load_piper_voice(voice)
    syn_cfg = SynthesisConfig(length_scale=length_scale)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as f:
        pv.synthesize_wav(text, f, syn_config=syn_cfg)


# --- edge-tts ------------------------------------------------------------------

def _is_403(error):
    msg = str(error)
    return "403" in msg or "Forbidden" in msg or "NoAudioReceived" in msg


@retry(attempts=2, base_delay=3)
def synthesize_edge(text, out_path, voice=DEFAULT_EDGE_VOICE, rate=DEFAULT_EDGE_RATE):
    """Render text to a wav file via Microsoft's free cloud TTS, converted from mp3."""
    import edge_tts
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mp3_path = out_path.with_suffix(".mp3")

    async def _run():
        communicate = edge_tts.Communicate(text, voice, rate=rate)
        await communicate.save(str(mp3_path))

    try:
        asyncio.run(_run())
    except Exception as e:
        if _is_403(e):
            # Not worth retrying: a 403 here means the calling IP is blocked, which
            # will not change between attempts.
            raise TTSError(
                "edge-tts got a 403 from Microsoft's endpoint. Some server or cloud IPs "
                "are blocked from this free service; it typically works from a home "
                "connection. Switch config tts.provider to \"piper\" if this keeps "
                "happening on this machine.") from e
        raise TTSError(f"edge-tts request failed: {e}") from e

    if not mp3_path.exists() or mp3_path.stat().st_size == 0:
        raise TTSError("edge-tts produced no audio (empty or missing file)")

    _mp3_to_wav(mp3_path, out_path)
    mp3_path.unlink(missing_ok=True)


def _mp3_to_wav(src, dst):
    result = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
         "-ar", str(SAMPLE_RATE), "-ac", "1", str(dst)],
        capture_output=True, text=True)
    if result.returncode != 0:
        raise TTSError(f"ffmpeg could not convert edge-tts output to wav: "
                       f"{result.stderr.strip()[-300:]}")


# --- shared --------------------------------------------------------------------

def wav_duration(path):
    with wave.open(str(path), "rb") as f:
        return f.getnframes() / f.getframerate()


def provider_settings(cfg):
    """The provider name plus its own settings block, filled in with defaults."""
    tts_cfg = cfg.get("tts") or {}
    provider = tts_cfg.get("provider", "piper")
    if provider == "piper":
        block = tts_cfg.get("piper") or {}
        return provider, {"voice": block.get("voice", DEFAULT_PIPER_VOICE),
                          "length_scale": block.get("length_scale", DEFAULT_PIPER_LENGTH_SCALE)}
    if provider == "edge":
        block = tts_cfg.get("edge") or {}
        return provider, {"voice": block.get("voice", DEFAULT_EDGE_VOICE),
                          "rate": block.get("rate", DEFAULT_EDGE_RATE)}
    raise TTSError(f"unknown tts provider: {provider!r} (use \"piper\" or \"edge\")")


def synthesize(text, out_path, cfg):
    """Render one piece of narration to a wav file. Returns its duration in seconds."""
    provider, settings = provider_settings(cfg)
    if provider == "piper":
        synthesize_piper(text, out_path, settings["voice"], settings["length_scale"])
    else:
        synthesize_edge(text, out_path, settings["voice"], settings["rate"])
    return wav_duration(out_path)


def concatenate(wav_paths, out_path):
    """Join scene wav files into one track with ffmpeg's concat demuxer."""
    out_path = Path(out_path)
    list_path = out_path.with_suffix(".txt")
    list_path.write_text(
        "\n".join(f"file '{Path(p).resolve().as_posix()}'" for p in wav_paths), encoding="utf-8")
    result = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", str(list_path), "-c", "copy", str(out_path)],
        capture_output=True, text=True)
    list_path.unlink(missing_ok=True)
    if result.returncode != 0:
        raise TTSError(f"ffmpeg could not concatenate {len(wav_paths)} scene files: "
                       f"{result.stderr.strip()[-300:]}")
    return wav_duration(out_path)
