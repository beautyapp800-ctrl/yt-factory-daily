import os


def file_exists(path):
    return bool(path) and os.path.isfile(path) and os.path.getsize(path) > 0


def audio_ok(path):
    # TODO: check codec/duration
    return file_exists(path)


def duration_within(actual, expected, tol):
    return abs(actual - expected) <= tol


def video_ok(path):
    # TODO: check streams/duration with ffprobe
    return file_exists(path)
