import functools
import time

from core.logger import get_logger

log = get_logger("retry")


def retry(attempts=3, base_delay=2):
    """Retry with exponential backoff: base_delay, 2*base_delay, 4*base_delay..."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            for attempt in range(1, attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    log.warning("%s failed (attempt %d/%d): %s", fn.__name__, attempt, attempts, e)
                    if attempt == attempts:
                        raise
                    time.sleep(base_delay * 2 ** (attempt - 1))
        return wrapper
    return decorator
