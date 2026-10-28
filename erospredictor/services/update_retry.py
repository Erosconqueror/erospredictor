"""Transient failure classification and interruptible, visible retry delays."""

import http.client
import time
from urllib.error import URLError


TRANSIENT_HTTP = {408, 425, 429, 500, 502, 503, 504}


def http_status(error):
    response = getattr(error, "response", None)
    return getattr(response, "status_code", None) if response is not None else getattr(error, "code", None)


def is_transient(error):
    status = http_status(error)
    if status is not None:
        return status in TRANSIENT_HTTP
    if isinstance(error, (ConnectionError, TimeoutError, URLError, http.client.HTTPException)):
        return True
    try:
        from requests.exceptions import ConnectionError as RequestsConnectionError, Timeout, ChunkedEncodingError
        return isinstance(error, (RequestsConnectionError, Timeout, ChunkedEncodingError))
    except ImportError:
        return False


def retry_delay(attempt):
    return min(120.0, 5.0 * (2 ** min(attempt, 5)))


def wait_for_retry(seconds, log=print, sleeper=None):
    """Small sleeps allow Ctrl+C at any point and regular progress messages."""
    sleeper = sleeper or time.sleep
    remaining = seconds
    while remaining > 0:
        chunk = min(5.0, remaining)
        sleeper(chunk)
        remaining -= chunk
        if log and remaining > 0 and (remaining % 30 == 0):
            log(f"[Update] Újrapróbálkozás {remaining:.0f} másodperc múlva; Ctrl+C: leállítás.")
