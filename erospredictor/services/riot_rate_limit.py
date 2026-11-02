"""One rate gate for RiotWatcher and direct Riot HTTP calls.

RiotWatcher learns when the *next* call may be made but does not retry a 429.
This limiter counts every actual attempt, honours both application and method
windows, and provides a bounded retry wrapper around the public API operation.
"""

from collections import deque
from datetime import timezone
from email.utils import parsedate_to_datetime
import hashlib
import math
import threading
import time
from services.update_retry import is_transient, retry_delay


def retry_after_seconds(value, wall_time=None):
    """Accept seconds or an HTTP date; malformed values require a fallback."""
    try:
        number = float(value)
        if math.isfinite(number) and number >= 0:
            return number
    except (TypeError, ValueError, OverflowError):
        pass
    try:
        parsed = parsedate_to_datetime(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return max(0.0, parsed.timestamp() - (time.time() if wall_time is None else wall_time))
    except (TypeError, ValueError, OverflowError):
        return None


def _windows(value):
    windows = {}
    for part in str(value or "").split(","):
        try:
            count, duration = part.strip().split(":")
            count, duration = int(count), float(duration)
            if count >= 0 and duration > 0 and math.isfinite(duration):
                windows[duration] = count
        except (ValueError, TypeError, OverflowError):
            continue
    return windows


def _limits(value):
    """Accept a complete positive server policy; malformed headers keep the old one."""
    if not isinstance(value, str) or not value.strip():
        return None
    parts = value.split(",")
    result = _windows(value)
    if len(result) != len(parts) or any(count <= 0 for count in result.values()):
        return None
    return result


class RiotRateLimiter:
    """Duck-types RiotWatcher's RateLimiter and can use fake time in tests.

    App limits are shared by every endpoint using the same key and route. The
    personal-key defaults apply until valid server headers replace them, including
    higher budgets and different windows. Header counts include other key users.
    """

    def __init__(self, clock=None, sleeper=None, logger=print, wall_clock=None,
                 app_limits=((20, 1), (100, 120))):
        self.clock = clock or time.monotonic
        self.sleeper = sleeper or time.sleep
        self.logger = logger
        self.wall_clock = wall_clock or time.time
        self.default_limits = {float(duration): int(limit) for limit, duration in app_limits}
        if any(limit <= 0 or duration <= 0 or not math.isfinite(duration)
               for duration, limit in self.default_limits.items()):
            raise ValueError("Rate-limit windows must have positive limits and durations.")
        self._apps = {}
        self._methods = {}
        self._blocked_until = {}
        self._lock = threading.RLock()

    @staticmethod
    def _state(limits):
        return {"limits": dict(limits), "attempts": deque(), "observed": {}}

    def _states(self, region, endpoint, method):
        app = self._apps.setdefault(region, self._state(self.default_limits))
        specific = self._methods.setdefault((region, endpoint, method), self._state({}))
        return app, specific

    @staticmethod
    def _delay(state, now):
        limits, attempts = state["limits"], state["attempts"]
        longest = max(limits, default=0)
        while attempts and attempts[0] <= now - longest:
            attempts.popleft()
        delay = 0.0
        for duration, limit in limits.items():
            recent = [stamp for stamp in attempts if stamp > now - duration]
            if len(recent) >= limit:
                delay = max(delay, recent[-limit] + duration - now)
            observed = state["observed"].get(duration)
            if observed and observed[1] > now and observed[0] >= limit:
                delay = max(delay, observed[1] - now)
        return delay

    def _pause(self, seconds):
        deadline = self.clock() + seconds
        if self.logger and seconds >= 1:
            self.logger(f"[Riot API] Kéréskorlát: várakozás {seconds:.1f} másodpercig...")
        while True:
            remaining = deadline - self.clock()
            if remaining <= 0:
                return
            self.sleeper(min(30.0, remaining))
            remaining = deadline - self.clock()
            if self.logger and remaining >= 1:
                self.logger(f"[Riot API] A várakozásból még {remaining:.1f} másodperc van hátra.")

    def acquire(self, region, endpoint, method):
        """Reserve exactly one actual request after all applicable windows clear."""
        while True:
            with self._lock:
                now = self.clock()
                states = self._states(region, endpoint, method)
                delay = max(0.0, self._blocked_until.get(region, 0) - now,
                            *(self._delay(state, now) for state in states))
                if delay <= 0:
                    for state in states:
                        state["attempts"].append(now)
                        for duration, (count, expiry) in list(state["observed"].items()):
                            if expiry > now:
                                state["observed"][duration] = (count + 1, expiry)
                    return
            self._pause(delay)

    def wait_until(self, region, endpoint_name, method_name):
        # Block here using the same injected monotonic clock as retry handling.
        # Returning None prevents RiotWatcher's adapter from sleeping a second time.
        self.acquire(region, endpoint_name, method_name)
        return None

    def record_response(self, region, endpoint_name, method_name, status, headers):
        headers = {str(k).lower(): v for k, v in (headers or {}).items()}
        with self._lock:
            now = self.clock()
            app, method = self._states(region, endpoint_name, method_name)
            for state, prefix in ((app, "x-app-rate-limit"), (method, "x-method-rate-limit")):
                advertised = _limits(headers.get(prefix))
                if advertised is not None:
                    if state is app and self.logger and (advertised != state["limits"] or not state.get("announced")):
                        budget = ", ".join(f"{limit} kérés/{duration:g} s" for duration, limit in advertised.items())
                        self.logger(f"[Riot API] {region}: szerver által engedett keret: {budget}")
                        state["announced"] = True
                    state["limits"] = advertised
                    state["observed"] = {duration: value for duration, value in state["observed"].items()
                                         if duration in advertised}
                for duration, count in _windows(headers.get(prefix + "-count")).items():
                    if duration not in state["limits"]:
                        continue
                    previous = state["observed"].get(duration)
                    expiry = previous[1] if previous and previous[1] > now else now + duration
                    # An older concurrent response cannot erase already reserved calls.
                    count = max(count, previous[0]) if previous and previous[1] > now else count
                    state["observed"][duration] = (count, expiry)
            if status == 429:
                delay = retry_after_seconds(headers.get("retry-after"), self.wall_clock())
                if delay is not None:
                    self._blocked_until[region] = max(self._blocked_until.get(region, 0), now + delay)

    def call(self, operation, region, max_retries=5, retry_transient=False):
        """Optionally keep retrying temporary network/server failures until cancelled."""
        if max_retries is not None and (not isinstance(max_retries, int) or max_retries < 0):
            raise ValueError("max_retries must be a non-negative integer.")
        attempt = 0
        while True:
            try:
                return operation()
            except Exception as exc:
                response = getattr(exc, "response", None)
                status = getattr(response, "status_code", None) if response is not None else None
                temporary = is_transient(exc) if retry_transient else status == 429
                if not temporary or (max_retries is not None and attempt >= max_retries):
                    raise
                headers = {str(k).lower(): v for k, v in (getattr(response, "headers", {}) or {}).items()}
                delay = retry_after_seconds(headers.get("retry-after"), self.wall_clock())
                if delay is None:
                    delay = retry_delay(attempt)
                with self._lock:
                    self._blocked_until[region] = max(self._blocked_until.get(region, 0), self.clock() + delay)
                if self.logger:
                    limit = str(max_retries) if max_retries is not None else "folyamatos"
                    reason = str(status) if status is not None else "kapcsolati hiba"
                    self.logger(f"[Riot API] {reason}: azonos kérés újrapróbálása várakozás után "
                                f"({attempt + 1}/{limit}); Ctrl+C: leállítás.")
                attempt += 1


_SHARED = {}
_SHARED_LOCK = threading.Lock()


def shared_limiter(api_key):
    """One in-process gate per key; registry keys never contain the actual token."""
    identity = hashlib.sha256(str(api_key).encode("utf-8")).digest()
    with _SHARED_LOCK:
        return _SHARED.setdefault(identity, RiotRateLimiter())
