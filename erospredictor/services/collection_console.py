"""Forward concise collection events without piping/blocking scraper output."""
from pathlib import Path
import json
import os
import threading
import time

_sent = {}
_lock = threading.Lock()


def api_event(route, kind, **fields):
    """Collector-only telemetry, without keys, URLs, identities or extra requests."""
    if "ERO_COLLECT_WINDOW" not in os.environ:
        return
    now = time.time()
    with _lock:
        key = (route, kind)
        signature = fields.get("status") if kind == "response" else fields.get("reason")
        stamp, previous, prior_deadline = _sent.get(key, (0, None, 0))
        deadline = now + fields.get("seconds", 0) if kind == "wait" else 0
        if signature == previous and now - stamp < 30 and deadline <= prior_deadline + 5:
            return
        _sent[key] = (now, signature, deadline)
        try:
            print("[CollectorAPI] " + json.dumps(dict(route=route, kind=kind, at=now, **fields)), flush=True)
        except OSError:
            pass


class ApiWatch:
    def __init__(self, label, clock=None):
        self.clock = clock or time.time
        self.label = label
        self.last_response = self.clock()
        self.wait_until = 0
        self.last_notice = 0
        self.stalled = False

    def observe(self, event):
        at = event["at"]
        if event["kind"] == "response":
            self.last_response = max(self.last_response, at)
            if self.stalled:
                print(f"[Collection] {self.label}: az API ujra valaszol (HTTP {event['status']}).", flush=True)
                self.stalled = False
            if event["status"] in (401, 403, 429) or event["status"] >= 500:
                print(f"[Collection] {self.label}: API HTTP {event['status']} ({event['route']}); reszletek a worker logban.", flush=True)
        elif event["kind"] == "wait" and event.get("reason") == "rate_limit":
            self.wait_until = max(self.wait_until, at + event["seconds"])

    def tick(self):
        now = self.clock()
        age = now - self.last_response
        if age < 180 or (self.last_notice and now - self.last_notice < 300):
            return
        if now < self.wait_until:
            print(f"[Collection] {self.label}: ismert API-keretvarakozas, meg kb. {self.wait_until-now:.0f} masodperc.", flush=True)
        else:
            print(f"[Collection] {self.label}: {age/60:.1f} perce nem erkezett API-valasz; "
                  "a worker el, az ok lehet halozati hiba vagy helyi feldolgozas. Ellenorizd a logjat.", flush=True)
            self.stalled = True
        self.last_notice = now


class CollectionLogTail:
    def __init__(self, path, api_watch=None):
        self.path = Path(path)
        self.offset = self.path.stat().st_size if self.path.exists() else 0
        self.api_watch = api_watch

    def drain(self, prefix=""):
        try:
            with self.path.open("rb") as stream:
                if self.path.stat().st_size < self.offset:
                    self.offset = 0
                stream.seek(self.offset)
                # Bound disk work per tick; partial writes wait for the next tick.
                end = self.offset + 1024 * 1024
                while stream.tell() < end:
                    line = stream.readline()
                    if not line or not line.endswith(b"\n"):
                        break
                    self.offset = stream.tell()
                    text = line.decode("utf-8", errors="replace").rstrip()
                    if self.api_watch and text.startswith("[CollectorAPI] "):
                        try:
                            self.api_watch.observe(json.loads(text[len("[CollectorAPI] "):]))
                        except (ValueError, KeyError, TypeError):
                            pass
                    if "[Collection]" in text:
                        print(prefix + text, flush=True)
        except OSError:
            pass  # Console reporting must never interrupt collection.
