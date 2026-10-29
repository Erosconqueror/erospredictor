"""Optional own-account history through the signed-in Riot client's service.

This is separate from the developer Match-v5 API. The caller must explicitly
enable it; session credentials stay in memory and go only to the Riot server
configured by the installed client. The default local history path never calls it.
"""

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
import re
import time
from urllib.parse import quote, urlsplit

import requests
from services.draft_import import DraftImportError


REGION = "/riotclient/region-locale"
ENTITLEMENTS = "/entitlements/v1/token"


def validate_server(value):
    if not isinstance(value, str):
        raise DraftImportError("A kliens előzményszerverének címe nem olvasható.")
    try:
        parsed = urlsplit(value)
        valid = (parsed.scheme == "https" and parsed.hostname is not None
            and re.fullmatch(r"[a-z0-9-]+\.pp\.sgp\.pvp\.net", parsed.hostname)
            and parsed.port in (None, 443) and not parsed.username and not parsed.password
            and parsed.path in ("", "/") and not parsed.query and not parsed.fragment)
    except ValueError:
        valid = False
    if not valid:
        raise DraftImportError("A kliens nem támogatott előzményszervert jelölt meg; hitelesítési adatot nem küldtem.")
    return f"https://{parsed.hostname}"


def server_from_config(path, region):
    """Read just one simple URL setting, without a general YAML dependency."""
    if not isinstance(region, str) or not re.fullmatch(r"[A-Z0-9_-]{1,20}", region):
        raise DraftImportError("A kliens régiója nem határozható meg.")
    try:
        path = Path(path)
        if path.stat().st_size > 2 * 1024 * 1024:
            raise OSError("oversized config")
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        raise DraftImportError("Nem olvasható a LoL-kliens system.yaml konfigurációja.") from None
    stack = []
    for line in lines:
        matched = re.fullmatch(r"( *)([A-Za-z0-9_-]+):(?:\s+(.*))?", line)
        if not matched:
            continue
        indent, key, value = len(matched[1]), matched[2], matched[3]
        while stack and stack[-1][0] >= indent:
            stack.pop()
        parents = [item[1] for item in stack]
        if key == "player_platform_edge_url" and parents[-3:] == [region, "servers", "player_platform_edge"]:
            return validate_server((value or "").strip().strip("\"'"))
        if not value:
            stack.append((indent, key))
    raise DraftImportError("A saját kliensrégió előzményszervere nem található a konfigurációban.")


def retry_seconds(headers, fallback):
    value = headers.get("Retry-After")
    try:
        return max(0, float(value))
    except (TypeError, ValueError):
        try:
            return max(0, (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError, IndexError):
            return fallback


class RiotClientHistory:
    _cooldowns = {}

    def __init__(self, server, token, puuid, transport=None, clock=time.monotonic, sleeper=time.sleep):
        self.server = validate_server(server)
        if not isinstance(token, str) or not token or not isinstance(puuid, str) or not puuid:
            raise DraftImportError("A kliens saját bejelentkezése nem érhető el. Jelentkezz be újra a LoL-kliensbe.")
        self._token, self._puuid = token, puuid
        self._transport = transport
        self._clock, self._sleep = clock, sleeper

    def _wait(self, seconds, cancelled):
        remaining = seconds
        while remaining > 0:
            if cancelled():
                raise DraftImportError("Az előzmények beolvasása leállítva; a mentett oldalak megmaradtak.")
            chunk = min(0.25, remaining)
            self._sleep(chunk)
            remaining -= chunk

    def pages(self, limit, cancelled=lambda: False):
        if limit not in (100, 500, 1000):
            raise ValueError("Unsupported history limit")
        key = (self.server, self._puuid)
        until = self._cooldowns.get(key, 0)
        if until > self._clock():
            raise DraftImportError(f"A Riot kliensszolgáltatás várakozást kér. Próbáld újra {until - self._clock():.0f} másodperc múlva.")
        transport = self._transport or requests.Session()
        if self._transport is None:
            transport.trust_env = False
        endpoint = f"{self.server}/match-history-query/v1/products/lol/player/{quote(self._puuid, safe='')}/SUMMARY"
        try:
            for start in range(0, limit, 50):
                count = min(50, limit - start)
                if start:
                    self._wait(0.5, cancelled)
                for attempt in range(3):
                    if cancelled():
                        raise DraftImportError("Az előzmények beolvasása leállítva; a mentett oldalak megmaradtak.")
                    try:
                        response = transport.get(endpoint, params={"startIndex": start, "count": count},
                            headers={"Authorization": "Bearer " + self._token}, timeout=(3, 10),
                            verify=True, allow_redirects=False)
                    except requests.RequestException:
                        if attempt == 2:
                            raise DraftImportError("A Riot kliensszolgáltatás nem érhető el. A már mentett adatok megmaradtak.") from None
                        self._wait(2 ** attempt, cancelled)
                        continue
                    status = response.status_code
                    if status == 429:
                        delay = retry_seconds(response.headers, 5 * (attempt + 1))
                        self._cooldowns[key] = self._clock() + delay
                        if attempt == 2 or delay > 120:
                            raise DraftImportError(f"A Riot kliensszolgáltatás korlátozta a lekérést. Várj legalább {delay:.0f} másodpercet. A mentett adatok megmaradtak.")
                        self._wait(delay, cancelled)
                        continue
                    if status in (500, 502, 503, 504) and attempt < 2:
                        self._wait(2 ** attempt, cancelled)
                        continue
                    if status in (401, 403):
                        raise DraftImportError("A kliens bejelentkezése lejárt vagy az előzmények nem hozzáférhetők. Jelentkezz be újra a LoL-kliensbe.")
                    if status != 200:
                        raise DraftImportError("A Riot kliensszolgáltatás nem adott használható előzményválaszt.")
                    try:
                        payload = response.json()
                    except ValueError:
                        raise DraftImportError("A Riot kliensszolgáltatás válasza hiányos. A mentett oldalak megmaradtak.") from None
                    games = payload.get("games") if isinstance(payload, dict) else None
                    if not isinstance(games, list):
                        raise DraftImportError("A Riot kliensszolgáltatás előzményformátuma nem olvasható.")
                    if not games:
                        return
                    page = []
                    for item in games[:count]:
                        game = item.get("json") if isinstance(item, dict) else None
                        if isinstance(game, dict):
                            page.append(game)
                    if not page:
                        raise DraftImportError("A Riot kliensszolgáltatás nem adott olvasható meccseket.")
                    yield page
                    if len(games) < count:
                        return
                    break
        finally:
            if self._transport is None:
                transport.close()
