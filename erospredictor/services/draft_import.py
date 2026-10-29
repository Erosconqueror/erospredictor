"""Read one visible draft snapshot from the local League client, without OCR.

The League Client API is an unsupported, local-only Riot interface. Keep its
shape handling here; never guess lanes or sides from the display/cell order.
Reference: https://developer.riotgames.com/docs/lol#league-client-api
"""

from dataclasses import dataclass, field, replace
import json
import os
from pathlib import Path
import subprocess
import warnings

import requests
from urllib3.exceptions import InsecureRequestWarning
from services.client_settings import (
    DEFAULT_LOCKFILE, normalize_lockfile, saved_lockfile, save_lockfile_path,
)


class DraftImportError(RuntimeError):
    """An actionable error safe to display without local authentication data."""


class DraftUnavailable(DraftImportError):
    """The local client is reachable but has no active champion-select session."""


@dataclass(frozen=True)
class LcuCredentials:
    port: int
    token: str = field(repr=False)
    pid: int = 0


@dataclass(frozen=True)
class DraftPick:
    team: str
    riot_id: int
    internal_id: int
    role: int | None
    cell_id: int


@dataclass(frozen=True)
class DraftSnapshot:
    picks: tuple[DraftPick, ...]
    blue_bans: tuple[int, ...] = ()
    red_bans: tuple[int, ...] = ()
    own_team: str | None = None
    own_bans: tuple[int, ...] = ()
    enemy_bans: tuple[int, ...] = ()
    warnings: tuple[str, ...] = ()
    session_key: str = ""
    own_role: int | None = None
    queue_id: int | None = None
    custom_game: bool = False
    roster_sizes: tuple[int, int] = (0, 0)


_ROLES = {
    "TOP": 0, "JUNGLE": 1, "MIDDLE": 2, "MID": 2,
    "BOTTOM": 3, "ADC": 3, "UTILITY": 4, "SUPPORT": 4,
}


def _number(value, default=0):
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def read_lockfile(path) -> LcuCredentials:
    """Read credentials, never include lockfile contents in errors or reprs."""
    path = Path(path)
    if path.is_dir():
        path = path / "lockfile"
    try:
        pieces = path.read_text(encoding="utf-8").strip().split(":")
    except (OSError, UnicodeError):
        raise DraftImportError("A LoL lockfile nem olvasható. Indítsd el a klienst, majd válaszd ki újra a fájlt.") from None
    if len(pieces) != 5:
        raise DraftImportError("Ez nem érvényes LoL lockfile. A League of Legends mappájának lockfile fájlját válaszd.")
    name, pid, port, token, protocol = pieces
    if (name not in ("LeagueClient", "LeagueClientUx") or protocol != "https"
            or _number(pid) <= 0 or not 1 <= _number(port) <= 65535 or not token):
        raise DraftImportError("A LoL lockfile adatai érvénytelenek. Indítsd újra a klienst, és próbáld ismét.")
    return LcuCredentials(_number(port), token, _number(pid))


def discover_lockfiles(explicit=None) -> list[Path]:
    """Find only likely client paths; never recursively scan the user's drives."""
    supplied = explicit or os.environ.get("EROSPREDICTOR_LCU_LOCKFILE")
    if supplied:
        return [normalize_lockfile(supplied)]
    for candidate in (saved_lockfile(), DEFAULT_LOCKFILE):
        if candidate is not None and candidate.is_file():
            return [candidate]
    return [root / "lockfile" for root in client_install_roots() if (root / "lockfile").is_file()]


def client_install_roots() -> list[Path]:
    """Find installations even when the client is closed and has no lockfile."""
    roots = [Path(f"{drive}:/Riot Games/League of Legends") for drive in ("C", "D", "E", "F", "G")]
    for variable in ("ProgramFiles", "ProgramFiles(x86)"):
        if os.environ.get(variable):
            roots.append(Path(os.environ[variable]) / "Riot Games/League of Legends")
    if os.name == "nt":
        try:
            import winreg
            for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
                for key_name in (r"SOFTWARE\Riot Games, Inc\League of Legends",
                                 r"SOFTWARE\WOW6432Node\Riot Games, Inc\League of Legends"):
                    try:
                        with winreg.OpenKey(hive, key_name) as key:
                            for value_name in ("Location", "InstallLocation", "Path"):
                                try:
                                    value, _ = winreg.QueryValueEx(key, value_name)
                                    roots.append(Path(value))
                                except OSError:
                                    pass
                    except OSError:
                        pass
        except ImportError:
            pass
        try:
            result = subprocess.run(
                ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
                 "Get-CimInstance Win32_Process -Filter \"Name = 'LeagueClientUx.exe' OR Name = 'LeagueClient.exe'\" | Select-Object -ExpandProperty ExecutablePath"],
                capture_output=True, text=True, timeout=4, check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            roots.extend(Path(line.strip()).parent for line in result.stdout.splitlines() if line.strip())
        except (OSError, subprocess.SubprocessError):
            pass
    return list(dict.fromkeys(roots))


def configure_client_location():
    """Installer setup: keep a discovered location or a useful default for later."""
    candidates = discover_lockfiles()
    for candidate in candidates:
        try:
            read_lockfile(candidate)
        except DraftImportError:
            continue
        return candidate, save_lockfile_path(candidate)
    # A remembered location remains useful while the client is not running.
    remembered = saved_lockfile()
    supplied = os.environ.get("EROSPREDICTOR_LCU_LOCKFILE")
    if supplied:
        target = normalize_lockfile(supplied)
    elif remembered is not None and remembered.parent.is_dir():
        target = remembered
    else:
        target = next((root / "lockfile" for root in client_install_roots()
                       if (root / "LeagueClient.exe").is_file()
                       or (root / "LeagueClientUx.exe").is_file()),
                      remembered or DEFAULT_LOCKFILE)
    return target, save_lockfile_path(target)


def _side_from_members(members):
    # Champ-select's explicit team enum uses 1/2; platform teamId uses 100/200.
    # Neither cellId nor display order establishes a side.
    sides = set()
    for member in members:
        if not isinstance(member, dict):
            continue
        platform = _number(member.get("teamId"))
        encoded = _number(member.get("team"))
        side = platform if platform in (100, 200) else {1: 100, 2: 200, 100: 100, 200: 200}.get(encoded)
        if side:
            sides.add(side)
    if sides == {100}:
        return "blue"
    if sides == {200}:
        return "red"
    return None


def _same_player(left, right):
    for key in ("puuid", "summonerId"):
        a, b = left.get(key), right.get(key)
        if a and b and str(a) != "0" and str(a) == str(b):
            return True
    return False


def resolve_own_side(session, gameflow=None, current_summoner=None):
    own_members = session.get("myTeam", [])
    own_members = [member for member in own_members if isinstance(member, dict)] if isinstance(own_members, list) else []
    side = _side_from_members(own_members)
    if side:
        return side
    explicit = session.get("myTeamTeamId")
    if _number(explicit) in (100, 200):
        return "blue" if _number(explicit) == 100 else "red"
    data = (gameflow or {}).get("gameData", {}) or {}
    if not isinstance(data, dict):
        return None
    identities = own_members + ([current_summoner] if current_summoner else [])
    found = set()
    for team_key, team_side in (("teamOne", "blue"), ("teamTwo", "red")):
        for member in data.get(team_key, []) or []:
            if not isinstance(member, dict):
                continue
            if any(_same_player(identity, member) for identity in identities):
                found.add(team_side)
    return found.pop() if len(found) == 1 else None


def _session_key(session, gameflow=None):
    """Use actual client identifiers; zero or absent IDs are not unique drafts."""
    game_id = _number(session.get("gameId"))
    if game_id > 0:
        return f"game:{game_id}"
    session_id = session.get("id")
    if isinstance(session_id, str) and 0 < len(session_id.strip()) <= 128:
        return "session:" + session_id.strip()
    data = gameflow.get("gameData") if isinstance(gameflow, dict) else None
    game_id = _number(data.get("gameId")) if isinstance(data, dict) else 0
    return f"game:{game_id}" if game_id > 0 else ""


def parse_draft_session(session: dict, mapping: dict, gameflow=None, current_summoner=None) -> DraftSnapshot:
    """Parse a snapshot. Riot id 0 is empty; internal id 0 remains a real champion."""
    if not isinstance(session, dict) or not isinstance(session.get("myTeam"), list):
        raise DraftImportError("A kliens nem adott érvényes hősválasztási állapotot. Nyiss meg egy draftot, majd próbáld újra.")
    if not isinstance(session.get("theirTeam", []), list):
        raise DraftImportError("A kliens hősválasztási adatai nem olvashatók. Próbáld újra a beolvasást.")
    own_team = resolve_own_side(session, gameflow, current_summoner)
    notes = []
    if own_team is None:
        notes.append("A kliens nem jelzi biztosan az oldaladat: alkalmazás előtt válaszd ki a Kék vagy Piros csapatot.")
    groups = session.get("actions", []) or []
    groups = groups if isinstance(groups, list) else []
    actions = []
    for group in groups:
        if isinstance(group, list):
            actions.extend(action for action in group if isinstance(action, dict))
        elif isinstance(group, dict):
            actions.append(group)
    pick_actions = {}
    for action in actions:
        if action.get("type") == "pick":
            cell = _number(action.get("actorCellId"), -1)
            if cell >= 0:
                ally = action.get("isAllyAction")
                pick_actions[(ally if isinstance(ally, bool) else None, cell)] = action

    def translate(riot_id):
        riot_id = _number(riot_id)
        if riot_id <= 0:
            return None
        value = mapping.get(str(riot_id), mapping.get(riot_id))
        if value is None:
            notes.append(f"Ismeretlen hősazonosító: {riot_id}. A hősadatok frissítése szükséges; ezt a bejegyzést kihagytam.")
            return None
        return _number(value, -1) if _number(value, -1) >= 0 else None

    picks, roster_cells = [], {}
    for source_key, source_team in (("myTeam", "own"), ("theirTeam", "enemy")):
        members = session.get(source_key, [])
        if len(members) > 5:
            raise DraftImportError("A beolvasás csak az 5 az 5 elleni draftot támogatja.")
        roster_cells[source_team] = set()
        for member in members:
            if not isinstance(member, dict):
                continue
            cell = _number(member.get("cellId"), -1)
            if cell >= 0:
                roster_cells[source_team].add(cell)
            action = pick_actions.get((source_team == "own", cell), pick_actions.get((None, cell)))
            if action is not None and action.get("completed") is not True:
                continue  # A hover/intent is never a locked pick.
            riot_id = _number(member.get("championId"))
            if riot_id <= 0 and action is not None and action.get("completed") is True:
                riot_id = _number(action.get("championId"))
            internal_id = translate(riot_id)
            if internal_id is None:
                continue
            role = _ROLES.get(str(member.get("assignedPosition", "")).upper())
            team = source_team
            if own_team:
                team = own_team if source_team == "own" else ("red" if own_team == "blue" else "blue")
            picks.append(DraftPick(team, riot_id, internal_id, role, cell))
    # Some client versions expose a locked action before adding its roster row.
    # Explicit isAllyAction is required; cell order never establishes a team.
    for (ally, cell), action in pick_actions.items():
        if ally is None or action.get("completed") is not True:
            continue
        source_team = "own" if ally else "enemy"
        if cell in roster_cells[source_team]:
            continue
        riot_id = _number(action.get("championId"))
        internal_id = translate(riot_id)
        if internal_id is None:
            continue
        team = (own_team if ally else ("red" if own_team == "blue" else "blue")) if own_team else source_team
        if sum(pick.team == team for pick in picks) >= 5 or any(pick.riot_id == riot_id for pick in picks):
            continue
        picks.append(DraftPick(team, riot_id, internal_id, None, cell))
    if any(p.role is None for p in picks):
        notes.append("A nem közölt szerepköröket kézzel rendeld pozícióhoz; a lista sorrendje nem jelenti az ösvényt.")

    bans = session.get("bans", {}) or {}
    bans = bans if isinstance(bans, dict) else {}
    def bans_for(source_key, members_key):
        champions = bans.get(source_key)
        champions = champions if isinstance(champions, (list, tuple)) else []
        if not any(_number(champion) > 0 for champion in champions):
            # cellId only establishes membership of myTeam/theirTeam. It is
            # never used to infer a blue/red side or lane.
            members = session.get(members_key, [])
            other_members = session.get("theirTeam" if members_key == "myTeam" else "myTeam", [])
            cells = {_number(member.get("cellId"), -1) for member in members if isinstance(member, dict)}
            other_cells = {_number(member.get("cellId"), -1) for member in other_members if isinstance(member, dict)}
            cells.difference_update(other_cells | {-1})
            champions = [action.get("championId") for action in actions
                         if action.get("type") == "ban" and action.get("completed") is True
                         and _number(action.get("actorCellId"), -1) in cells]
        return tuple(dict.fromkeys(value for champion in champions if (value := translate(champion)) is not None))

    own_bans = bans_for("myTeamBans", "myTeam")
    enemy_bans = bans_for("theirTeamBans", "theirTeam")
    blue_bans, red_bans = (), ()
    if own_team == "blue":
        blue_bans, red_bans = own_bans, enemy_bans
    elif own_team == "red":
        blue_bans, red_bans = enemy_bans, own_bans
    local_cell = _number(session.get("localPlayerCellId"), -1)
    local_members = [member for member in session["myTeam"] if isinstance(member, dict)
                     and local_cell >= 0 and _number(member.get("cellId"), -1) == local_cell]
    own_role = (_ROLES.get(str(local_members[0].get("assignedPosition", "")).upper())
                if len(local_members) == 1 else None)
    game_data = gameflow.get("gameData", {}) if isinstance(gameflow, dict) else {}
    queue = game_data.get("queue", {}) if isinstance(game_data, dict) else {}
    queue_id = _number(queue.get("id"), -1) if isinstance(queue, dict) else -1
    custom = session.get("isCustomGame") is True or (isinstance(game_data, dict) and game_data.get("isCustomGame") is True)
    sizes = (len(session["myTeam"]), len(session.get("theirTeam", [])))
    if custom:
        notes.append("Custom draft: csak a kliens által közölt, rögzített hősök olvashatók; a botok választásai hiányozhatnak.")
    return DraftSnapshot(tuple(picks), blue_bans, red_bans, own_team, own_bans, enemy_bans,
                         tuple(dict.fromkeys(notes)), _session_key(session, gameflow), own_role,
                         queue_id if queue_id >= 0 else None, custom, sizes)


class LcuDraftClient:
    """One-shot client. Call read_draft from a GUI worker, never the UI thread."""

    def __init__(self, lockfile_path=None, mapping=None, transport=None):
        self.lockfile_path = lockfile_path
        self.mapping = mapping
        self.transport = transport

    missing_message = "Jelenleg nincs aktív hősválasztás a LoL-kliensben. Lépj be egy draftba, majd olvasd be újra."

    def _allowed_endpoint(self, endpoint):
        return endpoint in {"/lol-champ-select/v1/session", "/lol-gameflow/v1/session", "/lol-summoner/v1/current-summoner"}

    def _get(self, transport, credentials, endpoint, optional=False):
        if not self._allowed_endpoint(endpoint):
            raise ValueError("Unsupported local endpoint")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", InsecureRequestWarning)
                response = transport.get(
                    f"https://127.0.0.1:{credentials.port}{endpoint}",
                    auth=("riot", credentials.token), verify=False,
                    timeout=(1, 2), allow_redirects=False,
                )
            if response.status_code in (204, 404):
                if optional:
                    return {}
                if endpoint == "/lol-champ-select/v1/session":
                    raise DraftUnavailable(self.missing_message)
                raise DraftImportError(self.missing_message)
            if response.status_code in (401, 403):
                raise DraftImportError("A kliens helyi kapcsolata lejárt. Indítsd újra a beolvasást, vagy válaszd ki az aktuális lockfile fájlt.")
            if not 200 <= response.status_code < 300:
                raise DraftImportError("A LoL-kliens helyi szolgáltatása nem válaszolt megfelelően. Próbáld újra a beolvasást.")
            value = response.json()
            return value if isinstance(value, dict) else {}
        except (requests.RequestException, ValueError):
            if optional:
                return {}
            raise DraftImportError("Nem érhető el a helyi LoL-kliens. Ellenőrizd, hogy fut-e, majd próbáld újra a beolvasást.") from None

    def read_draft(self) -> DraftSnapshot:
        candidates = discover_lockfiles(self.lockfile_path)
        if not candidates:
            raise DraftImportError("Nem található a LoL lockfile. Indítsd el a klienst, vagy válaszd ki a telepítési mappában lévő lockfile fájlt.")
        credentials = read_lockfile(candidates[0])
        path = normalize_lockfile(candidates[0])
        existing = saved_lockfile()
        remembered = (existing is not None and os.path.normcase(str(existing)) == os.path.normcase(str(path)))
        if not remembered:
            remembered = save_lockfile_path(path)
        mapping = self.mapping
        if mapping is None:
            path = Path(__file__).resolve().parents[1] / "data/champion_id.json"
            try:
                mapping = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                raise DraftImportError("Nem olvasható a helyi hősazonosító-lista. Ellenőrizd a data/champion_id.json fájlt.") from None
        transport = self.transport or requests.Session()
        if self.transport is None:
            transport.trust_env = False  # The local auth token must never go through a proxy.
        try:
            session = self._get(transport, credentials, "/lol-champ-select/v1/session")
            gameflow, current_summoner = {}, {}
            if resolve_own_side(session) is None:
                gameflow = self._get(transport, credentials, "/lol-gameflow/v1/session", optional=True)
                if resolve_own_side(session, gameflow) is None:
                    current_summoner = self._get(transport, credentials, "/lol-summoner/v1/current-summoner", optional=True)
            snapshot = parse_draft_session(session, mapping, gameflow, current_summoner)
            if not remembered:
                snapshot = replace(snapshot, warnings=snapshot.warnings + (
                    "A kliens útvonala ezen a gépen nem menthető. A beolvasás ettől még használható.",))
            return snapshot
        finally:
            if self.transport is None:
                transport.close()
