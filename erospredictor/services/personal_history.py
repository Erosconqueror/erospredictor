"""Own-account LCU history and descriptive evidence; never changes model probabilities.

Schema: https://github.com/KebsCS/lcu-and-riotclient-api (LCU extracted Swagger).
Local GETs by default; optional own-account Riot client service after opt-in.
No opponent account lookup and no Riot developer key required.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import re

import requests
from services import feature_flags
from services.client_settings import settings_path
from services.draft_import import LcuDraftClient, DraftImportError, discover_lockfiles, read_lockfile
from services.update_progress import atomic_json
from services.client_history_service import REGION, ENTITLEMENTS, RiotClientHistory, server_from_config


MIN_PLAYED = 3
MIN_GOOD_GAMES = 20
MIN_PAIR_GAMES = 20
GOOD_WINRATE = 55.0
MAX_STORED = 1000
CURRENT = "/lol-summoner/v1/current-summoner"
HISTORY = "/lol-match-history/v1/products/lol/current-summoner/matches"
ROLES = {"TOP": 0, "JUNGLE": 1, "MIDDLE": 2, "MID": 2, "BOTTOM": 3, "ADC": 3, "UTILITY": 4, "SUPPORT": 4}


def number(value):
    if isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return 0


def participant_role(participant):
    for key in ("teamPosition", "individualPosition"):
        role = ROLES.get(str(participant.get(key, "")).upper())
        if role is not None:
            return role
    timeline = participant.get("timeline") or {}
    if not isinstance(timeline, dict):
        return None
    lane, role = str(timeline.get("lane", "")).upper(), str(timeline.get("role", "")).upper()
    if lane in ("TOP", "JUNGLE", "MIDDLE", "MID"):
        return ROLES[lane]
    if lane in ("BOTTOM", "BOT"):
        return {"DUO_CARRY": 3, "CARRY": 3, "DUO_SUPPORT": 4, "SUPPORT": 4}.get(role)
    return None


def same_account(player, account):
    if player.get("puuid"):
        return player["puuid"] == account["puuid"]
    return number(player.get("summonerId")) > 0 and number(player.get("summonerId")) == number(account.get("summonerId"))


def parse_game(game, account):
    if not isinstance(game, dict) or number(game.get("queueId")) not in (420, 440):
        return None
    if number(game.get("mapId")) != 11 or number(game.get("gameDuration")) < 300:
        return None
    if number(game.get("gameId")) <= 0:
        return None
    participants = game.get("participants")
    identities = game.get("participantIdentities", [])
    if not isinstance(participants, list) or not isinstance(identities, list):
        return None
    ids = {number(item.get("participantId")) for item in identities if isinstance(item, dict)
           and isinstance(item.get("player"), dict) and same_account(item["player"], account)}
    own = [item for item in participants if isinstance(item, dict)
           and (number(item.get("participantId")) in ids or item.get("puuid") == account["puuid"])]
    if len(own) != 1:
        return None
    own = own[0]
    stats = own.get("stats") or {}
    win = stats.get("win", own.get("win")) if isinstance(stats, dict) else None
    # A normal early surrender (e.g. FF15) is a real result, including a loss.
    # Filtering it by the API flag would bias the personal win rate upward.
    if not isinstance(win, bool):
        return None
    champion, team, role = number(own.get("championId")), number(own.get("teamId")), participant_role(own)
    if champion <= 0 or team not in (100, 200):
        return None
    foes = [item for item in participants if isinstance(item, dict) and number(item.get("teamId")) in (100, 200)
            and number(item.get("teamId")) != team and role is not None and participant_role(item) == role]
    enemy = number(foes[0].get("championId")) if len(foes) == 1 else 0
    allies = [{"champion": number(item.get("championId")), "role": participant_role(item)}
              for item in participants if isinstance(item, dict) and item is not own
              and number(item.get("teamId")) == team and number(item.get("championId")) > 0]
    return {"match_id": f"{game.get('platformId', '')}:{number(game['gameId'])}",
            "created": number(game.get("gameCreation")), "champion": champion, "role": role,
            "win": win, "enemy": enemy or None, "allies": allies,
            "complete": len(participants) == 10}


def tally(rows):
    wins = sum(row["win"] for row in rows)
    return {"games": len(rows), "wins": wins, "winrate": 100 * wins / len(rows) if rows else None}


@dataclass(frozen=True)
class PersonalHistory:
    puuid: str
    name: str
    games: tuple
    updated_at: str = ""
    warnings: tuple = ()
    requested_games: int = 0
    examined_games: int = 0
    ranked_games_seen: int = 0
    source: str = "lcu"
    limited: bool = False

    def evidence(self, cid, role, enemy, allies, mapping):
        inverse = {int(index): number(riot) for riot, index in mapping.items()}
        riot = inverse.get(cid)
        played = [row for row in self.games if row["champion"] == riot] if riot else []
        if not played:
            return None
        on_role = [row for row in played if row["role"] == role]
        relevant = on_role if len(on_role) >= MIN_PLAYED else played
        summary = tally(relevant)
        marker = "good" if summary["games"] >= MIN_GOOD_GAMES and summary["winrate"] >= GOOD_WINRATE else (
            "played" if len(played) >= MIN_PLAYED else None)
        lane = [row for row in on_role if inverse.get(enemy) and row["enemy"] == inverse[enemy]]
        matchup = dict(tally(lane), champion_id=enemy, enough=len(lane) >= MIN_PAIR_GAMES) if lane else None
        synergy = []
        for ally_role, ally in enumerate(allies):
            if ally < 0 or ally_role == role or ally not in inverse:
                continue
            rows = [row for row in on_role if any(item["champion"] == inverse[ally] and item["role"] == ally_role
                                                 for item in row["allies"])]
            if rows:
                synergy.append(dict(tally(rows), champion_id=ally, role=ally_role, enough=len(rows) >= MIN_PAIR_GAMES))
        return dict(summary, marker=marker, total_games=len(played), role_games=len(on_role),
                    scope="role" if relevant is on_role else "all", matchup=matchup, synergies=synergy)


def annotate_recommendations(recommendations, history, mapping, blue, red, team, role):
    own, enemy = (blue, red) if team == "blue" else (red, blue)
    return [dict(item, personal=history.evidence(item["id"], role, enemy[role], own, mapping))
            for item in recommendations]


class LcuHistoryClient(LcuDraftClient):
    missing_message = "A saját meccselőzmények jelenleg nem érhetők el a LoL-kliensből. Próbáld újra a frissítésgombbal."

    def _allowed_endpoint(self, endpoint):
        return endpoint in (CURRENT, REGION, ENTITLEMENTS) or bool(re.fullmatch(
            r"/lol-match-history/v1/(?:products/lol/current-summoner/matches\?begIndex=\d{1,4}&endIndex=\d{1,4}|games/[1-9]\d{0,19})", endpoint))

    @contextmanager
    def _connection(self):
        paths = discover_lockfiles(self.lockfile_path)
        if not paths:
            raise DraftImportError("Indítsd el a LoL-kliensedet, majd frissítsd a saját előzményeket.")
        credentials = read_lockfile(paths[0])
        self._system_config_path = paths[0].parent / "system.yaml"
        transport = self.transport or requests.Session()
        if self.transport is None:
            transport.trust_env = False
        try:
            yield transport, credentials
        finally:
            if self.transport is None:
                transport.close()

    def _account(self, transport, credentials):
        account = self._get(transport, credentials, CURRENT)
        if not isinstance(account.get("puuid"), str) or not account["puuid"].strip():
            raise DraftImportError("Jelentkezz be a LoL-kliensbe a saját meccsek betöltéséhez.")
        return account

    def verify_account(self, history):
        if not feature_flags.PERSONAL_HISTORY_ENABLED:
            raise DraftImportError("A match history funkció jelenleg szüneteltetve van.")
        with self._connection() as (transport, credentials):
            if self._account(transport, credentials)["puuid"] != history.puuid:
                raise DraftImportError("Másik fiók van bejelentkezve. Frissítsd a saját meccselőzményeket!")

    @staticmethod
    def _cache_path(puuid):
        return settings_path().parent / "history" / (hashlib.sha256(puuid.encode()).hexdigest() + ".json")

    def _cached(self, account):
        try:
            cache = json.loads(self._cache_path(account["puuid"]).read_text(encoding="utf-8"))
            if cache.get("puuid") != account["puuid"] or cache.get("version") != 1:
                return {}
            rows = cache["games"]
            if not isinstance(rows, list) or len(rows) > MAX_STORED:
                return {}
            for row in rows:
                if (not isinstance(row, dict) or not {"match_id", "win", "champion", "role", "enemy", "allies", "created"}.issubset(row)
                        or not isinstance(row.get("match_id"), str)
                        or not isinstance(row.get("win"), bool) or number(row.get("champion")) <= 0
                        or row.get("role") not in (None, 0, 1, 2, 3, 4)
                        or not isinstance(row.get("allies"), list)
                        or "enemy" not in row
                        or not isinstance(row.get("created"), int)
                        or any(not isinstance(item, dict) or number(item.get("champion")) <= 0
                               or item.get("role") not in (None, 0, 1, 2, 3, 4) for item in row["allies"])):
                    return {}
            return {row["match_id"]: row for row in rows}
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            return {}

    def _local_pages(self, transport, credentials, limit, cancelled):
        for start in range(0, limit, 20):
            if cancelled():
                raise DraftImportError("Az előzmények beolvasása leállítva; a mentett oldalak megmaradtak.")
            response = self._get(transport, credentials, f"{HISTORY}?begIndex={start}&endIndex={start + 19}")
            group = response.get("games")
            games = group.get("games") if isinstance(group, dict) else None
            if not isinstance(games, list):
                raise DraftImportError(self.missing_message)
            if not games:
                return
            yield [dict(game, platformId=game.get("platformId") or response.get("platformId", ""))
                   for game in games[:20] if isinstance(game, dict)]
            if len(games) < 20:
                return

    def _client_service(self, transport, credentials, account):
        # Only called after explicit opt-in. The default local path reads neither
        # the session access token nor the remote server configuration.
        region = self._get(transport, credentials, REGION).get("region")
        server = server_from_config(self._system_config_path, region)
        session = self._get(transport, credentials, ENTITLEMENTS)
        if ((session.get("subject") and session["subject"] != account["puuid"])
                or self._account(transport, credentials)["puuid"] != account["puuid"]):
            raise DraftImportError("Fiókváltás történt beolvasás közben. Frissítsd újra az előzményeket!")
        return RiotClientHistory(server, session.get("accessToken"), account["puuid"])

    def read_history(self, limit=100, cancelled=lambda: False, allow_client_service=False):
        if not feature_flags.PERSONAL_HISTORY_ENABLED:
            raise DraftImportError("A match history funkció jelenleg szüneteltetve van.")
        if limit not in (100, 500, 1000):
            raise ValueError("History limit must be 100, 500 or 1000")
        with self._connection() as (transport, credentials):
            account = self._account(transport, credentials)
            rows, seen, detail_failures, warnings = self._cached(account), set(), 0, []
            ranked_seen, unreadable, limited = 0, 0, False
            source = "riot-client-service" if allow_client_service else "lcu"
            stamp = datetime.now(timezone.utc).isoformat()
            pages = (self._client_service(transport, credentials, account).pages(limit, cancelled)
                     if allow_client_service else self._local_pages(transport, credentials, limit, cancelled))
            for page_index, games in enumerate(pages):
                if cancelled():
                    raise DraftImportError("Az előzmények beolvasása leállítva. A már mentett oldalak megmaradtak.")
                fresh = 0
                for original in games:
                    if cancelled():
                        raise DraftImportError("Az előzmények beolvasása leállítva. A már mentett oldalak megmaradtak.")
                    if not isinstance(original, dict):
                        continue
                    game = dict(original)
                    key = f"{game.get('platformId', '')}:{number(game.get('gameId'))}"
                    if number(game.get("gameId")) <= 0 or key in seen:
                        continue
                    seen.add(key)
                    fresh += 1
                    queue = number(game.get("queueId"))
                    ranked_seen += queue in (420, 440)
                    row = parse_game(game, account)
                    if (queue not in (420, 440) or number(game.get("mapId")) != 11
                            or number(game.get("gameDuration")) < 300):
                        continue
                    if row is not None and row["match_id"] in rows and rows[row["match_id"]].get("complete"):
                        continue
                    if (row is None or not row["complete"]) and detail_failures < 3:
                        try:
                            detail = self._get(transport, credentials, f"/lol-match-history/v1/games/{number(game['gameId'])}", optional=True)
                            detail = dict(detail, platformId=detail.get("platformId") or game.get("platformId", ""))
                            parsed = parse_game(detail, account)
                            if (parsed and number(detail.get("gameId")) == number(game["gameId"])
                                    and (not game.get("platformId") or detail["platformId"] == game["platformId"])):
                                row = parsed
                            else:
                                detail_failures += 1
                        except DraftImportError:
                            detail_failures += 1
                    if row is None:
                        unreadable += 1
                        continue
                    rows[row["match_id"]] = row
                if self._account(transport, credentials)["puuid"] != account["puuid"]:
                    raise DraftImportError("Fiókváltás történt beolvasás közben. Frissítsd újra az előzményeket!")
                recent = sorted(rows.values(), key=lambda row: (row["created"], row["match_id"]), reverse=True)[:MAX_STORED]
                rows = {row["match_id"]: row for row in recent}
                try:
                    atomic_json(self._cache_path(account["puuid"]), {"version": 1, "puuid": account["puuid"], "games": recent})
                except OSError:
                    warnings.append("A helyi előzménygyorsítótár nem menthető; a betöltött adatok most használhatók.")
                # Some client builds repeat the first page for larger offsets.
                if not fresh and page_index > 0:
                    limited = True
                    warnings.append(f"A kliens ugyanazt az előzményoldalt ismételte: {len(seen)}/{limit} különböző meccs volt elérhető. "
                        "Ez nem azt jelenti, hogy nincs régebbi rangsorolt meccsed. " +
                        ("A régebbi előzményekhez próbáld ki a külön Riot-kliensszolgáltatás opciót."
                         if not allow_client_service else "A Riot szolgáltatása nem adott további különböző meccseket."))
                    break
                if len(seen) >= limit:
                    break
            if detail_failures:
                warnings.append("Néhány meccs részlete hiányzik; ismeretlen lane-ből nem számolunk párosítási statisztikát.")
            if unreadable:
                warnings.append(f"{unreadable} rangsorolt meccsben nem volt olvasható a saját résztvevő; ezeket kihagytuk.")
            if not ranked_seen and seen:
                warnings.append(f"A kapott {len(seen)} meccs között nincs Ranked Solo/Duo (420) vagy Ranked Flex (440). "
                                "A többi játékmódból nem számolunk rangsorolt hősstatisztikát.")
            if self._account(transport, credentials)["puuid"] != account["puuid"]:
                raise DraftImportError("Fiókváltás történt beolvasás közben. Frissítsd újra az előzményeket!")
            name = account.get("gameName") or account.get("displayName") or "Saját fiók"
            if account.get("tagLine"):
                name += "#" + account["tagLine"]
            return PersonalHistory(account["puuid"], str(name), tuple(sorted(rows.values(), key=lambda row: row["created"], reverse=True)),
                                   stamp, tuple(dict.fromkeys(warnings)), limit, len(seen), ranked_seen, source, limited)
