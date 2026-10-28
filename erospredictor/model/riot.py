import time
import os
import urllib.parse
import requests
from riotwatcher import LolWatcher
from configs import RIOT_API_KEY, REGION, CONTINENT
from services.riot_rate_limit import shared_limiter

class Riot:
    """Handles Riot API requests and data extraction."""
    
    def __init__(self, api_key=None, strict_errors=False, rate_limiter=None, max_rate_retries=5,
                 retry_transient=False):
        self.api_key = api_key or os.environ.get("RIOT_API_KEY") or RIOT_API_KEY
        self.strict_errors = strict_errors
        self.rate_limiter = rate_limiter if rate_limiter is not None else shared_limiter(self.api_key)
        self.max_rate_retries = max_rate_retries
        self.retry_transient = retry_transient
        self.api = LolWatcher(self.api_key, timeout=15, rate_limiter=self.rate_limiter)
        self.region = REGION
        self.cont = CONTINENT
    
    def get_account(self, name: str, tag: str) -> dict:
        """Fetches Riot account details."""
        try:
            q_name = urllib.parse.quote(name)
            q_tag = urllib.parse.quote(tag)
            url = f"https://{self.cont}.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{q_name}/{q_tag}"
            def request():
                self.rate_limiter.acquire(self.cont, "account", "by-riot-id")
                res = requests.get(url, headers={"X-Riot-Token": self.api_key}, timeout=15)
                self.rate_limiter.record_response(self.cont, "account", "by-riot-id", res.status_code, res.headers)
                res.raise_for_status()
                return res.json()
            return self._call(request, self.cont)
        except Exception as e:
            if getattr(getattr(e, "response", None), "status_code", None) == 404:
                return None
            if self.strict_errors:
                raise
            print(f"Account fetch error: {e}")
            return None
            
    def get_league_exp_players(self, tier: str, div: str, q_type: str = "RANKED_SOLO_5x5", page: int = 1) -> list:
        """Fetches players from a specific ranked division."""
        try:
            if tier in ["CHALLENGER", "GRANDMASTER", "MASTER"]:
                if page > 1 or div != "I":
                    return []
                
                if tier == "CHALLENGER":
                    league = self._call(lambda: self.api.league.challenger_by_queue(self.region, q_type), self.region)
                elif tier == "GRANDMASTER":
                    league = self._call(lambda: self.api.league.grandmaster_by_queue(self.region, q_type), self.region)
                else:
                    league = self._call(lambda: self.api.league.masters_by_queue(self.region, q_type), self.region)
                    
                return league.get("entries", [])
            return self._call(lambda: self.api.league.entries(self.region, q_type, tier, div, page=page), self.region)
        except requests.exceptions.ConnectionError:
            if self.strict_errors:
                raise
            time.sleep(15)
            return []
        except Exception as e:
            if getattr(getattr(e, "response", None), "status_code", None) == 404:
                return []
            if self.strict_errors:
                raise
            print(f"Player fetch error: {e}")
            return []

    def get_match_ids(self, puuid: str, q_id: int = 420, limit: int = 20) -> list:
        """Fetches match IDs for a specific player."""
        try:
            return self._call(lambda: self.api.match.matchlist_by_puuid(self.cont, puuid, queue=q_id, count=limit), self.cont)
        except requests.exceptions.ConnectionError:
            if self.strict_errors:
                raise
            time.sleep(3)
            return []
        except Exception as e:
            if getattr(getattr(e, "response", None), "status_code", None) == 404:
                return []
            if self.strict_errors:
                raise
            print(f"Match ID fetch error: {e}")
            return []

    def _clean_data(self, raw: dict, tier: str) -> dict:
        """Extracts minimal required data from raw Riot JSON."""
        info = raw.get("info")
        if not info or len(info.get("participants", [])) != 10:
            return None

        if info.get("queueId") != 420 or info.get("gameDuration", 0) < 300:
            return None
        parts = info["participants"]
        roles = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
        teams = {}
        for team_id in (100, 200):
            participants = [p for p in parts if p.get("teamId") == team_id]
            by_role = {p.get("teamPosition"): p.get("championId") for p in participants}
            if len(participants) != 5 or set(by_role) != set(roles):
                return None
            teams[team_id] = [by_role[role] for role in roles]
        outcomes = {team.get("teamId"): team.get("win") for team in info.get("teams", [])}
        if not isinstance(outcomes.get(100), bool):
            return None
        blue_win = outcomes[100]
        b_team, r_team = teams[100], teams[200]
        
        raw_patch = info.get("gameVersion", "UNKNOWN")
        patch = ".".join(raw_patch.split(".")[:2]) if raw_patch != "UNKNOWN" else "UNKNOWN"

        return {
            "tier": tier,
            "patch": patch,
            "blue_win": blue_win,
            "blue_team": b_team,
            "red_team": r_team
        }

    def get_match_data(self, match_id: str, tier: str = "UNKNOWN") -> dict:
        """Fetches and cleans a single match by ID."""
        try:
            raw = self._call(lambda: self.api.match.by_id(self.cont, match_id), self.cont)
            return self._clean_data(raw, tier)
        except requests.exceptions.ConnectionError:
            if self.strict_errors:
                raise
            time.sleep(3)
            return None
        except Exception as e:
            if getattr(getattr(e, "response", None), "status_code", None) == 404:
                return None
            if self.strict_errors:
                raise
            print(f"Match fetch error ({match_id}): {e}")
            return None

    def _call(self, operation, region):
        return self.rate_limiter.call(operation, region, self.max_rate_retries,
                                      retry_transient=self.retry_transient)
