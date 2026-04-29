import time
import urllib.parse
import requests
from riotwatcher import LolWatcher
from configs import RIOT_API_KEY, REGION, CONTINENT

class Riot:
    """Handles Riot API requests and data extraction."""
    
    def __init__(self):
        self.api = LolWatcher(RIOT_API_KEY)
        self.region = REGION
        self.cont = CONTINENT
    
    def get_account(self, name: str, tag: str) -> dict:
        """Fetches Riot account details."""
        try:
            q_name = urllib.parse.quote(name)
            q_tag = urllib.parse.quote(tag)
            url = f"https://{self.cont}.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{q_name}/{q_tag}"
            res = requests.get(url, headers={"X-Riot-Token": RIOT_API_KEY})
            res.raise_for_status()
            return res.json()
        except Exception as e:
            print(f"Account fetch error: {e}")
            return None
            
    def get_league_exp_players(self, tier: str, div: str, q_type: str = "RANKED_SOLO_5x5", page: int = 1) -> list:
        """Fetches players from a specific ranked division."""
        try:
            if tier in ["CHALLENGER", "GRANDMASTER", "MASTER"]:
                if page > 1 or div != "I":
                    return []
                
                if tier == "CHALLENGER":
                    league = self.api.league.challenger_by_queue(self.region, q_type)
                elif tier == "GRANDMASTER":
                    league = self.api.league.grandmaster_by_queue(self.region, q_type)
                else:
                    league = self.api.league.masters_by_queue(self.region, q_type)
                    
                return league.get("entries", [])
            return self.api.league.entries(self.region, q_type, tier, div, page=page)
        except requests.exceptions.ConnectionError:
            time.sleep(15)
            return []
        except Exception as e:
            print(f"Player fetch error: {e}")
            return []

    def get_match_ids(self, puuid: str, q_id: int = 420, limit: int = 20) -> list:
        """Fetches match IDs for a specific player."""
        try:
            return self.api.match.matchlist_by_puuid(self.cont, puuid, queue=q_id, count=limit)
        except requests.exceptions.ConnectionError:
            time.sleep(3)
            return []
        except Exception as e:
            print(f"Match ID fetch error: {e}")
            return []

    def _clean_data(self, raw: dict, tier: str) -> dict:
        """Extracts minimal required data from raw Riot JSON."""
        info = raw.get("info")
        if not info or len(info.get("participants", [])) != 10:
            return None

        parts = info["participants"]
        blue_win = info["teams"][0]["win"]
        b_team = [p["championId"] for p in parts[:5]]
        r_team = [p["championId"] for p in parts[5:]]
        
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
            raw = self.api.match.by_id(self.cont, match_id)
            return self._clean_data(raw, tier)
        except requests.exceptions.ConnectionError:
            time.sleep(3)
            return None
        except Exception as e:
            print(f"Match fetch error ({match_id}): {e}")
            return None