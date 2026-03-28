import requests
import urllib.parse
from riotwatcher import LolWatcher, ApiError
from configs import RIOT_API_KEY, REGION, CONTINENT
import time

class Riot:
    def __init__(self):
        self.watcher = LolWatcher(RIOT_API_KEY)
        self.region = REGION
        self.continent = CONTINENT
    
    def get_account_by_riot_id(self, game_name: str, tag_line: str):
        try:
            encoded_game_name = urllib.parse.quote(game_name)
            encoded_tag_line = urllib.parse.quote(tag_line)
            url = f"https://{self.continent}.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{encoded_game_name}/{encoded_tag_line}"
            headers = {"X-Riot-Token": RIOT_API_KEY}

            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.ConnectionError as err:
            print(f"Hálózati megszakadás a fiók lekérésekor: {err}")
            time.sleep(3)
            return None
        except requests.exceptions.HTTPError as err:
            print(f"Error fetching account by Riot ID: {err}")
            return None
        except Exception as e:
            print(f"Váratlan hiba a fiók lekérésekor: {e}")
            return None
            
    def get_league_exp_players(self, queue_type="RANKED_SOLO_5x5", tier="DIAMOND", division="I", page=1):
        try:
            if tier in ["CHALLENGER", "GRANDMASTER", "MASTER"]:
                if page > 1 or division != "I":
                    return []
                
                if tier == "CHALLENGER":
                    league = self.watcher.league.challenger_by_queue(self.region, queue_type)
                elif tier == "GRANDMASTER":
                    league = self.watcher.league.grandmaster_by_queue(self.region, queue_type)
                else:
                    league = self.watcher.league.masters_by_queue(self.region, queue_type)
                    
                return league.get("entries", [])
            else:
                players = self.watcher.league.entries(self.region, queue_type, tier, division, page=page)
                return players
        except ApiError as err:
            print(f"Riot API hiba a játékosok lekérésekor ({tier} {division}): {err}")
            return []
        except requests.exceptions.ConnectionError as err:
            print(f"Hálózati megszakadás (ConnectionReset)! Később újrapróbáljuk... ({err})")
            time.sleep(15)  
            return []
        except Exception as e:
            print(f"Váratlan hiba a játékosok lekérésekor: {e}")
            return []

    def get_match_ids_by_puuid(self, puuid, queue_id=420, match_limit=20):
        try:
            match_ids = self.watcher.match.matchlist_by_puuid(
                self.continent,
                puuid,
                queue=queue_id, 
                count=match_limit
            )
            return match_ids
        except ApiError as err:
            print(f"Error fetching match IDs for {puuid}: {err}")
            return []
        except requests.exceptions.ConnectionError:
            print(f"Hálózati megszakadás a meccs ID-k lekérésekor ({puuid}). Átugrás.")
            time.sleep(3)
            return []
        except Exception as e:
            print(f"Váratlan hiba a meccs ID-k lekérésekor: {e}")
            return []

    def _extract_minimal_match_data(self, raw_match, tier="UNKNOWN"):
        """Kivágja a felesleget a Riot JSON-ből, és csak a ML számára fontosat hagyja meg."""
        try:
            info = raw_match.get("info")
            if not info:
                return None

            participants = info.get("participants", [])
            if len(participants) != 10:
                return None 

            blue_win = info["teams"][0]["win"]
            
            blue_team = [p["championId"] for p in participants[:5]]
            red_team = [p["championId"] for p in participants[5:]]
            
            raw_patch = info.get("gameVersion", "UNKNOWN")
            patch = ".".join(raw_patch.split(".")[:2]) if raw_patch != "UNKNOWN" else raw_patch

            return {
                "tier": tier,
                "patch": patch,
                "blue_win": blue_win,
                "blue_team": blue_team,
                "red_team": red_team
            }
        except Exception as e:
            print(f"Hiba a meccs feldolgozásakor: {e}")
            return None

    def get_match_data(self, match_id: str, tier: str = "UNKNOWN"):
        """Lekéri a meccset, minimalizálja az adatot, majd visszatér vele."""
        try:
            raw_match_data = self.watcher.match.by_id(self.continent, match_id)
            
            clean_data = self._extract_minimal_match_data(raw_match_data, tier)
            return clean_data
            
        except ApiError as err:
            print(f"Error fetching full match data for {match_id}: {err}")
            return None
        except requests.exceptions.ConnectionError:
            print(f"Hálózati megszakadás meccs ({match_id}) lekérése közben. Átugrás.")
            time.sleep(3)
            return None
        except Exception as e:
            print(f"Váratlan hiba a meccs lekérésekor ({match_id}): {e}")
            return None