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

        except requests.exceptions.HTTPError as err:
            print(f"Error fetching account by Riot ID: {err}")
            print(f"Response text: {response.text}")
            return None
    
    def get_match_data(self, match_id: str):
        try:
            raw_data = self.watcher.match.by_id(self.continent, match_id)
            info = raw_data.get("info", {})
            metadata = raw_data.get("metadata", {})

            simplified = {
                "matchId": metadata.get("matchId"),
                "gameVersion": info.get("gameVersion"),
                "gameDuration": info.get("gameDuration"),
                "queueId": info.get("queueId"),
                "teams": [
                    {
                        "teamId": t["teamId"],
                        "win": t["win"],
                    }
                    for t in info.get("teams", [])
                ],
                "participants": [
                    {
                        "championId": p["championId"],
                        "teamId": p["teamId"],
                        "summonerName": p["summonerName"],
                        "individualPosition": p.get("individualPosition"),
                    }
                    for p in info.get("participants", [])
                ],
            }
            return simplified
        except Exception as e:
            print(f"Error fetching match {match_id}: {e}")
            return None


    def get_matches_by_id(self, puuid, match_limit=20, tier=None):
        try:
            match_ids = self.watcher.match.matchlist_by_puuid(
                self.continent,
                puuid,
                count=match_limit
            )
        except ApiError as err:
            print(f"Error fetching match list: {err}")
            return []

        
        simplified_matches = []

        for match_id in match_ids:
            try:
                match_data = self.watcher.match.by_id(self.continent, match_id)
                info = match_data.get("info", {})
                metadata = match_data.get("metadata", {})

                if info.get("queueId") not in [420, 700]:
                    continue

                patch = "unknown"
                if "gameVersion" in info:
                    patch = info["gameVersion"].split(".")
                    patch = ".".join(patch[:2])

                simplified = {
                    "matchId": metadata.get("matchId"),
                    "tier": tier,
                    "gameVersion": patch,
                    "gameDuration": info.get("gameDuration"),
                    "queueId": info.get("queueId"),
                    "teams": [
                        {"teamId": t["teamId"], "win": t["win"]}
                        for t in info.get("teams", [])
                    ],
                    "participants": [
                        {
                            "championId": p["championId"],
                            "teamId": p["teamId"],
                            "individualPosition": p.get("individualPosition")
                        }
                        for p in info.get("participants", [])
                    ],
                }

                simplified_matches.append(simplified)
                time.sleep(0.07)  

            except ApiError as e:
                print(f"Error fetching match {match_id}: {e}")
                continue

        return simplified_matches

    def get_match_data(self, match_id):
        try:
            match_data = self.watcher.match.by_id(self.continent, match_id)
            return match_data
        except ApiError as err:
            print(f"An error occurred: {err}")
            return None
    
    def get_ranked_players(self, queue_type="RANKED_SOLO_5x5", tier="DIAMOND", division="I"):
        try:
            ranked_players = self.watcher.league.entries(self.region, queue_type, tier, division)
            return ranked_players
        except ApiError as err:
            print(f"An error occurred: {err}")
            return None
    
    