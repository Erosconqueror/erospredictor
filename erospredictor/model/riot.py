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
                    league = self.watcher.league.master_by_queue(self.region, queue_type)
                    
                return league.get("entries", [])
            else:
                players = self.watcher.league.entries(self.region, queue_type, tier, division, page=page)
                return players
        except ApiError as err:
            print(f"Error fetching players for {tier} {division} page {page}: {err}")
            return []

    def get_match_ids_by_puuid(self, puuid, queue_id=420, match_limit=20):
        try:
            match_ids = self.watcher.match.matchlist_by_puuid(
                self.continent,
                puuid,
                queue=queue_id, # 420 a Ranked Solo/Duo
                count=match_limit
            )
            return match_ids
        except ApiError as err:
            print(f"Error fetching match IDs for {puuid}: {err}")
            return []

    def get_raw_match_data(self, match_id: str):

        try:
            match_data = self.watcher.match.by_id(self.continent, match_id)
            return match_data
        except ApiError as err:
            print(f"Error fetching full match data for {match_id}: {err}")
            return None