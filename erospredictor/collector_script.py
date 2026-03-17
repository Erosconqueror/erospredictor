import time
from model.riot import Riot
from model.data_manager import DataManager
from configs import TARGET_TIERS, TARGET_DIVISIONS, ALLOWED_PATCHES

def run_continuous_scraper():
    riot = Riot()
    db = DataManager()
    
    print("Starting continuous scraper...")
    print(f"Allowed patches: {ALLOWED_PATCHES}")
    
    while True:
        for tier in TARGET_TIERS:
            if tier in ["CHALLENGER", "GRANDMASTER", "MASTER"] and division != "I":
                    continue
            for division in TARGET_DIVISIONS:
                page = 1
                while True:
                    print(f"\n--- Fetching players: {tier} {division} Page {page} ---")
                    players = riot.get_league_exp_players(tier=tier, division=division, page=page)
                    
                    if not players:
                        print("Nincs tobb jatekos ezen az oldalon.")
                        break
                        
                    for player in players:
                        puuid = player.get("puuid")
                        summoner_name = player.get("summonerName", "Unknown")
                        
                        if not puuid:
                            continue
                            
                        print(f"\nJatekos: {summoner_name} | PUUID: {puuid[:15]}...")
                        match_ids = riot.get_match_ids_by_puuid(puuid)
                        print(f"  Talalt meccsek: {len(match_ids)} db -> {match_ids[:3]}...")
                        
                        for match_id in match_ids:
                            if db.get_match(match_id):
                                print(f"  [-] Mar az adatbazisban: {match_id}")
                                continue
                                
                            match_data = riot.get_raw_match_data(match_id)
                            if not match_data:
                                print(f"  [!] Hiba a letoltesnel: {match_id}")
                                continue
                                
                            patch = db.extract_patch_version(match_data)
                            
                            if patch in ALLOWED_PATCHES:
                                db.save_match(match_id, riot.region, match_data, rank_info=tier)
                                print(f"  [+] MENTVE! {match_id} | Patch: {patch}")
                            else:
                                print(f"  [x] Kiszurve (rossz patch): {match_id} | Patch: {patch}")
                                
                    page += 1
                    
        print("Cycle finished. Sleeping for 1 hour...")
        time.sleep(3600)

if __name__ == "__main__":
    try:
        run_continuous_scraper()
    except KeyboardInterrupt:
        print("\nScraper stopped by user.")