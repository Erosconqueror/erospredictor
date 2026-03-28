import time
from model.riot import Riot
from model.data_manager import DataManager
import configs as cfg

def run_continuous_scraper():
    riot = Riot()
    db = DataManager()
    
    print("Starting continuous scraper...")
    print(f"Allowed patches: {cfg.ALLOWED_PATCHES}")
    
    while True:
        for tier in cfg.TARGET_TIERS:
            
            for division in cfg.TARGET_DIVISIONS:
                if tier in ["CHALLENGER", "GRANDMASTER", "MASTER"] and division != "I":
                    continue
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
                                
                            match_data = riot.get_match_data(match_id, tier)
                            if not match_data:
                                print(f"  [!] Hiba a letoltesnel: {match_id}")
                                continue
                                
                            patch = match_data.get("patch", "UNKNOWN")
                            
                            if patch in cfg.ALLOWED_PATCHES:
                                if match_data:  
                                    db.save_match(match_id, cfg.REGION, match_data)
                                print(f"  [+] MENTVE! {match_id} | Patch: {patch}")
                            else:
                                print(f"  [x] Kiszurve (rossz patch): {match_id} | Patch: {patch}")
                                
                    page += 1
                    if page >= 10:  
                        print("Túl sok oldal, továbblépünk a következő ligára. ")
                        time.sleep(10)
                        break
                    
               

if __name__ == "__main__":
    try:
        run_continuous_scraper()
    except KeyboardInterrupt:
        print("\nScraper stopped by user.")