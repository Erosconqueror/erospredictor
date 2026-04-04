import time
import configs as cfg
from model.riot import Riot
from model.data_manager import DataManager

def run_scraper():
    """Continuously scrapes matches from Riot API based on config criteria."""
    riot = Riot()
    db = DataManager()
    
    print("Starting continuous scraper...")
    print(f"Allowed patches: {cfg.ALLOWED_PATCHES}")
    
    while True:
        for tier in cfg.TARGET_TIERS:
            for div in cfg.TARGET_DIVISIONS:
                if tier in ["CHALLENGER", "GRANDMASTER", "MASTER"] and div != "I":
                    continue
                    
                page = 1
                while True:
                    print(f"\n--- Fetching players: {tier} {div} Page {page} ---")
                    players = riot.get_league_exp_players(tier=tier, div=div, page=page)
                    
                    if not players:
                        print("No more players on this page")
                        break
                        
                    for p in players:
                        puuid = p.get("puuid")
                        name = p.get("summonerName", "Unknown")
                        
                        if not puuid:
                            continue
                            
                        print(f"\nNew Player --- ")
                        m_ids = riot.get_match_ids(puuid)
                        print(f"  FOUND MATCHES: {len(m_ids)} item(s)  -> {m_ids[:3]}...")
                        
                        for m_id in m_ids:
                            if db.get_match(m_id):
                                print(f"  [-] Already in database: {m_id}")
                                continue
                                
                            m_data = riot.get_match_data(m_id, tier)
                            if not m_data:
                                print(f"  [!] Hiba a letoltesnel: {m_id}")
                                continue
                                
                            patch = m_data.get("patch", "UNKNOWN")
                            if patch in cfg.ALLOWED_PATCHES:
                                db.save_match(m_id, cfg.REGION, m_data)
                                print(f"  [+] SAVED MATCH {m_id} | Patch: {patch}")
                            else:
                                print(f"  [x] WRONG PATCH: {m_id} | Patch: {patch}")
                                
                    page += 1
                    if page >= 10:  
                        print("Too many pages read, moving to the next tier")
                        time.sleep(10)
                        break

if __name__ == "__main__":
    try:
        run_scraper()
    except KeyboardInterrupt:
        print("\nScraper stopped by user.")