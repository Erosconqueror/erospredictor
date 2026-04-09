import time
import configs as cfg
from model.riot import Riot
from model.data_manager import DataManager

def run_scraper():
    print("=== Erospredictor Match Collector ===")
    print("1. Default continuous scraping (All config tiers, updated limits)")
    print("2. Custom scraping (Specific continent, region, and tier)")
    
    mode = input("Select mode (1/2) [1]: ").strip() or "1"
    
    if mode == "2":
        cont = input(f"Continent (asia/europe/americas) [{cfg.CONTINENT}]: ").strip().lower() or cfg.CONTINENT
        reg = input(f"Region (e.g., euw1, eun1, kr) [{cfg.REGION}]: ").strip().lower() or cfg.REGION
        tier_input = input("Tier (e.g., DIAMOND, MASTER) or ALL [ALL]: ").strip().upper() or "ALL"
        tiers_to_scrape = cfg.TARGET_TIERS if tier_input == "ALL" else [tier_input]
    else:
        cont = cfg.CONTINENT
        reg = cfg.REGION
        tiers_to_scrape = cfg.TARGET_TIERS
    
    riot = Riot()
    riot.cont = cont
    riot.region = reg
    db = DataManager(dev_mode=True)
    
    print("\nStarting scraper...")
    print(f"Server: {cont} / {reg} | Tiers: {', '.join(tiers_to_scrape)}")
    print(f"Allowed patches: {cfg.ALLOWED_PATCHES}")
    
    while True:
        for tier in tiers_to_scrape:
            divs = ["I"] if tier in ["CHALLENGER", "GRANDMASTER", "MASTER"] else cfg.TARGET_DIVISIONS
            
            for div in divs:
                page = 1
                while True:
                    print(f"\n--- Fetching players: {tier} {div} Page {page} ---")
                    players = riot.get_league_exp_players(tier=tier, div=div, page=page)
                    
                    if not players:
                        print("No more players on this page.")
                        break
                        
                    for p in players:
                        puuid = p.get("puuid")
                        if not puuid:
                            continue
                            
                        print(f"\nNew Player --- ")
                        m_ids = riot.get_match_ids(puuid, limit=50) 
                        print(f"  FOUND MATCHES: {len(m_ids)} item(s)  -> {m_ids[:3]}...")
                        
                        for m_id in m_ids:
                            if db.get_match(m_id):
                                print(f"  [-] Already in database: {m_id}")
                                continue
                                
                            m_data = riot.get_match_data(m_id, tier)
                            if not m_data:
                                print(f"  [!] Error downloading: {m_id}")
                                continue
                                
                            patch = m_data.get("patch", "UNKNOWN")
                            if patch in cfg.ALLOWED_PATCHES:
                                db.save_match(m_id, reg, m_data)
                                print(f"  [+] SAVED MATCH {m_id} | Patch: {patch}")
                            else:
                                print(f"  [x] WRONG PATCH: {m_id} | Patch: {patch}")
                                
                    page += 1
                    if page >= 3:  
                        print("Too many pages read, moving to the next division/tier.")
                        time.sleep(5)
                        break

if __name__ == "__main__":
    try:
        run_scraper()
    except KeyboardInterrupt:
        print("\nScraper stopped by user.")