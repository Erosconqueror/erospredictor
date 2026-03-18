import json
import psycopg2
import configs as cfg
from pathlib import Path

class DataManager:
    def __init__(self):
        self.conn = psycopg2.connect(
            dbname=getattr(cfg, 'DB_NAME', 'erospredictor'),
            user=getattr(cfg, 'DB_USER', 'postgres'),
            password=getattr(cfg, 'DB_PASS', 'Eros'),
            host=getattr(cfg, 'DB_HOST', 'localhost'),
            port=getattr(cfg, 'DB_PORT', '5432')
        )
        self.create_tables()

    def create_tables(self):
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS matches (
                    match_id VARCHAR(50) PRIMARY KEY,
                    region VARCHAR(20),
                    tier VARCHAR(20),
                    patch VARCHAR(20),
                    blue_win BOOLEAN,
                    blue_team INTEGER[],
                    red_team INTEGER[],
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
        self.conn.commit()

    def save_match(self, match_id: str, region: str, minimal_data: dict):
        if not minimal_data:
            return False
            
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO matches (match_id, region, tier, patch, blue_win, blue_team, red_team)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (match_id) DO NOTHING;
            """, (
                match_id, 
                region, 
                minimal_data.get("tier", "UNKNOWN"),
                minimal_data.get("patch", "UNKNOWN"), 
                minimal_data.get("blue_win"),
                minimal_data.get("blue_team"),
                minimal_data.get("red_team")
            ))
        self.conn.commit()
        return True

    def get_match(self, match_id: str):
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT region, patch, tier, blue_win, blue_team, red_team 
                FROM matches WHERE match_id = %s;
            """, (match_id,))
            row = cur.fetchone()
            
            if row:
                return {
                    "region": row[0],
                    "patch": row[1],
                    "tier": row[2],
                    "blue_win": row[3],
                    "blue_team": row[4],
                    "red_team": row[5]
                }
        return None
    
    def get_all_match_ids(self):
        with self.conn.cursor() as cur:
            cur.execute("SELECT match_id FROM matches;")
            return [row[0] for row in cur.fetchall()]

    def get_champindex_by_id(self, champ_id: int):
        champ_data_path = Path(cfg.CHAMPION_DATA_PATH)
        if not champ_data_path.exists():
            return None
        with open(champ_data_path, 'r') as f:
            champ_data = json.load(f)
            return int(champ_data.get(str(champ_id)))
        
    def get_champion_names(self):
        with open(cfg.CHAMPION_NAMES_PATH, "r", encoding="utf-8") as f:
            return {int(k): v for k, v in json.load(f).items()}

    def close(self):
        if self.conn:
            self.conn.close()