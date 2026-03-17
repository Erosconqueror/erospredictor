import json
import psycopg2
from psycopg2.extras import Json
import configs as cfg
from pathlib import Path
from data_manager import DataManager

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
                    game_version VARCHAR(20),
                    division VARCHAR(20),      
                    data JSONB,                
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
        self.conn.commit()

    def extract_patch_version(self, data: dict):
        try:
            info_dict = data.get('info', {})
            version_str = info_dict.get('gameVersion', '')
            
            if not version_str:
                version_str = data.get('gameVersion', '')
                
            if version_str:
                parts = version_str.split('.')
                if len(parts) >= 2:
                    return f"{parts[0]}.{parts[1]}"
            else:
                print(f"Figyelem! Nincs gameVersion. Info kulcsok: {list(info_dict.keys())[:10]}")
                
        except Exception as e:
            print(f"Hiba a patch verzio kinyeresekor: {e}")
        
        return "UNKNOWN"

    def save_match(self, match_id: str, region: str, data: dict, rank_info: str = None):
        patch = self.extract_patch_version(data)
        
        if rank_info:
            division = rank_info
        else:
            division = data.get("tier", "UNKNOWN")
            
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO matches (match_id, region, game_version, division, data)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (match_id) DO NOTHING;
            """, (match_id, region, patch, division, Json(data)))
        self.conn.commit()

    def get_match(self, match_id: str):
        with self.conn.cursor() as cur:
            cur.execute("SELECT region, data FROM matches WHERE match_id = %s;", (match_id,))
            row = cur.fetchone()
            if row:
                return {"region": row[0], "data": row[1]}
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

    def close(self):
        if self.conn:
            self.conn.close()