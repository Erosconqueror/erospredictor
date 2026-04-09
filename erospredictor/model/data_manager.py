import json
import psycopg2
import configs as cfg
from pathlib import Path

class DataManager:
    """Handles PostgreSQL database operations and static JSON parsing."""

    def __init__(self, dev_mode: bool = False ):
        if dev_mode:
            self.conn = psycopg2.connect(
                dbname=getattr(cfg, 'DB_NAME', 'erospredictor'),
                user=getattr(cfg, 'DB_USER', 'postgres'),
                password=getattr(cfg, 'DB_PASS', 'Eros'),
                host=getattr(cfg, 'DB_HOST', 'localhost'),
                port=getattr(cfg, 'DB_PORT', '5432')
            )
            self._init_tables()
        else:
            self.conn = None
            self.cur = None

    def _init_tables(self):
        """Creates the necessary schema if it does not exist."""
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

    def save_match(self, match_id: str, region: str, data: dict) -> bool:
        """Inserts a single match record into the database."""
        if not data:
            return False
            
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO matches (match_id, region, tier, patch, blue_win, blue_team, red_team)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (match_id) DO NOTHING;
            """, (
                match_id, 
                region, 
                data.get("tier", "UNKNOWN"),
                data.get("patch", "UNKNOWN"), 
                data.get("blue_win"),
                data.get("blue_team"),
                data.get("red_team")
            ))
        self.conn.commit()
        return True

    def get_match(self, match_id: str) -> dict:
        """Retrieves a single match record by ID."""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT region, patch, tier, blue_win, blue_team, red_team 
                FROM matches WHERE match_id = %s;
            """, (match_id,))
            row = cur.fetchone()
            
            if row:
                return {
                    "region": row[0], "patch": row[1], "tier": row[2],
                    "blue_win": row[3], "blue_team": row[4], "red_team": row[5]
                }
        return None
    
    def get_all_match_ids(self) -> list:
        """Returns a list of all stored match IDs."""
        with self.conn.cursor() as cur:
            cur.execute("SELECT match_id FROM matches;")
            return [row[0] for row in cur.fetchall()]

    def get_all_matches(self) -> list:
        """Returns a list of dictionaries containing all stored matches."""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT match_id, region, patch, tier, blue_win, blue_team, red_team 
                FROM matches;
            """)
            
            return [{
                "match_id": r[0], "region": r[1], "patch": r[2], "tier": r[3],
                "blue_win": r[4], "blue_team": r[5], "red_team": r[6]
            } for r in cur.fetchall()]

    def get_champindex_by_id(self, champ_id: int) -> int:
        """Maps a Riot champion ID to an internal neural network index."""
        path = Path(cfg.CHAMPION_DATA_PATH)
        if not path.exists():
            return None
        with open(path, 'r') as f:
            return int(json.load(f).get(str(champ_id)))
        
    def get_champion_names(self) -> dict:
        """Loads the mapping between internal indices and champion names."""
        with open(cfg.CHAMPION_NAMES_PATH, "r", encoding="utf-8") as f:
            return {int(k): v for k, v in json.load(f).items()}

    def close(self):
        """Closes the active database connection."""
        if self.conn:
            self.conn.close()