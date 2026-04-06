import os
import requests

REPOSITORY_URL = "https://raw.githubusercontent.com/Erosconqueror/ErospredictorModelUpdates/main/"
VERSION_FILE = "version.dat"

FILES_TO_UPDATE = [
    "configs.py",
    "data/champion_id.json",
    "data/champion_names.json",
    "data/meta_champs.json",
    "data/stats_cache.json",
    "models/BRONZE_gnn.pth",
    "models/BRONZE_roleaware.pth",
    "models/BRONZE_roleweighted.pth",
    "models/CHALLENGER_gnn.pth",
    "models/CHALLENGER_roleaware.pth",
    "models/CHALLENGER_roleweighted.pth",
    "models/DIAMOND_gnn.pth",
    "models/DIAMOND_roleaware.pth",
    "models/DIAMOND_roleweighted.pth",
    "models/EMERALD_gnn.pth",
    "models/EMERALD_roleaware.pth",
    "models/EMERALD_roleweighted.pth",
    "models/GOLD_gnn.pth",
    "models/GOLD_roleaware.pth",
    "models/GOLD_roleweighted.pth",
    "models/GRANDMASTER_gnn.pth",
    "models/GRANDMASTER_roleaware.pth",
    "models/GRANDMASTER_roleweighted.pth",
    "models/IRON_gnn.pth",
    "models/IRON_roleaware.pth",
    "models/IRON_roleweighted.pth",
    "models/MASTER_gnn.pth",
    "models/MASTER_roleaware.pth",
    "models/MASTER_roleweighted.pth",
    "models/MIXED_gnn.pth",
    "models/MIXED_roleaware.pth",
    "models/MIXED_roleweighted.pth",
    "models/PLATINUM_gnn.pth",
    "models/PLATINUM_roleaware.pth",
    "models/PLATINUM_roleweighted.pth",
    "models/SILVER_gnn.pth",
    "models/SILVER_roleaware.pth",
    "models/SILVER_roleweighted.pth"
]

def get_local_version() -> float:
    """Returns the current local version from the version file."""
    if os.path.exists(VERSION_FILE):
        try:
            with open(VERSION_FILE, "r", encoding="utf-8") as f:
                return float(f.read().strip())
        except Exception:
            return 0.0
    return 0.0

def update_files():
    """Checks for a new version on the server and downloads updated files if necessary."""
    print("[Updater] Checking for updates...")
    
    try:
        response = requests.get(REPOSITORY_URL + VERSION_FILE, timeout=3)
        response.raise_for_status()
        remote_version = float(response.text.strip())
    except Exception:
        print("[Updater] Could not connect to the server. Starting in offline mode.")
        return

    local_version = get_local_version()

    if remote_version <= local_version:
        print(f"[Updater] Software is up to date (v{local_version}).")
        return

    print(f"[Updater] Update found: {local_version} -> {remote_version}, Downloading files...")

    for file_path in FILES_TO_UPDATE:
        print(f" -> Downloading {file_path}...")
        try:
            file_resp = requests.get(REPOSITORY_URL + file_path, timeout=10)
            file_resp.raise_for_status()
            
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, "wb") as f:
                f.write(file_resp.content)
        except Exception as e:
            print(f"[Updater] Error downloading {file_path}: {e}")
            print("[Updater] Update aborted.")
            return

    with open(VERSION_FILE, "w", encoding="utf-8") as f:
        f.write(str(remote_version))

    print("[Updater] Update successful!")

if __name__ == "__main__":
    update_files()