"""Application configuration constants."""

RIOT_API_KEY = "RGAPI-c3d498f9-c7a6-4add-88a5-a2c5c12e7844"
REGION = "euw1"
CONTINENT = "europe"

CHAMPION_DATA_PATH = "data/champion_id.json"
CHAMPION_NAMES_PATH = "data/champion_names.json"
MODELS_DIR = "models"
CHAMPION_COUNT = 172
PREPROCESSED_DATA_PATH = "data/preprocessed"

DB_NAME = "erospredictor"
DB_USER = "postgres"
DB_PASS = "Eros"
DB_HOST = "localhost"
DB_PORT = "5432"

#BASED ON XPETU'S ROLE IMPORTANCE DATA (TRIPLED VALUES TO AVOID NEAR NULL-VECTORS!!!)
ROLE_WEIGHTS = {
    "TOP": 0.593,
    "JUNGLE": 0.878,
    "MIDDLE": 0.62,
    "BOTTOM": 0.52,
    "UTILITY": 0.39
}

ROLE_WEIGHTS_NO = {
    "TOP": 0.66,
    "JUNGLE": 0.84,
    "MIDDLE": 0.6,
    "BOTTOM": 0.45,
    "UTILITY": 0.45
}
#PLAYERS IN HIGER TIERSHAVE MORE CONSISTENT PERFORMANCE, SO THEIR MATCHES SHOULD BE WEIGHTED MORE HEAVILY IN THE MODEL TRAINING
DIVISION_WEIGHTS = {
    "IRON": 0.3,
    "BRONZE": 0.4,
    "SILVER": 0.5,
    "GOLD": 0.75,
    "PLATINUM": 0.85,
    "EMERALD": 0.9,
    "DIAMOND": 1.0,
    "MASTER": 1.0,
    "GRANDMASTER": 1.0,
    "CHALLENGER": 1.0
}

ALLOWED_PATCHES = ["16.3", "16.4", "16.5", "16.6", "16.7"]

TARGET_TIERS = [ "IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND", "MASTER", "GRANDMASTER", "CHALLENGER" ][::-1] 
TARGET_DIVISIONS = ["I", "II", "III", "IV"]

# Environment overrides allow the unattended updater to use local credentials.
# Existing local configuration remains the fallback for manual developer tools.
import os as _os
RIOT_API_KEY = _os.environ.get("RIOT_API_KEY", RIOT_API_KEY)
DB_NAME = _os.environ.get("DB_NAME", DB_NAME)
DB_USER = _os.environ.get("DB_USER", DB_USER)
DB_PASS = _os.environ.get("DB_PASS", DB_PASS)
DB_HOST = _os.environ.get("DB_HOST", DB_HOST)
DB_PORT = _os.environ.get("DB_PORT", DB_PORT)

# Approximate collection profile, with useful minimum samples for rare ranks.
DATASET_TARGETS = {
    "CHALLENGER": 2000, "GRANDMASTER": 3000, "MASTER": 5000,
    "DIAMOND": 8000, "EMERALD": 12000, "PLATINUM": 16000,
    "GOLD": 22000, "SILVER": 18000, "BRONZE": 7000, "IRON": 5000,
}


def _load_runtime_config():
    """Only published patch/catalog metadata may override these settings."""
    import json
    from pathlib import Path
    app_dir = Path(__file__).resolve().parent
    try:
        names = json.loads((app_dir / CHAMPION_NAMES_PATH).read_text(encoding="utf-8"))
        count = max(int(key) for key in names) + 1
        if count > 0:
            globals()["CHAMPION_COUNT"] = count
    except (OSError, ValueError, TypeError):
        pass
    try:
        runtime = json.loads((app_dir / "data/runtime_config.json").read_text(encoding="utf-8"))
        if not isinstance(runtime, dict):
            return
        patches = runtime.get("ALLOWED_PATCHES")
        if isinstance(patches, list) and patches and all(isinstance(patch, str) for patch in patches):
            globals()["ALLOWED_PATCHES"] = patches
        version = runtime.get("CHAMPION_CATALOG_VERSION")
        if isinstance(version, str):
            globals()["CHAMPION_CATALOG_VERSION"] = version
        # Published targets describe the previous training snapshot. They must
        # not overwrite the collection goals configured for the next update.
    except (OSError, ValueError, TypeError):
        pass


_load_runtime_config()
