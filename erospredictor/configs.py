"""Application configuration constants."""

RIOT_API_KEY = "RGAPI-c3d498f9-c7a6-4add-88a5-a2c5c12e7844"
REGION = "eun1"
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

ROLE_WEIGHTS = {
    "TOP": 1.0,
    "JUNGLE": 0.9,
    "MIDDLE": 0.8,
    "BOTTOM": 0.7,
    "UTILITY": 0.6
}

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

TARGET_TIERS = ["IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND", "MASTER", "GRANDMASTER", "CHALLENGER"][::-1]
TARGET_DIVISIONS = ["I", "II", "III", "IV"]