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
    "BOTTOM": 0.15,
    "UTILITY": 0.15
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
#These will be removed after im done with experimenting

#, "GOLD", "PLATINUM", "EMERALD", "DIAMOND", "MASTER", "GRANDMASTER",
#"IRON", "BRONZE",
# "SILVER", "PLATINUM","CHALLENGER"
#asia
# kr, jp1, oc1, ru, tr1
#europe
# eun1, euw1, ru, tr1
#americas
# br1, oc1, na1, la1, la2

#TARGET_TIERS = [ "IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND", "MASTER", "GRANDMASTER", "CHALLENGER" ][::-1] 