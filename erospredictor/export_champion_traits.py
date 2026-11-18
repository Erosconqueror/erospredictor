"""Refresh experimental observations without fitting or changing winrate models."""

import argparse
from pathlib import Path
import configs
from model.data_manager import DataManager
from services.champion_traits import export_observations, DATA


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patches", nargs="+", default=configs.ALLOWED_PATCHES)
    args = parser.parse_args()
    db = DataManager(True)
    try:
        export_observations(db, args.patches, DATA / "champion_trait_observations.json")
        print("Experimental champion observations exported; model files unchanged.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
