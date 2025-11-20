import os
import pickle
from model.data_manager import DataManager
from configs import CHAMPION_COUNT, ROLE_WEIGHTS, PREPROCESSED_DATA_PATH

class Preprocessor:
    def __init__(self):
        self.data_manager = DataManager()
        self.preprocessed_data_path = PREPROCESSED_DATA_PATH
        print(f"Preprocessor initialized with cache path: {self.preprocessed_data_path}")

    def _get_cache_path(self, data_type):
        os.makedirs(self.preprocessed_data_path, exist_ok=True)
        cache_file = os.path.join(self.preprocessed_data_path, f"{data_type}_preprocessed.pkl")
        return cache_file

    def _save_preprocessed_data(self, data, data_type):

        try:
            cache_path = self._get_cache_path(data_type)
            temp_path = cache_path + ".tmp"
            with open(temp_path, 'wb') as f:
                pickle.dump(data, f)

            if os.path.exists(cache_path):
                os.remove(cache_path)
            os.rename(temp_path, cache_path)
            print(f"✅ Saved preprocessed {data_type} data to {cache_path}")
            print(f"File exists: {os.path.exists(cache_path)}")
            if os.path.exists(cache_path):
                file_size = os.path.getsize(cache_path)
                print(f"File size: {file_size} bytes")
        except Exception as e:
            print(f"❌ Error saving cache for {data_type}: {e}")
            temp_path = self._get_cache_path(data_type) + ".tmp"
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def _load_preprocessed_data(self, data_type, use_cache=True):

        if not use_cache:
            print(f"🔄 Cache loading disabled for {data_type}")
            return None
            
        cache_path = self._get_cache_path(data_type)
        print(f"Looking for cache file: {cache_path}")
        print(f"Cache file exists: {os.path.exists(cache_path)}")
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                print(f"✅ Loaded preprocessed {data_type} data from cache")

                if isinstance(data, tuple) and len(data) == 3:
                    X, y, divisions = data
                    print(f"Cache validation: {len(X)} samples, {len(y)} labels, {len(divisions)} divisions")
                    return data
                else:
                    print(f"❌ Cache file has invalid structure: {type(data)}")
                    return None
            except (pickle.PickleError, EOFError, FileNotFoundError) as e:
                print(f"❌ Error loading cache for {data_type}: {e}")
                try:
                    os.remove(cache_path)
                    print(f"Removed corrupted cache file: {cache_path}")
                except:
                    pass
                return None
        else:
            print(f"❌ Cache file not found for {data_type}")
        return None

    def _is_cache_valid(self, data_type):

        cache_path = self._get_cache_path(data_type)
        
        if not os.path.exists(cache_path):
            print(f"❌ Cache does not exist: {cache_path}")
            return False

        file_size = os.path.getsize(cache_path)
        if file_size < 100:
            print(f"❌ Cache file too small ({file_size} bytes), likely corrupted")
            return False
        
        print(f"✅ Cache file exists and has reasonable size ({file_size} bytes)")
        return True

    def preprocess_all_matches(self, use_cache=True, always_save_cache=True):

        print(f"\n=== Preprocessing RoleWeighted Data ===")
        print(f"Use cache for loading: {use_cache}")
        print(f"Always save to cache: {always_save_cache}")
        

        if use_cache and self._is_cache_valid("roleweighted"):
            cached_data = self._load_preprocessed_data("roleweighted", use_cache)
            if cached_data is not None:
                return cached_data
            else:
                print("🔄 Cache load failed, processing from scratch")
        else:
            print("🔄 Cache not used or invalid, processing from scratch")

        match_ids = self.data_manager.get_all_match_ids()
        print(f"Found {len(match_ids)} matches to process")
        
        if len(match_ids) == 0:
            print("❌ No matches found to process!")
            return [], [], []
        
        preprocessed_data = []
        blue_wins = []
        divisions = []
        
        processed_count = 0
        for match_id in match_ids:
            match_data = self.data_manager.get_match(match_id)
            if not match_data:
                continue

            divisions.append(match_data["data"].get("tier", "UNKNOWN"))
            
            blue_win = 1 if match_data["data"]["teams"][0]["win"] else 0
            blue_wins.append(blue_win)

            champions = [0.0] * (CHAMPION_COUNT*2)
            for i in range(5):
                champ_id = self.data_manager.get_champindex_by_id(match_data["data"]["participants"][i]["championId"])
                if champ_id is not None:
                    champions[champ_id] = 1.0
            for i in range(5, 10):
                champ_id = self.data_manager.get_champindex_by_id(match_data["data"]["participants"][i]["championId"])
                if champ_id is not None:
                    champions[champ_id + CHAMPION_COUNT] = 1.0
            preprocessed_data.append(champions)
            processed_count += 1
            
            if processed_count % 100 == 0:
                print(f"Processed {processed_count} matches...")

        result = (preprocessed_data, blue_wins, divisions)
        print(f"✅ Processed {len(preprocessed_data)} roleweighted samples")
        
        if always_save_cache and len(preprocessed_data) > 0:
            print("💾 Saving to cache for future use...")
            self._save_preprocessed_data(result, "roleweighted")
        else:
            print("🚫 Skipping cache save")
        
        return result
 
    def preprocess_all_matches_roleaware(self, use_cache=True, always_save_cache=True):
        print(f"\n=== Preprocessing RoleAware Data ===")
        print(f"Use cache for loading: {use_cache}")
        print(f"Always save to cache: {always_save_cache}")
        
        if use_cache and self._is_cache_valid("roleaware"):
            cached_data = self._load_preprocessed_data("roleaware", use_cache)
            if cached_data is not None:
                return cached_data
            else:
                print("🔄 Cache load failed, processing from scratch")
        else:
            print("🔄 Cache not used or invalid, processing from scratch")

        match_ids = self.data_manager.get_all_match_ids()
        print(f"Found {len(match_ids)} matches to process")
        
        if len(match_ids) == 0:
            print("❌ No matches found to process!")
            return [], [], []
        
        preprocessed_data = []
        blue_wins = []
        divisions = []

        role_order = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
        role_to_idx = {r: i for i, r in enumerate(role_order)}

        processed_count = 0
        for match_id in match_ids:
            match_data = self.data_manager.get_match(match_id)
            if not match_data:
                continue

            match = match_data["data"]
            div = match.get("tier", "UNKNOWN")
            divisions.append(div)

            blue_win = 1 if match["teams"][0]["win"] else 0
            blue_wins.append(blue_win)

            champions = [0.0] * (CHAMPION_COUNT * 10)

            for p in match["participants"]:
                role = p.get("individualPosition", "").upper()
                role_idx = role_to_idx.get(role)
                if role_idx is None:
                    continue 

                champ_index = self.data_manager.get_champindex_by_id(p["championId"])
                if champ_index is None:
                    continue

                side_offset = 0 if p["teamId"] == 100 else 5
                absolute_role = side_offset + role_idx
                champions[absolute_role * CHAMPION_COUNT + champ_index] = 1.0

            preprocessed_data.append(champions)
            processed_count += 1
            
            if processed_count % 100 == 0:
                print(f"Processed {processed_count} matches...")

        result = (preprocessed_data, blue_wins, divisions)
        print(f"✅ Processed {len(preprocessed_data)} roleaware samples")
        
        if always_save_cache and len(preprocessed_data) > 0:
            print("💾 Saving to cache for future use...")
            self._save_preprocessed_data(result, "roleaware")
        else:
            print("🚫 Skipping cache save")
        
        return result

    def clear_cache(self):
        print(f"\n=== Clearing Cache ===")
        cache_files = [
            self._get_cache_path("roleweighted"),
            self._get_cache_path("roleaware"),
        ]
        
        cleared_count = 0
        for cache_file in cache_files:
            if os.path.exists(cache_file):
                try:
                    os.remove(cache_file)
                    print(f"🗑️ Removed cache file: {cache_file}")
                    cleared_count += 1
                except Exception as e:
                    print(f"❌ Error removing {cache_file}: {e}")
        
        if cleared_count == 0:
            print("ℹ️ No cache files found to clear")
        else:
            print(f"✅ Cleared {cleared_count} cache files")
            
    def list_cache_files(self):
        print(f"\n=== Cache Files in {self.preprocessed_data_path} ===")
        if os.path.exists(self.preprocessed_data_path):
            files = os.listdir(self.preprocessed_data_path)
            for file in files:
                file_path = os.path.join(self.preprocessed_data_path, file)
                file_size = os.path.getsize(file_path)
                print(f"  {file} ({file_size} bytes)")
        else:
            print("  Cache directory does not exist")
        print("=== End Cache Files ===")