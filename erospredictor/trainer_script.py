import json
import os
import time
import torch
import hashlib
import tempfile
from collections import defaultdict, deque
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Any

from model.preprocessor import Preprocessor
from model.train_model import DynamicTrainer
from model.data_manager import DataManager
from model.statistical import StatisticalModel
from configs import CHAMPION_COUNT, ROLE_WEIGHTS
from model.golden_dataset import GoldenDataset
from model.meta_calibrator import MetaLearningCalibrator
from model.gnn_predictor import LeagueGNN
from model.predictor import ChampionPredictor, RoleAwareEmbeddingPredictor
from services.role_distribution import build_role_distribution, save_role_distribution, DEFAULT_PATH as ROLE_DISTRIBUTION_PATH

INTERACTION_POLICY = "interaction-statistics-and-priors-v1"
GNN_PRIOR_POLICY = "training-only-role-priors-v1"


def interaction_fingerprint(document, tier=None):
    """Stable evidence identity; wall-clock generation time is not model input."""
    evidence = {key: value for key, value in document.items() if key != "generated_at"}
    if tier is not None:
        evidence = {key: evidence.get(key) for key in ("version", "roles", "min_samples", "shrinkage", "method")}
        evidence.update(tier=tier, bucket=document.get("tiers", {}).get(tier))
    raw = json.dumps(evidence, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def compact_json(path, document):
    """Atomically write derived tables without expanding their dense JSON cells."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".interactions-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(document, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _meta_for_records(records, mapping, patches=None):
    """Derive non-off-meta eligibility from these features, never held-out labels."""
    holder = {}
    window = list(patches) if patches is not None else list(dict.fromkeys(row.get("patch") for row in records))
    data = SimpleNamespace(get_all_matches=lambda: records, get_champion_mapping=lambda: mapping,
                           save_meta_champs=lambda value: holder.update(meta=value))
    Preprocessor(db=data, allowed_patches=window).gen_meta_champs()
    return holder.get("meta", {})


def make_interaction_prior_factory(records, mapping, tier, *, patches=None, region=None):
    """Fit a compact GNN prior only after the trainer selects its training rows."""
    records = tuple(records)
    cached_indices, cached_prior = None, None
    def fit(indices):
        nonlocal cached_indices, cached_prior
        key = tuple(indices)
        if key == cached_indices:
            return cached_prior
        if len(set(key)) != len(key) or any(type(index) is not int or not 0 <= index < len(records) for index in key):
            raise ValueError("Invalid GNN training-subset indices")
        from services.interaction_statistics import build_interaction_statistics, InteractionStatistics
        selected = [records[index] for index in key]
        meta = _meta_for_records(selected, mapping, patches)
        selected_tiers = sorted({row["tier"] for row in selected})
        document = build_interaction_statistics(selected, mapping, meta, patches=patches,
            region=region, tiers=selected_tiers, source="gnn_training_subset")
        cached_prior = InteractionStatistics(document, mapping).prior(tier)
        cached_indices = key
        return cached_prior
    return fit


def graph_training_records(graphs, divisions, mapping, source_records=None):
    """Align cached graphs with their exact Riot-ID records and window metadata."""
    if len(graphs) != len(divisions):
        raise ValueError("GNN graphs and divisions are not aligned")
    inverse = {int(index): int(riot) for riot, index in mapping.items()}
    source = defaultdict(deque)
    if source_records is not None:
        from services.role_distribution import _team
        for row in source_records:
            blue, red = _team(row.get("blue_team")), _team(row.get("red_team"))
            if blue is None or red is None or any(str(champion) not in mapping for champion in blue + red):
                continue
            key = (tuple(int(champion) for champion in blue + red), bool(row.get("blue_win")), row.get("tier"))
            source[key].append(row)
    result = []
    for index, (graph, tier) in enumerate(zip(graphs, divisions)):
        teams = [[None] * 5, [None] * 5]
        nodes = graph.x.detach().cpu().tolist()
        for cid, role, team in nodes:
            if cid not in inverse or team not in (0, 1) or not 0 <= role < 10 or role // 5 != team:
                raise ValueError("Invalid node in a cached GNN training graph")
            if teams[team][role % 5] is not None:
                raise ValueError("Duplicate role in a cached GNN training graph")
            teams[team][role % 5] = inverse[cid]
        if len(nodes) != 10 or any(champion is None for team in teams for champion in team):
            raise ValueError("Incomplete cached GNN training graph")
        value = float(graph.y.detach().cpu().reshape(-1)[0])
        if value not in (0.0, 1.0):
            raise ValueError("Invalid GNN training outcome")
        win = bool(value)
        key = (tuple(teams[0] + teams[1]), win, tier)
        if source_records is not None:
            if not source[key]:
                raise ValueError("Cached GNN graph does not match the frozen training records")
            result.append(source[key].popleft())
        else:
            result.append({"match_id": f"training-graph:{index}", "tier": tier,
                           "blue_team": teams[0], "red_team": teams[1], "blue_win": win})
    return result


class _FrozenTrainingData:
    def __init__(self, source, records):
        self.source, self.records = source, records

    def get_all_matches(self):
        return self.records

    def __getattr__(self, name):
        return getattr(self.source, name)


def freeze_training_window(preprocessor):
    """One SQL read per interactive action even if a scraper is collecting."""
    source = preprocessor.db
    while isinstance(source, _FrozenTrainingData):
        source = source.source
    current = Preprocessor(db=source, allowed_patches=preprocessor.allowed_patches)
    return Preprocessor(db=_FrozenTrainingData(source, current._get_matches()),
                        allowed_patches=preprocessor.allowed_patches)


def interaction_artifacts(records, mapping, meta_data, *, patches, tiers, region, output_dir,
                          previous_dir=None, source="training_snapshot"):
    """Build before fitting, keeping untouched rank tables and their own scopes."""
    from services.interaction_statistics import (build_interaction_statistics, InteractionStatistics,
        load_interaction_statistics, merge_interaction_statistics, save_interaction_statistics)
    from services.blind_pick import build_blind_profiles
    output_dir, previous_dir = Path(output_dir), Path(previous_dir or output_dir)
    document = build_interaction_statistics(records, mapping, meta_data, patches=patches,
        tiers=tiers, region=region, source=source)
    index = InteractionStatistics(document, mapping)
    profiles = build_blind_profiles(index, tiers, mapping)
    profiles["tier_scopes"] = {tier: bucket.get("scope", document["scope"])
                               for tier, bucket in document["tiers"].items()}
    try:
        old_profiles = json.loads((previous_dir / "blind_profiles.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        old_profiles = {}
    selected = set(tiers) | {"MIXED"}
    if isinstance(old_profiles, dict) and old_profiles.get("version") == profiles.get("version"):
        for field in ("tiers", "tier_scopes"):
            old = old_profiles.get(field, {})
            if isinstance(old, dict):
                profiles[field] = {**{tier: bucket for tier, bucket in old.items() if tier not in selected},
                                   **profiles.get(field, {})}
    merged = merge_interaction_statistics(load_interaction_statistics(previous_dir / "interaction_stats.json"),
                                          document, tiers)
    save_interaction_statistics(output_dir / "interaction_stats.json", merged)
    compact_json(output_dir / "blind_profiles.json", profiles)
    return index, document


def refresh_training_statistics(preprocessor, stat):
    """Interactive actions prepare all evidence from their frozen patch window."""
    records, mapping = preprocessor._get_matches(), preprocessor.c_map
    stat.db = preprocessor.db
    stat.c_map = mapping
    stat.build_stats()
    stat.save_cache()
    preprocessor.gen_meta_champs()
    refresh_role_statistics(preprocessor)
    tiers = sorted({record["tier"] for record in records})
    index, document = interaction_artifacts(records, mapping, _meta_for_records(records, mapping, preprocessor.allowed_patches),
        patches=preprocessor.allowed_patches, tiers=tiers, region=None, output_dir="data", source="interactive_trainer")
    stat.interactions = index
    print(f"Interaction matrices and blind profiles saved ({document['scope']['matches']} matches).")
    return records


def refresh_role_statistics(preprocessor):
    """Legacy trainer: export the same patch-filtered matches used by preprocessing."""
    document = build_role_distribution(preprocessor._get_matches(), patches=preprocessor.allowed_patches,
                                       source="interactive_trainer")
    if document["scope"]["matches"]:
        save_role_distribution(ROLE_DISTRIBUTION_PATH, document,
                               selected_tiers=document["scope"]["tiers"])
        print(f"Champion role distribution saved ({document['scope']['matches']} matches).")
    return document

def load_name_map() -> dict:
    """Loads champion ID to Name mapping and inverts it."""
    name_map = {}
    try:
        with open("data/champion_names.json", "r", encoding="utf-8") as f:
            id_to_name = json.load(f)
            name_map = {name: int(cid) for cid, name in id_to_name.items()}
    except Exception:
        pass
    return name_map

def quick_evaluate_model(model_id: str, model_obj: Any, div: str, dataset: GoldenDataset, device: torch.device) -> float:
    """Runs a quick evaluation of a single model against the Golden Dataset and returns the pass rate."""
    def predict_fn(div: str, blue: list, red: list) -> dict:
        prob = 0.5
        if model_id == "gnn":
            from model.gnn_predictor import predict_gnn
            try: prob = predict_gnn(model_obj, blue, red, device)
            except Exception: pass
        elif model_id == "roleweighted":
            c_rw = [0.0] * (CHAMPION_COUNT * 2)
            vw = list(ROLE_WEIGHTS.values())
            for i, c in enumerate(blue):
                if c >= 0: c_rw[c] = vw[i]
            for i, c in enumerate(red):
                if c >= 0: c_rw[c + CHAMPION_COUNT] = vw[i]
            x = torch.tensor([c_rw], dtype=torch.float32).to(device)
            with torch.no_grad(): prob = model_obj(x).item()
        elif model_id == "roleaware":
            c_ra = [0.0] * (CHAMPION_COUNT * 10)
            for i, c in enumerate(blue):
                if c >= 0: c_ra[i * CHAMPION_COUNT + c] = 1.0
            for i, c in enumerate(red):
                if c >= 0: c_ra[(i + 5) * CHAMPION_COUNT + c] = 1.0
            x = torch.tensor([c_ra], dtype=torch.float32).to(device)
            with torch.no_grad(): prob = model_obj(x).item()
        return {"blue_win_prob": prob * 100}

    def recommend_fn(div: str, blue: list, red: list, bans: list, team: str, r_idx: int, top_k: int) -> list:
        recs = []
        unavailable = {cid for cid in blue + red + bans if cid >= 0}
        for c_id in range(CHAMPION_COUNT):
            if c_id in unavailable:
                continue
            t_blue, t_red = list(blue), list(red)
            if team == "blue": t_blue[r_idx] = c_id
            else: t_red[r_idx] = c_id
            pred = predict_fn(div, t_blue, t_red)
            score = pred["blue_win_prob"] if team == "blue" else (100.0 - pred["blue_win_prob"])
            recs.append({"id": c_id, "score": score})
        recs.sort(key=lambda x: x["score"], reverse=True)
        return recs[:top_k]

    res_pred = dataset.validate_predictions(predict_fn, div)
    res_rec = dataset.validate_recommendations(recommend_fn, div) # <--- ITT VOLT A HIBA, javítva!
    
    total_tests = res_pred['total'] + res_rec['total']
    if total_tests == 0:
        return 1.0
        
    passed = res_pred['passed'] + res_rec.get('score', res_rec['passed'])
    return passed / total_tests

def run_trainer():
    """Enhanced training script with interactive custom parameters, QA fallback, and execution timers."""
    print("=== EROSPREDICTOR - ENHANCED TRAINING WITH QA ===")
    prep = Preprocessor()
    db = DataManager(True)
    stat = StatisticalModel(db)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    while True:
        print("\n1. Preprocess data")
        print("2. Train ALL models (with Golden Dataset QA Fallback)")
        print("3. Train CUSTOM model (Interactive)")
        print("4. Calibrate ensemble weights")
        print("5. EXIT")
        print("6. Database maintenance (keep latest 26 patches)")
        
        choice = input("Select: ").strip()
        
        if choice == "1":
            print("\nPreprocessing...")
            prep = freeze_training_window(prep)
            
            x_rw, y_rw, d_rw, w_rw = prep.process_matches(use_cache=False)
            x_ra, y_ra, d_ra, w_ra = prep.process_matches_ra(use_cache=False)
            graphs, d_gnn = prep.process_matches_gnn(use_cache=False)
            
            refresh_training_statistics(prep, stat)
            print("Preprocessing complete.")
        
        elif choice == "2":
            print("\nTraining with Golden Dataset QA...")
            prep = freeze_training_window(prep)
            name_map = load_name_map()
            dataset = GoldenDataset(path="data/golden_dataset.json", name_map=name_map)
            
            x_rw, y_rw, d_rw, w_rw = prep.process_matches(use_cache=True)
            x_ra, y_ra, d_ra, w_ra = prep.process_matches_ra(use_cache=True)
            graphs, d_gnn = prep.process_matches_gnn(use_cache=True)
            
            if not x_rw:
                continue

            frozen_records = refresh_training_statistics(prep, stat)
            graph_records = graph_training_records(graphs, d_gnn, prep.c_map, frozen_records)
            
            unique_divs = sorted(set(d_rw))
            if "MIXED" not in unique_divs:
                unique_divs.append("MIXED")
            
            total_start_time = time.time()
            
            for div in unique_divs:
                print(f"\n>>> DIV: {div} <<<")
                
                x_rw_d = x_rw if div == "MIXED" else [x for x, d in zip(x_rw, d_rw) if d == div]
                y_rw_d = y_rw if div == "MIXED" else [y for y, d in zip(y_rw, d_rw) if d == div]
                w_rw_d = w_rw if div == "MIXED" else [w for w, d in zip(w_rw, d_rw) if d == div]
                
                x_ra_d = x_ra if div == "MIXED" else [x for x, d in zip(x_ra, d_ra) if d == div]
                y_ra_d = y_ra if div == "MIXED" else [y for y, d in zip(y_ra, d_ra) if d == div]
                w_ra_d = w_ra if div == "MIXED" else [w for w, d in zip(w_ra, d_ra) if d == div]
                
                gr_d = graphs if div == "MIXED" else [g for g, d in zip(graphs, d_gnn) if d == div]
                prior_records = graph_records if div == "MIXED" else [m for m, d in zip(graph_records, d_gnn) if d == div]
                
                if len(x_rw_d) < 200:
                    continue
                
                trainer = DynamicTrainer(div)

                def train_and_qa(model_id: str, x_data, y_data, w_data, input_size, model_type):
                    t0 = time.time()
                    print(f"Training {model_id}...")
                    trainer.train_single_model(x_data, y_data, f"{div}_{model_id}", input_size, model_type, w_data, fallback=False)
                    
                    path = f"models/{div}_{model_id}.pth"
                    if os.path.exists(path):
                        model_obj = ChampionPredictor(input_size=input_size).to(device) if model_id == "roleweighted" else RoleAwareEmbeddingPredictor().to(device)
                        model_obj.load_state_dict(torch.load(path, map_location=device, weights_only=True))
                        model_obj.eval()
                        
                        score = quick_evaluate_model(model_id, model_obj, div, dataset, device)
                        print(f"  -> QA Pass Rate: {score*100:.1f}%")
                        
                        if score < 0.25:
                            print("  [!] QA FAILED (<25%). Retraining with generalized parameters...")
                            trainer.train_single_model(x_data, y_data, f"{div}_{model_id}", input_size, model_type, w_data, fallback=True)
                            model_obj.load_state_dict(torch.load(path, map_location=device, weights_only=True))
                            model_obj.eval()
                            score2 = quick_evaluate_model(model_id, model_obj, div, dataset, device)
                            print(f"  -> New QA Pass Rate: {score2*100:.1f}%. Saved for Meta-Calibrator.")
                    print(f"  -> Total time for {model_id}: {time.time() - t0:.1f} seconds.")

                train_and_qa("roleweighted", x_rw_d, y_rw_d, w_rw_d, CHAMPION_COUNT * 2, "standard")
                train_and_qa("roleaware", x_ra_d, y_ra_d, w_ra_d, CHAMPION_COUNT * 10, "roleaware")
                
                if len(gr_d) >= 200:
                    t_gnn = time.time()
                    print("Training GNN...")
                    prior_factory = make_interaction_prior_factory(prior_records, prep.c_map, div,
                                                                   patches=prep.allowed_patches)
                    trainer.train_gnn_model(gr_d, f"{div}_gnn", fallback=False,
                                            interaction_prior_factory=prior_factory)
                    gnn_path = f"models/{div}_gnn.pth"
                    if os.path.exists(gnn_path):
                        gnn_obj = LeagueGNN().to(device)
                        gnn_obj.load_state_dict(torch.load(gnn_path, map_location=device, weights_only=True))
                        gnn_obj.eval()
                        score = quick_evaluate_model("gnn", gnn_obj, div, dataset, device)
                        print(f"  -> GNN QA Pass Rate: {score*100:.1f}%")
                        if score < 0.25:
                            print("  [!] GNN QA FAILED (<25%). Retraining with generalized parameters...")
                            trainer.train_gnn_model(gr_d, f"{div}_gnn", fallback=True,
                                                    interaction_prior_factory=prior_factory)
                    print(f"  -> Total time for GNN: {time.time() - t_gnn:.1f} seconds.")
            
            print(f"\n✓ Training and QA complete! Total duration: {(time.time() - total_start_time)/60:.1f} minutes.")

        elif choice == "3":
            print("\n--- Train CUSTOM Model ---")
            prep = freeze_training_window(prep)
            x_rw, y_rw, d_rw, w_rw = prep.process_matches(use_cache=True)
            x_ra, y_ra, d_ra, w_ra = prep.process_matches_ra(use_cache=True)
            graphs, d_gnn = prep.process_matches_gnn(use_cache=True)

            if not x_rw:
                print("No data! Please run Preprocess (1) first.")
                continue

            frozen_records = refresh_training_statistics(prep, stat)
            graph_records = graph_training_records(graphs, d_gnn, prep.c_map, frozen_records)

            unique_divs = sorted(set(d_rw))
            if "MIXED" not in unique_divs:
                unique_divs.append("MIXED")

            print(f"Available divisions: {', '.join(unique_divs)}")
            target_div = input("Select division: ").strip().upper()
            if target_div not in unique_divs:
                print("Invalid division.")
                continue

            print("Available models: 1. roleweighted, 2. roleaware, 3. gnn")
            m_choice = input("Select model (1/2/3): ").strip()
            model_map = {"1": "roleweighted", "2": "roleaware", "3": "gnn"}
            target_model = model_map.get(m_choice)
            if not target_model:
                print("Invalid model.")
                continue

            fallback_input = input("Use fallback parameters (stricter regularization)? (y/n): ").strip().lower()
            use_fallback = fallback_input == 'y'

            custom_ep, custom_bs, custom_lr = None, None, None
            if not use_fallback:
                cust_param_input = input("Use custom hyperparameters? (y/n): ").strip().lower()
                if cust_param_input == 'y':
                    try:
                        custom_ep = int(input("Epochs (e.g. 30): ").strip())
                        custom_bs = int(input("Batch Size (e.g. 64): ").strip())
                        custom_lr = float(input("Learning Rate (e.g. 0.001): ").strip())
                    except ValueError:
                        print("Invalid numeric input. Proceeding with dynamic defaults.")

            trainer = DynamicTrainer(target_div)
            if custom_ep and custom_bs and custom_lr:
                trainer.config["epochs"] = custom_ep
                trainer.config["batch_size"] = custom_bs
                trainer.config["learning_rate"] = custom_lr

            if target_div == "MIXED":
                x_rw_d, y_rw_d, w_rw_d = x_rw, y_rw, w_rw
                x_ra_d, y_ra_d, w_ra_d = x_ra, y_ra, w_ra
                gr_d = graphs
                prior_records = graph_records
            else:
                x_rw_d = [x for x, d in zip(x_rw, d_rw) if d == target_div]
                y_rw_d = [y for y, d in zip(y_rw, d_rw) if d == target_div]
                w_rw_d = [w for w, d in zip(w_rw, d_rw) if d == target_div]
                x_ra_d = [x for x, d in zip(x_ra, d_ra) if d == target_div]
                y_ra_d = [y for y, d in zip(y_ra, d_ra) if d == target_div]
                w_ra_d = [w for w, d in zip(w_ra, d_ra) if d == target_div]
                gr_d = [g for g, d in zip(graphs, d_gnn) if d == target_div]
                prior_records = [m for m, d in zip(graph_records, d_gnn) if d == target_div]

            t_custom = time.time()
            print(f"\nTraining {target_div}_{target_model} (Fallback: {use_fallback})...")
            
            if target_model == "roleweighted":
                trainer.train_single_model(x_rw_d, y_rw_d, f"{target_div}_roleweighted", CHAMPION_COUNT * 2, "standard", w_rw_d, fallback=use_fallback)
            elif target_model == "roleaware":
                trainer.train_single_model(x_ra_d, y_ra_d, f"{target_div}_roleaware", CHAMPION_COUNT * 10, "roleaware", w_ra_d, fallback=use_fallback)
            elif target_model == "gnn":
                prior_factory = make_interaction_prior_factory(prior_records, prep.c_map, target_div,
                                                               patches=prep.allowed_patches)
                trainer.train_gnn_model(gr_d, f"{target_div}_gnn", fallback=use_fallback,
                                        interaction_prior_factory=prior_factory)

            run_qa = input("Run QA evaluation on this model? (y/n): ").strip().lower()
            if run_qa == 'y':
                name_map = load_name_map()
                dataset = GoldenDataset(path="data/golden_dataset.json", name_map=name_map)
                path = f"models/{target_div}_{target_model}.pth"
                if os.path.exists(path):
                    if target_model == "gnn":
                        gnn_obj = LeagueGNN().to(device)
                        gnn_obj.load_state_dict(torch.load(path, map_location=device, weights_only=True))
                        gnn_obj.eval()
                        score = quick_evaluate_model("gnn", gnn_obj, target_div, dataset, device)
                        print(f"  -> QA Pass Rate: {score*100:.1f}%")
                    else:
                        in_size = CHAMPION_COUNT * 2 if target_model == "roleweighted" else CHAMPION_COUNT * 10
                        model_obj = ChampionPredictor(input_size=in_size).to(device) if target_model == "roleweighted" else RoleAwareEmbeddingPredictor().to(device)
                        model_obj.load_state_dict(torch.load(path, map_location=device, weights_only=True))
                        model_obj.eval()
                        score = quick_evaluate_model(target_model, model_obj, target_div, dataset, device)
                        print(f"  -> QA Pass Rate: {score*100:.1f}%")
                else:
                    print("Model file not found for evaluation.")

            print(f"\n✓ Custom training complete in {time.time() - t_custom:.1f} seconds!")
        
        elif choice == "4":
            print("\nCalibrating ensemble weights using Golden Dataset...")
            prep = freeze_training_window(prep)
            refresh_training_statistics(prep, stat)
            name_map = load_name_map()
            dataset = GoldenDataset(path="data/golden_dataset.json", name_map=name_map)
            calibrator = MetaLearningCalibrator(dataset=dataset)
            
            _, _, d_rw, _ = prep.process_matches(use_cache=True)
            unique_divs = sorted(set(d_rw)) if d_rw else []
            if "MIXED" not in unique_divs:
                unique_divs.append("MIXED")
            
            for div in unique_divs:
                print(f"\nCalibrating {div}...")
                models_dict = {"gnn": None, "roleweighted": None, "roleaware": None, "statistical": stat}
                
                gnn_path = f"models/{div}_gnn.pth"
                if not os.path.exists(gnn_path): gnn_path = "models/gnn_model.pth"
                if os.path.exists(gnn_path):
                    gnn = LeagueGNN().to(device)
                    gnn.load_state_dict(torch.load(gnn_path, map_location=device, weights_only=True))
                    gnn.eval()
                    models_dict["gnn"] = gnn
                    
                rw_path = f"models/{div}_roleweighted.pth"
                if os.path.exists(rw_path):
                    rw = ChampionPredictor(input_size=CHAMPION_COUNT * 2).to(device)
                    rw.load_state_dict(torch.load(rw_path, map_location=device, weights_only=True))
                    rw.eval()
                    models_dict["roleweighted"] = rw
                    
                ra_path = f"models/{div}_roleaware.pth"
                if os.path.exists(ra_path):
                    ra = RoleAwareEmbeddingPredictor().to(device)
                    ra.load_state_dict(torch.load(ra_path, map_location=device, weights_only=True))
                    ra.eval()
                    models_dict["roleaware"] = ra
                    
                optimal_weights = calibrator.calibrate(models_dict, div)
                calibrator.save(f"models/{div}_meta_calibrator.pkl")
                print(f"  Weights: {optimal_weights}")
            
            print("\n✓ Calibration complete!")
        
        elif choice == "5":
            break
        elif choice == "6":
            from database_maintenance import main as maintenance
            if maintenance([]) == 0 and input("Type DELETE to back up and remove the old matches: ").strip() == "DELETE":
                maintenance(["--apply"])

if __name__ == "__main__":
    try: run_trainer()
    except KeyboardInterrupt: print("\nStopped.")
