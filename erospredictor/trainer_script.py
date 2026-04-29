import json
import os
import time
import torch
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
                if c > 0: c_rw[c] = vw[i]
            for i, c in enumerate(red):
                if c > 0: c_rw[c + CHAMPION_COUNT] = vw[i]
            x = torch.tensor([c_rw], dtype=torch.float32).to(device)
            with torch.no_grad(): prob = model_obj(x).item()
        elif model_id == "roleaware":
            c_ra = [0.0] * (CHAMPION_COUNT * 10)
            for i, c in enumerate(blue):
                if c > 0: c_ra[i * CHAMPION_COUNT + c] = 1.0
            for i, c in enumerate(red):
                if c > 0: c_ra[(i + 5) * CHAMPION_COUNT + c] = 1.0
            x = torch.tensor([c_ra], dtype=torch.float32).to(device)
            with torch.no_grad(): prob = model_obj(x).item()
        return {"blue_win_prob": prob * 100}

    def recommend_fn(div: str, blue: list, red: list, bans: list, team: str, r_idx: int, top_k: int) -> list:
        recs = []
        for c_id in range(1, CHAMPION_COUNT):
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
        
    passed = res_pred['passed'] + res_rec['passed']
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
        
        choice = input("Select: ").strip()
        
        if choice == "1":
            print("\nPreprocessing...")
            prep.clear_cache()
            
            x_rw, y_rw, d_rw, w_rw = prep.process_matches(use_cache=False)
            x_ra, y_ra, d_ra, w_ra = prep.process_matches_ra(use_cache=False)
            graphs, d_gnn = prep.process_matches_gnn(use_cache=False)
            
            stat.build_stats()
            stat.save_cache()
            prep.gen_meta_champs()
            print("Preprocessing complete.")
        
        elif choice == "2":
            print("\nTraining with Golden Dataset QA...")
            name_map = load_name_map()
            dataset = GoldenDataset(path="data/golden_dataset.json", name_map=name_map)
            
            x_rw, y_rw, d_rw, w_rw = prep.process_matches(use_cache=True)
            x_ra, y_ra, d_ra, w_ra = prep.process_matches_ra(use_cache=True)
            graphs, d_gnn = prep.process_matches_gnn(use_cache=True)
            
            if not x_rw:
                continue
            
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
                    trainer.train_gnn_model(gr_d, f"{div}_gnn", fallback=False)
                    gnn_path = f"models/{div}_gnn.pth"
                    if os.path.exists(gnn_path):
                        gnn_obj = LeagueGNN().to(device)
                        gnn_obj.load_state_dict(torch.load(gnn_path, map_location=device, weights_only=True))
                        gnn_obj.eval()
                        score = quick_evaluate_model("gnn", gnn_obj, div, dataset, device)
                        print(f"  -> GNN QA Pass Rate: {score*100:.1f}%")
                        if score < 0.25:
                            print("  [!] GNN QA FAILED (<25%). Retraining with generalized parameters...")
                            trainer.train_gnn_model(gr_d, f"{div}_gnn", fallback=True)
                    print(f"  -> Total time for GNN: {time.time() - t_gnn:.1f} seconds.")
            
            print(f"\n✓ Training and QA complete! Total duration: {(time.time() - total_start_time)/60:.1f} minutes.")

        elif choice == "3":
            print("\n--- Train CUSTOM Model ---")
            x_rw, y_rw, d_rw, w_rw = prep.process_matches(use_cache=True)
            x_ra, y_ra, d_ra, w_ra = prep.process_matches_ra(use_cache=True)
            graphs, d_gnn = prep.process_matches_gnn(use_cache=True)

            if not x_rw:
                print("No data! Please run Preprocess (1) first.")
                continue

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
            else:
                x_rw_d = [x for x, d in zip(x_rw, d_rw) if d == target_div]
                y_rw_d = [y for y, d in zip(y_rw, d_rw) if d == target_div]
                w_rw_d = [w for w, d in zip(w_rw, d_rw) if d == target_div]
                x_ra_d = [x for x, d in zip(x_ra, d_ra) if d == target_div]
                y_ra_d = [y for y, d in zip(y_ra, d_ra) if d == target_div]
                w_ra_d = [w for w, d in zip(w_ra, d_ra) if d == target_div]
                gr_d = [g for g, d in zip(graphs, d_gnn) if d == target_div]

            t_custom = time.time()
            print(f"\nTraining {target_div}_{target_model} (Fallback: {use_fallback})...")
            
            if target_model == "roleweighted":
                trainer.train_single_model(x_rw_d, y_rw_d, f"{target_div}_roleweighted", CHAMPION_COUNT * 2, "standard", w_rw_d, fallback=use_fallback)
            elif target_model == "roleaware":
                trainer.train_single_model(x_ra_d, y_ra_d, f"{target_div}_roleaware", CHAMPION_COUNT * 10, "roleaware", w_ra_d, fallback=use_fallback)
            elif target_model == "gnn":
                trainer.train_gnn_model(gr_d, f"{target_div}_gnn", fallback=use_fallback)

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

if __name__ == "__main__":
    try: run_trainer()
    except KeyboardInterrupt: print("\nStopped.")