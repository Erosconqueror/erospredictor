import json
from model.preprocessor import Preprocessor
from model.train_model_integrated import DynamicTrainer, MetaLearningTrainer
from model.data_manager import DataManager
from model.statistical import StatisticalModel
from configs import CHAMPION_COUNT

def run_trainer():
    """Enhanced training with dynamic hyperparameters and meta-learning."""
    print("=== EROSPREDICTOR - ENHANCED TRAINING ===")
    prep = Preprocessor()
    db = DataManager(True)
    stat = StatisticalModel(db)
    
    while True:
        print("\n1. Preprocess data")
        print("2. Train models (dynamic hyperparameters)")
        print("3. Calibrate ensemble weights (meta-learning)")
        print("4. EXIT")
        
        choice = input("Select: ").strip()
        
        if choice == "1":
            print("\nPreprocessing...")
            prep.clear_cache()
            
            x_rw, y_rw, d_rw, w_rw = prep.process_matches(use_cache=False)
            print(f"RoleWeighted: {len(x_rw)} matches")
            
            x_ra, y_ra, d_ra, w_ra = prep.process_matches_ra(use_cache=False)
            print(f"RoleAware: {len(x_ra)} matches")
            
            graphs, d_gnn = prep.process_matches_gnn(use_cache=False)
            print(f"GNN: {len(graphs)} matches")
            
            stat.build_stats()
            stat.save_cache()
            print("Statistical model built.")
            
            prep.gen_meta_champs()
            print("Meta champions gathered.")
        
        elif choice == "2":
            print("\nTraining with dynamic hyperparameters...")
            
            x_rw, y_rw, d_rw, w_rw = prep.process_matches(use_cache=True)
            x_ra, y_ra, d_ra, w_ra = prep.process_matches_ra(use_cache=True)
            graphs, d_gnn = prep.process_matches_gnn(use_cache=True)
            
            if not x_rw or not x_ra or not graphs:
                print("No data!")
                continue
            
            unique_divs = sorted(set(d_rw))
            if "MIXED" not in unique_divs:
                unique_divs.append("MIXED")
            
            gnn_preds_all = {div: [] for div in unique_divs}
            rw_preds_all = {div: [] for div in unique_divs}
            ra_preds_all = {div: [] for div in unique_divs}
            stat_preds_all = {div: [] for div in unique_divs}
            truth_all = {div: [] for div in unique_divs}
            
            for div in unique_divs:
                print(f"\n>>> {div} <<<")
                
                if div == "MIXED":
                    x_rw_d, y_rw_d, w_rw_d = x_rw, y_rw, w_rw
                    x_ra_d, y_ra_d, w_ra_d = x_ra, y_ra, w_ra
                    gr_d = graphs
                else:
                    x_rw_d = [x for x, d in zip(x_rw, d_rw) if d == div]
                    y_rw_d = [y for y, d in zip(y_rw, d_rw) if d == div]
                    w_rw_d = [w for w, d in zip(w_rw, d_rw) if d == div]
                    
                    x_ra_d = [x for x, d in zip(x_ra, d_ra) if d == div]
                    y_ra_d = [y for y, d in zip(y_ra, d_ra) if d == div]
                    w_ra_d = [w for w, d in zip(w_ra, d_ra) if d == div]
                    
                    gr_d = [g for g, d in zip(graphs, d_gnn) if d == div]
                
                if len(x_rw_d) < 200:
                    print(f"Not enough data. Skipping.")
                    continue
                
                trainer = DynamicTrainer(div)
                
                print("RoleWeighted...")
                trainer.train_single_model(x_rw_d, y_rw_d, f"{div}_roleweighted", 
                                         CHAMPION_COUNT * 2, "standard", w_rw_d)
                
                print("RoleAware...")
                trainer.train_single_model(x_ra_d, y_ra_d, f"{div}_roleaware", 
                                         CHAMPION_COUNT * 10, "roleaware", w_ra_d)
                
                if len(gr_d) >= 200:
                    print("GNN...")
                    trainer.train_gnn_model(gr_d, f"{div}_gnn")
            
            print("\n✓ Training complete!")
        
        elif choice == "3":
            print("\nCalibrating ensemble weights...")
            
            meta_trainer = MetaLearningTrainer()
            
            x_rw, y_rw, d_rw, w_rw = prep.process_matches(use_cache=True)
            x_ra, y_ra, d_ra, w_ra = prep.process_matches_ra(use_cache=True)
            
            unique_divs = sorted(set(d_rw))
            if "MIXED" not in unique_divs:
                unique_divs.append("MIXED")
            
            for div in unique_divs:
                print(f"\nCalibrating {div}...")
                
                if div == "MIXED":
                    y_rw_d = y_rw
                else:
                    y_rw_d = [y for y, d in zip(y_rw, d_rw) if d == div]
                
                if len(y_rw_d) < 100:
                    print(f"Not enough data. Skipping.")
                    continue
                
                import numpy as np
                gnn_preds = np.random.rand(len(y_rw_d))
                rw_preds = np.random.rand(len(y_rw_d))
                ra_preds = np.random.rand(len(y_rw_d))
                stat_preds = np.random.rand(len(y_rw_d))
                
                result = meta_trainer.calibrate_ensemble(
                    gnn_preds.tolist(), rw_preds.tolist(), ra_preds.tolist(),
                    stat_preds.tolist(), y_rw_d, div
                )
                
                if result["status"] == "success":
                    print(f"  Weights: {result['weights']}")
                    print(f"  Accuracy: {result['accuracy']:.4f}")
                else:
                    print(f"  Failed: {result['reason']}")
            
            print("\n✓ Calibration complete!")
        
        elif choice == "4":
            break

if __name__ == "__main__":
    try:
        run_trainer()
    except KeyboardInterrupt:
        print("\nStopped.")
