
#HELLO
import os
from model.preprocessor import Preprocessor
from model.train_model import train_single_model, train_gnn_model
from model.data_manager import DataManager
from model.statistical import StatisticalModel
from configs import CHAMPION_COUNT


def get_params(size: int, m_type: str) -> tuple:
    """Calculates optimal hyperparameters based on dataset size."""
    if size < 2000:
        ep, bs, lr = 100, 32, 0.001
    elif size < 10000:
        ep, bs, lr = 75, 64, 0.001
    elif size < 30000:
        ep, bs, lr = 50, 128, 0.0005
    else:
        ep, bs, lr = 40, 256, 0.0003
        
    if m_type == "gnn":
        lr *= 0.3
        ep = min(ep, 60)
        
    return ep, bs, lr

def run_trainer():
    """CLI application to train ML models with data fetching, RAM caching and on-the-fly preprocessing."""
    print("=== EROSPREDICTOR - TRAINING MODUL ===")
    prep = Preprocessor()
    db = DataManager(True)
    stat = StatisticalModel(db)
    
    while True:
        print("\n1. Preprocess ALL data (Loads into RAM)")
        print("2. Train individual model (via giving unique parameters)")
        print("3. Train every model (via auto-tune function)")
        print("4. EXIT")
        
        choice = input("Choose option: ").strip()
        
        if choice == "1":
            print("\nStarting to preprocess the data...")
            prep.clear_cache()
            
            x_rw, y_rw, d_rw, w_rw = prep.process_matches(use_cache=False)
            print(f"RoleWeighted data processed: {len(x_rw)} matches")
            
            x_ra, y_ra, d_ra, w_ra = prep.process_matches_ra(use_cache=False)
            print(f"RoleAware data processed: {len(x_ra)} matches")
            
            graphs, d_gnn = prep.process_matches_gnn(use_cache=False)
            print(f"GNN graphs processed: {len(graphs)} matches")
            
            stat.build_stats()
            stat.save_cache()
            print("Statistical model built.")
            
            print("Gathering meta champions...")
            prep.gen_meta_champs()
            print("All meta champions gathered.")
            
        elif choice == "2":
            print("\n1. RoleWeighted\n2. RoleAware\n3. GNN")
            m_choice = input("Which model? (1-3): ").strip()
            div = input("Division (pl. DIAMOND, MIXED): ").strip().upper()
            
            ep_in = input("Epochs (base: 50): ").strip()
            bs_in = input("Batch size (base: 32): ").strip()
            lr_in = input("Learning rate (base: 0.001): ").strip()
            
            ep = int(ep_in) if ep_in else 50
            bs = int(bs_in) if bs_in else 32
            lr = float(lr_in) if lr_in else 0.001
            
            print("Preparing data...")
            
            if m_choice == "1":
                X, y, divs, w = prep.process_matches(use_cache=True)
                if not X:
                    print("No matches found in database!")
                    continue
                train_single_model(X, y, divs, f"{div}_roleweighted", CHAMPION_COUNT * 2, ep, bs, lr, "standard", w)
            
            elif m_choice == "2":
                X, y, divs, w = prep.process_matches_ra(use_cache=True)
                if not X:
                    print("No matches found in database!")
                    continue
                train_single_model(X, y, divs, f"{div}_roleaware", CHAMPION_COUNT * 10, ep, bs, lr, "roleaware", w)

            elif m_choice == "3":
                graphs, divs = prep.process_matches_gnn(use_cache=True)
                if not graphs:
                    print("No matches found in database!")
                    continue
                if div != "MIXED":
                    graphs = [g for g, d in zip(graphs, divs) if d == div]
                print(f"Adatok szűrve: {len(graphs)} meccs a {div} divízióban.")
                
                train_gnn_model(graphs, f"{div}_gnn", ep, bs, lr)
                
        elif choice == "3":
            print("Preparing data for all models...")
            x_rw, y_rw, d_rw, w_rw = prep.process_matches(use_cache=True)
            x_ra, y_ra, d_ra, w_ra = prep.process_matches_ra(use_cache=True)
            graphs, d_gnn = prep.process_matches_gnn(use_cache=True)
            
            if not x_rw or not x_ra or not graphs:
                print("No matches found in database!")
                continue
                
            unique_divs = list(set(d_rw))
            if "MIXED" not in unique_divs:
                unique_divs.append("MIXED")
            
            for div in unique_divs:
                print(f"\n{'='*50}\n>>> Teaching: {div} DIVISION <<<\n{'='*50}")
                
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
                    print(f"Not enough data for {div} div ({len(x_rw_d)}). Skipping...")
                    continue
                    
                ep1, bs1, lr1 = get_params(len(x_rw_d), "roleweighted")
                print(f"\n[ {div} - RoleWeighted ] Epochs: {ep1}, Batch: {bs1}, LR: {lr1}, Matches: {len(x_rw_d)}")
                train_single_model(x_rw_d, y_rw_d, [div]*len(x_rw_d), f"{div}_roleweighted", CHAMPION_COUNT * 2, ep1, bs1, lr1, "standard", w_rw_d)
                
                ep2, bs2, lr2 = get_params(len(x_ra_d), "roleaware")
                print(f"\n[ {div} - RoleAware ] Epochs: {ep2}, Batch: {bs2}, LR: {lr2}, Matches: {len(x_ra_d)}")
                train_single_model(x_ra_d, y_ra_d, [div]*len(x_ra_d), f"{div}_roleaware", CHAMPION_COUNT * 10, ep2, bs2, lr2, "roleaware", w_ra_d)
                
                if len(gr_d) >= 200:
                    ep3, bs3, lr3 = get_params(len(gr_d), "gnn")
                    print(f"\n[ {div} - GNN ] Epochs: {ep3}, Batch: {bs3}, LR: {lr3}, Matches: {len(gr_d)}")
                    train_gnn_model(gr_d, f"{div}_gnn", ep3, bs3, lr3)
                else:
                    print(f"Not enough data for GNN in the {div} division.")
                
            print("\nEVERY MODEL SUCCESSFULLY TRAINED!")
            
        elif choice == "4":
            break

if __name__ == "__main__":
    try:
        run_trainer()
    except KeyboardInterrupt:
        print("\nTraining stopped by user.")