import os
from model.preprocessor import Preprocessor
from model.train_model import train_single_model
from model.data_manager import DataManager
from model.statistical import StatisticalModel
from model.gnn_predictor import preprocess_matches_for_gnn
from model.train_model import train_gnn_model
from configs import CHAMPION_COUNT
#ezt meg tweakelni kell elegge, hogy a parameterek optimalisak legyenek kulonbozo adatmeretekhez es modellekhez, de egy jo kiindulasi pont lehet az auto-tuninghoz (imadom hogy a copilot hogyan generalja a kommenteim veget :D)
def calculate_optimal_params(data_size, model_type):
    if data_size < 2000:
        epochs = 100
        batch_size = 32
        lr = 0.001
    elif data_size < 10000:
        epochs = 75
        batch_size = 64
        lr = 0.001
    elif data_size < 30000:
        epochs = 50
        batch_size = 128
        lr = 0.0005
    else:
        epochs = 40
        batch_size = 256
        lr = 0.0003
        
    if model_type == "gnn":
        lr = lr * 0.3
        epochs = min(epochs, 60)
        
    return epochs, batch_size, lr

def run_trainer():
    print("=== EROS PREDICTOR - TANITO MODUL ===")
    prep = Preprocessor()
    db = DataManager()
    stat_model = StatisticalModel(db)
    
    while True:
        print("\n1. Adatok elofeldolgozasa (Preprocess)")
        print("2. Egyedi modell tanitasa (Parameter megadassal)")
        print("3. Osszes modell automatikus tanitasa (Auto-tune)")
        print("4. Kilepes")
        
        choice = input("Valassz opciot: ").strip()
        
        if choice == "1":
            print("\nAdatok elofeldolgozasa folyamatban...")
            prep.clear_cache()
            
            X_rw, y_rw, div_rw, w_rw = prep.preprocess_all_matches(use_cache=False)
            print(f"RoleWeighted adatok feldolgozva: {len(X_rw)} minta")
            
            X_ra, y_ra, div_ra, w_ra = prep.preprocess_all_matches_roleaware(use_cache=False)
            print(f"RoleAware adatok feldolgozva: {len(X_ra)} minta")
            stat_model.build_from_matches()
            stat_model.save_cache()
            print("Statisztikai modell megepitve.")
            
        elif choice == "2":
            print("\n1. RoleWeighted\n2. RoleAware\n3. GNN (Graph Neural Network)")
            m_choice = input("Melyik modellt? (1-3): ").strip()
            division = input("Divizio (pl. DIAMOND, MIXED): ").strip().upper()
            
            ep = input("Epochs (alap 50): ").strip()
            bs = input("Batch size (alap 32): ").strip()
            lr_inp = input("Learning rate (alap 0.001): ").strip()
            
            epochs = int(ep) if ep else 50
            batch_size = int(bs) if bs else 32
            lr = float(lr_inp) if lr_inp else 0.001
            
            if m_choice == "1":
                X, y, divs, w = prep.preprocess_all_matches(use_cache=True)
                if not X:
                    print("Nincs adat! Futtasd az 1-es opciot elobb.")
                    continue
                train_single_model(X, y, divs, f"{division}_roleweighted", CHAMPION_COUNT * 2, epochs, batch_size, lr, "standard", w)
            
            elif m_choice == "2":
                X, y, divs, w = prep.preprocess_all_matches_roleaware(use_cache=True)
                if not X:
                    print("Nincs adat! Futtasd az 1-es opciot elobb.")
                    continue
                train_single_model(X, y, divs, f"{division}_roleaware", CHAMPION_COUNT * 10, epochs, batch_size, lr, "roleaware", w)

            elif m_choice == "3":
                graphs, divs = preprocess_matches_for_gnn(db, prep.patch_weights)
                if not graphs:
                    print("Nincs adat a GNN-hez!")
                    continue
                train_gnn_model(graphs, f"{division}_gnn", epochs, batch_size, lr)
                
        elif choice == "3":
            X_rw, y_rw, divs_rw, w_rw = prep.preprocess_all_matches(use_cache=True)
            X_ra, y_ra, divs_ra, w_ra = prep.preprocess_all_matches_roleaware(use_cache=True)
            
            print("GNN adatok (grafok) elofeldolgozasa...")
            graphs, divs_gnn = preprocess_matches_for_gnn(db, prep.patch_weights)
            
            if not X_rw or not X_ra or not graphs:
                print("Nincs a memoriaban eleg adat! Futtasd az 1-es opciot elobb.")
                continue
                
            unique_divisions = list(set(divs_rw))
            if "MIXED" not in unique_divisions:
                unique_divisions.append("MIXED")
            
            for division in unique_divisions:
                print(f"\n{'='*50}")
                print(f">>> TANITAS: {division} DIVIZIO <<<")
                print(f"{'='*50}")
                
                if division == "MIXED":
                    X_rw_div, y_rw_div, w_rw_div = X_rw, y_rw, w_rw
                    X_ra_div, y_ra_div, w_ra_div = X_ra, y_ra, w_ra
                    graphs_div = graphs
                else:
                    X_rw_div = [x for x, d in zip(X_rw, divs_rw) if d == division]
                    y_rw_div = [y for y, d in zip(y_rw, divs_rw) if d == division]
                    w_rw_div = [w for w, d in zip(w_rw, divs_rw) if d == division]
                    
                    X_ra_div = [x for x, d in zip(X_ra, divs_ra) if d == division]
                    y_ra_div = [y for y, d in zip(y_ra, divs_ra) if d == division]
                    w_ra_div = [w for w, d in zip(w_ra, divs_ra) if d == division]
                    
                    graphs_div = [g for g, d in zip(graphs, divs_gnn) if d == division]
                    
                if len(X_rw_div) < 200:
                    print(f"Nincs eleg adat a(z) {division} diviziohoz ({len(X_rw_div)} meccs). Atugras...")
                    continue
                    
                # RoleWeighted Tanítás
                ep_rw, bs_rw, lr_rw = calculate_optimal_params(len(X_rw_div), "roleweighted")
                print(f"\n[ {division} - RoleWeighted ] Epochs: {ep_rw}, Batch: {bs_rw}, LR: {lr_rw}, Meccsek: {len(X_rw_div)}")
                train_single_model(X_rw_div, y_rw_div, [division]*len(X_rw_div), f"{division}_roleweighted", CHAMPION_COUNT * 2, ep_rw, bs_rw, lr_rw, "standard", w_rw_div)
                
                # RoleAware Tanítás
                ep_ra, bs_ra, lr_ra = calculate_optimal_params(len(X_ra_div), "roleaware")
                print(f"\n[ {division} - RoleAware ] Epochs: {ep_ra}, Batch: {bs_ra}, LR: {lr_ra}, Meccsek: {len(X_ra_div)}")
                train_single_model(X_ra_div, y_ra_div, [division]*len(X_ra_div), f"{division}_roleaware", CHAMPION_COUNT * 10, ep_ra, bs_ra, lr_ra, "roleaware", w_ra_div)
                
                # GNN Tanítás
                if len(graphs_div) >= 200:
                    ep_gnn, bs_gnn, lr_gnn = calculate_optimal_params(len(graphs_div), "gnn")
                    print(f"\n[ {division} - GNN ] Epochs: {ep_gnn}, Batch: {bs_gnn}, LR: {lr_gnn}, Meccsek: {len(graphs_div)}")
                    train_gnn_model(graphs_div, f"{division}_gnn", ep_gnn, bs_gnn, lr_gnn)
                else:
                    print(f"Nincs eleg adat a GNN-hez a(z) {division} divizioban.")
                
            print("\n!!! MINDEN MODELL SIKERESEN BETANITVA !!!")
            
        elif choice == "4":
            break
        else:
            print("Ervenytelen valasztas.")

if __name__ == "__main__":
    try:
        run_trainer()
    except KeyboardInterrupt:
        print("\nTanitas leallitva.")