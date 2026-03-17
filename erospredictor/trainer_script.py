import os
from model.preprocessor import Preprocessor
from model.train_model import train_single_model
from model.data_manager import DataManager
from model.statistical import StatisticalModel
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
            print("Statisztikai modell megepitve.")
            
        elif choice == "2":
            print("\n1. RoleWeighted\n2. RoleAware")
            m_choice = input("Melyik modellt? (1-2): ").strip()
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
                    print("Nincs a memoriaban adat!")
                    continue
                train_single_model(X, y, divs, f"{division}_roleweighted", CHAMPION_COUNT * 2, epochs, batch_size, lr, "standard", w)
            
            elif m_choice == "2":
                X, y, divs, w = prep.preprocess_all_matches_roleaware(use_cache=True)
                if not X:
                    print("Nincs a memoriaban adat!")
                    continue
                train_single_model(X, y, divs, f"{division}_roleaware", CHAMPION_COUNT * 10, epochs, batch_size, lr, "roleaware", w)
                
        elif choice == "3":
            division = input("Divizio (pl. DIAMOND, MIXED): ").strip().upper()
            
            X_rw, y_rw, divs_rw, w_rw = prep.preprocess_all_matches(use_cache=True)
            X_ra, y_ra, divs_ra, w_ra = prep.preprocess_all_matches_roleaware(use_cache=True)
            
            if not X_rw or not X_ra:
                print("Nincs a memoriaban adat!")
                continue
                
            ep_rw, bs_rw, lr_rw = calculate_optimal_params(len(X_rw), "roleweighted")
            print(f"\nRoleWeighted Auto-tune -> Epochs: {ep_rw}, Batch: {bs_rw}, LR: {lr_rw}")
            train_single_model(X_rw, y_rw, divs_rw, f"{division}_roleweighted", CHAMPION_COUNT * 2, ep_rw, bs_rw, lr_rw, "standard", w_rw)
            
            ep_ra, bs_ra, lr_ra = calculate_optimal_params(len(X_ra), "roleaware")
            print(f"\nRoleAware Auto-tune -> Epochs: {ep_ra}, Batch: {bs_ra}, LR: {lr_ra}")
            train_single_model(X_ra, y_ra, divs_ra, f"{division}_roleaware", CHAMPION_COUNT * 10, ep_ra, bs_ra, lr_ra, "roleaware", w_ra)
            
            print("\nMinden modell sikeresen betanitva")
            
        elif choice == "4":
            break
        else:
            print("Ervenytelen valasztas.")

if __name__ == "__main__":
    try:
        run_trainer()
    except KeyboardInterrupt:
        print("\nTanitas leallitva.")