from controller.controller import Controller
#this thing will be moving to legacy files soon, due to separating the data collection and model training into different scripts, but for now it is here for ease of use -its already there
def main():
    controller = Controller()
    
    while True:
        print("\n=== Eros Predictor Data Collector ===")
        print("Select option:")
        print("1. Fetch matches for one summoner")
        print("2. Fetch matches from ranked division")
        print("3. Preprocess all training data (including statistics)")
        print("4. Train specific model")
        print("5. Predict match outcome")
        print("6. Clear preprocessed data cache")
        print("7. Recommend champions for a match")
        print("8. Exit")
        
        choice = input("Enter your choice: ").strip()

        if choice == "1":
            game_name = input("Enter summoner name (gameName): ").strip()
            tag_line = input("Enter tagLine: ").strip()
            controller.fetch_and_store_matches(game_name, tag_line, match_count=5)
        
        elif choice == "2":
            tier = input("Enter tier (e.g., DIAMOND): ").strip().upper()
            division = input("Enter division (e.g., I): ").strip().upper()
            playerlimit = int(input("Enter Player limit: ").strip())
            matchcount = int(input("Enter matchcount per player: ").strip())
            controller.fetch_from_division(tier=tier, division=division, players_limit = playerlimit, match_count=matchcount)
        
        elif choice == "3":
            use_cache = input("Use cached data if available? (y/n, default y): ").strip().lower()
            use_cache = use_cache != 'n'
            print("Preprocessing all training data including statistics...")
            controller.preprocess_all_training_data(use_cache=use_cache)
        
        elif choice == "4":
            use_cache = input("Use cached preprocessed data? (y/n, default y): ").strip().lower()
            use_cache = use_cache != 'n'
            
            print("Select model to train:")
            print("1. RoleWeighted Model")
            print("2. RoleAware Model")
            print("3. Statistical Model")
            print("4. GNN Model")
            model_choice = input("Enter model choice (1-4): ").strip()
            
            division = input("Enter division (e.g., DIAMOND, GOLD, MIXED): ").strip().upper()
            epochs = input("Enter number of training epochs (default varies by model): ").strip()
            batch_size = input("Enter batch size (default varies by model): ").strip()
            lr = input("Enter learning rate (default varies by model): ").strip()
            
            controller.train_specific_model(
                model_type=model_choice,
                division=division,
                epochs=int(epochs) if epochs else None,
                batch_size=int(batch_size) if batch_size else None,
                lr=float(lr) if lr else None,
                use_cache=use_cache
            )
        
        elif choice == "5":
            controller.predict_match()
        
        elif choice == "6":
            print("Clearing preprocessed data cache...")
            controller.clear_preprocessed_cache()
            print("Cache cleared!")
        
        elif choice == "7":
            division = input("Division (pl. DIAMOND): ").strip().upper()
            
            print("Give the picks ():")
            blue = [int(input(f"Blue {i+1}: ")) for i in range(5)]
            red = [int(input(f"Red {i+1}: ")) for i in range(5)]
            
            bans_raw = input("Banned cvhamps: ").strip()
            bans = [int(x) for x in bans_raw.split(",")] if bans_raw else []
            
            next_idx = int(input("Next position: (0-4: Blue, 5-9: Red): "))
            
            results = controller.recommend_champions(division, blue, red, next_idx, bans)
            
            print("\n=== Recommended champions ===")
            for i, (cid, score) in enumerate(results):
                print(f"{i+1}. Champion ID: {cid} | Projected winrate: {score*100:.2f}%")
        
        elif choice == "8":
            print("Exiting...")
            break
        
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()