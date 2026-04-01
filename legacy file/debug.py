import torch
from model.predictor import ChampionPredictor, RoleAwareEmbeddingPredictor
from configs import CHAMPION_COUNT, MODELS_DIR, ROLE_WEIGHTS
import os

#this will be moved to legacy code, just for debugging purposes to see if the models are working as expected on some test inputs.
def debug_predict(blue_team, red_team, division="MIXED"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = {}
    for variant in ["roleweighted", "roleaware"]:
        model_path = os.path.join(MODELS_DIR, f"{division}_{variant}.pth")
        if not os.path.exists(model_path):
            print(f"Skipping {variant}, file not found.")
            continue
            
        if variant == "roleweighted":
            input_size = CHAMPION_COUNT * 2
            model = ChampionPredictor(input_size=input_size).to(device)
        else:
            model = RoleAwareEmbeddingPredictor().to(device)
            
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        models[variant] = model

    if not models:
        print("No models loaded.")
        return
    
    vex = list(ROLE_WEIGHTS.values())

    champions_rw = [0.0] * (CHAMPION_COUNT * 2)
    for i, champ_id in enumerate(blue_team):
        champions_rw[champ_id] = vex[i]
    for i, champ_id in enumerate(red_team):
        champions_rw[champ_id + CHAMPION_COUNT] = vex[i]

    champions_ra = [0.0] * (CHAMPION_COUNT * 10)
    for i, champ_id in enumerate(blue_team):
        champions_ra[i * CHAMPION_COUNT + champ_id] = 1.0 
    for i, champ_id in enumerate(red_team):
        champions_ra[(i + 5) * CHAMPION_COUNT + champ_id] = 1.0  
    print("=== DEBUG INFO ===")
    print("Blue team:", blue_team)
    print("Red team:", red_team)
    print("Roleweighted vector sum:", sum(champions_rw))
    print("Roleaware vector sum:", sum(champions_ra))

    with torch.no_grad():
        if "roleweighted" in models:
            x_rw = torch.tensor([champions_rw], dtype=torch.float32).to(device)
            out_rw = models["roleweighted"](x_rw).item()
            print(f"Roleweighted output: {out_rw:.4f}")

        if "roleaware" in models:
            x_ra = torch.tensor([champions_ra], dtype=torch.float32).to(device)
            out_ra = models["roleaware"](x_ra).item()
            print(f"Roleaware output: {out_ra:.4f}")

blue_team = [1, 2, 3, 4, 5]
red_team = [6, 7, 8, 9, 10]
blue_team2 = [76, 105, 1, 9, 4]
red_team2 = [140, 106, 159, 138, 101]
blue_team = [123, 102, 81, 32, 128]
red_team = [70, 147, 124, 137, 25]

debug_predict(blue_team, red_team)
debug_predict(red_team, blue_team)
debug_predict(blue_team2, red_team2)
debug_predict(red_team2, blue_team2)