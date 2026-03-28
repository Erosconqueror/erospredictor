import torch
from PyQt6.QtWidgets import QApplication
from model.predictor import ChampionPredictor, RoleAwareEmbeddingPredictor
from model.gnn_predictor import load_gnn_model, predict_match_gnn
from model.data_manager import DataManager
from model.statistical import StatisticalModel
from configs import CHAMPION_COUNT, MODELS_DIR

class Controller:
    def __init__(self, view=None):
        self.data_manager = DataManager()
        self.champ_mapping = self.data_manager.get_champion_names()
        
        self.name_to_index = {v: k for k, v in self.champ_mapping.items()}
        self.champion_names_list = sorted(list(self.champ_mapping.values()))
        
        self.view = view
        if self.view:
            self.connect_signals() # <--- JAVÍTVA: MOST MÁR MŰKÖDNEK A GOMBOK!
        
        self.stat_model = StatisticalModel(self.data_manager)
        if not self.stat_model.load_cache():
            print("Statisztikai cache nem talalhato, ujraepites SQL-bol...")
            self.stat_model.build_from_matches()
            self.stat_model.save_cache()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.current_division = None
        self.model_rw = None
        self.model_ra = None
        self.model_gnn = None

    def _load_models_for_division(self, division):
        if self.current_division == division:
            return 
            
        print(f"Modellek betoltese a kovetkezo diviziohoz: {division}...")
        
        self.model_rw = ChampionPredictor(CHAMPION_COUNT * 2).to(self.device)
        try:
            self.model_rw.load_state_dict(torch.load(f"{MODELS_DIR}/{division}_roleweighted.pth", map_location=self.device, weights_only=True))
            self.model_rw.eval()
        except: self.model_rw = None
            
        self.model_ra = RoleAwareEmbeddingPredictor(CHAMPION_COUNT, 10).to(self.device)
        try:
            self.model_ra.load_state_dict(torch.load(f"{MODELS_DIR}/{division}_roleaware.pth", map_location=self.device, weights_only=True))
            self.model_ra.eval()
        except: self.model_ra = None
            
        try:
            self.model_gnn = load_gnn_model(self.device, f"{MODELS_DIR}/{division}_gnn.pth")
        except: self.model_gnn = None
            
        self.current_division = division

    def connect_signals(self):
        self.view.predict_btn.clicked.connect(self.on_predict_clicked)
        self.view.recommend_btn.clicked.connect(self.on_recommend_clicked)

    def _get_teams_from_view(self):
        blue_names = [combo.currentText() for combo in self.view.blue_combos]
        red_names = [combo.currentText() for combo in self.view.red_combos]
        return blue_names, red_names

    def _calculate_winrate(self, division, blue_indices, red_indices):
        self._load_models_for_division(division)
        predictions = []

        stat_prob = self.stat_model.predict(division, blue_indices, red_indices)
        predictions.append(stat_prob)

        with torch.no_grad():
            if self.model_rw:
                x_rw = [0.0] * (CHAMPION_COUNT * 2)
                for i, c in enumerate(blue_indices): x_rw[c] = 1.0
                for i, c in enumerate(red_indices): x_rw[c + CHAMPION_COUNT] = 1.0
                tensor_rw = torch.tensor([x_rw], dtype=torch.float32).to(self.device)
                predictions.append(self.model_rw(tensor_rw).item())

            if self.model_ra:
                x_ra = [0.0] * (CHAMPION_COUNT * 10)
                for i, c in enumerate(blue_indices): x_ra[i * CHAMPION_COUNT + c] = 1.0
                for i, c in enumerate(red_indices): x_ra[(i + 5) * CHAMPION_COUNT + c] = 1.0
                tensor_ra = torch.tensor([x_ra], dtype=torch.float32).to(self.device)
                predictions.append(self.model_ra(tensor_ra).item())

            if self.model_gnn:
                gnn_prob = predict_match_gnn(self.model_gnn, blue_indices, red_indices, self.device)
                predictions.append(gnn_prob)

        if not predictions: return 0.5
        return sum(predictions) / len(predictions)

    def on_predict_clicked(self):
        division = self.view.rank_combo.currentText()
        blue_names, red_names = self._get_teams_from_view()
        
        # Ellenőrizzük, hogy minden mező ki van-e töltve érvényes hőssel
        for name in blue_names + red_names:
            if name == "" or name not in self.name_to_index:
                self.view.result_label.setText("Kérlek, válassz ki minden mezőhöz egy létező hőst a listából!")
                self.view.result_label.setStyleSheet("color: orange; font-weight: bold;")
                return

        blue_indices = [self.name_to_index[n] for n in blue_names]
        red_indices = [self.name_to_index[n] for n in red_names]

        avg_prob = self._calculate_winrate(division, blue_indices, red_indices)
        
        if avg_prob >= 0.5:
            self.view.result_label.setText(f"KÉK csapat győzelme várható!\nEsély: {avg_prob * 100:.2f}%")
            self.view.result_label.setStyleSheet("color: blue; font-size: 20px; font-weight: bold; border: 2px solid blue;")
        else:
            self.view.result_label.setText(f"PIROS csapat győzelme várható!\nEsély: {(1.0 - avg_prob) * 100:.2f}%")
            self.view.result_label.setStyleSheet("color: red; font-size: 20px; font-weight: bold; border: 2px solid red;")

    def on_recommend_clicked(self):
        division = self.view.rank_combo.currentText()
        blue_names, red_names = self._get_teams_from_view()
        
        # Üres mezők (vagy olyanok amikbe érvénytelen nevet írtak) keresése
        empty_blue = [i for i, name in enumerate(blue_names) if name not in self.name_to_index]
        empty_red = [i for i, name in enumerate(red_names) if name not in self.name_to_index]
        
        if len(empty_blue) + len(empty_red) != 1:
            self.view.result_label.setText("Hiba: Pontosan 1 darab üres (vagy befejezetlen) mezőt hagyj a felületen az ajánláshoz!")
            self.view.result_label.setStyleSheet("color: orange; font-weight: bold;")
            return

        is_blue_turn = len(empty_blue) == 1
        empty_idx = empty_blue[0] if is_blue_turn else empty_red[0]
        
        self.view.result_label.setText(f"Ajánlás folyamatban a(z) {division} divízióban...\nKérlek várj, több mint 160 szimuláció fut a háttérben!")
        self.view.result_label.setStyleSheet("color: black;")
        QApplication.processEvents()

        picked_names = set([n for n in blue_names + red_names if n in self.name_to_index])
        
        results = []
        for champ_name in self.champion_names_list:
            if champ_name in picked_names:
                continue 
                
            temp_blue = list(blue_names)
            temp_red = list(red_names)
            
            if is_blue_turn:
                temp_blue[empty_idx] = champ_name
            else:
                temp_red[empty_idx] = champ_name
                
            b_idx = [self.name_to_index[n] for n in temp_blue]
            r_idx = [self.name_to_index[n] for n in temp_red]
            
            blue_winrate = self._calculate_winrate(division, b_idx, r_idx)
            
            target_winrate = blue_winrate if is_blue_turn else (1.0 - blue_winrate)
            results.append((champ_name, target_winrate))
            
        results.sort(key=lambda x: x[1], reverse=True)
        top_3 = results[:3]
        
        team_color = "KÉK" if is_blue_turn else "PIROS"
        res_text = f"Top 3 Ajánlás a {team_color} csapatnak ({division}):\n\n"
        for i, (champ, winrate) in enumerate(top_3):
            res_text += f"{i+1}. {champ} ➔ Győzelmi esély: {winrate * 100:.1f}%\n"
            
        self.view.result_label.setStyleSheet(f"color: {'blue' if is_blue_turn else 'red'}; font-size: 16px; font-weight: bold;")
        self.view.result_label.setText(res_text)