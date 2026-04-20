import json
import os
from typing import Dict, List, Callable

class GoldenDataset:
    """Validator for evaluating recommendation and prediction models using a domain-knowledge dataset."""
    
    def __init__(self, path: str = "data/golden_dataset.json", name_map: Dict[str, int] = None):
        self.recommendation_cases = []
        self.prediction_cases = []
        self.name_map = name_map or {}
        self.load(path)
    
    def load(self, path: str):
        """Loads recommendation and prediction cases from the specified JSON file."""
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.recommendation_cases = data.get('recommendation_cases', [])
                self.prediction_cases = data.get('prediction_cases', [])

    def _names_to_ids(self, names: List[str]) -> List[int]:
        """Converts a list of champion names to their corresponding numerical IDs."""
        return [self.name_map.get(n, 0) if n else 0 for n in names]

    def validate_recommendations(self, recommendation_fn: Callable, div: str) -> Dict:
        """Validates champion recommendations. Must-picks in Top 5, Avoid-picks NOT in Top 10."""
        results = []
        
        for case in self.recommendation_cases:
            blue_ids = self._names_to_ids(case.get('blue_team', []))
            red_ids = self._names_to_ids(case.get('red_team', []))
            must_pick_ids = set(self._names_to_ids(case.get('must_pick_from', [])))
            avoid_ids = set(self._names_to_ids(case.get('avoid_picks', [])))
            r_idx = case.get('position', 0)
            
            recs = recommendation_fn(div=div, blue=blue_ids, red=red_ids, bans=[], 
                                     team="blue" if r_idx < 5 else "red", 
                                     r_idx=r_idx % 5, top_k=50)
            
            rec_ids = [r['id'] for r in recs]
            passed = True
            reasons = []

            if must_pick_ids:
                top_5_ids = set(rec_ids[:5])
                if not must_pick_ids.intersection(top_5_ids):
                    passed = False
                    reasons.append("Failed: Must-pick champion not found in Top 5.")

            if avoid_ids:
                top_10_ids = set(rec_ids[:8])
                for avoid_id in avoid_ids:
                    if avoid_id in top_10_ids:
                        passed = False
                        reasons.append("Failed: Avoid-pick champion found in Top 8.")
                        break

            results.append({
                'scenario': case.get('scenario', 'Unknown'),
                'passed': passed,
                'reasons': reasons
            })
            
        passed_count = sum(1 for r in results if r['passed'])
        return {
            'passed': passed_count,
            'total': len(results),
            'rate': (passed_count / len(results)) if results else 0.0,
            'results': results
        }

    def validate_predictions(self, prediction_fn: Callable, div: str) -> Dict:
        """Validates full match win probability predictions against expected ranges."""
        results = []
        
        for case in self.prediction_cases:
            blue_ids = self._names_to_ids(case.get('blue_team', []))
            red_ids = self._names_to_ids(case.get('red_team', []))
            min_wr = case.get('expected_blue_winrate_min', 0.0)
            max_wr = case.get('expected_blue_winrate_max', 100.0)
            
            pred = prediction_fn(div=div, blue=blue_ids, red=red_ids)
            blue_wr = pred.get('blue_win_prob', 50.0)
            passed = min_wr <= blue_wr <= max_wr
            
            results.append({
                'scenario': case.get('scenario', 'Unknown'),
                'predicted_winrate': round(blue_wr, 2),
                'expected_range': f"{min_wr}% - {max_wr}%",
                'passed': passed
            })
            
        passed_count = sum(1 for r in results if r['passed'])
        return {
            'passed': passed_count,
            'total': len(results),
            'rate': (passed_count / len(results)) if results else 0.0,
            'results': results
        }