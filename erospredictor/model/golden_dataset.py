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
        return [self.name_map.get(n, -1) if n else -1 for n in names]

    def validate_recommendations(self, recommendation_fn: Callable, div: str) -> Dict:
        """Grade recommendation ranks with partial credit; prediction QA is separate.

        Must-picks: Top 5 = 1, Top 10 = .75, Top 20 = .5. Avoid-picks:
        Top 3 = 0, ranks 4-8 = .5, outside Top 8 = 1. Average the applicable
        criteria, so one near miss no longer erases an entire scenario.
        """
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
            criteria = []
            reasons = []

            if must_pick_ids:
                rank = min((rec_ids.index(cid) + 1 for cid in must_pick_ids if cid in rec_ids), default=999)
                criteria.append(1.0 if rank <= 5 else .75 if rank <= 10 else .5 if rank <= 20 else 0.0)
                if rank > 5:
                    reasons.append(f"Must-pick best rank: {rank if rank != 999 else 'not found'}; partial credit applied.")

            if avoid_ids:
                rank = min((rec_ids.index(cid) + 1 for cid in avoid_ids if cid in rec_ids), default=999)
                criteria.append(0.0 if rank <= 3 else .5 if rank <= 8 else 1.0)
                if rank <= 8:
                    reasons.append(f"Avoid-pick best rank: {rank}; partial credit applied.")

            score = sum(criteria) / len(criteria) if criteria else 1.0

            results.append({
                'scenario': case.get('scenario', 'Unknown'),
                'passed': score >= .75,
                'score': score,
                'reasons': reasons
            })
            
        passed_count = sum(1 for r in results if r['passed'])
        return {
            'passed': passed_count,
            'total': len(results),
            'score': sum(r['score'] for r in results),
            'rate': (sum(r['score'] for r in results) / len(results)) if results else 0.0,
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
