import json
import os
from typing import Dict, List, Tuple

class GoldenDataset:
    """Loads and validates recommendations against domain knowledge rules."""
    
    def __init__(self, path: str = "data/golden_dataset.json"):
        self.cases = []
        self.load(path)
    
    def load(self, path: str):
        """Load golden dataset from JSON file."""
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
                self.cases = data.get('cases', [])
    
    def calculate_penalty(self, recommendations: List[Tuple[int, float]], 
                         case: Dict, top_k: int = 5) -> float:
        """Calculate domain knowledge violation penalty."""
        penalty = 0.0
        top_k_champs = [champ_id for champ_id, _ in recommendations[:top_k]]
        
        if "avoid_picks" in case:
            for avoid_id in case["avoid_picks"]:
                if avoid_id in top_k_champs:
                    idx = top_k_champs.index(avoid_id)
                    penalty += (0.5 / (idx + 1))
        
        if "must_pick_from" in case:
            allowed_set = set(case["must_pick_from"])
            must_pick_count = sum(1 for c in top_k_champs if c in allowed_set)
            
            if must_pick_count == 0:
                penalty += 0.7
            elif must_pick_count < 3:
                penalty += 0.3 * (1 - must_pick_count / 3)
        
        return min(penalty, 1.0)
    
    def get_cases_by_tier(self, tier: str) -> List[Dict]:
        """Get applicable cases for tier."""
        return [c for c in self.cases if c.get('tier') == 'ALL' or c.get('tier') == tier]
    
    def validate(self, recommendation_fn, div: str, top_k: int = 5) -> Dict:
        """Validate recommendations against all cases."""
        cases = self.get_cases_by_tier(div)
        results = []
        
        for case in cases:
            recs = recommendation_fn(
                blue=case['blue_team'],
                red=case['red_team'],
                r_idx=case['position']
            )
            penalty = self.calculate_penalty(recs, case, top_k)
            results.append({
                'scenario': case['scenario'],
                'penalty': penalty,
                'pass': penalty < 0.2
            })
        
        passed = sum(1 for r in results if r['pass'])
        return {
            'passed': passed,
            'total': len(results),
            'rate': (passed / len(results)) if results else 0.0,
            'results': results
        }
