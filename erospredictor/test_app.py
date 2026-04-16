import unittest
import json
from unittest.mock import patch, MagicMock
from controller.controller import Controller
from model.core_model import CoreModel
from configs import CHAMPION_DATA_PATH

class TestController(unittest.TestCase):

    @patch('controller.controller.DataManager')
    @patch('controller.controller.StatisticalModel')
    @patch('controller.controller.CoreModel')
    def setUp(self, MockCore, MockStat, MockData):
        self.controller = Controller(dev_mode=True)
        self.controller.core_model = MockCore.return_value

    def test_validate_draft(self):
        b_picks, r_picks = [1, 2, 3, 4, 5], [6, 7, 8, 9, 10]
        b_bans, r_bans = [11, 12, 13, 14, 15], [16, 17, 18, 19, 20]
        is_valid, msg = self.controller.validate_draft(b_picks, r_picks, b_bans, r_bans)
        self.assertTrue(is_valid)

    def test_duplicate_ban(self):
        b_picks, r_picks = [1, 2, 3, 4, 5], [6, 7, 8, 9, 10]
        b_bans, r_bans = [11, 11, 13, 14, 15], [16, 17, 18, 19, 20]
        is_valid, msg = self.controller.validate_draft(b_picks, r_picks, b_bans, r_bans)
        self.assertFalse(is_valid)
        self.assertEqual(msg, "A Kék csapat nem bannolhatja ugyanazt kétszer!")
        
    def test_duplicate_pick(self):
        b_picks, r_picks = [1, 2, 3, 4, 5], [1, 7, 8, 9, 10]
        b_bans, r_bans = [11, 12, 13, 14, 15], [16, 17, 18, 19, 20]
        is_valid, msg = self.controller.validate_draft(b_picks, r_picks, b_bans, r_bans)
        self.assertFalse(is_valid)
        self.assertEqual(msg, "Egy hős csak egyszer lehet kiválasztva!")

    def test_pick_is_banned(self):
        b_picks, r_picks = [1, 2, 3, 4, 5], [6, 7, 8, 9, 10]
        b_bans, r_bans = [1, 12, 13, 14, 15], [16, 17, 18, 19, 20]
        is_valid, msg = self.controller.validate_draft(b_picks, r_picks, b_bans, r_bans)
        self.assertFalse(is_valid)
        self.assertEqual(msg, "Kiválasztott hős nem lehet a tiltólistán!")

class TestModelLogic(unittest.TestCase):

    def setUp(self):
        self.mock_stat = MagicMock()
        self.core = CoreModel(self.mock_stat)
        self.core.device = "cpu"

    def test_id_mapping(self):
        blue = [5, 0, 10, 0, 0] 
        red = [0, 0, 0, 0, 0]
        c_ra = [0.0] * (172 * 10)
        for i, c_id in enumerate(blue):
            if c_id > 0: c_ra[i * 172 + c_id] = 1.0
        self.assertEqual(c_ra[0 * 172 + 5], 1.0)
        self.assertEqual(c_ra[2 * 172 + 10], 1.0)
        self.assertEqual(c_ra[1 * 172 + 5], 0.0)

    def test_weight_normalization(self):
        weights = [0.58, 0.19]
        preds = [0.70, 0.60]
        tot_w = sum(weights)
        norm_w = [w / tot_w for w in weights]
        final_prob = sum(p * w for p, w in zip(preds, norm_w))
        self.assertAlmostEqual(final_prob, 0.6753, places=3)

class TestDataMappingRealFiles(unittest.TestCase):

    def setUp(self):
        with open("data/champion_names.json", "r", encoding="utf-8") as f:
            self.name_to_internal = json.load(f)
        with open(CHAMPION_DATA_PATH, "r", encoding="utf-8") as f:
            self.riot_to_internal = json.load(f)

    def test_id_translation(self):
        champ_name = "Aatrox"
        if champ_name in self.name_to_internal:
            internal_id = self.name_to_internal[champ_name]
            riot_ids = [r for r, i in self.riot_to_internal.items() if int(i) == internal_id]
            self.assertTrue(len(riot_ids) > 0)

    def test_riot_to_internal(self):
        sample_riot_id = list(self.riot_to_internal.keys())[0]
        internal_id = int(self.riot_to_internal.get(str(sample_riot_id), "-1"))
        self.assertNotEqual(internal_id, -1)

    def test_unknown_riot_id(self):
        unknown_riot_id = "999999"
        internal_id = int(self.riot_to_internal.get(unknown_riot_id, "-1"))
        self.assertEqual(internal_id, -1)

class TestRecommendationsRealFiles(unittest.TestCase):

    def setUp(self):
        self.mock_stat = MagicMock()
        self.core = CoreModel(self.mock_stat)
        try:
            with open("data/meta_champs.json", "r", encoding="utf-8") as f:
                self.core.meta_data = json.load(f)
        except Exception:
            self.core.meta_data = {}
            
        self.patcher = patch.object(CoreModel, 'calc_win_prob', return_value=0.55)
        self.mock_calc = self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def test_recommend_strict(self):
        div = "ALL RANKS" if "ALL RANKS" in self.core.meta_data else list(self.core.meta_data.keys())[0]
        if div in self.core.meta_data and "strict" in self.core.meta_data[div]:
            allowed = self.core.meta_data[div]["strict"]["0"]
            recs = self.core.recommend_champs(div, [0]*5, [0]*5, bans=[], team="blue", r_idx=0, top_k=5, filter_off_meta=True)
            for r in recs:
                self.assertIn(r['id'], allowed)

    def test_recommend_loose(self):
        div = "ALL RANKS" if "ALL RANKS" in self.core.meta_data else list(self.core.meta_data.keys())[0]
        if div in self.core.meta_data and "loose" in self.core.meta_data[div]:
            allowed = self.core.meta_data[div]["loose"]["0"]
            recs = self.core.recommend_champs(div, [0]*5, [0]*5, bans=[], team="blue", r_idx=0, top_k=5, filter_off_meta=False)
            for r in recs:
                self.assertIn(r['id'], allowed)

    def test_recommend_unavailable(self):
        div = "MIXED"
        bans = [1, 2, 3] 
        recs = self.core.recommend_champs(div, [0]*5, [0]*5, bans=bans, team="blue", r_idx=0, top_k=20, filter_off_meta=False)
        rec_ids = [r['id'] for r in recs]
        for b in bans:
            self.assertNotIn(b, rec_ids)

    def test_recommend_sorting(self):
        self.mock_calc.side_effect = lambda d, b, r: 0.50 + (b[0] * 0.001)
        recs = self.core.recommend_champs("MIXED", [0]*5, [0]*5, bans=[], team="blue", r_idx=0, top_k=3, filter_off_meta=False)
        if len(recs) == 3:
            self.assertTrue(recs[0]['wr'] >= recs[1]['wr'])
            self.assertTrue(recs[1]['wr'] >= recs[2]['wr'])

if __name__ == '__main__':
    unittest.main()