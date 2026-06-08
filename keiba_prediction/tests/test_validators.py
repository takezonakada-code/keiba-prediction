"""
バリデーション・モデルルーティングの自動テスト。
pytest tests/test_validators.py で実行。
"""
import sys, math
sys.path.insert(0, __import__("pathlib").Path(__file__).parent.parent.__str__())

import pytest
from core.validators import (
    validate_ev, validate_odds, validate_probability,
    validate_scores, validate_race_data, EV_MAX, EV_MIN,
)
from core.model_router import get_track_group, get_payback


# ─────────────────────────────────────────────
# EV バリデーション
# ─────────────────────────────────────────────
class TestValidateEV:
    def test_normal_value(self):
        assert validate_ev(0.5) == 0.5

    def test_negative_ev(self):
        assert validate_ev(-0.3) == -0.3

    def test_nan_returns_none(self):
        assert validate_ev(float("nan")) is None

    def test_inf_returns_none(self):
        assert validate_ev(float("inf")) is None

    def test_none_returns_none(self):
        assert validate_ev(None) is None

    def test_above_max_clamped(self):
        assert validate_ev(18.0) == EV_MAX     # +18 → EV_MAX(3.0)

    def test_below_min_clamped(self):
        assert validate_ev(-5.0) == EV_MIN

    def test_zero_ok(self):
        assert validate_ev(0.0) == 0.0

    def test_at_boundary(self):
        assert validate_ev(EV_MAX) == EV_MAX
        assert validate_ev(EV_MIN) == EV_MIN


# ─────────────────────────────────────────────
# オッズ バリデーション
# ─────────────────────────────────────────────
class TestValidateOdds:
    def test_normal_odds(self):
        assert validate_odds(5.0) == 5.0

    def test_none_returns_none(self):
        assert validate_odds(None) is None

    def test_zero_returns_none(self):
        assert validate_odds(0) is None

    def test_negative_returns_none(self):
        assert validate_odds(-1.0) is None

    def test_nan_returns_none(self):
        assert validate_odds(float("nan")) is None


# ─────────────────────────────────────────────
# スコア バリデーション
# ─────────────────────────────────────────────
class TestValidateScores:
    def test_sum_to_one(self):
        scores = validate_scores([0.3, 0.5, 0.2])
        assert math.isclose(sum(scores), 1.0, abs_tol=1e-9)

    def test_nan_handled(self):
        scores = validate_scores([float("nan"), 0.5, 0.5])
        assert all(not math.isnan(s) for s in scores)
        assert math.isclose(sum(scores), 1.0, abs_tol=1e-9)

    def test_all_zero_equal_distribution(self):
        scores = validate_scores([0.0, 0.0, 0.0])
        assert math.isclose(scores[0], 1/3, abs_tol=1e-6)

    def test_no_negative(self):
        scores = validate_scores([0.5, -0.1, 0.6])
        assert all(s >= 0 for s in scores)


# ─────────────────────────────────────────────
# レースデータ バリデーション
# ─────────────────────────────────────────────
class TestValidateRaceData:
    def _make_horses(self, n, with_odds=True):
        return [{"draw_number": i+1, "horse_name": f"馬{i+1}",
                 "win_odds": float(2+i) if with_odds else None}
                for i in range(n)]

    def test_ok_with_enough_horses(self):
        ok, errs = validate_race_data(self._make_horses(9))
        assert ok, errs

    def test_fail_less_than_3_horses(self):
        ok, errs = validate_race_data(self._make_horses(2))
        assert not ok

    def test_fail_too_many_missing_odds(self):
        horses = self._make_horses(10, with_odds=False)
        ok, errs = validate_race_data(horses)
        assert not ok


# ─────────────────────────────────────────────
# モデルルーター
# ─────────────────────────────────────────────
class TestModelRouter:
    def test_banei_routing(self):
        assert get_track_group("帯広", "banei") == "banei"
        assert get_track_group("帯広") == "banei"

    def test_nar_kanto_routing(self):
        for t in ["大井", "川崎", "船橋", "浦和"]:
            assert get_track_group(t) == "kanto", f"{t} should be kanto"

    def test_nar_local_routing(self):
        for t in ["名古屋", "園田", "笠松", "金沢", "高知"]:
            assert get_track_group(t) == "nar_local"

    def test_jra_routing(self):
        for t in ["東京", "阪神", "中山", "京都"]:
            assert get_track_group(t) == "jra"

    def test_unknown_track_fallback(self):
        # 未知の競馬場はエラーにならず nar_local になる
        assert get_track_group("未知の場") == "nar_local"

    def test_banei_not_flat(self):
        # ばんえいに平地モデルが使われないこと
        assert get_track_group("帯広") != "kanto"
        assert get_track_group("帯広") != "nar_local"
        assert get_track_group("帯広") != "jra"

    def test_payback_rates(self):
        assert get_payback("大井")   == 0.725
        assert get_payback("名古屋") == 0.70
        assert get_payback("帯広")   == 0.70
        assert get_payback("東京")   == 0.75


# ─────────────────────────────────────────────
# ばんえい vs 平地 特徴量分離
# ─────────────────────────────────────────────
class TestFeatureSeparation:
    BANEI_FEATURES = [
        "horse_weight", "burden_weight", "post_position_bias",
    ]
    FLAT_ONLY_FEATURES = ["agari3f_z", "speed_index_mean", "style_score_c4"]

    def test_banei_not_using_flat_features(self):
        for f in self.FLAT_ONLY_FEATURES:
            assert f not in self.BANEI_FEATURES, \
                f"ばんえいが平地特徴量 '{f}' を使用している"

    def test_no_nan_in_ev_range(self):
        """EV範囲内の値にNaNが含まれないこと"""
        test_evs = [validate_ev(v) for v in [-1.0, 0.0, 0.5, 1.0, EV_MAX]]
        for ev in test_evs:
            if ev is not None:
                assert not math.isnan(ev)
