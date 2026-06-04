"""
データリーク自動検証テスト。
pytest tests/test_no_leakage.py で実行。
"""
from __future__ import annotations

import pandas as pd
import pytest

from features.feature_specs import (
    FORBIDDEN_FEATURES,
    ALLOWED_FEATURES,
    assert_no_forbidden,
)
from models.walk_forward import monthly_expanding_walk_forward


# ────────────────────────────────────────────────
# 禁止特徴量チェック
# ────────────────────────────────────────────────
class TestForbiddenFeatures:
    def test_forbidden_list_not_empty(self):
        assert len(FORBIDDEN_FEATURES) > 0

    def test_key_forbidden_features_in_list(self):
        must_be_forbidden = [
            "closing_odds",
            "finish_position",
            "agari3f_seconds",   # 当走値
            "agari3f_rank",      # 当走値
            "passing_order_4c",
            "race_time",
        ]
        for f in must_be_forbidden:
            assert f in FORBIDDEN_FEATURES, f"{f} が禁止リストにない"

    def test_allowed_features_not_in_forbidden(self):
        overlap = set(ALLOWED_FEATURES) & set(FORBIDDEN_FEATURES)
        assert not overlap, f"許可と禁止の重複: {overlap}"

    def test_assert_no_forbidden_raises_on_violation(self):
        with pytest.raises(ValueError, match="データリーク"):
            assert_no_forbidden(["horse_weight", "finish_position"])

    def test_assert_no_forbidden_passes_on_clean(self):
        assert_no_forbidden(["horse_weight", "draw_number", "jockey_id"])

    def test_hist_agari_is_allowed(self):
        """過去走の上がりは使えるが当走の値は禁止。"""
        assert "agari3f_rank_pct_hist" in ALLOWED_FEATURES
        assert "agari3f_seconds" in FORBIDDEN_FEATURES
        assert "agari3f_rank" in FORBIDDEN_FEATURES


# ────────────────────────────────────────────────
# walk-forward 時系列整合性
# ────────────────────────────────────────────────
class TestWalkForward:
    def _make_races(self, n_months: int = 36) -> pd.DataFrame:
        dates = pd.date_range("2022-01-01", periods=n_months * 4, freq="7D")
        return pd.DataFrame({
            "race_date": dates,
            "race_id":   [f"R{i:04d}" for i in range(len(dates))],
            "horse_id":  [f"H{i:04d}" for i in range(len(dates))],
        })

    def test_no_future_data_in_train(self):
        """train に test 以降のデータが混入していないか。"""
        races = self._make_races(36)
        for train_idx, test_idx in monthly_expanding_walk_forward(races, min_train_months=24):
            train_dates = pd.to_datetime(races.loc[train_idx, "race_date"])
            test_dates  = pd.to_datetime(races.loc[test_idx, "race_date"])
            assert train_dates.max() < test_dates.min(), \
                f"時系列リーク: train最大={train_dates.max()}, test最小={test_dates.min()}"

    def test_embargo_gap_respected(self):
        """train末尾とtest開始の間に7日以上のギャップがあるか。"""
        from datetime import timedelta
        races = self._make_races(36)
        for train_idx, test_idx in monthly_expanding_walk_forward(
            races, min_train_months=24, embargo_days=7
        ):
            train_max = pd.to_datetime(races.loc[train_idx, "race_date"]).max()
            test_min  = pd.to_datetime(races.loc[test_idx, "race_date"]).min()
            gap = (test_min - train_max).days
            assert gap >= 7, f"embargo不足: gap={gap}日"

    def test_folds_increase_training_size(self):
        """foldが進むにつれてtrainサイズが増えているか（expanding）。"""
        races = self._make_races(36)
        prev_train_size = 0
        for train_idx, _ in monthly_expanding_walk_forward(races, min_train_months=24):
            assert len(train_idx) >= prev_train_size, "expanding になっていない"
            prev_train_size = len(train_idx)

    def test_min_train_months_enforced(self):
        """min_train_months=24 → 24ヶ月未満のデータで ValueError 発生。"""
        small_races = pd.DataFrame({
            "race_date": pd.date_range("2024-01-01", periods=10, freq="30D"),
            "race_id":   [f"R{i}" for i in range(10)],
            "horse_id":  [f"H{i}" for i in range(10)],
        })
        with pytest.raises(ValueError, match="データが不足"):
            list(monthly_expanding_walk_forward(small_races, min_train_months=24))

    def test_no_overlap_between_train_and_test(self):
        """train と test のインデックスが重複していないか。"""
        races = self._make_races(36)
        for train_idx, test_idx in monthly_expanding_walk_forward(races, min_train_months=24):
            overlap = set(train_idx) & set(test_idx)
            assert not overlap, f"train/test インデックス重複: {len(overlap)}件"
